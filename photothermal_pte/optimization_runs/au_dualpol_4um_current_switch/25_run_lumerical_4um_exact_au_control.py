#!/usr/bin/env python3
"""Run one fail-closed GPU exact-Au 4-um Lumerical control.

Run ``source_only`` first for each polarization and numerical mesh.  An
empty/full/simple_L material run refuses to start unless the matching passed
source-only JSON is supplied.  Raw FSP/NPZ/JSON artifacts are written outside
the Git worktree and are never overwritten.

The default accelerator policy remains strict B200.  The explicit
``development`` policy permits debugging on another NVIDIA GPU while marking
the result as non-promotable and requiring a later B200 rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.native_yee_q import (  # noqa: E402
    extract_native_yee_q,
    frequency_slice,
    integrate_xyz,
)
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_FIELD,
    PABS_GROUP,
    PABS_INDEX,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (  # noqa: E402
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_forward import (  # noqa: E402
    C0_M_S,
    DENSITY_CONTROL,
    ENDPOINT_FIELD_MONITOR,
    EXACT_BINARY_CONTROL,
    SOURCE_NAME,
    TARGET_MONITOR,
    build_layout,
    coordinate_material_partition,
    control_volume_bounds,
    material_fit_readback,
    requested_mesh_readback_gates,
    source_calibration_contract,
    validate_source_calibration_record,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (  # noqa: E402
    density_state_audit,
    density_nodes,
    load_projected_density_file,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_mesh_contract import (  # noqa: E402
    BASELINE,
    BASELINE_SOURCE_OBJECT_W0_UM,
    GEOMETRY_CONTROLS,
    LumericalMeshSpec,
    MESH_REFINEMENT_CANDIDATES,
    POLARIZATIONS,
    Q_FLUX_GATE,
    RELATIVE_GATE,
    SOURCE_PROFILE_GATE,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_exact_au import (  # noqa: E402
    AU_MATERIAL_MAX_COEFFICIENTS,
    MATERIAL_FIT_TOLERANCE,
    au_fit_configuration,
    control_geometry_audits,
    design_edges,
    exact_control_masks,
    mask_rectangles,
    material_contract_audit,
    measurement_electrode_bounds,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_maxwell_contract import (  # noqa: E402
    ACCELERATOR_POLICIES,
    LUMAPI_PATH,
    LUMERICAL_ROOT,
    audit_environment,
    require_lumerical_gpu,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_only_boundary import (  # noqa: E402
    require_lumerical_only_source_boundary,
)
from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    validate_paper_ir_source_only_gpu as lumerical_audit,
)


HERE = Path(__file__).resolve().parent
PHYSICAL_DEVICE_CONTRACT = HERE / "physical_device_contract.json"
DEFAULT_RAW_ROOT = Path("/home/seunghyun/tairte4_raw_artifacts")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _scalar(value: Any, label: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise RuntimeError(f"{label} is not scalar: {array.shape}")
    result = float(np.real(array[0]))
    if not np.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def _single_frequency_cube(
    value: Any,
    shape: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (*shape, 1):
        array = array[..., 0]
    if array.shape != shape:
        raise RuntimeError(f"{label} shape {array.shape} != {shape}")
    return array


def _normalized_gpu_uuid(value: str) -> str:
    return value.removeprefix("GPU-").strip().lower()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=(
            "source_only",
            *GEOMETRY_CONTROLS,
            DENSITY_CONTROL,
            EXACT_BINARY_CONTROL,
        ),
    )
    parser.add_argument("--rho", type=float)
    parser.add_argument(
        "--rho-file",
        type=Path,
        help=(
            "NPY or NPZ containing one nonuniform 81x81 projected nodal "
            "density. This is mutually exclusive with scalar --rho."
        ),
    )
    parser.add_argument(
        "--rho-key",
        default="projected_density_nodal",
        help="NPZ array key used by --rho-file (default: projected_density_nodal).",
    )
    parser.add_argument(
        "--binary-mask-file",
        type=Path,
        help="NPY/NPZ containing the exact 80x80 zero/one Au cell mask.",
    )
    parser.add_argument(
        "--binary-mask-key",
        default="binary_mask",
        help="NPZ key used by --binary-mask-file (default: binary_mask).",
    )
    parser.add_argument("--polarization", required=True, choices=POLARIZATIONS)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument(
        "--accelerator-policy",
        choices=ACCELERATOR_POLICIES,
        default=os.environ.get("AU_LUMERICAL_ACCELERATOR_POLICY", "b200"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-calibration-json", type=Path)
    parser.add_argument(
        "--source-object-w0-um",
        type=float,
        default=BASELINE_SOURCE_OBJECT_W0_UM,
    )
    parser.add_argument("--mesh-label", default=BASELINE.label)
    parser.add_argument("--flake-dxy-nm", type=float, default=BASELINE.flake_dxy_m * 1e9)
    parser.add_argument("--stack-dz-nm", type=float, default=BASELINE.stack_dz_m * 1e9)
    parser.add_argument("--bulk-dz-nm", type=float, default=BASELINE.bulk_dz_m * 1e9)
    parser.add_argument("--outer-dxy-nm", type=float, default=BASELINE.outer_dxy_m * 1e9)
    parser.add_argument("--mesh-accuracy", type=int, default=BASELINE.mesh_accuracy)
    parser.add_argument(
        "--au-max-coefficients",
        type=int,
        default=AU_MATERIAL_MAX_COEFFICIENTS,
        help=(
            "Maximum Lumerical multi-coefficient terms for sampled-data Au only. "
            "TaIrTe4 and SiO2 remain fixed at the audited default."
        ),
    )
    parser.add_argument(
        "--au-fit-tolerance",
        type=float,
        default=MATERIAL_FIT_TOLERANCE,
        help="Lumerical sampled-data fit tolerance for Au only.",
    )
    parser.add_argument(
        "--mesh-refinement",
        choices=MESH_REFINEMENT_CANDIDATES,
        default=BASELINE.conformal_mesh,
        help=(
            "Lumerical metal-interface mesh method. Au promotion requires "
            "an explicit CV0/CV1/staircase convergence comparison."
        ),
    )
    parser.add_argument("--pml-layers", type=int, default=BASELINE.pml_layers)
    parser.add_argument("--lateral-span-um", type=float, default=BASELINE.lateral_span_m * 1e6)
    parser.add_argument("--z-min-um", type=float, default=BASELINE.z_min_m * 1e6)
    parser.add_argument("--z-max-um", type=float, default=BASELINE.z_max_m * 1e6)
    parser.add_argument(
        "--simulation-time-ps",
        type=float,
        default=BASELINE.simulation_time_s * 1e12,
    )
    parser.add_argument(
        "--auto-shutoff-min", type=float, default=BASELINE.auto_shutoff_min
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--include-adjoint-field-region",
        action="store_true",
        help=(
            "Add a source-disabled FieldRegion during an import-density "
            "forward so the frozen FSP can seed a later distributed adjoint."
        ),
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--recover-completed-fsp",
        action="store_true",
        help=(
            "Load an already completed exact-control FSP and regenerate only "
            "the NPZ/JSON postprocessing artifacts. This path never calls "
            "runsetup(), run(), strict_gpu_run(), or save()."
        ),
    )
    args = parser.parse_args()
    if not np.isfinite(args.source_object_w0_um) or args.source_object_w0_um <= 0:
        parser.error("--source-object-w0-um must be finite and positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        au_fit_configuration(
            max_coefficients=args.au_max_coefficients,
            tolerance=args.au_fit_tolerance,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.case == DENSITY_CONTROL:
        if (args.rho is None) == (args.rho_file is None):
            parser.error(
                "import_density requires exactly one of scalar --rho or --rho-file"
            )
        if args.rho is not None and (
            not np.isfinite(args.rho) or not 0.0 <= args.rho <= 1.0
        ):
            parser.error("--rho must be finite and in [0,1]")
        if args.binary_mask_file is not None:
            parser.error("import_density does not accept --binary-mask-file")
    elif args.rho is not None or args.rho_file is not None:
        parser.error("--rho/--rho-file are valid only for import_density")
    if args.case == EXACT_BINARY_CONTROL:
        if args.binary_mask_file is None:
            parser.error("exact_binary requires --binary-mask-file")
    elif args.binary_mask_file is not None:
        parser.error("--binary-mask-file is valid only for exact_binary")
    if args.include_adjoint_field_region and args.case != DENSITY_CONTROL:
        parser.error(
            "--include-adjoint-field-region is valid only for import_density"
        )
    if args.recover_completed_fsp and args.audit_only:
        parser.error("--recover-completed-fsp and --audit-only are mutually exclusive")
    if args.recover_completed_fsp and args.case in (
        "source_only",
        DENSITY_CONTROL,
        EXACT_BINARY_CONTROL,
    ):
        parser.error(
            "--recover-completed-fsp currently accepts only exact material controls"
        )
    if (
        args.case != "source_only"
        and args.source_calibration_json is None
        and not args.audit_only
    ):
        parser.error("material cases require --source-calibration-json")
    if not args.audit_only and args.gpu_index is None:
        env_index = os.environ.get("LUMERICAL_GPU_INDEX")
        if env_index is None and args.accelerator_policy == "b200":
            env_index = os.environ.get("LUMERICAL_B200_GPU_INDEX")
        if env_index is None:
            parser.error(
                "a Maxwell run requires --gpu-index or LUMERICAL_GPU_INDEX"
            )
        args.gpu_index = int(env_index)
    return args


def _mesh_spec(args: argparse.Namespace) -> LumericalMeshSpec:
    return LumericalMeshSpec(
        label=args.mesh_label,
        flake_dxy_m=args.flake_dxy_nm * 1e-9,
        stack_dz_m=args.stack_dz_nm * 1e-9,
        bulk_dz_m=args.bulk_dz_nm * 1e-9,
        outer_dxy_m=args.outer_dxy_nm * 1e-9,
        mesh_accuracy=args.mesh_accuracy,
        pml_layers=args.pml_layers,
        lateral_span_m=args.lateral_span_um * 1e-6,
        z_min_m=args.z_min_um * 1e-6,
        z_max_m=args.z_max_um * 1e-6,
        simulation_time_s=args.simulation_time_ps * 1e-12,
        auto_shutoff_min=args.auto_shutoff_min,
        conformal_mesh=args.mesh_refinement,
    ).validate()


def _projected_density(
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if args.case != DENSITY_CONTROL:
        return None, None
    if args.rho_file is None:
        assert args.rho is not None
        density = np.full(CONTRACT.design_node_shape, args.rho, dtype=np.float64)
        provenance: dict[str, Any] = {
            "kind": "uniform_scalar_cli",
            "rho": float(args.rho),
        }
    else:
        resolved = args.rho_file.expanduser().resolve()
        density = load_projected_density_file(resolved, key=args.rho_key)
        provenance = {
            "kind": "nonuniform_file",
            "path": str(resolved),
            "sha256": _sha256(resolved),
            "array_key": args.rho_key if resolved.suffix.lower() == ".npz" else None,
        }
    provenance["density_state"] = density_state_audit(density)
    return density, provenance


def _exact_binary_mask(
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if args.case != EXACT_BINARY_CONTROL:
        return None, None
    assert args.binary_mask_file is not None
    resolved = args.binary_mask_file.expanduser().resolve()
    if resolved.suffix.lower() == ".npy":
        mask = np.load(resolved, allow_pickle=False)
        key: str | None = None
    elif resolved.suffix.lower() == ".npz":
        with np.load(resolved, allow_pickle=False) as data:
            if args.binary_mask_key not in data:
                raise KeyError(
                    f"missing binary mask key {args.binary_mask_key!r}: {resolved}"
                )
            mask = np.asarray(data[args.binary_mask_key])
        key = args.binary_mask_key
    else:
        raise ValueError("exact binary mask must use NPY or NPZ")
    if (
        mask.shape != CONTRACT.design_shape
        or not np.all(np.isfinite(mask))
        or not np.all((mask == 0) | (mask == 1))
    ):
        raise ValueError("exact binary mask must be finite 80x80 with values 0/1")
    value = np.asarray(mask, dtype=np.uint8)
    return value, {
        "kind": "exact_binary_cell_mask",
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "array_key": key,
        "solid_cell_count": int(np.sum(value)),
    }


def _case_label(
    args: argparse.Namespace,
    projected_density: np.ndarray | None = None,
    exact_binary_mask: np.ndarray | None = None,
) -> str:
    if args.case == EXACT_BINARY_CONTROL:
        if exact_binary_mask is None:
            raise ValueError("exact-binary label requires the cell mask")
        digest = hashlib.sha256(
            np.ascontiguousarray(exact_binary_mask, dtype=np.uint8).tobytes()
        ).hexdigest()
        return f"exact_binary_{digest[:12]}"
    if args.case != DENSITY_CONTROL:
        return str(args.case)
    if projected_density is None:
        raise ValueError("import-density label requires the projected state")
    if args.rho_file is not None:
        state_hash = density_state_audit(projected_density)[
            "density_state_sha256"
        ]
        return f"import_density_{state_hash[:12]}"
    assert args.rho is not None
    token = f"{args.rho:.8f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"import_rho_{token}"


def _output_paths(
    args: argparse.Namespace,
    spec: LumericalMeshSpec,
    projected_density: np.ndarray | None = None,
    exact_binary_mask: np.ndarray | None = None,
) -> tuple[Path, Path, Path, Path]:
    output = args.output_dir
    if output is None:
        output = (
            DEFAULT_RAW_ROOT
            / "au_dualpol_4um_lumerical"
            / spec.label
            / f"{_case_label(args, projected_density, exact_binary_mask)}_{args.polarization}"
        )
    output = output.expanduser().resolve()
    try:
        output.relative_to(REPOSITORY.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "raw Lumerical outputs must be outside the Git worktree: "
            f"{output}"
        )
    stem = (
        f"{_case_label(args, projected_density, exact_binary_mask)}_"
        f"{args.polarization}_{spec.label}"
    )
    return (
        output,
        output / f"{stem}.fsp",
        output / f"{stem}_raw.npz",
        output / f"{stem}.json",
    )


def _load_source_record(
    path: Path | None,
    expected_contract: dict[str, Any],
    *,
    expected_accelerator_policy: str,
    expected_gpu_uuid: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    resolved = path.expanduser().resolve()
    record = json.loads(resolved.read_text(encoding="utf-8"))
    validation = validate_source_calibration_record(
        record,
        expected_contract,
        expected_accelerator_policy=expected_accelerator_policy,
        expected_gpu_uuid=expected_gpu_uuid,
    )
    validation["path"] = str(resolved)
    validation["sha256"] = _sha256(resolved)
    if not validation["passed"]:
        raise RuntimeError(f"source calibration gate failed: {validation}")
    return record, validation


def _configure_environment(gpu_index: int, threads: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() != str(gpu_index):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES does not match the requested physical GPU: "
            f"{visible!r} != {gpu_index}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    gpu_device = f"GPU {gpu_index}"
    os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = gpu_device
    os.environ["CL_GPU_DEVICE"] = gpu_device
    os.environ["FDTD_THREADS"] = str(threads)
    os.environ["VC_LUMERICAL_ROOT"] = str(LUMERICAL_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(LUMERICAL_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(LUMAPI_PATH.parent)
    os.environ["PATH"] = f"{LUMERICAL_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
    return gpu_device


def _configure_gpu_resource(fdtd: Any, gpu_device: str, threads: int) -> None:
    fdtd.setresource("FDTD", 1, "active", 0)
    fdtd.setresource("FDTD", 2, "active", 1)
    fdtd.setresource("FDTD", 2, "processes", "1")
    fdtd.setresource("FDTD", 2, "threads", str(threads))
    fdtd.setresource("FDTD", 2, "device type", gpu_device)
    fdtd.setresource("FDTD", 2, "solver extra command line options", "-gpu")


def _gpu_log_evidence(
    output: Path,
    *,
    requested_gpu_index: int,
    requested_gpu_uuid: str | None,
) -> dict[str, Any]:
    logs = sorted(output.glob("*.log"))
    text = "\n".join(path.read_text(errors="replace") for path in logs)
    configured = re.findall(
        r"Configured with CUDA_VISIBLE_DEVICES=(\d+)", text
    )
    detected = re.findall(r"Detected GPU\s+(\d+):", text)
    detected_uuid = re.findall(
        r"Detected GPU\s+\d+:.*?\(UUID:\s*([^)]+)\)", text
    )
    expected_uuid = (
        _normalized_gpu_uuid(requested_gpu_uuid) if requested_gpu_uuid else None
    )
    logged_uuid = (
        _normalized_gpu_uuid(detected_uuid[-1]) if detected_uuid else None
    )
    evidence = {
        "log_paths": [str(path) for path in logs],
        "requested_gpu_index": requested_gpu_index,
        "configured_cuda_visible_devices": (
            int(configured[-1]) if configured else None
        ),
        "detected_gpu_index": int(detected[-1]) if detected else None,
        "requested_gpu_uuid": requested_gpu_uuid,
        "detected_gpu_uuid": detected_uuid[-1] if detected_uuid else None,
        "detected_gpu_uuid_matches_inventory": bool(
            expected_uuid is not None and logged_uuid == expected_uuid
        ),
        "engine_command_contains_gpu_flag": bool(
            re.search(r"fdtd-engine.*(?:^|\s)-gpu(?:\s|$)", text, re.MULTILINE)
        ),
        "gpu_timestep_timing_present": "time to run GPU simulation" in text,
        "gpu_datatype_present": "Using datatype: real32" in text,
        "simulation_completed_successfully": (
            "Simulation completed successfully" in text
        ),
    }
    evidence["passed"] = bool(
        logs
        and evidence["configured_cuda_visible_devices"] == requested_gpu_index
        and (
            evidence["detected_gpu_index"] == requested_gpu_index
            or evidence["detected_gpu_uuid_matches_inventory"]
        )
        and evidence["engine_command_contains_gpu_flag"]
        and evidence["gpu_timestep_timing_present"]
        and evidence["simulation_completed_successfully"]
    )
    return evidence


def _completed_run_wall_time_s(output: Path) -> float | None:
    """Read the solver-reported wall time without estimating from process time."""

    values: list[float] = []
    for path in sorted(output.glob("*.log")):
        text = path.read_text(errors="replace")
        values.extend(
            float(value)
            for value in re.findall(
                r"Overall wall time measurements in seconds:\s*"
                r"([0-9]+(?:\.[0-9]+)?)",
                text,
            )
        )
    return values[-1] if values else None


def _exact_layout_audit_without_mutation(
    args: argparse.Namespace,
    spec: LumericalMeshSpec,
    source_contract: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild deterministic metadata without changing a loaded result session."""

    if args.case not in GEOMETRY_CONTROLS:
        raise ValueError("completed-FSP recovery requires an exact geometry control")
    mask = exact_control_masks()[args.case]
    x_edges, y_edges = design_edges()
    z_bounds = np.asarray([0.0, CONTRACT.design_thickness_m])
    bounds = control_volume_bounds(spec)
    faces = {
        f"{axis}_{side}": {
            "name": f"au_dualpol_4um_flux_{axis}_{side}",
            "axis": axis,
            "side": side,
            "outward_sign": -1.0 if side == "min" else 1.0,
        }
        for axis in "xyz"
        for side in ("min", "max")
    }
    design_rectangle_count = len(
        mask_rectangles(
            mask,
            x_edges_m=x_edges,
            y_edges_m=y_edges,
            z_bounds_m=z_bounds,
        )
    )
    electrode_bounds = measurement_electrode_bounds()
    geometry = {
        "status": "PROVISIONAL_UNCONFIRMED_DEVICE_GEOMETRY",
        "exact_au_geometry": control_geometry_audits()[args.case],
        "Au_rectangle_count": design_rectangle_count + len(electrode_bounds),
        "design_Au_rectangle_count": design_rectangle_count,
        "fixed_measurement_electrode_rectangle_count": len(electrode_bounds),
        "measurement_electrodes": {
            side: {axis: list(values) for axis, values in bounds.items()}
            for side, bounds in electrode_bounds.items()
        },
        "measurement_electrode_material": CONTRACT.measurement_electrode_material,
        "measurement_electrodes_are_fixed_not_design_variables": True,
        "layers_z_m": {
            "Si": [spec.z_min_m, -385.0e-9],
            "SiO2": [-385.0e-9, -100.0e-9],
            "TaIrTe4": [-100.0e-9, 0.0],
            "Au": [0.0, CONTRACT.design_thickness_m],
        },
        "device_confirmation_required": True,
    }
    return {
        "case": args.case,
        "classification": (
            "provisional exact-Au or relaxed-density Maxwell/Q control; "
            "no thermal, electrical, PTE, adjoint, or optimization solve"
        ),
        "source_calibration_contract": source_contract,
        "material_input_audit": material_contract_audit(
            au_max_coefficients=args.au_max_coefficients,
            au_fit_tolerance=args.au_fit_tolerance,
        ),
        "geometry": geometry,
        "control_volume_bounds_m": {
            axis: list(values) for axis, values in bounds.items()
        },
        "flux_faces": faces,
        "adjoint_field_region": None,
    }


def _face_fluxes(
    fdtd: Any, faces: dict[str, dict[str, Any]], source_power_w: float
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    net_outward = 0.0
    for key, face in faces.items():
        normalized = _scalar(fdtd.transmission(face["name"]), face["name"])
        signed_axis_power = normalized * source_power_w
        outward = float(face["outward_sign"]) * signed_axis_power
        result[key] = {
            "normalized_signed_axis_flux": normalized,
            "signed_axis_power_W": signed_axis_power,
            "outward_power_W": outward,
        }
        net_outward += outward
    return {
        "faces": result,
        "net_outward_power_W": net_outward,
        "net_inward_power_W": -net_outward,
    }


def _source_postprocess(
    fdtd: Any, source_power_w: float
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, bool]]:
    fields = lumerical_audit.monitor_fields(fdtd, TARGET_MONITOR)
    target_metrics, target_arrays = lumerical_audit.plane_metrics(
        fields, source_power_w
    )
    source_result = fdtd.getresult(SOURCE_NAME, "fields")
    source_metrics, source_arrays = lumerical_audit.source_profile_from_arrays(
        source_result["x"], source_result["y"], source_result["E"]
    )
    target_w0 = CONTRACT.gaussian_waist_m
    gates = {
        "realized_waist_x_within_0p5pct": abs(
            target_metrics["fitted_waist_x_m"] - target_w0
        )
        / target_w0
        < SOURCE_PROFILE_GATE,
        "realized_waist_y_within_0p5pct": abs(
            target_metrics["fitted_waist_y_m"] - target_w0
        )
        / target_w0
        < SOURCE_PROFILE_GATE,
        "Gaussian_fit_NRMSE_lt_0p5pct": (
            target_metrics["Gaussian_fit_NRMSE"] < SOURCE_PROFILE_GATE
        ),
        "realized_ellipticity_lt_1pct": (
            target_metrics["fitted_xy_ellipticity"] < 1.0e-2
        ),
        "center_displacement_lt_50nm": (
            target_metrics["beam_center_error_m"] < 50.0e-9
        ),
        "incident_power_closure_lt_0p5pct": abs(
            target_metrics["downward_Poynting_power_over_sourcepower"] - 1.0
        )
        < RELATIVE_GATE,
        "all_fields_finite": bool(target_metrics["all_fields_finite"]),
    }
    arrays = {
        **{f"target_{key}": value for key, value in target_arrays.items()},
        **source_arrays,
    }
    return {
        "target_plane_metrics": target_metrics,
        "source_object_metrics": source_metrics,
    }, arrays, gates


def _endpoint_field_postprocess(
    fdtd: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read a fixed air-side field plane without impedance assumptions."""

    fields = lumerical_audit.monitor_fields(fdtd, ENDPOINT_FIELD_MONITOR)
    x = np.asarray(fields["coordinates"]["x"], float).reshape(-1)
    y = np.asarray(fields["coordinates"]["y"], float).reshape(-1)
    electric = {
        axis: np.asarray(fields["electric"][axis]).squeeze() for axis in "xyz"
    }
    expected = (x.size, y.size)
    if any(value.shape != expected for value in electric.values()):
        raise RuntimeError(
            "unexpected endpoint-field shapes: "
            f"{[value.shape for value in electric.values()]} != {expected}"
        )
    e2 = np.asarray(sum(np.abs(value) ** 2 for value in electric.values()), float)
    finite = bool(
        np.all(np.isfinite(e2))
        and all(np.all(np.isfinite(value)) for value in electric.values())
    )
    metrics = {
        "shape_xy": list(e2.shape),
        "z_m": float(np.asarray(fields["coordinates"]["z"]).reshape(-1)[0]),
        "mean_E2_V2_m2": float(np.mean(e2)),
        "maximum_E2_V2_m2": float(np.max(e2)),
        "component_L2": {
            axis: float(np.linalg.norm(value)) for axis, value in electric.items()
        },
        "all_fields_finite": finite,
        "normalization": "raw common source; no field rescaling",
    }
    arrays = {
        "endpoint_field_x_m": x,
        "endpoint_field_y_m": y,
        "endpoint_field_E2_V2_m2": e2,
        **{
            f"endpoint_field_E{axis}_V_m": value
            for axis, value in electric.items()
        },
    }
    return metrics, arrays


def _material_postprocess(
    fdtd: Any,
    *,
    flux_faces: dict[str, dict[str, Any]],
    source_power_w: float,
    source_incident_power_w: float,
    au_mask: np.ndarray | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, bool]]:
    six_face = _face_fluxes(fdtd, flux_faces, source_power_w)
    fdtd.runanalysis(PABS_GROUP)
    q = extract_native_yee_q(
        fdtd,
        field_monitor=PABS_FIELD,
        index_monitor=PABS_INDEX,
        wavelength_m=CONTRACT.wavelength_m,
    )
    p_native = float(q["P_Q_W"])
    p_pabs = _scalar(
        fdtd.getresult(PABS_GROUP, "Pabs_total")["Pabs_total"],
        "Pabs_total",
    ) * source_power_w
    official_pabs = fdtd.getresult(PABS_GROUP, "Pabs")
    official_index = fdtd.getresult(PABS_INDEX, "index")
    official_coordinates = {
        axis: np.asarray(official_pabs[axis], dtype=np.float64).reshape(-1)
        for axis in "xyz"
    }
    official_shape = tuple(official_coordinates[axis].size for axis in "xyz")
    official_coordinate_difference_m: dict[str, float] = {}
    for axis in "xyz":
        index_axis = np.asarray(official_index[axis], dtype=np.float64).reshape(-1)
        difference = float(
            np.max(np.abs(index_axis - official_coordinates[axis]))
        )
        official_coordinate_difference_m[axis] = difference
        if difference > 1.0e-18:
            raise RuntimeError(
                f"official Pabs/index {axis} coordinates differ by {difference} m"
            )
    official_pabs_W_m3 = np.asarray(
        _single_frequency_cube(
            official_pabs["Pabs"], official_shape, "official Pabs"
        ),
        dtype=np.float64,
    ) * source_power_w
    official_index_x = np.asarray(
        _single_frequency_cube(
            official_index["index_x"], official_shape, "official index_x"
        ),
        dtype=np.complex128,
    )
    if not np.all(np.isfinite(official_pabs_W_m3)):
        raise RuntimeError("official spatial Pabs contains NaN or Inf")
    if not np.all(np.isfinite(official_index_x)):
        raise RuntimeError("official index_x contains NaN or Inf")
    official_negative_power = integrate_xyz(
        np.where(official_pabs_W_m3 < 0.0, -official_pabs_W_m3, 0.0),
        official_coordinates["x"],
        official_coordinates["y"],
        official_coordinates["z"],
    )
    official_spatial_total = integrate_xyz(
        official_pabs_W_m3,
        official_coordinates["x"],
        official_coordinates["y"],
        official_coordinates["z"],
    )
    official_spatial_error = abs(official_spatial_total - p_pabs) / max(
        abs(p_pabs), np.finfo(float).tiny
    )
    official_negative_relative = official_negative_power / max(
        abs(official_spatial_total), np.finfo(float).tiny
    )
    p_six = float(six_face["net_inward_power_W"])
    closure = abs(p_native - p_six) / max(
        abs(p_native), abs(p_six), np.finfo(float).tiny
    )
    native_pabs_error = abs(p_native - p_pabs) / max(
        abs(p_native), abs(p_pabs), np.finfo(float).tiny
    )
    negative = {
        component: int(
            np.count_nonzero(np.asarray(q["Q_components"][component]) < 0.0)
        )
        for component in "xyz"
    }
    finite = all(
        np.all(np.isfinite(np.asarray(q["Q_components"][component])))
        for component in "xyz"
    )
    endpoint_field, arrays = _endpoint_field_postprocess(fdtd)
    arrays.update(
        Pabs_W_m3=official_pabs_W_m3,
        Pabs_index_x=official_index_x,
        **{
            f"Pabs_{axis}_m": official_coordinates[axis]
            for axis in "xyz"
        },
    )
    material_power: dict[str, dict[str, float]] = {}
    spatial_shape = tuple(
        np.asarray(q["base_coordinates"][axis]).size for axis in "xyz"
    )
    for component in "xyz":
        arrays[f"Q{component}_W_m3"] = np.asarray(
            q["Q_components"][component]
        )
        for axis in "xyz":
            arrays[f"Q{component}_{axis}_m"] = np.asarray(
                q["native_coordinates"][component][axis]
            )
        refractive_index = frequency_slice(
            np.asarray(fdtd.getdata(PABS_INDEX, f"index_{component}", 1)),
            spatial_shape,
            int(q["frequency_index_zero_based"]),
            int(q["frequency_count"]),
            f"index_{component}",
        )
        arrays[f"epsilon_{component}"] = np.asarray(refractive_index) ** 2
        if au_mask is not None:
            partitions = coordinate_material_partition(
                q["native_coordinates"][component], au_mask
            )
            for material, partition in partitions.items():
                material_power.setdefault(material, {})[component] = integrate_xyz(
                    np.asarray(q["Q_components"][component]) * partition,
                    q["native_coordinates"][component]["x"],
                    q["native_coordinates"][component]["y"],
                    q["native_coordinates"][component]["z"],
                )
    material_power_total = {
        material: float(sum(components.values()))
        for material, components in material_power.items()
    }
    partition_closure = (
        abs(sum(material_power_total.values()) - p_native)
        / max(abs(p_native), np.finfo(float).tiny)
        if au_mask is not None
        else None
    )
    reporting_scale = CONTRACT.reporting_incident_power_W / source_incident_power_w
    gates = {
        "native_Yee_Q_finite": bool(finite),
        "native_Yee_Q_nonnegative": sum(negative.values()) == 0,
        "Q_vs_six_face_flux_lt_2pct": closure < Q_FLUX_GATE,
        "native_Q_vs_pabs_analysis_lt_0p5pct": (
            native_pabs_error < RELATIVE_GATE
        ),
        "official_spatial_Pabs_closes_Pabs_total_lt_1e-12": (
            official_spatial_error < 1.0e-12
        ),
        "official_Pabs_negative_interpolation_artifact_lt_1e-12": (
            official_negative_relative < 1.0e-12
        ),
        "positive_source_calibration_power": source_incident_power_w > 0.0,
        "endpoint_field_finite": bool(endpoint_field["all_fields_finite"]),
        "coordinate_material_partition_closes_native_Q_or_not_applicable": (
            partition_closure is None or partition_closure < 1.0e-12
        ),
    }
    metrics = {
        "P_Q_native_W_raw": p_native,
        "P_Q_pabs_W_raw": p_pabs,
        "P_six_face_W_raw": p_six,
        "six_face_closure_relative": closure,
        "native_vs_pabs_relative": native_pabs_error,
        "official_spatial_Pabs_W_raw": official_spatial_total,
        "official_spatial_Pabs_vs_total_relative": official_spatial_error,
        "official_spatial_Pabs_negative_sample_count": int(
            np.count_nonzero(official_pabs_W_m3 < 0.0)
        ),
        "official_spatial_Pabs_negative_magnitude_W": official_negative_power,
        "official_spatial_Pabs_negative_relative": official_negative_relative,
        "official_Pabs_vs_index_coordinate_max_abs_difference_m": (
            official_coordinate_difference_m
        ),
        "Q_component_power_native_W_raw": q["component_power_W"],
        "coordinate_partition_component_power_W_raw": material_power,
        "coordinate_partition_material_power_W_raw": material_power_total,
        "coordinate_partition_closure_relative": partition_closure,
        "coordinate_partition_scope": (
            "not applicable to gray imported density"
            if au_mask is None
            else "deterministic native-sample convergence diagnostic; interface "
            "cut cells remain represented by the saved raw epsilon arrays"
        ),
        "endpoint_field": endpoint_field,
        "component_hotspots": q["component_hotspots"],
        "negative_Q_cell_count": negative,
        "all_Q_arrays_finite": bool(finite),
        "six_face": six_face,
        "reporting_normalization": {
            "source_only_incident_power_W_raw": source_incident_power_w,
            "target_reporting_incident_power_W": CONTRACT.reporting_incident_power_W,
            "scalar_reporting_factor": reporting_scale,
            "P_Q_W_at_reporting_power": p_native * reporting_scale,
            "only_scalar_reports_scaled": True,
            "field_or_Q_array_rescaling": False,
        },
        "Q_processing": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "field_or_Q_rescaling": False,
            "global_rescaling": False,
            "tiling": False,
        },
    }
    return metrics, arrays, gates


def main() -> int:
    require_lumerical_only_source_boundary()
    args = _parse_args()
    spec = _mesh_spec(args)
    projected_density, density_input = _projected_density(args)
    exact_binary_mask, binary_input = _exact_binary_mask(args)
    source_object_w0_m = args.source_object_w0_um * 1.0e-6
    source_contract = source_calibration_contract(
        spec,
        args.polarization,
        source_object_w0_m=source_object_w0_m,
    )
    audit_payload = {
        "status": "AUDITED_EXACT_AU_4UM_CONTROL_NOT_RUN",
        "case": args.case,
        "case_label": _case_label(args, projected_density, exact_binary_mask),
        "projected_density": args.rho,
        "projected_density_input": density_input,
        "exact_binary_input": binary_input,
        "polarization": args.polarization,
        "mesh_spec": spec.audit(),
        "source_calibration_contract": source_contract,
        "environment": audit_environment(
            requested_gpu_index=args.gpu_index,
            accelerator_policy=args.accelerator_policy,
        ),
        "scope": "audit only; no Maxwell or downstream PDE solve",
        "include_adjoint_field_region": args.include_adjoint_field_region,
    }
    if args.audit_only:
        print(json.dumps(audit_payload, indent=2, default=_json_default))
        return 0

    assert args.gpu_index is not None
    preflight = require_lumerical_gpu(
        requested_gpu_index=args.gpu_index,
        accelerator_policy=args.accelerator_policy,
    )
    selected_gpu = preflight["matching_requested_gpu"]
    if len(selected_gpu) != 1:
        raise RuntimeError(f"requested GPU inventory is ambiguous: {selected_gpu}")
    requested_gpu_uuid = str(selected_gpu[0]["uuid"])
    source_record, source_validation = _load_source_record(
        args.source_calibration_json,
        source_contract,
        expected_accelerator_policy=args.accelerator_policy,
        expected_gpu_uuid=requested_gpu_uuid,
    )
    output, fsp_path, npz_path, result_path = _output_paths(
        args, spec, projected_density, exact_binary_mask
    )
    protected_paths = (
        (npz_path, result_path)
        if args.recover_completed_fsp
        else (fsp_path, npz_path, result_path)
    )
    for protected in protected_paths:
        if protected.exists():
            raise RuntimeError(f"refusing to overwrite raw artifact: {protected}")
    if args.recover_completed_fsp and not fsp_path.is_file():
        raise RuntimeError(f"completed FSP does not exist: {fsp_path}")
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "BLOCKED_EXACT_AU_4UM_CONTROL",
        "case": args.case,
        "case_label": _case_label(args, projected_density, exact_binary_mask),
        "projected_density": args.rho,
        "projected_density_input": density_input,
        "exact_binary_input": binary_input,
        "polarization": args.polarization,
        "git_commit": _git_commit(),
        "mesh_spec": spec.audit(),
        "source_calibration_contract": source_contract,
        "source_calibration_sha256": source_contract[
            "source_calibration_sha256"
        ],
        "source_calibration_validation": source_validation,
        "accelerator_policy": args.accelerator_policy,
        "accelerator_preflight": preflight,
        "B200_preflight": (
            preflight if args.accelerator_policy == "b200" else None
        ),
        "B200_promotion_certified": bool(preflight["b200_promotion_certified"]),
        "physical_device_contract": {
            "path": str(PHYSICAL_DEVICE_CONTRACT.relative_to(REPOSITORY)),
            "sha256": _sha256(PHYSICAL_DEVICE_CONTRACT),
            "status": json.loads(
                PHYSICAL_DEVICE_CONTRACT.read_text(encoding="utf-8")
            ).get("status"),
        },
        "Q_processing": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "field_or_Q_rescaling": False,
        },
        "execution_mode": (
            "recover_completed_fsp_without_Maxwell_rerun"
            if args.recover_completed_fsp
            else "new_Maxwell_run"
        ),
        "include_adjoint_field_region": args.include_adjoint_field_region,
    }
    fdtd = None
    try:
        gpu_device = _configure_environment(args.gpu_index, args.threads)
        sys.path.insert(0, str(LUMAPI_PATH.parent))
        import lumapi

        if args.recover_completed_fsp:
            log = lumerical_audit.log_audit(output)
            gpu_log = _gpu_log_evidence(
                output,
                requested_gpu_index=args.gpu_index,
                requested_gpu_uuid=requested_gpu_uuid,
            )
            if not log["simulation_completed_successfully"]:
                raise RuntimeError(
                    "refusing completed-FSP recovery because the engine log "
                    "does not prove successful completion"
                )
            if not gpu_log["passed"]:
                raise RuntimeError(
                    "refusing completed-FSP recovery because the engine log "
                    "does not prove the requested GPU execution"
                )
            fdtd = lumapi.FDTD(
                str(fsp_path),
                hide=True,
                serverArgs={"platform": "offscreen"},
            )
        else:
            fdtd = lumapi.FDTD(
                hide=True, serverArgs={"platform": "offscreen"}
            )
        solver_version = str(fdtd.version())
        result["solver_version"] = solver_version
        if source_record is not None:
            assert source_validation is not None
            source_validation["gates"]["solver_version_matches"] = (
                source_record.get("solver_version") == solver_version
            )
            source_validation["passed"] = all(
                source_validation["gates"].values()
            )
            if not source_validation["passed"]:
                raise RuntimeError(
                    "source calibration solver-version gate failed: "
                    f"{source_validation}"
                )
        if args.recover_completed_fsp:
            layout = _exact_layout_audit_without_mutation(
                args, spec, source_contract
            )
            pre_mesh = lumerical_audit.mesh_readback(fdtd)
            if not pre_mesh.get("available"):
                raise RuntimeError(
                    f"completed-FSP mesh readback unavailable: {pre_mesh}"
                )
            mesh_gate = requested_mesh_readback_gates(
                pre_mesh["coordinate_arrays"], spec
            )
            if not mesh_gate["all"]:
                raise RuntimeError(
                    f"completed FSP does not realize requested mesh: {mesh_gate}"
                )
            dt_s = _scalar(fdtd.getnamed("FDTD", "dt"), "FDTD.dt")
            material_readback = material_fit_readback(fdtd, dt_s=dt_s)
            if not all(material_readback["gates"].values()):
                raise RuntimeError(
                    f"material fit readback failed: {material_readback}"
                )
            resource = "recovered completed FSP; GPU proven by original engine log"
            wall_time_s = _completed_run_wall_time_s(output)
            result["completed_FSP_recovery"] = {
                "FSP_loaded": str(fsp_path),
                "original_solver_wall_time_source": "engine log",
                "Maxwell_run_called": False,
                "runsetup_called": False,
                "FSP_save_called": False,
            }
        else:
            layout = build_layout(
                fdtd,
                case=args.case,
                polarization=args.polarization,
                spec=spec,
                source_object_w0_m=source_object_w0_m,
                projected_density=projected_density,
                exact_binary_mask=exact_binary_mask,
                include_adjoint_field_region=args.include_adjoint_field_region,
                au_max_coefficients=args.au_max_coefficients,
                au_fit_tolerance=args.au_fit_tolerance,
            )
            _configure_gpu_resource(fdtd, gpu_device, args.threads)
            fdtd.runsetup()
            pre_mesh = lumerical_audit.mesh_readback(fdtd)
            if not pre_mesh.get("available"):
                raise RuntimeError(f"pre-run mesh readback unavailable: {pre_mesh}")
            mesh_gate = requested_mesh_readback_gates(
                pre_mesh["coordinate_arrays"], spec
            )
            if not mesh_gate["all"]:
                raise RuntimeError(f"requested mesh was not realized: {mesh_gate}")
            material_readback = None
            if args.case != "source_only":
                dt_s = _scalar(fdtd.getnamed("FDTD", "dt"), "FDTD.dt")
                material_readback = material_fit_readback(fdtd, dt_s=dt_s)
                if not all(material_readback["gates"].values()):
                    raise RuntimeError(
                        f"material fit readback failed: {material_readback}"
                    )
            fdtd.save(str(fsp_path))
            started = time.monotonic()
            resource = lumerical_audit.strict_gpu_run(
                fdtd, f"au_dualpol_4um_{args.case}_{args.polarization}"
            )
            wall_time_s = time.monotonic() - started
            fdtd.save(str(fsp_path))
            log = lumerical_audit.log_audit(output)
            gpu_log = _gpu_log_evidence(
                output,
                requested_gpu_index=args.gpu_index,
                requested_gpu_uuid=requested_gpu_uuid,
            )
        result.update(
            {
                "GPU_resource_used": resource,
                "solver_wall_time_s": wall_time_s,
                "log_audit": log,
                "GPU_log_evidence": gpu_log,
            }
        )
        if not log["simulation_completed_successfully"]:
            raise RuntimeError(
                "Lumerical solver did not complete successfully; inspect "
                f"{log['logs']}"
            )
        frequency_hz = C0_M_S / CONTRACT.wavelength_m
        source_power_w = _scalar(
            fdtd.sourcepower(frequency_hz, 2, SOURCE_NAME), "sourcepower"
        )
        post_mesh = lumerical_audit.mesh_readback(fdtd)
        common_gates = {
            "accelerator_preflight_passed": preflight["status"].startswith("READY"),
            "engine_log_proves_requested_GPU": bool(gpu_log["passed"]),
            "simulation_completed_successfully": bool(
                log["simulation_completed_successfully"]
            ),
            "auto_shutoff_lt_1e_5": (
                log["final_auto_shutoff"] is not None
                and log["final_auto_shutoff"] < 1.0e-5
            ),
            "pre_run_mesh_readback_passed": bool(mesh_gate["all"]),
            "post_run_mesh_readback_available": bool(post_mesh.get("available")),
        }
        if args.recover_completed_fsp:
            common_gates[
                "completed_FSP_recovered_without_Maxwell_rerun"
            ] = True
        arrays: dict[str, np.ndarray] = {
            f"pre_mesh_{axis}_m": np.asarray(
                pre_mesh["coordinate_arrays"][axis]
            )
            for axis in "xyz"
        }
        if projected_density is not None:
            density_x, density_y, _ = density_nodes()
            arrays.update(
                projected_density_nodal=np.asarray(projected_density),
                projected_density_x_m=density_x,
                projected_density_y_m=density_y,
            )
        if exact_binary_mask is not None:
            x_edges, y_edges = design_edges()
            arrays.update(
                exact_binary_cell_mask=np.asarray(exact_binary_mask),
                exact_binary_x_edges_m=x_edges,
                exact_binary_y_edges_m=y_edges,
            )
        if post_mesh.get("available"):
            arrays.update(
                {
                    f"post_mesh_{axis}_m": np.asarray(
                        post_mesh["coordinate_arrays"][axis]
                    )
                    for axis in "xyz"
                }
            )
        if args.case == "source_only":
            metrics, case_arrays, case_gates = _source_postprocess(
                fdtd, source_power_w
            )
        else:
            assert source_record is not None and source_validation is not None
            incident_power = float(source_validation["incident_power_W"])
            metrics, case_arrays, case_gates = _material_postprocess(
                fdtd,
                flux_faces=layout["flux_faces"],
                source_power_w=source_power_w,
                source_incident_power_w=incident_power,
                au_mask=(
                    None
                    if args.case == DENSITY_CONTROL
                    else (
                        exact_binary_mask
                        if args.case == EXACT_BINARY_CONTROL
                        else exact_control_masks()[args.case]
                    )
                ),
            )
            case_gates.update(
                matching_source_calibration_passed=bool(
                    source_validation["passed"]
                ),
                material_fit_readback_passed=bool(
                    material_readback is not None
                    and all(material_readback["gates"].values())
                ),
            )
            if args.case == DENSITY_CONTROL:
                case_gates["canonical_density_state_hash_present"] = bool(
                    layout["geometry"]["density_state"]["density_state_sha256"]
                )
            else:
                case_gates["canonical_exact_Au_geometry_hash_present"] = bool(
                    layout["geometry"]["exact_au_geometry"]["geometry_sha256"]
                )
        arrays.update(case_arrays)
        np.savez_compressed(npz_path, **arrays)
        all_gates = {**common_gates, **case_gates}
        passed = all(all_gates.values())
        status = (
            "PASSED_EXACT_AU_4UM_SOURCE_ONLY_NUMERICAL_GATE"
            if args.case == "source_only" and passed
            else (
                f"PASSED_PROVISIONAL_LUMERICAL_4UM_{_case_label(args, projected_density, exact_binary_mask)}_{args.polarization}_CONTROL"
                if passed
                else f"FAILED_LUMERICAL_4UM_{_case_label(args, projected_density, exact_binary_mask)}_{args.polarization}_GATE"
            )
        )
        if passed and args.accelerator_policy != "b200":
            status += "_DEVELOPMENT_GPU_NOT_B200_CERTIFIED"
        result.update(
            {
                "status": status,
                "all_gates_passed": passed,
                "promotion_to_physical_device_result": False,
                "provisional_device_contract_confirmation_required": True,
                "solver_version": solver_version,
                "GPU_resource_used": resource,
                "solver_wall_time_s": wall_time_s,
                "source_power_W_raw": source_power_w,
                "layout": layout,
                "material_fit_readback": material_readback,
                "mesh_region_readback": mesh_gate,
                "mesh_readback": {
                    "pre_run": {
                        key: value
                        for key, value in pre_mesh.items()
                        if key != "coordinate_arrays"
                    },
                    "post_run": {
                        key: value
                        for key, value in post_mesh.items()
                        if key != "coordinate_arrays"
                    },
                },
                "log_audit": log,
                "GPU_log_evidence": gpu_log,
                "gates": all_gates,
                **metrics,
            }
        )
        result["raw_artifacts"] = [_artifact(fsp_path), _artifact(npz_path)]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        _write_json(result_path, result)
    print(json.dumps(result, indent=2, default=_json_default))
    return 0 if str(result["status"]).startswith("PASSED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
