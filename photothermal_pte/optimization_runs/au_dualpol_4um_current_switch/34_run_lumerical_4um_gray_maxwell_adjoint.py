#!/usr/bin/env python3
"""Run one frozen-grid Lumerical adjoint for the gray Au PTE current.

The completed forward and custom-CUDA pullback are reused. This invocation
runs zero forward Maxwell solves and exactly one distributed-source adjoint.
It prepares a physical-density gradient but does not start an optimizer and
does not claim AD--FD until the independent centered-forward pair is run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_FIELD,
)
from photothermal_pte.finite_inverse_design.run_v261_large_background_mixed_optical_adfd import (  # noqa: E402
    fieldregion_profile,
    import_named_fieldregion_profile,
    monitor_electric,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_adjoint import (  # noqa: E402
    COMPONENTS,
    load_component_yee_jacobian,
    material_jacobian_reuse_audit,
    native_adjoint_source,
    optical_density_gradient,
    reconstruct_fieldregion_only_cw,
    validate_raw_against_jacobian,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (  # noqa: E402
    density_state_audit,
    load_projected_density_file,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_forward import (  # noqa: E402
    ADJOINT_FIELD_REGION,
    SOURCE_NAME,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_yee_jacobian import (  # noqa: E402
    validate_completed_density_record,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_maxwell_contract import (  # noqa: E402
    ACCELERATOR_POLICIES,
    LUMAPI_PATH,
    LUMERICAL_ROOT,
    require_lumerical_gpu,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (  # noqa: E402
    sha256,
)
from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    validate_paper_ir_source_only_gpu as lumerical_audit,
)


STATUS = "COMPLETED_LUMERICAL_4UM_GRAY_MAXWELL_ADJOINT_PREPARATION"


def _artifact(path: Path) -> dict[str, Any]:
    value = path.expanduser().resolve()
    return {
        "path": str(value),
        "size_bytes": value.stat().st_size,
        "sha256": sha256(value),
    }


def _matching_artifact(result: dict[str, Any], suffix: str) -> dict[str, Any]:
    matches = [
        item
        for item in result.get("raw_artifacts", [])
        if str(item.get("path", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {suffix} artifact in forward result")
    return matches[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-result", required=True, type=Path)
    parser.add_argument("--forward-fsp", required=True, type=Path)
    parser.add_argument("--forward-raw-npz", required=True, type=Path)
    parser.add_argument("--density-file", required=True, type=Path)
    parser.add_argument("--density-key", default="rho")
    parser.add_argument("--jacobian-dir", required=True, type=Path)
    parser.add_argument("--pde-result", required=True, type=Path)
    parser.add_argument("--pde-pullback-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument(
        "--accelerator-policy",
        choices=ACCELERATOR_POLICIES,
        default=os.environ.get("AU_LUMERICAL_ACCELERATOR_POLICY", "b200"),
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--artifact-audit", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not args.artifact_audit and args.gpu_index is None:
        value = os.environ.get("LUMERICAL_GPU_INDEX")
        if value is None:
            parser.error("adjoint run requires --gpu-index or LUMERICAL_GPU_INDEX")
        args.gpu_index = int(value)
    return args


def _configure_environment(gpu_index: int, threads: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() != str(gpu_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES differs from requested GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    device = f"GPU {gpu_index}"
    os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = device
    os.environ["CL_GPU_DEVICE"] = device
    os.environ["FDTD_THREADS"] = str(threads)
    os.environ["VC_LUMERICAL_ROOT"] = str(LUMERICAL_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(LUMERICAL_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(LUMAPI_PATH.parent)
    os.environ["PATH"] = f"{LUMERICAL_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
    return device


def _configure_gpu(fdtd: Any, device: str, threads: int) -> None:
    fdtd.setresource("FDTD", 1, "active", 0)
    fdtd.setresource("FDTD", 2, "active", 1)
    fdtd.setresource("FDTD", 2, "processes", "1")
    fdtd.setresource("FDTD", 2, "threads", str(threads))
    fdtd.setresource("FDTD", 2, "device type", device)
    fdtd.setresource("FDTD", 2, "solver extra command line options", "-gpu")


def _mesh_difference(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(
        float(
            np.max(
                np.abs(
                    np.asarray(left["coordinate_arrays"][axis], float)
                    - np.asarray(right["coordinate_arrays"][axis], float)
                )
            )
        )
        for axis in COMPONENTS
    )


def _grid_difference(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> float:
    return max(
        float(np.max(np.abs(np.asarray(left[key]) - np.asarray(right[key]))))
        for key in ("x", "y", "z", "delta_x", "delta_y", "delta_z")
    )


def _gpu_log_evidence(
    output: Path, *, requested_gpu_index: int, requested_gpu_uuid: str
) -> dict[str, Any]:
    logs = sorted(output.glob("*.log"))
    text = "\n".join(path.read_text(errors="replace") for path in logs)
    configured = re.findall(r"Configured with CUDA_VISIBLE_DEVICES=(\d+)", text)
    detected = re.findall(r"Detected GPU\s+(\d+):", text)
    detected_uuid = re.findall(r"Detected GPU\s+\d+:.*?\(UUID:\s*([^)]+)\)", text)
    normalized_expected = requested_gpu_uuid.removeprefix("GPU-").lower()
    normalized_detected = detected_uuid[-1].removeprefix("GPU-").lower() if detected_uuid else None
    evidence = {
        "log_paths": [str(path) for path in logs],
        "configured_cuda_visible_devices": int(configured[-1]) if configured else None,
        "detected_gpu_index": int(detected[-1]) if detected else None,
        "requested_gpu_uuid": requested_gpu_uuid,
        "detected_gpu_uuid": detected_uuid[-1] if detected_uuid else None,
        "detected_gpu_uuid_matches_inventory": normalized_detected == normalized_expected,
        "engine_command_contains_gpu_flag": bool(
            re.search(r"fdtd-engine.*(?:^|\s)-gpu(?:\s|$)", text, re.MULTILINE)
        ),
        "gpu_timestep_timing_present": "time to run GPU simulation" in text,
        "simulation_completed_successfully": "Simulation completed successfully" in text,
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


def _load_and_validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    forward_path = args.forward_result.expanduser().resolve()
    forward_fsp = args.forward_fsp.expanduser().resolve()
    forward_raw = args.forward_raw_npz.expanduser().resolve()
    density_path = args.density_file.expanduser().resolve()
    pde_path = args.pde_result.expanduser().resolve()
    pde_raw = args.pde_pullback_npz.expanduser().resolve()
    forward = json.loads(forward_path.read_text(encoding="utf-8"))
    pde = json.loads(pde_path.read_text(encoding="utf-8"))
    rho = load_projected_density_file(density_path, key=args.density_key)
    polarization = str(forward.get("polarization"))
    if polarization not in ("Ea", "Eb"):
        raise RuntimeError("forward polarization must be Ea or Eb")
    if pde.get("polarization") != polarization:
        raise RuntimeError("custom-CUDA PDE polarization differs from forward")
    fsp_record = _matching_artifact(forward, ".fsp")
    raw_record = _matching_artifact(forward, "_raw.npz")
    if forward_fsp != Path(fsp_record["path"]).resolve() or sha256(forward_fsp) != fsp_record["sha256"]:
        raise RuntimeError("adjoint-ready forward FSP path/SHA differs")
    if forward_raw != Path(raw_record["path"]).resolve() or sha256(forward_raw) != raw_record["sha256"]:
        raise RuntimeError("adjoint-ready forward raw path/SHA differs")
    binding = validate_completed_density_record(
        forward, rho, forward_fsp_sha256=str(fsp_record["sha256"])
    )
    if not binding["passed"]:
        raise RuntimeError("adjoint-ready forward density binding failed")
    region = forward.get("layout", {}).get("adjoint_field_region")
    if (
        forward.get("include_adjoint_field_region") is not True
        or not isinstance(region, dict)
        or region.get("name") != ADJOINT_FIELD_REGION
        or region.get("source_mode_during_forward") is not False
    ):
        raise RuntimeError("forward did not freeze the authorized adjoint FieldRegion")
    if pde.get("status") != "VALIDATED_LUMERICAL_4UM_GRAY_Q_CUSTOM_CUDA_PDE" or pde.get("passed") is not True:
        raise RuntimeError("custom-CUDA PDE pullback certificate did not pass")
    pde_record = pde.get("raw_output", {})
    if pde_raw != Path(pde_record.get("path", "")).resolve() or sha256(pde_raw) != pde_record.get("sha256"):
        raise RuntimeError("custom-CUDA PDE pullback path/SHA differs")
    operator, operator_meta = load_component_yee_jacobian(args.jacobian_dir, rho)
    source_forward_result = Path(operator_meta["source_forward_result"]["path"])
    if sha256(source_forward_result) != operator_meta["source_forward_result"]["sha256"]:
        raise RuntimeError("Jacobian source-forward JSON changed")
    source_forward = json.loads(source_forward_result.read_text(encoding="utf-8"))
    source_raw_record = _matching_artifact(source_forward, "_raw.npz")
    with np.load(forward_raw, allow_pickle=False) as raw:
        raw_binding = validate_raw_against_jacobian(raw, operator_meta)
        epsilon = {
            component: np.asarray(raw[f"epsilon_{component}"], np.complex128)
            for component in COMPONENTS
        }
    jacobian_reuse = material_jacobian_reuse_audit(
        raw_binding,
        source_raw_sha256=str(source_raw_record["sha256"]),
        target_raw_sha256=sha256(forward_raw),
        source_polarization=str(source_forward.get("polarization")),
        target_polarization=polarization,
    )
    if not jacobian_reuse["passed"]:
        raise RuntimeError("adjoint-ready forward Yee state differs from Jacobian")
    with np.load(pde_raw, allow_pickle=False) as pullback:
        if not np.array_equal(np.asarray(pullback["rho_nodal"], float), rho):
            raise RuntimeError("PDE pullback density differs")
        scale = float(pde["source_scale_to_reporting_power"])
        native_q_sensitivity_raw = {
            component: scale
            * np.asarray(
                pullback[f"native_Q{component}_sensitivity_A_m3_W_reporting"],
                float,
            )
            for component in COMPONENTS
        }
        gradient_direct_nodal = np.asarray(pullback["gradient_direct_nodal_A"], float)
    return {
        "forward": forward,
        "forward_path": forward_path,
        "forward_fsp": forward_fsp,
        "forward_raw": forward_raw,
        "density_path": density_path,
        "rho": rho,
        "binding": binding,
        "raw_binding": raw_binding,
        "operator": operator,
        "operator_meta": operator_meta,
        "epsilon": epsilon,
        "pde": pde,
        "pde_path": pde_path,
        "pde_raw": pde_raw,
        "polarization": polarization,
        "native_q_sensitivity_raw": native_q_sensitivity_raw,
        "gradient_direct_nodal": gradient_direct_nodal,
        "source_scale": scale,
        "jacobian_reuse": jacobian_reuse,
    }


def main() -> int:
    args = _parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "gray_maxwell_adjoint_result.json"
    result: dict[str, Any] = {
        "status": "FAILED_LUMERICAL_4UM_GRAY_MAXWELL_ADJOINT_PREPARATION",
        "passed": False,
        "Maxwell_forward_solves_this_invocation": 0,
        "Maxwell_adjoint_solves_this_invocation": 0,
        "Lumerical_HEAT_or_CHARGE_solves": 0,
        "optimizer_iterations": 0,
    }
    fdtd = None
    started = time.monotonic()
    try:
        inputs = _load_and_validate_inputs(args)
        if args.artifact_audit:
            result = {
                "status": "PASSED_LUMERICAL_4UM_GRAY_MAXWELL_ADJOINT_ARTIFACT_AUDIT",
                "passed": True,
                "polarization": inputs["polarization"],
                "density_state": density_state_audit(inputs["rho"]),
                "forward_binding": inputs["binding"],
                "forward_vs_jacobian": inputs["raw_binding"],
                "jacobian_reuse": inputs["jacobian_reuse"],
                "operator": {
                    key: value
                    for key, value in inputs["operator_meta"].items()
                    if key != "component_coordinates_m"
                },
                "Maxwell_forward_solves_this_invocation": 0,
                "Maxwell_adjoint_solves_this_invocation": 0,
                "Lumerical_HEAT_or_CHARGE_solves": 0,
                "optimizer_iterations": 0,
                "wall_s": time.monotonic() - started,
            }
        else:
            assert args.gpu_index is not None
            preflight = require_lumerical_gpu(
                requested_gpu_index=args.gpu_index,
                accelerator_policy=args.accelerator_policy,
            )
            selected = preflight["matching_requested_gpu"]
            if len(selected) != 1:
                raise RuntimeError("requested GPU inventory is ambiguous")
            requested_uuid = str(selected[0]["uuid"])
            device = _configure_environment(args.gpu_index, args.threads)
            sys.path.insert(0, str(LUMAPI_PATH.parent))
            import lumapi

            fdtd = lumapi.FDTD(
                str(inputs["forward_fsp"]),
                hide=True,
                serverArgs={"platform": "offscreen"},
            )
            if str(fdtd.version()) != str(inputs["forward"]["solver_version"]):
                raise RuntimeError("forward/adjoint Lumerical solver version differs")
            if int(fdtd.getnamednumber(ADJOINT_FIELD_REGION)) != 1:
                raise RuntimeError("adjoint-ready FSP lost its FieldRegion")
            if bool(fdtd.getnamed(ADJOINT_FIELD_REGION, "source mode")):
                raise RuntimeError("forward FieldRegion was unexpectedly in source mode")
            forward_electric, forward_grid = monitor_electric(fdtd, PABS_FIELD)
            if forward_electric.shape[:3] != inputs["operator"].component_shapes["x"]:
                raise RuntimeError("forward E and component-J shapes differ")
            source_native = native_adjoint_source(
                forward_electric,
                inputs["epsilon"],
                inputs["native_q_sensitivity_raw"],
            )
            profile, profile_scale = fieldregion_profile(source_native)
            forward_mesh = lumerical_audit.mesh_readback(fdtd)
            if not forward_mesh.get("available"):
                raise RuntimeError("completed forward mesh readback unavailable")
            fdtd.switchtolayout()
            original_amplitude = float(fdtd.getnamed(SOURCE_NAME, "amplitude"))
            fdtd.setnamed(SOURCE_NAME, "amplitude", 0.0)
            fdtd.setnamed(SOURCE_NAME, "enabled", True)
            for axis in COMPONENTS:
                fdtd.setnamed(
                    ADJOINT_FIELD_REGION,
                    f"{axis} min",
                    float(forward_grid[axis][0]),
                )
                fdtd.setnamed(
                    ADJOINT_FIELD_REGION,
                    f"{axis} max",
                    float(forward_grid[axis][-1]),
                )
            fdtd.setnamed(ADJOINT_FIELD_REGION, "source mode", True)
            try:
                fdtd.setnamed(ADJOINT_FIELD_REGION, "nuttall window pulse", False)
            except Exception:
                pass
            roundtrip = import_named_fieldregion_profile(
                fdtd, ADJOINT_FIELD_REGION, forward_grid, profile
            )
            base_amplitude = float(
                fdtd.getnamed(ADJOINT_FIELD_REGION, "base amplitude")
            )
            _configure_gpu(fdtd, device, args.threads)
            fdtd.runsetup()
            adjoint_mesh = lumerical_audit.mesh_readback(fdtd)
            if not adjoint_mesh.get("available"):
                raise RuntimeError("adjoint pre-run mesh readback unavailable")
            mesh_difference = _mesh_difference(forward_mesh, adjoint_mesh)
            template = output / "gray_maxwell_adjoint_template.fsp"
            fdtd.save(str(template))
            adjoint_fsp = output / "gray_maxwell_adjoint_gpu.fsp"
            fdtd.save(str(adjoint_fsp))
            solve_started = time.monotonic()
            resource = lumerical_audit.strict_gpu_run(
                fdtd,
                f"au_dualpol_4um_gray_{inputs['polarization']}_maxwell_adjoint",
            )
            solver_wall = time.monotonic() - solve_started
            result["Maxwell_adjoint_solves_this_invocation"] = 1
            fdtd.save(str(adjoint_fsp))
            log = lumerical_audit.log_audit(output)
            gpu_log = _gpu_log_evidence(
                output,
                requested_gpu_index=args.gpu_index,
                requested_gpu_uuid=requested_uuid,
            )
            fdtd.cwnorm(1)
            adjoint_first, adjoint_grid = monitor_electric(fdtd, PABS_FIELD)
            fdtd.cwnorm(2)
            adjoint_average, average_grid = monitor_electric(fdtd, PABS_FIELD)
            average_grid_difference = _grid_difference(adjoint_grid, average_grid)
            adjoint_electric, normalization = reconstruct_fieldregion_only_cw(
                adjoint_first, adjoint_average
            )
            forward_adjoint_grid_difference = _grid_difference(
                forward_grid, adjoint_grid
            )
            gradient_optical, optical = optical_density_gradient(
                inputs["operator"],
                forward_electric=forward_electric,
                adjoint_electric=adjoint_electric,
                epsilon=inputs["epsilon"],
                native_q_sensitivity_A_m3_W_raw=inputs[
                    "native_q_sensitivity_raw"
                ],
                grid=forward_grid,
                profile_scale=profile_scale,
                fieldregion_base_amplitude=base_amplitude,
            )
            gradient_direct = inputs["gradient_direct_nodal"]
            gradient_total = gradient_optical + gradient_direct
            finite_nonzero = bool(
                np.all(np.isfinite(gradient_optical))
                and np.all(np.isfinite(gradient_direct))
                and np.all(np.isfinite(gradient_total))
                and np.linalg.norm(gradient_optical) > 0.0
                and np.linalg.norm(gradient_total) > 0.0
            )
            gates = {
                "accelerator_preflight_passed": bool(preflight["all_required_gates_passed"]),
                "GPU_log_evidence_passed": bool(gpu_log["passed"]),
                "simulation_completed_successfully": bool(log["simulation_completed_successfully"]),
                "adjoint_auto_shutoff_lt_1e_5": log["final_auto_shutoff"] is not None
                and float(log["final_auto_shutoff"]) < 1.0e-5,
                "forward_vs_jacobian_state_passed": bool(inputs["raw_binding"]["passed"]),
                "forward_adjoint_solver_mesh_match_lt_2e_18": mesh_difference < 2.0e-18,
                "forward_adjoint_monitor_grid_match_lt_2e_18": forward_adjoint_grid_difference < 2.0e-18,
                "adjoint_cwnorm_grid_match_lt_2e_18": average_grid_difference < 2.0e-18,
                "source_profile_roundtrip_exact": roundtrip == 0.0,
                "cwnorm_reconstruction_residual_lt_1e_12": float(
                    normalization["two_normalization_state_spatial_residual"]
                )
                < 1.0e-12,
                "finite_nonzero_physical_density_gradient": finite_nonzero,
            }
            raw_output = output / "gray_maxwell_adjoint_gradient.npz"
            np.savez_compressed(
                raw_output,
                rho_nodal=inputs["rho"],
                gradient_optical_A=gradient_optical,
                gradient_optical_indirect_A=optical["indirect_gradient"],
                gradient_optical_direct_loss_A=optical["direct_loss_gradient"],
                gradient_direct_pde_A=gradient_direct,
                gradient_total_A=gradient_total,
            )
            passed = all(gates.values())
            result = {
                "status": STATUS if passed else "FAILED_LUMERICAL_4UM_GRAY_MAXWELL_ADJOINT_PREPARATION",
                "passed": passed,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": (
                    f"one completed nonuniform {inputs['polarization']} forward plus "
                    "custom-CUDA Q pullback through one frozen-grid distributed-source "
                    "Lumerical adjoint"
                ),
                "polarization": inputs["polarization"],
                "density_state": density_state_audit(inputs["rho"]),
                "current_A": float(inputs["pde"]["current_A"]),
                "source_scale_to_reporting_power": inputs["source_scale"],
                "forward_binding": inputs["binding"],
                "forward_vs_jacobian": inputs["raw_binding"],
                "jacobian_reuse": inputs["jacobian_reuse"],
                "operator": {
                    key: value
                    for key, value in inputs["operator_meta"].items()
                    if key != "component_coordinates_m"
                },
                "adjoint_source": {
                    "method": "official common-grid FieldRegion vector source with solver-handled component staggering",
                    "profile_scale": profile_scale,
                    "fieldregion_base_amplitude": base_amplitude,
                    "source_profile_roundtrip_max_abs_error": roundtrip,
                    "forward_Gaussian_original_amplitude": original_amplitude,
                    "forward_Gaussian_adjoint_amplitude": 0.0,
                    "forward_Gaussian_enabled_as_mesh_anchor": True,
                    "empirical_normalization": False,
                    "gradient_rescaling": False,
                },
                "normalization": normalization,
                "optical_gradient": {
                    key: value
                    for key, value in optical.items()
                    if key not in {"indirect_gradient", "direct_loss_gradient"}
                },
                "gradient_norms_A": {
                    "optical": float(np.linalg.norm(gradient_optical)),
                    "direct_PDE": float(np.linalg.norm(gradient_direct)),
                    "total": float(np.linalg.norm(gradient_total)),
                },
                "solver": {
                    "resource": resource,
                    "solver_wall_time_s": solver_wall,
                    "log_audit": log,
                    "GPU_log_evidence": gpu_log,
                    "forward_adjoint_solver_mesh_max_abs_difference_m": mesh_difference,
                    "forward_adjoint_monitor_grid_max_abs_difference_m": forward_adjoint_grid_difference,
                    "cwnorm_grid_max_abs_difference_m": average_grid_difference,
                },
                "gates": gates,
                "artifacts": {
                    "forward_result": _artifact(inputs["forward_path"]),
                    "forward_FSP": _artifact(inputs["forward_fsp"]),
                    "forward_raw_NPZ": _artifact(inputs["forward_raw"]),
                    "density_file": _artifact(inputs["density_path"]),
                    "PDE_result": _artifact(inputs["pde_path"]),
                    "PDE_pullback_NPZ": _artifact(inputs["pde_raw"]),
                    "adjoint_template_FSP": _artifact(template),
                    "adjoint_FSP": _artifact(adjoint_fsp),
                    "gradient_NPZ": _artifact(raw_output),
                },
                "Maxwell_forward_solves_this_invocation": 0,
                "Maxwell_adjoint_solves_this_invocation": 1,
                "custom_CUDA_solves_this_invocation": 0,
                "Lumerical_HEAT_or_CHARGE_solves": 0,
                "optimizer_iterations": 0,
                "AD_FD_claimed": False,
                "wall_s": time.monotonic() - started,
            }
    except Exception as error:
        result.update(
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            wall_s=time.monotonic() - started,
        )
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
