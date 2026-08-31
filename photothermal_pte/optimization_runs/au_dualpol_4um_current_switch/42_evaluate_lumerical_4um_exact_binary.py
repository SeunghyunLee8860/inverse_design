#!/usr/bin/env python3
"""Fresh dual-polarization evaluation of one exact dispersive-Au cell mask."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.dfm import (
    exact_500nm_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_gray_q_coupling import (
    GrayYeeQCoupling,
    component_coordinates_from_raw,
    component_q_from_raw,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_mesh_contract import (
    LumericalMeshSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_maxwell_contract import (
    binary_mask_sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    refine_exact_binary_density,
    thermal_edges,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.volumetric_electrical_4um import (
    evaluate_fixed_source_volumetric,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.objective import (
    exact_binary_promotion_passed,
    opposite_current_switching_achieved,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.pde_mesh_convergence_4um import (
    PDE_STEPS_M,
    pde_mesh_convergence_audit,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
MESH_LABEL = "fine_z2p5_bulk50_xy100_cv0_pml8_span20_z6_t1ps"
SOURCE_OBJECT_W0_UM = 3.9561433030461415


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary-mask-npz", required=True, type=Path)
    parser.add_argument("--binary-mask-key", default="binary_mask")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument(
        "--accelerator-policy", choices=("development", "b200"), required=True
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ea-source-calibration", type=Path)
    parser.add_argument("--eb-source-calibration", type=Path)
    parser.add_argument("--ea-forward-result", type=Path)
    parser.add_argument("--ea-raw-npz", type=Path)
    parser.add_argument("--eb-forward-result", type=Path)
    parser.add_argument("--eb-raw-npz", type=Path)
    parser.add_argument("--mesh-label", default=MESH_LABEL)
    parser.add_argument("--flake-dxy-nm", type=float, default=100.0)
    parser.add_argument("--stack-dz-nm", type=float, default=2.5)
    parser.add_argument("--bulk-dz-nm", type=float, default=50.0)
    parser.add_argument("--outer-dxy-nm", type=float, default=200.0)
    parser.add_argument(
        "--require-pde-through-nm",
        type=float,
        choices=tuple(step * 1.0e9 for step in PDE_STEPS_M),
        help=(
            "Do not stop adaptive PDE refinement before this core step. "
            "Used by the terminal 100/50-nm optical-downstream comparison."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    value = path.resolve()
    return {
        "path": str(value),
        "size_bytes": value.stat().st_size,
        "sha256": _sha256(value),
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _matching_artifact(
    result: dict[str, Any], suffix: str, *, override_path: Path | None = None
) -> Path:
    matches = [
        row
        for row in result.get("raw_artifacts", [])
        if str(row.get("path", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact-forward {suffix} artifact")
    record = matches[0]
    path = Path(override_path or record["path"]).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"missing exact-forward {suffix} artifact")
    if int(record.get("size_bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"exact-forward {suffix} size changed")
    if str(record.get("sha256", "")) != _sha256(path):
        raise RuntimeError(f"exact-forward {suffix} SHA256 changed")
    return path


def _requested_mesh_spec(args: argparse.Namespace) -> dict[str, Any]:
    return LumericalMeshSpec(
        label=args.mesh_label,
        flake_dxy_m=args.flake_dxy_nm * 1.0e-9,
        stack_dz_m=args.stack_dz_nm * 1.0e-9,
        bulk_dz_m=args.bulk_dz_nm * 1.0e-9,
        outer_dxy_m=args.outer_dxy_nm * 1.0e-9,
        mesh_accuracy=3,
        pml_layers=8,
        lateral_span_m=20.0e-6,
        z_min_m=-3.0e-6,
        z_max_m=3.0e-6,
        simulation_time_s=1.0e-12,
        auto_shutoff_min=1.0e-7,
        conformal_mesh="conformal variant 0",
    ).audit()


def _validate_forward_record(
    *,
    forward: dict[str, Any],
    polarization: str,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, bool]:
    geometry = (
        forward.get("layout", {}).get("geometry", {}).get("exact_au_geometry", {})
    )
    processing = forward.get("Q_processing", {})
    gates = {
        "all_original_forward_gates_passed": forward.get("all_gates_passed") is True,
        "exact_binary_case_matches": forward.get("case") == "exact_binary",
        "polarization_matches": forward.get("polarization") == polarization,
        "mesh_spec_matches": forward.get("mesh_spec") == _requested_mesh_spec(args),
        "accelerator_policy_matches": forward.get("accelerator_policy")
        == args.accelerator_policy,
        "binary_mask_payload_sha256_matches": geometry.get("mask_payload_sha256")
        == binary_mask_sha256(mask),
        "raw_Q_processing_is_unmodified": all(
            processing.get(name) is False
            for name in (
                "clipping",
                "smoothing",
                "gain",
                "field_or_Q_rescaling",
            )
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(
            f"{polarization} exact-forward provenance gates failed: {gates}"
        )
    return gates


def _reuse_forward_requested(args: argparse.Namespace) -> bool:
    values = (
        args.ea_forward_result,
        args.ea_raw_npz,
        args.eb_forward_result,
        args.eb_raw_npz,
    )
    if any(value is not None for value in values) and not all(
        value is not None for value in values
    ):
        raise ValueError(
            "reused mode requires Ea/Eb forward-result and raw-NPZ arguments"
        )
    return all(value is not None for value in values)


def _visible_cuda_device(
    args: argparse.Namespace, *, environ: dict[str, str] | None = None
) -> str:
    environment = os.environ if environ is None else environ
    devices = [
        item.strip()
        for item in environment.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    if len(devices) != 1 or devices[0] == "-1":
        raise RuntimeError(
            "set CUDA_VISIBLE_DEVICES to exactly one physical GPU for custom PDE solves"
        )
    if devices[0] != str(args.gpu_index):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must equal the requested physical --gpu-index"
        )
    return devices[0]


def _cuda_device_audit(args: argparse.Namespace) -> dict[str, Any]:
    physical = _visible_cuda_device(args)
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical}",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or len(rows) != 1:
        raise RuntimeError("could not prove the single visible custom CUDA GPU")
    fields = [item.strip() for item in rows[0].split(",", maxsplit=2)]
    if len(fields) != 3 or fields[0] != physical:
        raise RuntimeError("nvidia-smi custom CUDA GPU record is inconsistent")
    return {
        "CUDA_VISIBLE_DEVICES": physical,
        "physical_index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "local_cuda_device_used": 0,
    }


def _pde_step_label(step_m: float) -> str:
    return f"{float(step_m) * 1.0e9:g}".replace(".", "p") + "nm"


def _pde_resolution(
    *,
    mask: np.ndarray,
    component_coordinates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    q: dict[str, np.ndarray],
    reporting_scale: float,
    core_step_m: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    density = refine_exact_binary_density(mask, target_step_m=core_step_m)
    coupling = GrayYeeQCoupling.from_component_coordinates(
        component_coordinates,
        thermal_edges(core_step_m=core_step_m),
    )
    source_power_raw, mapping = coupling.map_power(q)
    source_power = source_power_raw * reporting_scale
    evaluated = evaluate_fixed_source_volumetric(
        np.asarray(density, dtype=np.float64),
        source_power,
        0,
        need_gradient=False,
        exact_binary_geometry=True,
    )
    thermal = evaluated["thermal_audit"]
    electrical = evaluated["electrical_audit"]
    gates = {
        "Q_mapping_conservation_lt_1e_12": float(mapping["relative_power_error"])
        < 1.0e-12,
        "thermal_residual_lt_1e_8": float(thermal["relative_residual"]) < 1.0e-8,
        "thermal_energy_balance_lt_1pct": float(thermal["energy_balance_relative"])
        < 1.0e-2,
        "electrical_residual_lt_1e_8": float(electrical["relative_residual"]) < 1.0e-8,
        "electrical_terminal_balance_lt_1pct": float(
            electrical["terminal_balance_relative"]
        )
        < 1.0e-2,
        "exact_binary_void_Au_nodes_removed": bool(
            electrical["exact_binary_geometry"]
            and electrical["electrical_void_Au_nodes_removed"]
            and int(electrical["inactive_void_Au_cell_count"])
            == int(np.count_nonzero(density == 0))
        ),
        "finite_nonzero_current": bool(
            np.isfinite(evaluated["objective_A"])
            and float(evaluated["objective_A"]) != 0.0
        ),
    }
    public = {
        "passed": all(gates.values()),
        "core_step_m": core_step_m,
        "design_shape": list(density.shape),
        "ta_temperature_shape": list(np.asarray(evaluated["ta_temperature"]).shape),
        "current_A": float(evaluated["objective_A"]),
        "current_nA": 1.0e9 * float(evaluated["objective_A"]),
        "mapped_source_power_W_reporting": float(np.sum(source_power)),
        "peak_temperature_K": float(np.max(evaluated["temperature"])),
        "ta_mean_temperature_K": float(np.mean(evaluated["ta_temperature"])),
        "mapping": mapping,
        "thermal": thermal,
        "electrical": electrical,
        "gates": gates,
    }
    arrays = {
        "density": np.asarray(density, dtype=np.uint8),
        "ta_temperature_K": np.asarray(evaluated["ta_temperature"], dtype=np.float64),
    }
    return public, arrays


def _forward_command(
    *,
    args: argparse.Namespace,
    polarization: str,
    source_calibration: Path,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(HERE / "25_run_lumerical_4um_exact_au_control.py"),
        "--case",
        "exact_binary",
        "--binary-mask-file",
        str(args.binary_mask_npz.resolve()),
        "--binary-mask-key",
        args.binary_mask_key,
        "--polarization",
        polarization,
        "--gpu-index",
        str(args.gpu_index),
        "--accelerator-policy",
        args.accelerator_policy,
        "--output-dir",
        str(output),
        "--source-calibration-json",
        str(source_calibration.resolve()),
        "--source-object-w0-um",
        str(SOURCE_OBJECT_W0_UM),
        "--mesh-label",
        args.mesh_label,
        "--flake-dxy-nm",
        str(args.flake_dxy_nm),
        "--stack-dz-nm",
        str(args.stack_dz_nm),
        "--bulk-dz-nm",
        str(args.bulk_dz_nm),
        "--outer-dxy-nm",
        str(args.outer_dxy_nm),
        "--mesh-accuracy",
        "3",
        "--au-max-coefficients",
        "6",
        "--au-fit-tolerance",
        "0",
        "--mesh-refinement",
        "conformal variant 0",
        "--pml-layers",
        "8",
        "--lateral-span-um",
        "20",
        "--z-min-um",
        "-3",
        "--z-max-um",
        "3",
        "--simulation-time-ps",
        "1",
        "--auto-shutoff-min",
        "1e-7",
        "--threads",
        str(args.threads),
    ]


def _evaluate_polarization(
    *,
    args: argparse.Namespace,
    polarization: str,
    source_calibration: Path | None,
    mask: np.ndarray,
    output: Path,
    forward_result: Path | None = None,
    raw_override: Path | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True)
    reuse_forward = forward_result is not None or raw_override is not None
    if reuse_forward and (forward_result is None or raw_override is None):
        raise ValueError(
            f"{polarization} reuse requires both forward result and raw NPZ"
        )
    log_path: Path | None = None
    if reuse_forward:
        forward_path = Path(forward_result).expanduser().resolve()
        if not forward_path.is_file():
            raise RuntimeError(f"missing {polarization} reused forward JSON")
        forward = json.loads(forward_path.read_text(encoding="utf-8"))
    else:
        if source_calibration is None:
            raise ValueError(
                f"{polarization} source calibration is required for a fresh forward"
            )
        command = _forward_command(
            args=args,
            polarization=polarization,
            source_calibration=source_calibration,
            output=output,
        )
        log_path = output.parent / f"exact_forward_{polarization}.log"
        with log_path.open("w", encoding="utf-8", errors="replace") as stream:
            completed = subprocess.run(
                command,
                cwd=REPOSITORY,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
            raise RuntimeError(
                f"{polarization} exact forward failed with exit "
                f"{completed.returncode}\n{tail}"
            )
        json_paths = sorted(output.glob("*.json"))
        if len(json_paths) != 1:
            raise RuntimeError(f"expected one {polarization} exact-forward JSON")
        forward_path = json_paths[0].resolve()
        forward = json.loads(forward_path.read_text(encoding="utf-8"))
    forward_validation = _validate_forward_record(
        forward=forward,
        polarization=polarization,
        mask=mask,
        args=args,
    )
    raw_path = _matching_artifact(
        forward,
        "_raw.npz",
        override_path=raw_override,
    )
    reporting_scale = float(
        forward["reporting_normalization"]["scalar_reporting_factor"]
    )
    if not np.isfinite(reporting_scale) or reporting_scale <= 0.0:
        raise RuntimeError("invalid exact-forward reporting scale")
    resolutions: dict[str, dict[str, Any]] = {}
    arrays_by_label: dict[str, dict[str, np.ndarray]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    previous_label: str | None = None
    required_step_m = (
        None
        if getattr(args, "require_pde_through_nm", None) is None
        else float(args.require_pde_through_nm) * 1.0e-9
    )
    with np.load(raw_path, allow_pickle=False) as raw:
        component_coordinates = component_coordinates_from_raw(raw)
        q = component_q_from_raw(raw)
        for step_m in PDE_STEPS_M:
            label = _pde_step_label(step_m)
            resolution, arrays = _pde_resolution(
                mask=mask,
                component_coordinates=component_coordinates,
                q=q,
                reporting_scale=reporting_scale,
                core_step_m=step_m,
            )
            resolutions[label] = resolution
            arrays_by_label[label] = arrays
            if previous_label is not None:
                coarse = resolutions[previous_label]
                comparison = pde_mesh_convergence_audit(
                    coarse_current_A=coarse["current_A"],
                    fine_current_A=resolution["current_A"],
                    coarse_ta_temperature_K=arrays_by_label[previous_label][
                        "ta_temperature_K"
                    ],
                    fine_ta_temperature_K=arrays["ta_temperature_K"],
                    coarse_peak_temperature_K=coarse["peak_temperature_K"],
                    fine_peak_temperature_K=resolution["peak_temperature_K"],
                    coarse_step_m=coarse["core_step_m"],
                    fine_step_m=resolution["core_step_m"],
                )
                comparison_label = f"{previous_label}_to_{label}"
                comparisons[comparison_label] = comparison
                required_reached = bool(
                    required_step_m is None
                    or step_m <= required_step_m * (1.0 + 1.0e-12)
                )
                if comparison["passed"] and required_reached:
                    break
            previous_label = label
    if not comparisons:
        raise RuntimeError("adaptive PDE sequence did not produce one comparison")
    selected_label = next(reversed(resolutions))
    selected = resolutions[selected_label]
    final_comparison_label = next(reversed(comparisons))
    convergence = comparisons[final_comparison_label]
    evidence_path = output.parent / f"pde_mesh_convergence_{polarization}.npz"
    evidence_arrays: dict[str, np.ndarray] = {}
    for label, arrays in arrays_by_label.items():
        evidence_arrays[f"density_{label}"] = arrays["density"]
        evidence_arrays[f"ta_temperature_K_{label}"] = arrays["ta_temperature_K"]
        evidence_arrays[f"current_A_{label}"] = np.asarray(
            resolutions[label]["current_A"]
        )
    np.savez_compressed(evidence_path, **evidence_arrays)
    first_label = next(iter(resolutions))
    native_q_power_raw = float(resolutions[first_label]["mapping"]["input_power_W"])
    expected_native_q_power_raw = float(forward["P_Q_native_W_raw"])
    native_q_json_error = abs(native_q_power_raw - expected_native_q_power_raw) / max(
        abs(expected_native_q_power_raw), np.finfo(float).tiny
    )
    gates = {
        "ordinary_dispersive_exact_Au_forward": True,
        "native_Q_json_power_match_lt_1e_12": native_q_json_error < 1.0e-12,
        "all_executed_PDE_resolution_gates_passed": all(
            row["passed"] for row in resolutions.values()
        ),
        "adaptive_adjacent_PDE_pair_converged_below_0p5pct": bool(
            convergence["passed"]
        ),
    }
    forward_evidence = {
        "forward_result": _artifact(forward_path),
        "forward_raw": _artifact(raw_path),
        "PDE_mesh_convergence_evidence": _artifact(evidence_path),
    }
    if log_path is not None:
        forward_evidence["forward_log"] = _artifact(log_path)
    return {
        "passed": all(gates.values()),
        "polarization": polarization,
        "current_A": selected["current_A"],
        "current_nA": selected["current_nA"],
        "reference_PDE_core_step_m": selected["core_step_m"],
        "selected_PDE_resolution": selected_label,
        "required_PDE_refinement_through_m": required_step_m,
        "mapped_source_power_W_reporting": selected["mapped_source_power_W_reporting"],
        "mapping": selected["mapping"],
        "thermal": selected["thermal"],
        "electrical": selected["electrical"],
        "PDE_resolutions": resolutions,
        "PDE_mesh_comparisons": comparisons,
        "PDE_mesh_convergence": convergence,
        "final_PDE_comparison": final_comparison_label,
        "native_Q_json_power_relative_error": native_q_json_error,
        "gates": gates,
        "forward_mode": (
            "reused_hash_bound_artifacts_without_Maxwell"
            if reuse_forward
            else "fresh_Lumerical_Maxwell"
        ),
        "forward_validation": forward_validation,
        **forward_evidence,
    }


def main() -> int:
    args = _parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty exact-binary output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "exact_binary_dualpol_result.json"
    result: dict[str, Any] = {
        "status": (
            "FAILED_LUMERICAL_4UM_EXACT_BINARY_SINGLE_MAXWELL_MESH_"
            "ADAPTIVE_PDE_GATE"
        ),
        "passed": False,
        "final_lateral_certificate_claimed": False,
        "requires_script_43_100_to_50nm_optical_comparison": True,
        "Lumerical_HEAT_or_CHARGE_solves": 0,
        "FDTDX_Maxwell_solves": 0,
    }
    started = time.monotonic()
    try:
        reuse_forward = _reuse_forward_requested(args)
        if not reuse_forward and (
            args.ea_source_calibration is None or args.eb_source_calibration is None
        ):
            raise ValueError("fresh mode requires both Ea and Eb source calibrations")
        result["Maxwell_forward_mode"] = (
            "reused_hash_bound_artifacts_without_Maxwell"
            if reuse_forward
            else "fresh_Lumerical_Maxwell"
        )
        result["Lumerical_Maxwell_solves"] = {
            "forward": 0 if reuse_forward else 2,
            "adjoint": 0,
        }
        with np.load(args.binary_mask_npz, allow_pickle=False) as data:
            mask = np.asarray(data[args.binary_mask_key])
        if mask.shape != CONTRACT.design_shape or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("binary candidate must be exact 80x80 zero/one")
        mask = np.asarray(mask, dtype=np.uint8)
        exact = exact_500nm_audit(
            mask,
            spacing_m=CONTRACT.design_pitch_m,
            minimum_feature_m=250.0e-9,
        )
        if not bool(exact["solid_pass"] and exact["void_pass"]):
            raise RuntimeError("exact mask failed 250 nm solid/void audit")
        custom_cuda_device = _cuda_device_audit(args)
        result["custom_CUDA_device"] = custom_cuda_device
        rows = {
            "Ea": _evaluate_polarization(
                args=args,
                polarization="Ea",
                source_calibration=args.ea_source_calibration,
                mask=mask,
                output=output / "forward_Ea",
                forward_result=args.ea_forward_result,
                raw_override=args.ea_raw_npz,
            ),
            "Eb": _evaluate_polarization(
                args=args,
                polarization="Eb",
                source_calibration=args.eb_source_calibration,
                mask=mask,
                output=output / "forward_Eb",
                forward_result=args.eb_forward_result,
                raw_override=args.eb_raw_npz,
            ),
        }
        pde_forward_solve_count = sum(
            len(row["PDE_resolutions"]) for row in rows.values()
        )
        currents = {key: row["current_A"] for key, row in rows.items()}
        switching = opposite_current_switching_achieved(currents["Ea"], currents["Eb"])
        numerical_pass = all(row["passed"] for row in rows.values())
        promotion_pass = exact_binary_promotion_passed(
            numerical_pass, currents["Ea"], currents["Eb"]
        )
        result = {
            "status": (
                "PASSED_LUMERICAL_4UM_EXACT_BINARY_SINGLE_MAXWELL_MESH_"
                "ADAPTIVE_PDE_GATE"
                if promotion_pass
                else (
                    "FAILED_LUMERICAL_4UM_EXACT_BINARY_SINGLE_MAXWELL_MESH_"
                    "ADAPTIVE_PDE_GATE"
                )
            ),
            "passed": promotion_pass,
            "final_lateral_certificate_claimed": False,
            "requires_script_43_100_to_50nm_optical_comparison": True,
            "numerical_gates_passed": numerical_pass,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "material_geometry": (
                "ordinary dispersive Au rectangles coalesced from the exact 80x80 mask"
            ),
            "binary_mask": _artifact(args.binary_mask_npz.resolve()),
            "exact_250nm_audit": {
                key: value
                for key, value in exact.items()
                if key not in {"binary", "bad_solid", "bad_void"}
            },
            "currents_A": currents,
            "currents_nA": {key: 1.0e9 * value for key, value in currents.items()},
            "balanced_utility_A": min(currents["Ea"], -currents["Eb"]),
            "opposite_current_switching_achieved": switching,
            "polarizations": rows,
            "Maxwell_solver": "Lumerical FDTD 2026 R1.2 build 4522",
            "Maxwell_forward_mode": result["Maxwell_forward_mode"],
            "Lumerical_Maxwell_solves": result["Lumerical_Maxwell_solves"],
            "custom_CUDA_device": custom_cuda_device,
            "custom_CUDA_thermal_solves": {
                "forward": pde_forward_solve_count,
                "adjoint": 0,
            },
            "custom_CUDA_electrical_solves": {
                "forward": pde_forward_solve_count,
                "adjoint": 0,
            },
            "Lumerical_HEAT_or_CHARGE_solves": 0,
            "FDTDX_Maxwell_solves": 0,
            "wall_s": time.monotonic() - started,
        }
    except Exception as error:
        result.update(
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            wall_s=time.monotonic() - started,
        )
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
