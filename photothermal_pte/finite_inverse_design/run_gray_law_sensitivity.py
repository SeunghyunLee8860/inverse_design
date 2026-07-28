#!/usr/bin/env python3
"""Coupled optical/thermal/PTE sensitivity to endpoint-preserving gray laws."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .probe_v261_cpu_tfsf_device import PABS_FIELD
from .probe_v261_gpu_plane_wave_roi import load_lumapi
from .run_combined_physical_rho_pte_adfd import (
    array_sha256,
    compact_forward,
    native_weight_and_source,
    physical_state,
    run_forward_density,
)
from .run_corrected_combined_physical_rho_pte_adfd import (
    FLUX_SIGNS,
    completed_forward,
    component_gradient,
    load_operator,
    prepare_corrected_source,
    thermal_state,
)
from .run_v261_large_background_mixed_optical_adfd import (
    monitor_electric,
    run_adjoint,
)
from .run_v261_large_background_tfsf_forward import sha256


STATUS = "COMPLETED_COUPLED_GRAY_LAW_SENSITIVITY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-forward", required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--jacobian-dir", required=True)
    parser.add_argument("--linear-adjoint-4um", required=True)
    parser.add_argument("--linear-adjoint-4um-sha256", required=True)
    parser.add_argument("--linear-adjoint-6um", required=True)
    parser.add_argument("--linear-adjoint-6um-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gpu-device", default="GPU 1")
    return parser.parse_args()


def checked(path_text: str, expected: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"missing or SHA-mismatched artifact: {path}")
    return path


def relative(value: float, reference: float) -> float:
    return abs(value - reference) / max(
        abs(value), abs(reference), np.finfo(float).tiny
    )


def nrmse(value: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.sqrt(np.mean(np.asarray(reference, float) ** 2)))
    return float(
        np.sqrt(
            np.mean(
                (np.asarray(value, float) - np.asarray(reference, float)) ** 2
            )
        )
        / max(scale, np.finfo(float).tiny)
    )


def gradient_angle(value: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(value) * np.linalg.norm(reference)),
        np.finfo(float).tiny,
    )
    cosine = float(np.vdot(value, reference).real) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "gray_law_sensitivity.json"
    npz_path = output / "gray_law_sensitivity_arrays.npz"
    result: dict[str, object] = {
        "status": "BLOCKED_GRAY_LAW_SENSITIVITY_NOT_RUN",
        "passed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "optimization_run": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "gray_laws": {
            "definition": (
                "phi_p(rho)=rho**p is applied consistently to optical "
                "epsilon, bulk kappa, and interface G through the existing "
                "linear effective-material operator"
            ),
            "exponents": [1.0, 2.0, 3.0],
            "endpoints_preserved": True,
            "interpretation": (
                "named numerical relaxations, not material measurements or "
                "a confidence interval"
            ),
        },
    }
    fdtd = None
    started = time.monotonic()
    arrays: dict[str, np.ndarray] = {}
    try:
        base_project = checked(args.base_forward, args.base_sha256)
        linear_adjoints = {
            4.0: checked(
                args.linear_adjoint_4um,
                args.linear_adjoint_4um_sha256,
            ),
            6.0: checked(
                args.linear_adjoint_6um,
                args.linear_adjoint_6um_sha256,
            ),
        }
        nominal_rho, _ = physical_state()
        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        linear_forward = run_forward_density(
            fdtd,
            rho=nominal_rho,
            project=base_project,
            threads=args.threads,
            flux_signs=FLUX_SIGNS,
            reuse_completed=True,
        )
        operator, operator_meta = load_operator(
            Path(args.jacobian_dir).expanduser().resolve(),
            linear_forward["electric"].shape[:3],
        )
        cases: dict[str, object] = {}
        runtime: dict[float, dict[str, object]] = {}
        for exponent in (1.0, 2.0, 3.0):
            effective_rho = nominal_rho**exponent
            label = f"p{int(exponent)}"
            if exponent == 1.0:
                forward = linear_forward
                forward_meta = {
                    **compact_forward(forward),
                    "reused_completed": True,
                }
            else:
                forward, forward_meta = completed_forward(
                    fdtd,
                    rho=effective_rho,
                    base_project=base_project,
                    output=output,
                    role=f"gray_{label}_forward_cpu_tfsf",
                    threads=args.threads,
                )
            scenario_data: dict[str, object] = {}
            runtime[exponent] = {}
            for flake_um in (4.0, 6.0):
                kwargs, coupling, geometry, mapping, evaluation = thermal_state(
                    effective_rho, flake_um, forward["native"]
                )
                coefficient, weighted_source, pullback = (
                    native_weight_and_source(
                        evaluation=evaluation,
                        native=forward["native"],
                        mapping=mapping,
                        electric=forward["electric"],
                        epsilon=forward["epsilon"],
                        frequency_Hz=float(forward["grid"]["f"][0]),
                    )
                )
                template, profile_scale, collocation, source_meta = (
                    prepare_corrected_source(
                        fdtd,
                        base_project=forward["project"]["path"],
                        grid=forward["grid"],
                        weighted_source=weighted_source,
                        output=output,
                        label=f"gray_{label}_{flake_um:g}um",
                    )
                )
                if exponent == 1.0:
                    adjoint_project = linear_adjoints[flake_um]
                    adjoint_meta = {
                        "engine": "GPU",
                        "reused_completed": True,
                        "project": {
                            "path": str(adjoint_project),
                            "byte_size": adjoint_project.stat().st_size,
                            "sha256": sha256(adjoint_project),
                        },
                    }
                else:
                    adjoint_project = (
                        output / f"gray_{label}_{flake_um:g}um_adjoint_gpu.fsp"
                    )
                    adjoint_meta = run_adjoint(
                        fdtd,
                        template=template,
                        project=adjoint_project,
                        engine="GPU",
                        threads=args.threads,
                        gpu_device=args.gpu_device,
                    )
                fdtd.load(str(adjoint_project))
                adjoint_electric, adjoint_grid = monitor_electric(
                    fdtd, PABS_FIELD
                )
                coordinate_mismatch = max(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(forward["grid"][key])
                                - np.asarray(adjoint_grid[key])
                            )
                        )
                    )
                    for key in (
                        "x",
                        "y",
                        "z",
                        "delta_x",
                        "delta_y",
                        "delta_z",
                    )
                )
                optical_gradient, optical_parts = component_gradient(
                    operator=operator,
                    base=forward,
                    adjoint_electric=adjoint_electric,
                    coefficient=coefficient,
                    profile_scale=profile_scale,
                    base_amplitude=source_meta[
                        "fieldregion_base_amplitude"
                    ],
                )
                thermal_gradient = coupling.thermal_vjp(
                    evaluation.gradient_rho_A
                )
                effective_gradient = optical_gradient + thermal_gradient
                nominal_gradient = (
                    effective_gradient
                    * exponent
                    * nominal_rho ** (exponent - 1.0)
                )
                scenario_key = f"{flake_um:g}um"
                arrays[f"{label}_{scenario_key}_temperature_K"] = (
                    evaluation.solved.temperature_K
                )
                arrays[f"{label}_{scenario_key}_source_W_m3"] = (
                    mapping["source_W_m3"]
                )
                arrays[f"{label}_{scenario_key}_effective_gradient_A"] = (
                    effective_gradient
                )
                arrays[f"{label}_{scenario_key}_nominal_gradient_A"] = (
                    nominal_gradient
                )
                runtime[exponent][scenario_key] = {
                    "temperature": evaluation.solved.temperature_K,
                    "source": mapping["source_W_m3"],
                    "nominal_gradient": nominal_gradient,
                }
                scenario_data[scenario_key] = {
                    "objective_A": float(evaluation.objective_A),
                    "effective_gradient_L2_A": float(
                        np.linalg.norm(effective_gradient)
                    ),
                    "nominal_gradient_L2_A": float(
                        np.linalg.norm(nominal_gradient)
                    ),
                    "optical_gradient_L2_A": float(
                        np.linalg.norm(optical_gradient)
                    ),
                    "thermal_gradient_L2_A": float(
                        np.linalg.norm(thermal_gradient)
                    ),
                    "energy_balance_relative_error": float(
                        evaluation.solved.energy_balance_relative_error
                    ),
                    "forward_linear_residual_relative": float(
                        evaluation.solved.linear_residual_relative
                    ),
                    "adjoint_linear_residual_relative": float(
                        evaluation.adjoint_linear_residual_relative
                    ),
                    "Q_mapping_relative_error": float(
                        mapping["relative_power_error"]
                    ),
                    "pullback": pullback,
                    "collocation": collocation,
                    "coordinate_mismatch_m": coordinate_mismatch,
                    "optical_gradient_parts": optical_parts,
                    "adjoint": {
                        key: value
                        for key, value in adjoint_meta.items()
                        if key not in {"electric", "grid"}
                    },
                }
            cases[label] = {
                "exponent": exponent,
                "nominal_rho_sha256": array_sha256(nominal_rho),
                "effective_rho_sha256": array_sha256(effective_rho),
                "effective_rho_bounds": [
                    float(np.min(effective_rho)),
                    float(np.max(effective_rho)),
                ],
                "P_Q_W": float(forward["P_Q_W"]),
                "P_six_W": float(forward["P_six_W"]),
                "six_face_closure_relative_error": float(
                    forward["six_face_closure_relative_error"]
                ),
                "forward": forward_meta,
                "scenarios": scenario_data,
            }
            print(
                f"GRAY_LAW exponent={exponent:g} "
                f"P_Q={forward['P_Q_W']:.9e}",
                flush=True,
            )
        comparisons = {}
        linear = cases["p1"]
        for label in ("p2", "p3"):
            case = cases[label]
            per_scenario = {}
            exponent = float(case["exponent"])
            for flake_um in (4.0, 6.0):
                scenario_key = f"{flake_um:g}um"
                current = runtime[exponent][scenario_key]
                reference = runtime[1.0][scenario_key]
                per_scenario[scenario_key] = {
                    "objective_relative_change": relative(
                        case["scenarios"][scenario_key]["objective_A"],
                        linear["scenarios"][scenario_key]["objective_A"],
                    ),
                    "temperature_field_NRMSE": nrmse(
                        current["temperature"], reference["temperature"]
                    ),
                    "mapped_Q_field_NRMSE": nrmse(
                        current["source"], reference["source"]
                    ),
                    "nominal_gradient_L2_relative_change": relative(
                        float(np.linalg.norm(current["nominal_gradient"])),
                        float(np.linalg.norm(reference["nominal_gradient"])),
                    ),
                    "nominal_gradient_angle_deg": gradient_angle(
                        current["nominal_gradient"],
                        reference["nominal_gradient"],
                    ),
                }
            comparisons[f"{label}_vs_p1"] = {
                "P_Q_relative_change": relative(
                    case["P_Q_W"], linear["P_Q_W"]
                ),
                "P_six_relative_change": relative(
                    case["P_six_W"], linear["P_six_W"]
                ),
                "scenarios": per_scenario,
            }
        np.savez_compressed(npz_path, **arrays)
        all_cases = list(cases.values())
        result.update(
            {
                "status": STATUS,
                "passed": True,
                "operator": operator_meta,
                "cases": cases,
                "comparisons": comparisons,
                "gates": {
                    "worst_optical_closure_relative_error": max(
                        case["six_face_closure_relative_error"]
                        for case in all_cases
                    ),
                    "optical_closure_limit": 5.0e-3,
                    "worst_Q_mapping_relative_error": max(
                        scenario["Q_mapping_relative_error"]
                        for case in all_cases
                        for scenario in case["scenarios"].values()
                    ),
                    "Q_mapping_limit": 5.0e-3,
                    "worst_thermal_energy_balance_relative_error": max(
                        scenario["energy_balance_relative_error"]
                        for case in all_cases
                        for scenario in case["scenarios"].values()
                    ),
                    "thermal_energy_balance_limit": 1.0e-2,
                    "worst_linear_residual_relative": max(
                        max(
                            scenario["forward_linear_residual_relative"],
                            scenario["adjoint_linear_residual_relative"],
                        )
                        for case in all_cases
                        for scenario in case["scenarios"].values()
                    ),
                    "linear_residual_limit": 1.0e-8,
                },
                "arrays": {
                    "path": str(npz_path),
                    "byte_size": npz_path.stat().st_size,
                    "sha256": sha256(npz_path),
                },
                "next_gate": "FULL_LATENT_COMBINED_PTE_ADFD",
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "FAILED_GRAY_LAW_SENSITIVITY",
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result["wall_s"] = time.monotonic() - started
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "path": str(result_path)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
