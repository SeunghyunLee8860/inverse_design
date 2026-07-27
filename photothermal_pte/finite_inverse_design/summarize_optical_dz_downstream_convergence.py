#!/usr/bin/env python3
"""Summarize optical-dz downstream PTE and component-J gradient convergence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .native_yee_q import EPS0
from .run_v261_large_background_mixed_optical_adfd import component_volumes
from .run_v261_large_background_tfsf_forward import sha256
from .yee_material_jacobian import SparseYeeMaterialJacobian


STATUS_PASS = "VALIDATED_OPTICAL_DZ_DOWNSTREAM_PTE_GRADIENT_CONVERGENCE"
STATUS_FAIL = "FAILED_OPTICAL_DZ_DOWNSTREAM_PTE_GRADIENT_CONVERGENCE"
CONVERGENCE_LIMIT = 5.0e-3
DOT_LIMIT = 1.0e-12


def parse_keyed(values: list[str], label: str) -> dict[float, Path]:
    result = {}
    for value in values:
        try:
            key_text, path_text = value.split("=", 1)
            key = float(key_text)
        except ValueError as exc:
            raise ValueError(f"invalid {label} {value!r}") from exc
        if key in result:
            raise ValueError(f"duplicate {label} dz={key:g}")
        result[key] = Path(path_text).expanduser().resolve()
    expected = {2.5, 1.25, 0.625}
    if set(result) != expected:
        raise ValueError(f"{label} dz set {set(result)} != {expected}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--jacobian", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def relative(value: float, reference: float) -> float:
    return abs(value - reference) / max(
        abs(value), abs(reference), np.finfo(float).tiny
    )


def nrmse(value: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(value - reference)
        / max(
            np.linalg.norm(value),
            np.linalg.norm(reference),
            np.finfo(float).tiny,
        )
    )


def load_operator(directory: Path, shape: tuple[int, ...]):
    matrices = {
        component: sparse.load_npz(directory / f"J_{component}.npz")
        for component in "xyz"
    }
    return SparseYeeMaterialJacobian(
        density_shape=(81, 81),
        component_shapes={component: shape for component in "xyz"},
        matrices=matrices,
    )


def component_grid(raw: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    return {
        "x": raw["pabs_x_m"],
        "y": raw["pabs_y_m"],
        "z": raw["pabs_z_m"],
        "delta_x": raw["pabs_delta_x_m"],
        "delta_y": raw["pabs_delta_y_m"],
        "delta_z": raw["pabs_delta_z_m"],
    }


def gradient_record(
    *,
    raw: np.lib.npyio.NpzFile,
    operator: SparseYeeMaterialJacobian,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    forward = np.asarray(raw["forward_electric"], np.complex128)
    adjoint = np.asarray(raw["adjoint_electric"], np.complex128)
    if forward.shape != adjoint.shape:
        raise ValueError("forward and adjoint PABS field shapes differ")
    shape = forward.shape[:3]
    if any(operator.component_shapes[c] != shape for c in "xyz"):
        raise ValueError("component-J and PABS field shapes differ")
    volumes = component_volumes(component_grid(raw))
    profile_scale = float(raw["profile_scale"][0])
    base_amplitude = float(raw["fieldregion_base_amplitude"][0])
    cotangent = {}
    component_inner = {}
    for index, component in enumerate("xyz"):
        cotangent[component] = (
            (2.0 * EPS0 / base_amplitude)
            * volumes[index]
            * forward[..., 0, index]
            * (adjoint[..., 0, index] * profile_scale)
        )
        component_inner[component] = float(
            np.real(
                np.sum(
                    cotangent[component]
                    * operator.jvp(raw["direction"])[component]
                )
            )
        )
    optical = operator.vjp(cotangent)
    thermal = np.asarray(raw["thermal_gradient_nodal_A"], float)
    combined = optical + thermal
    direction = np.asarray(raw["direction"], float)
    optical_directional = float(np.sum(optical * direction))
    component_sum = float(sum(component_inner.values()))
    component_sum_error = relative(component_sum, optical_directional)
    rng = np.random.default_rng(2026072721)
    dot_direction = rng.normal(size=(81, 81))
    tangent = operator.jvp(dot_direction)
    left = float(
        np.real(
            sum(
                np.sum(cotangent[c] * tangent[c]) for c in "xyz"
            )
        )
    )
    right = float(np.sum(operator.vjp(cotangent) * dot_direction))
    dot_error = relative(left, right)
    return (
        {
            "optical_directional_gradient_A": optical_directional,
            "thermal_directional_gradient_A": float(
                np.sum(thermal * direction)
            ),
            "combined_directional_gradient_A": float(
                np.sum(combined * direction)
            ),
            "optical_gradient_L2_A": float(np.linalg.norm(optical)),
            "thermal_gradient_L2_A": float(np.linalg.norm(thermal)),
            "combined_gradient_L2_A": float(np.linalg.norm(combined)),
            "optical_component_directional_xyz_A": component_inner,
            "component_sum_relative_error": component_sum_error,
            "JVP_VJP_dot_relative_error": dot_error,
            "empirical_normalization": False,
            "gradient_rescaling": False,
        },
        {
            "optical_gradient_A": optical,
            "thermal_gradient_A": thermal,
            "combined_gradient_A": combined,
        },
    )


def main() -> int:
    args = parse_args()
    cases = parse_keyed(args.case, "case")
    jacobians = parse_keyed(args.jacobian, "jacobian")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    gradient_artifacts = []
    for dz_nm in sorted(cases, reverse=True):
        result_path = cases[dz_nm]
        case = json.loads(result_path.read_text())
        if not case.get("passed"):
            raise RuntimeError(f"downstream case dz={dz_nm:g} did not pass")
        j_dir = jacobians[dz_nm]
        j_result_path = j_dir / "component_yee_jacobian_result.json"
        j_result = json.loads(j_result_path.read_text())
        if not j_result.get("passed"):
            raise RuntimeError(f"component J dz={dz_nm:g} did not pass")
        dz_records = {
            "flake_dz_nm": dz_nm,
            "case_result": {
                "path": str(result_path),
                "byte_size": result_path.stat().st_size,
                "sha256": sha256(result_path),
            },
            "jacobian_result": {
                "path": str(j_result_path),
                "byte_size": j_result_path.stat().st_size,
                "sha256": sha256(j_result_path),
            },
            "P_Q_W": case["forward"]["P_Q_W"],
            "P_six_W": case["forward"]["P_six_W"],
            "six_face_closure_relative_error": case["forward"][
                "six_face_closure_relative_error"
            ],
            "scenarios": {},
        }
        for name, scenario in case["scenarios"].items():
            raw_path = Path(scenario["raw_artifact"]["path"])
            if sha256(raw_path) != scenario["raw_artifact"]["sha256"]:
                raise RuntimeError(f"raw artifact SHA mismatch: {raw_path}")
            with np.load(raw_path) as raw:
                shape = tuple(raw["forward_electric"].shape[:3])
                operator = load_operator(j_dir, shape)
                gradient, arrays = gradient_record(
                    raw=raw, operator=operator
                )
                gradient_path = (
                    output
                    / f"dz_{dz_nm:g}nm_{name}_nodal_gradients.npz"
                )
                np.savez_compressed(gradient_path, **arrays)
                gradient_artifacts.append(
                    {
                        "path": str(gradient_path),
                        "byte_size": gradient_path.stat().st_size,
                        "sha256": sha256(gradient_path),
                    }
                )
                dz_records["scenarios"][name] = {
                    "mapped_Q_power_W": scenario["mapped_Q_power_W"],
                    "Tmax_DeltaT_K": scenario["Tmax_DeltaT_K"],
                    "PTE_objective_A": scenario["PTE_objective_A"],
                    "Q_mapping_relative_power_error": scenario[
                        "Q_mapping_relative_power_error"
                    ],
                    "thermal_energy_balance_relative_error": scenario[
                        "thermal_energy_balance_relative_error"
                    ],
                    "thermal_forward_linear_residual_relative": scenario[
                        "thermal_forward_linear_residual_relative"
                    ],
                    "thermal_adjoint_linear_residual_relative": scenario[
                        "thermal_adjoint_linear_residual_relative"
                    ],
                    "raw_path": str(raw_path),
                    "gradient": gradient,
                    "_mapped_Q": np.asarray(raw["mapped_Q_W_m3"], float),
                    "_temperature": np.asarray(raw["temperature_K"], float),
                    "_flake_mask": np.asarray(raw["flake_mask"], bool),
                }
        records[dz_nm] = dz_records

    comparisons = []
    for coarse, fine in ((2.5, 1.25), (1.25, 0.625), (2.5, 0.625)):
        item = {
            "coarse_dz_nm": coarse,
            "fine_dz_nm": fine,
            "P_Q_relative_difference": relative(
                records[coarse]["P_Q_W"], records[fine]["P_Q_W"]
            ),
            "scenarios": {},
        }
        for name in ("4um", "6um"):
            a = records[coarse]["scenarios"][name]
            b = records[fine]["scenarios"][name]
            if not np.array_equal(a["_flake_mask"], b["_flake_mask"]):
                raise RuntimeError("thermal flake masks differ across dz")
            mask = a["_flake_mask"]
            comparison = {
                "remapped_Q_field_NRMSE": nrmse(
                    a["_mapped_Q"], b["_mapped_Q"]
                ),
                "Tmax_relative_difference": relative(
                    a["Tmax_DeltaT_K"], b["Tmax_DeltaT_K"]
                ),
                "TaIrTe4_temperature_field_NRMSE": nrmse(
                    a["_temperature"][mask], b["_temperature"][mask]
                ),
                "PTE_objective_relative_difference": relative(
                    a["PTE_objective_A"], b["PTE_objective_A"]
                ),
                "optical_directional_gradient_relative_difference": relative(
                    a["gradient"]["optical_directional_gradient_A"],
                    b["gradient"]["optical_directional_gradient_A"],
                ),
                "combined_directional_gradient_relative_difference": (
                    relative(
                        a["gradient"][
                            "combined_directional_gradient_A"
                        ],
                        b["gradient"][
                            "combined_directional_gradient_A"
                        ],
                    )
                ),
            }
            comparison["decisive_maximum"] = max(
                comparison["PTE_objective_relative_difference"],
                comparison[
                    "optical_directional_gradient_relative_difference"
                ],
                comparison[
                    "combined_directional_gradient_relative_difference"
                ],
            )
            comparison["passed_0p5pct"] = bool(
                comparison["decisive_maximum"] < CONVERGENCE_LIMIT
            )
            item["scenarios"][name] = comparison
        comparisons.append(item)

    production_dz_nm = None
    for candidate in (2.5, 1.25):
        reference = 0.625
        match = next(
            item
            for item in comparisons
            if item["coarse_dz_nm"] == candidate
            and item["fine_dz_nm"] == reference
        )
        if all(
            value["passed_0p5pct"]
            for value in match["scenarios"].values()
        ):
            production_dz_nm = candidate
            break
    worst_dot = max(
        scenario["gradient"]["JVP_VJP_dot_relative_error"]
        for value in records.values()
        for scenario in value["scenarios"].values()
    )
    passed = bool(production_dz_nm is not None and worst_dot < DOT_LIMIT)

    for value in records.values():
        for scenario in value["scenarios"].values():
            for private in ("_mapped_Q", "_temperature", "_flake_mask"):
                del scenario[private]
    result = {
        "status": STATUS_PASS if passed else STATUS_FAIL,
        "passed": passed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "nonuniform physical-rho optical dz downstream convergence "
            "through conservative Q remap, explicit thermal/PTE, "
            "component-wise Yee J_c, and combined directional gradient"
        ),
        "records": {f"{key:g}nm": value for key, value in records.items()},
        "comparisons": comparisons,
        "production_flake_dz_nm": production_dz_nm,
        "production_selection_rule": (
            "coarsest candidate whose 4um and 6um raw PTE, optical "
            "directional gradient, and combined directional gradient all "
            "differ from the 0.625 nm reference by <0.5%"
        ),
        "gates": {
            "convergence_limit": CONVERGENCE_LIMIT,
            "worst_JVP_VJP_dot_relative_error": worst_dot,
            "mapping_transpose_limit": DOT_LIMIT,
        },
        "gradient_artifacts": gradient_artifacts,
        "gray_law_sensitivity_run": False,
        "full_latent_adfd_run": False,
        "optimization_run": False,
    }
    result_path = output / "optical_dz_downstream_convergence_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "production_flake_dz_nm": production_dz_nm,
                "result_path": str(result_path),
            }
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
