#!/usr/bin/env python3
"""Pull Run-002 thermal/PTE sensitivity back to native component Yee Q."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.finite_q_mapping import (  # noqa: E402
    apply_material_intersection_density_separable,
    nodal_control_volume_edges,
    transpose_material_intersection_density_separable,
)
from photothermal_pte.optimization_runs.cuda_thermal_adjoint import (  # noqa: E402
    solve_forward_adjoint_cuda,
)
from validate_production_thermal_material_adfd import (  # noqa: E402
    boundary_energy,
    build_state,
    nodal_to_cell,
)


MATERIALS = ("Si", "bottom_SiO2", "physical_TaIrTe4", "design_effective_SiO2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapped-q", type=Path, required=True)
    parser.add_argument("--mapped-q-sha256", required=True)
    parser.add_argument("--native-q", type=Path, required=True)
    parser.add_argument("--native-q-sha256", required=True)
    parser.add_argument("--scenario", default="grown_grown")
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing non-empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    mapped_path = args.mapped_q.expanduser().resolve()
    native_path = args.native_q.expanduser().resolve()
    if sha256(mapped_path) != args.mapped_q_sha256:
        raise RuntimeError("mapped-Q SHA mismatch")
    if sha256(native_path) != args.native_q_sha256:
        raise RuntimeError("native-Q SHA mismatch")
    mapped = np.load(mapped_path)
    native = np.load(native_path)
    target_edges = tuple(np.asarray(mapped[f"{axis}_edges_m"], float) for axis in "xyz")
    masks = {name: np.asarray(mapped[f"mask_{name}"], bool) for name in MATERIALS}

    design_mask = masks["design_effective_SiO2"]
    nx = int(np.count_nonzero(np.any(design_mask, axis=(1, 2))))
    ny = int(np.count_nonzero(np.any(design_mask, axis=(0, 2))))
    uniform_nodal = np.full((nx + 1, ny + 1), 0.5)
    state = build_state(mapped, args.scenario, nodal_to_cell(uniform_nodal))
    pair = solve_forward_adjoint_cuda(
        state.system.matrix_W_K,
        state.source_power_W,
        state.c_A_K,
        cuda_device=args.cuda_device,
        relative_tolerance=1e-10,
        max_iterations=30000,
    )
    objective = float(np.dot(state.c_A_K, pair.forward.solution))
    reciprocal = float(np.dot(pair.adjoint.solution, state.source_power_W))
    target_active_sensitivity = np.asarray(
        state.system.source_volume_operator_m3.T @ pair.adjoint.solution
    ).reshape(-1)
    target_sensitivity = np.zeros(state.active.shape, float)
    target_sensitivity[state.active] = target_active_sensitivity

    component_sensitivity: dict[str, np.ndarray] = {}
    component_weight: dict[str, np.ndarray] = {}
    component_records: dict[str, object] = {}
    actual_pullback = 0.0
    cauchy_terms = []
    rng = np.random.default_rng(2026080605)
    for component in "xyz":
        source_density = np.asarray(native[f"Q{component}_W_m3"], float)
        source_edges = tuple(
            nodal_control_volume_edges(
                np.asarray(native[f"Q{component}_{axis}_m"], float)
            )
            for axis in "xyz"
        )
        pulled_total = np.zeros(source_density.shape, float)
        perturbation = rng.normal(size=source_density.shape)
        left = 0.0
        material_records = {}
        for material in MATERIALS:
            pulled = transpose_material_intersection_density_separable(
                target_density_sensitivity=target_sensitivity,
                source_edges_m=source_edges,
                target_edges_m=target_edges,
                target_material_support_mask=masks[material],
            )
            pulled_total += pulled
            mapped_perturbation, _, metrics = apply_material_intersection_density_separable(
                source_density=perturbation,
                source_edges_m=source_edges,
                target_edges_m=target_edges,
                target_material_support_mask=masks[material],
            )
            left += float(np.sum(target_sensitivity * mapped_perturbation))
            material_records[material] = metrics
        right = float(np.sum(pulled_total * perturbation))
        dot_error = relative(left, right)
        component_sensitivity[component] = pulled_total
        source_volume = (
            np.diff(source_edges[0])[:, None, None]
            * np.diff(source_edges[1])[None, :, None]
            * np.diff(source_edges[2])[None, None, :]
        )
        if np.any(source_volume <= 0.0):
            raise RuntimeError("native component volume is nonpositive")
        component_weight[component] = pulled_total / source_volume
        contribution = float(np.sum(pulled_total * source_density))
        actual_pullback += contribution
        cauchy_terms.append(float(np.linalg.norm(pulled_total) * np.linalg.norm(source_density)))
        component_records[component] = {
            "shape": list(source_density.shape),
            "transpose_dot_error": dot_error,
            "actual_Q_objective_contribution_A": contribution,
            "sensitivity_all_finite": bool(np.all(np.isfinite(pulled_total))),
            "weight_all_finite": bool(np.all(np.isfinite(component_weight[component]))),
            "material_forward_mapping_audit": material_records,
        }

    raw_identity = relative(actual_pullback, objective)
    cauchy_scale = max(sum(cauchy_terms), float(np.linalg.norm(pair.adjoint.solution) * np.linalg.norm(state.source_power_W)), np.finfo(float).tiny)
    cauchy_identity = abs(actual_pullback - objective) / cauchy_scale
    raw_reciprocity = relative(objective, reciprocal)
    cauchy_reciprocity = abs(objective - reciprocal) / max(
        float(np.linalg.norm(state.c_A_K) * np.linalg.norm(pair.forward.solution)),
        float(np.linalg.norm(pair.adjoint.solution) * np.linalg.norm(state.source_power_W)),
        np.finfo(float).tiny,
    )
    worst_dot = max(record["transpose_dot_error"] for record in component_records.values())
    energy = boundary_energy(state, pair.forward.solution)
    passed = bool(
        worst_dot < 1e-12
        and cauchy_identity < 1e-8
        and cauchy_reciprocity < 1e-8
        and max(pair.forward.explicit_relative_residual, pair.adjoint.explicit_relative_residual) < 1e-8
        and energy < 0.01
        and all(record["sensitivity_all_finite"] and record["weight_all_finite"] for record in component_records.values())
    )
    raw = output / f"thermal_to_native_yee_pullback_{args.scenario}.npz"
    np.savez_compressed(
        raw,
        thermal_Q_density_sensitivity_A_m3_W=target_sensitivity,
        **{f"native_Q{component}_density_sensitivity_A_m3_W": value for component, value in component_sensitivity.items()},
        **{f"native_Q{component}_absorption_weight_A_W": value for component, value in component_weight.items()},
        **{f"Q{component}_{axis}_m": np.asarray(native[f"Q{component}_{axis}_m"], float) for component in "xyz" for axis in "xyz"},
    )
    result = {
        "status": "VALIDATED_PRODUCTION_THERMAL_PTE_TO_NATIVE_YEE_PULLBACK" if passed else "FAILED_PRODUCTION_THERMAL_PTE_TO_NATIVE_YEE_PULLBACK",
        "passed": passed,
        "scenario": args.scenario,
        "scope": "uniform rho=0.5 fixed production thermal/PTE adjoint through exact material-intersection Q deposition transpose",
        "objective_A": objective,
        "reciprocal_A": reciprocal,
        "native_pullback_actual_Q_objective_A": actual_pullback,
        "raw_near_null_objective_identity_relative_error": raw_identity,
        "Cauchy_normalized_objective_identity_error": cauchy_identity,
        "raw_near_null_reciprocity_relative_error": raw_reciprocity,
        "Cauchy_normalized_reciprocity_error": cauchy_reciprocity,
        "component_records": component_records,
        "worst_transpose_dot_error": worst_dot,
        "forward": {"iterations": pair.forward.iterations, "residual": pair.forward.explicit_relative_residual, "seconds": pair.forward.solve_seconds},
        "adjoint": {"iterations": pair.adjoint.iterations, "residual": pair.adjoint.explicit_relative_residual, "seconds": pair.adjoint.solve_seconds},
        "energy_balance_error": energy,
        "raw_artifact": {"path": str(raw), "size_bytes": raw.stat().st_size, "sha256": sha256(raw)},
        "inputs": {
            "mapped_Q": {"path": str(mapped_path), "sha256": args.mapped_q_sha256},
            "native_Q": {"path": str(native_path), "sha256": args.native_q_sha256},
        },
        "full_3D_Kronecker_operator_materialized": False,
        "empirical_normalization_or_gradient_rescaling": False,
        "CPU_linear_solve_fallback": False,
        "Maxwell_solves": 0,
        "optimization_iterations": 0,
    }
    (output / "production_thermal_to_native_yee_pullback_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
