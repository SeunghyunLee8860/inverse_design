#!/usr/bin/env python3
"""Compare Run009 thermal lateral domains without rerunning Maxwell.

The physical 32 um TaIrTe4 flake, 18.6 um design, z grid, Q artifact,
weighting field, material properties, and boundary types are held fixed.
Only the outer x/y thermal-domain location is changed.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

RUN002 = REPOSITORY / "photothermal_pte" / "optimization_runs" / "run_002_gaussian10_w8p5_current_max"
if str(RUN002) not in sys.path:
    sys.path.insert(0, str(RUN002))

import run_production_combined_adfd_smoke as combined  # noqa: E402
from map_production_q_to_thermal_grid import growing_positions  # noqa: E402
from photothermal_pte.optimization_runs.axis_contract import X_B_Y_A  # noqa: E402
from selected_thermal_density_mapping import (  # noqa: E402
    selected_nodal_to_thermal_cell,
    selected_nodal_to_thermal_cell_transpose,
)


DEFAULT_EVALUATION = Path(
    "/data/seunghyun/tairte4/raw_artifacts/"
    "run009_uniform_Ea_optimization_20260809/"
    "b0002_s028_g028_retry1_evaluation"
)


def thermal_edges(span_um: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half_span = 0.5 * float(span_um) * 1e-6
    flake_half_span = 16.0e-6
    if half_span < flake_half_span:
        raise ValueError("thermal domain cannot be smaller than the physical flake")
    core = np.linspace(-flake_half_span, flake_half_span, 321)
    outer_length = half_span - flake_half_span
    if outer_length > 1e-18:
        outer = growing_positions(outer_length, 100e-9, 1e-6, 1.45)[1:]
        lateral = np.concatenate(
            (-flake_half_span - outer[::-1], core, flake_half_span + outer)
        )
    else:
        lateral = core
    si_top = -0.385e-6
    si = si_top - growing_positions(20e-6, 25e-9, 0.5e-6, 1.35)[::-1]
    oxide = np.linspace(-0.385e-6, -0.100e-6, 20)
    flake = np.linspace(-0.100e-6, 0.0, 5)
    design = np.linspace(0.0, 1.0e-6, 21)
    z = np.concatenate((si, oxide[1:], flake[1:], design[1:]))
    return lateral, lateral.copy(), z


def load_q(native_path: Path, result_path: Path) -> dict:
    native = np.load(native_path, mmap_mode="r")
    result = json.loads(result_path.read_text())
    return {
        "Q_components": {
            component: np.asarray(native[f"Q{component}_W_m3"])
            for component in "xyz"
        },
        "native_coordinates": {
            component: {
                axis: np.asarray(native[f"Q{component}_{axis}_m"])
                for axis in "xyz"
            }
            for component in "xyz"
        },
        "P_Q_W": float(result["base_forward"]["P_Q_W"]),
    }


def relative(value: float, reference: float) -> float:
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1e-300)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--spans-um", type=float, nargs="+", default=(64.0, 48.0, 35.0))
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation = args.evaluation.resolve()
    native_path = evaluation / "full_latent_base_native_q.npz"
    result_path = evaluation / "selected_full_latent_adjoint_preparation_result.json"
    state_path = evaluation / "selected_full_latent_adjoint_preparation.npz"
    q = load_q(native_path, result_path)
    rho = np.asarray(np.load(state_path)["rho"], float)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    comparison_arrays: dict[float, dict[str, np.ndarray]] = {}

    for span_um in args.spans_um:
        edges = thermal_edges(span_um)
        combined.thermal_edges = lambda edges=edges: edges
        mapping_start = perf_counter()
        data, mapping = combined.map_q(q, design_half_span_m=9.3e-6)
        mapping_seconds = perf_counter() - mapping_start

        solve_start = perf_counter()
        state, pair, objective, nodal_parts, target_sensitivity = combined.solve_base_thermal(
            data,
            rho,
            args.cuda_device,
            selected_nodal_to_thermal_cell,
            selected_nodal_to_thermal_cell_transpose,
            axis_contract=X_B_Y_A,
        )
        thermal_total_seconds = perf_counter() - solve_start
        theta_full = state.system.full_field(pair.forward.solution)
        flake_mask = np.asarray(data["mask_physical_TaIrTe4"], bool)
        flake_temperature = np.asarray(theta_full[flake_mask], float)
        thermal_gradient = np.asarray(nodal_parts["total"], float)
        boundary_power = {
            name: float(np.sum(g * pair.forward.solution[cell_ids]))
            for name, (cell_ids, g, _) in state.system.boundary_terms.items()
        }
        source_power = float(np.sum(state.source_power_W))
        lateral_power = sum(
            value
            for name, value in boundary_power.items()
            if name in {"x_min", "x_max", "y_min", "y_max"}
        )
        total_boundary = float(sum(boundary_power.values()))
        case = {
            "span_um": float(span_um),
            "half_span_um": 0.5 * float(span_um),
            "padding_beyond_flake_per_side_um": 0.5 * float(span_um) - 16.0,
            "shape_xyz": [int(v) for v in state.active.shape],
            "active_unknowns": int(state.system.matrix_W_K.shape[0]),
            "matrix_nnz": int(state.system.matrix_W_K.nnz),
            "mapped_source_power_W": float(mapping["mapped_power_W"]),
            "physical_fraction_of_native_P_Q": float(mapping["physical_fraction_of_native_P_Q"]),
            "objective_A": float(objective),
            "Tmax_rise_K": float(np.nanmax(theta_full)),
            "flake_average_rise_K": float(np.mean(flake_temperature)),
            "forward_iterations": int(pair.forward.iterations),
            "adjoint_iterations": int(pair.adjoint.iterations),
            "forward_residual": float(pair.forward.explicit_relative_residual),
            "adjoint_residual": float(pair.adjoint.explicit_relative_residual),
            "energy_balance_error": abs(total_boundary - source_power) / max(abs(source_power), 1e-300),
            "lateral_numerical_boundary_power_fraction": float(lateral_power / max(abs(source_power), 1e-300)),
            "mapping_seconds": float(mapping_seconds),
            "thermal_total_seconds": float(thermal_total_seconds),
            "forward_solve_seconds": float(pair.forward.solve_seconds),
            "adjoint_solve_seconds": float(pair.adjoint.solve_seconds),
        }
        cases.append(case)
        comparison_arrays[float(span_um)] = {
            "flake_temperature": flake_temperature,
            "thermal_gradient": thermal_gradient,
        }
        print(json.dumps(case, indent=2), flush=True)
        del data, state, pair, theta_full, flake_mask, nodal_parts, target_sensitivity
        gc.collect()

    reference = cases[0]
    reference_arrays = comparison_arrays[float(reference["span_um"])]
    for case in cases:
        arrays = comparison_arrays[float(case["span_um"])]
        case["relative_to_reference"] = {
            "mapped_source_power": relative(case["mapped_source_power_W"], reference["mapped_source_power_W"]),
            "objective": relative(case["objective_A"], reference["objective_A"]),
            "Tmax": relative(case["Tmax_rise_K"], reference["Tmax_rise_K"]),
            "flake_average_temperature": relative(case["flake_average_rise_K"], reference["flake_average_rise_K"]),
            "flake_temperature_NRMSE": float(
                np.linalg.norm(arrays["flake_temperature"] - reference_arrays["flake_temperature"])
                / max(np.linalg.norm(reference_arrays["flake_temperature"]), 1e-300)
            ),
            "thermal_design_gradient_NRMSE": float(
                np.linalg.norm(arrays["thermal_gradient"] - reference_arrays["thermal_gradient"])
                / max(np.linalg.norm(reference_arrays["thermal_gradient"]), 1e-300)
            ),
            "thermal_design_gradient_angle_deg": float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.vdot(arrays["thermal_gradient"], reference_arrays["thermal_gradient"]).real
                            / max(
                                np.linalg.norm(arrays["thermal_gradient"])
                                * np.linalg.norm(reference_arrays["thermal_gradient"]),
                                1e-300,
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                )
            ),
        }

    result = {
        "status": "THERMAL_LATERAL_DOMAIN_DIAGNOSTIC",
        "reference_span_um": float(reference["span_um"]),
        "fixed_contract": {
            "physical_flake_span_um": 32.0,
            "design_span_um": 18.6,
            "si_depth_um": 20.0,
            "axis_contract": X_B_Y_A.name,
            "weighting_field_held_fixed_m_inv": [1.0 / 64e-6, 1.0 / 64e-6],
            "Maxwell_rerun": False,
        },
        "cases": cases,
    }
    result_path = output / "run009_thermal_lateral_domain_comparison.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
