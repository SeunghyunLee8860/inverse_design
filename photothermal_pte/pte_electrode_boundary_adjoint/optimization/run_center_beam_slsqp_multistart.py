from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tairte4_boundary_adjoint.baseline import (  # noqa: E402
    BASELINE_ROOT,
    ElectricalModel,
    load_config,
)
from tairte4_boundary_adjoint.optimization import (  # noqa: E402
    SignedSLSQPResult,
    run_signed_slsqp,
)
from tairte4_boundary_adjoint.perimeter import PerimeterParameters  # noqa: E402
from tairte4_boundary_adjoint.robin import DifferentiableContactModel  # noqa: E402
from tairte4_boundary_adjoint.scaled import SignedBranchObjective  # noqa: E402


CONTACT_CONDUCTANCE_S_M2 = 1.0e12
TRANSITION_WIDTH_M = 0.75e-6
BOUNDARY_QUADRATURE_ORDER = 5
CONTACT_DISCRETIZATION = "nodal_lumped"
MAX_ITERATIONS = 250
FUNCTION_TOLERANCE = 1.0e-11
FEASIBILITY_TOLERANCE = 1.0e-8


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def deterministic_starts(perimeter_m: float) -> list[np.ndarray]:
    """Feasible, asymmetric perimeter starts; centers remain lifted."""

    um_over_p = 1e-6 / perimeter_m
    starts_physical_um = [
        # Phase-3 validation point and converted legacy-DE center-beam winner.
        (6.1, 7.3, 58.4, 10.7),
        (35.26207216724098, 20.66452892501721, 13.589892863387605, 8.540358266092685),
        # Opposite contacts at four rotations, with unequal lengths.
        (0.0, 6.0, 48.0, 14.0),
        (12.0, 14.0, 60.0, 6.0),
        (24.0, 9.0, 72.0, 18.0),
        (36.0, 18.0, 84.0, 9.0),
        # Adjacent-side-scale separations and their rotations.
        (8.0, 5.0, 31.0, 12.0),
        (32.0, 12.0, 55.0, 5.0),
        (56.0, 7.0, 79.0, 17.0),
        (80.0, 17.0, 7.0, 7.0),
        # Deliberately short/long asymmetric contacts.
        (4.0, 2.0, 44.0, 20.0),
        (52.0, 20.0, 92.0, 2.0),
    ]
    return [np.asarray(row, dtype=float) * um_over_p for row in starts_physical_um]


def contact_record(model: DifferentiableContactModel, center_m: float, length_m: float) -> dict:
    center_wrapped = model.perimeter.wrap_center(center_m)
    side, tangent_m = model.perimeter.s_to_side_coordinate(center_wrapped)
    width = model.perimeter.width_m
    height = model.perimeter.height_m
    corners = np.asarray([0.0, width, width + height, 2.0 * width + height])
    corner_distance = np.min(
        model.perimeter.periodic_distance(corners, center_wrapped)
    )
    return {
        "center_perimeter_um": center_wrapped * 1e6,
        "length_um": length_m * 1e6,
        "center_side": side,
        "center_side_tangent_um": tangent_m * 1e6,
        "crosses_corner": bool(corner_distance < 0.5 * length_m),
    }


def serialize_result(
    run: SignedSLSQPResult,
    model: DifferentiableContactModel,
) -> dict:
    p = run.smooth.canonical_design.to_physical(model.perimeter.perimeter_m)
    scipy_result = run.scipy_result
    return {
        "branch_sign": run.branch_sign,
        "start_scaled": run.start_scaled.tolist(),
        "start_physical_um": (run.start_scaled * model.perimeter.perimeter_m * 1e6).tolist(),
        "success": bool(scipy_result.success),
        "status_code": int(scipy_result.status),
        "message": str(scipy_result.message),
        "iterations": int(scipy_result.nit),
        "function_evaluations_reported": int(scipy_result.nfev),
        "gradient_evaluations_reported": int(scipy_result.njev),
        "unique_forward_adjoint_evaluations": run.unique_forward_adjoint_evaluations,
        "x_lifted_scaled": np.asarray(scipy_result.x, dtype=float).tolist(),
        "x_canonical_scaled": run.smooth.canonical_design.as_array().tolist(),
        "x_canonical_physical_um": (p.as_array() * 1e6).tolist(),
        "contact_0": contact_record(model, p.center_0_m, p.length_0_m),
        "contact_1": contact_record(model, p.center_1_m, p.length_1_m),
        "minimum_constraint": float(np.min(run.constraints)),
        "constraints": run.constraints.tolist(),
        "smooth_current_A": run.smooth.current_A,
        "smooth_abs_current_A": abs(run.smooth.current_A),
        "smooth_signed_response": run.smooth.signed_response,
        "dimensionless_minimization_objective": run.smooth.minimization_objective,
        "smooth_gradient_dimensionless": run.smooth.minimization_gradient_scaled.tolist(),
        "smooth_gradient_linf": float(np.max(np.abs(run.smooth.minimization_gradient_scaled))),
        "smooth_state_residual_relative": run.smooth.forward.state_residual_relative,
        "smooth_adjoint_residual_relative": run.smooth.forward.adjoint_residual_relative,
        "hard_current_A": run.hard.current_A,
        "hard_abs_current_A": abs(run.hard.current_A),
        "hard_terminal_conductance_S": run.hard.terminal_conductance_S,
        "hard_contact_0_node_count": int(run.hard.contact_0_nodes.size),
        "hard_contact_1_node_count": int(run.hard.contact_1_nodes.size),
        "hard_residual_relative": run.hard.residual_relative,
        "history": [
            {
                "iteration": item.iteration,
                "x_scaled": item.x_scaled.tolist(),
                "smooth_current_A": item.smooth_current_A,
                "dimensionless_minimization_objective": item.minimization_objective,
                "minimum_constraint": item.minimum_constraint,
            }
            for item in run.history
        ],
    }


def main() -> int:
    config_path = BASELINE_ROOT / "configs" / "per_beam_500nm.json"
    fields_path = BASELINE_ROOT / "results" / "per_beam_500nm_final" / "per_beam_fields.npz"
    legacy_results_path = (
        BASELINE_ROOT / "results" / "per_beam_500nm_final" / "per_beam_results.json"
    )
    config = load_config(config_path)
    with np.load(fields_path) as fields:
        centers_m = np.asarray(fields["beam_centers_m"], dtype=float)
        beam_index = int(np.argmin(np.linalg.norm(centers_m, axis=1)))
        temperature_nodes_K = np.asarray(fields["temperature_nodes_K"][beam_index], dtype=float)

    electrical = ElectricalModel(config)
    model = DifferentiableContactModel(
        electrical,
        temperature_nodes_K,
        contact_conductance_S_m2=CONTACT_CONDUCTANCE_S_M2,
        transition_m=TRANSITION_WIDTH_M,
        quadrature_order=BOUNDARY_QUADRATURE_ORDER,
        contact_discretization=CONTACT_DISCRETIZATION,
    )
    objective = SignedBranchObjective(model)
    electrode_config = config["electrodes"]
    minimum_length_m = electrode_config["min_contact_length_um"] * 1e-6
    usable_side_m = (
        config["geometry"]["flake_width_um"]
        - 2.0 * electrode_config["edge_clearance_um"]
    ) * 1e-6
    maximum_length_m = electrode_config["max_contact_fraction"] * usable_side_m
    minimum_gap_m = electrode_config["same_side_min_gap_um"] * 1e-6
    starts = deterministic_starts(model.perimeter.perimeter_m)

    all_runs: list[SignedSLSQPResult] = []
    for branch_sign in (+1, -1):
        for index, start in enumerate(starts):
            print(
                f"branch={branch_sign:+d} start={index + 1:02d}/{len(starts)}",
                flush=True,
            )
            run = run_signed_slsqp(
                objective,
                start,
                branch_sign=branch_sign,
                minimum_length_m=minimum_length_m,
                maximum_length_m=maximum_length_m,
                minimum_gap_m=minimum_gap_m,
                max_iterations=MAX_ITERATIONS,
                function_tolerance=FUNCTION_TOLERANCE,
            )
            all_runs.append(run)
            print(
                f"  success={run.scipy_result.success} nit={run.scipy_result.nit} "
                f"I_smooth={run.smooth.current_A:+.6e} "
                f"I_hard={run.hard.current_A:+.6e} "
                f"min_constraint={np.min(run.constraints):+.3e}",
                flush=True,
            )

    records = [serialize_result(run, model) for run in all_runs]
    acceptable = [
        (run, record)
        for run, record in zip(all_runs, records)
        if np.min(run.constraints) >= -FEASIBILITY_TOLERANCE
        and np.isfinite(run.hard.current_A)
    ]
    if not acceptable:
        raise RuntimeError("no feasible finite hard-re-evaluated candidate")
    best_run, best_record = max(acceptable, key=lambda pair: abs(pair[0].hard.current_A))

    legacy = json.loads(legacy_results_path.read_text(encoding="utf-8"))
    legacy_center = next(
        item for item in legacy["per_beam_results"] if item["beam_index"] == beam_index
    )
    legacy_e0 = legacy_center["best_electrode_0"]
    legacy_e1 = legacy_center["best_electrode_1"]
    legacy_parameters = PerimeterParameters(
        model.perimeter.side_coordinate_to_s(legacy_e0["side"], legacy_e0["center_m"]),
        legacy_e0["length_m"],
        model.perimeter.side_coordinate_to_s(legacy_e1["side"], legacy_e1["center_m"]),
        legacy_e1["length_m"],
    )
    legacy_hard = model.hard_evaluate(legacy_parameters)
    legacy_reported_current = legacy_center["signed_short_circuit_current_A"]
    legacy_reproduction_relative_error = abs(
        legacy_hard.current_A - legacy_reported_current
    ) / max(abs(legacy_reported_current), np.finfo(float).tiny)

    summary = {
        "status": "COMPLETED" if all(r.scipy_result.success for r in all_runs) else "COMPLETED_WITH_SLSQP_WARNINGS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one actual 0.5 um center-beam two-signed-branch SLSQP multi-start run",
        "baseline_root": str(BASELINE_ROOT),
        "config_sha256": file_sha256(config_path),
        "temperature_fields_sha256": file_sha256(fields_path),
        "legacy_results_sha256": file_sha256(legacy_results_path),
        "beam_index": beam_index,
        "beam_center_um": (centers_m[beam_index] * 1e6).tolist(),
        "mesh_step_um": electrical.step_m * 1e6,
        "perimeter_um": model.perimeter.perimeter_m * 1e6,
        "contact_discretization": CONTACT_DISCRETIZATION,
        "contact_conductance_S_m2": CONTACT_CONDUCTANCE_S_M2,
        "transition_width_um": TRANSITION_WIDTH_M * 1e6,
        "boundary_quadrature_order": BOUNDARY_QUADRATURE_ORDER,
        "current_scale_A": objective.current_scale_A,
        "minimum_contact_length_um": minimum_length_m * 1e6,
        "maximum_contact_length_um": maximum_length_m * 1e6,
        "minimum_gap_um": minimum_gap_m * 1e6,
        "signed_branches": [+1, -1],
        "starts_per_branch": len(starts),
        "slsqp_max_iterations": MAX_ITERATIONS,
        "slsqp_function_tolerance": FUNCTION_TOLERANCE,
        "hard_ranking_rule": "maximum abs(hard_current_A) over feasible finite candidates",
        "best_candidate": best_record,
        "legacy_de_comparison": {
            "reported_signed_current_A": legacy_reported_current,
            "reported_abs_current_A": legacy_center["best_abs_short_circuit_current_A"],
            "recomputed_full_perimeter_hard_current_A": legacy_hard.current_A,
            "reproduction_relative_error": legacy_reproduction_relative_error,
            "new_to_legacy_hard_abs_ratio": abs(best_run.hard.current_A) / abs(legacy_hard.current_A),
            "legacy_electrode_0": legacy_e0,
            "legacy_electrode_1": legacy_e1,
        },
        "overall_hard_winner": {
            "source": (
                "slsqp_smooth_then_hard"
                if abs(best_run.hard.current_A) > abs(legacy_hard.current_A)
                else "legacy_de_hard"
            ),
            "abs_current_A": max(
                abs(best_run.hard.current_A), abs(legacy_hard.current_A)
            ),
        },
        "runs": records,
    }
    output_json = HERE / "center_beam_slsqp_multistart.json"
    output_plot = HERE / "center_beam_slsqp_multistart.png"
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for branch_sign, marker, color in ((+1, "o", "tab:blue"), (-1, "s", "tab:orange")):
        branch_records = [r for r in records if r["branch_sign"] == branch_sign]
        indices = np.arange(1, len(branch_records) + 1)
        axes[0].plot(
            indices,
            np.asarray([r["hard_abs_current_A"] for r in branch_records]) * 1e9,
            marker=marker,
            color=color,
            linestyle="none",
            label=f"branch {branch_sign:+d}",
        )
    axes[0].axhline(abs(legacy_hard.current_A) * 1e9, color="black", linestyle="--", label="legacy DE")
    axes[0].set_xlabel("multi-start index")
    axes[0].set_ylabel("hard-contact |I| (nA)")
    axes[0].set_title("Hard re-evaluation of every smooth optimum")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for run in all_runs:
        if run.history:
            axes[1].plot(
                [item.iteration for item in run.history],
                [item.minimization_objective for item in run.history],
                alpha=0.45,
                color="tab:blue" if run.branch_sign == +1 else "tab:orange",
            )
    axes[1].set_xlabel("SLSQP iteration")
    axes[1].set_ylabel("dimensionless minimization objective")
    axes[1].set_title("All signed-branch optimization traces")
    axes[1].grid(True, alpha=0.3)
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)

    print(json.dumps({
        "status": summary["status"],
        "best_hard_current_A": best_run.hard.current_A,
        "best_physical_um": best_record["x_canonical_physical_um"],
        "legacy_hard_current_A": legacy_hard.current_A,
        "new_to_legacy_hard_abs_ratio": summary["legacy_de_comparison"]["new_to_legacy_hard_abs_ratio"],
        "output": str(output_json),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
