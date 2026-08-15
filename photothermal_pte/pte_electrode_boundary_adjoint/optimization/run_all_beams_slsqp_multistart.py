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
sys.path.insert(0, str(HERE))

from run_center_beam_slsqp_multistart import (  # noqa: E402
    BOUNDARY_QUADRATURE_ORDER,
    CONTACT_CONDUCTANCE_S_M2,
    CONTACT_DISCRETIZATION,
    FEASIBILITY_TOLERANCE,
    FUNCTION_TOLERANCE,
    MAX_ITERATIONS,
    TRANSITION_WIDTH_M,
    deterministic_starts,
    serialize_result,
)
from tairte4_boundary_adjoint.baseline import (  # noqa: E402
    BASELINE_ROOT,
    ElectricalModel,
    load_config,
)
from tairte4_boundary_adjoint.optimization import run_signed_slsqp  # noqa: E402
from tairte4_boundary_adjoint.perimeter import PerimeterParameters  # noqa: E402
from tairte4_boundary_adjoint.robin import DifferentiableContactModel  # noqa: E402
from tairte4_boundary_adjoint.scaled import SignedBranchObjective  # noqa: E402


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def legacy_parameters(model: DifferentiableContactModel, record: dict) -> PerimeterParameters:
    e0 = record["best_electrode_0"]
    e1 = record["best_electrode_1"]
    return PerimeterParameters(
        model.perimeter.side_coordinate_to_s(e0["side"], e0["center_m"]),
        e0["length_m"],
        model.perimeter.side_coordinate_to_s(e1["side"], e1["center_m"]),
        e1["length_m"],
    )


def main() -> int:
    # Never run production if either completed blocker gate has regressed.
    for gate in (
        PROJECT_ROOT / "validation" / "phase3_gradient_check.json",
        PROJECT_ROOT / "validation" / "phase3_robin_hard_convergence.json",
    ):
        data = json.loads(gate.read_text(encoding="utf-8"))
        if data.get("status") != "PASS":
            raise RuntimeError(f"production gate is not PASS: {gate}")

    config_path = BASELINE_ROOT / "configs" / "per_beam_500nm.json"
    fields_path = BASELINE_ROOT / "results" / "per_beam_500nm_final" / "per_beam_fields.npz"
    legacy_results_path = BASELINE_ROOT / "results" / "per_beam_500nm_final" / "per_beam_results.json"
    config = load_config(config_path)
    with np.load(fields_path) as fields:
        centers_m = np.asarray(fields["beam_centers_m"], dtype=float)
        temperatures_K = np.asarray(fields["temperature_nodes_K"], dtype=float)
    legacy_data = json.loads(legacy_results_path.read_text(encoding="utf-8"))
    legacy_by_index = {item["beam_index"]: item for item in legacy_data["per_beam_results"]}

    electrical = ElectricalModel(config)
    electrode_config = config["electrodes"]
    minimum_length_m = electrode_config["min_contact_length_um"] * 1e-6
    usable_side_m = (
        config["geometry"]["flake_width_um"]
        - 2.0 * electrode_config["edge_clearance_um"]
    ) * 1e-6
    maximum_length_m = electrode_config["max_contact_fraction"] * usable_side_m
    minimum_gap_m = electrode_config["same_side_min_gap_um"] * 1e-6

    beam_records: list[dict] = []
    all_slsqp_success = True
    for beam_index, (center_m, temperature_K) in enumerate(zip(centers_m, temperatures_K)):
        print(
            f"beam={beam_index + 1:02d}/{centers_m.shape[0]} "
            f"center_um={(center_m * 1e6).tolist()}",
            flush=True,
        )
        model = DifferentiableContactModel(
            electrical,
            temperature_K,
            contact_conductance_S_m2=CONTACT_CONDUCTANCE_S_M2,
            transition_m=TRANSITION_WIDTH_M,
            quadrature_order=BOUNDARY_QUADRATURE_ORDER,
            contact_discretization=CONTACT_DISCRETIZATION,
        )
        objective = SignedBranchObjective(model)
        legacy_record = legacy_by_index[beam_index]
        legacy_p = legacy_parameters(model, legacy_record)
        legacy_hard = model.hard_evaluate(legacy_p)
        legacy_reported = legacy_record["signed_short_circuit_current_A"]
        reproduction_error = abs(legacy_hard.current_A - legacy_reported) / max(
            abs(legacy_reported), np.finfo(float).tiny
        )

        starts = deterministic_starts(model.perimeter.perimeter_m)
        legacy_start = legacy_p.as_array() / model.perimeter.perimeter_m
        gap_fraction = minimum_gap_m / model.perimeter.perimeter_m
        legacy_constraints, _ = model.perimeter.separation_constraints_scaled(
            legacy_start, gap_fraction
        )
        if np.min(legacy_constraints) >= -1e-12:
            starts[1] = legacy_start

        runs = []
        for branch_sign in (+1, -1):
            for start_index, start in enumerate(starts):
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
                runs.append(run)
                all_slsqp_success = all_slsqp_success and bool(run.scipy_result.success)
                print(
                    f"  branch={branch_sign:+d} start={start_index + 1:02d} "
                    f"ok={run.scipy_result.success} nit={run.scipy_result.nit:3d} "
                    f"hard|I|={abs(run.hard.current_A):.6e}",
                    flush=True,
                )

        serialized = [serialize_result(run, model) for run in runs]
        feasible = [
            (run, record)
            for run, record in zip(runs, serialized)
            if np.min(run.constraints) >= -FEASIBILITY_TOLERANCE
            and np.isfinite(run.hard.current_A)
        ]
        if not feasible:
            raise RuntimeError(f"beam {beam_index}: no feasible hard candidate")
        best_run, best_serialized = max(
            feasible, key=lambda pair: abs(pair[0].hard.current_A)
        )
        slsqp_abs = abs(best_run.hard.current_A)
        legacy_abs = abs(legacy_hard.current_A)
        source = "slsqp_smooth_then_hard" if slsqp_abs > legacy_abs else "legacy_de_hard"
        beam_records.append(
            {
                "beam_index": beam_index,
                "beam_center_um": (center_m * 1e6).tolist(),
                "current_scale_A": objective.current_scale_A,
                "all_slsqp_runs_successful": all(r.scipy_result.success for r in runs),
                "best_slsqp_hard_candidate": best_serialized,
                "legacy_de_hard": {
                    "reported_current_A": legacy_reported,
                    "recomputed_current_A": legacy_hard.current_A,
                    "reproduction_relative_error": reproduction_error,
                    "electrode_0": legacy_record["best_electrode_0"],
                    "electrode_1": legacy_record["best_electrode_1"],
                },
                "slsqp_to_legacy_abs_ratio": slsqp_abs / legacy_abs,
                "overall_hard_winner": {
                    "source": source,
                    "abs_current_A": max(slsqp_abs, legacy_abs),
                },
                "runs": serialized,
            }
        )
        print(
            f"  winner={source} SLSQP/DE={slsqp_abs / legacy_abs:.6f}",
            flush=True,
        )

    summary = {
        "status": "COMPLETED" if all_slsqp_success else "COMPLETED_WITH_SLSQP_WARNINGS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "independent per-beam 0.5 um signed-branch SLSQP multi-start with hard re-evaluation",
        "gradient_gate": "PASS",
        "robin_hard_gate": "PASS",
        "baseline_root": str(BASELINE_ROOT),
        "config_sha256": file_sha256(config_path),
        "temperature_fields_sha256": file_sha256(fields_path),
        "legacy_results_sha256": file_sha256(legacy_results_path),
        "mesh_step_um": electrical.step_m * 1e6,
        "contact_discretization": CONTACT_DISCRETIZATION,
        "contact_conductance_S_m2": CONTACT_CONDUCTANCE_S_M2,
        "transition_width_um": TRANSITION_WIDTH_M * 1e6,
        "boundary_quadrature_order": BOUNDARY_QUADRATURE_ORDER,
        "minimum_contact_length_um": minimum_length_m * 1e6,
        "maximum_contact_length_um": maximum_length_m * 1e6,
        "minimum_gap_um": minimum_gap_m * 1e6,
        "starts_per_branch": len(deterministic_starts(2.0 * (24e-6 + 24e-6))),
        "signed_branches": [+1, -1],
        "ranking_rule": "for each beam compare all feasible SLSQP hard re-evaluations and legacy DE hard current by abs(I)",
        "beams": beam_records,
    }
    output_json = HERE / "all_beams_slsqp_multistart.json"
    output_plot = HERE / "all_beams_slsqp_multistart.png"
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    labels = [f"({x:.0f},{y:.0f})" for x, y in (centers_m * 1e6)]
    slsqp_nA = np.asarray(
        [b["best_slsqp_hard_candidate"]["hard_abs_current_A"] for b in beam_records]
    ) * 1e9
    legacy_nA = np.asarray(
        [abs(b["legacy_de_hard"]["recomputed_current_A"]) for b in beam_records]
    ) * 1e9
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    axes[0].plot(x, legacy_nA, "o--", label="legacy DE hard")
    axes[0].plot(x, slsqp_nA, "s-", label="SLSQP -> hard")
    axes[0].set_xticks(x, labels, rotation=45)
    axes[0].set_xlabel("beam center (um)")
    axes[0].set_ylabel("hard-contact |I| (nA)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    ratios = slsqp_nA / legacy_nA
    axes[1].bar(x, ratios)
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set_xticks(x, labels, rotation=45)
    axes[1].set_xlabel("beam center (um)")
    axes[1].set_ylabel("SLSQP hard / legacy-DE hard")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "beam_count": len(beam_records),
                "slsqp_wins": sum(
                    b["overall_hard_winner"]["source"] == "slsqp_smooth_then_hard"
                    for b in beam_records
                ),
                "output": str(output_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
