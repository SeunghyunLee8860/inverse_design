from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(HERE))

from run_all_beams_slsqp_multistart import legacy_parameters  # noqa: E402
from run_center_beam_slsqp_multistart import (  # noqa: E402
    BOUNDARY_QUADRATURE_ORDER,
    CONTACT_CONDUCTANCE_S_M2,
    CONTACT_DISCRETIZATION,
    FEASIBILITY_TOLERANCE,
    FUNCTION_TOLERANCE,
    MAX_ITERATIONS,
    TRANSITION_WIDTH_M,
    contact_record,
)
from tairte4_boundary_adjoint.baseline import (  # noqa: E402
    BASELINE_ROOT,
    ElectricalModel,
    load_config,
)
from tairte4_boundary_adjoint.optimization import run_signed_slsqp  # noqa: E402
from tairte4_boundary_adjoint.robin import DifferentiableContactModel  # noqa: E402
from tairte4_boundary_adjoint.scaled import ScaledDesign, SignedBranchObjective  # noqa: E402


BUDGETS = (12, 24, 48, 96)
PLATEAU_RELATIVE_IMPROVEMENT_TOLERANCE = 1.0e-3
SOBOL_SEED = 20260815


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def swapped(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return values[[2, 3, 0, 1]].copy()


def canonical_key(x: np.ndarray) -> tuple[float, ...]:
    values = np.asarray(x, dtype=float).copy()
    values[[0, 2]] %= 1.0
    return tuple(np.round(values, 13))


def nested_systematic_starts(
    model: DifferentiableContactModel,
    *,
    legacy_scaled: np.ndarray,
    incumbent_scaled: np.ndarray,
    minimum_length_m: float,
    maximum_length_m: float,
    minimum_gap_m: float,
    count: int,
) -> list[np.ndarray]:
    """Nested incumbent seeds followed by a deterministic scrambled Sobol set."""

    perimeter_m = model.perimeter.perimeter_m
    length_lo = minimum_length_m / perimeter_m
    length_hi = maximum_length_m / perimeter_m
    gap = minimum_gap_m / perimeter_m
    starts: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()

    def append_unique(values: np.ndarray) -> None:
        x = np.asarray(values, dtype=float).copy()
        key = canonical_key(x)
        if key not in seen:
            constraints, _ = model.perimeter.separation_constraints_scaled(x, gap)
            if np.min(constraints) < -1e-12:
                raise ValueError(f"generated infeasible seed: {constraints.tolist()}")
            starts.append(x)
            seen.add(key)

    # Every budget contains both terminal labelings of the legacy DE solution
    # and the previous campaign's best SLSQP-derived hard candidate.
    append_unique(legacy_scaled)
    append_unique(swapped(legacy_scaled))
    append_unique(incumbent_scaled)
    append_unique(swapped(incumbent_scaled))

    sampler = qmc.Sobol(d=5, scramble=True, seed=SOBOL_SEED)
    samples = sampler.random_base2(m=8)  # 256 nested candidates, more than needed.
    for sample in samples:
        center_0 = sample[0]
        length_0 = length_lo + sample[1] * (length_hi - length_lo)
        length_1 = length_lo + sample[2] * (length_hi - length_lo)
        required_distance = 0.5 * (length_0 + length_1) + gap
        distance = required_distance + sample[3] * (0.5 - required_distance)
        orientation = -1.0 if sample[4] < 0.5 else +1.0
        center_1 = center_0 + orientation * distance
        append_unique(np.asarray([center_0, length_0, center_1, length_1]))
        if len(starts) >= count:
            break
    if len(starts) != count:
        raise RuntimeError(f"could generate only {len(starts)} of {count} starts")
    return starts


def compact_run_record(
    *,
    beam_index: int,
    start_index: int,
    branch_sign: int,
    start_scaled: np.ndarray,
    start_hard_current_A: float,
    run,
) -> dict:
    # Geometry serialization is completed by the caller, which owns P.
    return {
        "beam_index": beam_index,
        "start_index": start_index,
        "branch_sign": branch_sign,
        "start_scaled": np.asarray(start_scaled, dtype=float).tolist(),
        "start_hard_current_A": start_hard_current_A,
        "start_hard_abs_current_A": abs(start_hard_current_A),
        "slsqp_success": bool(run.scipy_result.success),
        "slsqp_status_code": int(run.scipy_result.status),
        "slsqp_message": str(run.scipy_result.message),
        "iterations": int(run.scipy_result.nit),
        "unique_forward_adjoint_evaluations": run.unique_forward_adjoint_evaluations,
        "endpoint_lifted_scaled": np.asarray(run.scipy_result.x, dtype=float).tolist(),
        "endpoint_canonical_scaled": run.smooth.canonical_design.as_array().tolist(),
        "endpoint_smooth_current_A": run.smooth.current_A,
        "endpoint_hard_current_A": run.hard.current_A,
        "endpoint_hard_abs_current_A": abs(run.hard.current_A),
        "minimum_constraint": float(np.min(run.constraints)),
        "state_residual_relative": run.smooth.forward.state_residual_relative,
        "adjoint_residual_relative": run.smooth.forward.adjoint_residual_relative,
        "hard_residual_relative": run.hard.residual_relative,
    }


def candidate_from_record(record: dict, source: str) -> dict:
    if source == "start":
        return {
            "hard_abs_current_A": record["start_hard_abs_current_A"],
            "hard_current_A": record["start_hard_current_A"],
            "scaled": record["start_scaled"],
            "source": "initial_hard",
            "beam_index": record["beam_index"],
            "branch_sign": record["branch_sign"],
            "start_index": record["start_index"],
        }
    return {
        "hard_abs_current_A": record["endpoint_hard_abs_current_A"],
        "hard_current_A": record["endpoint_hard_current_A"],
        "scaled": record["endpoint_canonical_scaled"],
        "source": "slsqp_endpoint_hard",
        "beam_index": record["beam_index"],
        "branch_sign": record["branch_sign"],
        "start_index": record["start_index"],
    }


def best_at_budget(records: list[dict], beam_index: int, budget: int) -> dict:
    eligible = [
        record
        for record in records
        if record["beam_index"] == beam_index
        and record["start_index"] < budget
        and record["minimum_constraint"] >= -FEASIBILITY_TOLERANCE
    ]
    candidates = []
    for record in eligible:
        candidates.append(candidate_from_record(record, "start"))
        candidates.append(candidate_from_record(record, "endpoint"))
    if not candidates:
        raise RuntimeError(f"beam {beam_index}, budget {budget}: no candidate")
    return max(candidates, key=lambda item: item["hard_abs_current_A"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Nested 0.5 um SLSQP search-budget plateau audit"
    )
    parser.add_argument("--max-starts", type=int, choices=BUDGETS, default=48)
    args = parser.parse_args(argv)

    for gate in (
        PROJECT_ROOT / "validation" / "phase3_gradient_check.json",
        PROJECT_ROOT / "validation" / "phase3_robin_hard_convergence.json",
    ):
        if json.loads(gate.read_text(encoding="utf-8")).get("status") != "PASS":
            raise RuntimeError(f"production gate is not PASS: {gate}")

    config_path = BASELINE_ROOT / "configs" / "per_beam_500nm.json"
    fields_path = BASELINE_ROOT / "results" / "per_beam_500nm_final" / "per_beam_fields.npz"
    legacy_path = BASELINE_ROOT / "results" / "per_beam_500nm_final" / "per_beam_results.json"
    incumbent_path = HERE / "all_beams_slsqp_multistart.json"
    checkpoint_path = HERE / "search_plateau_checkpoint.json"
    output_path = HERE / "search_plateau_results.json"
    plot_path = HERE / "search_plateau.png"

    config = load_config(config_path)
    with np.load(fields_path) as fields:
        centers_m = np.asarray(fields["beam_centers_m"], dtype=float)
        temperatures_K = np.asarray(fields["temperature_nodes_K"], dtype=float)
    legacy_data = json.loads(legacy_path.read_text(encoding="utf-8"))
    legacy_by_index = {item["beam_index"]: item for item in legacy_data["per_beam_results"]}
    incumbent_data = json.loads(incumbent_path.read_text(encoding="utf-8"))
    incumbent_by_index = {item["beam_index"]: item for item in incumbent_data["beams"]}

    electrical = ElectricalModel(config)
    electrode_config = config["electrodes"]
    minimum_length_m = electrode_config["min_contact_length_um"] * 1e-6
    usable_side_m = (
        config["geometry"]["flake_width_um"]
        - 2.0 * electrode_config["edge_clearance_um"]
    ) * 1e-6
    maximum_length_m = electrode_config["max_contact_fraction"] * usable_side_m
    minimum_gap_m = electrode_config["same_side_min_gap_um"] * 1e-6

    checkpoint_identity = {
        "config_sha256": file_sha256(config_path),
        "temperature_fields_sha256": file_sha256(fields_path),
        "legacy_results_sha256": file_sha256(legacy_path),
        "incumbent_results_sha256": file_sha256(incumbent_path),
        "sobol_seed": SOBOL_SEED,
        "contact_conductance_S_m2": CONTACT_CONDUCTANCE_S_M2,
        "transition_width_m": TRANSITION_WIDTH_M,
    }
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != checkpoint_identity:
            raise RuntimeError("checkpoint identity does not match current inputs")
    else:
        checkpoint = {"identity": checkpoint_identity, "records": []}
    records: list[dict] = checkpoint["records"]
    completed = {
        (r["beam_index"], r["branch_sign"], r["start_index"]) for r in records
    }

    for beam_index, temperature_K in enumerate(temperatures_K):
        model = DifferentiableContactModel(
            electrical,
            temperature_K,
            contact_conductance_S_m2=CONTACT_CONDUCTANCE_S_M2,
            transition_m=TRANSITION_WIDTH_M,
            quadrature_order=BOUNDARY_QUADRATURE_ORDER,
            contact_discretization=CONTACT_DISCRETIZATION,
        )
        objective = SignedBranchObjective(model)
        legacy_p = legacy_parameters(model, legacy_by_index[beam_index])
        legacy_scaled = legacy_p.as_array() / model.perimeter.perimeter_m
        incumbent_scaled = np.asarray(
            incumbent_by_index[beam_index]["best_slsqp_hard_candidate"][
                "x_canonical_scaled"
            ],
            dtype=float,
        )
        starts = nested_systematic_starts(
            model,
            legacy_scaled=legacy_scaled,
            incumbent_scaled=incumbent_scaled,
            minimum_length_m=minimum_length_m,
            maximum_length_m=maximum_length_m,
            minimum_gap_m=minimum_gap_m,
            count=args.max_starts,
        )
        hard_start_cache = {}
        print(
            f"beam={beam_index + 1:02d}/9 center_um={(centers_m[beam_index] * 1e6).tolist()}",
            flush=True,
        )
        for branch_sign in (+1, -1):
            for start_index, start in enumerate(starts):
                key = (beam_index, branch_sign, start_index)
                if key in completed:
                    continue
                if start_index not in hard_start_cache:
                    physical = ScaledDesign.from_array(start).canonical().to_physical(
                        model.perimeter.perimeter_m
                    )
                    hard_start_cache[start_index] = model.hard_evaluate(physical)
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
                record = compact_run_record(
                    beam_index=beam_index,
                    start_index=start_index,
                    branch_sign=branch_sign,
                    start_scaled=start,
                    start_hard_current_A=hard_start_cache[start_index].current_A,
                    run=run,
                )
                perimeter_m = model.perimeter.perimeter_m
                endpoint_p = ScaledDesign.from_array(
                    np.asarray(record["endpoint_canonical_scaled"])
                ).to_physical(perimeter_m)
                record["start_physical_um"] = (
                    ScaledDesign.from_array(start).canonical().to_physical(perimeter_m).as_array()
                    * 1e6
                ).tolist()
                record["endpoint_physical_um"] = (endpoint_p.as_array() * 1e6).tolist()
                records.append(record)
                completed.add(key)
                checkpoint["records"] = records
                atomic_write_json(checkpoint_path, checkpoint)
                if (start_index + 1) % 4 == 0 or start_index + 1 == args.max_starts:
                    print(
                        f"  branch={branch_sign:+d} starts={start_index + 1:02d}/{args.max_starts} "
                        f"ok={run.scipy_result.success} "
                        f"start/end hard=({abs(record['start_hard_current_A']):.6e},"
                        f"{abs(record['endpoint_hard_current_A']):.6e})",
                        flush=True,
                    )

    active_budgets = [budget for budget in BUDGETS if budget <= args.max_starts]
    beams = []
    for beam_index, center_m in enumerate(centers_m):
        budget_rows = []
        for budget in active_budgets:
            best = best_at_budget(records, beam_index, budget)
            model = DifferentiableContactModel(
                electrical,
                temperatures_K[beam_index],
                contact_conductance_S_m2=CONTACT_CONDUCTANCE_S_M2,
                transition_m=TRANSITION_WIDTH_M,
                quadrature_order=BOUNDARY_QUADRATURE_ORDER,
                contact_discretization=CONTACT_DISCRETIZATION,
            )
            x = np.asarray(best["scaled"], dtype=float)
            p = ScaledDesign.from_array(x).canonical().to_physical(
                model.perimeter.perimeter_m
            )
            best["canonical_scaled"] = ScaledDesign.from_array(x).canonical().as_array().tolist()
            best["physical_um"] = (p.as_array() * 1e6).tolist()
            best["contact_0"] = contact_record(model, p.center_0_m, p.length_0_m)
            best["contact_1"] = contact_record(model, p.center_1_m, p.length_1_m)
            budget_rows.append({"starts_per_branch": budget, "best": best})
        gains = []
        for previous, current in zip(budget_rows[:-1], budget_rows[1:]):
            old = previous["best"]["hard_abs_current_A"]
            new = current["best"]["hard_abs_current_A"]
            gains.append(
                {
                    "from_starts_per_branch": previous["starts_per_branch"],
                    "to_starts_per_branch": current["starts_per_branch"],
                    "relative_improvement": (new - old) / max(new, np.finfo(float).tiny),
                }
            )
        beams.append(
            {
                "beam_index": beam_index,
                "beam_center_um": (center_m * 1e6).tolist(),
                "budgets": budget_rows,
                "budget_gains": gains,
            }
        )

    latest_gains = [beam["budget_gains"][-1]["relative_improvement"] for beam in beams]
    plateau_pass = bool(
        max(latest_gains) <= PLATEAU_RELATIVE_IMPROVEMENT_TOLERANCE
    )
    status = "PLATEAU_PASS" if plateau_pass else (
        "NEEDS_96" if args.max_starts < 96 else "PLATEAU_NOT_REACHED_AT_96"
    )
    summary = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "0.5 um nested systematic multi-start hard-current search plateau",
        "budgets_starts_per_signed_branch": active_budgets,
        "signed_branches": [+1, -1],
        "plateau_relative_improvement_tolerance": PLATEAU_RELATIVE_IMPROVEMENT_TOLERANCE,
        "plateau_definition": "maximum over beams of (I_latest-I_previous)/I_latest",
        "latest_max_relative_improvement": max(latest_gains),
        "latest_all_beam_relative_improvements": latest_gains,
        "start_sequence": "legacy DE, swapped legacy DE, prior SLSQP incumbent, swapped incumbent, then nested scrambled Sobol feasible designs",
        "hard_ranking_rule": "maximum over initial hard and SLSQP endpoint hard abs(current), both signed branches",
        "identity": checkpoint_identity,
        "all_slsqp_runs_successful": all(r["slsqp_success"] for r in records),
        "run_count": len(records),
        "beams": beams,
        "runs": records,
    }
    atomic_write_json(output_path, summary)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    for beam in beams:
        budgets = [row["starts_per_branch"] for row in beam["budgets"]]
        currents = [row["best"]["hard_abs_current_A"] * 1e9 for row in beam["budgets"]]
        label = str(tuple(int(v) for v in beam["beam_center_um"]))
        axes[0].plot(budgets, currents, "o-", label=label)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(active_budgets, [str(v) for v in active_budgets])
    axes[0].set_xlabel("starts per signed branch")
    axes[0].set_ylabel("best hard-contact |I| (nA)")
    axes[0].set_title("Nested search-budget plateau")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)
    labels = [str(tuple(int(v) for v in beam["beam_center_um"])) for beam in beams]
    axes[1].bar(np.arange(len(beams)), np.asarray(latest_gains) * 100.0)
    axes[1].axhline(
        PLATEAU_RELATIVE_IMPROVEMENT_TOLERANCE * 100.0,
        color="black",
        linestyle="--",
        label="plateau tolerance",
    )
    axes[1].set_xticks(np.arange(len(beams)), labels, rotation=45)
    axes[1].set_ylabel("latest doubling improvement (%)")
    axes[1].set_title(status)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(
        json.dumps(
            {
                "status": status,
                "active_budgets": active_budgets,
                "latest_max_relative_improvement": max(latest_gains),
                "run_count": len(records),
                "output": str(output_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
