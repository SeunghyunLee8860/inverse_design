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
sys.path.insert(0, str(PROJECT_ROOT / "optimization"))

from run_center_beam_slsqp_multistart import contact_record  # noqa: E402
from tairte4_boundary_adjoint.baseline import ElectricalModel, load_config  # noqa: E402
from tairte4_boundary_adjoint.optimization import run_signed_slsqp  # noqa: E402
from tairte4_boundary_adjoint.robin import DifferentiableContactModel  # noqa: E402
from tairte4_boundary_adjoint.scaled import ScaledDesign, SignedBranchObjective  # noqa: E402


CONTACT_G_S_M2 = 1e13
TRANSITION_M = 0.75e-6
LOCAL_PERTURBATION_M = 0.25e-6
STARTS_PER_BRANCH = 8


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def periodic_delta(a: float, b: float) -> float:
    return float((a - b + 0.5) % 1.0 - 0.5)


def reflected_scaled(x: np.ndarray, *, flip_x: bool, flip_y: bool, perimeter) -> np.ndarray:
    result = np.asarray(x, dtype=float).copy()
    for center_index in (0, 2):
        center_m = result[center_index] * perimeter.perimeter_m
        side, tangent_m = perimeter.s_to_side_coordinate(center_m)
        if flip_x:
            if side == "left":
                side = "right"
            elif side == "right":
                side = "left"
            else:
                tangent_m = -tangent_m
        if flip_y:
            if side == "bottom":
                side = "top"
            elif side == "top":
                side = "bottom"
            else:
                tangent_m = -tangent_m
        result[center_index] = (
            perimeter.side_coordinate_to_s(side, tangent_m)
            / perimeter.perimeter_m
        )
    return result


def geometry_change_um(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    perimeter,
    beam_center_um: np.ndarray,
) -> dict:
    def difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                periodic_delta(first[0], second[0]),
                first[1] - second[1],
                periodic_delta(first[2], second[2]),
                first[3] - second[3],
            ]
        )

    flip_x_options = (False, True) if abs(beam_center_um[0]) < 1e-12 else (False,)
    flip_y_options = (False, True) if abs(beam_center_um[1]) < 1e-12 else (False,)
    alignments = []
    for flip_x in flip_x_options:
        for flip_y in flip_y_options:
            reflected = reflected_scaled(
                reference, flip_x=flip_x, flip_y=flip_y, perimeter=perimeter
            )
            for swap_terminals in (False, True):
                aligned = reflected[[2, 3, 0, 1]] if swap_terminals else reflected
                delta = difference(candidate, aligned)
                alignments.append(
                    (np.linalg.norm(delta), delta, flip_x, flip_y, swap_terminals)
                )
    _, delta, flip_x, flip_y, swap_terminals = min(
        alignments, key=lambda item: item[0]
    )
    delta_um = delta * perimeter.perimeter_m * 1e6
    return {
        "flip_x_alignment": flip_x,
        "flip_y_alignment": flip_y,
        "terminal_alignment": "swapped" if swap_terminals else "direct",
        "delta_c0_L0_c1_L1_um": delta_um.tolist(),
        "maximum_absolute_parameter_change_um": float(np.max(np.abs(delta_um))),
        "l2_parameter_change_um": float(np.linalg.norm(delta_um)),
    }


def local_starts(
    base: np.ndarray,
    *,
    perimeter,
    minimum_length_m: float,
    maximum_length_m: float,
    minimum_gap_m: float,
) -> list[np.ndarray]:
    delta = LOCAL_PERTURBATION_M / perimeter.perimeter_m
    lo = minimum_length_m / perimeter.perimeter_m
    hi = maximum_length_m / perimeter.perimeter_m
    directions = [
        np.zeros(4),
        np.asarray([0.0, 0.0, 0.0, 0.0]),  # replaced by swapped base below
        np.asarray([+1.0, 0.0, -1.0, 0.0]),
        np.asarray([-1.0, 0.0, +1.0, 0.0]),
        np.asarray([0.0, +1.0, 0.0, -1.0]),
        np.asarray([0.0, -1.0, 0.0, +1.0]),
        np.asarray([+1.0, +1.0, -1.0, -1.0]),
        np.asarray([-1.0, -1.0, +1.0, +1.0]),
    ]
    candidates = [base.copy(), base[[2, 3, 0, 1]].copy()]
    for direction in directions[2:]:
        x = base + delta * direction
        x[1] = np.clip(x[1], lo, hi)
        x[3] = np.clip(x[3], lo, hi)
        constraints, _ = perimeter.separation_constraints_scaled(
            x, minimum_gap_m / perimeter.perimeter_m
        )
        if np.min(constraints) >= -1e-12:
            candidates.append(x)
    unique = []
    keys = set()
    for x in candidates:
        canonical = x.copy()
        canonical[[0, 2]] %= 1.0
        key = tuple(np.round(canonical, 13))
        if key not in keys:
            unique.append(x)
            keys.add(key)
    # If clipping/deduplication removed a perturbation, add smaller center shifts.
    multiplier = 0.5
    while len(unique) < STARTS_PER_BRANCH:
        for sign in (-1.0, +1.0):
            x = base + multiplier * delta * np.asarray([sign, 0.0, sign, 0.0])
            canonical = x.copy()
            canonical[[0, 2]] %= 1.0
            key = tuple(np.round(canonical, 13))
            if key not in keys:
                unique.append(x)
                keys.add(key)
            if len(unique) == STARTS_PER_BRANCH:
                break
        multiplier *= 0.5
    return unique[:STARTS_PER_BRANCH]


def main() -> int:
    config_path = PROJECT_ROOT / "configs" / "per_beam_250nm.json"
    fields_path = HERE / "per_beam_250nm_fields.npz"
    plateau_path = PROJECT_ROOT / "optimization" / "search_plateau_results.json"
    relaxation_path = HERE / "relaxation_250nm.json"
    if json.loads(relaxation_path.read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("0.25 um relaxation/gradient gate is not PASS")
    config = load_config(config_path)
    electrical = ElectricalModel(config)
    with np.load(fields_path) as fields:
        centers = np.asarray(fields["beam_centers_m"])
        temperatures = np.asarray(fields["temperature_nodes_K"])
    plateau = json.loads(plateau_path.read_text(encoding="utf-8"))

    econf = config["electrodes"]
    minimum_length_m = econf["min_contact_length_um"] * 1e-6
    usable_side_m = (
        config["geometry"]["flake_width_um"] - 2.0 * econf["edge_clearance_um"]
    ) * 1e-6
    maximum_length_m = econf["max_contact_fraction"] * usable_side_m
    minimum_gap_m = econf["same_side_min_gap_um"] * 1e-6

    beam_rows = []
    all_success = True
    for beam_index, temperature in enumerate(temperatures):
        print(f"beam={beam_index + 1:02d}/9 center_um={(centers[beam_index] * 1e6).tolist()}", flush=True)
        model = DifferentiableContactModel(
            electrical,
            temperature,
            contact_conductance_S_m2=CONTACT_G_S_M2,
            transition_m=TRANSITION_M,
            contact_discretization="nodal_lumped",
        )
        objective = SignedBranchObjective(model)
        source_best = plateau["beams"][beam_index]["budgets"][-1]["best"]
        base = np.asarray(source_best["canonical_scaled"], dtype=float)
        base_physical = ScaledDesign.from_array(base).to_physical(
            model.perimeter.perimeter_m
        )
        transferred = model.hard_evaluate(base_physical)
        starts = local_starts(
            base,
            perimeter=model.perimeter,
            minimum_length_m=minimum_length_m,
            maximum_length_m=maximum_length_m,
            minimum_gap_m=minimum_gap_m,
        )
        candidates = [
            {
                "source": "transferred_500nm_best_hard",
                "hard_current_A": transferred.current_A,
                "hard_abs_current_A": abs(transferred.current_A),
                "canonical_scaled": ScaledDesign.from_array(base).canonical().as_array().tolist(),
            }
        ]
        runs = []
        hard_start_cache = {}
        for branch_sign in (+1, -1):
            for start_index, start in enumerate(starts):
                if start_index not in hard_start_cache:
                    p = ScaledDesign.from_array(start).canonical().to_physical(
                        model.perimeter.perimeter_m
                    )
                    hard_start_cache[start_index] = model.hard_evaluate(p)
                run = run_signed_slsqp(
                    objective,
                    start,
                    branch_sign=branch_sign,
                    minimum_length_m=minimum_length_m,
                    maximum_length_m=maximum_length_m,
                    minimum_gap_m=minimum_gap_m,
                    max_iterations=250,
                    function_tolerance=1e-11,
                )
                all_success = all_success and bool(run.scipy_result.success)
                endpoint = run.smooth.canonical_design.as_array()
                runs.append(
                    {
                        "branch_sign": branch_sign,
                        "start_index": start_index,
                        "success": bool(run.scipy_result.success),
                        "message": str(run.scipy_result.message),
                        "iterations": int(run.scipy_result.nit),
                        "start_scaled": start.tolist(),
                        "start_hard_current_A": hard_start_cache[start_index].current_A,
                        "endpoint_scaled": endpoint.tolist(),
                        "endpoint_smooth_current_A": run.smooth.current_A,
                        "endpoint_hard_current_A": run.hard.current_A,
                        "minimum_constraint": float(np.min(run.constraints)),
                    }
                )
                candidates.append(
                    {
                        "source": "local_slsqp_endpoint_hard",
                        "branch_sign": branch_sign,
                        "start_index": start_index,
                        "hard_current_A": run.hard.current_A,
                        "hard_abs_current_A": abs(run.hard.current_A),
                        "canonical_scaled": endpoint.tolist(),
                    }
                )
        best = max(candidates, key=lambda row: row["hard_abs_current_A"])
        best_x = np.asarray(best["canonical_scaled"])
        best_p = ScaledDesign.from_array(best_x).to_physical(model.perimeter.perimeter_m)
        best["physical_um"] = (best_p.as_array() * 1e6).tolist()
        best["contact_0"] = contact_record(model, best_p.center_0_m, best_p.length_0_m)
        best["contact_1"] = contact_record(model, best_p.center_1_m, best_p.length_1_m)
        old_current = source_best["hard_abs_current_A"]
        beam_rows.append(
            {
                "beam_index": beam_index,
                "beam_center_um": (centers[beam_index] * 1e6).tolist(),
                "source_500nm_best": source_best,
                "transferred_250nm_hard_current_A": transferred.current_A,
                "transferred_250nm_hard_abs_current_A": abs(transferred.current_A),
                "relative_current_change_500_to_250_same_geometry": (
                    abs(transferred.current_A) - old_current
                ) / old_current,
                "best_250nm": best,
                "relative_local_improvement_over_transfer": (
                    best["hard_abs_current_A"] - abs(transferred.current_A)
                ) / abs(transferred.current_A),
                "geometry_change_from_500nm": geometry_change_um(
                    base,
                    best_x,
                    perimeter=model.perimeter,
                    beam_center_um=centers[beam_index] * 1e6,
                ),
                "runs": runs,
            }
        )
        print(
            f"  transfer={abs(transferred.current_A):.6e} "
            f"local_best={best['hard_abs_current_A']:.6e} "
            f"gain={(best['hard_abs_current_A']/abs(transferred.current_A)-1)*100:.3f}%",
            flush=True,
        )

    summary = {
        "status": "COMPLETED" if all_success else "COMPLETED_WITH_SLSQP_WARNINGS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "transfer 0.5 um plateau winners to 0.25 um and local signed-branch reoptimization",
        "mesh_step_um": 0.25,
        "contact_g_S_m2": CONTACT_G_S_M2,
        "transition_width_um": TRANSITION_M * 1e6,
        "starts_per_signed_branch": STARTS_PER_BRANCH,
        "local_perturbation_um": LOCAL_PERTURBATION_M * 1e6,
        "ranking_rule": "maximum hard abs(current) over transferred geometry and all local SLSQP endpoints",
        "config_sha256": digest(config_path),
        "fields_sha256": digest(fields_path),
        "source_500nm_plateau_sha256": digest(plateau_path),
        "relaxation_gate_sha256": digest(relaxation_path),
        "beams": beam_rows,
    }
    output_path = HERE / "local_refinement_250nm.json"
    plot_path = HERE / "local_refinement_250nm.png"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    labels = [str(tuple(int(v) for v in row["beam_center_um"])) for row in beam_rows]
    old = np.asarray([row["source_500nm_best"]["hard_abs_current_A"] for row in beam_rows]) * 1e9
    transfer = np.asarray([row["transferred_250nm_hard_abs_current_A"] for row in beam_rows]) * 1e9
    local = np.asarray([row["best_250nm"]["hard_abs_current_A"] for row in beam_rows]) * 1e9
    x = np.arange(len(beam_rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    axes[0].plot(x, old, "o--", label="0.5 um final")
    axes[0].plot(x, transfer, "s--", label="same geometry at 0.25 um")
    axes[0].plot(x, local, "^-", label="0.25 um local best")
    axes[0].set_xticks(x, labels, rotation=45)
    axes[0].set_ylabel("hard-contact |I| (nA)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    changes = [row["geometry_change_from_500nm"]["maximum_absolute_parameter_change_um"] for row in beam_rows]
    axes[1].bar(x, changes)
    axes[1].set_xticks(x, labels, rotation=45)
    axes[1].set_ylabel("max |geometry parameter change| (um)")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(json.dumps({
        "status": summary["status"],
        "max_same_geometry_current_change": max(abs(row["relative_current_change_500_to_250_same_geometry"]) for row in beam_rows),
        "max_local_improvement": max(row["relative_local_improvement_over_transfer"] for row in beam_rows),
        "max_geometry_parameter_change_um": max(row["geometry_change_from_500nm"]["maximum_absolute_parameter_change_um"] for row in beam_rows),
        "output": str(output_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
