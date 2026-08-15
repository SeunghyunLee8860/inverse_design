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

from run_250nm_local_refinement import geometry_change_um  # noqa: E402
from tairte4_boundary_adjoint.baseline import ElectricalModel, load_config  # noqa: E402
from tairte4_boundary_adjoint.optimization import run_signed_slsqp  # noqa: E402
from tairte4_boundary_adjoint.robin import DifferentiableContactModel  # noqa: E402
from tairte4_boundary_adjoint.scaled import ScaledDesign, SignedBranchObjective  # noqa: E402


WIDTHS_UM = (0.25, 0.50, 0.75, 1.00)
CONTACT_G_S_M2 = 1e14
STARTS_PER_BRANCH = 4
CLOSURE_TOLERANCE = 1e-3
GEOMETRY_TOLERANCE_UM = 2.0


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def key(x: np.ndarray) -> tuple[float, ...]:
    value = np.asarray(x).copy()
    value[[0, 2]] %= 1.0
    return tuple(np.round(value, 13))


def common_starts(
    strongest: np.ndarray,
    reference: np.ndarray,
    perimeter,
    *,
    minimum_gap_m: float,
) -> list[np.ndarray]:
    delta = 0.25e-6 / perimeter.perimeter_m
    candidates = [
        strongest,
        strongest[[2, 3, 0, 1]],
        reference,
        reference[[2, 3, 0, 1]],
        strongest + delta * np.asarray([+1.0, 0.0, -1.0, 0.0]),
        strongest + delta * np.asarray([-1.0, 0.0, +1.0, 0.0]),
        strongest + delta * np.asarray([+1.0, 0.0, +1.0, 0.0]),
        strongest + delta * np.asarray([-1.0, 0.0, -1.0, 0.0]),
    ]
    unique = []
    seen = set()
    for candidate in candidates:
        constraints, _ = perimeter.separation_constraints_scaled(
            candidate, minimum_gap_m / perimeter.perimeter_m
        )
        if np.min(constraints) < -1e-12:
            continue
        if key(candidate) not in seen:
            unique.append(np.asarray(candidate).copy())
            seen.add(key(candidate))
        if len(unique) == STARTS_PER_BRANCH:
            break
    if len(unique) < STARTS_PER_BRANCH:
        raise RuntimeError("could not construct four distinct common starts")
    return unique


def width_candidate(row: dict) -> tuple[float, np.ndarray]:
    """Return the hard-ranked candidate from either audit JSON schema."""
    if "best" in row:
        return (
            float(row["best"]["hard_abs_current_A"]),
            np.asarray(row["best"]["canonical_scaled"], dtype=float),
        )
    return (
        float(row["best_hard_abs_current_A"]),
        np.asarray(row["best_canonical_scaled"], dtype=float),
    )


def run_cross_seed(
    *,
    input_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    plot_path: Path,
    beam_indices: tuple[int, ...] | None = None,
) -> int:
    failed_path = input_path
    local_path = HERE / "local_refinement_250nm.json"
    fields_path = HERE / "per_beam_250nm_fields.npz"
    relaxation_path = HERE / "relaxation_250nm.json"
    if json.loads(relaxation_path.read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("corrected 0.25 um relaxation gate is not PASS")
    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    local = json.loads(local_path.read_text(encoding="utf-8"))
    config = load_config(PROJECT_ROOT / "configs" / "per_beam_250nm.json")
    electrical = ElectricalModel(config)
    with np.load(fields_path) as fields:
        centers = np.asarray(fields["beam_centers_m"])
        temperatures = np.asarray(fields["temperature_nodes_K"])
    econf = config["electrodes"]
    minimum_length_m = econf["min_contact_length_um"] * 1e-6
    maximum_length_m = econf["max_contact_fraction"] * (
        config["geometry"]["flake_width_um"] - 2 * econf["edge_clearance_um"]
    ) * 1e-6
    minimum_gap_m = econf["same_side_min_gap_um"] * 1e-6

    selected_beams = tuple(range(len(temperatures))) if beam_indices is None else beam_indices
    identity = {
        "failed_audit_sha256": digest(failed_path),
        "local_refinement_sha256": digest(local_path),
        "fields_sha256": digest(fields_path),
        "relaxation_sha256": digest(relaxation_path),
        "g": CONTACT_G_S_M2,
        "widths": list(WIDTHS_UM),
        "starts_per_branch": STARTS_PER_BRANCH,
        "beam_indices": list(selected_beams),
    }
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        legacy_identity = dict(identity)
        legacy_identity.pop("starts_per_branch")
        legacy_identity.pop("beam_indices")
        if checkpoint.get("identity") == legacy_identity:
            checkpoint["identity"] = identity
            atomic_write(checkpoint_path, checkpoint)
        elif checkpoint.get("identity") != identity:
            raise RuntimeError("cross-seed checkpoint identity mismatch")
    else:
        checkpoint = {"identity": identity, "results": []}
    completed = {
        (item["beam_index"], item["width_um"]): item["result"]
        for item in checkpoint["results"]
    }

    beam_rows = []
    all_success = True
    for beam_index, temperature in enumerate(temperatures):
        if beam_index not in selected_beams:
            continue
        strongest_width = max(
            failed["beams"][beam_index]["widths"],
            key=lambda row: width_candidate(row)[0],
        )
        seed_abs_current, strongest = width_candidate(strongest_width)
        reference = np.asarray(
            local["beams"][beam_index]["best_250nm"]["canonical_scaled"]
        )
        width_rows = []
        for width_um in WIDTHS_UM:
            completed_key = (beam_index, width_um)
            if completed_key in completed:
                print(f"beam={beam_index + 1:02d}/9 width={width_um:.2f} resumed", flush=True)
                width_rows.append(completed[completed_key])
                continue
            print(f"beam={beam_index + 1:02d}/9 width={width_um:.2f}", flush=True)
            model = DifferentiableContactModel(
                electrical,
                temperature,
                contact_conductance_S_m2=CONTACT_G_S_M2,
                transition_m=width_um * 1e-6,
                contact_discretization="nodal_lumped",
            )
            objective = SignedBranchObjective(model)
            starts = common_starts(
                strongest,
                reference,
                model.perimeter,
                minimum_gap_m=minimum_gap_m,
            )
            candidates = []
            hard_starts = []
            for start in starts:
                p = ScaledDesign.from_array(start).canonical().to_physical(
                    model.perimeter.perimeter_m
                )
                hard = model.hard_evaluate(p)
                hard_starts.append(hard)
                candidates.append(
                    {
                        "source": "common_initial_hard",
                        "hard_current_A": hard.current_A,
                        "hard_abs_current_A": abs(hard.current_A),
                        "canonical_scaled": ScaledDesign.from_array(start).canonical().as_array().tolist(),
                    }
                )
            strongest_p = ScaledDesign.from_array(strongest).to_physical(
                model.perimeter.perimeter_m
            )
            strongest_hard = model.hard_evaluate(strongest_p)
            strongest_smooth = model.evaluate(strongest_p)
            run_rows = []
            successes = []
            for branch_sign in (+1, -1):
                for start_index, start in enumerate(starts):
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
                    successes.append(bool(run.scipy_result.success))
                    all_success = all_success and bool(run.scipy_result.success)
                    endpoint = run.smooth.canonical_design.as_array()
                    candidates.append(
                        {
                            "source": "slsqp_endpoint_hard",
                            "branch_sign": branch_sign,
                            "start_index": start_index,
                            "hard_current_A": run.hard.current_A,
                            "hard_abs_current_A": abs(run.hard.current_A),
                            "canonical_scaled": endpoint.tolist(),
                        }
                    )
                    run_rows.append(
                        {
                            "branch_sign": branch_sign,
                            "start_index": start_index,
                            "success": bool(run.scipy_result.success),
                            "iterations": int(run.scipy_result.nit),
                            "endpoint_hard_current_A": run.hard.current_A,
                            "endpoint_smooth_current_A": run.smooth.current_A,
                            "endpoint_canonical_scaled": endpoint.tolist(),
                        }
                    )
            best = max(candidates, key=lambda row: row["hard_abs_current_A"])
            result = {
                "transition_width_um": width_um,
                "common_seed_best_hard_abs_current_A": seed_abs_current,
                "strongest_seed_relaxation_current_relative_error": abs(
                    strongest_smooth.current_A - strongest_hard.current_A
                ) / abs(strongest_hard.current_A),
                "best": best,
                "relative_improvement_over_common_seed": (
                    best["hard_abs_current_A"] - seed_abs_current
                ) / seed_abs_current,
                "all_slsqp_runs_successful": all(successes),
                "runs": run_rows,
            }
            width_rows.append(result)
            checkpoint["results"].append(
                {"beam_index": beam_index, "width_um": width_um, "result": result}
            )
            completed[completed_key] = result
            atomic_write(checkpoint_path, checkpoint)
        currents = np.asarray([row["best"]["hard_abs_current_A"] for row in width_rows])
        spread = float((np.max(currents) - np.min(currents)) / np.max(currents))
        max_improvement = max(row["relative_improvement_over_common_seed"] for row in width_rows)
        geometry_changes = []
        alignment_model = DifferentiableContactModel(
            electrical,
            temperature,
            contact_conductance_S_m2=CONTACT_G_S_M2,
            transition_m=0.75e-6,
        )
        final_reference = np.asarray(max(width_rows, key=lambda r: r["best"]["hard_abs_current_A"])["best"]["canonical_scaled"])
        for row in width_rows:
            geometry_changes.append(
                geometry_change_um(
                    final_reference,
                    np.asarray(row["best"]["canonical_scaled"]),
                    perimeter=alignment_model.perimeter,
                    beam_center_um=centers[beam_index] * 1e6,
                )
            )
        beam_rows.append(
            {
                "beam_index": beam_index,
                "beam_center_um": (centers[beam_index] * 1e6).tolist(),
                "input_common_seed_hard_abs_current_A": seed_abs_current,
                "hard_current_relative_spread": spread,
                "maximum_improvement_over_input_pool": max_improvement,
                "maximum_symmetry_aligned_geometry_change_um": max(
                    g["maximum_absolute_parameter_change_um"] for g in geometry_changes
                ),
                "widths": [
                    {**row, "geometry_change_from_final_reference": change}
                    for row, change in zip(width_rows, geometry_changes)
                ],
            }
        )

    max_spread = max(row["hard_current_relative_spread"] for row in beam_rows)
    max_improvement = max(row["maximum_improvement_over_input_pool"] for row in beam_rows)
    max_geometry = max(row["maximum_symmetry_aligned_geometry_change_um"] for row in beam_rows)
    max_relaxation_error = max(
        width["strongest_seed_relaxation_current_relative_error"]
        for beam in beam_rows
        for width in beam["widths"]
    )
    closure_pass = max_spread <= CLOSURE_TOLERANCE and max_improvement <= CLOSURE_TOLERANCE
    status = "PASS" if (
        all_success
        and closure_pass
        and max_geometry <= GEOMETRY_TOLERANCE_UM
        and max_relaxation_error <= 0.01
    ) else "NEEDS_ANOTHER_CROSS_SEED_ITERATION" if max_improvement > CLOSURE_TOLERANCE else "FAIL"
    summary = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mesh_step_um": 0.25,
        "contact_g_S_m2": CONTACT_G_S_M2,
        "transition_widths_um": list(WIDTHS_UM),
        "common_starts_per_signed_branch": STARTS_PER_BRANCH,
        "input_audit": str(failed_path),
        "beam_indices": list(selected_beams),
        "closure_tolerance": CLOSURE_TOLERANCE,
        "geometry_tolerance_um": GEOMETRY_TOLERANCE_UM,
        "all_slsqp_runs_successful": all_success,
        "maximum_hard_current_relative_spread": max_spread,
        "maximum_improvement_over_input_pool": max_improvement,
        "maximum_symmetry_aligned_geometry_change_um": max_geometry,
        "maximum_strongest_seed_relaxation_current_relative_error": max_relaxation_error,
        "beams": beam_rows,
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    labels = [str(tuple(int(v) for v in row["beam_center_um"])) for row in beam_rows]
    x = np.arange(len(beam_rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    for width_um in WIDTHS_UM:
        axes[0].plot(
            x,
            [next(w["best"]["hard_abs_current_A"] for w in b["widths"] if w["transition_width_um"] == width_um) * 1e9 for b in beam_rows],
            "o-",
            label=f"{width_um:g} um",
        )
    axes[0].set_xticks(x, labels, rotation=45)
    axes[0].set_ylabel("cross-seeded best hard |I| (nA)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].bar(x, [100*b["hard_current_relative_spread"] for b in beam_rows])
    axes[1].axhline(100*CLOSURE_TOLERANCE, color="black", linestyle="--")
    axes[1].set_xticks(x, labels, rotation=45)
    axes[1].set_ylabel("width spread (%)")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(json.dumps({
        "status": status,
        "maximum_hard_current_relative_spread": max_spread,
        "maximum_improvement_over_input_pool": max_improvement,
        "maximum_geometry_change_um": max_geometry,
        "maximum_relaxation_error": max_relaxation_error,
        "output": str(output_path),
    }, indent=2))
    return 0


def main() -> int:
    return run_cross_seed(
        input_path=HERE / "transition_robustness_250nm.json",
        checkpoint_path=HERE / "transition_cross_seed_checkpoint.json",
        output_path=HERE / "transition_cross_seed_250nm.json",
        plot_path=HERE / "transition_cross_seed_250nm.png",
    )


if __name__ == "__main__":
    raise SystemExit(main())
