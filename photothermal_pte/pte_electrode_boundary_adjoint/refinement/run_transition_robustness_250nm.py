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

from run_250nm_local_refinement import geometry_change_um, local_starts  # noqa: E402
from tairte4_boundary_adjoint.baseline import ElectricalModel, load_config  # noqa: E402
from tairte4_boundary_adjoint.optimization import run_signed_slsqp  # noqa: E402
from tairte4_boundary_adjoint.robin import DifferentiableContactModel  # noqa: E402
from tairte4_boundary_adjoint.scaled import ScaledDesign, SignedBranchObjective  # noqa: E402


WIDTHS_UM_TO_RUN = (0.25, 0.50, 1.00)
REFERENCE_WIDTH_UM = 0.75
CONTACT_G_S_M2 = 1e13
STARTS_PER_BRANCH = 4
CURRENT_SPREAD_TOLERANCE = 0.01
GEOMETRY_CHANGE_TOLERANCE_UM = 2.0


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    config = load_config(PROJECT_ROOT / "configs" / "per_beam_250nm.json")
    electrical = ElectricalModel(config)
    with np.load(HERE / "per_beam_250nm_fields.npz") as fields:
        centers = np.asarray(fields["beam_centers_m"])
        temperatures = np.asarray(fields["temperature_nodes_K"])
    local_path = HERE / "local_refinement_250nm.json"
    fields_path = HERE / "per_beam_250nm_fields.npz"
    local = json.loads(local_path.read_text(encoding="utf-8"))
    checkpoint_path = HERE / "transition_robustness_checkpoint.json"
    checkpoint_identity = {
        "local_refinement_sha256": digest(local_path),
        "fields_sha256": digest(fields_path),
        "widths_um": list(WIDTHS_UM_TO_RUN),
        "contact_g_S_m2": CONTACT_G_S_M2,
        "starts_per_branch": STARTS_PER_BRANCH,
    }
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != checkpoint_identity:
            raise RuntimeError("transition checkpoint identity mismatch")
    else:
        checkpoint = {"identity": checkpoint_identity, "width_results": []}
    completed_widths = {
        (row["beam_index"], row["transition_width_um"]): row["result"]
        for row in checkpoint["width_results"]
    }
    econf = config["electrodes"]
    minimum_length_m = econf["min_contact_length_um"] * 1e-6
    maximum_length_m = econf["max_contact_fraction"] * (
        config["geometry"]["flake_width_um"] - 2 * econf["edge_clearance_um"]
    ) * 1e-6
    minimum_gap_m = econf["same_side_min_gap_um"] * 1e-6

    beam_rows = []
    all_success = True
    for beam_index, temperature in enumerate(temperatures):
        reference = local["beams"][beam_index]["best_250nm"]
        reference_x = np.asarray(reference["canonical_scaled"])
        width_rows = [
            {
                "transition_width_um": REFERENCE_WIDTH_UM,
                "best_hard_current_A": reference["hard_current_A"],
                "best_hard_abs_current_A": reference["hard_abs_current_A"],
                "best_canonical_scaled": reference_x.tolist(),
                "source": "existing_0.75um_local_refinement",
                "all_slsqp_runs_successful": True,
            }
        ]
        for width_um in WIDTHS_UM_TO_RUN:
            completed_key = (beam_index, width_um)
            if completed_key in completed_widths:
                print(
                    f"beam={beam_index + 1:02d}/9 width={width_um:.2f}um resumed",
                    flush=True,
                )
                width_rows.append(completed_widths[completed_key])
                continue
            print(
                f"beam={beam_index + 1:02d}/9 width={width_um:.2f}um",
                flush=True,
            )
            model = DifferentiableContactModel(
                electrical,
                temperature,
                contact_conductance_S_m2=CONTACT_G_S_M2,
                transition_m=width_um * 1e-6,
                contact_discretization="nodal_lumped",
            )
            objective = SignedBranchObjective(model)
            starts = local_starts(
                reference_x,
                perimeter=model.perimeter,
                minimum_length_m=minimum_length_m,
                maximum_length_m=maximum_length_m,
                minimum_gap_m=minimum_gap_m,
            )[:STARTS_PER_BRANCH]
            reference_p = ScaledDesign.from_array(reference_x).to_physical(
                model.perimeter.perimeter_m
            )
            hard_reference = model.hard_evaluate(reference_p)
            smooth_reference = model.evaluate(reference_p)
            candidates = [
                {
                    "hard_current_A": hard_reference.current_A,
                    "hard_abs_current_A": abs(hard_reference.current_A),
                    "canonical_scaled": reference_x.tolist(),
                    "source": "reference_geometry_hard",
                }
            ]
            run_success = []
            run_records = []
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
                    run_success.append(bool(run.scipy_result.success))
                    all_success = all_success and bool(run.scipy_result.success)
                    endpoint = run.smooth.canonical_design.as_array()
                    candidates.append(
                        {
                            "hard_current_A": run.hard.current_A,
                            "hard_abs_current_A": abs(run.hard.current_A),
                            "canonical_scaled": endpoint.tolist(),
                            "source": "slsqp_endpoint_hard",
                            "branch_sign": branch_sign,
                            "start_index": start_index,
                        }
                    )
                    run_records.append(
                        {
                            "branch_sign": branch_sign,
                            "start_index": start_index,
                            "success": bool(run.scipy_result.success),
                            "iterations": int(run.scipy_result.nit),
                            "endpoint_smooth_current_A": run.smooth.current_A,
                            "endpoint_hard_current_A": run.hard.current_A,
                            "endpoint_canonical_scaled": endpoint.tolist(),
                            "minimum_constraint": float(np.min(run.constraints)),
                        }
                    )
            best = max(candidates, key=lambda item: item["hard_abs_current_A"])
            width_result = {
                    "transition_width_um": width_um,
                    "reference_smooth_current_A": smooth_reference.current_A,
                    "reference_hard_current_A": hard_reference.current_A,
                    "reference_relaxation_current_relative_error": abs(
                        smooth_reference.current_A - hard_reference.current_A
                    ) / abs(hard_reference.current_A),
                    "best_hard_current_A": best["hard_current_A"],
                    "best_hard_abs_current_A": best["hard_abs_current_A"],
                    "best_canonical_scaled": best["canonical_scaled"],
                    "source": best["source"],
                    "all_slsqp_runs_successful": all(run_success),
                    "runs": run_records,
                }
            width_rows.append(width_result)
            checkpoint["width_results"].append(
                {
                    "beam_index": beam_index,
                    "transition_width_um": width_um,
                    "result": width_result,
                }
            )
            completed_widths[completed_key] = width_result
            atomic_write_json(checkpoint_path, checkpoint)
        width_rows.sort(key=lambda item: item["transition_width_um"])
        currents = np.asarray([row["best_hard_abs_current_A"] for row in width_rows])
        spread = float((np.max(currents) - np.min(currents)) / np.max(currents))
        geometry_changes = []
        reference_model = DifferentiableContactModel(
            electrical,
            temperature,
            contact_conductance_S_m2=CONTACT_G_S_M2,
            transition_m=REFERENCE_WIDTH_UM * 1e-6,
        )
        for row in width_rows:
            geometry_changes.append(
                geometry_change_um(
                    reference_x,
                    np.asarray(row["best_canonical_scaled"]),
                    perimeter=reference_model.perimeter,
                    beam_center_um=centers[beam_index] * 1e6,
                )
            )
        max_geometry = max(
            item["maximum_absolute_parameter_change_um"]
            for item in geometry_changes
        )
        beam_rows.append(
            {
                "beam_index": beam_index,
                "beam_center_um": (centers[beam_index] * 1e6).tolist(),
                "hard_current_relative_spread": spread,
                "maximum_symmetry_aligned_geometry_change_um": max_geometry,
                "current_robustness_pass": spread <= CURRENT_SPREAD_TOLERANCE,
                "geometry_robustness_pass": max_geometry <= GEOMETRY_CHANGE_TOLERANCE_UM,
                "widths": [
                    {**row, "geometry_change_from_reference": change}
                    for row, change in zip(width_rows, geometry_changes)
                ],
            }
        )

    current_pass = all(row["current_robustness_pass"] for row in beam_rows)
    geometry_pass = all(row["geometry_robustness_pass"] for row in beam_rows)
    status = (
        "PASS"
        if all_success and current_pass and geometry_pass
        else "CURRENT_PASS_GEOMETRY_SENSITIVE"
        if all_success and current_pass
        else "FAIL"
    )
    summary = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mesh_step_um": 0.25,
        "transition_widths_um": sorted((*WIDTHS_UM_TO_RUN, REFERENCE_WIDTH_UM)),
        "contact_g_S_m2": CONTACT_G_S_M2,
        "starts_per_signed_branch_for_new_widths": STARTS_PER_BRANCH,
        "current_relative_spread_tolerance": CURRENT_SPREAD_TOLERANCE,
        "symmetry_aligned_geometry_change_tolerance_um": GEOMETRY_CHANGE_TOLERANCE_UM,
        "all_slsqp_runs_successful": all_success,
        "all_beam_current_robustness_pass": current_pass,
        "all_beam_geometry_robustness_pass": geometry_pass,
        "maximum_hard_current_relative_spread": max(
            row["hard_current_relative_spread"] for row in beam_rows
        ),
        "maximum_symmetry_aligned_geometry_change_um": max(
            row["maximum_symmetry_aligned_geometry_change_um"] for row in beam_rows
        ),
        "beams": beam_rows,
    }
    output_path = HERE / "transition_robustness_250nm.json"
    plot_path = HERE / "transition_robustness_250nm.png"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    labels = [str(tuple(int(v) for v in row["beam_center_um"])) for row in beam_rows]
    x = np.arange(len(beam_rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    for width_um in sorted((*WIDTHS_UM_TO_RUN, REFERENCE_WIDTH_UM)):
        values = []
        for beam in beam_rows:
            values.append(
                next(
                    row["best_hard_abs_current_A"]
                    for row in beam["widths"]
                    if row["transition_width_um"] == width_um
                ) * 1e9
            )
        axes[0].plot(x, values, "o-", label=f"{width_um:g} um")
    axes[0].set_xticks(x, labels, rotation=45)
    axes[0].set_ylabel("best hard-contact |I| (nA)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title="transition")
    axes[1].bar(
        x,
        [100 * row["hard_current_relative_spread"] for row in beam_rows],
    )
    axes[1].axhline(100 * CURRENT_SPREAD_TOLERANCE, color="black", linestyle="--")
    axes[1].set_xticks(x, labels, rotation=45)
    axes[1].set_ylabel("hard-current spread across widths (%)")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(json.dumps({
        "status": status,
        "maximum_hard_current_relative_spread": summary["maximum_hard_current_relative_spread"],
        "maximum_symmetry_aligned_geometry_change_um": summary["maximum_symmetry_aligned_geometry_change_um"],
        "output": str(output_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
