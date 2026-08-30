from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MESH_TOLERANCE = 0.01


def main() -> int:
    thermal = json.loads(
        (HERE / "per_beam_125nm_thermal.json").read_text(encoding="utf-8")
    )
    final_250 = json.loads(
        (PROJECT_ROOT / "refinement" / "transition_width_final_250nm.json").read_text(
            encoding="utf-8"
        )
    )
    refinement_paths = sorted(HERE.glob("local_refinement_beam??_125nm.json"))
    refined = {}
    for path in refinement_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] != "PASS":
            raise RuntimeError(f"selective refinement did not pass: {path}")
        refined[int(result["beam_index"])] = (path, result)

    beams = []
    for beam_index in range(9):
        thermal_row = thermal["beams"][beam_index]
        transferred = abs(float(thermal_row["same_geometry_hard_current_125nm_A"]))
        if beam_index in refined:
            source_path, local = refined[beam_index]
            best = local["best"]
            all_success = local["all_slsqp_runs_successful"]
            improvement = local["relative_improvement_over_transferred"]
        else:
            source_path = HERE / "per_beam_125nm_thermal.json"
            best = {
                "source": "transferred_250nm_hard",
                "hard_abs_current_A": transferred,
                "hard_current_A": None,
                "canonical_scaled": final_250["beams"][beam_index]["final_best"][
                    "canonical_scaled"
                ],
            }
            all_success = True
            improvement = 0.0
        beams.append(
            {
                "beam_index": beam_index,
                "beam_center_um": thermal_row["beam_center_um"],
                "hard_abs_current_250nm_A": final_250["beams"][beam_index][
                    "final_best"
                ]["hard_abs_current_A"],
                "transferred_hard_abs_current_125nm_A": transferred,
                "best_hard_abs_current_125nm_A": best["hard_abs_current_A"],
                "relative_mesh_change_250_to_125": thermal_row[
                    "relative_current_change_250_to_125"
                ],
                "relative_local_improvement_at_125nm": improvement,
                "local_refinement_run": beam_index in refined,
                "all_slsqp_runs_successful": all_success,
                "best": best,
                "accepted_source": str(source_path),
            }
        )

    max_mesh_change = max(abs(row["relative_mesh_change_250_to_125"]) for row in beams)
    max_local_improvement = max(row["relative_local_improvement_at_125nm"] for row in beams)
    all_success = all(row["all_slsqp_runs_successful"] for row in beams)
    geometry_unchanged = all(row["relative_local_improvement_at_125nm"] <= 1e-10 for row in beams)
    status = (
        "MESH_CONVERGED_PASS" if max_mesh_change <= MESH_TOLERANCE
        else "BEST_FOUND_PASS_MESH_NOT_CONVERGED"
    )
    if not all_success:
        status = "FAIL"
    summary = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mesh_step_um": 0.125,
        "mesh_relative_tolerance": MESH_TOLERANCE,
        "maximum_absolute_relative_mesh_change_250_to_125": max_mesh_change,
        "maximum_relative_local_improvement_at_125nm": max_local_improvement,
        "all_selective_slsqp_runs_successful": all_success,
        "transferred_geometry_remains_best": geometry_unchanged,
        "selectively_refined_beam_indices": sorted(refined),
        "interpretation": (
            "The 0.25 um electrode geometries remain the 0.125 um local hard "
            "winners, but corner and x-edge currents still change by more than "
            "1%; another thermal refinement is required for mesh convergence."
        ),
        "beams": beams,
    }
    output = HERE / "final_125nm.json"
    plot = HERE / "final_125nm.png"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    x = np.arange(9)
    labels = [str(tuple(int(v) for v in row["beam_center_um"])) for row in beams]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    axes[0].plot(x, [b["hard_abs_current_250nm_A"] * 1e9 for b in beams], "o-", label="0.25 um")
    axes[0].plot(x, [b["best_hard_abs_current_125nm_A"] * 1e9 for b in beams], "o-", label="0.125 um")
    axes[0].set_xticks(x, labels, rotation=45)
    axes[0].set_ylabel("best hard |I| (nA)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].bar(x, [100 * abs(b["relative_mesh_change_250_to_125"]) for b in beams])
    axes[1].axhline(100 * MESH_TOLERANCE, color="black", linestyle="--")
    axes[1].set_xticks(x, labels, rotation=45)
    axes[1].set_ylabel("same-geometry mesh change (%)")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.savefig(plot, dpi=180)
    plt.close(fig)
    print(json.dumps({
        "status": status,
        "maximum_absolute_relative_mesh_change_250_to_125": max_mesh_change,
        "maximum_relative_local_improvement_at_125nm": max_local_improvement,
        "transferred_geometry_remains_best": geometry_unchanged,
        "output": str(output),
    }, indent=2))
    return 0 if status != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
