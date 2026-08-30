from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
CURRENT_TOLERANCE = 1e-3
GEOMETRY_TOLERANCE_UM = 2.0
RELAXATION_TOLERANCE = 0.01
WIDTHS_UM = (0.25, 0.50, 0.75, 1.00)


def main() -> int:
    first_path = HERE / "transition_cross_seed_250nm.json"
    first = json.loads(first_path.read_text(encoding="utf-8"))
    first_by_beam = {int(row["beam_index"]): row for row in first["beams"]}
    closure_paths = sorted(HERE.glob("transition_cross_seed_closure_beam??_250nm.json"))
    closure_by_beam = {}
    for path in closure_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] != "PASS" or len(result["beams"]) != 1:
            raise RuntimeError(f"closure file is not a one-beam PASS: {path}")
        beam = result["beams"][0]
        closure_by_beam[int(beam["beam_index"])] = (path, beam)

    beams = []
    for beam_index in range(9):
        first_beam = first_by_beam[beam_index]
        first_needs_closure = (
            first_beam["hard_current_relative_spread"] > CURRENT_TOLERANCE
            or first_beam["maximum_improvement_over_input_pool"] > CURRENT_TOLERANCE
        )
        if first_needs_closure:
            if beam_index not in closure_by_beam:
                raise RuntimeError(f"missing closure result for beam {beam_index}")
            source_path, chosen = closure_by_beam[beam_index]
            iteration = 2
        else:
            source_path, chosen = first_path, first_beam
            iteration = 1
        best_width = max(
            chosen["widths"], key=lambda row: row["best"]["hard_abs_current_A"]
        )
        beams.append(
            {
                **chosen,
                "accepted_iteration": iteration,
                "accepted_source": str(source_path),
                "final_best": best_width["best"],
                "final_best_transition_width_um": best_width["transition_width_um"],
            }
        )

    max_spread = max(row["hard_current_relative_spread"] for row in beams)
    max_improvement = max(row["maximum_improvement_over_input_pool"] for row in beams)
    max_geometry = max(
        row["maximum_symmetry_aligned_geometry_change_um"] for row in beams
    )
    max_relaxation = max(
        width["strongest_seed_relaxation_current_relative_error"]
        for beam in beams
        for width in beam["widths"]
    )
    all_success = all(
        width["all_slsqp_runs_successful"]
        for beam in beams
        for width in beam["widths"]
    )
    status = "PASS" if (
        all_success
        and max_spread <= CURRENT_TOLERANCE
        and max_improvement <= CURRENT_TOLERANCE
        and max_geometry <= GEOMETRY_TOLERANCE_UM
        and max_relaxation <= RELAXATION_TOLERANCE
    ) else "FAIL"
    summary = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mesh_step_um": 0.25,
        "contact_g_S_m2": 1e14,
        "transition_widths_um": list(WIDTHS_UM),
        "current_relative_tolerance": CURRENT_TOLERANCE,
        "geometry_tolerance_um": GEOMETRY_TOLERANCE_UM,
        "relaxation_relative_tolerance": RELAXATION_TOLERANCE,
        "all_slsqp_runs_successful": all_success,
        "maximum_hard_current_relative_spread": max_spread,
        "maximum_improvement_over_latest_input_pool": max_improvement,
        "maximum_symmetry_aligned_geometry_change_um": max_geometry,
        "maximum_smooth_hard_current_relative_error": max_relaxation,
        "closure_beam_indices": sorted(closure_by_beam),
        "beams": beams,
    }
    output_path = HERE / "transition_width_final_250nm.json"
    plot_path = HERE / "transition_width_final_250nm.png"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    labels = [str(tuple(int(v) for v in row["beam_center_um"])) for row in beams]
    x = np.arange(len(beams))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    for width_um in WIDTHS_UM:
        axes[0].plot(
            x,
            [
                next(
                    width["best"]["hard_abs_current_A"]
                    for width in beam["widths"]
                    if width["transition_width_um"] == width_um
                ) * 1e9
                for beam in beams
            ],
            "o-",
            label=f"{width_um:g} um",
        )
    axes[0].set_xticks(x, labels, rotation=45)
    axes[0].set_ylabel("accepted hard |I| (nA)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].bar(x, [100 * b["hard_current_relative_spread"] for b in beams])
    axes[1].axhline(100 * CURRENT_TOLERANCE, color="black", linestyle="--")
    axes[1].set_xticks(x, labels, rotation=45)
    axes[1].set_ylabel("transition-width spread (%)")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(json.dumps({
        "status": status,
        "maximum_hard_current_relative_spread": max_spread,
        "maximum_improvement_over_latest_input_pool": max_improvement,
        "maximum_geometry_change_um": max_geometry,
        "maximum_relaxation_error": max_relaxation,
        "output": str(output_path),
    }, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
