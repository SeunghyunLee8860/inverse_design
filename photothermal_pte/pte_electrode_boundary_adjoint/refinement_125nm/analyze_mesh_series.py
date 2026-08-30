from __future__ import annotations

from datetime import datetime, timezone
import json
from math import log2
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def main() -> int:
    final_250 = json.loads(
        (PROJECT_ROOT / "refinement" / "transition_width_final_250nm.json").read_text(
            encoding="utf-8"
        )
    )
    final_125 = json.loads((HERE / "final_125nm.json").read_text(encoding="utf-8"))
    pilot_62 = json.loads(
        (HERE / "targeted_62p5nm_pilot.json").read_text(encoding="utf-8")
    )
    pilot_by_beam = {int(row["beam_index"]): row for row in pilot_62["beams"]}
    rows = []
    for beam_index in pilot_62["target_indices"]:
        i250 = float(final_250["beams"][beam_index]["final_best"]["hard_abs_current_A"])
        i125 = float(final_125["beams"][beam_index]["best_hard_abs_current_125nm_A"])
        i62 = float(pilot_by_beam[beam_index]["same_geometry_hard_current_62p5nm_A"])
        coarse_difference = abs(i125 - i250)
        fine_difference = abs(i62 - i125)
        observed_order = log2(coarse_difference / fine_difference)
        ratio = 2.0**observed_order
        extrapolated = i62 + (i62 - i125) / (ratio - 1.0)
        predicted_31 = i62 + (i62 - i125) / ratio
        rows.append(
            {
                "beam_index": beam_index,
                "beam_center_um": final_125["beams"][beam_index]["beam_center_um"],
                "hard_abs_current_250nm_A": i250,
                "hard_abs_current_125nm_A": i125,
                "hard_abs_current_62p5nm_A": i62,
                "observed_order_from_successive_differences": observed_order,
                "richardson_extrapolated_hard_abs_current_A": extrapolated,
                "estimated_remaining_relative_error_at_62p5nm": abs(extrapolated - i62)
                / abs(extrapolated),
                "predicted_hard_abs_current_31p25nm_A": predicted_31,
                "predicted_relative_change_62p5_to_31p25": abs(predicted_31 - i62)
                / abs(i62),
            }
        )
    orders = [row["observed_order_from_successive_differences"] for row in rows]
    report = {
        "status": "DIRECT_SUCCESSIVE_MESH_NOT_CONVERGED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "direct_mesh_relative_tolerance": 0.01,
        "maximum_direct_change_250_to_125": final_125[
            "maximum_absolute_relative_mesh_change_250_to_125"
        ],
        "maximum_targeted_direct_change_125_to_62p5": pilot_62[
            "maximum_absolute_relative_current_change"
        ],
        "observed_order_range": [min(orders), max(orders)],
        "interpretation": (
            "Both nonconverged beam types show a consistent approximately "
            "1.4-order monotone trend. Richardson estimates are diagnostic, "
            "not a substitute for the direct 31.25 nm check."
        ),
        "rows": rows,
    }
    output = HERE / "mesh_series_analysis.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "maximum_targeted_direct_change_125_to_62p5": report[
            "maximum_targeted_direct_change_125_to_62p5"
        ],
        "observed_order_range": report["observed_order_range"],
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
