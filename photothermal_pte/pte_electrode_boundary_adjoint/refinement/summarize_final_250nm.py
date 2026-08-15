from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(HERE))

from run_250nm_local_refinement import geometry_change_um  # noqa: E402
from tairte4_boundary_adjoint.baseline import ElectricalModel, load_config  # noqa: E402
from tairte4_boundary_adjoint.robin import DifferentiableContactModel  # noqa: E402


def main() -> int:
    plateau = json.loads(
        (PROJECT_ROOT / "optimization" / "search_plateau_results.json").read_text(
            encoding="utf-8"
        )
    )
    mesh = json.loads((HERE / "mesh_change_analysis.json").read_text(encoding="utf-8"))
    final = json.loads(
        (HERE / "transition_width_final_250nm.json").read_text(encoding="utf-8")
    )
    config = load_config(PROJECT_ROOT / "configs" / "per_beam_250nm.json")
    electrical = ElectricalModel(config)
    with np.load(HERE / "per_beam_250nm_fields.npz") as fields:
        temperatures = np.asarray(fields["temperature_nodes_K"])

    rows = []
    for beam_index in range(9):
        beam_500 = plateau["beams"][beam_index]["budgets"][-1]["best"]
        beam_mesh = mesh["beams"][beam_index]
        beam_250 = final["beams"][beam_index]
        best_500 = float(beam_500["hard_abs_current_A"])
        transferred_250 = abs(float(beam_mesh["currents"]["T250_E250_A"]))
        best_250 = float(beam_250["final_best"]["hard_abs_current_A"])
        model = DifferentiableContactModel(
            electrical,
            temperatures[beam_index],
            contact_conductance_S_m2=1e14,
            transition_m=0.75e-6,
            contact_discretization="nodal_lumped",
        )
        change = geometry_change_um(
            np.asarray(beam_500["canonical_scaled"]),
            np.asarray(beam_250["final_best"]["canonical_scaled"]),
            perimeter=model.perimeter,
            beam_center_um=np.asarray(beam_250["beam_center_um"]),
        )
        rows.append(
            {
                "beam_index": beam_index,
                "beam_center_um": beam_250["beam_center_um"],
                "best_500nm_hard_abs_current_A": best_500,
                "transferred_same_geometry_250nm_hard_abs_current_A": transferred_250,
                "final_best_250nm_hard_abs_current_A": best_250,
                "same_geometry_mesh_relative_change": (transferred_250 - best_500)
                / best_500,
                "optimization_uplift_at_250nm_over_transferred": (
                    best_250 - transferred_250
                )
                / transferred_250,
                "total_best_found_relative_change": (best_250 - best_500) / best_500,
                "symmetry_aligned_geometry_change": change,
                "final_250nm_canonical_scaled": beam_250["final_best"]["canonical_scaled"],
                "final_250nm_physical_um": (
                    np.asarray(beam_250["final_best"]["canonical_scaled"])
                    * model.perimeter.perimeter_m
                    * 1e6
                ).tolist(),
            }
        )

    result = {
        "status": "REFINED_BUT_NOT_MESH_CONVERGED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "The 0.25 um result is the current best-found refined design. The "
            "0.5-to-0.25 um same-geometry current change is too large to claim "
            "mesh convergence; a finer thermal mesh is required for that claim."
        ),
        "maximum_absolute_same_geometry_mesh_relative_change": max(
            abs(row["same_geometry_mesh_relative_change"]) for row in rows
        ),
        "maximum_optimization_uplift_at_250nm": max(
            row["optimization_uplift_at_250nm_over_transferred"] for row in rows
        ),
        "maximum_symmetry_aligned_geometry_parameter_change_um": max(
            row["symmetry_aligned_geometry_change"][
                "maximum_absolute_parameter_change_um"
            ]
            for row in rows
        ),
        "beams": rows,
    }
    output = HERE / "final_refinement_summary_250nm.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "maximum_absolute_same_geometry_mesh_relative_change": result[
                    "maximum_absolute_same_geometry_mesh_relative_change"
                ],
                "maximum_optimization_uplift_at_250nm": result[
                    "maximum_optimization_uplift_at_250nm"
                ],
                "maximum_geometry_change_um": result[
                    "maximum_symmetry_aligned_geometry_parameter_change_um"
                ],
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
