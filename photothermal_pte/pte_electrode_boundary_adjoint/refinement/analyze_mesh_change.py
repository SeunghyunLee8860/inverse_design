from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(HERE))

from run_250nm_local_refinement import geometry_change_um  # noqa: E402
from tairte4_boundary_adjoint.baseline import (  # noqa: E402
    BASELINE_ROOT,
    ElectricalModel,
    load_config,
)
from tairte4_boundary_adjoint.robin import DifferentiableContactModel  # noqa: E402
from tairte4_boundary_adjoint.scaled import ScaledDesign  # noqa: E402


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def hard_current(electrical, temperature, physical) -> float:
    model = DifferentiableContactModel(
        electrical,
        temperature,
        contact_conductance_S_m2=1e13,
        transition_m=0.75e-6,
        contact_discretization="nodal_lumped",
    )
    return model.hard_evaluate(physical).current_A


def main() -> int:
    plateau_path = PROJECT_ROOT / "optimization" / "search_plateau_results.json"
    local_path = HERE / "local_refinement_250nm.json"
    fields_250_path = HERE / "per_beam_250nm_fields.npz"
    fields_500_path = BASELINE_ROOT / "results" / "per_beam_500nm_final" / "per_beam_fields.npz"
    plateau = json.loads(plateau_path.read_text(encoding="utf-8"))
    local = json.loads(local_path.read_text(encoding="utf-8"))
    with np.load(fields_500_path) as data:
        temperature_500 = np.asarray(data["temperature_nodes_K"])
    with np.load(fields_250_path) as data:
        temperature_250 = np.asarray(data["temperature_nodes_K"])
        centers = np.asarray(data["beam_centers_m"])
    electrical_500 = ElectricalModel(
        load_config(BASELINE_ROOT / "configs" / "per_beam_500nm.json")
    )
    electrical_250 = ElectricalModel(
        load_config(PROJECT_ROOT / "configs" / "per_beam_250nm.json")
    )
    xx, yy = np.meshgrid(
        electrical_250.mesh.x_m, electrical_250.mesh.y_m, indexing="ij"
    )
    query_250 = np.column_stack((xx.ravel(), yy.ravel()))

    rows = []
    for beam_index, beam in enumerate(local["beams"]):
        source = plateau["beams"][beam_index]["budgets"][-1]["best"]
        x_500 = np.asarray(source["canonical_scaled"])
        p_500 = ScaledDesign.from_array(x_500).to_physical(96e-6)
        t250_on_500 = temperature_250[beam_index][::2, ::2]
        interpolator = RegularGridInterpolator(
            (electrical_500.mesh.x_m, electrical_500.mesh.y_m),
            temperature_500[beam_index],
            method="linear",
        )
        t500_on_250 = interpolator(query_250).reshape(electrical_250.mesh.shape)
        currents = {
            "T500_E500_A": hard_current(
                electrical_500, temperature_500[beam_index], p_500
            ),
            "T250_downsampled_E500_A": hard_current(
                electrical_500, t250_on_500, p_500
            ),
            "T500_interpolated_E250_A": hard_current(
                electrical_250, t500_on_250, p_500
            ),
            "T250_E250_A": hard_current(
                electrical_250, temperature_250[beam_index], p_500
            ),
        }
        denominator = abs(currents["T500_E500_A"])
        best_250_x = np.asarray(beam["best_250nm"]["canonical_scaled"])
        geometry = geometry_change_um(
            x_500,
            best_250_x,
            perimeter=DifferentiableContactModel(
                electrical_250,
                temperature_250[beam_index],
                contact_conductance_S_m2=1e13,
                transition_m=0.75e-6,
            ).perimeter,
            beam_center_um=centers[beam_index] * 1e6,
        )
        rows.append(
            {
                "beam_index": beam_index,
                "beam_center_um": (centers[beam_index] * 1e6).tolist(),
                "currents": currents,
                "relative_thermal_change_on_E500": (
                    abs(currents["T250_downsampled_E500_A"]) / denominator - 1.0
                ),
                "relative_electrical_change_with_interpolated_T500": (
                    abs(currents["T500_interpolated_E250_A"]) / denominator - 1.0
                ),
                "relative_combined_same_geometry_change": (
                    abs(currents["T250_E250_A"]) / denominator - 1.0
                ),
                "symmetry_aligned_geometry_change": geometry,
                "local_best_250nm_hard_abs_current_A": beam["best_250nm"][
                    "hard_abs_current_A"
                ],
            }
        )
    summary = {
        "status": "COMPLETED",
        "scope": "decompose 0.5-to-0.25 current change into thermal and electrical discretization effects",
        "temperature_transfer_250_to_500": "exact nested-node downsampling [::2,::2]",
        "temperature_transfer_500_to_250": "bilinear interpolation",
        "geometry_alignment": "periodic centers + terminal swap + only beam-preserving x/y reflections",
        "max_absolute_thermal_change_on_E500": max(
            abs(r["relative_thermal_change_on_E500"]) for r in rows
        ),
        "max_absolute_electrical_change_with_interpolated_T500": max(
            abs(r["relative_electrical_change_with_interpolated_T500"]) for r in rows
        ),
        "max_absolute_combined_same_geometry_change": max(
            abs(r["relative_combined_same_geometry_change"]) for r in rows
        ),
        "max_symmetry_aligned_geometry_parameter_change_um": max(
            r["symmetry_aligned_geometry_change"][
                "maximum_absolute_parameter_change_um"
            ]
            for r in rows
        ),
        "plateau_sha256": digest(plateau_path),
        "local_refinement_sha256": digest(local_path),
        "fields_500_sha256": digest(fields_500_path),
        "fields_250_sha256": digest(fields_250_path),
        "beams": rows,
    }
    output_path = HERE / "mesh_change_analysis.json"
    plot_path = HERE / "mesh_change_analysis.png"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    labels = [str(tuple(int(v) for v in r["beam_center_um"])) for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    axes[0].bar(x - 0.25, [100*r["relative_thermal_change_on_E500"] for r in rows], 0.25, label="thermal")
    axes[0].bar(x, [100*r["relative_electrical_change_with_interpolated_T500"] for r in rows], 0.25, label="electrical")
    axes[0].bar(x + 0.25, [100*r["relative_combined_same_geometry_change"] for r in rows], 0.25, label="combined")
    axes[0].set_xticks(x, labels, rotation=45)
    axes[0].set_ylabel("same-geometry |I| change (%)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    axes[1].bar(x, [r["symmetry_aligned_geometry_change"]["maximum_absolute_parameter_change_um"] for r in rows])
    axes[1].set_xticks(x, labels, rotation=45)
    axes[1].set_ylabel("symmetry-aligned max parameter change (um)")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
