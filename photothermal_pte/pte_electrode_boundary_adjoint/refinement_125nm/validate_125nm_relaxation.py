from __future__ import annotations

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

from tairte4_boundary_adjoint.baseline import ElectricalModel, load_config  # noqa: E402
from tairte4_boundary_adjoint.robin import DifferentiableContactModel  # noqa: E402
from tairte4_boundary_adjoint.scaled import ScaledDesign  # noqa: E402


G_SWEEP = np.asarray([1e13, 1e14, 1e15, 1e16, 1e17, 1e18])
REPRESENTATIVE_INDICES = (0, 1, 3, 4)
TRANSITION_M = 0.50e-6
FD_STEP = 2e-6
CURRENT_TOLERANCE = 5e-3
PSI_TOLERANCE = 1e-2


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config_path = PROJECT_ROOT / "configs" / "per_beam_125nm.json"
    fields_path = HERE / "per_beam_125nm_fields.npz"
    final_250_path = PROJECT_ROOT / "refinement" / "transition_width_final_250nm.json"
    config = load_config(config_path)
    electrical = ElectricalModel(config)
    with np.load(fields_path) as fields:
        centers = np.asarray(fields["beam_centers_m"])
        temperatures = np.asarray(fields["temperature_nodes_K"])
    final_250 = json.loads(final_250_path.read_text(encoding="utf-8"))
    perimeter_m = 2.0 * (
        config["geometry"]["flake_width_um"]
        + config["geometry"]["flake_height_um"]
    ) * 1e-6

    beam_rows = []
    for beam_index in REPRESENTATIVE_INDICES:
        x = np.asarray(
            final_250["beams"][beam_index]["final_best"]["canonical_scaled"],
            dtype=float,
        )
        p = ScaledDesign.from_array(x).canonical().to_physical(perimeter_m)

        def build(g: float) -> DifferentiableContactModel:
            return DifferentiableContactModel(
                electrical,
                temperatures[beam_index],
                contact_conductance_S_m2=g,
                transition_m=TRANSITION_M,
                contact_discretization="nodal_lumped",
            )

        hard = build(float(G_SWEEP[-1])).hard_evaluate(p)
        evaluations = []
        for g in G_SWEEP:
            smooth = build(float(g)).evaluate(p)
            evaluations.append(
                {
                    "g_S_m2": float(g),
                    "current_A": smooth.current_A,
                    "hard_current_A": hard.current_A,
                    "current_relative_error": abs(smooth.current_A - hard.current_A)
                    / abs(hard.current_A),
                    "psi_relative_l2_error": float(
                        np.linalg.norm(
                            smooth.weighting_potential - hard.weighting_potential
                        ) / np.linalg.norm(hard.weighting_potential)
                    ),
                    "scaled_gradient_l2_A": float(
                        np.linalg.norm(perimeter_m * smooth.current_gradient_A_per_m)
                    ),
                    "state_residual_relative": smooth.state_residual_relative,
                    "adjoint_residual_relative": smooth.adjoint_residual_relative,
                }
            )
        beam_rows.append(
            {
                "beam_index": beam_index,
                "beam_center_um": (centers[beam_index] * 1e6).tolist(),
                "canonical_scaled": x.tolist(),
                "evaluations": evaluations,
            }
        )

    maxima_by_g = []
    for g_index, g in enumerate(G_SWEEP):
        rows = [beam["evaluations"][g_index] for beam in beam_rows]
        maxima_by_g.append(
            {
                "g_S_m2": float(g),
                "maximum_current_relative_error": max(
                    row["current_relative_error"] for row in rows
                ),
                "maximum_psi_relative_l2_error": max(
                    row["psi_relative_l2_error"] for row in rows
                ),
                "minimum_scaled_gradient_l2_A": min(
                    row["scaled_gradient_l2_A"] for row in rows
                ),
                "maximum_state_residual_relative": max(
                    row["state_residual_relative"] for row in rows
                ),
                "maximum_adjoint_residual_relative": max(
                    row["adjoint_residual_relative"] for row in rows
                ),
            }
        )
    eligible = [
        row for row in maxima_by_g
        if row["maximum_current_relative_error"] <= CURRENT_TOLERANCE
        and row["maximum_psi_relative_l2_error"] <= PSI_TOLERANCE
        and row["minimum_scaled_gradient_l2_A"] > 0.0
    ]
    if not eligible:
        raise RuntimeError("no finite g passes the representative relaxation criteria")
    selected_g = eligible[0]["g_S_m2"]

    fd_beam_index = 0
    x = np.asarray(
        final_250["beams"][fd_beam_index]["final_best"]["canonical_scaled"],
        dtype=float,
    )
    model = DifferentiableContactModel(
        electrical,
        temperatures[fd_beam_index],
        contact_conductance_S_m2=selected_g,
        transition_m=TRANSITION_M,
        contact_discretization="nodal_lumped",
    )
    p = ScaledDesign.from_array(x).canonical().to_physical(perimeter_m)
    base = model.evaluate(p)
    adjoint = perimeter_m * base.current_gradient_A_per_m
    fd = np.zeros(4)
    for component in range(4):
        plus = x.copy()
        minus = x.copy()
        plus[component] += FD_STEP
        minus[component] -= FD_STEP
        i_plus = model.evaluate(
            ScaledDesign.from_array(plus).to_physical(perimeter_m)
        ).current_A
        i_minus = model.evaluate(
            ScaledDesign.from_array(minus).to_physical(perimeter_m)
        ).current_A
        fd[component] = (i_plus - i_minus) / (2.0 * FD_STEP)
    denominator = np.maximum.reduce(
        [np.abs(adjoint), np.abs(fd), np.full(4, np.linalg.norm(adjoint) * 1e-12)]
    )
    fd_errors = np.abs(adjoint - fd) / denominator
    tail = maxima_by_g[-1]
    selected = next(row for row in maxima_by_g if row["g_S_m2"] == selected_g)
    passed = bool(
        selected["maximum_current_relative_error"] <= CURRENT_TOLERANCE
        and selected["maximum_psi_relative_l2_error"] <= PSI_TOLERANCE
        and tail["maximum_current_relative_error"] <= 1e-4
        and tail["maximum_psi_relative_l2_error"] <= PSI_TOLERANCE
        and np.max(fd_errors) <= 1e-4
    )
    summary = {
        "status": "PASS" if passed else "FAIL",
        "mesh_step_um": 0.125,
        "representative_indices": list(REPRESENTATIVE_INDICES),
        "transition_width_um": TRANSITION_M * 1e6,
        "selected_g_S_m2": selected_g,
        "current_relative_tolerance": CURRENT_TOLERANCE,
        "psi_relative_l2_tolerance": PSI_TOLERANCE,
        "selected_maxima": selected,
        "hard_limit_maxima": tail,
        "hard_limit_note": (
            "At this mesh, g->infinity current converges below 1e-4 relative "
            "error while psi plateaus below the declared 1% tolerance because "
            "the nominal hard contact includes endpoint nodes at equality."
        ),
        "gradient_fd_beam_index": fd_beam_index,
        "gradient_fd_step_scaled": FD_STEP,
        "adjoint_gradient_A_per_scaled_variable": adjoint.tolist(),
        "central_fd_gradient_A_per_scaled_variable": fd.tolist(),
        "component_relative_errors": fd_errors.tolist(),
        "maximum_component_relative_error": float(np.max(fd_errors)),
        "config_sha256": digest(config_path),
        "fields_sha256": digest(fields_path),
        "source_250nm_sha256": digest(final_250_path),
        "maxima_by_g": maxima_by_g,
        "beams": beam_rows,
    }
    output = HERE / "relaxation_125nm.json"
    plot = HERE / "relaxation_125nm.png"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].loglog(
        G_SWEEP,
        [row["maximum_current_relative_error"] for row in maxima_by_g],
        "o-",
    )
    axes[0].axhline(CURRENT_TOLERANCE, color="black", linestyle="--")
    axes[0].set_xlabel("g (S/m2)")
    axes[0].set_ylabel("max relative current error")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].loglog(
        G_SWEEP,
        [row["maximum_psi_relative_l2_error"] for row in maxima_by_g],
        "o-",
    )
    axes[1].axhline(PSI_TOLERANCE, color="black", linestyle="--")
    axes[1].set_xlabel("g (S/m2)")
    axes[1].set_ylabel("max relative L2 psi error")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.savefig(plot, dpi=180)
    plt.close(fig)
    print(json.dumps({
        "status": summary["status"],
        "selected_g_S_m2": selected_g,
        "selected_maxima": selected,
        "maximum_component_relative_error": float(np.max(fd_errors)),
        "output": str(output),
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
