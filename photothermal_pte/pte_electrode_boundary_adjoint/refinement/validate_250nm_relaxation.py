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


G_SWEEP = np.asarray([1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18])
SELECTED_G = 1e14
TRANSITION_M = 0.75e-6
FD_STEP = 2e-6


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config_path = PROJECT_ROOT / "configs" / "per_beam_250nm.json"
    fields_path = HERE / "per_beam_250nm_fields.npz"
    plateau_path = PROJECT_ROOT / "optimization" / "search_plateau_results.json"
    config = load_config(config_path)
    electrical = ElectricalModel(config)
    with np.load(fields_path) as fields:
        centers = np.asarray(fields["beam_centers_m"])
        index = int(np.argmin(np.linalg.norm(centers, axis=1)))
        temperature = np.asarray(fields["temperature_nodes_K"][index])
    plateau = json.loads(plateau_path.read_text(encoding="utf-8"))
    physical_um = plateau["beams"][index]["budgets"][-1]["best"]["physical_um"]
    perimeter_m = 2.0 * (
        config["geometry"]["flake_width_um"]
        + config["geometry"]["flake_height_um"]
    ) * 1e-6
    x = np.asarray(physical_um, dtype=float) * 1e-6 / perimeter_m
    p = ScaledDesign.from_array(x).to_physical(perimeter_m)

    def build(g: float) -> DifferentiableContactModel:
        return DifferentiableContactModel(
            electrical,
            temperature,
            contact_conductance_S_m2=g,
            transition_m=TRANSITION_M,
            contact_discretization="nodal_lumped",
        )

    hard = build(SELECTED_G).hard_evaluate(p)
    hard_psi = hard.weighting_potential
    rows = []
    for g in G_SWEEP:
        result = build(float(g)).evaluate(p)
        rows.append(
            {
                "g_S_m2": float(g),
                "current_A": result.current_A,
                "current_relative_error": abs(result.current_A - hard.current_A)
                / abs(hard.current_A),
                "psi_relative_l2_error": float(
                    np.linalg.norm(result.weighting_potential - hard_psi)
                    / np.linalg.norm(hard_psi)
                ),
                "scaled_gradient_l2_A": float(
                    np.linalg.norm(perimeter_m * result.current_gradient_A_per_m)
                ),
                "state_residual_relative": result.state_residual_relative,
                "adjoint_residual_relative": result.adjoint_residual_relative,
            }
        )

    selected = rows[int(np.flatnonzero(G_SWEEP == SELECTED_G)[0])]
    model = build(SELECTED_G)
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
    passed = bool(
        selected["current_relative_error"] <= 0.01
        and selected["psi_relative_l2_error"] <= 0.05
        and rows[-1]["current_relative_error"] <= 1e-5
        and rows[-1]["psi_relative_l2_error"] <= 1e-5
        and np.max(fd_errors) <= 1e-4
    )
    summary = {
        "status": "PASS" if passed else "FAIL",
        "mesh_step_um": electrical.step_m * 1e6,
        "beam_index": index,
        "beam_center_um": (centers[index] * 1e6).tolist(),
        "geometry_from_500nm_um": physical_um,
        "hard_current_A": hard.current_A,
        "selected_g_S_m2": SELECTED_G,
        "selected_result": selected,
        "g_1e12_result": rows[int(np.flatnonzero(G_SWEEP == 1e12)[0])],
        "hard_limit_result": rows[-1],
        "gradient_fd_step_scaled": FD_STEP,
        "adjoint_gradient_A_per_scaled_variable": adjoint.tolist(),
        "central_fd_gradient_A_per_scaled_variable": fd.tolist(),
        "component_relative_errors": fd_errors.tolist(),
        "maximum_component_relative_error": float(np.max(fd_errors)),
        "config_sha256": digest(config_path),
        "fields_sha256": digest(fields_path),
        "source_500nm_plateau_sha256": digest(plateau_path),
        "evaluations": rows,
    }
    output_path = HERE / "relaxation_250nm.json"
    plot_path = HERE / "relaxation_250nm.png"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].loglog(G_SWEEP, [r["current_relative_error"] for r in rows], "o-")
    axes[0].axhline(0.01, color="black", linestyle="--")
    axes[0].set_xlabel("g (S/m2)")
    axes[0].set_ylabel("relative current error")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].loglog(G_SWEEP, [r["psi_relative_l2_error"] for r in rows], "o-")
    axes[1].axhline(0.05, color="black", linestyle="--")
    axes[1].set_xlabel("g (S/m2)")
    axes[1].set_ylabel("relative L2 psi error")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
