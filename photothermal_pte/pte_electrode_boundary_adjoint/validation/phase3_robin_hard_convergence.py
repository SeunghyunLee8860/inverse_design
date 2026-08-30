from __future__ import annotations

import csv
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

from tairte4_boundary_adjoint.baseline import (  # noqa: E402
    BASELINE_ROOT,
    ElectricalModel,
    load_config,
)
from tairte4_boundary_adjoint.perimeter import PerimeterParameters  # noqa: E402
from tairte4_boundary_adjoint.robin import DifferentiableContactModel  # noqa: E402


DESIGN_PHYSICAL_UM = np.asarray([6.1, 7.3, 58.4, 10.7], dtype=float)
G_SWEEP_S_M2 = np.asarray(
    [1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 3e16, 1e17, 3e17, 1e18],
    dtype=float,
)
SELECTED_OPTIMIZATION_G_S_M2 = 1.0e12
TRANSITION_WIDTH_M = 0.75e-6
BOUNDARY_QUADRATURE_ORDER = 5
CONTACT_DISCRETIZATION = "nodal_lumped"

# Separate criteria for proving the asymptotic connection and selecting a
# finite, still-differentiable optimization relaxation.
HARD_LIMIT_CURRENT_REL_TOL = 1.0e-4
HARD_LIMIT_PSI_L2_REL_TOL = 1.0e-4
HARD_LIMIT_PSI_LINF_TOL = 1.0e-4
RELAXATION_CURRENT_REL_TOL = 1.0e-2
RELAXATION_PSI_L2_REL_TOL = 5.0e-2
RESIDUAL_TOL = 1.0e-10


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    config_path = BASELINE_ROOT / "configs" / "per_beam_500nm.json"
    fields_path = (
        BASELINE_ROOT / "results" / "per_beam_500nm_final" / "per_beam_fields.npz"
    )
    config = load_config(config_path)
    with np.load(fields_path) as fields:
        centers_m = np.asarray(fields["beam_centers_m"], dtype=float)
        beam_index = int(np.argmin(np.linalg.norm(centers_m, axis=1)))
        temperature_nodes_K = np.asarray(
            fields["temperature_nodes_K"][beam_index], dtype=float
        )

    electrical = ElectricalModel(config)
    parameters = PerimeterParameters.from_array(DESIGN_PHYSICAL_UM * 1e-6)

    def build(g_value: float, mode: str = CONTACT_DISCRETIZATION):
        return DifferentiableContactModel(
            electrical,
            temperature_nodes_K,
            contact_conductance_S_m2=g_value,
            transition_m=TRANSITION_WIDTH_M,
            quadrature_order=BOUNDARY_QUADRATURE_ORDER,
            contact_discretization=mode,
        )

    reference_model = build(SELECTED_OPTIMIZATION_G_S_M2)
    hard = reference_model.hard_evaluate(parameters)
    hard_psi = hard.weighting_potential.reshape(-1)
    hard_norm = np.linalg.norm(hard_psi)
    current_denominator = max(abs(hard.current_A), np.finfo(float).tiny)
    perimeter_m = reference_model.perimeter.perimeter_m

    rows: list[dict[str, float]] = []
    for g_value in G_SWEEP_S_M2:
        result = build(float(g_value)).evaluate(parameters)
        psi = result.weighting_potential.reshape(-1)
        delta = psi - hard_psi
        gradient_scaled = perimeter_m * result.current_gradient_A_per_m
        row = {
            "g_S_m2": float(g_value),
            "current_A": result.current_A,
            "current_relative_error": abs(result.current_A - hard.current_A)
            / current_denominator,
            "psi_relative_l2_error": float(np.linalg.norm(delta) / hard_norm),
            "psi_absolute_linf_error": float(np.max(np.abs(delta))),
            "state_residual_relative": result.state_residual_relative,
            "adjoint_residual_relative": result.adjoint_residual_relative,
            "psi_min": float(np.min(psi)),
            "psi_max": float(np.max(psi)),
            "conductance_S": result.terminal_conductance_S,
            "scaled_gradient_l2_A": float(np.linalg.norm(gradient_scaled)),
            "scaled_gradient_min_to_max_abs_ratio": float(
                np.min(np.abs(gradient_scaled))
                / max(np.max(np.abs(gradient_scaled)), np.finfo(float).tiny)
            ),
        }
        rows.append(row)

    selected_index = int(
        np.flatnonzero(G_SWEEP_S_M2 == SELECTED_OPTIMIZATION_G_S_M2)[0]
    )
    selected = rows[selected_index]
    tail = rows[-1]
    hard_limit_pass = bool(
        tail["current_relative_error"] <= HARD_LIMIT_CURRENT_REL_TOL
        and tail["psi_relative_l2_error"] <= HARD_LIMIT_PSI_L2_REL_TOL
        and tail["psi_absolute_linf_error"] <= HARD_LIMIT_PSI_LINF_TOL
        and tail["state_residual_relative"] <= RESIDUAL_TOL
    )
    selected_relaxation_pass = bool(
        selected["current_relative_error"] <= RELAXATION_CURRENT_REL_TOL
        and selected["psi_relative_l2_error"] <= RELAXATION_PSI_L2_REL_TOL
        and selected["state_residual_relative"] <= RESIDUAL_TOL
    )

    # Preserve the failure that motivated nodal mass lumping.  Consistent edge
    # integration has a different fixed-mesh g->infinity node constraint.
    old_mode_rows = []
    for g_value in (1e12, 1e14, 1e16, 1e18):
        result = build(g_value, mode="consistent_edge").evaluate(parameters)
        psi = result.weighting_potential.reshape(-1)
        old_mode_rows.append(
            {
                "g_S_m2": g_value,
                "current_relative_error": abs(result.current_A - hard.current_A)
                / current_denominator,
                "psi_relative_l2_error": float(
                    np.linalg.norm(psi - hard_psi) / hard_norm
                ),
            }
        )

    passed = bool(hard_limit_pass and selected_relaxation_pass)
    summary = {
        "status": "PASS" if passed else "FAIL",
        "scope": "Robin-to-hard convergence only; no optimizer was called",
        "baseline_root": str(BASELINE_ROOT),
        "config_sha256": file_sha256(config_path),
        "temperature_fields_sha256": file_sha256(fields_path),
        "beam_index": beam_index,
        "beam_center_um": (centers_m[beam_index] * 1e6).tolist(),
        "mesh_step_um": electrical.step_m * 1e6,
        "design_physical_um": DESIGN_PHYSICAL_UM.tolist(),
        "transition_width_um": TRANSITION_WIDTH_M * 1e6,
        "boundary_quadrature_order": BOUNDARY_QUADRATURE_ORDER,
        "contact_discretization": CONTACT_DISCRETIZATION,
        "hard_current_A": hard.current_A,
        "hard_conductance_S": hard.terminal_conductance_S,
        "hard_contact_0_node_count": int(hard.contact_0_nodes.size),
        "hard_contact_1_node_count": int(hard.contact_1_nodes.size),
        "selected_optimization_g_S_m2": SELECTED_OPTIMIZATION_G_S_M2,
        "selected_relaxation": selected,
        "selected_relaxation_tolerances": {
            "current_relative_error": RELAXATION_CURRENT_REL_TOL,
            "psi_relative_l2_error": RELAXATION_PSI_L2_REL_TOL,
            "state_residual_relative": RESIDUAL_TOL,
        },
        "selected_relaxation_pass": selected_relaxation_pass,
        "hard_limit_g_S_m2": tail["g_S_m2"],
        "hard_limit_result": tail,
        "hard_limit_tolerances": {
            "current_relative_error": HARD_LIMIT_CURRENT_REL_TOL,
            "psi_relative_l2_error": HARD_LIMIT_PSI_L2_REL_TOL,
            "psi_absolute_linf_error": HARD_LIMIT_PSI_LINF_TOL,
            "state_residual_relative": RESIDUAL_TOL,
        },
        "hard_limit_pass": hard_limit_pass,
        "consistent_edge_mismatch_diagnostic": old_mode_rows,
        "evaluations": rows,
    }
    json_path = HERE / "phase3_robin_hard_convergence.json"
    csv_path = HERE / "phase3_robin_hard_convergence.csv"
    plot_path = HERE / "phase3_robin_hard_convergence.png"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    g_values = np.asarray([row["g_S_m2"] for row in rows])
    axes[0].loglog(
        g_values,
        [row["current_relative_error"] for row in rows],
        "o-",
        label="nodal-lumped Robin",
    )
    axes[0].loglog(
        [row["g_S_m2"] for row in old_mode_rows],
        [row["current_relative_error"] for row in old_mode_rows],
        "x--",
        label="old consistent-edge diagnostic",
    )
    axes[0].axhline(HARD_LIMIT_CURRENT_REL_TOL, color="black", linestyle=":")
    axes[0].set_xlabel("contact conductance g (S/m2)")
    axes[0].set_ylabel("relative current error")
    axes[0].set_title("I_Robin versus I_hard")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[1].loglog(
        g_values,
        [row["psi_relative_l2_error"] for row in rows],
        "o-",
        label="relative L2",
    )
    axes[1].loglog(
        g_values,
        [row["psi_absolute_linf_error"] for row in rows],
        "s-",
        label="absolute Linf",
    )
    axes[1].axhline(HARD_LIMIT_PSI_L2_REL_TOL, color="black", linestyle=":")
    axes[1].set_xlabel("contact conductance g (S/m2)")
    axes[1].set_ylabel("weighting-potential error")
    axes[1].set_title("psi_Robin versus psi_hard")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
