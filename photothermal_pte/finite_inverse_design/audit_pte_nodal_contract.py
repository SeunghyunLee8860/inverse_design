#!/usr/bin/env python3
"""Audit the approved uniform-45-degree PTE and 81x81 nodal contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .contract import (
    DESIGN_BOUNDS_M,
    DESIGN_DX_M,
    DESIGN_DY_M,
    DESIGN_NX,
    DESIGN_NY,
    DESIGN_NZ,
    SEEBECK_A_V_K,
    SEEBECK_B_V_K,
    SIGMA_A_S_M,
    SIGMA_B_S_M,
    WEIGHTING_FIELD_X_M_INV,
    WEIGHTING_FIELD_Y_M_INV,
    design_extrusion_nodes_m,
    design_nodal_coordinates_m,
)
from .uniform_weighting_pte import build_uniform_45deg_weighting_pte
from ..validation.photothermal_stage1.anisotropic_heat_fvm import (
    assemble_steady_diagonal_kappa,
)


STATUS = "AUDITED_PTE_DISCRETE_OPERATOR_AND_81X81_NODAL_CONTRACT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    nodal_x, nodal_y = design_nodal_coordinates_m()
    extrusion_z = design_extrusion_nodes_m()
    x = np.linspace(-2.0e-6, 2.0e-6, 18)
    y = np.linspace(-2.0e-6, 2.0e-6, 16)
    z = np.linspace(-100.0e-9, 0.0, 5)
    shape = (x.size - 1, y.size - 1, z.size - 1)
    kappa = np.broadcast_to([14.4, 3.8, 1.0], (*shape, 3)).copy()
    system = assemble_steady_diagonal_kappa(
        x_edges_m=x,
        y_edges_m=y,
        z_edges_m=z,
        kappa_W_mK=kappa,
        dirichlet_temperature_K={"z_min": 0.0},
    )
    pte = build_uniform_45deg_weighting_pte(
        x_edges_m=x,
        y_edges_m=y,
        z_edges_m=z,
        active_mask=system.active_mask,
        active_ids=system.active_ids,
        flake_mask=system.active_mask,
    )
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    a, b = 2.3e6, -1.7e6
    affine = (
        4.0
        + a * xc[:, None, None]
        + b * yc[None, :, None]
        + np.zeros((1, 1, z.size - 1))
    )[system.active_mask]
    volume = (x[-1] - x[0]) * (y[-1] - y[0]) * (z[-1] - z[0])
    analytic = -volume * (
        SIGMA_A_S_M
        * SEEBECK_A_V_K
        * a
        * WEIGHTING_FIELD_X_M_INV
        + SIGMA_B_S_M
        * SEEBECK_B_V_K
        * b
        * WEIGHTING_FIELD_Y_M_INV
    )
    affine_forward = pte.evaluate_active(affine)
    affine_transpose = float(np.dot(pte.temperature_source_A_K, affine))
    rng = np.random.default_rng(2026072705)
    temperature = rng.normal(size=system.matrix_W_K.shape[0])
    direction = rng.normal(size=temperature.size)
    random_forward = pte.evaluate_active(temperature)
    random_transpose = float(
        np.dot(pte.temperature_source_A_K, temperature)
    )
    step = 1.0e-5
    finite_difference = (
        pte.evaluate_active(temperature + step * direction)
        - pte.evaluate_active(temperature - step * direction)
    ) / (2.0 * step)
    analytic_directional = float(
        np.dot(pte.temperature_source_A_K, direction)
    )

    def relative(first: float, second: float) -> float:
        return abs(first - second) / max(
            abs(first), abs(second), np.finfo(float).tiny
        )

    metrics = {
        "affine_vs_analytic_relative_error": relative(
            affine_forward, analytic
        ),
        "affine_forward_vs_source_relative_error": relative(
            affine_forward, affine_transpose
        ),
        "random_forward_vs_source_relative_error": relative(
            random_forward, random_transpose
        ),
        "temperature_source_FD_relative_error": relative(
            analytic_directional, finite_difference
        ),
    }
    passed = bool(
        max(metrics.values()) < 1.0e-10
        and pte.base.periodic_axes == ()
        and nodal_x.size == 81
        and nodal_y.size == 81
        and extrusion_z.size == 13
    )
    result = {
        "status": STATUS if passed else "FAILED_PTE_OR_NODAL_CONTRACT_AUDIT",
        "passed": passed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pte_objective": (
            "-integral[sigma_a*S_a*dTdx*dpsidx + "
            "sigma_b*S_b*dTdy*dpsidy]dV"
        ),
        "weighting_field_m_inv": {
            "x": WEIGHTING_FIELD_X_M_INV,
            "y": WEIGHTING_FIELD_Y_M_INV,
        },
        "periodic_axes": list(pte.base.periodic_axes),
        "forward_and_adjoint_use_identical_derivative_matrices": True,
        "metrics": metrics,
        "nodal_density": {
            "shape": [DESIGN_NX, DESIGN_NY],
            "x_bounds_m": [float(nodal_x[0]), float(nodal_x[-1])],
            "y_bounds_m": [float(nodal_y[0]), float(nodal_y[-1])],
            "dx_m": DESIGN_DX_M,
            "dy_m": DESIGN_DY_M,
            "coordinate_role": (
                "nodal physical density on exact finite support; not "
                "81 finite-width pixels"
            ),
            "opposite_edge_wrap": False,
        },
        "z_extrusion": {
            "node_count": DESIGN_NZ,
            "bounds_m": [
                float(extrusion_z[0]),
                float(extrusion_z[-1]),
            ],
            "two_dimensional_density_repeated_in_z": True,
        },
        "design_bounds_m": DESIGN_BOUNDS_M,
        "surrogate_warning": (
            "uniform-45-degree weighting approximation; not a solved "
            "finite-contact terminal weighting potential"
        ),
        "optimization_run": False,
    }
    (output / "pte_nodal_contract_audit_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    report = f"""# PTE discrete operator and 81x81 nodal contract audit

Status: `{result["status"]}`

- physical-density coordinates: 81x81 nodes on exact
  `[-1,1] um x [-1,1] um` support;
- spacing: `25 nm`; opposite-edge wrap: absent;
- design is a 2D density extruded from `z=0` to `600 nm`;
- PTE weighting:
  `dpsi/dx=dpsi/dy=1/(4 um)`;
- periodic derivatives: absent;
- forward functional and `c_T` use the same sparse `D_x,D_y` matrices.

Errors:

- affine analytic control:
  `{metrics["affine_vs_analytic_relative_error"]:.8e}`;
- affine forward/source identity:
  `{metrics["affine_forward_vs_source_relative_error"]:.8e}`;
- random forward/source identity:
  `{metrics["random_forward_vs_source_relative_error"]:.8e}`;
- temperature-source centered FD:
  `{metrics["temperature_source_FD_relative_error"]:.8e}`.

This is a uniform-45-degree PTE surrogate, not a solved finite-contact
terminal current. No Maxwell solve, thermal solve, or optimization was run.
"""
    (output / "PTE_81X81_NODAL_CONTRACT_AUDIT.md").write_text(report)
    print(json.dumps(result, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
