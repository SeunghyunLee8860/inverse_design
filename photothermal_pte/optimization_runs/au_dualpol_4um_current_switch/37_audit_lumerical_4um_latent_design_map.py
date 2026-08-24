#!/usr/bin/env python3
"""Audit the 81x81 latent/filter/projection/PDE/DFM transpose chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.dfm import (
    DEFAULT_POSITIVE_PART_TAU,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    NOMINAL_MAPPING,
    design_state_audit,
    projected_cell_density,
    projected_cell_jvp,
    projected_cell_vjp,
    smooth_lumerical_500nm_constraints,
)


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)


def latent_design_map_audit() -> dict[str, object]:
    rng = np.random.default_rng(20260824)
    x = np.linspace(-1.0, 1.0, CONTRACT.design_node_shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, CONTRACT.design_node_shape[1])[None, :]
    latent = 0.5 + 0.16 * np.sin(0.8 * np.pi * x) * np.cos(0.6 * np.pi * y)
    direction = rng.standard_normal(CONTRACT.design_node_shape)
    direction /= np.max(np.abs(direction))
    nodal_cotangent = rng.standard_normal(CONTRACT.design_node_shape)
    cell_cotangent = rng.standard_normal(CONTRACT.design_shape)
    beta = 4.0
    step = 1.0e-6

    projected = NOMINAL_MAPPING.physical(latent, beta)
    nodal_tangent = NOMINAL_MAPPING.jvp(latent, direction, beta)
    nodal_pullback = NOMINAL_MAPPING.vjp(latent, nodal_cotangent, beta)
    nodal_transpose_error = _relative_difference(
        float(np.vdot(nodal_cotangent, nodal_tangent)),
        float(np.vdot(nodal_pullback, direction)),
    )
    nodal_fd = (
        NOMINAL_MAPPING.physical(latent + step * direction, beta)
        - NOMINAL_MAPPING.physical(latent - step * direction, beta)
    ) / (2.0 * step)
    nodal_fd_error = float(
        np.linalg.norm(nodal_tangent - nodal_fd)
        / max(np.linalg.norm(nodal_fd), np.finfo(float).tiny)
    )

    cell_tangent = projected_cell_jvp(latent, direction, beta)
    cell_pullback = projected_cell_vjp(latent, cell_cotangent, beta)
    cell_transpose_error = _relative_difference(
        float(np.vdot(cell_cotangent, cell_tangent)),
        float(np.vdot(cell_pullback, direction)),
    )
    cell_fd = (
        projected_cell_density(latent + step * direction, beta)
        - projected_cell_density(latent - step * direction, beta)
    ) / (2.0 * step)
    cell_fd_error = float(
        np.linalg.norm(cell_tangent - cell_fd)
        / max(np.linalg.norm(cell_fd), np.finfo(float).tiny)
    )

    dfm_step = 2.5e-4
    dfm_values, dfm_gradients, _ = smooth_lumerical_500nm_constraints(
        latent, beta
    )
    dfm_ad = dfm_gradients.reshape(2, -1) @ direction.ravel()
    dfm_plus = smooth_lumerical_500nm_constraints(
        latent + dfm_step * direction, beta
    )[0]
    dfm_minus = smooth_lumerical_500nm_constraints(
        latent - dfm_step * direction, beta
    )[0]
    dfm_fd = (dfm_plus - dfm_minus) / (2.0 * dfm_step)
    dfm_errors = np.abs(dfm_ad - dfm_fd) / np.maximum(
        np.maximum(np.abs(dfm_ad), np.abs(dfm_fd)), 1.0e-14
    )

    mapping = NOMINAL_MAPPING.audit()
    state = design_state_audit(latent, beta)
    positive_tau = DEFAULT_POSITIVE_PART_TAU
    gates = {
        "latent_and_projected_are_81x81": projected.shape
        == CONTRACT.design_node_shape,
        "PDE_and_DFM_cells_are_80x80": projected_cell_density(
            latent, beta
        ).shape
        == CONTRACT.design_shape,
        "filter_preserves_constants_lt_1e_14": mapping[
            "constant_preservation_max_abs"
        ]
        < 1.0e-14,
        "filter_projection_transpose_lt_1e_12": nodal_transpose_error
        < 1.0e-12,
        "filter_projection_directional_FD_lt_1e_7": nodal_fd_error < 1.0e-7,
        "cell_chain_transpose_lt_1e_12": cell_transpose_error < 1.0e-12,
        "cell_chain_directional_FD_lt_1e_7": cell_fd_error < 1.0e-7,
        "smooth_DFM_directional_FD_lt_0p5pct": float(np.max(dfm_errors))
        < 5.0e-3,
        "one_shared_projected_density_hash": state[
            "shared_projected_density"
        ]["all_constitutive_maps_derive_from_this_nodal_state"]
        is True,
        "no_rho3_and_no_np_density": mapping["optical_rho_power"] is None
        and mapping["np_density_used"] is False,
    }
    passed = all(gates.values())
    return {
        "status": (
            "VALIDATED_SOLVER_FREE_LUMERICAL_4UM_LATENT_DESIGN_MAP"
            if passed
            else "FAILED_SOLVER_FREE_LUMERICAL_4UM_LATENT_DESIGN_MAP"
        ),
        "passed": passed,
        "scope": (
            "81x81 nodal latent -> finite conic filter -> tanh projection -> "
            "shared projected nodes -> exact 80x80 cell average -> smooth DFM"
        ),
        "beta": beta,
        "mapping": mapping,
        "state": state,
        "metrics": {
            "filter_projection_transpose_relative_error": nodal_transpose_error,
            "filter_projection_directional_FD_relative_error": nodal_fd_error,
            "cell_chain_transpose_relative_error": cell_transpose_error,
            "cell_chain_directional_FD_relative_error": cell_fd_error,
            "smooth_DFM_values": dfm_values.tolist(),
            "smooth_DFM_AD": dfm_ad.tolist(),
            "smooth_DFM_FD": dfm_fd.tolist(),
            "smooth_DFM_directional_relative_errors": dfm_errors.tolist(),
            "smooth_positive_part_tau": positive_tau,
            "smooth_positive_part_pointwise_upper_error_bound": positive_tau
            * float(np.log(2.0)),
        },
        "gates": gates,
        "Maxwell_solves": 0,
        "custom_CUDA_solves": 0,
        "Lumerical_HEAT_or_CHARGE_solves": 0,
        "optimizer_iterations": 0,
        "remaining_gates": [
            "complete latent AD-FD through Lumerical and custom CUDA PDEs",
            "Eb and signed dual-polarization derivative",
            "selected-mesh and B200 repetition",
            "final ordinary dispersive-Au binary reevaluation",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    payload = latent_design_map_audit()
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
