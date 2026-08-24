#!/usr/bin/env python3
"""Audit the shared 81x81 nodal state and its 80x80 PDE pullback.

This is a solver-free discrete-map audit.  It does not launch Lumerical and
cannot replace the B200 component-Yee material Jacobian or full latent AD-FD
gates.
"""

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
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (
    density_state_audit,
    nodal_to_cell_average,
    nodal_to_cell_jvp,
    nodal_to_cell_vjp,
)


def mapping_audit() -> dict[str, object]:
    rng = np.random.default_rng(20260824)
    nodes = 0.1 + 0.8 * rng.random(CONTRACT.design_node_shape)
    direction = rng.standard_normal(CONTRACT.design_node_shape)
    direction /= np.max(np.abs(direction))
    cell_cotangent = rng.standard_normal(CONTRACT.design_shape)

    forward_inner_product = float(
        np.vdot(nodal_to_cell_jvp(direction), cell_cotangent)
    )
    transpose_inner_product = float(
        np.vdot(direction, nodal_to_cell_vjp(cell_cotangent))
    )
    transpose_relative_error = abs(
        forward_inner_product - transpose_inner_product
    ) / max(abs(forward_inner_product), 1.0e-300)

    cells = nodal_to_cell_average(nodes)
    gradient = nodal_to_cell_vjp(2.0 * cells)
    analytic_directional = float(np.vdot(gradient, direction))
    finite_difference_rows = []
    for step in (1.0e-4, 5.0e-5, 2.5e-5, 1.25e-5):
        plus = float(
            np.sum(nodal_to_cell_average(nodes + step * direction) ** 2)
        )
        minus = float(
            np.sum(nodal_to_cell_average(nodes - step * direction) ** 2)
        )
        finite_difference = (plus - minus) / (2.0 * step)
        relative_error = abs(finite_difference - analytic_directional) / max(
            abs(analytic_directional), 1.0e-300
        )
        finite_difference_rows.append(
            {"step": step, "directional_relative_error": relative_error}
        )

    state = density_state_audit(nodes)
    passed = bool(
        transpose_relative_error < 1.0e-12
        and max(row["directional_relative_error"] for row in finite_difference_rows)
        < 1.0e-7
        and state["nodal_shape_xy"] == [81, 81]
        and state["pde_cell_shape_xy"] == [80, 80]
    )
    return {
        "status": (
            "VALIDATED_SOLVER_FREE_SHARED_DENSITY_MAP"
            if passed
            else "FAILED_SOLVER_FREE_SHARED_DENSITY_MAP"
        ),
        "passed": passed,
        "state": state,
        "transpose_relative_error": transpose_relative_error,
        "finite_difference": finite_difference_rows,
        "Lumerical_solve_run": False,
        "remaining_gates": [
            "B200 nonuniform density-to-component-Yee Jacobian",
            "B200 complete latent/filter/projection AD-FD",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    payload = mapping_audit()
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
