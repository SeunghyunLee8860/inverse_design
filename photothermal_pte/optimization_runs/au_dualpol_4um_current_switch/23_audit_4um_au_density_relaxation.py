#!/usr/bin/env python3
"""Audit the solver-free 4-um Au density constitutive law.

This does not launch Lumerical and cannot replace the B200 field/resonance or
AD-FD gates.  It proves the endpoint, passivity, no-rho-cubed, and analytic
complex-derivative parts of the contract before an expensive solver run.
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

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.au_density_relaxation import (
    audit,
    d_epsilon_d_projected_density,
    epsilon_relaxation,
)


def derivative_audit() -> dict[str, object]:
    rng = np.random.default_rng(20260824)
    rho = 0.1 + 0.8 * rng.random((11, 13))
    direction = rng.standard_normal(rho.shape)
    direction /= np.max(np.abs(direction))
    analytic = d_epsilon_d_projected_density(rho) * direction
    rows = []
    for step in (1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4):
        finite_difference = (
            epsilon_relaxation(rho + step * direction)
            - epsilon_relaxation(rho - step * direction)
        ) / (2.0 * step)
        error = float(
            np.linalg.norm(finite_difference - analytic)
            / max(np.linalg.norm(analytic), 1.0e-300)
        )
        rows.append({"step": step, "complex_directional_relative_error": error})
    return {
        "method": "independent centered finite difference of complex epsilon map",
        "shape": list(rho.shape),
        "steps": rows,
        "maximum_relative_error": max(row["complex_directional_relative_error"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    payload = audit()
    payload["complex_derivative_audit"] = derivative_audit()
    payload["status"] = (
        "VALIDATED_SOLVER_FREE_4UM_AU_NK_DENSITY_LAW"
        if payload["passive_on_uniform_density_sweep"]
        and payload["exact_background_endpoint"]
        and payload["exact_au_endpoint"]
        and not payload["rho_cubed_used"]
        and payload["complex_derivative_audit"]["maximum_relative_error"] < 1.0e-9
        else "FAILED_SOLVER_FREE_4UM_AU_NK_DENSITY_LAW"
    )
    payload["passed"] = payload["status"].startswith("VALIDATED")
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
