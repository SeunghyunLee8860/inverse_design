#!/usr/bin/env python3
"""Directional AD--FD check for the explicit 3-D electrical operator."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.ndimage import gaussian_filter

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    build_thermal_state,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.volumetric_electrical_4um import (
    build_volumetric_electrical_system,
    solve_volumetric_electrical,
    solve_volumetric_electrical_adjoint,
    volumetric_electrical_density_gradient,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--step", type=float, default=1.0e-4)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report: dict[str, object] = {"passed": False}
    try:
        rho = np.full((80, 80), 0.5, dtype=np.float64)
        state = build_thermal_state(rho)
        x, y, z = state.centers
        temperature = (
            0.7
            * np.exp(
                -(
                    (x[:, None, None] - 1.2e-6) ** 2
                    + (y[None, :, None] + 0.9e-6) ** 2
                )
                / (2.0 * (2.3e-6) ** 2)
            )
            * np.exp(-((z[None, None, :] + 0.03e-6) / (0.65e-6)) ** 2)
            + 0.08 * x[:, None, None] / 8.0e-6
            - 0.04 * y[None, :, None] / 8.0e-6
        )
        system = build_volumetric_electrical_system(state, temperature)
        psi, current, forward_audit = solve_volumetric_electrical(
            system, args.cuda_device
        )
        adjoint, adjoint_audit = solve_volumetric_electrical_adjoint(
            system, args.cuda_device
        )
        gradient = volumetric_electrical_density_gradient(system, psi, adjoint)
        rng = np.random.default_rng(73491)
        direction = gaussian_filter(rng.normal(size=rho.shape), sigma=4.0)
        direction /= np.max(np.abs(direction))
        h = float(args.step)
        currents = []
        for sign in (-1.0, 1.0):
            perturbed_rho = rho + sign * h * direction
            perturbed_state = replace(state, rho=perturbed_rho)
            perturbed_system = build_volumetric_electrical_system(
                perturbed_state, temperature
            )
            _, perturbed_current, _ = solve_volumetric_electrical(
                perturbed_system, args.cuda_device
            )
            currents.append(float(perturbed_current))
        finite_difference = (currents[1] - currents[0]) / (2.0 * h)
        adjoint_directional = float(np.sum(gradient * direction))
        relative_error = abs(adjoint_directional - finite_difference) / max(
            abs(adjoint_directional),
            abs(finite_difference),
            np.finfo(float).tiny,
        )
        passed = bool(
            relative_error < 1.0e-4
            and np.sign(adjoint_directional) == np.sign(finite_difference)
            and float(forward_audit["relative_residual"]) < 1.0e-8
            and float(adjoint_audit["relative_residual"]) < 1.0e-8
            and float(
                forward_audit["volumetric_integral_normwise_relative_error"]
            )
            < 1.0e-12
        )
        report = {
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "fixed-temperature electrical-only 3-D volumetric AD--FD",
            "rho": 0.5,
            "step": h,
            "direction_seed": 73491,
            "direction_gaussian_sigma_cells": 4.0,
            "base_current_A": current,
            "minus_current_A": currents[0],
            "plus_current_A": currents[1],
            "adjoint_directional_A_per_unit": adjoint_directional,
            "finite_difference_directional_A_per_unit": finite_difference,
            "relative_error": relative_error,
            "sign_match": bool(
                np.sign(adjoint_directional) == np.sign(finite_difference)
            ),
            "forward": forward_audit,
            "adjoint": adjoint_audit,
            "wall_s": time.monotonic() - started,
        }
    except Exception as error:
        report.update(
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            wall_s=time.monotonic() - started,
        )
    output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0 if report.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
