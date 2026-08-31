#!/usr/bin/env python3
"""Full custom thermal/electrical AD--FD and 3-D scenario preflight."""

from __future__ import annotations

import argparse
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
    evaluate_fixed_source_volumetric,
    solve_volumetric_electrical,
)


def _source(state) -> np.ndarray:
    x, y, z = state.centers
    envelope = np.exp(
        -2.0
        * ((x[:, None, None] - 0.7e-6) ** 2 + (y[None, :, None] + 1.0e-6) ** 2)
        / (4.0e-6) ** 2
    )
    material_weight = (
        0.65 * np.asarray(state.masks["design_au"], dtype=np.float64)
        + 0.35 * np.asarray(state.masks["tairte4"], dtype=np.float64)
    )
    value = envelope * material_weight
    value *= 80.0e-6 / np.sum(value)
    return value


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
        reference_state = build_thermal_state(rho)
        source = _source(reference_state)
        base = evaluate_fixed_source_volumetric(
            rho, source, args.cuda_device, need_gradient=True
        )
        rng = np.random.default_rng(93217)
        direction = gaussian_filter(rng.normal(size=rho.shape), sigma=4.0)
        direction /= np.max(np.abs(direction))
        h = float(args.step)
        currents = []
        for sign in (-1.0, 1.0):
            perturbed = evaluate_fixed_source_volumetric(
                rho + sign * h * direction,
                source,
                args.cuda_device,
                need_gradient=False,
            )
            currents.append(float(perturbed["objective_A"]))
        finite_difference = (currents[1] - currents[0]) / (2.0 * h)
        adjoint_directional = float(
            np.sum(np.asarray(base["gradient_direct_A"]) * direction)
        )
        relative_error = abs(adjoint_directional - finite_difference) / max(
            abs(adjoint_directional),
            abs(finite_difference),
            np.finfo(float).tiny,
        )

        temperature = np.asarray(base["temperature"])
        sigma_z_sensitivity: dict[str, float] = {}
        for sigma_z in (1.10e3, 1.10e4, 1.10e5):
            system = build_volumetric_electrical_system(
                base["state"], temperature, sigma_z_S_m=sigma_z
            )
            _, value, _ = solve_volumetric_electrical(system, args.cuda_device)
            sigma_z_sensitivity[f"{sigma_z:.2e}"] = float(value)

        exact_mask = np.zeros((80, 80), dtype=np.float64)
        exact_mask[18:62, 24:56] = 1.0
        exact_state = build_thermal_state(exact_mask)
        exact_system = build_volumetric_electrical_system(
            exact_state,
            temperature,
            exact_binary_geometry=True,
        )
        _, exact_current, exact_audit = solve_volumetric_electrical(
            exact_system, args.cuda_device
        )
        expected_removed_cells = int(np.count_nonzero(exact_mask == 0.0))

        thermal = base["thermal_audit"]
        electrical = base["electrical_audit"]
        thermal_adjoint = base["thermal_adjoint_audit"]
        electrical_adjoint = base["electrical_adjoint_audit"]
        gates = {
            "full_pde_adfd_relative_lt_1e_4": relative_error < 1.0e-4,
            "full_pde_adfd_sign_match": bool(
                np.sign(adjoint_directional) == np.sign(finite_difference)
            ),
            "thermal_forward_residual_lt_1e_8": float(
                thermal["relative_residual"]
            )
            < 1.0e-8,
            "thermal_adjoint_residual_lt_1e_8": float(
                thermal_adjoint["relative_residual"]
            )
            < 1.0e-8,
            "electrical_forward_residual_lt_1e_8": float(
                electrical["relative_residual"]
            )
            < 1.0e-8,
            "electrical_adjoint_residual_lt_1e_8": float(
                electrical_adjoint["relative_residual"]
            )
            < 1.0e-8,
            "volumetric_integral_lt_1e_12": float(
                electrical["volumetric_integral_relative_error"]
            )
            < 1.0e-12,
            "exact_binary_void_cells_removed": bool(
                exact_audit["exact_binary_geometry"]
                and exact_audit["electrical_void_Au_nodes_removed"]
                and int(exact_audit["inactive_void_Au_cell_count"])
                == expected_removed_cells
            ),
            "all_sigma_z_scenarios_finite": bool(
                all(np.isfinite(value) for value in sigma_z_sensitivity.values())
            ),
        }
        report = {
            "passed": all(gates.values()),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": (
                "fixed-source custom 3-D thermal plus 3-D electrical AD--FD, "
                "exact binary topology, and unmeasured sigma_z scenarios"
            ),
            "source_power_W": float(np.sum(source)),
            "rho": 0.5,
            "step": h,
            "direction_seed": 93217,
            "direction_gaussian_sigma_cells": 4.0,
            "base_current_A": float(base["objective_A"]),
            "minus_current_A": currents[0],
            "plus_current_A": currents[1],
            "adjoint_directional_A_per_unit": adjoint_directional,
            "finite_difference_directional_A_per_unit": finite_difference,
            "relative_error": relative_error,
            "sigma_z_sensitivity_current_A": sigma_z_sensitivity,
            "exact_binary": {
                "current_A": exact_current,
                "solid_cells": int(np.count_nonzero(exact_mask)),
                "expected_removed_void_cells": expected_removed_cells,
                "audit": exact_audit,
            },
            "thermal": thermal,
            "thermal_adjoint": thermal_adjoint,
            "electrical": electrical,
            "electrical_adjoint": electrical_adjoint,
            "gates": gates,
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
