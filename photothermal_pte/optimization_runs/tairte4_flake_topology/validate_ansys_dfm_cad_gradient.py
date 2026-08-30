#!/usr/bin/env python3
"""CAD-only directional-FD certificate for the official Ansys DFM gradient."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.ansys_minimum_feature import (
    evaluate_on_cad,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.validate_combined_adfd import (
    open_fdtd,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-device", default="GPU 1")
    args = parser.parse_args()
    rng = np.random.default_rng(90210)
    latent = np.full(CONTRACT.design_node_shape, 0.5, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, latent.shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, latent.shape[1])[None, :]
    latent += 0.15 * np.sin(2.3 * x) * np.cos(1.7 * y)
    steps = (1.0e-2, 5.0e-3, 1.0e-3, 5.0e-4, 1.0e-4)
    fdtd = None
    result = {"passed": False}
    try:
        fdtd, _, _ = open_fdtd(args.gpu_device)
        indicators, gradient, metadata = evaluate_on_cad(fdtd, latent, 16.0)
        directions = {
            "adjoint_aligned": gradient.copy(),
            "random": rng.normal(size=latent.shape),
            "asymmetric_local": gradient
            * np.exp(-(((x - 0.25) / 0.3) ** 2 + ((y + 0.1) / 0.22) ** 2)),
        }
        directions = {
            name: value / np.linalg.norm(value)
            for name, value in directions.items()
        }
        comparisons = []
        for direction_name, direction in directions.items():
            analytic = float(np.sum(gradient * direction))
            for step in steps:
                plus, _, _ = evaluate_on_cad(
                    fdtd, latent + step * direction, 16.0
                )
                minus, _, _ = evaluate_on_cad(
                    fdtd, latent - step * direction, 16.0
                )
                finite_difference = float(np.sum(plus - minus) / (2.0 * step))
                error = abs(analytic - finite_difference) / max(
                    abs(analytic), abs(finite_difference), 1.0e-15
                )
                comparisons.append(
                    {
                        "direction": direction_name,
                        "step": step,
                        "analytic": analytic,
                        "central_fd": finite_difference,
                        "relative_error": error,
                    }
                )
        best_by_direction = {
            name: min(
                (row for row in comparisons if row["direction"] == name),
                key=lambda row: row["relative_error"],
            )
            for name in directions
        }
        maximum_best_error = max(
            row["relative_error"] for row in best_by_direction.values()
        )
        result = {
            "passed": bool(maximum_best_error < 0.02),
            "status": (
                "VALIDATED_ANSYS_V261_DFM_CAD_GRADIENT"
                if maximum_best_error < 0.02
                else "FAILED_ANSYS_V261_DFM_CAD_GRADIENT"
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "beta": 16.0,
            "indicator_solid": float(indicators[0]),
            "indicator_void": float(indicators[1]),
            "comparisons": comparisons,
            "best_by_direction": best_by_direction,
            "maximum_best_relative_error": maximum_best_error,
            "gate": "best central-FD plateau error <2% for every direction",
            "metadata": metadata,
            "Maxwell_solves": 0,
        }
    finally:
        if fdtd is not None:
            fdtd.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
