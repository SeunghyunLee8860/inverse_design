#!/usr/bin/env python3
"""Validate the selected 373-node to 186-cell thermal-density operator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from selected_thermal_density_mapping import (
    SELECTED_BOUNDS_M,
    SELECTED_NODAL_SHAPE,
    SELECTED_THERMAL_CELL_SHAPE,
    THERMAL_CELL_STEP_M,
    bilinear_integral,
    selected_nodal_to_thermal_cell,
    selected_nodal_to_thermal_cell_transpose,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


def fields() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x = np.linspace(-1.0, 1.0, 373)[:, None]
    y = np.linspace(-1.0, 1.0, 373)[None, :]
    base = 0.5 + 0.035 * x + 0.025 * np.sin(np.pi * x) * np.cos(0.5 * np.pi * y)
    rng = np.random.default_rng(2026080605)
    raw = {
        "uniform": np.ones(SELECTED_NODAL_SHAPE),
        "smooth_asymmetric": 0.65 * np.sin(np.pi * x) * np.cos(0.5 * np.pi * y) + 0.35 * x,
        "central_localized": np.exp(-((x + 0.15) ** 2 + (y - 0.10) ** 2) / 0.02),
        "design_edge_localized": np.exp(-((x - 0.92) ** 2 + (y + 0.72) ** 2) / 0.01),
        "fixed_seed_random": rng.normal(size=SELECTED_NODAL_SHAPE),
    }
    return base, {name: value / np.max(np.abs(value)) for name, value in raw.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    base, directions = fields()
    rng = np.random.default_rng(2026080606)
    rows = []
    for name, direction in directions.items():
        dual = rng.normal(size=SELECTED_THERMAL_CELL_SHAPE)
        mapped = selected_nodal_to_thermal_cell(direction)
        left = float(np.sum(mapped * dual))
        right = float(
            np.sum(direction * selected_nodal_to_thermal_cell_transpose(dual))
        )
        step = 2.5e-3
        finite_difference = (
            selected_nodal_to_thermal_cell(base + step * direction)
            - selected_nodal_to_thermal_cell(base - step * direction)
        ) / (2.0 * step)
        rows.append(
            {
                "direction": name,
                "transpose_relative_error": relative(left, right),
                "mapping_FD_relative_error": float(
                    np.linalg.norm(finite_difference - mapped)
                    / max(np.linalg.norm(mapped), np.finfo(float).tiny)
                ),
            }
        )

    constant_error = float(
        np.max(np.abs(selected_nodal_to_thermal_cell(np.ones(SELECTED_NODAL_SHAPE)) - 1.0))
    )
    impulse = np.zeros(SELECTED_NODAL_SHAPE)
    impulse[0, 0] = 1.0
    mapped_impulse = selected_nodal_to_thermal_cell(impulse)
    opposite_edge_wrap = float(
        max(np.max(np.abs(mapped_impulse[-1, :])), np.max(np.abs(mapped_impulse[:, -1])))
    )
    mapped_base = selected_nodal_to_thermal_cell(base)
    nodal_integral = bilinear_integral(base)
    cell_integral = float(np.sum(mapped_base) * THERMAL_CELL_STEP_M**2)
    integral_error = relative(nodal_integral, cell_integral)
    worst_dot = max(row["transpose_relative_error"] for row in rows)
    worst_fd = max(row["mapping_FD_relative_error"] for row in rows)
    passed = bool(
        constant_error < 1e-15
        and opposite_edge_wrap == 0.0
        and integral_error < 1e-14
        and worst_dot < 1e-12
        and worst_fd < 1e-10
    )
    artifact = output / "selected_thermal_density_mapping.npz"
    np.savez_compressed(
        artifact,
        base_nodal_density=base,
        base_thermal_cell_density=mapped_base,
        **{f"direction_{name}": value for name, value in directions.items()},
    )
    result = {
        "status": "VALIDATED_SELECTED_373_NODE_TO_186_THERMAL_CELL_MAPPING" if passed else "FAILED_SELECTED_373_NODE_TO_186_THERMAL_CELL_MAPPING",
        "passed": passed,
        "scope": "solver-free exact bilinear area averaging and Euclidean transpose",
        "bounds_m": list(SELECTED_BOUNDS_M),
        "nodal_shape": list(SELECTED_NODAL_SHAPE),
        "thermal_cell_shape": list(SELECTED_THERMAL_CELL_SHAPE),
        "weights_1d": [0.25, 0.5, 0.25],
        "constant_preservation_error": constant_error,
        "opposite_edge_wrap_error": opposite_edge_wrap,
        "bilinear_nodal_integral": nodal_integral,
        "thermal_cell_integral": cell_integral,
        "integral_relative_error": integral_error,
        "directions": rows,
        "worst_transpose_relative_error": worst_dot,
        "worst_mapping_FD_relative_error": worst_fd,
        "artifact": {
            "path": str(artifact),
            "size_bytes": artifact.stat().st_size,
            "sha256": sha256(artifact),
        },
        "Maxwell_solves": 0,
        "thermal_solves": 0,
        "optimization_iterations": 0,
    }
    result_path = output / "selected_thermal_density_mapping_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
