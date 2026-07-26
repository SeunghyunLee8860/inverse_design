from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
THERMAL_ADJOINT = HERE.parent
if str(THERMAL_ADJOINT) not in sys.path:
    sys.path.insert(0, str(THERMAL_ADJOINT))

from paper_reduced_thermal import (  # noqa: E402
    DesignSurfaceMap,
    evaluate_reduced_paper_thermal,
    reduced_flake_grid,
)


def _source(shape: tuple[int, int, int]) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, shape[0])[:, None, None]
    y = np.linspace(-1.0, 1.0, shape[1])[None, :, None]
    source = np.exp(-2.0 * ((x - 0.2) ** 2 + (y + 0.1) ** 2))
    source = np.broadcast_to(source, shape).copy()
    source *= 1.0e-12 / np.sum(source)
    # Unit test cells have unitless volume here only until the evaluator
    # multiplies by its actual grid volumes, so normalize once more there.
    return source


def test_design_surface_map_transpose_identity():
    mapping = DesignSurfaceMap(
        physical_shape=(21, 31, 4),
        face_shape=(4, 5),
    )
    rng = np.random.default_rng(2026072601)
    physical = rng.normal(size=mapping.physical_shape)
    face = rng.normal(size=mapping.face_shape)
    left = float(np.sum(mapping.apply(physical) * face))
    right = float(np.sum(physical * mapping.transpose(face)))
    assert np.isclose(left, right, rtol=2e-15, atol=1e-14)
    transpose = mapping.transpose(face)
    assert np.all(transpose[-1, :, :] == 0.0)
    assert np.all(transpose[:, -1, :] == 0.0)


def test_reduced_robin_thermal_material_gradient_matches_fd():
    grid = reduced_flake_grid(nx=8, ny=8, nz=2)
    shape = tuple(axis.size - 1 for axis in grid)
    x = 0.5 * (grid[0][:-1] + grid[0][1:])
    y = 0.5 * (grid[1][:-1] + grid[1][1:])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    rho = 0.45 + 0.15 * np.sin(2.0 * np.pi * xx / 6.0e-6) * np.cos(
        2.0 * np.pi * yy / 6.0e-6
    )
    source = np.zeros(shape, float)
    source[:, :, :] = np.exp(
        -2.0
        * (
            ((xx - 0.4e-6) / 1.1e-6) ** 2
            + ((yy + 0.3e-6) / 0.9e-6) ** 2
        )
    )[:, :, None]
    volume = (
        np.diff(grid[0])[:, None, None]
        * np.diff(grid[1])[None, :, None]
        * np.diff(grid[2])[None, None, :]
    )
    source *= 1.0e-12 / np.sum(source * volume)
    direction = np.cos(2.0 * np.pi * xx / 6.0e-6) * np.sin(
        2.0 * np.pi * yy / 6.0e-6
    )
    direction /= np.max(np.abs(direction))
    baseline = evaluate_reduced_paper_thermal(
        rho_face=rho,
        source_W_m3=source,
        grid=grid,
    )
    analytic = float(np.sum(baseline.gradient_rho_face_A_m * direction))
    step = 1.0e-5
    plus = evaluate_reduced_paper_thermal(
        rho_face=rho + step * direction,
        source_W_m3=source,
        grid=grid,
    ).objective_A_m
    minus = evaluate_reduced_paper_thermal(
        rho_face=rho - step * direction,
        source_W_m3=source,
        grid=grid,
    ).objective_A_m
    finite_difference = (plus - minus) / (2.0 * step)
    error = abs(finite_difference - analytic) / max(
        abs(finite_difference), abs(analytic), 1e-300
    )
    assert error < 1.0e-5
