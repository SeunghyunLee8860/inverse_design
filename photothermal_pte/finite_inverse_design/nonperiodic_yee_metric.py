"""Finite-support quadrature for a non-periodic raw Yee design monitor."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def clipped_voronoi_weights(
    coordinates_m: np.ndarray,
    lower_m: float,
    upper_m: float,
) -> np.ndarray:
    """Intersect component-grid control cells with a physical interval.

    The component coordinates may extend beyond the geometry because a raw
    no-interpolation Yee monitor reports staggered samples owned by adjacent
    FDTD cells.  A material derivative has support only inside its geometry,
    so its overlap metric is the intersection of those control cells with the
    exact design bounds.
    """

    coordinate = np.asarray(coordinates_m, float).reshape(-1)
    lower = float(lower_m)
    upper = float(upper_m)
    if coordinate.size < 2 or np.any(np.diff(coordinate) <= 0.0):
        raise ValueError("coordinates must be strictly increasing")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("invalid finite-support bounds")
    edges = np.empty(coordinate.size + 1, float)
    edges[1:-1] = 0.5 * (coordinate[:-1] + coordinate[1:])
    # Only the intersection with the physical support is used, so the two
    # unconstrained exterior Voronoi edges can be set to the support faces.
    edges[0] = lower
    edges[-1] = upper
    left = np.maximum(edges[:-1], lower)
    right = np.minimum(edges[1:], upper)
    return np.maximum(right - left, 0.0)


def clipped_component_yee_volumes(
    grid: Mapping[str, np.ndarray],
    bounds_m: Mapping[str, tuple[float, float]],
) -> dict[int, np.ndarray]:
    """Return Ex/Ey/Ez volumes clipped to an exact non-periodic box."""

    base = [np.asarray(grid[axis], float).reshape(-1) for axis in "xyz"]
    delta = [
        np.asarray(grid[f"delta_{axis}"], float).reshape(-1)
        for axis in "xyz"
    ]
    for coordinate, shift in zip(base, delta):
        if shift.shape != coordinate.shape:
            raise ValueError("coordinate and Yee shift shapes differ")
    volumes = {}
    for component in range(3):
        coordinates = [np.array(axis, copy=True) for axis in base]
        coordinates[component] += delta[component]
        weights = [
            clipped_voronoi_weights(coordinate, *bounds_m[axis])
            for axis, coordinate in zip("xyz", coordinates)
        ]
        volumes[component] = (
            weights[0][:, None, None]
            * weights[1][None, :, None]
            * weights[2][None, None, :]
        )
    return volumes
