"""Shared numerical metrics for fresh FDTDX field and absorption gates."""

from __future__ import annotations

from typing import Any

import numpy as np


def electric_yee_dual_volumes(
    grid: Any,
    grid_slice: tuple[slice, slice, slice],
) -> np.ndarray:
    """Return physical integration volumes for staggered Ex, Ey, and Ez."""

    if len(grid_slice) != 3:
        raise ValueError("grid_slice must contain exactly three spatial slices")
    widths = [
        np.asarray(grid.cell_widths(axis), dtype=np.float64) for axis in range(3)
    ]
    if any(value.ndim != 1 or np.any(value <= 0.0) for value in widths):
        raise ValueError("grid cell widths must be positive one-dimensional arrays")
    edge_dual = [
        0.5 * (np.concatenate((value[:1], value[:-1])) + value)
        for value in widths
    ]
    bounds = tuple((int(part.start), int(part.stop)) for part in grid_slice)
    if any(lower < 0 or upper <= lower for lower, upper in bounds):
        raise ValueError("grid_slice must contain nonempty non-negative bounds")

    volumes = []
    for component in range(3):
        selected = []
        for axis, (lower, upper) in enumerate(bounds):
            metric = widths[axis] if axis == component else edge_dual[axis]
            if upper > metric.size:
                raise ValueError("grid_slice exceeds the realized grid")
            selected.append(metric[lower:upper])
        volumes.append(
            selected[0][:, None, None]
            * selected[1][None, :, None]
            * selected[2][None, None, :]
        )
    return np.stack(volumes)


def weighted_complex_nrmse(
    late: np.ndarray,
    previous: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return a volume-weighted complex-field NRMSE without phase removal."""

    late_value = np.asarray(late)
    previous_value = np.asarray(previous)
    weight = np.asarray(weights, dtype=np.float64)
    if late_value.shape != previous_value.shape or late_value.ndim != 4:
        raise ValueError("fields must have identical (component,x,y,z) shapes")
    if weight.shape == late_value.shape[1:]:
        weight = np.broadcast_to(weight[None], late_value.shape)
    elif weight.shape != late_value.shape:
        raise ValueError("weights must be spatial or component-specific field weights")
    if np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("weights must be finite and non-negative")
    numerator = float(np.sum(np.abs(late_value - previous_value) ** 2 * weight))
    denominator = float(np.sum(np.abs(late_value) ** 2 * weight))
    return float(
        np.sqrt(numerator / max(denominator, np.finfo(float).tiny))
    )
