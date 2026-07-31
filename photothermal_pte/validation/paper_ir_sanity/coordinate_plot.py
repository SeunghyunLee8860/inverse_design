"""Coordinate-faithful plotting helpers for physical cell fields.

``imshow`` assigns equal display width to every array element.  That is not a
valid representation of the nonuniform Cartesian grids used by the thermal
FVM, and using only the first/last cell centres as an ``extent`` also shifts
uniform-grid boundaries by half a cell.  Physical field maps in this package
must therefore be drawn from cell edges with ``pcolormesh``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def strict_centered_xy_mask(material_mask: np.ndarray) -> np.ndarray:
    """Cells having same-material neighbours in all four x/y directions."""

    mask = np.asarray(material_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("material_mask must be two-dimensional")
    valid = np.zeros_like(mask)
    valid[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return valid


def dual_edges_from_centers(coordinates: np.ndarray) -> np.ndarray:
    """Return control-volume edges for a strictly increasing centre array."""

    centers = np.asarray(coordinates, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("coordinates must be a one-dimensional array of length >= 2")
    spacing = np.diff(centers)
    if not np.all(np.isfinite(centers)) or np.any(spacing <= 0.0):
        raise ValueError("coordinates must be finite and strictly increasing")
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * spacing[0]
    edges[-1] = centers[-1] + 0.5 * spacing[-1]
    return edges


def cell_field(
    axis: Any,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    values: np.ndarray,
    *,
    coordinate_scale: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Draw an ``(nx, ny)`` cell field at its exact physical boundaries."""

    x_boundaries = np.asarray(x_edges, dtype=float)
    y_boundaries = np.asarray(y_edges, dtype=float)
    field = np.asarray(values)
    expected = (x_boundaries.size - 1, y_boundaries.size - 1)
    if field.shape != expected:
        raise ValueError(f"field shape {field.shape} does not match cell grid {expected}")
    if np.any(np.diff(x_boundaries) <= 0.0) or np.any(np.diff(y_boundaries) <= 0.0):
        raise ValueError("cell edges must be strictly increasing")
    image = axis.pcolormesh(
        x_boundaries * coordinate_scale,
        y_boundaries * coordinate_scale,
        field.T,
        shading="flat",
        **kwargs,
    )
    axis.set_aspect("equal")
    return image


def center_field(
    axis: Any,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    values: np.ndarray,
    *,
    coordinate_scale: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Draw a nodal/common-grid field using its dual control-volume edges."""

    return cell_field(
        axis,
        dual_edges_from_centers(x_centers),
        dual_edges_from_centers(y_centers),
        values,
        coordinate_scale=coordinate_scale,
        **kwargs,
    )
