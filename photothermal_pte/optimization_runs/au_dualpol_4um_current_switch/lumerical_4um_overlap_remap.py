"""Solver-independent Cartesian remap used by the Lumerical/CUDA route.

The Maxwell samples supplied to these helpers come from Lumerical component
Yee grids.  The functions map component-dual-cell power conservatively to the
custom finite-volume thermal grid and provide the exact transpose operation.
This module contains no Maxwell solver API and is safe to import in the
Lumerical-only production path.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def component_yee_coordinates(realized_grid, grid_slice, component: int):
    """Return coordinates and dual widths for one component-Yee grid slice."""

    if component not in (0, 1, 2):
        raise ValueError("component must be 0, 1, or 2")
    coordinates = []
    widths = []
    for axis, part in enumerate(grid_slice):
        edges = np.asarray(realized_grid.edges(axis), dtype=np.float64)
        if (
            edges.ndim != 1
            or edges.size < 2
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError(f"invalid realized-grid edges on axis {axis}")
        centers = 0.5 * (edges[:-1] + edges[1:])
        primal_width = np.diff(edges)
        edge_dual_width = 0.5 * (
            np.concatenate((primal_width[:1], primal_width[:-1])) + primal_width
        )
        sample = centers if axis == component else edges[:-1]
        metric = primal_width if axis == component else edge_dual_width
        lower, upper = int(part.start), int(part.stop)
        coordinates.append(sample[lower:upper])
        widths.append(metric[lower:upper])
    return tuple(coordinates), tuple(widths)


def primal_edges(coordinate: np.ndarray, width: np.ndarray) -> np.ndarray:
    """Recover contiguous primal-cell edges from centers and cell widths."""

    coordinate = np.asarray(coordinate, dtype=np.float64)
    width = np.asarray(width, dtype=np.float64)
    if (
        coordinate.ndim != 1
        or coordinate.shape != width.shape
        or coordinate.size == 0
        or not np.all(np.isfinite(coordinate))
        or not np.all(np.isfinite(width))
        or np.any(width <= 0.0)
    ):
        raise ValueError("coordinate and width must be finite positive 1-D arrays")
    lower = coordinate - 0.5 * width
    upper = coordinate + 0.5 * width
    mismatch = (
        float(np.max(np.abs(upper[:-1] - lower[1:])))
        if len(width) > 1
        else 0.0
    )
    if mismatch > 5.0e-13:
        raise RuntimeError(f"non-contiguous primal cell edges: {mismatch:.6e} m")
    return np.concatenate((lower[:1], upper))


def overlap_operator(
    coordinate: np.ndarray,
    width: np.ndarray,
    target_edges: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Map source dual-cell power to target cells by normalized overlap."""

    coordinate = np.asarray(coordinate, dtype=np.float64)
    width = np.asarray(width, dtype=np.float64)
    target_edges = np.asarray(target_edges, dtype=np.float64)
    if (
        coordinate.ndim != 1
        or coordinate.shape != width.shape
        or coordinate.size == 0
        or not np.all(np.isfinite(coordinate))
        or not np.all(np.isfinite(width))
        or np.any(width <= 0.0)
    ):
        raise ValueError("coordinate and width must be finite positive 1-D arrays")
    if (
        target_edges.ndim != 1
        or target_edges.size < 2
        or not np.all(np.isfinite(target_edges))
        or np.any(np.diff(target_edges) <= 0.0)
    ):
        raise ValueError("target_edges must be finite and strictly increasing")

    target_count = len(target_edges) - 1
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    retained = np.zeros(len(coordinate), dtype=np.float64)
    domain_low, domain_high = target_edges[0], target_edges[-1]
    for source_index, (center, full_width) in enumerate(
        zip(coordinate, width, strict=True)
    ):
        full_low = center - 0.5 * full_width
        full_high = center + 0.5 * full_width
        low = max(full_low, domain_low)
        high = min(full_high, domain_high)
        denominator = high - low
        if denominator <= 0.0:
            raise RuntimeError(
                f"source dual cell {source_index} has no target-domain overlap"
            )
        retained[source_index] = denominator / full_width
        first = max(int(np.searchsorted(target_edges, low, side="right")) - 1, 0)
        last = min(
            int(np.searchsorted(target_edges, high, side="left")) + 1,
            target_count,
        )
        for target_index in range(first, last):
            overlap = max(
                0.0,
                min(high, target_edges[target_index + 1])
                - max(low, target_edges[target_index]),
            )
            if overlap > 0.0:
                rows.append(target_index)
                columns.append(source_index)
                values.append(overlap / denominator)
    operator = sparse.coo_matrix(
        (values, (rows, columns)), shape=(target_count, len(coordinate))
    ).tocsr()
    column_sum = np.asarray(operator.sum(axis=0)).reshape(-1)
    if not np.allclose(column_sum, 1.0, rtol=0.0, atol=2.0e-13):
        raise RuntimeError(
            "overlap operator is not conservative: max error "
            f"{np.max(np.abs(column_sum - 1.0))}"
        )
    return operator, retained


def _apply_axis(
    array: np.ndarray, operator: sparse.spmatrix, axis: int
) -> np.ndarray:
    moved = np.moveaxis(array, axis, 0)
    transformed = operator @ moved.reshape(moved.shape[0], -1)
    reshaped = np.asarray(transformed).reshape(
        (operator.shape[0],) + moved.shape[1:]
    )
    return np.moveaxis(reshaped, 0, axis)


def forward_remap(
    array: np.ndarray,
    operators: tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix],
) -> np.ndarray:
    """Apply the separable conservative forward remap."""

    result = np.asarray(array)
    for axis, operator in enumerate(operators):
        result = _apply_axis(result, operator, axis)
    return result


def transpose_remap(
    array: np.ndarray,
    operators: tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix],
) -> np.ndarray:
    """Apply the exact algebraic transpose of :func:`forward_remap`."""

    result = np.asarray(array)
    for axis in reversed(range(3)):
        result = _apply_axis(result, operators[axis].transpose().tocsr(), axis)
    return result
