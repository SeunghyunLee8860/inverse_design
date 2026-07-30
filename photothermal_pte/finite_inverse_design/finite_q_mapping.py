"""Conservative embedding of finite native-Yee Q into a thermal grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def nodal_control_volume_edges(coordinates_m: np.ndarray) -> np.ndarray:
    """Edges whose widths are the non-periodic trapezoid weights."""

    coordinate = np.asarray(coordinates_m, float).reshape(-1)
    if coordinate.size < 2 or np.any(np.diff(coordinate) <= 0.0):
        raise ValueError("coordinates must be strictly increasing")
    return np.concatenate(
        (
            coordinate[:1],
            0.5 * (coordinate[:-1] + coordinate[1:]),
            coordinate[-1:],
        )
    )


def _overlap_1d(target: np.ndarray, source: np.ndarray) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    target_index = 0
    for source_index in range(source.size - 1):
        lower, upper = source[source_index : source_index + 2]
        while (
            target_index < target.size - 1
            and target[target_index + 1] <= lower
        ):
            target_index += 1
        index = target_index
        while index < target.size - 1 and target[index] < upper:
            overlap = min(upper, target[index + 1]) - max(
                lower, target[index]
            )
            if overlap > 0.0:
                rows.append(index)
                columns.append(source_index)
                values.append(float(overlap))
            index += 1
    return sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(target.size - 1, source.size - 1),
    ).tocsr()


def _volumes(edges: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    dx, dy, dz = (np.diff(axis) for axis in edges)
    return dx[:, None, None] * dy[None, :, None] * dz[None, None, :]


@dataclass(frozen=True)
class ConservativeEmbeddingRemap:
    """Density map from a contained source box into a larger target box."""

    density_operator: sparse.csr_matrix
    source_volume_m3: np.ndarray
    target_volume_m3: np.ndarray
    source_shape: tuple[int, int, int]
    target_shape: tuple[int, int, int]

    def apply(self, source_density: np.ndarray) -> np.ndarray:
        source = np.asarray(source_density, float)
        if source.shape != self.source_shape:
            raise ValueError(
                f"source shape {source.shape} != {self.source_shape}"
            )
        return np.asarray(
            self.density_operator @ source.reshape(-1)
        ).reshape(self.target_shape)

    def transpose(self, target_sensitivity: np.ndarray) -> np.ndarray:
        target = np.asarray(target_sensitivity, float)
        if target.shape != self.target_shape:
            raise ValueError(
                f"target shape {target.shape} != {self.target_shape}"
            )
        return np.asarray(
            self.density_operator.T @ target.reshape(-1)
        ).reshape(self.source_shape)

    def power_source(self, source_density: np.ndarray) -> float:
        return float(np.sum(self.source_volume_m3 * source_density))

    def power_target(self, target_density: np.ndarray) -> float:
        return float(np.sum(self.target_volume_m3 * target_density))


def build_conservative_embedding_remap(
    *,
    source_edges_m: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_edges_m: tuple[np.ndarray, np.ndarray, np.ndarray],
    bounds_tolerance_m: float = 1.0e-15,
) -> ConservativeEmbeddingRemap:
    """Build a no-gain map when the target fully contains the source.

    Target-only cells are exactly zero after ``apply``.  Source cropping,
    periodic tiling, smoothing, gain, and global rescaling are absent.
    """

    source = tuple(
        np.asarray(axis, float).reshape(-1) for axis in source_edges_m
    )
    target = tuple(
        np.asarray(axis, float).reshape(-1) for axis in target_edges_m
    )
    for label, axes in (("source", source), ("target", target)):
        for axis, name in zip(axes, "xyz"):
            if (
                axis.size < 2
                or not np.all(np.isfinite(axis))
                or np.any(np.diff(axis) <= 0.0)
            ):
                raise ValueError(f"invalid {label} {name} edges")
    for name, source_axis, target_axis in zip("xyz", source, target):
        if (
            source_axis[0] < target_axis[0] - bounds_tolerance_m
            or source_axis[-1] > target_axis[-1] + bounds_tolerance_m
        ):
            raise ValueError(
                f"{name} source is not fully contained; fail closed"
            )
    overlaps = [
        _overlap_1d(target_axis, source_axis)
        for target_axis, source_axis in zip(target, source)
    ]
    overlap_3d = sparse.kron(
        overlaps[0],
        sparse.kron(overlaps[1], overlaps[2]),
        format="csr",
    )
    source_volume = _volumes(source)
    target_volume = _volumes(target)
    covered_source = np.asarray(overlap_3d.sum(axis=0)).reshape(-1)
    if not np.allclose(
        covered_source,
        source_volume.reshape(-1),
        rtol=2.0e-13,
        atol=1.0e-30,
    ):
        raise RuntimeError("source cells are not covered exactly once")
    density_operator = (
        sparse.diags(
            1.0 / target_volume.reshape(-1), format="csr"
        )
        @ overlap_3d
    )
    return ConservativeEmbeddingRemap(
        density_operator=density_operator,
        source_volume_m3=source_volume,
        target_volume_m3=target_volume,
        source_shape=source_volume.shape,
        target_shape=target_volume.shape,
    )


def project_remap_to_material_support_along_axis(
    remap: ConservativeEmbeddingRemap,
    *,
    target_edges_m: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_support_mask: np.ndarray,
    axis: int = 2,
) -> ConservativeEmbeddingRemap:
    """Conservatively relocate target energy to material-support cells.

    A native staggered Yee control volume may straddle a material boundary.
    Its integrated absorption belongs to the lossy material even when the
    unconstrained geometric embedding overlaps a neighboring thermal cell.
    This operator moves each target-cell energy to the nearest supported cell
    along ``axis`` while preserving every source-cell energy exactly.

    The operation is linear, contains no empirical or global rescaling, and
    returns the literal density operator so ``transpose`` remains exact.
    Columns with no supported cell are left unchanged; callers must fail
    closed if a nonzero source reaches such a column.
    """

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    target_edges = tuple(
        np.asarray(values, float).reshape(-1)
        for values in target_edges_m
    )
    target_shape = tuple(values.size - 1 for values in target_edges)
    if target_shape != remap.target_shape:
        raise ValueError(
            f"target edge shape {target_shape} != {remap.target_shape}"
        )
    support = np.asarray(target_support_mask, bool)
    if support.shape != remap.target_shape:
        raise ValueError(
            f"support shape {support.shape} != {remap.target_shape}"
        )
    centers = 0.5 * (
        target_edges[axis][:-1] + target_edges[axis][1:]
    )
    destination = np.arange(support.size, dtype=np.int64).reshape(
        support.shape
    )
    transverse_shape = tuple(
        size for index, size in enumerate(support.shape) if index != axis
    )
    for transverse_index in np.ndindex(transverse_shape):
        selector: list[object] = []
        cursor = iter(transverse_index)
        for index in range(3):
            selector.append(slice(None) if index == axis else next(cursor))
        line_selector = tuple(selector)
        supported_indices = np.flatnonzero(support[line_selector])
        if supported_indices.size == 0:
            continue
        nearest = supported_indices[
            np.argmin(
                np.abs(
                    centers[:, None] - centers[supported_indices][None, :]
                ),
                axis=1,
            )
        ]
        line_destinations = destination[line_selector]
        for source_index, destination_index in enumerate(nearest):
            destination_full = list(selector)
            destination_full[axis] = int(destination_index)
            line_destinations[source_index] = np.ravel_multi_index(
                tuple(destination_full), support.shape
            )

    source_target_ids = np.arange(support.size, dtype=np.int64)
    energy_projection = sparse.coo_matrix(
        (
            np.ones(support.size, float),
            (destination.reshape(-1), source_target_ids),
        ),
        shape=(support.size, support.size),
    ).tocsr()
    target_volume = remap.target_volume_m3.reshape(-1)
    density_projection = (
        sparse.diags(1.0 / target_volume, format="csr")
        @ energy_projection
        @ sparse.diags(target_volume, format="csr")
    )
    return ConservativeEmbeddingRemap(
        density_operator=(
            density_projection @ remap.density_operator
        ).tocsr(),
        source_volume_m3=remap.source_volume_m3,
        target_volume_m3=remap.target_volume_m3,
        source_shape=remap.source_shape,
        target_shape=remap.target_shape,
    )


def project_remap_to_nearest_material_support(
    remap: ConservativeEmbeddingRemap,
    *,
    target_edges_m: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_support_mask: np.ndarray,
    nearest_neighbours: int = 16,
) -> ConservativeEmbeddingRemap:
    """Relocate target energy to nearest support in physical 3D distance.

    Unlike repeated axis-wise projection, this construction has no preferred
    coordinate-axis order.  Only target rows reached by ``remap`` are
    considered, so the sparse operator remains practical on a large thermal
    grid.  Exact nearest-distance ties are split uniformly; this preserves
    reflection/rotation symmetry on a Cartesian grid instead of resolving a
    tie with a hidden x/y preference.

    The returned operator is still literal and linear.  It preserves each
    source-column energy, supports an exact transpose, and performs no
    clipping, smoothing, gain, or global rescaling.
    """

    target_edges = tuple(
        np.asarray(values, float).reshape(-1)
        for values in target_edges_m
    )
    target_shape = tuple(values.size - 1 for values in target_edges)
    if target_shape != remap.target_shape:
        raise ValueError(
            f"target edge shape {target_shape} != {remap.target_shape}"
        )
    support = np.asarray(target_support_mask, bool)
    if support.shape != remap.target_shape:
        raise ValueError(
            f"support shape {support.shape} != {remap.target_shape}"
        )
    support_flat = np.flatnonzero(support.reshape(-1))
    if support_flat.size == 0:
        raise ValueError("material support is empty")
    if nearest_neighbours < 1:
        raise ValueError("nearest_neighbours must be positive")

    source_rows = np.unique(remap.density_operator.nonzero()[0])
    supported_rows = source_rows[support.reshape(-1)[source_rows]]
    outside_rows = source_rows[~support.reshape(-1)[source_rows]]

    projection_source = [supported_rows]
    projection_destination = [supported_rows]
    projection_weight = [np.ones(supported_rows.size, float)]
    if outside_rows.size:
        centers = tuple(
            0.5 * (axis[:-1] + axis[1:]) for axis in target_edges
        )

        def physical_points(flat_indices: np.ndarray) -> np.ndarray:
            indices = np.unravel_index(flat_indices, target_shape)
            return np.column_stack(
                [centers[axis][indices[axis]] for axis in range(3)]
            )

        support_points = physical_points(support_flat)
        query_points = physical_points(outside_rows)
        neighbours = min(int(nearest_neighbours), support_flat.size)
        distances, neighbour_indices = cKDTree(support_points).query(
            query_points,
            k=neighbours,
        )
        if neighbours == 1:
            distances = np.asarray(distances)[:, None]
            neighbour_indices = np.asarray(neighbour_indices)[:, None]
        else:
            distances = np.asarray(distances)
            neighbour_indices = np.asarray(neighbour_indices)
        minimum = distances[:, :1]
        # Coordinates are in metres.  The absolute term is far below any
        # production cell size but large enough to catch round-off in exact
        # diagonal-grid ties.
        tie_tolerance = np.maximum(1.0e-18, minimum * 1.0e-10)
        tied = distances <= minimum + tie_tolerance
        if (
            neighbours < support_flat.size
            and np.any(tied[:, -1])
        ):
            raise RuntimeError(
                "nearest-support tie reaches the candidate limit; increase "
                "nearest_neighbours so every exact physical-distance tie is "
                "split explicitly"
            )
        tie_count = np.sum(tied, axis=1)
        repeated_source = np.repeat(outside_rows, tie_count)
        selected_destination = support_flat[neighbour_indices[tied]]
        selected_weight = np.concatenate(
            [
                np.full(count, 1.0 / count, float)
                for count in tie_count
            ]
        )
        projection_source.append(repeated_source)
        projection_destination.append(selected_destination)
        projection_weight.append(selected_weight)

    source_row = np.concatenate(projection_source)
    destination_row = np.concatenate(projection_destination)
    energy_weight = np.concatenate(projection_weight)
    target_volume = remap.target_volume_m3.reshape(-1)
    density_weight = (
        energy_weight
        * target_volume[source_row]
        / target_volume[destination_row]
    )
    projection = sparse.coo_matrix(
        (density_weight, (destination_row, source_row)),
        shape=(support.size, support.size),
    ).tocsr()
    projected_operator = (
        projection @ remap.density_operator
    ).tocsr()
    return ConservativeEmbeddingRemap(
        density_operator=projected_operator,
        source_volume_m3=remap.source_volume_m3,
        target_volume_m3=remap.target_volume_m3,
        source_shape=remap.source_shape,
        target_shape=remap.target_shape,
    )


def exact_nonzero_box(
    density: np.ndarray,
) -> tuple[tuple[slice, slice, slice], int]:
    """Return the smallest box containing all exact nonzero entries."""

    value = np.asarray(density, float)
    if value.ndim != 3:
        raise ValueError("density must be three-dimensional")
    active = np.argwhere(value != 0.0)
    if active.size == 0:
        raise ValueError("density is identically zero")
    lower = np.min(active, axis=0)
    upper = np.max(active, axis=0) + 1
    box = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(lower, upper)
    )
    outside = np.array(value, copy=True)
    outside[box] = 0.0
    return box, int(np.count_nonzero(outside))
