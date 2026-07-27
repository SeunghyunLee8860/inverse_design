"""Conservative embedding of finite native-Yee Q into a thermal grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


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
