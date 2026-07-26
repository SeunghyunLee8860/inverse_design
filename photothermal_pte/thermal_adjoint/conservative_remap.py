"""Exact conservative Cartesian cell-density transfer and transpose."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


def _edges(values: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(values, float).reshape(-1)
    if out.size < 2 or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} edges must be finite")
    if not np.all(np.diff(out) > 0.0):
        raise ValueError(f"{name} edges must be strictly increasing")
    return out


def _overlap_1d(target: np.ndarray, source: np.ndarray) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    source_index = 0
    for target_index in range(target.size - 1):
        lower, upper = target[target_index : target_index + 2]
        while source_index < source.size - 1 and source[source_index + 1] <= lower:
            source_index += 1
        index = source_index
        while index < source.size - 1 and source[index] < upper:
            overlap = min(upper, source[index + 1]) - max(lower, source[index])
            if overlap > 0.0:
                rows.append(target_index)
                columns.append(index)
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
class ConservativeDensityRemap:
    """Cell-average density map ``q_target = R q_source``."""

    density_operator: sparse.csr_matrix
    overlap_volume_operator_m3: sparse.csr_matrix
    source_volume_m3: np.ndarray
    target_volume_m3: np.ndarray
    source_shape: tuple[int, int, int]
    target_shape: tuple[int, int, int]

    def apply(self, source_density: np.ndarray) -> np.ndarray:
        source = np.asarray(source_density, float)
        if source.shape != self.source_shape:
            raise ValueError(f"source shape {source.shape} != {self.source_shape}")
        return np.asarray(
            self.density_operator @ source.reshape(-1)
        ).reshape(self.target_shape)

    def transpose(self, target_density_sensitivity: np.ndarray) -> np.ndarray:
        target = np.asarray(target_density_sensitivity, float)
        if target.shape != self.target_shape:
            raise ValueError(f"target shape {target.shape} != {self.target_shape}")
        return np.asarray(
            self.density_operator.T @ target.reshape(-1)
        ).reshape(self.source_shape)

    def power_source(self, source_density: np.ndarray) -> float:
        return float(np.sum(self.source_volume_m3 * source_density))

    def power_target(self, target_density: np.ndarray) -> float:
        return float(np.sum(self.target_volume_m3 * target_density))


def build_conservative_density_remap(
    *,
    source_edges_m: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_edges_m: tuple[np.ndarray, np.ndarray, np.ndarray],
    bounds_tolerance_m: float = 1e-15,
) -> ConservativeDensityRemap:
    """Build a no-gain overlap map for equal Cartesian bounds.

    Equal bounds are required so the operator cannot silently crop or pad the
    optical source.  Periodic tiling and global source rescaling are absent.
    """
    source = tuple(
        _edges(axis, f"source {name}")
        for axis, name in zip(source_edges_m, "xyz")
    )
    target = tuple(
        _edges(axis, f"target {name}")
        for axis, name in zip(target_edges_m, "xyz")
    )
    for name, source_axis, target_axis in zip("xyz", source, target):
        if not np.allclose(
            source_axis[[0, -1]],
            target_axis[[0, -1]],
            rtol=0.0,
            atol=bounds_tolerance_m,
        ):
            raise ValueError(
                f"{name} bounds differ; fail closed rather than crop/pad/tile"
            )
    overlaps = [
        _overlap_1d(target_axis, source_axis)
        for target_axis, source_axis in zip(target, source)
    ]
    overlap_3d = sparse.kron(
        overlaps[0], sparse.kron(overlaps[1], overlaps[2]), format="csr"
    )
    source_volume = _volumes(source)
    target_volume = _volumes(target)
    density_operator = (
        sparse.diags(1.0 / target_volume.reshape(-1), format="csr")
        @ overlap_3d
    )
    covered_source = np.asarray(overlap_3d.sum(axis=0)).reshape(-1)
    covered_target = np.asarray(overlap_3d.sum(axis=1)).reshape(-1)
    if not np.allclose(
        covered_source,
        source_volume.reshape(-1),
        rtol=1e-12,
        atol=1e-30,
    ):
        raise RuntimeError("source cells are not covered exactly once")
    if not np.allclose(
        covered_target,
        target_volume.reshape(-1),
        rtol=1e-12,
        atol=1e-30,
    ):
        raise RuntimeError("target cells are not filled exactly once")
    return ConservativeDensityRemap(
        density_operator=density_operator,
        overlap_volume_operator_m3=overlap_3d,
        source_volume_m3=source_volume,
        target_volume_m3=target_volume,
        source_shape=source_volume.shape,
        target_shape=target_volume.shape,
    )
