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


@dataclass(frozen=True)
class NodalDensityRemap1D:
    """Mass-conservative nodal density map along one array axis."""

    density_operator: sparse.csr_matrix
    source_weight_m: np.ndarray
    target_weight_m: np.ndarray
    periodic: bool
    period_m: float | None

    def apply_axis(self, values: np.ndarray, axis: int) -> np.ndarray:
        moved = np.moveaxis(np.asarray(values, float), axis, 0)
        if moved.shape[0] != self.density_operator.shape[1]:
            raise ValueError("source data axis does not match nodal remap")
        mapped = self.density_operator @ moved.reshape(moved.shape[0], -1)
        shape = (self.density_operator.shape[0], *moved.shape[1:])
        return np.moveaxis(np.asarray(mapped).reshape(shape), 0, axis)

    def transpose_axis(self, sensitivity: np.ndarray, axis: int) -> np.ndarray:
        moved = np.moveaxis(np.asarray(sensitivity, float), axis, 0)
        if moved.shape[0] != self.density_operator.shape[0]:
            raise ValueError("target sensitivity axis does not match nodal remap")
        mapped = self.density_operator.T @ moved.reshape(moved.shape[0], -1)
        shape = (self.density_operator.shape[1], *moved.shape[1:])
        return np.moveaxis(np.asarray(mapped).reshape(shape), 0, axis)


def _trapezoid_weights(coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinates, float).reshape(-1)
    weights = np.empty_like(values)
    weights[0] = 0.5 * (values[1] - values[0])
    weights[-1] = 0.5 * (values[-1] - values[-2])
    weights[1:-1] = 0.5 * (values[2:] - values[:-2])
    return weights


def build_nodal_density_remap_1d(
    *,
    source_coordinates_m: np.ndarray,
    target_coordinates_m: np.ndarray,
    periodic: bool,
) -> NodalDensityRemap1D:
    """Build the sparse form of the validated nodal mass-deposition rule."""
    source = _edges(source_coordinates_m, "source coordinates")
    target = _edges(target_coordinates_m, "target coordinates")
    source_weight = _trapezoid_weights(source)
    target_weight = _trapezoid_weights(target)
    locations = np.array(source, copy=True)
    period = None
    if periodic:
        period = float(target[-1] - target[0])
        if not np.isclose(
            source[-1] - source[0], period, rtol=1e-12, atol=1e-15
        ):
            raise ValueError("periodic source and target periods differ")
        locations = (locations - target[0]) % period + target[0]
    elif (
        np.min(locations) < target[0] - 1e-15
        or np.max(locations) > target[-1] + 1e-15
    ):
        raise ValueError("nonperiodic nodal remap would crop the source")
    else:
        locations = np.clip(locations, target[0], target[-1])

    upper = np.searchsorted(target, locations, side="right")
    upper = np.clip(upper, 1, target.size - 1)
    lower = upper - 1
    fraction = (locations - target[lower]) / (
        target[upper] - target[lower]
    )
    rows = np.concatenate([lower, upper])
    columns = np.concatenate(
        [np.arange(source.size), np.arange(source.size)]
    )
    deposited_mass = np.concatenate(
        [source_weight * (1.0 - fraction), source_weight * fraction]
    )
    operator = sparse.coo_matrix(
        (
            deposited_mass
            / np.concatenate([target_weight[lower], target_weight[upper]]),
            (rows, columns),
        ),
        shape=(target.size, source.size),
    ).tocsr()
    column_mass = np.asarray(
        target_weight @ operator
    ).reshape(-1)
    if not np.allclose(
        column_mass, source_weight, rtol=2e-13, atol=1e-30
    ):
        raise RuntimeError("nodal remap does not conserve each source mass")
    return NodalDensityRemap1D(
        density_operator=operator,
        source_weight_m=source_weight,
        target_weight_m=target_weight,
        periodic=bool(periodic),
        period_m=period,
    )


def nodal_control_volume_edges(coordinates_m: np.ndarray) -> np.ndarray:
    """Edges whose widths equal trapezoid weights at the input nodes."""
    coordinates = _edges(coordinates_m, "nodal coordinates")
    return np.concatenate(
        [
            coordinates[:1],
            0.5 * (coordinates[:-1] + coordinates[1:]),
            coordinates[-1:],
        ]
    )


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
