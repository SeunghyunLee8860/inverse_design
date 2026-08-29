"""Linear local-device to global-crystal scalar-field rotation map."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import (
    TaIrTe4FlakeContract,
)


@dataclass(frozen=True)
class RotatedScalarMap:
    operator: sparse.csr_matrix
    source_shape: tuple[int, int, int]
    target_shape: tuple[int, int, int]
    target_flat_indices: np.ndarray

    def apply(self, source: np.ndarray) -> np.ndarray:
        values = np.asarray(source, dtype=np.float64)
        if values.shape != self.source_shape:
            raise ValueError(f"source shape {values.shape} != {self.source_shape}")
        target = np.zeros(int(np.prod(self.target_shape)), dtype=np.float64)
        target[self.target_flat_indices] = self.operator @ values.ravel()
        return target.reshape(self.target_shape)

    def transpose(self, target: np.ndarray) -> np.ndarray:
        values = np.asarray(target, dtype=np.float64)
        if values.shape != self.target_shape:
            raise ValueError(f"target shape {values.shape} != {self.target_shape}")
        selected = values.ravel()[self.target_flat_indices]
        return np.asarray(self.operator.T @ selected).reshape(self.source_shape)


def _interval_and_fraction(
    coordinate: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    points = np.asarray(query, dtype=np.float64).reshape(-1)
    if nodes.size < 2 or not np.all(np.diff(nodes) > 0.0):
        raise ValueError("source coordinate must be strictly increasing")
    lower = np.searchsorted(nodes, points, side="right") - 1
    valid = (points >= nodes[0]) & (points <= nodes[-1])
    lower = np.clip(lower, 0, nodes.size - 2)
    fraction = (points - nodes[lower]) / (nodes[lower + 1] - nodes[lower])
    fraction = np.clip(fraction, 0.0, 1.0)
    return lower, fraction, valid


def build_rotated_scalar_map(
    source_coordinates_m: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_centers_m: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_support_mask: np.ndarray,
) -> RotatedScalarMap:
    """Build trilinear Q(u,v,z) -> Q(x,y,z) for a +45-degree device."""

    source = tuple(np.asarray(value, dtype=np.float64).reshape(-1) for value in source_coordinates_m)
    target = tuple(np.asarray(value, dtype=np.float64).reshape(-1) for value in target_centers_m)
    target_shape = tuple(value.size for value in target)
    support = np.asarray(target_support_mask, dtype=bool)
    if support.shape != target_shape:
        raise ValueError(f"support shape {support.shape} != {target_shape}")
    selected = np.flatnonzero(support.ravel())
    ix, iy, iz = np.unravel_index(selected, target_shape)
    x = target[0][ix]
    y = target[1][iy]
    u, v = TaIrTe4FlakeContract.rotated_uv(x, y)
    queries = (u, v, target[2][iz])
    interval = [_interval_and_fraction(nodes, query) for nodes, query in zip(source, queries)]
    valid = interval[0][2] & interval[1][2] & interval[2][2]
    selected = selected[valid]
    lower = [value[0][valid] for value in interval]
    fraction = [value[1][valid] for value in interval]
    rows = []
    columns = []
    data = []
    source_shape = tuple(value.size for value in source)
    for ox in (0, 1):
        wx = fraction[0] if ox else 1.0 - fraction[0]
        for oy in (0, 1):
            wy = fraction[1] if oy else 1.0 - fraction[1]
            for oz in (0, 1):
                wz = fraction[2] if oz else 1.0 - fraction[2]
                rows.append(np.arange(selected.size, dtype=np.int64))
                columns.append(
                    np.ravel_multi_index(
                        (lower[0] + ox, lower[1] + oy, lower[2] + oz),
                        source_shape,
                    )
                )
                data.append(wx * wy * wz)
    operator = sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
        shape=(selected.size, int(np.prod(source_shape))),
    )
    return RotatedScalarMap(operator, source_shape, target_shape, selected)


def make_control_volume_conservative(
    mapping: RotatedScalarMap,
    source_volume_m3: np.ndarray,
    target_volume_m3: np.ndarray,
) -> tuple[RotatedScalarMap, np.ndarray]:
    """Scale source columns so every represented control-volume power is conserved."""

    source_volume = np.asarray(source_volume_m3, dtype=np.float64)
    target_volume = np.asarray(target_volume_m3, dtype=np.float64)
    if source_volume.shape != mapping.source_shape:
        raise ValueError("source control-volume shape does not match rotation map")
    if target_volume.shape != mapping.target_shape:
        raise ValueError("target control-volume shape does not match rotation map")
    selected_volume = target_volume.ravel()[mapping.target_flat_indices]
    represented_volume = np.asarray(
        mapping.operator.T @ selected_volume, dtype=np.float64
    ).reshape(-1)
    desired_volume = source_volume.reshape(-1)
    represented = represented_volume > 0.0
    scale = np.zeros_like(represented_volume)
    scale[represented] = desired_volume[represented] / represented_volume[represented]
    conservative = mapping.operator @ sparse.diags(scale, format="csr")
    return RotatedScalarMap(
        conservative.tocsr(),
        mapping.source_shape,
        mapping.target_shape,
        mapping.target_flat_indices,
    ), represented.reshape(mapping.source_shape)
