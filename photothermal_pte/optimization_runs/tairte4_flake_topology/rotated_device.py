"""Linear maps between the local device grid and fixed crystal coordinates."""

from __future__ import annotations

from functools import lru_cache
from math import sqrt

import numpy as np
from scipy import sparse

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import (
    CONTRACT,
)


ROTATION_DEG = 45.0


def device_to_crystal_coordinates(
    u_m: np.ndarray, v_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map local device coordinates to global crystal ``x=b, y=a``."""

    u = np.asarray(u_m, dtype=np.float64)
    v = np.asarray(v_m, dtype=np.float64)
    return (u - v) / sqrt(2.0), (u + v) / sqrt(2.0)


def crystal_to_device_coordinates(
    x_m: np.ndarray, y_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return (x_m + y_m) / sqrt(2.0), (-x_m + y_m) / sqrt(2.0)


def device_nodes_m() -> tuple[np.ndarray, np.ndarray]:
    return tuple(
        np.linspace(*CONTRACT.design_bounds_m[axis], CONTRACT.design_node_shape[index])
        for index, axis in enumerate("xy")
    )


def crystal_nodes_m() -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * CONTRACT.crystal_bounding_span_m
    count = CONTRACT.crystal_bounding_node_shape[0]
    values = np.linspace(-half, half, count)
    return values, values.copy()


def _bilinear_operator(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    *,
    active: np.ndarray,
) -> sparse.csr_matrix:
    """Return row-major bilinear sampling from a uniform source grid."""

    sx = np.asarray(source_x, dtype=np.float64)
    sy = np.asarray(source_y, dtype=np.float64)
    tx = np.asarray(target_x, dtype=np.float64).reshape(-1)
    ty = np.asarray(target_y, dtype=np.float64).reshape(-1)
    selected = np.flatnonzero(np.asarray(active, dtype=bool).reshape(-1))
    dx = float(sx[1] - sx[0])
    dy = float(sy[1] - sy[0])
    fx = np.clip((tx[selected] - sx[0]) / dx, 0.0, sx.size - 1.0)
    fy = np.clip((ty[selected] - sy[0]) / dy, 0.0, sy.size - 1.0)
    ix0 = np.minimum(np.floor(fx).astype(np.int64), sx.size - 2)
    iy0 = np.minimum(np.floor(fy).astype(np.int64), sy.size - 2)
    wx = fx - ix0
    wy = fy - iy0
    rows = np.repeat(selected, 4)
    columns = np.column_stack(
        (
            ix0 * sy.size + iy0,
            (ix0 + 1) * sy.size + iy0,
            ix0 * sy.size + iy0 + 1,
            (ix0 + 1) * sy.size + iy0 + 1,
        )
    ).reshape(-1)
    weights = np.column_stack(
        (
            (1.0 - wx) * (1.0 - wy),
            wx * (1.0 - wy),
            (1.0 - wx) * wy,
            wx * wy,
        )
    ).reshape(-1)
    return sparse.coo_matrix(
        (weights, (rows, columns)),
        shape=(tx.size, sx.size * sy.size),
    ).tocsr()


@lru_cache(maxsize=1)
def device_to_crystal_operator() -> sparse.csr_matrix:
    """Sample a local 24 um device field onto the global diamond box."""

    u, v = device_nodes_m()
    x, y = crystal_nodes_m()
    xx, yy = np.meshgrid(x, y, indexing="ij")
    uu, vv = crystal_to_device_coordinates(xx, yy)
    half = 0.5 * CONTRACT.flake_span_m
    active = (np.abs(uu) <= half + 1.0e-18) & (np.abs(vv) <= half + 1.0e-18)
    return _bilinear_operator(u, v, uu, vv, active=active)


@lru_cache(maxsize=1)
def crystal_to_device_operator() -> sparse.csr_matrix:
    """Sample a global crystal-grid field at the rotated device nodes."""

    x, y = crystal_nodes_m()
    u, v = device_nodes_m()
    uu, vv = np.meshgrid(u, v, indexing="ij")
    xx, yy = device_to_crystal_coordinates(uu, vv)
    return _bilinear_operator(
        x,
        y,
        xx,
        yy,
        active=np.ones_like(xx, dtype=bool),
    )


def device_to_crystal_field(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    if source.shape != CONTRACT.design_node_shape:
        raise ValueError(f"device field shape {source.shape} != {CONTRACT.design_node_shape}")
    return np.asarray(device_to_crystal_operator() @ source.reshape(-1)).reshape(
        CONTRACT.crystal_bounding_node_shape
    )


def crystal_to_device_field(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    if source.shape != CONTRACT.crystal_bounding_node_shape:
        raise ValueError(
            f"crystal field shape {source.shape} != {CONTRACT.crystal_bounding_node_shape}"
        )
    return np.asarray(crystal_to_device_operator() @ source.reshape(-1)).reshape(
        CONTRACT.design_node_shape
    )


def device_to_crystal_transpose(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    if source.shape != CONTRACT.crystal_bounding_node_shape:
        raise ValueError("crystal sensitivity has the wrong shape")
    return np.asarray(device_to_crystal_operator().T @ source.reshape(-1)).reshape(
        CONTRACT.design_node_shape
    )


def crystal_to_device_transpose(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    if source.shape != CONTRACT.design_node_shape:
        raise ValueError("device sensitivity has the wrong shape")
    return np.asarray(crystal_to_device_operator().T @ source.reshape(-1)).reshape(
        CONTRACT.crystal_bounding_node_shape
    )
