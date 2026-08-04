"""Exact discrete local-PTE functional and transpose source.

The forward functional and ``c_T = dF/dT`` are built from the same sparse
derivative matrices. No continuous re-discretization is used for the adjoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


SIGMA_A_S_M = 4.91e5
SIGMA_B_S_M = 1.10e5
SEEBECK_A_V_K = -6.0e-6
SEEBECK_B_V_K = 27.0e-6


def _centres(edges: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(edges, float).reshape(-1)
    if values.size < 2 or not np.all(np.diff(values) > 0.0):
        raise ValueError(f"{name} edges must be strictly increasing")
    return 0.5 * (values[:-1] + values[1:])


def _derivative_weights(x0: float, samples: np.ndarray) -> np.ndarray:
    """Polynomial-exact first-derivative weights at ``x0``."""
    points = np.asarray(samples, float).reshape(-1)
    offsets = points - float(x0)
    if points.size < 2 or np.unique(points).size != points.size:
        raise ValueError("derivative stencil needs distinct samples")
    matrix = np.vstack([offsets**order for order in range(points.size)])
    rhs = np.zeros(points.size, float)
    rhs[1] = 1.0
    return np.linalg.solve(matrix, rhs)


def _axis_stencil(
    *,
    index: int,
    coordinates: np.ndarray,
    selected_line: np.ndarray,
    periodic: bool,
    period: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.flatnonzero(selected_line)
    if index not in selected:
        raise ValueError("requested derivative row is outside the FOM mask")
    if selected.size < 3:
        raise ValueError("each derivative line needs at least three FOM cells")
    if periodic:
        if selected.size != coordinates.size or period is None:
            raise ValueError(
                "periodic PTE derivatives require a complete selected axis"
            )
        n = coordinates.size
        neighbours = np.asarray([(index - 1) % n, index, (index + 1) % n])
        samples = coordinates[neighbours].astype(float, copy=True)
        if neighbours[0] > index:
            samples[0] -= period
        if neighbours[2] < index:
            samples[2] += period
        return neighbours, _derivative_weights(coordinates[index], samples)

    position = int(np.flatnonzero(selected == index)[0])
    if position == 0:
        neighbours = selected[:3]
    elif position == selected.size - 1:
        neighbours = selected[-3:]
    else:
        neighbours = selected[position - 1 : position + 2]
    return neighbours, _derivative_weights(
        coordinates[index], coordinates[neighbours]
    )


def _derivative_matrix(
    *,
    axis: int,
    selected_cells: np.ndarray,
    fom_mask: np.ndarray,
    active_ids: np.ndarray,
    coordinates: np.ndarray,
    periodic: bool,
    period: float | None,
) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, cell in enumerate(selected_cells):
        i, j, k = (int(value) for value in cell)
        if axis == 0:
            line = fom_mask[:, j, k]
            indices, weights = _axis_stencil(
                index=i,
                coordinates=coordinates,
                selected_line=line,
                periodic=periodic,
                period=period,
            )
            cells = [(int(index), j, k) for index in indices]
        elif axis == 1:
            line = fom_mask[i, :, k]
            indices, weights = _axis_stencil(
                index=j,
                coordinates=coordinates,
                selected_line=line,
                periodic=periodic,
                period=period,
            )
            cells = [(i, int(index), k) for index in indices]
        else:
            raise ValueError("local PTE supports only x/y derivatives")
        ids = np.asarray(
            [active_ids[cell_index] for cell_index in cells], dtype=np.int64
        )
        if np.any(ids < 0):
            raise ValueError("PTE derivative stencil enters an inactive cell")
        rows.extend([row] * ids.size)
        columns.extend(ids.tolist())
        values.extend(np.asarray(weights, float).tolist())
    return sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(selected_cells.shape[0], int(np.max(active_ids)) + 1),
    ).tocsr()


@dataclass(frozen=True)
class LocalPTEFunctional:
    """One linear signed local-PTE surrogate on active temperatures."""

    derivative_x_m_inv: sparse.csr_matrix
    derivative_y_m_inv: sparse.csr_matrix
    integration_volume_m3: np.ndarray
    selected_cells: np.ndarray
    temperature_source_A_m_K: np.ndarray
    sigma_a_S_m: float
    sigma_b_S_m: float
    seebeck_a_V_K: float
    seebeck_b_V_K: float
    periodic_axes: tuple[str, ...]

    def evaluate_active(self, temperature_active_K: np.ndarray) -> float:
        temperature = np.asarray(temperature_active_K, float).reshape(-1)
        if temperature.size != self.temperature_source_A_m_K.size:
            raise ValueError("active temperature has the wrong length")
        dtx = self.derivative_x_m_inv @ temperature
        dty = self.derivative_y_m_inv @ temperature
        local_current_A_m2 = (
            self.sigma_a_S_m * self.seebeck_a_V_K * dtx
            + self.sigma_b_S_m * self.seebeck_b_V_K * dty
        )
        return float(
            -np.dot(self.integration_volume_m3, local_current_A_m2)
            / np.sqrt(2.0)
        )


def build_local_pte_functional(
    *,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    z_edges_m: np.ndarray,
    active_mask: np.ndarray,
    active_ids: np.ndarray,
    fom_mask: np.ndarray,
    periodic_axes: tuple[str, ...] = (),
    sigma_a_S_m: float = SIGMA_A_S_M,
    sigma_b_S_m: float = SIGMA_B_S_M,
    seebeck_a_V_K: float = SEEBECK_A_V_K,
    seebeck_b_V_K: float = SEEBECK_B_V_K,
) -> LocalPTEFunctional:
    """Build the forward functional and its literal sparse transpose."""
    x = _centres(x_edges_m, "x")
    y = _centres(y_edges_m, "y")
    z = _centres(z_edges_m, "z")
    shape = (x.size, y.size, z.size)
    active = np.asarray(active_mask, bool)
    ids = np.asarray(active_ids, np.int64)
    mask = np.asarray(fom_mask, bool)
    if active.shape != shape or ids.shape != shape or mask.shape != shape:
        raise ValueError("active/id/FOM masks must match the grid shape")
    if np.any(mask & ~active):
        raise ValueError("FOM mask contains inactive cells")
    selected = np.argwhere(mask)
    if selected.size == 0:
        raise ValueError("FOM mask is empty")
    periodic = tuple(dict.fromkeys(str(axis) for axis in periodic_axes))
    if set(periodic) - {"x", "y"}:
        raise ValueError("local PTE periodic axes may only be x and/or y")
    dx = sparse.csr_matrix(
        _derivative_matrix(
            axis=0,
            selected_cells=selected,
            fom_mask=mask,
            active_ids=ids,
            coordinates=x,
            periodic="x" in periodic,
            period=float(np.asarray(x_edges_m)[-1] - np.asarray(x_edges_m)[0]),
        )
    )
    dy = sparse.csr_matrix(
        _derivative_matrix(
            axis=1,
            selected_cells=selected,
            fom_mask=mask,
            active_ids=ids,
            coordinates=y,
            periodic="y" in periodic,
            period=float(np.asarray(y_edges_m)[-1] - np.asarray(y_edges_m)[0]),
        )
    )
    widths = (
        np.diff(np.asarray(x_edges_m, float)),
        np.diff(np.asarray(y_edges_m, float)),
        np.diff(np.asarray(z_edges_m, float)),
    )
    volume = (
        widths[0][:, None, None]
        * widths[1][None, :, None]
        * widths[2][None, None, :]
    )
    integration_volume = volume[mask].reshape(-1)
    c_t = -(
        sigma_a_S_m
        * seebeck_a_V_K
        * (dx.T @ integration_volume)
        + sigma_b_S_m
        * seebeck_b_V_K
        * (dy.T @ integration_volume)
    ) / np.sqrt(2.0)
    return LocalPTEFunctional(
        derivative_x_m_inv=dx,
        derivative_y_m_inv=dy,
        integration_volume_m3=integration_volume,
        selected_cells=selected,
        temperature_source_A_m_K=np.asarray(c_t, float).reshape(-1),
        sigma_a_S_m=float(sigma_a_S_m),
        sigma_b_S_m=float(sigma_b_S_m),
        seebeck_a_V_K=float(seebeck_a_V_K),
        seebeck_b_V_K=float(seebeck_b_V_K),
        periodic_axes=periodic,
    )

