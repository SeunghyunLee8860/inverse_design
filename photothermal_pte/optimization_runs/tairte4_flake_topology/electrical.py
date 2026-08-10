"""Density-dependent anisotropic weighting-potential FEM and implicit adjoint.

The electrical problem is a two-dimensional thin-sheet model.  It is exact
for a uniform-through-thickness TaIrTe4 sheet with top/bottom full-width
contacts.  The production optimization will solve the sparse systems on CUDA;
the small SciPy reference path exists only for deterministic analytic and
AD--FD tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


Array = np.ndarray
LinearSolve = Callable[[sparse.csr_matrix, Array], Array]


@dataclass(frozen=True)
class RectangularTriangularMesh:
    x_m: Array
    y_m: Array
    nodes_m: Array
    triangles: Array
    triangle_area_m2: Array
    gradients_m_inv: Array
    top_nodes: Array
    bottom_nodes: Array

    @property
    def shape(self) -> tuple[int, int]:
        return self.x_m.size, self.y_m.size


@dataclass(frozen=True)
class ElectricalResult:
    weighting_potential: Array
    weighting_gradient_element_m_inv: Array
    current_A: float
    gradient_temperature_K_inv: Array
    gradient_rho_A: Array
    weighting_residual: float
    adjoint_residual: float
    terminal_conductance_S: float
    gradient_terminal_conductance_S: Array


def build_rectangular_mesh(span_x_m: float, span_y_m: float, step_m: float) -> RectangularTriangularMesh:
    nx = int(round(span_x_m / step_m)) + 1
    ny = int(round(span_y_m / step_m)) + 1
    if not np.isclose((nx - 1) * step_m, span_x_m, rtol=0.0, atol=1e-18):
        raise ValueError("step must divide x span")
    if not np.isclose((ny - 1) * step_m, span_y_m, rtol=0.0, atol=1e-18):
        raise ValueError("step must divide y span")
    x = np.linspace(-0.5 * span_x_m, 0.5 * span_x_m, nx)
    y = np.linspace(-0.5 * span_y_m, 0.5 * span_y_m, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    nodes = np.column_stack((xx.ravel(), yy.ravel()))
    ids = np.arange(nx * ny, dtype=np.int64).reshape(nx, ny)
    lower = np.column_stack((ids[:-1, :-1].ravel(), ids[1:, :-1].ravel(), ids[1:, 1:].ravel()))
    upper = np.column_stack((ids[:-1, :-1].ravel(), ids[1:, 1:].ravel(), ids[:-1, 1:].ravel()))
    triangles = np.vstack((lower, upper))
    coordinates = nodes[triangles]
    twice_area = (
        (coordinates[:, 1, 0] - coordinates[:, 0, 0])
        * (coordinates[:, 2, 1] - coordinates[:, 0, 1])
        - (coordinates[:, 2, 0] - coordinates[:, 0, 0])
        * (coordinates[:, 1, 1] - coordinates[:, 0, 1])
    )
    if np.any(twice_area <= 0.0):
        raise RuntimeError("triangles must have positive orientation")
    gradients = np.empty((triangles.shape[0], 2, 3), dtype=np.float64)
    gradients[:, 0, :] = np.column_stack(
        (
            coordinates[:, 1, 1] - coordinates[:, 2, 1],
            coordinates[:, 2, 1] - coordinates[:, 0, 1],
            coordinates[:, 0, 1] - coordinates[:, 1, 1],
        )
    ) / twice_area[:, None]
    gradients[:, 1, :] = np.column_stack(
        (
            coordinates[:, 2, 0] - coordinates[:, 1, 0],
            coordinates[:, 0, 0] - coordinates[:, 2, 0],
            coordinates[:, 1, 0] - coordinates[:, 0, 0],
        )
    ) / twice_area[:, None]
    return RectangularTriangularMesh(
        x_m=x,
        y_m=y,
        nodes_m=nodes,
        triangles=triangles,
        triangle_area_m2=0.5 * twice_area,
        gradients_m_inv=gradients,
        top_nodes=ids[:, -1].copy(),
        bottom_nodes=ids[:, 0].copy(),
    )


def _element_tensor(values_xy: tuple[float, float], scale: Array) -> Array:
    output = np.zeros((scale.size, 2, 2), dtype=np.float64)
    output[:, 0, 0] = float(values_xy[0]) * scale
    output[:, 1, 1] = float(values_xy[1]) * scale
    return output


def _assemble(mesh: RectangularTriangularMesh, tensor: Array, thickness_m: float) -> sparse.csr_matrix:
    element = thickness_m * mesh.triangle_area_m2[:, None, None] * np.einsum(
        "eai,eab,ebj->eij", mesh.gradients_m_inv, tensor, mesh.gradients_m_inv
    )
    rows = np.repeat(mesh.triangles, 3, axis=1).ravel()
    columns = np.tile(mesh.triangles, (1, 3)).ravel()
    return sparse.coo_matrix(
        (element.ravel(), (rows, columns)),
        shape=(mesh.nodes_m.shape[0], mesh.nodes_m.shape[0]),
    ).tocsr()


def _reference_solve(matrix: sparse.csr_matrix, rhs: Array) -> Array:
    return np.asarray(spla.spsolve(matrix.tocsc(), rhs), dtype=np.float64)


def solve_weighting_and_adjoint(
    mesh: RectangularTriangularMesh,
    rho_nodal: Array,
    temperature_K: Array,
    *,
    thickness_m: float,
    sigma_xy_S_m: tuple[float, float],
    seebeck_xy_V_K: tuple[float, float],
    sigma_void_fraction: float = 1.0e-8,
    sigma_penalty: float = 2.0,
    alpha_penalty: float = 2.0,
    linear_solve: LinearSolve | None = None,
) -> ElectricalResult:
    """Return the PTE current and exact discrete rho/T derivatives.

    ``rho_nodal`` includes the fixed frame.  Production callers set frame
    nodes exactly to one and optimize only the enclosed design subset.
    """

    rho = np.asarray(rho_nodal, dtype=np.float64).reshape(-1)
    temperature = np.asarray(temperature_K, dtype=np.float64).reshape(-1)
    node_count = mesh.nodes_m.shape[0]
    if rho.size != node_count or temperature.size != node_count:
        raise ValueError("rho and temperature must match the electrical mesh")
    if np.any((rho < 0.0) | (rho > 1.0)) or not np.all(np.isfinite(rho)):
        raise ValueError("rho must be finite in [0,1]")
    if not np.all(np.isfinite(temperature)):
        raise ValueError("temperature must be finite")
    if not 0.0 < sigma_void_fraction < 1.0:
        raise ValueError("invalid void-conductivity regularization")
    solve = _reference_solve if linear_solve is None else linear_solve

    tri = mesh.triangles
    rho_element = np.mean(rho[tri], axis=1)
    sigma_scale = sigma_void_fraction + (1.0 - sigma_void_fraction) * rho_element**sigma_penalty
    alpha_scale = rho_element**alpha_penalty
    sigma = _element_tensor(sigma_xy_S_m, sigma_scale)
    alpha_base = (
        float(sigma_xy_S_m[0]) * float(seebeck_xy_V_K[0]),
        float(sigma_xy_S_m[1]) * float(seebeck_xy_V_K[1]),
    )
    alpha = _element_tensor(alpha_base, alpha_scale)
    matrix = _assemble(mesh, sigma, thickness_m)

    fixed = np.unique(np.concatenate((mesh.bottom_nodes, mesh.top_nodes)))
    fixed_values = np.zeros(fixed.size, dtype=np.float64)
    top_lookup = np.isin(fixed, mesh.top_nodes)
    fixed_values[top_lookup] = 1.0
    free_mask = np.ones(node_count, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)
    reduced = matrix[free][:, free].tocsr()
    rhs = -np.asarray(matrix[free][:, fixed] @ fixed_values).reshape(-1)
    psi = np.zeros(node_count, dtype=np.float64)
    psi[fixed] = fixed_values
    psi[free] = solve(reduced, rhs)
    weighting_residual = float(
        np.linalg.norm(reduced @ psi[free] - rhs)
        / max(np.linalg.norm(rhs), np.finfo(float).tiny)
    )

    grad_psi = np.einsum("eai,ei->ea", mesh.gradients_m_inv, psi[tri])
    grad_temperature = np.einsum("eai,ei->ea", mesh.gradients_m_inv, temperature[tri])
    element_current = -thickness_m * mesh.triangle_area_m2 * np.einsum(
        "ea,eab,eb->e", grad_psi, alpha, grad_temperature
    )
    current = float(np.sum(element_current))

    dcurrent_dpsi_local = -thickness_m * mesh.triangle_area_m2[:, None] * np.einsum(
        "eai,eab,eb->ei", mesh.gradients_m_inv, alpha, grad_temperature
    )
    dcurrent_dpsi = np.zeros(node_count, dtype=np.float64)
    np.add.at(dcurrent_dpsi, tri.ravel(), dcurrent_dpsi_local.ravel())
    adjoint_free = solve(reduced.T.tocsr(), dcurrent_dpsi[free])
    adjoint_residual = float(
        np.linalg.norm(reduced.T @ adjoint_free - dcurrent_dpsi[free])
        / max(np.linalg.norm(dcurrent_dpsi[free]), np.finfo(float).tiny)
    )
    adjoint = np.zeros(node_count, dtype=np.float64)
    adjoint[free] = adjoint_free

    dcurrent_dtemperature_local = -thickness_m * mesh.triangle_area_m2[:, None] * np.einsum(
        "eai,eab,eb->ei", mesh.gradients_m_inv, np.swapaxes(alpha, 1, 2), grad_psi
    )
    gradient_temperature = np.zeros(node_count, dtype=np.float64)
    np.add.at(
        gradient_temperature,
        tri.ravel(),
        dcurrent_dtemperature_local.ravel(),
    )

    dsigma_scale = (
        (1.0 - sigma_void_fraction)
        * sigma_penalty
        * np.where(rho_element > 0.0, rho_element ** (sigma_penalty - 1.0), 0.0)
        / 3.0
    )
    dalpha_scale = (
        alpha_penalty
        * np.where(rho_element > 0.0, rho_element ** (alpha_penalty - 1.0), 0.0)
        / 3.0
    )
    dsigma = _element_tensor(sigma_xy_S_m, dsigma_scale)
    dalpha = _element_tensor(alpha_base, dalpha_scale)
    direct_element = -thickness_m * mesh.triangle_area_m2 * np.einsum(
        "ea,eab,eb->e", grad_psi, dalpha, grad_temperature
    )
    grad_adjoint = np.einsum("eai,ei->ea", mesh.gradients_m_inv, adjoint[tri])
    implicit_element = -thickness_m * mesh.triangle_area_m2 * np.einsum(
        "ea,eab,eb->e", grad_adjoint, dsigma, grad_psi
    )
    element_node = np.repeat(
        (direct_element + implicit_element)[:, None], 3, axis=1
    )
    gradient_rho = np.zeros(node_count, dtype=np.float64)
    np.add.at(
        gradient_rho,
        tri.ravel(),
        element_node.ravel(),
    )

    terminal_load = np.zeros(node_count, dtype=np.float64)
    terminal_load[fixed] = matrix[fixed] @ psi
    terminal_conductance = float(np.sum(terminal_load[mesh.top_nodes]))
    terminal_element_derivative = thickness_m * mesh.triangle_area_m2 * np.einsum(
        "ea,eab,eb->e", grad_psi, dsigma, grad_psi
    )
    terminal_gradient = np.zeros(node_count, dtype=np.float64)
    np.add.at(
        terminal_gradient,
        tri.ravel(),
        np.repeat(terminal_element_derivative[:, None], 3, axis=1).ravel(),
    )
    return ElectricalResult(
        weighting_potential=psi.reshape(mesh.shape),
        weighting_gradient_element_m_inv=grad_psi,
        current_A=current,
        gradient_temperature_K_inv=gradient_temperature.reshape(mesh.shape),
        gradient_rho_A=gradient_rho.reshape(mesh.shape),
        weighting_residual=weighting_residual,
        adjoint_residual=adjoint_residual,
        terminal_conductance_S=terminal_conductance,
        gradient_terminal_conductance_S=terminal_gradient.reshape(mesh.shape),
    )
