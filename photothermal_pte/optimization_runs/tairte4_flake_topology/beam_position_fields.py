"""Spatial diagnostics for fixed-geometry beam-position response scans."""

from __future__ import annotations

from typing import Callable

import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.electrical import (
    build_rectangular_mesh,
    solve_short_circuit_current_density,
)


Array = np.ndarray
LinearSolve = Callable[[object, Array], Array]
SIGMA_XY_S_M = (1.10e5, 4.91e5)
SEEBECK_XY_V_K = (27.0e-6, -6.0e-6)
FIELD_SCHEMA = "exact-binary-fixed-flake-au-position-spatial-fields-v1"


def full_flake_density(rho: Array) -> Array:
    values = np.asarray(rho, dtype=np.float64)
    if values.shape != CONTRACT.design_node_shape:
        raise ValueError("invalid exact-binary design density shape")
    full = np.ones(CONTRACT.flake_node_shape, dtype=np.float64)
    full[CONTRACT.design_node_slices] = values
    return full


def element_to_cell_average(values: Array, cell_shape: tuple[int, int]) -> Array:
    array = np.asarray(values, dtype=np.float64)
    cell_count = int(np.prod(cell_shape))
    if array.shape[0] != 2 * cell_count:
        raise ValueError("element array does not match the rectangular mesh")
    tail = array.shape[1:]
    return 0.5 * (
        array[:cell_count].reshape(*cell_shape, *tail)
        + array[cell_count:].reshape(*cell_shape, *tail)
    )


def strict_centered_gradient(
    temperature_K: Array, solid_nodal: Array, step_m: float
) -> tuple[Array, Array, Array]:
    temperature = np.asarray(temperature_K, dtype=np.float64)
    solid = np.asarray(solid_nodal, dtype=bool)
    valid = np.zeros_like(solid)
    valid[1:-1, 1:-1] = (
        solid[1:-1, 1:-1]
        & solid[:-2, 1:-1]
        & solid[2:, 1:-1]
        & solid[1:-1, :-2]
        & solid[1:-1, 2:]
    )
    gx = np.full_like(temperature, np.nan)
    gy = np.full_like(temperature, np.nan)
    central_x = (temperature[2:, 1:-1] - temperature[:-2, 1:-1]) / (2.0 * step_m)
    central_y = (temperature[1:-1, 2:] - temperature[1:-1, :-2]) / (2.0 * step_m)
    interior = valid[1:-1, 1:-1]
    gx[1:-1, 1:-1][interior] = central_x[interior]
    gy[1:-1, 1:-1][interior] = central_y[interior]
    return gx, gy, np.hypot(gx, gy)


def absorption_maps(coupled: dict[str, object], scale: float) -> dict[str, Array]:
    state = coupled["state"]
    q = np.asarray(coupled["mapped_q"], dtype=np.float64) * scale
    dz = np.asarray(state.widths_m[2], dtype=np.float64)
    dx = np.asarray(state.widths_m[0], dtype=np.float64)
    dy = np.asarray(state.widths_m[1], dtype=np.float64)

    def areal(mask_name: str) -> Array:
        mask = np.asarray(state.masks[mask_name], dtype=bool)
        return np.sum(np.where(mask, q, 0.0) * dz[None, None, :], axis=2)

    return {
        "thermal_x_cell_m": 0.5 * (state.edges_m[0][:-1] + state.edges_m[0][1:]),
        "thermal_y_cell_m": 0.5 * (state.edges_m[1][:-1] + state.edges_m[1][1:]),
        "thermal_z_cell_m": 0.5 * (state.edges_m[2][:-1] + state.edges_m[2][1:]),
        "thermal_x_cell_width_m": dx,
        "thermal_y_cell_width_m": dy,
        "absorbed_power_density_total_W_m2": np.sum(q * dz[None, None, :], axis=2),
        "absorbed_power_density_Au_W_m2": areal("Au_electrodes"),
        "absorbed_power_density_TaIrTe4_W_m2": areal("flake_support"),
        "absorbed_power_density_SiO2_W_m2": areal("SiO2"),
        "absorbed_power_density_Si_W_m2": areal("Si"),
        "absorbed_power_by_z_W": np.sum(
            q * dx[:, None, None] * dy[None, :, None] * dz[None, None, :],
            axis=(0, 1),
        ),
    }


def build_position_field_arrays(
    coupled: dict[str, object],
    rho: Array,
    scale: float,
    contact_axis: str,
    *,
    linear_solve: LinearSolve | None = None,
) -> tuple[dict[str, Array], dict[str, float | bool | str]]:
    """Build self-contained 2-D fields and independent current identities."""

    mesh = build_rectangular_mesh(
        CONTRACT.flake_span_m, CONTRACT.flake_span_m, CONTRACT.design_step_m
    )
    rho_full = full_flake_density(rho)
    temperature = np.asarray(coupled["temperature"], dtype=np.float64) * scale
    electrical = coupled["electrical"]
    grad_psi = np.asarray(
        electrical.weighting_gradient_element_m_inv, dtype=np.float64
    )
    short = solve_short_circuit_current_density(
        mesh,
        rho_full,
        temperature,
        thickness_m=CONTRACT.flake_thickness_m,
        sigma_xy_S_m=SIGMA_XY_S_M,
        seebeck_xy_V_K=SEEBECK_XY_V_K,
        sigma_void_fraction=CONTRACT.sigma_void_fraction,
        sigma_penalty=CONTRACT.sigma_penalty,
        alpha_penalty=CONTRACT.alpha_penalty,
        linear_solve=linear_solve,
        terminal_axis=contact_axis,
    )

    tri = mesh.triangles
    rho_element = np.mean(rho_full.ravel()[tri], axis=1)
    alpha = np.empty((tri.shape[0], 2), dtype=np.float64)
    alpha[:, 0] = (
        SIGMA_XY_S_M[0] * SEEBECK_XY_V_K[0]
        * rho_element**CONTRACT.alpha_penalty
    )
    alpha[:, 1] = (
        SIGMA_XY_S_M[1] * SEEBECK_XY_V_K[1]
        * rho_element**CONTRACT.alpha_penalty
    )
    grad_t = short.temperature_gradient_element_K_m
    pte_contribution_xy = (
        -CONTRACT.flake_thickness_m * grad_psi * alpha * grad_t
    )
    total_weighted_contribution_xy = (
        CONTRACT.flake_thickness_m
        * grad_psi
        * short.total_current_density_element_A_m2
    )
    area = mesh.triangle_area_m2
    weighting_current = float(
        np.sum(pte_contribution_xy.sum(axis=1) * area)
    )
    total_weighted_current = float(
        np.sum(total_weighted_contribution_xy.sum(axis=1) * area)
    )
    certified_current = float(electrical.current_A) * scale
    gross_contribution = float(
        np.sum(np.abs(pte_contribution_xy.sum(axis=1)) * area)
    )
    denominator = max(
        abs(certified_current), gross_contribution, np.finfo(float).tiny
    )
    cell_shape = (mesh.x_m.size - 1, mesh.y_m.size - 1)

    grad_t_cell = element_to_cell_average(grad_t, cell_shape)
    grad_psi_cell = element_to_cell_average(grad_psi, cell_shape)
    electric_cell = element_to_cell_average(short.electric_field_element_V_m, cell_shape)
    j_pte_cell = element_to_cell_average(
        short.thermoelectric_current_density_element_A_m2, cell_shape
    )
    j_conductive_cell = element_to_cell_average(
        short.conductive_current_density_element_A_m2, cell_shape
    )
    j_total_cell = element_to_cell_average(
        short.total_current_density_element_A_m2, cell_shape
    )
    contribution_cell_xy = element_to_cell_average(
        pte_contribution_xy, cell_shape
    )
    contribution_total_cell_xy = element_to_cell_average(
        total_weighted_contribution_xy, cell_shape
    )
    contribution_total = np.sum(contribution_cell_xy, axis=2)
    total_weighted_contribution = np.sum(contribution_total_cell_xy, axis=2)
    gx_node, gy_node, gmag_node = strict_centered_gradient(
        temperature, rho_full >= 0.5, CONTRACT.design_step_m
    )

    arrays: dict[str, Array] = {
        "schema": np.asarray(FIELD_SCHEMA),
        "axis_contract": np.asarray("Lumerical x=b, y=a, z=c"),
        "electrical_x_node_m": mesh.x_m,
        "electrical_y_node_m": mesh.y_m,
        "electrical_x_cell_m": 0.5 * (mesh.x_m[:-1] + mesh.x_m[1:]),
        "electrical_y_cell_m": 0.5 * (mesh.y_m[:-1] + mesh.y_m[1:]),
        "rho_exact_binary_nodal": rho_full.astype(np.uint8),
        "rho_exact_binary_cell": element_to_cell_average(rho_element, cell_shape),
        "temperature_rise_nodal_K": temperature,
        "temperature_gradient_strict_x_nodal_K_m": gx_node,
        "temperature_gradient_strict_y_nodal_K_m": gy_node,
        "temperature_gradient_strict_magnitude_nodal_K_m": gmag_node,
        "temperature_gradient_x_cell_K_m": grad_t_cell[:, :, 0],
        "temperature_gradient_y_cell_K_m": grad_t_cell[:, :, 1],
        "temperature_gradient_magnitude_cell_K_m": np.linalg.norm(grad_t_cell, axis=2),
        "weighting_potential_nodal": np.asarray(electrical.weighting_potential, dtype=np.float64),
        "weighting_gradient_x_cell_m_inv": grad_psi_cell[:, :, 0],
        "weighting_gradient_y_cell_m_inv": grad_psi_cell[:, :, 1],
        "short_circuit_potential_nodal_V": short.potential_V,
        "electric_field_x_cell_V_m": electric_cell[:, :, 0],
        "electric_field_y_cell_V_m": electric_cell[:, :, 1],
        "local_J_thermoelectric_x_A_m2": j_pte_cell[:, :, 0],
        "local_J_thermoelectric_y_A_m2": j_pte_cell[:, :, 1],
        "local_J_conductive_x_A_m2": j_conductive_cell[:, :, 0],
        "local_J_conductive_y_A_m2": j_conductive_cell[:, :, 1],
        "local_J_total_x_A_m2": j_total_cell[:, :, 0],
        "local_J_total_y_A_m2": j_total_cell[:, :, 1],
        "local_J_total_magnitude_A_m2": np.linalg.norm(j_total_cell, axis=2),
        "terminal_current_contribution_x_A_m2": contribution_cell_xy[:, :, 0],
        "terminal_current_contribution_y_A_m2": contribution_cell_xy[:, :, 1],
        "terminal_current_contribution_total_A_m2": contribution_total,
        "terminal_current_contribution_positive_A_m2": np.maximum(contribution_total, 0.0),
        "terminal_current_contribution_negative_A_m2": np.minimum(contribution_total, 0.0),
        "total_J_weighted_contribution_x_A_m2": contribution_total_cell_xy[:, :, 0],
        "total_J_weighted_contribution_y_A_m2": contribution_total_cell_xy[:, :, 1],
        "total_J_weighted_contribution_total_A_m2": total_weighted_contribution,
        **absorption_maps(coupled, scale),
    }
    finite_arrays = all(
        np.all(np.isfinite(value))
        for key, value in arrays.items()
        if value.dtype.kind not in "US" and "strict" not in key
    )
    metrics: dict[str, float | bool | str] = {
        "field_schema": FIELD_SCHEMA,
        "terminal_current_A": certified_current,
        "terminal_current_from_pte_contribution_A": weighting_current,
        "terminal_current_from_short_circuit_flux_A": short.terminal_current_A,
        "terminal_current_from_total_J_weighting_A": total_weighted_current,
        "gross_terminal_current_contribution_A": gross_contribution,
        "current_identity_normalization_A": denominator,
        "pte_contribution_relative_error": abs(weighting_current - certified_current) / denominator,
        "short_circuit_flux_relative_error": abs(short.terminal_current_A - certified_current) / denominator,
        "total_J_weighting_relative_error": abs(total_weighted_current - certified_current) / denominator,
        "short_circuit_continuity_residual": short.continuity_residual,
        "temperature_max_K": float(np.max(temperature)),
        "temperature_gradient_max_K_m": float(np.max(np.linalg.norm(grad_t, axis=1))),
        "local_J_total_max_A_m2": float(np.max(np.linalg.norm(short.total_current_density_element_A_m2, axis=1))),
        "local_contribution_positive_A": float(
            np.sum(np.maximum(pte_contribution_xy.sum(axis=1), 0.0) * area)
        ),
        "local_contribution_negative_A": float(
            np.sum(np.minimum(pte_contribution_xy.sum(axis=1), 0.0) * area)
        ),
        "all_finite": finite_arrays,
    }
    return arrays, metrics
