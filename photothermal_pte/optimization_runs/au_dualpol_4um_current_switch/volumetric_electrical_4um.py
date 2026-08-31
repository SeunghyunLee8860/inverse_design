"""Explicit 3-D electrical readout on the existing thermal finite-volume grid.

The measurement is a terminal current in ampere.  It is evaluated by the
reciprocity identity

    I = sum_edges G_e S_e (T_r - T_l) (psi_l - psi_r),

where ``psi`` is the dimensionless weighting potential produced by applying
0 V and 1 V to the two fixed top-Au measurement strips.  The same edge
contributions are distributed over their adjacent cell volumes to provide an
A/m^3 diagnostic whose volume integral is exactly the terminal current.

This module deliberately does not use Lumerical HEAT/CHARGE.  Maxwell power is
handed to the existing custom 3-D thermal CUDA solve; electrical forward and
adjoint systems are assembled here and solved with the same float64 CUDA PCG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_material_fraction import (
    au_material_fraction,
    d_au_material_fraction_drho,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    CONTACT_FLOOR_FRACTION,
    SEEBECK_AU_TA_CONTACT_V_K,
    SEEBECK_AU_V_K,
    SEEBECK_TA_XY_V_K,
    SIGMA_AU_S_M,
    SIGMA_FLOOR_FRACTION,
    SIGMA_TA_XY_S_M,
    ThermalState,
    au_temperature,
    build_thermal_state,
    solve_thermal,
    solve_thermal_adjoint,
    tairte4_temperature,
    thermal_density_gradient,
)
from photothermal_pte.optimization_runs.cuda_thermal_adjoint import PersistentCudaCSR


@dataclass(frozen=True)
class VolumetricElectricalSystem:
    full_matrix_S: sparse.csr_matrix
    reduced_matrix_S: sparse.csr_matrix
    reduced_rhs_A: np.ndarray
    free: np.ndarray
    fixed: np.ndarray
    fixed_low: np.ndarray
    fixed_high: np.ndarray
    fixed_values_V: np.ndarray
    objective_gradient_psi_A: np.ndarray
    tairte4_thermoelectric_load_A: np.ndarray
    au_thermoelectric_load_A: np.ndarray
    conductance_derivative_left: np.ndarray
    conductance_derivative_right: np.ndarray
    conductance_derivative_rho: np.ndarray
    conductance_derivative_S: np.ndarray
    thermoelectric_left: np.ndarray
    thermoelectric_right: np.ndarray
    thermoelectric_temperature_left: np.ndarray
    thermoelectric_temperature_right: np.ndarray
    thermoelectric_coefficient_A_K: np.ndarray
    thermoelectric_material: np.ndarray
    thermoelectric_derivative_edge: np.ndarray
    thermoelectric_derivative_rho: np.ndarray
    thermoelectric_derivative_A_K: np.ndarray
    node_to_thermal_flat: np.ndarray
    temperature_K: np.ndarray
    thermal_cell_volume_m3: np.ndarray
    rho: np.ndarray
    material_fraction: np.ndarray
    exact_binary_geometry: bool
    sigma_z_S_m: float
    active_tairte4_nodes: int
    active_design_au_nodes: int
    fixed_electrode_nodes: int
    removed_void_au_nodes: int


def _pair_coordinates(mask: np.ndarray, axis: int) -> tuple[np.ndarray, ...]:
    left_slice = [slice(None)] * 3
    right_slice = [slice(None)] * 3
    left_slice[axis] = slice(0, -1)
    right_slice[axis] = slice(1, None)
    return tuple(np.asarray(value, dtype=np.int64) for value in np.nonzero(
        mask[tuple(left_slice)] & mask[tuple(right_slice)]
    ))


def _right_coordinates(
    coordinates: tuple[np.ndarray, ...], axis: int
) -> tuple[np.ndarray, ...]:
    result = [value.copy() for value in coordinates]
    result[axis] += 1
    return tuple(result)


def _pair_geometry(
    state: ThermalState,
    coordinates: tuple[np.ndarray, ...],
    axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    right = _right_coordinates(coordinates, axis)
    left_width = state.widths[axis][coordinates[axis]]
    right_width = state.widths[axis][right[axis]]
    area = np.ones(coordinates[0].size, dtype=np.float64)
    for other_axis in range(3):
        if other_axis != axis:
            area *= state.widths[other_axis][coordinates[other_axis]]
    left_flat = np.ravel_multi_index(coordinates, state.system.shape)
    right_flat = np.ravel_multi_index(right, state.system.shape)
    return left_width, right_width, area, np.stack((left_flat, right_flat))


def _assemble_laplacian(
    node_count: int,
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> sparse.csr_matrix:
    left = np.concatenate([item[0] for item in groups])
    right = np.concatenate([item[1] for item in groups])
    conductance = np.concatenate([item[2] for item in groups])
    if (
        np.any(left < 0)
        or np.any(right < 0)
        or np.any(conductance <= 0.0)
        or not np.all(np.isfinite(conductance))
    ):
        raise RuntimeError("invalid edge in 3-D electrical conductance graph")
    rows = np.concatenate((left, right, left, right))
    columns = np.concatenate((left, right, right, left))
    values = np.concatenate((conductance, conductance, -conductance, -conductance))
    matrix = sparse.coo_matrix(
        (values, (rows, columns)), shape=(node_count, node_count)
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def build_volumetric_electrical_system(
    state: ThermalState,
    temperature_K: np.ndarray,
    *,
    exact_binary_geometry: bool = False,
    sigma_z_S_m: float | None = None,
) -> VolumetricElectricalSystem:
    """Assemble TaIrTe4, floating design-Au, top-pad, and contact conductance."""

    temperature = np.asarray(temperature_K, dtype=np.float64)
    if temperature.shape != state.system.shape or not np.all(np.isfinite(temperature)):
        raise ValueError("temperature must match the full 3-D thermal grid")
    density = np.asarray(state.rho, dtype=np.float64)
    exact = bool(exact_binary_geometry)
    if exact and not np.all((density == 0.0) | (density == 1.0)):
        raise ValueError("exact-binary electrical geometry requires exact 0/1 rho")
    sigma_z = CONTRACT.tairte4_sigma_z_S_m if sigma_z_S_m is None else float(sigma_z_S_m)
    if not np.isfinite(sigma_z) or sigma_z <= 0.0:
        raise ValueError("TaIrTe4 sigma_z must be finite and positive")

    shape = state.system.shape
    ta_mask = np.asarray(state.masks["tairte4"], dtype=bool)
    design_mask = np.asarray(state.masks["design_au"], dtype=bool)
    electrode_mask = np.asarray(state.masks["measurement_electrodes"], dtype=bool)

    rho_index = np.full(shape, -1, dtype=np.int64)
    x, y, z = state.centers
    ix_design = np.flatnonzero((x >= -4.0e-6) & (x < 4.0e-6))
    iy_design = np.flatnonzero((y >= -4.0e-6) & (y < 4.0e-6))
    iz_au = np.flatnonzero((z >= 0.0) & (z < CONTRACT.design_thickness_m))
    if (ix_design.size, iy_design.size) != density.shape:
        raise RuntimeError("design density and 3-D electrical footprint disagree")
    local_rho_index = np.arange(density.size, dtype=np.int64).reshape(density.shape)
    rho_index[np.ix_(ix_design, iy_design, iz_au)] = local_rho_index[:, :, None]

    if exact:
        solid = np.zeros(shape, dtype=bool)
        solid[np.ix_(ix_design, iy_design, iz_au)] = (density == 1.0)[:, :, None]
        active_design = design_mask & solid
    else:
        active_design = design_mask.copy()
    active_au = active_design | electrode_mask

    node_grid = np.full(shape, -1, dtype=np.int64)
    ta_flat = np.flatnonzero(ta_mask.reshape(-1))
    au_flat = np.flatnonzero(active_au.reshape(-1))
    node_grid.reshape(-1)[ta_flat] = np.arange(ta_flat.size, dtype=np.int64)
    node_grid.reshape(-1)[au_flat] = ta_flat.size + np.arange(
        au_flat.size, dtype=np.int64
    )
    node_count = int(ta_flat.size + au_flat.size)
    node_to_thermal = np.concatenate((ta_flat, au_flat))

    matrix_groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    derivative_left: list[np.ndarray] = []
    derivative_right: list[np.ndarray] = []
    derivative_rho: list[np.ndarray] = []
    derivative_value: list[np.ndarray] = []
    te_left: list[np.ndarray] = []
    te_right: list[np.ndarray] = []
    te_temperature_left: list[np.ndarray] = []
    te_temperature_right: list[np.ndarray] = []
    te_coefficient: list[np.ndarray] = []
    te_material: list[np.ndarray] = []
    te_derivative_edge: list[np.ndarray] = []
    te_derivative_rho: list[np.ndarray] = []
    te_derivative_value: list[np.ndarray] = []
    te_count = 0

    sigma_ta = (float(SIGMA_TA_XY_S_M[0]), float(SIGMA_TA_XY_S_M[1]), sigma_z)
    seebeck_ta = (
        float(SEEBECK_TA_XY_V_K[0]),
        float(SEEBECK_TA_XY_V_K[1]),
        float(CONTRACT.tairte4_seebeck_z_V_K),
    )
    for axis in range(3):
        coordinates = _pair_coordinates(ta_mask, axis)
        right_coordinates = _right_coordinates(coordinates, axis)
        left_width, right_width, area, thermal_pair = _pair_geometry(
            state, coordinates, axis
        )
        conductance = area * sigma_ta[axis] / (0.5 * left_width + 0.5 * right_width)
        left_node = node_grid[coordinates]
        right_node = node_grid[right_coordinates]
        matrix_groups.append((left_node, right_node, conductance))
        if seebeck_ta[axis] != 0.0:
            coefficient = conductance * seebeck_ta[axis]
            te_left.append(left_node)
            te_right.append(right_node)
            te_temperature_left.append(thermal_pair[0])
            te_temperature_right.append(thermal_pair[1])
            te_coefficient.append(coefficient)
            te_material.append(np.zeros(coefficient.size, dtype=np.int8))
            te_count += coefficient.size

    fraction = np.asarray(au_material_fraction(density), dtype=np.float64)
    d_fraction = np.asarray(d_au_material_fraction_drho(density), dtype=np.float64)
    sigma_floor = 0.0 if exact else SIGMA_AU_S_M * SIGMA_FLOOR_FRACTION
    contact_floor = 0.0 if exact else (
        CONTRACT.electrical_contact_S_m2 * CONTACT_FLOOR_FRACTION
    )
    sigma_au = np.zeros(shape, dtype=np.float64)
    dsigma_au = np.zeros(shape, dtype=np.float64)
    sigma_design = sigma_floor + fraction * (SIGMA_AU_S_M - sigma_floor)
    dsigma_design = d_fraction * (SIGMA_AU_S_M - sigma_floor)
    sigma_au[electrode_mask] = SIGMA_AU_S_M
    for k in iz_au:
        sigma_au[np.ix_(ix_design, iy_design, [k])] = sigma_design[:, :, None]
        dsigma_au[np.ix_(ix_design, iy_design, [k])] = dsigma_design[:, :, None]

    for axis in range(3):
        coordinates = _pair_coordinates(active_au, axis)
        right_coordinates = _right_coordinates(coordinates, axis)
        left_width, right_width, area, thermal_pair = _pair_geometry(
            state, coordinates, axis
        )
        sigma_left = sigma_au[coordinates]
        sigma_right = sigma_au[right_coordinates]
        resistance = 0.5 * left_width / sigma_left + 0.5 * right_width / sigma_right
        conductance = area / resistance
        left_node = node_grid[coordinates]
        right_node = node_grid[right_coordinates]
        matrix_groups.append((left_node, right_node, conductance))

        left_rho = rho_index[coordinates]
        right_rho = rho_index[right_coordinates]
        left_design = left_rho >= 0
        right_design = right_rho >= 0
        if np.any(left_design != right_design):
            raise RuntimeError("fixed measurement electrode touches floating design Au")
        design_pair = left_design & right_design
        physical_conductance = conductance.copy()
        if not exact and np.any(design_pair):
            baseline_resistance = (
                0.5 * left_width[design_pair] / sigma_floor
                + 0.5 * right_width[design_pair] / sigma_floor
            )
            physical_conductance[design_pair] -= area[design_pair] / baseline_resistance
            physical_conductance[design_pair] = np.maximum(
                physical_conductance[design_pair], 0.0
            )
        coefficient = physical_conductance * SEEBECK_AU_V_K
        te_left.append(left_node)
        te_right.append(right_node)
        te_temperature_left.append(thermal_pair[0])
        te_temperature_right.append(thermal_pair[1])
        te_coefficient.append(coefficient)
        te_material.append(np.ones(coefficient.size, dtype=np.int8))

        if not exact:
            for endpoint, endpoint_rho, endpoint_sigma, endpoint_width, endpoint_dsigma in (
                (coordinates, left_rho, sigma_left, left_width, dsigma_au[coordinates]),
                (
                    right_coordinates,
                    right_rho,
                    sigma_right,
                    right_width,
                    dsigma_au[right_coordinates],
                ),
            ):
                varying = endpoint_rho >= 0
                dg = (
                    area[varying]
                    / resistance[varying] ** 2
                    * 0.5
                    * endpoint_width[varying]
                    / endpoint_sigma[varying] ** 2
                    * endpoint_dsigma[varying]
                )
                derivative_left.append(left_node[varying])
                derivative_right.append(right_node[varying])
                derivative_rho.append(endpoint_rho[varying])
                derivative_value.append(dg)
                te_derivative_edge.append(
                    te_count + np.flatnonzero(varying).astype(np.int64)
                )
                te_derivative_rho.append(endpoint_rho[varying])
                te_derivative_value.append(dg * SEEBECK_AU_V_K)
        te_count += coefficient.size

    top_face = int(state.faces["TaIrTe4_Au_or_air"])
    bottom_au_layer = top_face + 1
    contact_active = active_au[:, :, bottom_au_layer]
    ci, cj = np.nonzero(contact_active)
    ta_coordinates = (ci, cj, np.full(ci.size, top_face, dtype=np.int64))
    au_coordinates = (ci, cj, np.full(ci.size, bottom_au_layer, dtype=np.int64))
    contact_rho = rho_index[au_coordinates]
    contact_fraction = np.zeros(ci.size, dtype=np.float64)
    contact_derivative = np.zeros(ci.size, dtype=np.float64)
    is_design_contact = contact_rho >= 0
    if np.any(is_design_contact):
        flat_fraction = fraction.reshape(-1)
        flat_derivative = d_fraction.reshape(-1)
        contact_fraction[is_design_contact] = flat_fraction[
            contact_rho[is_design_contact]
        ]
        contact_derivative[is_design_contact] = flat_derivative[
            contact_rho[is_design_contact]
        ]
    contact_conductance_per_area = np.full(
        ci.size, CONTRACT.measurement_electrode_contact_S_m2, dtype=np.float64
    )
    contact_conductance_per_area[is_design_contact] = contact_floor + (
        contact_fraction[is_design_contact]
        * (CONTRACT.electrical_contact_S_m2 - contact_floor)
    )
    contact_area = state.widths[0][ci] * state.widths[1][cj]
    contact_g = contact_area * contact_conductance_per_area
    contact_left = node_grid[ta_coordinates]
    contact_right = node_grid[au_coordinates]
    matrix_groups.append((contact_left, contact_right, contact_g))
    if not exact and np.any(is_design_contact):
        contact_dg = (
            contact_area[is_design_contact]
            * (CONTRACT.electrical_contact_S_m2 - contact_floor)
            * contact_derivative[is_design_contact]
        )
        derivative_left.append(contact_left[is_design_contact])
        derivative_right.append(contact_right[is_design_contact])
        derivative_rho.append(contact_rho[is_design_contact])
        derivative_value.append(contact_dg)

    matrix = _assemble_laplacian(node_count, matrix_groups)
    electrode_flat = np.flatnonzero(electrode_mask.reshape(-1))
    electrode_nodes = node_grid.reshape(-1)[electrode_flat]
    electrode_x = np.unravel_index(electrode_flat, shape)[0]
    x_coordinates = state.centers[0][electrode_x]
    fixed_low = np.unique(electrode_nodes[x_coordinates < 0.0])
    fixed_high = np.unique(electrode_nodes[x_coordinates > 0.0])
    if fixed_low.size == 0 or fixed_high.size == 0:
        raise RuntimeError("both fixed top measurement electrodes are required")
    fixed = np.concatenate((fixed_low, fixed_high))
    fixed_values = np.concatenate((np.zeros(fixed_low.size), np.ones(fixed_high.size)))
    free_mask = np.ones(node_count, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)
    reduced = matrix[free][:, free].tocsr()
    rhs = -np.asarray(matrix[free][:, fixed] @ fixed_values).reshape(-1)

    te_left_array = np.concatenate(te_left)
    te_right_array = np.concatenate(te_right)
    te_temperature_left_array = np.concatenate(te_temperature_left)
    te_temperature_right_array = np.concatenate(te_temperature_right)
    te_coefficient_array = np.concatenate(te_coefficient)
    te_material_array = np.concatenate(te_material)
    delta_temperature = temperature.reshape(-1)[te_temperature_right_array] - (
        temperature.reshape(-1)[te_temperature_left_array]
    )
    edge_load = te_coefficient_array * delta_temperature
    load_ta = np.zeros(node_count, dtype=np.float64)
    load_au = np.zeros(node_count, dtype=np.float64)
    for material, destination in ((0, load_ta), (1, load_au)):
        selected = te_material_array == material
        np.add.at(destination, te_left_array[selected], edge_load[selected])
        np.add.at(destination, te_right_array[selected], -edge_load[selected])

    widths = state.widths
    cell_volume = (
        widths[0][:, None, None]
        * widths[1][None, :, None]
        * widths[2][None, None, :]
    )
    removed = int(np.count_nonzero(design_mask) - np.count_nonzero(active_design))
    return VolumetricElectricalSystem(
        full_matrix_S=matrix,
        reduced_matrix_S=reduced,
        reduced_rhs_A=rhs,
        free=free,
        fixed=fixed,
        fixed_low=fixed_low,
        fixed_high=fixed_high,
        fixed_values_V=fixed_values,
        objective_gradient_psi_A=load_ta + load_au,
        tairte4_thermoelectric_load_A=load_ta,
        au_thermoelectric_load_A=load_au,
        conductance_derivative_left=np.concatenate(derivative_left) if derivative_left else np.empty(0, dtype=np.int64),
        conductance_derivative_right=np.concatenate(derivative_right) if derivative_right else np.empty(0, dtype=np.int64),
        conductance_derivative_rho=np.concatenate(derivative_rho) if derivative_rho else np.empty(0, dtype=np.int64),
        conductance_derivative_S=np.concatenate(derivative_value) if derivative_value else np.empty(0, dtype=np.float64),
        thermoelectric_left=te_left_array,
        thermoelectric_right=te_right_array,
        thermoelectric_temperature_left=te_temperature_left_array,
        thermoelectric_temperature_right=te_temperature_right_array,
        thermoelectric_coefficient_A_K=te_coefficient_array,
        thermoelectric_material=te_material_array,
        thermoelectric_derivative_edge=np.concatenate(te_derivative_edge) if te_derivative_edge else np.empty(0, dtype=np.int64),
        thermoelectric_derivative_rho=np.concatenate(te_derivative_rho) if te_derivative_rho else np.empty(0, dtype=np.int64),
        thermoelectric_derivative_A_K=np.concatenate(te_derivative_value) if te_derivative_value else np.empty(0, dtype=np.float64),
        node_to_thermal_flat=node_to_thermal,
        temperature_K=temperature.copy(),
        thermal_cell_volume_m3=cell_volume,
        rho=density.copy(),
        material_fraction=fraction.copy(),
        exact_binary_geometry=exact,
        sigma_z_S_m=sigma_z,
        active_tairte4_nodes=int(ta_flat.size),
        active_design_au_nodes=int(np.count_nonzero(active_design)),
        fixed_electrode_nodes=int(np.count_nonzero(electrode_mask)),
        removed_void_au_nodes=removed,
    )


def solve_volumetric_electrical(
    system: VolumetricElectricalSystem, cuda_device: int
) -> tuple[np.ndarray, float, dict[str, object]]:
    operator = PersistentCudaCSR(system.reduced_matrix_S, cuda_device=cuda_device)
    solved = operator.solve(
        system.reduced_rhs_A,
        relative_tolerance=1.0e-10,
        max_iterations=30000,
        residual_check_interval=10,
    )
    psi = np.zeros(system.full_matrix_S.shape[0], dtype=np.float64)
    psi[system.fixed] = system.fixed_values_V
    psi[system.free] = solved.solution
    current = float(system.objective_gradient_psi_A @ psi)
    residual = np.asarray(system.full_matrix_S @ psi).reshape(-1)
    low = float(np.sum(residual[system.fixed_low]))
    high = float(np.sum(residual[system.fixed_high]))
    balance = abs(low + high) / max(abs(low), abs(high), np.finfo(float).tiny)
    explicit_free = np.linalg.norm(residual[system.free]) / max(
        np.linalg.norm(system.reduced_rhs_A), np.finfo(float).tiny
    )
    integrand = volumetric_current_integrand(system, psi)
    integrated_cell_current = integrand * system.thermal_cell_volume_m3
    integrated = float(np.sum(integrated_cell_current))
    integration_absolute_error = abs(integrated - current)
    integration_current_relative_error = integration_absolute_error / max(
        abs(current), np.finfo(float).tiny
    )
    current_absolute_scale = float(np.sum(np.abs(integrated_cell_current)))
    integration_normwise_error = integration_absolute_error / max(
        current_absolute_scale, np.finfo(float).tiny
    )
    cancellation_ratio = abs(current) / max(
        current_absolute_scale, np.finfo(float).tiny
    )
    matrix_asymmetry = system.full_matrix_S - system.full_matrix_S.T
    matrix_scale = max(
        float(np.max(np.abs(system.full_matrix_S.data))), np.finfo(float).tiny
    )
    asymmetry = 0.0 if matrix_asymmetry.nnz == 0 else (
        float(np.max(np.abs(matrix_asymmetry.data))) / matrix_scale
    )
    return psi, current, {
        "electrical_model": CONTRACT.electrical_model,
        "relative_residual": float(solved.explicit_relative_residual),
        "explicit_free_residual": float(explicit_free),
        "iterations": int(solved.iterations),
        "terminal_balance_relative": float(balance),
        "low_terminal_A_per_V": low,
        "high_terminal_A_per_V": high,
        "matrix_symmetry_relative": float(asymmetry),
        "volumetric_integral_current_A": integrated,
        "volumetric_integral_absolute_error_A": float(
            integration_absolute_error
        ),
        "volumetric_current_absolute_scale_A": current_absolute_scale,
        "volumetric_current_cancellation_ratio": float(cancellation_ratio),
        "volumetric_integral_relative_error": float(
            integration_current_relative_error
        ),
        "volumetric_integral_normwise_relative_error": float(
            integration_normwise_error
        ),
        "tairte4_thermoelectric_current_A": float(
            system.tairte4_thermoelectric_load_A @ psi
        ),
        "au_thermoelectric_current_A": float(system.au_thermoelectric_load_A @ psi),
        "S_Au_V_K": float(SEEBECK_AU_V_K),
        "S_Au_Ta_contact_V_K": float(SEEBECK_AU_TA_CONTACT_V_K),
        "S_TaIrTe4_z_V_K": float(CONTRACT.tairte4_seebeck_z_V_K),
        "sigma_TaIrTe4_z_S_m": float(system.sigma_z_S_m),
        "sigma_TaIrTe4_z_scenario": CONTRACT.tairte4_sigma_z_scenario,
        "S_TaIrTe4_z_scenario": CONTRACT.tairte4_seebeck_z_scenario,
        "Au_thermopower_model": "bulk_isotropic_3d_Au_edges_floor_subtracted",
        "Au_transport_parameter_scope": CONTRACT.au_transport_parameter_scope,
        "measurement_electrode_contact_S_m2": float(
            CONTRACT.measurement_electrode_contact_S_m2
        ),
        "measurement_electrode_geometry_is_named_baseline": True,
        "active_tairte4_nodes": system.active_tairte4_nodes,
        "active_design_au_nodes": system.active_design_au_nodes,
        "fixed_electrode_nodes": system.fixed_electrode_nodes,
        "exact_binary_geometry": system.exact_binary_geometry,
        "inactive_void_Au_node_count": system.removed_void_au_nodes,
        "inactive_void_Au_cell_count": int(
            np.count_nonzero(system.rho == 0.0)
        ),
        "electrical_void_Au_nodes_removed": bool(system.exact_binary_geometry),
    }


def volumetric_current_integrand(
    system: VolumetricElectricalSystem, psi: np.ndarray
) -> np.ndarray:
    """Return A/m^3 on thermal cells with exact integral equal to terminal I."""

    values = np.asarray(psi, dtype=np.float64)
    if values.shape != (system.full_matrix_S.shape[0],):
        raise ValueError("psi does not match the 3-D electrical system")
    temperature_flat = system.temperature_K.reshape(-1)
    delta_temperature = (
        temperature_flat[system.thermoelectric_temperature_right]
        - temperature_flat[system.thermoelectric_temperature_left]
    )
    contribution = (
        system.thermoelectric_coefficient_A_K
        * delta_temperature
        * (values[system.thermoelectric_left] - values[system.thermoelectric_right])
    )
    integrated_per_cell = np.zeros(temperature_flat.size, dtype=np.float64)
    np.add.at(
        integrated_per_cell,
        system.thermoelectric_temperature_left,
        0.5 * contribution,
    )
    np.add.at(
        integrated_per_cell,
        system.thermoelectric_temperature_right,
        0.5 * contribution,
    )
    return integrated_per_cell.reshape(system.temperature_K.shape) / (
        system.thermal_cell_volume_m3
    )


def volumetric_temperature_pullback(
    system: VolumetricElectricalSystem, psi: np.ndarray
) -> np.ndarray:
    """Return dI/dT on every explicit 3-D thermal cell."""

    values = np.asarray(psi, dtype=np.float64)
    if values.shape != (system.full_matrix_S.shape[0],):
        raise ValueError("psi does not match the 3-D electrical system")
    edge_scale = system.thermoelectric_coefficient_A_K * (
        values[system.thermoelectric_left] - values[system.thermoelectric_right]
    )
    gradient = np.zeros(system.temperature_K.size, dtype=np.float64)
    np.add.at(gradient, system.thermoelectric_temperature_left, -edge_scale)
    np.add.at(gradient, system.thermoelectric_temperature_right, edge_scale)
    return gradient.reshape(system.temperature_K.shape)


def solve_volumetric_electrical_adjoint(
    system: VolumetricElectricalSystem, cuda_device: int
) -> tuple[np.ndarray, dict[str, float | int]]:
    operator = PersistentCudaCSR(system.reduced_matrix_S, cuda_device=cuda_device)
    solved = operator.solve(
        system.objective_gradient_psi_A[system.free],
        relative_tolerance=1.0e-10,
        max_iterations=30000,
        residual_check_interval=10,
    )
    adjoint = np.zeros(system.full_matrix_S.shape[0], dtype=np.float64)
    adjoint[system.free] = solved.solution
    return adjoint, {
        "relative_residual": float(solved.explicit_relative_residual),
        "iterations": int(solved.iterations),
    }


def volumetric_electrical_density_gradient(
    system: VolumetricElectricalSystem,
    psi: np.ndarray,
    adjoint: np.ndarray,
) -> np.ndarray:
    gradient = np.zeros(system.rho.size, dtype=np.float64)
    if system.conductance_derivative_rho.size:
        value = -system.conductance_derivative_S * (
            adjoint[system.conductance_derivative_left]
            - adjoint[system.conductance_derivative_right]
        ) * (
            psi[system.conductance_derivative_left]
            - psi[system.conductance_derivative_right]
        )
        np.add.at(gradient, system.conductance_derivative_rho, value)
    if system.thermoelectric_derivative_rho.size:
        edge = system.thermoelectric_derivative_edge
        temperature_flat = system.temperature_K.reshape(-1)
        delta_temperature = (
            temperature_flat[system.thermoelectric_temperature_right[edge]]
            - temperature_flat[system.thermoelectric_temperature_left[edge]]
        )
        weighting_drop = (
            psi[system.thermoelectric_left[edge]]
            - psi[system.thermoelectric_right[edge]]
        )
        np.add.at(
            gradient,
            system.thermoelectric_derivative_rho,
            system.thermoelectric_derivative_A_K
            * delta_temperature
            * weighting_drop,
        )
    return gradient.reshape(system.rho.shape)


def evaluate_fixed_source_volumetric(
    rho: np.ndarray,
    source_power_W: np.ndarray,
    cuda_device: int,
    *,
    need_gradient: bool,
    exact_binary_geometry: bool = False,
    sigma_z_S_m: float | None = None,
) -> dict[str, object]:
    if exact_binary_geometry and need_gradient:
        raise ValueError(
            "exact-binary topology removal is nondifferentiable; set need_gradient=False"
        )
    state = build_thermal_state(rho)
    temperature, thermal_audit = solve_thermal(state, source_power_W, cuda_device)
    electrical = build_volumetric_electrical_system(
        state,
        temperature,
        exact_binary_geometry=exact_binary_geometry,
        sigma_z_S_m=sigma_z_S_m,
    )
    psi, current, electrical_audit = solve_volumetric_electrical(
        electrical, cuda_device
    )
    result: dict[str, object] = {
        "objective_A": current,
        "state": state,
        "temperature": temperature,
        "ta_temperature": tairte4_temperature(state, temperature),
        "au_temperature": au_temperature(state, temperature),
        "electrical_system": electrical,
        "weighting": psi,
        "volumetric_current_density_A_m3": volumetric_current_integrand(
            electrical, psi
        ),
        "thermal_audit": thermal_audit,
        "electrical_audit": electrical_audit,
    }
    if need_gradient:
        electrical_adjoint, electrical_adjoint_audit = (
            solve_volumetric_electrical_adjoint(electrical, cuda_device)
        )
        thermal_rhs = volumetric_temperature_pullback(electrical, psi)
        thermal_adjoint, thermal_adjoint_audit = solve_thermal_adjoint(
            state, thermal_rhs, cuda_device
        )
        gradient_thermal = thermal_density_gradient(
            state, temperature, thermal_adjoint
        )
        gradient_electrical = volumetric_electrical_density_gradient(
            electrical, psi, electrical_adjoint
        )
        result.update(
            electrical_adjoint=electrical_adjoint,
            electrical_adjoint_audit=electrical_adjoint_audit,
            thermal_adjoint=thermal_adjoint,
            thermal_adjoint_audit=thermal_adjoint_audit,
            gradient_thermal_A=gradient_thermal,
            gradient_electrical_A=gradient_electrical,
            gradient_direct_A=gradient_thermal + gradient_electrical,
        )
    return result
