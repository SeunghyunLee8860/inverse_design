"""Explicit finite-device thermal model and exact discrete rho adjoint.

This is a differentiable *thermal control* for the finite, non-periodic
inverse-design contract.  It intentionally accepts a fixed volumetric heat
source so the thermal-material/interface branch can be certified before an
optical device source is promoted.

The Cartesian domain contains explicit Si, bottom SiO2, TaIrTe4, design
SiO2/air relaxation, and surrounding air cells.  Every conductance used by
the rho gradient is the literal conductance used by the forward sparse
matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import linalg as sparse_linalg

from .contract import (
    BOTTOM_SIO2_BOUNDS_Z_M,
    DESIGN_BOUNDS_M,
    FLAKE_BOUNDS_Z_M,
    G_SIO2_SI_W_M2K,
    G_TAIRTE4_AIR_W_M2K,
    G_TAIRTE4_BOTTOM_SIO2_W_M2K,
    G_TAIRTE4_DEPOSITED_SIO2_W_M2K,
    H_SIO2_AIR_W_M2K,
    KAPPA_AIR_W_MK,
    KAPPA_SI_W_MK,
    KAPPA_SIO2_W_MK,
    KAPPA_TAIRTE4_W_MK,
)
from .uniform_weighting_pte import (
    UniformWeightingPTE,
    build_uniform_45deg_weighting_pte,
)


HERE = Path(__file__).resolve().parent
FVM_DIR = HERE.parent / "validation" / "photothermal_stage1"
if str(FVM_DIR) not in sys.path:
    sys.path.insert(0, str(FVM_DIR))

from anisotropic_heat_fvm import (  # noqa: E402
    AssembledThermalSystem,
    SteadyHeatResult,
    assemble_steady_diagonal_kappa,
    solve_assembled_thermal_system,
)


MATERIAL_AIR = np.uint8(0)
MATERIAL_SI = np.uint8(1)
MATERIAL_BOTTOM_SIO2 = np.uint8(2)
MATERIAL_TAIRTE4 = np.uint8(3)
MATERIAL_DESIGN = np.uint8(4)


@dataclass(frozen=True)
class ExplicitThermalGeometry:
    x_edges_m: np.ndarray
    y_edges_m: np.ndarray
    z_edges_m: np.ndarray
    material_id: np.ndarray
    flake_mask: np.ndarray
    design_mask: np.ndarray
    design_rho_id: np.ndarray
    interface_resistance_m2K_W: dict[str, np.ndarray]
    kappa_W_mK: np.ndarray
    rho: np.ndarray
    flake_span_m: float
    lateral_domain_m: float
    si_depth_m: float
    cell_size_m: float


@dataclass(frozen=True)
class ExplicitThermalForward:
    geometry: ExplicitThermalGeometry
    system: AssembledThermalSystem
    solved: SteadyHeatResult
    pte: UniformWeightingPTE
    theta_active_K: np.ndarray
    objective_A: float
    source_W_m3: np.ndarray


@dataclass(frozen=True)
class ExplicitThermalEvaluation:
    geometry: ExplicitThermalGeometry
    system: AssembledThermalSystem
    solved: SteadyHeatResult
    pte: UniformWeightingPTE
    theta_active_K: np.ndarray
    adjoint_active: np.ndarray
    objective_A: float
    gradient_rho_A: np.ndarray
    gradient_bulk_k_A: np.ndarray
    gradient_interface_G_A: np.ndarray
    gradient_top_convection_k_A: np.ndarray
    gradient_Q_active_A_m3_W: np.ndarray
    source_W_m3: np.ndarray
    adjoint_linear_residual_relative: float
    adjoint_iterations: int


def _growing_positions(
    length_m: float,
    *,
    initial_step_m: float,
    maximum_step_m: float,
    growth: float = 1.45,
) -> np.ndarray:
    if length_m <= 0.0:
        return np.asarray([0.0])
    values = [0.0]
    step = initial_step_m
    while values[-1] + step < length_m:
        values.append(values[-1] + step)
        step = min(maximum_step_m, step * growth)
    if length_m - values[-1] < 0.35 * step and len(values) > 1:
        values[-1] = length_m
    else:
        values.append(length_m)
    return np.asarray(values)


def _lateral_edges(
    *,
    lateral_domain_m: float,
    flake_span_m: float,
    cell_size_m: float,
) -> np.ndarray:
    half_domain = 0.5 * lateral_domain_m
    half_flake = 0.5 * flake_span_m
    if half_domain <= half_flake:
        raise ValueError("thermal domain must extend beyond the flake")
    cells = int(round(flake_span_m / cell_size_m))
    if not np.isclose(cells * cell_size_m, flake_span_m):
        raise ValueError("flake span must be an integer number of core cells")
    core = np.linspace(-half_flake, half_flake, cells + 1)
    outer = _growing_positions(
        half_domain - half_flake,
        initial_step_m=cell_size_m,
        maximum_step_m=1.0e-6,
    )[1:]
    return np.concatenate(
        (-half_flake - outer[::-1], core, half_flake + outer)
    )


def _z_edges(
    *,
    si_depth_m: float,
    cell_size_m: float,
) -> np.ndarray:
    si_top = BOTTOM_SIO2_BOUNDS_Z_M[0]
    si = si_top - _growing_positions(
        si_depth_m,
        initial_step_m=25.0e-9,
        maximum_step_m=0.5e-6,
        growth=1.35,
    )[::-1]
    oxide_cells = max(
        1,
        int(
            round(
                (
                    BOTTOM_SIO2_BOUNDS_Z_M[1]
                    - BOTTOM_SIO2_BOUNDS_Z_M[0]
                )
                / 15.0e-9
            )
        ),
    )
    oxide = np.linspace(
        BOTTOM_SIO2_BOUNDS_Z_M[0],
        BOTTOM_SIO2_BOUNDS_Z_M[1],
        oxide_cells + 1,
    )
    flake_cells = max(
        1,
        int(
            round(
                (FLAKE_BOUNDS_Z_M[1] - FLAKE_BOUNDS_Z_M[0])
                / min(cell_size_m, 25.0e-9)
            )
        ),
    )
    flake = np.linspace(
        FLAKE_BOUNDS_Z_M[0],
        FLAKE_BOUNDS_Z_M[1],
        flake_cells + 1,
    )
    design_cells = int(
        round(
            (DESIGN_BOUNDS_M["z"][1] - DESIGN_BOUNDS_M["z"][0])
            / cell_size_m
        )
    )
    design = np.linspace(
        DESIGN_BOUNDS_M["z"][0],
        DESIGN_BOUNDS_M["z"][1],
        design_cells + 1,
    )
    edges = np.concatenate((si, oxide[1:], flake[1:], design[1:]))
    if not np.all(np.diff(edges) > 0.0):
        raise RuntimeError("constructed z grid is not strictly increasing")
    return edges


def _face_before(edges: np.ndarray, coordinate_m: float) -> int:
    matches = np.flatnonzero(
        np.isclose(edges, coordinate_m, rtol=0.0, atol=2.0e-18)
    )
    if matches.size != 1 or matches[0] in (0, edges.size - 1):
        raise ValueError(f"cannot locate internal face at {coordinate_m}")
    return int(matches[0] - 1)


def _design_indices(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    selected_x = np.flatnonzero(
        (x > DESIGN_BOUNDS_M["x"][0])
        & (x < DESIGN_BOUNDS_M["x"][1])
    )
    selected_y = np.flatnonzero(
        (y > DESIGN_BOUNDS_M["y"][0])
        & (y < DESIGN_BOUNDS_M["y"][1])
    )
    if selected_x.size == 0 or selected_y.size == 0:
        raise ValueError("thermal grid contains no design cells")
    return selected_x, selected_y, (selected_x.size, selected_y.size)


def build_explicit_geometry(
    rho: np.ndarray,
    *,
    lateral_domain_m: float = 32.0e-6,
    si_depth_m: float = 20.0e-6,
    flake_span_m: float = 4.0e-6,
    cell_size_m: float = 50.0e-9,
) -> ExplicitThermalGeometry:
    x_edges = _lateral_edges(
        lateral_domain_m=lateral_domain_m,
        flake_span_m=flake_span_m,
        cell_size_m=cell_size_m,
    )
    y_edges = x_edges.copy()
    z_edges = _z_edges(
        si_depth_m=si_depth_m,
        cell_size_m=cell_size_m,
    )
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    selected_x, selected_y, rho_shape = _design_indices(x, y)
    density = np.asarray(rho, float)
    if density.shape != rho_shape:
        raise ValueError(f"rho shape {density.shape} != {rho_shape}")
    if not np.all(np.isfinite(density)):
        raise ValueError("rho contains NaN or Inf")
    if np.any(density < 0.0) or np.any(density > 1.0):
        raise ValueError("rho must lie in [0, 1]")

    shape = (x.size, y.size, z.size)
    material = np.full(shape, MATERIAL_AIR, dtype=np.uint8)
    si_z = z < BOTTOM_SIO2_BOUNDS_Z_M[0]
    oxide_z = (
        (z > BOTTOM_SIO2_BOUNDS_Z_M[0])
        & (z < BOTTOM_SIO2_BOUNDS_Z_M[1])
    )
    flake_z = (z > FLAKE_BOUNDS_Z_M[0]) & (z < FLAKE_BOUNDS_Z_M[1])
    design_z = (
        (z > DESIGN_BOUNDS_M["z"][0])
        & (z < DESIGN_BOUNDS_M["z"][1])
    )
    material[:, :, si_z] = MATERIAL_SI
    material[:, :, oxide_z] = MATERIAL_BOTTOM_SIO2
    flake_xy = (
        (np.abs(x[:, None]) < 0.5 * flake_span_m)
        & (np.abs(y[None, :]) < 0.5 * flake_span_m)
    )
    flake_mask = flake_xy[:, :, None] & flake_z[None, None, :]
    material[flake_mask] = MATERIAL_TAIRTE4
    design_xy = np.zeros((x.size, y.size), bool)
    design_xy[np.ix_(selected_x, selected_y)] = True
    design_mask = design_xy[:, :, None] & design_z[None, None, :]
    material[design_mask] = MATERIAL_DESIGN

    rho_id = np.full(shape, -1, dtype=np.int64)
    design_ids_2d = np.arange(density.size, dtype=np.int64).reshape(
        density.shape
    )
    for z_index in np.flatnonzero(design_z):
        rho_id[np.ix_(selected_x, selected_y, [z_index])] = (
            design_ids_2d[:, :, None]
        )

    kappa = np.full((*shape, 3), KAPPA_AIR_W_MK, float)
    kappa[material == MATERIAL_SI] = KAPPA_SI_W_MK
    kappa[material == MATERIAL_BOTTOM_SIO2] = KAPPA_SIO2_W_MK
    kappa[material == MATERIAL_TAIRTE4] = KAPPA_TAIRTE4_W_MK
    design_k = KAPPA_AIR_W_MK + density * (
        KAPPA_SIO2_W_MK - KAPPA_AIR_W_MK
    )
    for z_index in np.flatnonzero(design_z):
        for component in range(3):
            kappa[
                selected_x[:, None],
                selected_y[None, :],
                z_index,
                component,
            ] = design_k

    resistance_x = np.zeros((shape[0] - 1, shape[1], shape[2]), float)
    resistance_y = np.zeros((shape[0], shape[1] - 1, shape[2]), float)
    resistance_z = np.zeros((shape[0], shape[1], shape[2] - 1), float)

    left = material[:-1, :, :]
    right = material[1:, :, :]
    lateral_flake_air_x = (
        ((left == MATERIAL_TAIRTE4) & (right == MATERIAL_AIR))
        | ((left == MATERIAL_AIR) & (right == MATERIAL_TAIRTE4))
    )
    resistance_x[lateral_flake_air_x] = 1.0 / G_TAIRTE4_AIR_W_M2K
    lower = material[:, :-1, :]
    upper = material[:, 1:, :]
    lateral_flake_air_y = (
        ((lower == MATERIAL_TAIRTE4) & (upper == MATERIAL_AIR))
        | ((lower == MATERIAL_AIR) & (upper == MATERIAL_TAIRTE4))
    )
    resistance_y[lateral_flake_air_y] = 1.0 / G_TAIRTE4_AIR_W_M2K

    oxide_si_face = _face_before(z_edges, BOTTOM_SIO2_BOUNDS_Z_M[0])
    resistance_z[:, :, oxide_si_face] = 1.0 / G_SIO2_SI_W_M2K
    flake_bottom_face = _face_before(z_edges, FLAKE_BOUNDS_Z_M[0])
    bottom_connected = (
        (material[:, :, flake_bottom_face] == MATERIAL_BOTTOM_SIO2)
        & (material[:, :, flake_bottom_face + 1] == MATERIAL_TAIRTE4)
    )
    resistance_z[:, :, flake_bottom_face][bottom_connected] = (
        1.0 / G_TAIRTE4_BOTTOM_SIO2_W_M2K
    )
    flake_top_face = _face_before(z_edges, FLAKE_BOUNDS_Z_M[1])
    top_flake = material[:, :, flake_top_face] == MATERIAL_TAIRTE4
    top_density = np.zeros((x.size, y.size), float)
    top_density[np.ix_(selected_x, selected_y)] = density
    top_g = G_TAIRTE4_AIR_W_M2K + top_density * (
        G_TAIRTE4_DEPOSITED_SIO2_W_M2K
        - G_TAIRTE4_AIR_W_M2K
    )
    resistance_z[:, :, flake_top_face][top_flake] = 1.0 / top_g[top_flake]

    return ExplicitThermalGeometry(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        material_id=material,
        flake_mask=flake_mask,
        design_mask=design_mask,
        design_rho_id=rho_id,
        interface_resistance_m2K_W={
            "x": resistance_x,
            "y": resistance_y,
            "z": resistance_z,
        },
        kappa_W_mK=kappa,
        rho=density,
        flake_span_m=float(flake_span_m),
        lateral_domain_m=float(lateral_domain_m),
        si_depth_m=float(si_depth_m),
        cell_size_m=float(cell_size_m),
    )


def fixed_control_source(
    geometry: ExplicitThermalGeometry,
    *,
    total_power_W: float = 2.56071371086521e-12,
) -> np.ndarray:
    """Return a positive asymmetric fixed-Q control, not an optical artifact."""
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
    z = 0.5 * (geometry.z_edges_m[:-1] + geometry.z_edges_m[1:])
    pattern = (
        1.0
        + 0.22 * x[:, None, None] / (0.5 * geometry.flake_span_m)
        - 0.17 * y[None, :, None] / (0.5 * geometry.flake_span_m)
        + 0.08
        * (z[None, None, :] - np.mean(FLAKE_BOUNDS_Z_M))
        / (0.5 * (FLAKE_BOUNDS_Z_M[1] - FLAKE_BOUNDS_Z_M[0]))
    )
    source = np.zeros(geometry.material_id.shape, float)
    source[geometry.flake_mask] = pattern[geometry.flake_mask]
    volume = (
        np.diff(geometry.x_edges_m)[:, None, None]
        * np.diff(geometry.y_edges_m)[None, :, None]
        * np.diff(geometry.z_edges_m)[None, None, :]
    )
    normalization = float(np.sum(source * volume))
    if normalization <= 0.0:
        raise RuntimeError("fixed control source has nonpositive integral")
    source *= float(total_power_W) / normalization
    return source


def _rho_conductance_gradient(
    *,
    geometry: ExplicitThermalGeometry,
    theta: np.ndarray,
    adjoint: np.ndarray,
) -> dict[str, np.ndarray]:
    """Differentiate every rho-dependent face conductance exactly."""
    bulk_gradient = np.zeros(geometry.rho.size, float)
    interface_gradient = np.zeros(geometry.rho.size, float)
    convection_gradient = np.zeros(geometry.rho.size, float)
    widths = (
        np.diff(geometry.x_edges_m),
        np.diff(geometry.y_edges_m),
        np.diff(geometry.z_edges_m),
    )
    kappa = geometry.kappa_W_mK
    rho_ids = geometry.design_rho_id
    dk = KAPPA_SIO2_W_MK - KAPPA_AIR_W_MK

    for axis in range(3):
        if axis == 0:
            left_theta, right_theta = theta[:-1, :, :], theta[1:, :, :]
            left_lambda, right_lambda = (
                adjoint[:-1, :, :],
                adjoint[1:, :, :],
            )
            left_k, right_k = kappa[:-1, :, :, 0], kappa[1:, :, :, 0]
            left_d = widths[0][:-1, None, None]
            right_d = widths[0][1:, None, None]
            area = widths[1][None, :, None] * widths[2][None, None, :]
            left_id, right_id = rho_ids[:-1, :, :], rho_ids[1:, :, :]
            resistance = geometry.interface_resistance_m2K_W["x"]
        elif axis == 1:
            left_theta, right_theta = theta[:, :-1, :], theta[:, 1:, :]
            left_lambda, right_lambda = (
                adjoint[:, :-1, :],
                adjoint[:, 1:, :],
            )
            left_k, right_k = kappa[:, :-1, :, 1], kappa[:, 1:, :, 1]
            left_d = widths[1][None, :-1, None]
            right_d = widths[1][None, 1:, None]
            area = widths[0][:, None, None] * widths[2][None, None, :]
            left_id, right_id = rho_ids[:, :-1, :], rho_ids[:, 1:, :]
            resistance = geometry.interface_resistance_m2K_W["y"]
        else:
            left_theta, right_theta = theta[:, :, :-1], theta[:, :, 1:]
            left_lambda, right_lambda = (
                adjoint[:, :, :-1],
                adjoint[:, :, 1:],
            )
            left_k, right_k = kappa[:, :, :-1, 2], kappa[:, :, 1:, 2]
            left_d = widths[2][None, None, :-1]
            right_d = widths[2][None, None, 1:]
            area = widths[0][:, None, None] * widths[1][None, :, None]
            left_id, right_id = rho_ids[:, :, :-1], rho_ids[:, :, 1:]
            resistance = geometry.interface_resistance_m2K_W["z"]
        area = np.broadcast_to(area, left_theta.shape)
        total_resistance = (
            0.5 * left_d / left_k
            + resistance
            + 0.5 * right_d / right_k
        )
        common = -(
            left_lambda - right_lambda
        ) * (left_theta - right_theta)
        left_active = left_id >= 0
        if np.any(left_active):
            dconductance = (
                area
                / total_resistance**2
                * 0.5
                * left_d
                / left_k**2
                * dk
            )
            np.add.at(
                bulk_gradient,
                left_id[left_active],
                (common * dconductance)[left_active],
            )
        right_active = right_id >= 0
        if np.any(right_active):
            dconductance = (
                area
                / total_resistance**2
                * 0.5
                * right_d
                / right_k**2
                * dk
            )
            np.add.at(
                bulk_gradient,
                right_id[right_active],
                (common * dconductance)[right_active],
            )

    # The flake/design z face has rho-dependent r=1/G in addition to the
    # upper half-cell conductivity already accumulated above.
    top_face = _face_before(geometry.z_edges_m, FLAKE_BOUNDS_Z_M[1])
    upper_id = rho_ids[:, :, top_face + 1]
    selected = upper_id >= 0
    if np.any(selected):
        dx, dy, dz = widths
        area = dx[:, None] * dy[None, :]
        lower_k = kappa[:, :, top_face, 2]
        upper_k = kappa[:, :, top_face + 1, 2]
        resistance = geometry.interface_resistance_m2K_W["z"][
            :, :, top_face
        ]
        total_resistance = (
            0.5 * dz[top_face] / lower_k
            + resistance
            + 0.5 * dz[top_face + 1] / upper_k
        )
        g = np.empty_like(resistance)
        g[selected] = 1.0 / resistance[selected]
        delta_g = (
            G_TAIRTE4_DEPOSITED_SIO2_W_M2K
            - G_TAIRTE4_AIR_W_M2K
        )
        drho_resistance = np.zeros_like(g)
        drho_resistance[selected] = -delta_g / g[selected] ** 2
        dconductance = -area / total_resistance**2 * drho_resistance
        common = -(
            adjoint[:, :, top_face] - adjoint[:, :, top_face + 1]
        ) * (
            theta[:, :, top_face] - theta[:, :, top_face + 1]
        )
        np.add.at(
            interface_gradient,
            upper_id[selected],
            (common * dconductance)[selected],
        )

    # Explicit top air/design surface convection includes the top half-cell.
    top_id = rho_ids[:, :, -1]
    selected = top_id >= 0
    if np.any(selected):
        dx, dy, dz = widths
        area = dx[:, None] * dy[None, :]
        top_k = kappa[:, :, -1, 2]
        resistance = (
            0.5 * dz[-1] / top_k + 1.0 / H_SIO2_AIR_W_M2K
        )
        dconductance = (
            area
            / resistance**2
            * 0.5
            * dz[-1]
            / top_k**2
            * dk
        )
        contribution = (
            -adjoint[:, :, -1] * theta[:, :, -1] * dconductance
        )
        np.add.at(
            convection_gradient,
            top_id[selected],
            contribution[selected],
        )
    shape = geometry.rho.shape
    components = {
        "bulk_k": bulk_gradient.reshape(shape),
        "interface_G": interface_gradient.reshape(shape),
        "top_convection_k": convection_gradient.reshape(shape),
    }
    components["total"] = sum(components.values())
    return components


def solve_explicit_forward(
    *,
    rho: np.ndarray,
    source_W_m3: np.ndarray | None = None,
    lateral_domain_m: float = 32.0e-6,
    si_depth_m: float = 20.0e-6,
    flake_span_m: float = 4.0e-6,
    cell_size_m: float = 50.0e-9,
) -> ExplicitThermalForward:
    geometry = build_explicit_geometry(
        rho,
        lateral_domain_m=lateral_domain_m,
        si_depth_m=si_depth_m,
        flake_span_m=flake_span_m,
        cell_size_m=cell_size_m,
    )
    source = (
        fixed_control_source(geometry)
        if source_W_m3 is None
        else np.asarray(source_W_m3, float)
    )
    if source.shape != geometry.material_id.shape:
        raise ValueError(
            f"source shape {source.shape} != {geometry.material_id.shape}"
        )
    system = assemble_steady_diagonal_kappa(
        x_edges_m=geometry.x_edges_m,
        y_edges_m=geometry.y_edges_m,
        z_edges_m=geometry.z_edges_m,
        kappa_W_mK=geometry.kappa_W_mK,
        dirichlet_temperature_K={
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
            "z_min": 0.0,
        },
        interface_resistance_m2K_W=(
            geometry.interface_resistance_m2K_W
        ),
        active_mask=np.ones(geometry.material_id.shape, bool),
        exposed_heat_transfer_W_m2K=H_SIO2_AIR_W_M2K,
        ambient_temperature_K=0.0,
    )
    solved = solve_assembled_thermal_system(
        system,
        source_W_m3=source,
        relative_tolerance=1.0e-11,
        max_iterations=12000,
    )
    theta_active = solved.temperature_K[system.active_mask]
    pte = build_uniform_45deg_weighting_pte(
        x_edges_m=geometry.x_edges_m,
        y_edges_m=geometry.y_edges_m,
        z_edges_m=geometry.z_edges_m,
        active_mask=system.active_mask,
        active_ids=system.active_ids,
        flake_mask=geometry.flake_mask,
    )
    objective = pte.evaluate_active(theta_active)
    return ExplicitThermalForward(
        geometry=geometry,
        system=system,
        solved=solved,
        pte=pte,
        theta_active_K=theta_active,
        objective_A=objective,
        source_W_m3=source,
    )


def evaluate_explicit_thermal(
    *,
    rho: np.ndarray,
    source_W_m3: np.ndarray | None = None,
    lateral_domain_m: float = 32.0e-6,
    si_depth_m: float = 20.0e-6,
    flake_span_m: float = 4.0e-6,
    cell_size_m: float = 50.0e-9,
) -> ExplicitThermalEvaluation:
    forward = solve_explicit_forward(
        rho=rho,
        source_W_m3=source_W_m3,
        lateral_domain_m=lateral_domain_m,
        si_depth_m=si_depth_m,
        flake_span_m=flake_span_m,
        cell_size_m=cell_size_m,
    )
    system = forward.system
    matrix = system.matrix_W_K
    diagonal = system.diagonal_W_K
    preconditioner = sparse_linalg.LinearOperator(
        matrix.shape,
        matvec=lambda vector: vector / diagonal,
    )
    iterations = 0

    def count_iteration(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    adjoint_active, info = sparse_linalg.cg(
        matrix.T,
        forward.pte.temperature_source_A_K,
        rtol=1.0e-11,
        atol=0.0,
        maxiter=12000,
        M=preconditioner,
        callback=count_iteration,
    )
    if info != 0:
        raise RuntimeError(f"thermal adjoint CG did not converge: info={info}")
    residual = (
        matrix.T @ adjoint_active
        - forward.pte.temperature_source_A_K
    )
    residual_relative = float(
        np.linalg.norm(residual)
        / max(
            np.linalg.norm(forward.pte.temperature_source_A_K),
            np.finfo(float).tiny,
        )
    )
    gradient_q = np.asarray(
        system.source_volume_operator_m3.T @ adjoint_active
    ).reshape(-1)
    theta = system.full_field(forward.theta_active_K)
    adjoint = system.full_field(adjoint_active)
    gradient_components = _rho_conductance_gradient(
        geometry=forward.geometry,
        theta=theta,
        adjoint=adjoint,
    )
    return ExplicitThermalEvaluation(
        geometry=forward.geometry,
        system=system,
        solved=forward.solved,
        pte=forward.pte,
        theta_active_K=forward.theta_active_K,
        adjoint_active=np.asarray(adjoint_active),
        objective_A=forward.objective_A,
        gradient_rho_A=gradient_components["total"],
        gradient_bulk_k_A=gradient_components["bulk_k"],
        gradient_interface_G_A=gradient_components["interface_G"],
        gradient_top_convection_k_A=gradient_components[
            "top_convection_k"
        ],
        gradient_Q_active_A_m3_W=gradient_q,
        source_W_m3=forward.source_W_m3,
        adjoint_linear_residual_relative=residual_relative,
        adjoint_iterations=iterations,
    )
