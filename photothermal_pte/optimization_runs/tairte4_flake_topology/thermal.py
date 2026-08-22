"""Explicit 3-D thermal model for the TaIrTe4-to-void topology.

The optical design changes the TaIrTe4 sheet itself.  This module therefore
does not reuse Run 009's upper-SiO2 ``G(rho)`` law.  Air, bottom SiO2, and Si
are explicit FVM cells.  At the bottom sheet face, gray density is an
area-fraction relaxation of two parallel paths: TaIrTe4/SiO2 with finite G,
and air/SiO2 without a TaIrTe4 contact.  The endpoint face conductances are
exact by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

from photothermal_pte.finite_inverse_design.finite_q_mapping import (
    apply_material_intersection_density_separable,
    nodal_control_volume_edges,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import (
    CONTRACT,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
FVM = REPOSITORY / "photothermal_pte" / "validation" / "photothermal_stage1"
if str(FVM) not in sys.path:
    sys.path.insert(0, str(FVM))

from anisotropic_heat_fvm import (  # noqa: E402
    AssembledThermalSystem,
    assemble_steady_diagonal_kappa,
)


K_AIR_W_MK = 0.026
K_SIO2_W_MK = 1.38
K_SI_W_MK = 145.0
# Room-temperature bulk value used only when an explicit Au terminal layer is
# requested by a forward-response calculation.
K_AU_W_MK = 317.0
# Lumerical x=b, y=a, z=c.
K_TAIRTE4_XYZ_W_MK = np.asarray((3.8, 14.4, 1.0), dtype=np.float64)
TAIRTE4_SIO2_INTERFACE_CONDUCTANCE_W_M2K = {
    "thermally_grown": 7.37e6,
    "evaporated": 7.37e4,
}


def selected_tairte4_sio2_interface() -> tuple[str, float]:
    """Return the explicitly selected paper-derived interface scenario."""

    scenario = os.environ.get(
        "TAIRTE4_SIO2_INTERFACE_SCENARIO", "thermally_grown"
    ).strip()
    if scenario not in TAIRTE4_SIO2_INTERFACE_CONDUCTANCE_W_M2K:
        allowed = ", ".join(sorted(TAIRTE4_SIO2_INTERFACE_CONDUCTANCE_W_M2K))
        raise RuntimeError(
            f"unknown TAIRTE4_SIO2_INTERFACE_SCENARIO={scenario!r}; "
            f"expected one of: {allowed}"
        )
    return scenario, float(TAIRTE4_SIO2_INTERFACE_CONDUCTANCE_W_M2K[scenario])


TAIRTE4_SIO2_INTERFACE_SCENARIO, G_TAIRTE4_SIO2_W_M2K = (
    selected_tairte4_sio2_interface()
)
G_SIO2_SI_W_M2K = 1.1e9
TOP_AIR_CONVECTION_W_M2K = 10.0
THERMAL_SI_DEPTH_M = 20.0e-6
THERMAL_AIR_HEIGHT_M = 2.0e-6


def thermal_interface_contract() -> dict[str, object]:
    """Serializable interface provenance for coupled objective evaluations."""

    return {
        "model": "explicit_3d_bulk_SiO2_and_Si_with_finite_interfaces",
        "TaIrTe4_SiO2_scenario": TAIRTE4_SIO2_INTERFACE_SCENARIO,
        "G_TaIrTe4_SiO2_W_m2K": G_TAIRTE4_SIO2_W_M2K,
        "G_SiO2_Si_W_m2K": G_SIO2_SI_W_M2K,
        "interpretation": (
            "paper-derived interface scenario; bulk SiO2 geometry and "
            "optical material are unchanged"
        ),
    }


def _piecewise_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    negative_outer = np.asarray((-32, -28, -24, -20, -16, -14), float) * 1e-6
    negative_shoulder = np.arange(-14.0, -12.0, 0.25) * 1e-6
    core = np.arange(-12.0, 12.0 + 0.05, 0.1) * 1e-6
    positive_shoulder = np.arange(12.25, 14.0 + 0.125, 0.25) * 1e-6
    positive_outer = np.asarray((16, 20, 24, 28, 32), float) * 1e-6
    lateral = np.unique(
        np.concatenate((negative_outer, negative_shoulder, core, positive_shoulder, positive_outer))
    )
    z = np.asarray(
        (
            -20.0, -12.0, -8.0, -5.0, -3.0, -2.0, -1.25,
            -0.8, -0.55, -0.385, -0.30, -0.20, -0.10,
            -0.09, -0.08, -0.07, -0.06, -0.05,
            -0.04, -0.03, -0.02, -0.01, 0.0,
            0.01, 0.02, 0.05, 0.10, 0.20, 0.40,
            0.70, 1.0, 1.25, 1.50, 2.0,
        ),
        float,
    ) * 1e-6
    return lateral, lateral.copy(), z


def _centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def _face_before(edges: np.ndarray, value: float) -> int:
    match = np.flatnonzero(np.isclose(edges, value, rtol=0.0, atol=2e-18))
    if match.size != 1 or match[0] == 0:
        raise RuntimeError(f"required face {value:.9e} m is absent")
    return int(match[0] - 1)


def nodal_to_cell(density: np.ndarray) -> np.ndarray:
    values = np.asarray(density, dtype=np.float64)
    if values.shape != CONTRACT.design_node_shape:
        raise ValueError(f"density shape {values.shape} != {CONTRACT.design_node_shape}")
    return 0.25 * (
        values[:-1, :-1] + values[1:, :-1]
        + values[:-1, 1:] + values[1:, 1:]
    )


def nodal_to_cell_transpose(values: np.ndarray) -> np.ndarray:
    cell = np.asarray(values, dtype=np.float64)
    expected = CONTRACT.design_intervals
    if cell.shape != expected:
        raise ValueError(f"cell values shape {cell.shape} != {expected}")
    result = np.zeros(CONTRACT.design_node_shape, dtype=np.float64)
    result[:-1, :-1] += 0.25 * cell
    result[1:, :-1] += 0.25 * cell
    result[:-1, 1:] += 0.25 * cell
    result[1:, 1:] += 0.25 * cell
    return result


@dataclass(frozen=True)
class ThermalState:
    edges_m: tuple[np.ndarray, np.ndarray, np.ndarray]
    widths_m: tuple[np.ndarray, np.ndarray, np.ndarray]
    system: AssembledThermalSystem
    kappa_W_mK: np.ndarray
    interface_resistance_m2K_W: dict[str, np.ndarray]
    material_id: np.ndarray
    masks: dict[str, np.ndarray]
    rho_nodal: np.ndarray
    rho_cell: np.ndarray
    rho_id: np.ndarray
    dphi_drho_cell: np.ndarray
    gray_exponent: float
    bottom_face: int
    bottom_air_path_resistance_m2K_W: float
    bottom_tairte4_path_resistance_m2K_W: float


def build_state(
    rho_nodal: np.ndarray,
    *,
    gray_exponent: float = 1.0,
    au_contact_axis: str | None = None,
    au_thickness_m: float = 50.0e-9,
    au_tairte4_interface_conductance_W_m2K: float = 19.89e6,
) -> ThermalState:
    CONTRACT.validate()
    rho_nodal = CONTRACT.apply_fixed_contact_density(
        np.asarray(rho_nodal, dtype=np.float64)
    )
    if rho_nodal.shape != CONTRACT.design_node_shape:
        raise ValueError("invalid design density shape")
    if np.any((rho_nodal < 0.0) | (rho_nodal > 1.0)):
        raise ValueError("density must remain in [0,1]")
    if gray_exponent <= 0.0:
        raise ValueError("gray exponent must be positive")
    if au_contact_axis not in (None, "x", "y"):
        raise ValueError("au_contact_axis must be None, 'x', or 'y'")
    if au_contact_axis is not None:
        if not np.isclose(au_thickness_m, 50.0e-9, rtol=0.0, atol=1.0e-18):
            raise ValueError("explicit Au model requires the paper's 50 nm thickness")
        if au_tairte4_interface_conductance_W_m2K <= 0.0:
            raise ValueError("Au/TaIrTe4 interface conductance must be positive")
    rho_cell = nodal_to_cell(rho_nodal)
    phi_cell = rho_cell**gray_exponent
    # d(rho**p)/d(rho) is exactly one at rho=0 when p=1.  The former
    # ``where(rho > 0, ..., 0)`` silently set that endpoint derivative to
    # zero, even for the production linear gray law.  Keep the mathematically
    # exact endpoint and use the limiting zero derivative only for p>1.
    if np.isclose(gray_exponent, 1.0, rtol=0.0, atol=1.0e-15):
        dphi = np.ones_like(rho_cell)
    else:
        dphi = gray_exponent * np.where(
            rho_cell > 0.0, rho_cell ** (gray_exponent - 1.0), 0.0
        )

    edges = _piecewise_edges()
    widths = tuple(np.diff(value) for value in edges)
    centers = tuple(_centers(value) for value in edges)
    shape = tuple(value.size for value in centers)
    xx, yy, zz = np.meshgrid(*centers, indexing="ij")
    si = zz < -0.385e-6
    sio2 = (zz >= -0.385e-6) & (zz < -0.100e-6)
    flake_z = (zz >= -0.100e-6) & (zz < 0.0)
    flake_xy = (np.abs(xx) < 12.0e-6) & (np.abs(yy) < 12.0e-6)
    x_bounds = CONTRACT.design_bounds_m["x"]
    y_bounds = CONTRACT.design_bounds_m["y"]
    design_xy = (
        (xx >= x_bounds[0]) & (xx < x_bounds[1])
        & (yy >= y_bounds[0]) & (yy < y_bounds[1])
    )
    fixed_flake = flake_z & flake_xy & ~design_xy
    design_flake = flake_z & design_xy
    if au_contact_axis == "x":
        contact_xy = (
            (np.abs(xx) < 12.0e-6)
            & (np.abs(yy) < 12.0e-6)
            & (np.abs(xx) >= 10.0e-6)
        )
    elif au_contact_axis == "y":
        contact_xy = (
            (np.abs(xx) < 12.0e-6)
            & (np.abs(yy) < 12.0e-6)
            & (np.abs(yy) >= 10.0e-6)
        )
    else:
        contact_xy = np.zeros_like(xx, dtype=bool)
    au = (
        contact_xy
        & (zz >= 0.0)
        & (zz < au_thickness_m)
    )
    air = ~(si | sio2 | fixed_flake | design_flake | au)
    masks = {
        "Si": si,
        "SiO2": sio2,
        "air": air,
        "fixed_TaIrTe4": fixed_flake,
        "design_effective": design_flake,
        "Au_electrodes": au,
        "flake_support": flake_z & flake_xy,
        "physical_absorbing_support": si | sio2 | (flake_z & flake_xy) | au,
    }
    material = np.zeros(shape, dtype=np.uint8)
    material[air] = 1
    material[si] = 2
    material[sio2] = 3
    material[fixed_flake] = 4
    material[design_flake] = 5
    material[au] = 6

    x_design = np.flatnonzero((centers[0] >= x_bounds[0]) & (centers[0] < x_bounds[1]))
    y_design = np.flatnonzero((centers[1] >= y_bounds[0]) & (centers[1] < y_bounds[1]))
    z_flake = np.flatnonzero((centers[2] >= -0.100e-6) & (centers[2] < 0.0))
    if (x_design.size, y_design.size) != rho_cell.shape or z_flake.size != 10:
        raise RuntimeError("thermal design/flake grid does not match 100/10 nm contract")

    kappa = np.full((*shape, 3), K_AIR_W_MK, dtype=np.float64)
    kappa[si] = K_SI_W_MK
    kappa[sio2] = K_SIO2_W_MK
    kappa[fixed_flake] = K_TAIRTE4_XYZ_W_MK
    kappa[au] = K_AU_W_MK
    effective = K_AIR_W_MK + phi_cell[..., None] * (
        K_TAIRTE4_XYZ_W_MK[None, None, :] - K_AIR_W_MK
    )
    for z_index in z_flake:
        kappa[x_design[:, None], y_design[None, :], z_index, :] = effective

    rho_id = np.full(shape, -1, dtype=np.int64)
    ids = np.arange(rho_cell.size, dtype=np.int64).reshape(rho_cell.shape)
    for z_index in z_flake:
        rho_id[np.ix_(x_design, y_design, [z_index])] = ids[:, :, None]

    rx = np.zeros((shape[0] - 1, shape[1], shape[2]), dtype=np.float64)
    ry = np.zeros((shape[0], shape[1] - 1, shape[2]), dtype=np.float64)
    rz = np.zeros((shape[0], shape[1], shape[2] - 1), dtype=np.float64)
    sio2_si_face = _face_before(edges[2], -0.385e-6)
    rz[:, :, sio2_si_face] = 1.0 / G_SIO2_SI_W_M2K

    bottom = _face_before(edges[2], -0.100e-6)
    lower_half = 0.5 * widths[2][bottom] / K_SIO2_W_MK
    air_path = lower_half + 0.5 * widths[2][bottom + 1] / K_AIR_W_MK
    tair_path = (
        lower_half
        + 1.0 / G_TAIRTE4_SIO2_W_M2K
        + 0.5 * widths[2][bottom + 1] / K_TAIRTE4_XYZ_W_MK[2]
    )
    x_flake = np.flatnonzero((centers[0] >= -12.0e-6) & (centers[0] < 12.0e-6))
    y_flake = np.flatnonzero((centers[1] >= -12.0e-6) & (centers[1] < 12.0e-6))
    phi_full = np.ones((x_flake.size, y_flake.size), dtype=np.float64)
    x_offset = int(np.flatnonzero(x_flake == x_design[0])[0])
    y_offset = int(np.flatnonzero(y_flake == y_design[0])[0])
    phi_full[
        x_offset : x_offset + x_design.size,
        y_offset : y_offset + y_design.size,
    ] = phi_cell
    upper_k = kappa[np.ix_(x_flake, y_flake, [bottom + 1], [2])][:, :, 0, 0]
    face_conductance_per_area = (1.0 - phi_full) / air_path + phi_full / tair_path
    equivalent = (
        1.0 / face_conductance_per_area
        - lower_half
        - 0.5 * widths[2][bottom + 1] / upper_k
    )
    if np.any(equivalent < -1e-15):
        raise RuntimeError("bottom gray parallel-path relaxation produced negative resistance")
    rz[np.ix_(x_flake, y_flake, [bottom])] = np.maximum(equivalent, 0.0)[:, :, None]

    if au_contact_axis is not None:
        top_flake_face = _face_before(edges[2], 0.0)
        x_contact = np.flatnonzero(
            (centers[0] >= -12.0e-6)
            & (centers[0] < 12.0e-6)
            & (
                (np.abs(centers[0]) >= 10.0e-6)
                if au_contact_axis == "x"
                else np.ones_like(centers[0], dtype=bool)
            )
        )
        y_contact = np.flatnonzero(
            (centers[1] >= -12.0e-6)
            & (centers[1] < 12.0e-6)
            & (
                (np.abs(centers[1]) >= 10.0e-6)
                if au_contact_axis == "y"
                else np.ones_like(centers[1], dtype=bool)
            )
        )
        rz[np.ix_(x_contact, y_contact, [top_flake_face])] = (
            1.0 / au_tairte4_interface_conductance_W_m2K
        )

    system = assemble_steady_diagonal_kappa(
        x_edges_m=edges[0],
        y_edges_m=edges[1],
        z_edges_m=edges[2],
        kappa_W_mK=kappa,
        active_mask=np.ones(shape, dtype=bool),
        interface_resistance_m2K_W={"x": rx, "y": ry, "z": rz},
        dirichlet_temperature_K={
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
            "z_min": 0.0,
        },
        surface_robin_heat_transfer_W_m2K={"z_max": TOP_AIR_CONVECTION_W_M2K},
        surface_robin_temperature_K={"z_max": 0.0},
    )
    return ThermalState(
        edges_m=edges,
        widths_m=widths,
        system=system,
        kappa_W_mK=kappa,
        interface_resistance_m2K_W={"x": rx, "y": ry, "z": rz},
        material_id=material,
        masks=masks,
        rho_nodal=rho_nodal.copy(),
        rho_cell=rho_cell,
        rho_id=rho_id,
        dphi_drho_cell=dphi,
        gray_exponent=float(gray_exponent),
        bottom_face=bottom,
        bottom_air_path_resistance_m2K_W=float(air_path),
        bottom_tairte4_path_resistance_m2K_W=float(tair_path),
    )


def map_native_q(native_q: np.lib.npyio.NpzFile, state: ThermalState) -> tuple[np.ndarray, dict[str, object]]:
    """Deposit literal material-intersection Q without nearest-cell relocation."""
    mapped = np.zeros(state.system.shape, dtype=np.float64)
    records: dict[str, object] = {}
    native_total = 0.0
    attributed_total = 0.0
    for component in "xyz":
        source = np.asarray(native_q[f"Q{component}_W_m3"], dtype=np.float64)
        source_edges = tuple(
            nodal_control_volume_edges(
                np.asarray(native_q[f"Q{component}_{axis}_m"], dtype=np.float64)
            )
            for axis in "xyz"
        )
        source_volume = (
            np.diff(source_edges[0])[:, None, None]
            * np.diff(source_edges[1])[None, :, None]
            * np.diff(source_edges[2])[None, None, :]
        )
        native_power = float(np.sum(source * source_volume))
        density, overlap, metrics = apply_material_intersection_density_separable(
            source_density=source,
            source_edges_m=source_edges,
            target_edges_m=state.edges_m,
            target_material_support_mask=state.masks["physical_absorbing_support"],
        )
        mapped += density
        native_total += native_power
        attributed_total += float(metrics["material_attributed_source_power_W"])
        records[component] = {
            "native_power_W": native_power,
            **metrics,
            "source_cells_with_zero_material_overlap": int(np.count_nonzero(overlap == 0.0)),
        }
    target_volume = state.system.cell_volume_m3
    target_power = float(np.sum(mapped * target_volume))
    return mapped, {
        "method": "literal optical-dual-cell/thermal-material intersection",
        "full_cell_power_forced_into_TaIrTe4": False,
        "nearest_cell_relocation": False,
        "clipping_smoothing_gain_or_rescaling": False,
        "native_total_power_W": native_total,
        "material_attributed_power_W": attributed_total,
        "target_power_W": target_power,
        "relative_mapping_error": abs(target_power - attributed_total)
        / max(abs(attributed_total), np.finfo(float).tiny),
        "material_attributed_fraction_of_native_Q": attributed_total
        / max(abs(native_total), np.finfo(float).tiny),
        "components": records,
    }


def flake_cell_temperature(state: ThermalState, active_temperature: np.ndarray) -> np.ndarray:
    full = state.system.full_field(active_temperature)
    x = _centers(state.edges_m[0])
    y = _centers(state.edges_m[1])
    z = _centers(state.edges_m[2])
    ix = np.flatnonzero((x >= -12e-6) & (x < 12e-6))
    iy = np.flatnonzero((y >= -12e-6) & (y < 12e-6))
    iz = np.flatnonzero((z >= -0.1e-6) & (z < 0.0))
    weights = state.widths_m[2][iz]
    return np.tensordot(full[np.ix_(ix, iy, iz)], weights / np.sum(weights), axes=(2, 0))


def cell_to_node(cell: np.ndarray) -> np.ndarray:
    values = np.asarray(cell, dtype=np.float64)
    if values.shape != (240, 240):
        raise ValueError("flake cell temperature must be 240x240")
    result = np.zeros((241, 241), dtype=np.float64)
    weight = np.zeros_like(result)
    for di in (0, 1):
        for dj in (0, 1):
            result[di : di + 240, dj : dj + 240] += values
            weight[di : di + 240, dj : dj + 240] += 1.0
    return result / weight


def cell_to_node_transpose(node: np.ndarray) -> np.ndarray:
    values = np.asarray(node, dtype=np.float64)
    if values.shape != (241, 241):
        raise ValueError("flake nodal sensitivity must be 241x241")
    node_weight = np.ones_like(values)
    node_weight[1:-1, :] *= 2.0
    node_weight[:, 1:-1] *= 2.0
    weighted = values / node_weight
    return (
        weighted[:-1, :-1] + weighted[1:, :-1]
        + weighted[:-1, 1:] + weighted[1:, 1:]
    )


def flake_temperature_transpose(state: ThermalState, nodal_sensitivity: np.ndarray) -> np.ndarray:
    cell_sensitivity = cell_to_node_transpose(nodal_sensitivity)
    full = np.zeros(state.system.shape, dtype=np.float64)
    x = _centers(state.edges_m[0])
    y = _centers(state.edges_m[1])
    z = _centers(state.edges_m[2])
    ix = np.flatnonzero((x >= -12e-6) & (x < 12e-6))
    iy = np.flatnonzero((y >= -12e-6) & (y < 12e-6))
    iz = np.flatnonzero((z >= -0.1e-6) & (z < 0.0))
    weights = state.widths_m[2][iz]
    for local, z_index in enumerate(iz):
        full[np.ix_(ix, iy, [z_index])] = (
            cell_sensitivity[:, :, None] * weights[local] / np.sum(weights)
        )
    return full[state.system.active_mask]


def thermal_density_gradient(
    state: ThermalState,
    temperature_active: np.ndarray,
    adjoint_active: np.ndarray,
) -> np.ndarray:
    """Exact discrete -lambda^T(dK/drho)T for the thermal operator."""
    temperature = state.system.full_field(temperature_active)
    adjoint = state.system.full_field(adjoint_active)
    count = state.rho_cell.size
    gradient = np.zeros(count, dtype=np.float64)
    dphi = state.dphi_drho_cell.reshape(-1)
    dk_xyz = K_TAIRTE4_XYZ_W_MK - K_AIR_W_MK
    widths = state.widths_m

    for axis in range(3):
        if axis == 0:
            lt, rt = temperature[:-1], temperature[1:]
            ll, rl = adjoint[:-1], adjoint[1:]
            lk, rk = state.kappa_W_mK[:-1, :, :, 0], state.kappa_W_mK[1:, :, :, 0]
            ld, rd = widths[0][:-1, None, None], widths[0][1:, None, None]
            area = widths[1][None, :, None] * widths[2][None, None, :]
            lid, rid = state.rho_id[:-1], state.rho_id[1:]
            resistance = state.interface_resistance_m2K_W["x"]
        elif axis == 1:
            lt, rt = temperature[:, :-1], temperature[:, 1:]
            ll, rl = adjoint[:, :-1], adjoint[:, 1:]
            lk, rk = state.kappa_W_mK[:, :-1, :, 1], state.kappa_W_mK[:, 1:, :, 1]
            ld, rd = widths[1][None, :-1, None], widths[1][None, 1:, None]
            area = widths[0][:, None, None] * widths[2][None, None, :]
            lid, rid = state.rho_id[:, :-1], state.rho_id[:, 1:]
            resistance = state.interface_resistance_m2K_W["y"]
        else:
            lt, rt = temperature[:, :, :-1], temperature[:, :, 1:]
            ll, rl = adjoint[:, :, :-1], adjoint[:, :, 1:]
            lk, rk = state.kappa_W_mK[:, :, :-1, 2], state.kappa_W_mK[:, :, 1:, 2]
            ld, rd = widths[2][None, None, :-1], widths[2][None, None, 1:]
            area = widths[0][:, None, None] * widths[1][None, :, None]
            lid, rid = state.rho_id[:, :, :-1], state.rho_id[:, :, 1:]
            resistance = state.interface_resistance_m2K_W["z"]
        area = np.broadcast_to(area, lt.shape)
        total_r = 0.5 * ld / lk + resistance + 0.5 * rd / rk
        common = -(ll - rl) * (lt - rt)
        left = lid >= 0
        right = rid >= 0
        # The bottom contact face is replaced below by the exact parallel-path
        # derivative; its k-dependent equivalent resistance cancels the
        # upper-half-cell derivative.
        if axis == 2:
            left[:, :, state.bottom_face] = False
            right[:, :, state.bottom_face] = False
        left_dk = np.zeros_like(lk)
        right_dk = np.zeros_like(rk)
        left_dk[left] = dk_xyz[axis] * dphi[lid[left]]
        right_dk[right] = dk_xyz[axis] * dphi[rid[right]]
        left_dg = area / total_r**2 * 0.5 * ld / lk**2 * left_dk
        right_dg = area / total_r**2 * 0.5 * rd / rk**2 * right_dk
        np.add.at(gradient, lid[left], (common * left_dg)[left])
        np.add.at(gradient, rid[right], (common * right_dg)[right])

    bottom = state.bottom_face
    upper_id = state.rho_id[:, :, bottom + 1]
    selected = upper_id >= 0
    area = widths[0][:, None] * widths[1][None, :]
    dg_dphi = area * (
        1.0 / state.bottom_tairte4_path_resistance_m2K_W
        - 1.0 / state.bottom_air_path_resistance_m2K_W
    )
    common = -(
        adjoint[:, :, bottom] - adjoint[:, :, bottom + 1]
    ) * (
        temperature[:, :, bottom] - temperature[:, :, bottom + 1]
    )
    face_dphi = np.zeros(upper_id.shape, dtype=np.float64)
    face_dphi[selected] = dphi[upper_id[selected]]
    np.add.at(
        gradient,
        upper_id[selected],
        (common * dg_dphi * face_dphi)[selected],
    )
    return nodal_to_cell_transpose(gradient.reshape(state.rho_cell.shape))


def boundary_energy_error(
    state: ThermalState, temperature_active: np.ndarray, source_power_W: np.ndarray
) -> tuple[float, dict[str, float]]:
    powers = {
        name: float(np.sum(g * temperature_active[cell_ids]))
        for name, (cell_ids, g, _) in state.system.boundary_terms.items()
    }
    source = float(np.sum(source_power_W))
    error = abs(sum(powers.values()) - source) / max(
        abs(source), max((abs(value) for value in powers.values()), default=0.0), np.finfo(float).tiny
    )
    return float(error), powers
