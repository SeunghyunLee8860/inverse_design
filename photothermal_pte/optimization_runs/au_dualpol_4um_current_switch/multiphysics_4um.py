"""Explicit thermal and floating-Au electrical operators for the 4 um design.

Coordinates are fixed to Lumerical x=b and y=a.  The fixed TaIrTe4 flake is
16 x 16 x 0.1 um, the floating Au design window is 8 x 8 x 0.05 um, and both
operators use a 100 nm lateral grid over the physical device.  Optical
electrodes are absent; the electrical readout is represented by psi=0 on the
left flake boundary and psi=1 on the right flake boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    au_material_fraction,
    d_au_material_fraction_drho,
)
from photothermal_pte.optimization_runs.au_on_fixed_tairte4_validation.material_model import (
    AU_BULK_ELECTRICAL_CONDUCTIVITY_S_M,
)
from photothermal_pte.optimization_runs.cuda_thermal_adjoint import PersistentCudaCSR


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
FVM_PATH = (
    REPOSITORY
    / "photothermal_pte/validation/photothermal_stage1/anisotropic_heat_fvm.py"
)
OVERLAP_PATH = (
    HERE.parent
    / "au_on_fixed_tairte4_validation/64_validate_fdtdx_material_overlap_thermal_remap.py"
)

STEP_M = CONTRACT.design_pitch_m
N_TA = int(round(CONTRACT.flake_span_x_m / STEP_M))
N_DESIGN = CONTRACT.design_shape[0]
DESIGN_OFFSET = (N_TA - N_DESIGN) // 2
TA_THICKNESS_M = CONTRACT.flake_thickness_m
AU_THICKNESS_M = CONTRACT.design_thickness_m

K_AIR_W_MK = 0.026
K_SIO2_W_MK = 1.38
K_SI_W_MK = 145.0
K_TA_XYZ_W_MK = np.asarray((3.8, 14.4, 1.0), dtype=np.float64)
K_AU_W_MK = 317.0
G_SIO2_SI_W_M2K = 1.1e9
G_TA_AIR_W_M2K = CONTRACT.g_ta_air_W_m2K
G_TA_SIO2_W_M2K = CONTRACT.g_ta_sio2_W_m2K
G_AU_TA_W_M2K = CONTRACT.g_au_ta_W_m2K
TOP_AIR_CONVECTION_W_M2K = 10.0

# x=b, y=a.
SIGMA_TA_XY_S_M = np.asarray((1.10e5, 4.91e5), dtype=np.float64)
SEEBECK_TA_XY_V_K = np.asarray((27.0e-6, -6.0e-6), dtype=np.float64)
SIGMA_AU_S_M = float(AU_BULK_ELECTRICAL_CONDUCTIVITY_S_M)
SIGMA_FLOOR_FRACTION = 1.0e-8
CONTACT_FLOOR_FRACTION = 1.0e-10
ELECTRICAL_CONTACT_S_M2 = CONTRACT.electrical_contact_S_m2


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def _face_before(edges: np.ndarray, value: float) -> int:
    match = np.flatnonzero(np.isclose(edges, value, rtol=0.0, atol=2e-18))
    if match.size != 1 or match[0] == 0:
        raise RuntimeError(f"required face {value:.9e} m is absent")
    return int(match[0] - 1)


def _refine_edges(edges: np.ndarray, factor: int) -> np.ndarray:
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
        raise ValueError("thermal refinement factor must be a positive integer")
    value = np.asarray(edges, dtype=np.float64)
    if (
        value.ndim != 1
        or value.size < 2
        or not np.all(np.isfinite(value))
        or np.any(np.diff(value) <= 0.0)
    ):
        raise ValueError("thermal edges must be finite and strictly increasing")
    if factor == 1:
        return value.copy()
    refined = [
        np.linspace(value[index], value[index + 1], factor, endpoint=False)
        for index in range(value.size - 1)
    ]
    return np.concatenate((*refined, value[-1:]))


def thermal_edges(
    z_refinement_factor: int = 1,
    *,
    xy_refinement_factor: int = 1,
    lateral_half_span_um: int = 32,
    substrate_depth_um: int = 20,
    top_air_height_um: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lateral_half_span_um not in (32, 48, 64):
        raise ValueError("thermal lateral half-span must be 32, 48, or 64 um")
    if substrate_depth_um not in (20, 30, 40):
        raise ValueError("thermal substrate depth must be 20, 30, or 40 um")
    if top_air_height_um not in (2.0, 3.0, 4.0):
        raise ValueError("thermal top-air height must be 2, 3, or 4 um")
    negative_extension_um = -np.arange(
        float(lateral_half_span_um), 32.0, -4.0
    )
    positive_extension_um = np.arange(
        36.0, lateral_half_span_um + 2.0, 4.0
    )
    negative_outer = np.asarray((-32, -28, -24, -20, -16, -14), float) * 1e-6
    negative_shoulder = np.arange(-14.0, -12.0, 0.25) * 1e-6
    core = np.arange(-12.0, 12.0 + 0.05, 0.1) * 1e-6
    positive_shoulder = np.arange(12.25, 14.0 + 0.125, 0.25) * 1e-6
    positive_outer = np.asarray((16, 20, 24, 28, 32), float) * 1e-6
    lateral = np.unique(
        np.concatenate(
            (
                negative_extension_um * 1e-6,
                negative_outer,
                negative_shoulder,
                core,
                positive_shoulder,
                positive_outer,
                positive_extension_um * 1e-6,
            )
        )
    )
    base_z_um = np.asarray(
        (
            -20.0, -12.0, -8.0, -5.0, -3.0, -2.0, -1.25,
            -0.8, -0.55, -0.385, -0.30, -0.20, -0.10,
            -0.09, -0.08, -0.07, -0.06, -0.05,
            -0.04, -0.03, -0.02, -0.01, 0.0,
            0.01, 0.02, 0.05, 0.10, 0.20, 0.40,
            0.70, 1.0, 1.25, 1.50, 2.0,
        ),
        float,
    )
    substrate_extension_um = -np.arange(
        float(substrate_depth_um), 20.0, -10.0
    )
    top_extension_um = np.arange(2.5, top_air_height_um + 0.25, 0.5)
    z = np.concatenate(
        (substrate_extension_um, base_z_um, top_extension_um)
    ) * 1e-6
    refined_lateral = _refine_edges(lateral, xy_refinement_factor)
    return (
        refined_lateral,
        refined_lateral.copy(),
        _refine_edges(z, z_refinement_factor),
    )


@dataclass(frozen=True)
class ThermalState:
    edges: tuple[np.ndarray, np.ndarray, np.ndarray]
    widths: tuple[np.ndarray, np.ndarray, np.ndarray]
    centers: tuple[np.ndarray, np.ndarray, np.ndarray]
    system: object
    kappa: np.ndarray
    masks: dict[str, np.ndarray]
    interface_resistance: dict[str, np.ndarray]
    rho: np.ndarray
    material_fraction: np.ndarray
    faces: dict[str, int]


def build_thermal_state(
    rho: np.ndarray,
    *,
    z_refinement_factor: int = 1,
    xy_refinement_factor: int = 1,
    lateral_half_span_um: int = 32,
    substrate_depth_um: int = 20,
    top_air_height_um: float = 2.0,
) -> ThermalState:
    density = np.asarray(rho, dtype=np.float64)
    if density.shape != CONTRACT.design_shape or np.any((density < 0) | (density > 1)):
        raise ValueError("rho must be an 80x80 physical density in [0,1]")
    fvm = _load(FVM_PATH, "au_dualpol_4um_fvm")
    edges = thermal_edges(
        z_refinement_factor,
        xy_refinement_factor=xy_refinement_factor,
        lateral_half_span_um=lateral_half_span_um,
        substrate_depth_um=substrate_depth_um,
        top_air_height_um=top_air_height_um,
    )
    widths = tuple(np.diff(axis) for axis in edges)
    centers = tuple(_centers(axis) for axis in edges)
    x, y, z = centers
    shape = tuple(len(value) for value in centers)
    x_ta = (x >= -8e-6) & (x < 8e-6)
    y_ta = (y >= -8e-6) & (y < 8e-6)
    z_ta = (z >= -0.1e-6) & (z < 0.0)
    x_au = (x >= -4e-6) & (x < 4e-6)
    y_au = (y >= -4e-6) & (y < 4e-6)
    z_au = (z >= 0.0) & (z < 0.05e-6)
    z_sio2 = (z >= -0.385e-6) & (z < -0.1e-6)
    z_si = z < -0.385e-6
    ta = x_ta[:, None, None] & y_ta[None, :, None] & z_ta[None, None, :]
    au = x_au[:, None, None] & y_au[None, :, None] & z_au[None, None, :]
    sio2 = np.broadcast_to(z_sio2[None, None, :], shape)
    si = np.broadcast_to(z_si[None, None, :], shape)
    ix_au, iy_au, iz_au = map(np.flatnonzero, (x_au, y_au, z_au))
    expected_ta = N_TA * xy_refinement_factor
    expected_design = tuple(
        value * xy_refinement_factor for value in CONTRACT.design_shape
    )
    if (np.count_nonzero(x_ta), np.count_nonzero(y_ta)) != (
        expected_ta,
        expected_ta,
    ):
        raise RuntimeError(
            "TaIrTe4 thermal footprint does not match xy refinement"
        )
    if (ix_au.size, iy_au.size) != expected_design:
        raise RuntimeError("Au thermal footprint does not match design density")

    fraction = np.asarray(au_material_fraction(density), dtype=np.float64)
    refined_fraction = np.repeat(
        np.repeat(fraction, xy_refinement_factor, axis=0),
        xy_refinement_factor,
        axis=1,
    )
    k_au = K_AIR_W_MK + refined_fraction * (K_AU_W_MK - K_AIR_W_MK)
    kappa = np.full((*shape, 3), K_AIR_W_MK, dtype=np.float64)
    kappa[si] = K_SI_W_MK
    kappa[sio2] = K_SIO2_W_MK
    kappa[ta] = K_TA_XYZ_W_MK
    for iz in iz_au:
        for component in range(3):
            kappa[np.ix_(ix_au, iy_au, [iz], [component])] = k_au[:, :, None, None]

    rx = np.zeros((shape[0] - 1, shape[1], shape[2]), dtype=np.float64)
    ry = np.zeros((shape[0], shape[1] - 1, shape[2]), dtype=np.float64)
    rz = np.zeros((shape[0], shape[1], shape[2] - 1), dtype=np.float64)
    sio2_si_face = _face_before(edges[2], -0.385e-6)
    ta_sio2_face = _face_before(edges[2], -0.1e-6)
    ta_top_face = _face_before(edges[2], 0.0)
    rz[:, :, sio2_si_face] = 1.0 / G_SIO2_SI_W_M2K
    rz[np.ix_(np.flatnonzero(x_ta), np.flatnonzero(y_ta), [ta_sio2_face])] = (
        1.0 / G_TA_SIO2_W_M2K
    )
    rz[np.ix_(np.flatnonzero(x_ta), np.flatnonzero(y_ta), [ta_top_face])] = (
        1.0 / G_TA_AIR_W_M2K
    )

    lower_dz = widths[2][ta_top_face]
    upper_dz = widths[2][ta_top_face + 1]
    r_air = (
        0.5 * lower_dz / K_TA_XYZ_W_MK[2]
        + 1.0 / G_TA_AIR_W_M2K
        + 0.5 * upper_dz / K_AIR_W_MK
    )
    r_au = (
        0.5 * lower_dz / K_TA_XYZ_W_MK[2]
        + 1.0 / G_AU_TA_W_M2K
        + 0.5 * upper_dz / K_AU_W_MK
    )
    g_area = (1.0 - refined_fraction) / r_air + refined_fraction / r_au
    r_interface = (
        1.0 / g_area
        - 0.5 * lower_dz / K_TA_XYZ_W_MK[2]
        - 0.5 * upper_dz / k_au
    )
    if np.min(r_interface) < -1e-15:
        raise RuntimeError("negative equivalent Au/Ta interface resistance")
    rz[np.ix_(ix_au, iy_au, [ta_top_face])] = np.maximum(r_interface, 0.0)[:, :, None]

    system = fvm.assemble_steady_diagonal_kappa(
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
        edges=edges,
        widths=widths,
        centers=centers,
        system=system,
        kappa=kappa,
        masks={"au": au, "tairte4": ta, "sio2": sio2, "si": si},
        interface_resistance={"x": rx, "y": ry, "z": rz},
        rho=density.copy(),
        material_fraction=fraction.copy(),
        faces={
            "SiO2_Si": sio2_si_face,
            "TaIrTe4_SiO2": ta_sio2_face,
            "TaIrTe4_Au_or_air": ta_top_face,
        },
    )


def map_native_q_to_thermal(
    state: ThermalState,
    *,
    q_fields_W_m3: dict[str, np.ndarray],
    dual_volumes_m3: dict[str, np.ndarray],
    material_slices: dict[str, tuple[slice, slice, slice]],
    realized_grid,
) -> tuple[np.ndarray, dict[str, dict[str, float]], dict[str, dict[str, object]]]:
    """Conservatively map component-specific material power to explicit cells."""

    overlap = _load(OVERLAP_PATH, "au_dualpol_4um_overlap")
    from photothermal_pte.optimization_runs.au_on_fixed_tairte4_validation.fdtdx_dynamic_pte import (
        component_coordinates,
    )

    source_power = np.zeros(state.system.shape, dtype=np.float64)
    records: dict[str, dict[str, float]] = {}
    contexts: dict[str, dict[str, object]] = {}
    for material in ("au", "tairte4"):
        mask = state.masks[material]
        indices = (
            np.flatnonzero(np.any(mask, axis=(1, 2))),
            np.flatnonzero(np.any(mask, axis=(0, 2))),
            np.flatnonzero(np.any(mask, axis=(0, 1))),
        )
        target_edges = tuple(
            state.edges[axis][index[0] : index[-1] + 2]
            for axis, index in enumerate(indices)
        )
        geometry = {
            name: component_coordinates(
                realized_grid, material_slices[material], component
            )
            for component, name in enumerate(("x", "y", "z"))
        }
        primal_edges = tuple(
            overlap._primal_edges(
                geometry[("x", "y", "z")[axis]][0][axis],
                geometry[("x", "y", "z")[axis]][1][axis],
            )
            for axis in range(3)
        )
        primal_total = np.zeros(tuple(len(edge) - 1 for edge in primal_edges))
        native_total = 0.0
        first_operators: dict[str, tuple[object, object, object]] = {}
        for component, name in enumerate(("x", "y", "z")):
            coordinates, widths = geometry[name]
            operators = tuple(
                overlap._overlap_operator(
                    coordinates[axis], widths[axis], primal_edges[axis]
                )[0]
                for axis in range(3)
            )
            first_operators[name] = operators
            power = (
                np.asarray(q_fields_W_m3[material][component], dtype=np.float64)
                * np.asarray(dual_volumes_m3[material][component], dtype=np.float64)
            )
            native_total += float(np.sum(power))
            primal_total += overlap._forward(power, operators)
        primal_centers = tuple(0.5 * (edge[:-1] + edge[1:]) for edge in primal_edges)
        primal_widths = tuple(np.diff(edge) for edge in primal_edges)
        second = tuple(
            overlap._overlap_operator(
                primal_centers[axis], primal_widths[axis], target_edges[axis]
            )[0]
            for axis in range(3)
        )
        mapped = overlap._forward(primal_total, second)
        source_power[np.ix_(*indices)] += mapped
        mapped_total = float(np.sum(mapped))
        records[material] = {
            "native_power_W": native_total,
            "mapped_power_W": mapped_total,
            "relative_error": abs(native_total - mapped_total)
            / max(abs(native_total), np.finfo(float).tiny),
        }
        contexts[material] = {
            "indices": indices,
            "first": first_operators,
            "second": second,
        }
    return source_power, records, contexts


def pullback_thermal_source_weights(
    thermal_adjoint: np.ndarray,
    mapping_context: dict[str, dict[str, object]],
) -> dict[str, np.ndarray]:
    """Transpose the conservative remap to native component Yee power."""

    overlap = _load(OVERLAP_PATH, "au_dualpol_4um_overlap_transpose")
    result: dict[str, np.ndarray] = {}
    for material, context in mapping_context.items():
        explicit = np.asarray(thermal_adjoint, dtype=np.float64)[
            np.ix_(*context["indices"])
        ]
        primal = overlap._transpose(explicit, context["second"])
        result[material] = np.stack(
            [
                overlap._transpose(primal, context["first"][name])
                for name in ("x", "y", "z")
            ]
        )
    return result


def solve_thermal(
    state: ThermalState, source_power_W: np.ndarray, cuda_device: int
) -> tuple[np.ndarray, dict[str, object]]:
    rhs = np.asarray(source_power_W, dtype=np.float64).reshape(-1)
    operator = PersistentCudaCSR(state.system.matrix_W_K, cuda_device=cuda_device)
    result = operator.solve(
        rhs,
        relative_tolerance=1e-9,
        max_iterations=30000,
        residual_check_interval=25,
    )
    temperature = result.solution.reshape(state.system.shape)
    boundary = {
        name: float(np.sum(conductance * result.solution[cell_ids]))
        for name, (cell_ids, conductance, _) in state.system.boundary_terms.items()
    }
    source = float(np.sum(rhs))
    balance = abs(sum(boundary.values()) - source) / max(abs(source), np.finfo(float).tiny)
    return temperature, {
        "relative_residual": float(result.explicit_relative_residual),
        "iterations": int(result.iterations),
        "boundary_power_W": boundary,
        "energy_balance_relative": balance,
    }


def solve_thermal_adjoint(
    state: ThermalState, rhs: np.ndarray, cuda_device: int
) -> tuple[np.ndarray, dict[str, float | int]]:
    operator = PersistentCudaCSR(state.system.matrix_W_K, cuda_device=cuda_device)
    result = operator.solve(
        np.asarray(rhs, dtype=np.float64).reshape(-1),
        relative_tolerance=1e-9,
        max_iterations=30000,
        residual_check_interval=25,
    )
    return result.solution.reshape(state.system.shape), {
        "relative_residual": float(result.explicit_relative_residual),
        "iterations": int(result.iterations),
    }


def tairte4_temperature(state: ThermalState, temperature: np.ndarray) -> np.ndarray:
    x, y, z = state.centers
    ix = np.flatnonzero((x >= -8e-6) & (x < 8e-6))
    iy = np.flatnonzero((y >= -8e-6) & (y < 8e-6))
    iz = np.flatnonzero((z >= -0.1e-6) & (z < 0.0))
    weights = state.widths[2][iz]
    result = np.tensordot(
        temperature[np.ix_(ix, iy, iz)], weights / np.sum(weights), axes=(2, 0)
    )
    if result.shape != (ix.size, iy.size):
        raise RuntimeError(f"unexpected Ta temperature shape {result.shape}")
    return result


@dataclass(frozen=True)
class EdgeDerivative:
    left: int
    right: int
    rho_index: int
    dg_drho_S: float
    label: str


@dataclass(frozen=True)
class ElectricalSystem:
    full_matrix_S: sparse.csr_matrix
    reduced_matrix_S: sparse.csr_matrix
    reduced_rhs_A: np.ndarray
    free: np.ndarray
    fixed: np.ndarray
    fixed_values_V: np.ndarray
    objective_gradient_psi_A: np.ndarray
    derivative_terms: tuple[EdgeDerivative, ...]
    rho: np.ndarray
    material_fraction: np.ndarray


def ta_id(i: int, j: int) -> int:
    return i * N_TA + j


def au_id(i: int, j: int) -> int:
    return N_TA * N_TA + i * N_DESIGN + j


def _add_edge(rows, cols, data, left: int, right: int, g: float) -> None:
    rows.extend((left, right, left, right))
    cols.extend((left, right, right, left))
    data.extend((g, g, -g, -g))


def electrical_load(temperature_K: np.ndarray) -> np.ndarray:
    """Return the Shockley--Ramo objective vector for ``Jloc=-sigma*S*grad(T)``."""

    temperature = np.asarray(temperature_K, dtype=np.float64)
    if temperature.shape != (N_TA, N_TA):
        raise ValueError("temperature must be 160x160")
    load = np.zeros(N_TA * N_TA + N_DESIGN * N_DESIGN, dtype=np.float64)
    for i in range(N_TA):
        for j in range(N_TA):
            left = ta_id(i, j)
            if i + 1 < N_TA:
                right = ta_id(i + 1, j)
                value = (
                    SIGMA_TA_XY_S_M[0]
                    * TA_THICKNESS_M
                    * SEEBECK_TA_XY_V_K[0]
                    * (temperature[i + 1, j] - temperature[i, j])
                )
                load[left] += value
                load[right] -= value
            if j + 1 < N_TA:
                right = ta_id(i, j + 1)
                value = (
                    SIGMA_TA_XY_S_M[1]
                    * TA_THICKNESS_M
                    * SEEBECK_TA_XY_V_K[1]
                    * (temperature[i, j + 1] - temperature[i, j])
                )
                load[left] += value
                load[right] -= value
    return load


def build_electrical_system(rho: np.ndarray, temperature_K: np.ndarray) -> ElectricalSystem:
    density = np.asarray(rho, dtype=np.float64)
    if density.shape != CONTRACT.design_shape or np.any((density < 0) | (density > 1)):
        raise ValueError("rho must be 80x80 in [0,1]")
    node_count = N_TA * N_TA + N_DESIGN * N_DESIGN
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    derivatives: list[EdgeDerivative] = []
    for i in range(N_TA):
        for j in range(N_TA):
            node = ta_id(i, j)
            if i + 1 < N_TA:
                _add_edge(rows, cols, data, node, ta_id(i + 1, j), SIGMA_TA_XY_S_M[0] * TA_THICKNESS_M)
            if j + 1 < N_TA:
                _add_edge(rows, cols, data, node, ta_id(i, j + 1), SIGMA_TA_XY_S_M[1] * TA_THICKNESS_M)
    sigma_floor = SIGMA_AU_S_M * SIGMA_FLOOR_FRACTION
    fraction = np.asarray(au_material_fraction(density), dtype=np.float64)
    d_fraction = np.asarray(d_au_material_fraction_drho(density), dtype=np.float64)
    sigma = sigma_floor + fraction * (SIGMA_AU_S_M - sigma_floor)
    dsigma = d_fraction * (SIGMA_AU_S_M - sigma_floor)
    contact_floor = ELECTRICAL_CONTACT_S_M2 * CONTACT_FLOOR_FRACTION
    for i in range(N_DESIGN):
        for j in range(N_DESIGN):
            node = au_id(i, j)
            for di, dj, label in ((1, 0, "Au_sheet_x"), (0, 1, "Au_sheet_y")):
                ni, nj = i + di, j + dj
                if ni >= N_DESIGN or nj >= N_DESIGN:
                    continue
                right = au_id(ni, nj)
                resistance = 0.5 * STEP_M / sigma[i, j] + 0.5 * STEP_M / sigma[ni, nj]
                g = AU_THICKNESS_M * STEP_M / resistance
                _add_edge(rows, cols, data, node, right, g)
                for ii, jj in ((i, j), (ni, nj)):
                    dg = (
                        AU_THICKNESS_M
                        * STEP_M
                        / resistance**2
                        * 0.5
                        * STEP_M
                        / sigma[ii, jj] ** 2
                        * dsigma[ii, jj]
                    )
                    derivatives.append(
                        EdgeDerivative(node, right, ii * N_DESIGN + jj, dg, label)
                    )
            ti, tj = DESIGN_OFFSET + i, DESIGN_OFFSET + j
            g_contact = STEP_M**2 * (
                contact_floor
                + fraction[i, j]
                * (ELECTRICAL_CONTACT_S_M2 - contact_floor)
            )
            _add_edge(rows, cols, data, ta_id(ti, tj), node, g_contact)
            derivatives.append(
                EdgeDerivative(
                    ta_id(ti, tj),
                    node,
                    i * N_DESIGN + j,
                    STEP_M**2
                    * (ELECTRICAL_CONTACT_S_M2 - contact_floor)
                    * d_fraction[i, j],
                    "vertical_Au_Ta_contact",
                )
            )
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(node_count, node_count)).tocsr()
    matrix.sum_duplicates()
    # Left x-min terminal psi=0; right x-max terminal psi=1.
    low = np.asarray([ta_id(0, j) for j in range(N_TA)], dtype=np.int64)
    high = np.asarray([ta_id(N_TA - 1, j) for j in range(N_TA)], dtype=np.int64)
    fixed = np.concatenate((low, high))
    fixed_values = np.concatenate((np.zeros(low.size), np.ones(high.size)))
    free_mask = np.ones(node_count, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)
    reduced = matrix[free][:, free].tocsr()
    rhs = -np.asarray(matrix[free][:, fixed] @ fixed_values).reshape(-1)
    return ElectricalSystem(
        full_matrix_S=matrix,
        reduced_matrix_S=reduced,
        reduced_rhs_A=rhs,
        free=free,
        fixed=fixed,
        fixed_values_V=fixed_values,
        objective_gradient_psi_A=electrical_load(temperature_K),
        derivative_terms=tuple(derivatives),
        rho=density.copy(),
        material_fraction=fraction.copy(),
    )


def solve_electrical(
    system: ElectricalSystem, cuda_device: int
) -> tuple[np.ndarray, float, dict[str, float | int]]:
    operator = PersistentCudaCSR(system.reduced_matrix_S, cuda_device=cuda_device)
    result = operator.solve(
        system.reduced_rhs_A,
        relative_tolerance=1e-10,
        max_iterations=30000,
        residual_check_interval=10,
    )
    psi = np.zeros(system.full_matrix_S.shape[0], dtype=np.float64)
    psi[system.fixed] = system.fixed_values_V
    psi[system.free] = result.solution
    current = float(system.objective_gradient_psi_A @ psi)
    residual = np.asarray(system.full_matrix_S @ psi).reshape(-1)
    low = float(np.sum(residual[system.fixed[:N_TA]]))
    high = float(np.sum(residual[system.fixed[N_TA:]]))
    balance = abs(low + high) / max(abs(low), abs(high), np.finfo(float).tiny)
    free_residual = np.linalg.norm(residual[system.free]) / max(
        np.linalg.norm(system.reduced_rhs_A), np.finfo(float).tiny
    )
    return psi, current, {
        "relative_residual": float(result.explicit_relative_residual),
        "explicit_free_residual": float(free_residual),
        "iterations": int(result.iterations),
        "terminal_balance_relative": float(balance),
        "low_terminal_A_per_V": low,
        "high_terminal_A_per_V": high,
    }


def current_integrand(temperature_K: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Cell-centred A/m2 map whose area integral is the total PTE current."""

    temperature = np.asarray(temperature_K, dtype=np.float64)
    values = np.zeros((N_TA, N_TA), dtype=np.float64)
    for i in range(N_TA):
        for j in range(N_TA):
            node = ta_id(i, j)
            if i + 1 < N_TA:
                right = ta_id(i + 1, j)
                contribution = -(
                    SIGMA_TA_XY_S_M[0]
                    * TA_THICKNESS_M
                    * SEEBECK_TA_XY_V_K[0]
                    * (temperature[i + 1, j] - temperature[i, j])
                    * (psi[right] - psi[node])
                )
                values[i, j] += 0.5 * contribution / STEP_M**2
                values[i + 1, j] += 0.5 * contribution / STEP_M**2
            if j + 1 < N_TA:
                right = ta_id(i, j + 1)
                contribution = -(
                    SIGMA_TA_XY_S_M[1]
                    * TA_THICKNESS_M
                    * SEEBECK_TA_XY_V_K[1]
                    * (temperature[i, j + 1] - temperature[i, j])
                    * (psi[right] - psi[node])
                )
                values[i, j] += 0.5 * contribution / STEP_M**2
                values[i, j + 1] += 0.5 * contribution / STEP_M**2
    return values


def temperature_pullback(psi: np.ndarray) -> np.ndarray:
    """Return dI/dT for the 160x160 thickness-averaged Ta field."""

    gradient = np.zeros((N_TA, N_TA), dtype=np.float64)
    for i in range(N_TA):
        for j in range(N_TA):
            node = ta_id(i, j)
            if i + 1 < N_TA:
                right = ta_id(i + 1, j)
                scale = (
                    SIGMA_TA_XY_S_M[0]
                    * TA_THICKNESS_M
                    * SEEBECK_TA_XY_V_K[0]
                )
                contribution = -scale * (psi[right] - psi[node])
                gradient[i, j] -= contribution
                gradient[i + 1, j] += contribution
            if j + 1 < N_TA:
                right = ta_id(i, j + 1)
                scale = (
                    SIGMA_TA_XY_S_M[1]
                    * TA_THICKNESS_M
                    * SEEBECK_TA_XY_V_K[1]
                )
                contribution = -scale * (psi[right] - psi[node])
                gradient[i, j] -= contribution
                gradient[i, j + 1] += contribution
    return gradient


def explicit_temperature_pullback(state: ThermalState, psi: np.ndarray) -> np.ndarray:
    coarse = temperature_pullback(psi)
    x, y, z = state.centers
    ix = np.flatnonzero((x >= -8e-6) & (x < 8e-6))
    iy = np.flatnonzero((y >= -8e-6) & (y < 8e-6))
    iz = np.flatnonzero((z >= -0.1e-6) & (z < 0.0))
    z_weight = state.widths[2][iz]
    z_weight = z_weight / np.sum(z_weight)
    result = np.zeros(state.system.shape, dtype=np.float64)
    result[np.ix_(ix, iy, iz)] = coarse[:, :, None] * z_weight[None, None, :]
    return result


def solve_electrical_adjoint(
    system: ElectricalSystem, cuda_device: int
) -> tuple[np.ndarray, dict[str, float | int]]:
    operator = PersistentCudaCSR(system.reduced_matrix_S, cuda_device=cuda_device)
    result = operator.solve(
        system.objective_gradient_psi_A[system.free],
        relative_tolerance=1e-10,
        max_iterations=30000,
        residual_check_interval=10,
    )
    adjoint = np.zeros(system.full_matrix_S.shape[0], dtype=np.float64)
    adjoint[system.free] = result.solution
    return adjoint, {
        "relative_residual": float(result.explicit_relative_residual),
        "iterations": int(result.iterations),
    }


def electrical_density_gradient(
    system: ElectricalSystem, psi: np.ndarray, adjoint: np.ndarray
) -> np.ndarray:
    gradient = np.zeros(N_DESIGN * N_DESIGN, dtype=np.float64)
    for term in system.derivative_terms:
        gradient[term.rho_index] += -term.dg_drho_S * (
            adjoint[term.left] - adjoint[term.right]
        ) * (psi[term.left] - psi[term.right])
    return gradient.reshape(CONTRACT.design_shape)


def _generic_face_dg(
    state: ThermalState,
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    axis: int,
    derivative_cell: tuple[int, int, int],
    dk_drho: float,
) -> float:
    widths = state.widths
    li, lj, lk = left
    if axis == 0:
        face_r = state.interface_resistance["x"][li, lj, lk]
        area = widths[1][lj] * widths[2][lk]
    elif axis == 1:
        face_r = state.interface_resistance["y"][li, lj, lk]
        area = widths[0][li] * widths[2][lk]
    else:
        face_r = state.interface_resistance["z"][li, lj, lk]
        area = widths[0][li] * widths[1][lj]
    total_r = (
        0.5 * widths[axis][left[axis]] / state.kappa[left + (axis,)]
        + face_r
        + 0.5 * widths[axis][right[axis]] / state.kappa[right + (axis,)]
    )
    local_k = state.kappa[derivative_cell + (axis,)]
    return float(
        area
        / total_r**2
        * 0.5
        * widths[axis][derivative_cell[axis]]
        / local_k**2
        * dk_drho
    )


def thermal_density_gradient(
    state: ThermalState,
    temperature: np.ndarray,
    adjoint: np.ndarray,
) -> np.ndarray:
    """Return -lambda^T(dK/drho)T for Au k and Au/Ta contact."""

    ids = np.arange(np.prod(state.system.shape), dtype=np.int64).reshape(
        state.system.shape
    )
    x, y, z = state.centers
    ix = np.flatnonzero((x >= -4e-6) & (x < 4e-6))
    iy = np.flatnonzero((y >= -4e-6) & (y < 4e-6))
    iz = np.flatnonzero((z >= 0.0) & (z < 0.05e-6))
    result = np.zeros(CONTRACT.design_shape, dtype=np.float64)
    d_fraction = np.asarray(
        d_au_material_fraction_drho(state.rho), dtype=np.float64
    )
    bottom_face = state.faces["TaIrTe4_Au_or_air"]
    lower_dz = state.widths[2][bottom_face]
    upper_dz = state.widths[2][bottom_face + 1]
    r_air = (
        0.5 * lower_dz / K_TA_XYZ_W_MK[2]
        + 1.0 / G_TA_AIR_W_M2K
        + 0.5 * upper_dz / K_AIR_W_MK
    )
    r_au = (
        0.5 * lower_dz / K_TA_XYZ_W_MK[2]
        + 1.0 / G_AU_TA_W_M2K
        + 0.5 * upper_dz / K_AU_W_MK
    )

    def add(ii: int, jj: int, left_id: int, right_id: int, dg: float) -> None:
        result[ii, jj] += -dg * (
            adjoint.reshape(-1)[left_id] - adjoint.reshape(-1)[right_id]
        ) * (
            temperature.reshape(-1)[left_id] - temperature.reshape(-1)[right_id]
        )

    for local_i, i in enumerate(ix):
        for local_j, j in enumerate(iy):
            for k in iz:
                cell = (i, j, k)
                for axis in (0, 1):
                    lower = list(cell)
                    lower[axis] -= 1
                    lower = tuple(lower)
                    add(
                        local_i,
                        local_j,
                        int(ids[lower]),
                        int(ids[cell]),
                        _generic_face_dg(
                            state,
                            lower,
                            cell,
                            axis,
                            cell,
                            d_fraction[local_i, local_j]
                            * (K_AU_W_MK - K_AIR_W_MK),
                        ),
                    )
                    upper = list(cell)
                    upper[axis] += 1
                    upper = tuple(upper)
                    add(
                        local_i,
                        local_j,
                        int(ids[cell]),
                        int(ids[upper]),
                        _generic_face_dg(
                            state,
                            cell,
                            upper,
                            axis,
                            cell,
                            d_fraction[local_i, local_j]
                            * (K_AU_W_MK - K_AIR_W_MK),
                        ),
                    )
                lower = (i, j, k - 1)
                if k == iz[0]:
                    area = state.widths[0][i] * state.widths[1][j]
                    dg = (
                        area
                        * (1.0 / r_au - 1.0 / r_air)
                        * d_fraction[local_i, local_j]
                    )
                else:
                    dg = _generic_face_dg(
                        state,
                        lower,
                        cell,
                        2,
                        cell,
                        d_fraction[local_i, local_j]
                        * (K_AU_W_MK - K_AIR_W_MK),
                    )
                add(local_i, local_j, int(ids[lower]), int(ids[cell]), float(dg))
                upper = (i, j, k + 1)
                add(
                    local_i,
                    local_j,
                    int(ids[cell]),
                    int(ids[upper]),
                    _generic_face_dg(
                        state,
                        cell,
                        upper,
                        2,
                        cell,
                        d_fraction[local_i, local_j]
                        * (K_AU_W_MK - K_AIR_W_MK),
                    ),
                )
    return result


def evaluate_fixed_source(
    rho: np.ndarray,
    source_power_W: np.ndarray,
    cuda_device: int,
    *,
    need_gradient: bool,
) -> dict[str, object]:
    state = build_thermal_state(rho)
    temperature, thermal_audit = solve_thermal(state, source_power_W, cuda_device)
    ta_temperature = tairte4_temperature(state, temperature)
    electrical = build_electrical_system(rho, ta_temperature)
    psi, current, electrical_audit = solve_electrical(electrical, cuda_device)
    result: dict[str, object] = {
        "objective_A": current,
        "state": state,
        "temperature": temperature,
        "ta_temperature": ta_temperature,
        "electrical_system": electrical,
        "weighting": psi,
        "thermal_audit": thermal_audit,
        "electrical_audit": electrical_audit,
    }
    if need_gradient:
        electrical_adjoint, electrical_adjoint_audit = solve_electrical_adjoint(
            electrical, cuda_device
        )
        thermal_rhs = explicit_temperature_pullback(state, psi)
        thermal_adjoint, thermal_adjoint_audit = solve_thermal_adjoint(
            state, thermal_rhs, cuda_device
        )
        gradient_thermal = thermal_density_gradient(
            state, temperature, thermal_adjoint
        )
        gradient_electrical = electrical_density_gradient(
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
