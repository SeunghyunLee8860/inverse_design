from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PHOTOTHERMAL = Path(__file__).resolve().parents[2]
FVM = PHOTOTHERMAL / "validation" / "photothermal_stage1"
if str(FVM) not in sys.path:
    sys.path.insert(0, str(FVM))

from anisotropic_heat_fvm import (
    assemble_steady_diagonal_kappa,
    solve_assembled_thermal_system,
    solve_steady_diagonal_kappa,
)


def _case(periodic=()):
    x = np.linspace(0.0, 2.0e-6, 9)
    y = np.linspace(0.0, 1.5e-6, 7)
    z = np.asarray([-1.0e-6, -0.6e-6, -0.2e-6, 0.0, 0.1e-6])
    shape = (8, 6, 4)
    kappa = np.empty((*shape, 3))
    kappa[:, :, :2, :] = [145.0, 145.0, 145.0]
    kappa[:, :, 2, :] = [1.38, 1.38, 1.38]
    kappa[:, :, 3, :] = [14.4, 3.8, 1.0]
    interface = {"z": np.zeros((8, 6, 3))}
    interface["z"][:, :, 1] = 1.0 / 1.1e9
    interface["z"][:, :, 2] = 1.0 / 7.37e6
    source = np.zeros(shape)
    source[:, :, 3] = 2.0e8 * (
        1.0 + 0.2 * np.sin(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False))
    )[:, None]
    return x, y, z, kappa, interface, source


def test_exposed_operator_matches_compatibility_forward():
    x, y, z, kappa, interface, source = _case()
    system = assemble_steady_diagonal_kappa(
        x_edges_m=x,
        y_edges_m=y,
        z_edges_m=z,
        kappa_W_mK=kappa,
        dirichlet_temperature_K={"z_min": 0.0},
        interface_resistance_m2K_W=interface,
    )
    split = solve_assembled_thermal_system(system, source_W_m3=source)
    legacy = solve_steady_diagonal_kappa(
        x_edges_m=x,
        y_edges_m=y,
        z_edges_m=z,
        kappa_W_mK=kappa,
        source_W_m3=source,
        dirichlet_temperature_K={"z_min": 0.0},
        interface_resistance_m2K_W=interface,
    )
    assert np.array_equal(split.temperature_K, legacy.temperature_K)
    assert split.boundary_power_out_W == legacy.boundary_power_out_W
    assert split.source_power_W == legacy.source_power_W


def test_periodic_matrix_is_symmetric_and_connects_seams():
    x, y, z, kappa, interface, source = _case(periodic=("x", "y"))
    system = assemble_steady_diagonal_kappa(
        x_edges_m=x,
        y_edges_m=y,
        z_edges_m=z,
        kappa_W_mK=kappa,
        dirichlet_temperature_K={"z_min": 0.0},
        interface_resistance_m2K_W=interface,
        periodic_axes=("x", "y"),
    )
    difference = system.matrix_W_K - system.matrix_W_K.T
    assert difference.nnz == 0 or np.max(np.abs(difference.data)) == 0.0
    first = int(system.active_ids[0, 0, 2])
    last_x = int(system.active_ids[-1, 0, 2])
    last_y = int(system.active_ids[0, -1, 2])
    assert system.matrix_W_K[first, last_x] < 0.0
    assert system.matrix_W_K[first, last_y] < 0.0
    result = solve_assembled_thermal_system(system, source_W_m3=source)
    assert result.linear_residual_relative < 1.0e-8
    assert result.energy_balance_relative_error < 1.0e-8


def test_source_volume_operator_owns_volume_exactly_once():
    x, y, z, kappa, interface, source = _case()
    system = assemble_steady_diagonal_kappa(
        x_edges_m=x,
        y_edges_m=y,
        z_edges_m=z,
        kappa_W_mK=kappa,
        dirichlet_temperature_K={"z_min": 0.0},
        interface_resistance_m2K_W=interface,
    )
    active = system.active_source(source)
    power_from_operator = float(
        np.sum(system.source_volume_operator_m3 @ active)
    )
    power_direct = float(np.sum(source * system.cell_volume_m3))
    assert power_from_operator == power_direct


def test_direct_surface_robin_matches_reduced_sheet_balance():
    x = np.asarray([0.0, 2.0e-6])
    y = np.asarray([0.0, 3.0e-6])
    z = np.asarray([0.0, 1.0e-7])
    kappa = np.full((1, 1, 1, 3), 2.0)
    source = np.full((1, 1, 1), 4.0e8)
    bath = 300.0
    g_bottom = 7.37e6
    g_top = 1.0
    system = assemble_steady_diagonal_kappa(
        x_edges_m=x,
        y_edges_m=y,
        z_edges_m=z,
        kappa_W_mK=kappa,
        dirichlet_temperature_K={},
        surface_robin_heat_transfer_W_m2K={
            "z_min": g_bottom,
            "z_max": g_top,
        },
        surface_robin_temperature_K={
            "z_min": bath,
            "z_max": bath,
        },
    )
    result = solve_assembled_thermal_system(
        system, source_W_m3=source
    )
    expected_rise = source[0, 0, 0] * (z[-1] - z[0]) / (
        g_bottom + g_top
    )
    assert np.isclose(
        result.temperature_K[0, 0, 0],
        bath + expected_rise,
        rtol=1e-13,
    )
    assert result.linear_residual_relative < 1e-13
    # Boundary power subtracts two ~300 K values, so absolute-temperature
    # roundoff is larger than the linear residual while remaining negligible.
    assert result.energy_balance_relative_error < 1e-8


def test_spatial_surface_robin_is_area_times_G():
    x = np.asarray([0.0, 1.0e-6, 3.0e-6])
    y = np.asarray([0.0, 2.0e-6])
    z = np.asarray([0.0, 1.0e-7])
    kappa = np.full((2, 1, 1, 3), 2.0)
    top_g = np.asarray([[1.0], [7.37e6]])
    system = assemble_steady_diagonal_kappa(
        x_edges_m=x,
        y_edges_m=y,
        z_edges_m=z,
        kappa_W_mK=kappa,
        dirichlet_temperature_K={},
        surface_robin_heat_transfer_W_m2K={
            "z_min": 7.37e6,
            "z_max": top_g,
        },
        surface_robin_temperature_K={
            "z_min": 300.0,
            "z_max": 300.0,
        },
    )
    ids, conductance, temperature = system.boundary_terms[
        "surface_robin_z_max"
    ]
    expected = np.diff(x) * np.diff(y)[0] * top_g[:, 0]
    assert np.array_equal(ids, np.asarray([0, 1]))
    assert np.allclose(conductance, expected, rtol=0.0, atol=0.0)
    assert temperature == 300.0
