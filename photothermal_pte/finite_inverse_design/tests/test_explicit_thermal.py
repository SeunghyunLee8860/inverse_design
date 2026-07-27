from __future__ import annotations

import numpy as np

from photothermal_pte.finite_inverse_design.explicit_thermal import (
    build_explicit_geometry,
    evaluate_explicit_thermal,
)


def test_explicit_thermal_geometry_has_all_named_materials_and_interfaces():
    rho = np.full((8, 8), 0.42)
    geometry = build_explicit_geometry(
        rho,
        lateral_domain_m=8e-6,
        si_depth_m=2e-6,
        flake_span_m=4e-6,
        cell_size_m=250e-9,
    )
    assert set(np.unique(geometry.material_id)) == {0, 1, 2, 3, 4}
    assert geometry.interface_resistance_m2K_W["x"].shape[0] == (
        geometry.material_id.shape[0] - 1
    )
    assert np.max(geometry.interface_resistance_m2K_W["z"]) > 0.0
    assert np.allclose(geometry.rho, 0.42)


def test_explicit_thermal_xy_flake_z_and_design_z_are_independent():
    rho = np.full((8, 8), 0.5)
    geometry = build_explicit_geometry(
        rho,
        lateral_domain_m=8e-6,
        si_depth_m=2e-6,
        flake_span_m=4e-6,
        core_xy_cell_size_m=250e-9,
        flake_dz_m=12.5e-9,
        design_dz_m=50e-9,
    )
    z = 0.5 * (geometry.z_edges_m[:-1] + geometry.z_edges_m[1:])
    assert geometry.core_xy_cell_size_m == 250e-9
    assert geometry.flake_dz_m == 12.5e-9
    assert geometry.design_dz_m == 50e-9
    assert np.count_nonzero((z > -100e-9) & (z < 0.0)) == 8
    assert np.count_nonzero((z > 0.0) & (z < 600e-9)) == 12


def test_explicit_thermal_rho_adjoint_matches_directional_fd_small_grid():
    rho = np.full((8, 8), 0.48)
    x = np.linspace(-1.0, 1.0, rho.shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, rho.shape[1])[None, :]
    direction = np.sin(np.pi * x) * np.cos(0.5 * np.pi * y)
    direction /= np.max(np.abs(direction))
    kwargs = {
        "lateral_domain_m": 8e-6,
        "si_depth_m": 2e-6,
        "flake_span_m": 4e-6,
        "cell_size_m": 250e-9,
    }
    base = evaluate_explicit_thermal(rho=rho, **kwargs)
    analytic = float(np.sum(base.gradient_rho_A * direction))
    step = 2e-4
    plus = evaluate_explicit_thermal(
        rho=rho + step * direction,
        **kwargs,
    )
    minus = evaluate_explicit_thermal(
        rho=rho - step * direction,
        **kwargs,
    )
    finite_difference = (plus.objective_A - minus.objective_A) / (2.0 * step)
    assert np.isclose(analytic, finite_difference, rtol=2e-4, atol=1e-22)
