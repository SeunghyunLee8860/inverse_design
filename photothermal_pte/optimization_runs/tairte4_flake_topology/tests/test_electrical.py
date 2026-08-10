import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.electrical import (
    build_rectangular_mesh,
    solve_weighting_and_adjoint,
)


SIGMA = (1.10e5, 4.91e5)  # x=b, y=a
SEEBECK = (27.0e-6, -6.0e-6)


def test_uniform_weighting_is_linear_and_conductance_is_analytic() -> None:
    mesh = build_rectangular_mesh(1.0e-6, 2.0e-6, 0.1e-6)
    xx, yy = np.meshgrid(mesh.x_m, mesh.y_m, indexing="ij")
    result = solve_weighting_and_adjoint(
        mesh,
        np.ones(mesh.shape),
        300.0 + 2.0e5 * xx - 1.0e5 * yy,
        thickness_m=100.0e-9,
        sigma_xy_S_m=SIGMA,
        seebeck_xy_V_K=SEEBECK,
    )
    expected = (yy - yy.min()) / (yy.max() - yy.min())
    assert np.max(np.abs(result.weighting_potential - expected)) < 2.0e-12
    expected_conductance = SIGMA[1] * 100.0e-9 * 1.0e-6 / 2.0e-6
    assert abs(result.terminal_conductance_S / expected_conductance - 1.0) < 2.0e-12
    assert result.weighting_residual < 1.0e-11
    assert result.adjoint_residual < 1.0e-10


def test_constant_temperature_has_zero_current() -> None:
    mesh = build_rectangular_mesh(0.8e-6, 1.0e-6, 0.1e-6)
    rho = np.full(mesh.shape, 0.63)
    result = solve_weighting_and_adjoint(
        mesh,
        rho,
        np.full(mesh.shape, 300.0),
        thickness_m=100.0e-9,
        sigma_xy_S_m=SIGMA,
        seebeck_xy_V_K=SEEBECK,
    )
    assert abs(result.current_A) < 1.0e-20


def test_density_gradient_matches_directional_fd() -> None:
    mesh = build_rectangular_mesh(0.8e-6, 1.0e-6, 0.1e-6)
    xx, yy = np.meshgrid(mesh.x_m, mesh.y_m, indexing="ij")
    rho = 0.55 + 0.08 * np.cos(2.0 * np.pi * xx / 0.8e-6) * np.sin(np.pi * yy / 1.0e-6)
    temperature = 300.0 + 0.7 * np.exp(-((xx / 0.25e-6) ** 2 + ((yy + 0.1e-6) / 0.3e-6) ** 2))
    rng = np.random.default_rng(19)
    direction = rng.normal(size=mesh.shape)
    direction[0, :] = 0.0
    direction[-1, :] = 0.0
    direction[:, 0] = 0.0
    direction[:, -1] = 0.0
    direction /= np.max(np.abs(direction))
    kwargs = dict(
        thickness_m=100.0e-9,
        sigma_xy_S_m=SIGMA,
        seebeck_xy_V_K=SEEBECK,
        sigma_void_fraction=1.0e-8,
        sigma_penalty=2.0,
        alpha_penalty=2.0,
    )
    base = solve_weighting_and_adjoint(mesh, rho, temperature, **kwargs)
    h = 2.0e-5
    plus = solve_weighting_and_adjoint(mesh, rho + h * direction, temperature, **kwargs)
    minus = solve_weighting_and_adjoint(mesh, rho - h * direction, temperature, **kwargs)
    fd = (plus.current_A - minus.current_A) / (2.0 * h)
    ad = float(np.sum(base.gradient_rho_A * direction))
    scale = max(abs(fd), abs(ad), 1.0e-30)
    assert abs(ad - fd) / scale < 2.0e-6


def test_temperature_gradient_matches_directional_fd() -> None:
    mesh = build_rectangular_mesh(0.8e-6, 1.0e-6, 0.1e-6)
    xx, yy = np.meshgrid(mesh.x_m, mesh.y_m, indexing="ij")
    rho = 0.4 + 0.3 * (xx / 0.8e-6 + 0.5)
    temperature = 300.0 + 0.4 * np.cos(np.pi * xx / 0.8e-6) * np.sin(np.pi * yy / 1.0e-6)
    direction = np.exp(
        -(((xx - 0.12e-6) / 0.23e-6) ** 2 + ((yy + 0.17e-6) / 0.31e-6) ** 2)
    )
    kwargs = dict(
        thickness_m=100.0e-9,
        sigma_xy_S_m=SIGMA,
        seebeck_xy_V_K=SEEBECK,
    )
    base = solve_weighting_and_adjoint(mesh, rho, temperature, **kwargs)
    h = 1.0e-3
    plus = solve_weighting_and_adjoint(mesh, rho, temperature + h * direction, **kwargs)
    minus = solve_weighting_and_adjoint(mesh, rho, temperature - h * direction, **kwargs)
    fd = (plus.current_A - minus.current_A) / (2.0 * h)
    ad = float(np.sum(base.gradient_temperature_K_inv * direction))
    scale = max(abs(fd), abs(ad), 1.0e-30)
    assert abs(ad - fd) / scale < 2.0e-7
