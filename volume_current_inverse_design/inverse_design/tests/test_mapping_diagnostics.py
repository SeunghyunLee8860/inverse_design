import numpy as np

from mapping_diagnostics import assert_mapping_contract, mapping_diagnostics


def periodic_vertical_density():
    base = np.array([[0.2, 0.4], [0.6, 0.8]])
    tiled = np.empty((3, 3, 4))
    tiled[:-1, :-1, :] = base[:, :, None]
    tiled[-1, :-1, :] = tiled[0, :-1, :]
    tiled[:, -1, :] = tiled[:, 0, :]
    return tiled


def test_periodic_vertical_mapping_contract():
    rho = periodic_vertical_density()
    result = mapping_diagnostics(rho)
    assert result.periodic_x_max_abs_error == 0.0
    assert result.periodic_y_max_abs_error == 0.0
    assert result.z_extrusion_max_abs_error == 0.0
    assert result.unique_shape == (2, 2)
    assert_mapping_contract(result, rho.shape)


def test_rail_margin_is_informational_not_a_failure():
    # A near-rail density (0.001) MUST NOT raise: the solver-safe affine layer,
    # not a mapping gate, keeps the Jacobian probe valid.  The old fail-closed
    # gate caused the beta deadlock and has been removed.
    rho = periodic_vertical_density()
    rho[0, 0, :] = 0.001
    rho[-1, 0, :] = rho[0, 0, :]
    rho[0, -1, :] = rho[0, 0, :]
    rho[-1, -1, :] = rho[0, 0, :]
    result = mapping_diagnostics(rho)
    assert result.centered_probe_margin <= 0.001
    assert_mapping_contract(result, rho.shape)  # does not raise


def test_exact_binary_flag():
    rho = np.zeros((3, 3, 2))
    rho[0, 0, :] = 1.0
    rho[-1, 0, :] = 1.0
    rho[0, -1, :] = 1.0
    rho[-1, -1, :] = 1.0
    result = mapping_diagnostics(rho)
    assert result.is_exact_binary is True
    assert result.fraction_between_rails == 0.0
