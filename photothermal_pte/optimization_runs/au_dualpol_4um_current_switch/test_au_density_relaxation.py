from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.au_density_relaxation import (
    CONTRACT,
    audit,
    d_epsilon_d_projected_density,
    epsilon_relaxation,
    lumerical_import_index,
    nk_relaxation,
    ordal_au_index,
)


def test_4um_ordal_endpoint_and_no_rho_cubed_contract() -> None:
    au = ordal_au_index()
    assert au == 2.2 + 28.9j
    assert np.isclose((au**2).real, -830.37)
    assert np.isclose((au**2).imag, 127.16)
    assert CONTRACT.optical_rho_power is None
    payload = audit()
    assert payload["rho_cubed_used"] is False
    assert payload["exact_background_endpoint"] is True
    assert payload["exact_au_endpoint"] is True
    assert payload["passive_on_uniform_density_sweep"] is True


def test_nk_then_square_is_nonlinear_in_epsilon() -> None:
    rho = np.asarray([0.0, 0.5, 1.0])
    index = nk_relaxation(rho)
    epsilon = epsilon_relaxation(rho)
    au_epsilon = ordal_au_index() ** 2
    assert index[0] == 1.0 + 0.0j
    assert index[-1] == ordal_au_index()
    assert epsilon[0] == 1.0 + 0.0j
    assert epsilon[-1] == au_epsilon
    assert not np.isclose(epsilon[1], 0.5 * (1.0 + au_epsilon))
    assert np.all(epsilon.imag >= 0.0)


def test_complex_analytic_derivative_matches_centered_fd() -> None:
    rho = np.asarray([[0.1, 0.35], [0.6, 0.9]])
    direction = np.asarray([[0.3, -0.2], [0.1, -0.25]])
    step = 1.0e-7
    finite_difference = (
        epsilon_relaxation(rho + step * direction)
        - epsilon_relaxation(rho - step * direction)
    ) / (2.0 * step)
    analytic = d_epsilon_d_projected_density(rho) * direction
    assert np.allclose(finite_difference, analytic, rtol=2.0e-9, atol=2.0e-7)


def test_lumerical_import_map_extrudes_the_same_density() -> None:
    rho = np.asarray([[0.0, 0.25], [0.5, 1.0]])
    index = lumerical_import_index(rho, z_samples=3)
    assert index.shape == (2, 2, 3)
    assert np.array_equal(index[:, :, 0], index[:, :, 1])
    assert np.array_equal(index[:, :, 1], index[:, :, 2])
    assert index[0, 0, 0] == 1.0 + 0.0j
    assert index[1, 1, 0] == ordal_au_index()


@pytest.mark.parametrize(
    "bad",
    (
        np.asarray([[0.0, 1.1]]),
        np.asarray([[np.nan, 0.5]]),
        np.asarray([]),
    ),
)
def test_relaxation_rejects_out_of_contract_density(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        epsilon_relaxation(bad)
