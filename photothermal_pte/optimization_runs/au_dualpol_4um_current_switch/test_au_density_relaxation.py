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
    assert payload["positive_relaxed_path_stays_below_Re_epsilon_one"] is True


def test_low_density_tail_avoids_internal_lumerical_metal_threshold() -> None:
    transition = CONTRACT.optical_n_low_density_transition_rho
    rho = np.geomspace(3.0e-5, 1.0, 100_001)
    epsilon = epsilon_relaxation(rho)
    assert np.all(epsilon.real < 1.0)

    unchanged = np.asarray([transition, 0.1, 0.5, 1.0])
    expected = 1.0 + unchanged * (ordal_au_index() - 1.0)
    assert np.array_equal(nk_relaxation(unchanged), expected)


def test_low_density_analytic_derivative_and_c1_transition() -> None:
    transition = CONTRACT.optical_n_low_density_transition_rho
    rho = np.asarray([3.0e-5, 0.0028783, 0.019, transition, 0.021])
    direction = np.asarray([0.2, -0.3, 0.1, -0.25, 0.15])
    step = 1.0e-8
    finite_difference = (
        epsilon_relaxation(rho + step * direction)
        - epsilon_relaxation(rho - step * direction)
    ) / (2.0 * step)
    analytic = d_epsilon_d_projected_density(rho) * direction
    assert np.allclose(finite_difference, analytic, rtol=2.0e-8, atol=2.0e-6)


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
