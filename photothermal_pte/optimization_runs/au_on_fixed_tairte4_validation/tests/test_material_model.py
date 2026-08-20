from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from material_model import (  # noqa: E402
    EPSILON_AIR,
    EPSILON_AU_ORDAL_10UM,
    N_AU_ORDAL_10UM,
    linear_epsilon_path,
    nonlinear_index_path,
)


def test_ordal_10um_endpoint() -> None:
    assert N_AU_ORDAL_10UM == 12.1 + 69.2j
    assert np.isclose(EPSILON_AU_ORDAL_10UM, -4642.23 + 1674.64j)


def test_density_paths_have_exact_endpoints_and_are_passive() -> None:
    rho = np.linspace(0.0, 1.0, 1001)
    for path in (linear_epsilon_path(rho), nonlinear_index_path(rho)):
        assert path.epsilon[0] == EPSILON_AIR
        assert path.epsilon[-1] == EPSILON_AU_ORDAL_10UM
        assert np.all(path.epsilon.imag >= -1e-12)


def test_nonlinear_analytic_derivative_matches_centered_fd() -> None:
    rho = np.asarray([0.1, 0.5, 0.9])
    step = 1e-7
    analytic = nonlinear_index_path(rho).derivative
    numerical = (
        nonlinear_index_path(rho + step).epsilon
        - nonlinear_index_path(rho - step).epsilon
    ) / (2.0 * step)
    assert np.allclose(analytic, numerical, rtol=1e-8, atol=1e-6)
