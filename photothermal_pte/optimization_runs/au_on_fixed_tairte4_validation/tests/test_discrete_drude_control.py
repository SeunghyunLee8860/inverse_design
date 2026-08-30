from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(HERE / "38_validate_discrete_drude_adjoint_control.py"))


def test_passive_drude_fit_matches_frozen_au_endpoint() -> None:
    omega = 2.0 * np.pi * MODULE["C0"] / MODULE["WAVELENGTH_M"]
    target = complex(MODULE["N_AU"], MODULE["K_AU"]) ** 2
    pole = MODULE["fit_single_frequency_passive_drude"](target, omega)
    readback = complex(pole.epsilon(omega, 1.0))
    assert pole.omega_p > 0.0
    assert pole.gamma > 0.0
    assert abs(readback - target) / abs(target) < 1.0e-12


def test_pole_strength_interpolation_has_exact_binary_endpoints() -> None:
    strength, derivative = MODULE["interpolation"](np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(strength[[0, 2]], [0.0, 1.0], atol=0.0)
    np.testing.assert_allclose(derivative, [0.0, 0.75, 3.0], atol=0.0)


def test_fixed_grid_discrete_adjoint_matches_directional_fd() -> None:
    u = np.linspace(-1.0, 1.0, 41)
    rho = 0.50 + 0.08 * np.cos(np.pi * u)
    direction = np.sin(1.2 * np.pi * u) + 0.2 * np.cos(2.1 * np.pi * u)
    direction /= np.max(np.abs(direction))
    baseline = MODULE["solve_control"](rho, cells=161)
    h = 5.0e-4
    plus = MODULE["solve_control"](rho + h * direction, cells=161)
    minus = MODULE["solve_control"](rho - h * direction, cells=161)
    fd = (plus.objective - minus.objective) / (2.0 * h)
    ad = float(np.dot(baseline.gradient, direction))
    assert baseline.residual < 1.0e-10
    assert abs(ad - fd) / max(abs(ad), abs(fd)) < 1.0e-5
