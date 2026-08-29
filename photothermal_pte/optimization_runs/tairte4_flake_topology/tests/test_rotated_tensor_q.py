from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.rotated_tensor_q import (
    principal_absorption_density,
    principal_fields,
)


def test_principal_fields_align_global_axes() -> None:
    scale = 1.0 / np.sqrt(2.0)
    ea_local = np.asarray([scale, scale, 0.0], complex)
    eb_local = np.asarray([scale, -scale, 0.0], complex)
    ea = principal_fields(ea_local)
    eb = principal_fields(eb_local)
    assert abs(ea["a"] - 1.0) < 1.0e-15
    assert abs(ea["b"]) < 1.0e-15
    assert abs(eb["b"] - 1.0) < 1.0e-15
    assert abs(eb["a"]) < 1.0e-15


def test_principal_q_matches_full_rotated_tensor_quadratic_form() -> None:
    rng = np.random.default_rng(64)
    electric = rng.normal(size=(5, 4, 3)) + 1j * rng.normal(size=(5, 4, 3))
    rho = rng.uniform(size=(5, 4))
    epsilon = {"b": 3.0 + 2.0j, "a": -5.0 + 7.0j, "c": 2.0 + 1.0j}
    omega = 1.7
    q = principal_absorption_density(
        electric, rho, omega_rad_s=omega, epsilon_abc=epsilon
    )
    fields = principal_fields(electric)
    expected = sum(
        0.5
        * 8.8541878128e-12
        * omega
        * np.imag(1.0 + rho * (epsilon[axis] - 1.0))
        * np.abs(fields[axis]) ** 2
        for axis in "bac"
    )
    assert np.allclose(sum(q.values()), expected, rtol=1.0e-13, atol=0.0)
