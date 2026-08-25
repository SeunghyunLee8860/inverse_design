from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_ade import (
    BASES,
    carrier_audit,
    coefficient_hash,
    coefficient_jvp_analytic,
    coefficients_jax,
    coefficient_vector_numpy,
    coefficients_numpy,
    fdtdx_api_coefficient_audit,
    jvp_audit,
    lorentz_parameters,
    realized_epsilon,
    recurrence_roots,
    target_epsilon,
)


def test_nk_square_decomposes_into_linear_and_quadratic_susceptibility() -> None:
    rho = np.linspace(0.0, 1.0, 101)
    decomposed = 1.0 + rho * (2.4 + 57.8j) + rho**2 * (-833.77 + 69.36j)
    assert np.allclose(target_epsilon(rho), decomposed, rtol=2e-16, atol=2e-13)
    assert target_epsilon(0.0) == 1.0 + 0.0j
    assert np.allclose(target_epsilon(1.0), -830.37 + 127.16j, rtol=0.0, atol=2e-13)


def test_three_pole_map_uses_one_shared_density_and_zero_c4() -> None:
    rho = np.linspace(0.0, 1.0, 80 * 80, dtype=np.float32).reshape(80, 80)
    c1, c2, c3, c4 = coefficients_numpy(rho)
    assert c1.shape == c2.shape == c3.shape == c4.shape == (3, 80, 80)
    assert np.all(c1 == c1[:, :1, :1])
    assert np.all(c2 == c2[:, :1, :1])
    assert np.all(c4 == 0.0)
    assert np.allclose(c3[0], np.float32(BASES[0].c3_at_unit_weight) * rho)
    assert np.allclose(c3[1], np.float32(BASES[1].c3_at_unit_weight) * rho**2)
    assert np.allclose(c3[2], np.float32(BASES[2].c3_at_unit_weight) * rho**2)

    jax_coefficients = tuple(np.asarray(value) for value in coefficients_jax(rho))
    assert all(value.shape == (3, 80, 80) for value in jax_coefficients)
    for numpy_value, jax_value in zip((c1, c2, c3, c4), jax_coefficients, strict=True):
        assert np.array_equal(numpy_value, jax_value)


def test_carrier_is_not_linear_epsilon_or_historical_linear_au_c3() -> None:
    rho = 0.5
    endpoint_linear_epsilon = 1.0 + rho * ((-830.37 + 127.16j) - 1.0)
    selected_target = complex(target_epsilon(rho))
    assert abs(selected_target - endpoint_linear_epsilon) > 200.0
    assert abs(complex(realized_epsilon(rho)) - selected_target) / abs(selected_target) < 1e-5


def test_all_lorentz_parameters_and_float32_recurrences_are_passive_stable() -> None:
    for basis in BASES:
        params = lorentz_parameters(basis)
        assert all(value > 0.0 for value in params.values())
        roots = recurrence_roots(basis)
        assert np.max(np.abs(roots)) < 1.0
        assert 1.0 - np.float32(basis.c2) - abs(np.float32(basis.c1)) > 0.0
    assert np.all(coefficients_numpy(1.0)[3] == 0.0)


def test_uniform_density_certificate_has_margin_and_frozen_hash() -> None:
    audit = carrier_audit()
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert audit["density_count"] == 101
    assert audit["maximum_relative_complex_epsilon_error"] < 1.1e-6
    assert audit["au_endpoint_relative_error"] < 1.5e-7
    assert audit["minimum_implicit_divisor"] == 1.0
    assert coefficient_hash() == "71f6738a4c587387c334c3a31edcf8df1ff9415b8fdf2d66537b7a65b6b07b0f"
    assert audit["field_control_gate"] == {
        "status": "PENDING",
        "required_densities": [0.0, 0.25, 0.5, 0.75, 1.0],
        "optimizer_allowed": False,
    }


def test_pinned_fdtdx_api_reproduces_coefficients_exactly() -> None:
    audit = fdtdx_api_coefficient_audit()
    assert audit["status"] == "PASS"
    assert all(item["exact"] for item in audit["bases"].values())


def test_jax_coefficient_jvp_matches_analytic_and_centered_fd() -> None:
    audit = jvp_audit()
    assert audit["status"] == "PASS"
    assert audit["max_jax_vs_analytic_relative_l2"] == 0.0
    assert audit["max_jax_vs_centered_fd_relative_l2"] < 3e-5
    rho = 0.7
    h = 1e-3
    centered = (coefficient_vector_numpy(rho + h) - coefficient_vector_numpy(rho - h)) / (2 * h)
    analytic = coefficient_jvp_analytic(rho)
    assert np.linalg.norm(centered - analytic) / np.linalg.norm(analytic) < 3e-5
