from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    N_DESIGN,
    N_TA,
    SEEBECK_AU_TA_CONTACT_V_K,
    SEEBECK_AU_V_K,
    STEP_M,
    au_temperature_pullback,
    build_electrical_system,
    current_integrand,
    electrical_density_gradient,
    ta_id,
    temperature_pullback,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)


def _linear_weighting() -> np.ndarray:
    values = np.zeros(N_TA * N_TA + N_DESIGN * N_DESIGN, dtype=np.float64)
    for i in range(N_TA):
        values[[ta_id(i, j) for j in range(N_TA)]] = i / (N_TA - 1)
    au = np.broadcast_to(
        np.linspace(0.25, 0.75, N_DESIGN)[:, None],
        (N_DESIGN, N_DESIGN),
    )
    values[N_TA * N_TA :] = au.ravel()
    return values


def test_bulk_au_thermopower_is_active_and_integrates_exactly() -> None:
    assert SEEBECK_AU_V_K == CONTRACT.au_bulk_seebeck_V_K == 1.94e-6
    assert SEEBECK_AU_TA_CONTACT_V_K == 0.0
    rho = np.full((N_DESIGN, N_DESIGN), 0.65, dtype=np.float64)
    ta_temperature = np.zeros((N_TA, N_TA), dtype=np.float64)
    au_temperature = np.broadcast_to(
        np.linspace(0.0, 3.0, N_DESIGN)[:, None],
        (N_DESIGN, N_DESIGN),
    ).copy()
    system = build_electrical_system(rho, ta_temperature, au_temperature)
    psi = _linear_weighting()

    au_current = float(system.au_thermoelectric_load_A @ psi)
    total_current = float(system.objective_gradient_psi_A @ psi)
    integrand_current = float(
        np.sum(
            current_integrand(
                ta_temperature,
                psi,
                electrical_system=system,
            )
        )
        * STEP_M**2
    )
    assert au_current != 0.0
    assert np.isclose(total_current, au_current, rtol=2e-13, atol=0.0)
    assert np.isclose(integrand_current, total_current, rtol=2e-13, atol=0.0)


def test_tairte4_and_au_temperature_pullbacks_match_the_objective() -> None:
    x_ta = np.arange(N_TA, dtype=np.float64)[:, None]
    y_ta = np.arange(N_TA, dtype=np.float64)[None, :]
    ta_temperature = 0.01 * x_ta + 0.004 * y_ta
    x_au = np.arange(N_DESIGN, dtype=np.float64)[:, None]
    y_au = np.arange(N_DESIGN, dtype=np.float64)[None, :]
    au_temperature = 0.02 * x_au - 0.007 * y_au
    rho = np.full((N_DESIGN, N_DESIGN), 0.55, dtype=np.float64)
    system = build_electrical_system(rho, ta_temperature, au_temperature)
    psi = _linear_weighting()

    ta_contraction = float(
        np.vdot(
            temperature_pullback(psi, n_ta=N_TA, n_design=N_DESIGN),
            ta_temperature,
        )
    )
    au_contraction = float(
        np.vdot(au_temperature_pullback(system, psi), au_temperature)
    )
    assert np.isclose(
        ta_contraction,
        float(system.tairte4_thermoelectric_load_A @ psi),
        rtol=2e-13,
        atol=0.0,
    )
    assert np.isclose(
        au_contraction,
        float(system.au_thermoelectric_load_A @ psi),
        rtol=2e-13,
        atol=0.0,
    )
    assert np.isclose(
        ta_contraction + au_contraction,
        float(system.objective_gradient_psi_A @ psi),
        rtol=2e-13,
        atol=0.0,
    )


def test_au_thermopower_density_derivative_matches_central_fd() -> None:
    rng = np.random.default_rng(20260826)
    rho = 0.35 + 0.30 * rng.random((N_DESIGN, N_DESIGN))
    direction = rng.standard_normal(rho.shape)
    direction /= np.max(np.abs(direction))
    ta_temperature = np.zeros((N_TA, N_TA), dtype=np.float64)
    x = np.linspace(-1.0, 1.0, N_DESIGN)[:, None]
    y = np.linspace(-1.0, 1.0, N_DESIGN)[None, :]
    au_temperature = 1.7 * x - 0.8 * y + 0.3 * x * y
    psi = rng.standard_normal(N_TA * N_TA + N_DESIGN * N_DESIGN)
    zero_adjoint = np.zeros_like(psi)

    system = build_electrical_system(rho, ta_temperature, au_temperature)
    analytic = float(
        np.vdot(
            electrical_density_gradient(system, psi, zero_adjoint),
            direction,
        )
    )
    step = 2.0e-6

    def objective(value: np.ndarray) -> float:
        candidate = build_electrical_system(
            value,
            ta_temperature,
            au_temperature,
        )
        return float(candidate.objective_gradient_psi_A @ psi)

    finite_difference = (
        objective(rho + step * direction) - objective(rho - step * direction)
    ) / (2.0 * step)
    relative_error = abs(analytic - finite_difference) / max(
        abs(analytic), abs(finite_difference), np.finfo(float).tiny
    )
    assert abs(analytic) > 0.0
    assert relative_error < 2.0e-7


def test_exact_void_has_no_au_thermopower_despite_conductivity_floor() -> None:
    rho = np.zeros((N_DESIGN, N_DESIGN), dtype=np.float64)
    ta_temperature = np.zeros((N_TA, N_TA), dtype=np.float64)
    au_temperature = np.broadcast_to(
        np.linspace(0.0, 5.0, N_DESIGN)[:, None],
        (N_DESIGN, N_DESIGN),
    ).copy()
    system = build_electrical_system(rho, ta_temperature, au_temperature)
    assert all(edge.coefficient_A_K == 0.0 for edge in system.thermoelectric_edges)
    assert np.count_nonzero(system.au_thermoelectric_load_A) == 0


def test_exact_binary_mode_keeps_au_thermopower_only_on_solid_edges() -> None:
    mask = np.zeros((N_DESIGN, N_DESIGN), dtype=np.float64)
    mask[20:60, 30:50] = 1.0
    ta_temperature = np.zeros((N_TA, N_TA), dtype=np.float64)
    au_temperature = np.broadcast_to(
        np.linspace(0.0, 4.0, N_DESIGN)[:, None],
        (N_DESIGN, N_DESIGN),
    ).copy()
    system = build_electrical_system(
        mask,
        ta_temperature,
        au_temperature,
        exact_binary_geometry=True,
    )
    assert system.exact_binary_geometry
    assert system.inactive.size == int(np.count_nonzero(mask == 0.0))
    assert system.thermoelectric_edges
    assert all(edge.coefficient_A_K > 0.0 for edge in system.thermoelectric_edges)
    assert all(
        edge.coefficient_derivatives_A_K == ()
        for edge in system.thermoelectric_edges
    )
    for edge in system.thermoelectric_edges:
        li, lj = divmod(edge.temperature_left_index, N_DESIGN)
        ri, rj = divmod(edge.temperature_right_index, N_DESIGN)
        assert mask[li, lj] == mask[ri, rj] == 1.0
    assert np.count_nonzero(system.au_thermoelectric_load_A) > 0
