from __future__ import annotations

import numpy as np

from optimization_support import (
    ProductionDensityMapping,
    adaptive_move_ceiling,
    candidate_acceptance,
    constraint_contract,
    constraint_values_and_gradients,
    design_metrics,
    exact_binary_audit,
    exact_safe_move_retries,
    mma_effective_constraint_caps,
    stage_caps,
    stage_convergence,
    transient_license_failure,
)


def test_exact_safe_move_retries_resolve_threshold_events_without_changing_ceiling():
    retries = exact_safe_move_retries(0.0025)
    assert len(retries) == 9
    assert retries[0] == 0.0025
    assert retries[-1] == 0.0025 / 256.0
    assert all(a > b for a, b in zip(retries, retries[1:]))


def test_effective_caps_allow_only_one_percent_or_absolute_floor_slack():
    values = np.asarray([1.0e-4, 5.0e-6])
    np.testing.assert_allclose(
        mma_effective_constraint_caps(values, 2.0),
        stage_caps(2.0),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        mma_effective_constraint_caps(values, 32.0),
        [1.01e-4, 5.1e-6],
        rtol=1e-14,
        atol=1e-16,
    )


def test_constraint_jvp_matches_centered_fd() -> None:
    mapping = ProductionDensityMapping(shape=(41, 39), spacing_m=50e-9, radius_m=500e-9)
    rng = np.random.default_rng(20260807)
    latent = np.clip(0.5 + 0.08 * rng.standard_normal(mapping.shape), 0.2, 0.8)
    direction = rng.standard_normal(mapping.shape)
    direction /= np.linalg.norm(direction)
    values, gradients, _ = constraint_values_and_gradients(latent, 4.0, mapping)
    step = 1.0e-4
    plus, _, _ = constraint_values_and_gradients(latent + step * direction, 4.0, mapping)
    minus, _, _ = constraint_values_and_gradients(latent - step * direction, 4.0, mapping)
    fd = (plus - minus) / (2.0 * step)
    ad = gradients.reshape(2, -1) @ direction.ravel()
    np.testing.assert_allclose(ad, fd, rtol=3e-5, atol=1e-9)
    assert np.all(values > 0.0)


def test_disk_constraint_jvp_matches_centered_fd() -> None:
    mapping = ProductionDensityMapping(shape=(31, 29), spacing_m=50e-9, radius_m=500e-9)
    rng = np.random.default_rng(320085)
    latent = np.clip(0.5 + 0.08 * rng.standard_normal(mapping.shape), 0.2, 0.8)
    direction = rng.standard_normal(mapping.shape)
    direction /= np.linalg.norm(direction)
    values, gradients, _ = constraint_values_and_gradients(latent, 32.0, mapping)
    step = 5.0e-5
    plus, _, _ = constraint_values_and_gradients(latent + step * direction, 32.0, mapping)
    minus, _, _ = constraint_values_and_gradients(latent - step * direction, 32.0, mapping)
    fd = (plus - minus) / (2.0 * step)
    ad = gradients.reshape(2, -1) @ direction.ravel()
    np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-9)
    assert np.all(values > 0.0)


def test_exact_audit_reports_kernel_and_domain_counts_separately() -> None:
    rho = np.zeros((21, 23))
    audit = exact_binary_audit(rho, 50e-9)
    assert audit["structuring_element_pixel_count"] == 81
    assert audit["design_pixel_count"] == 21 * 23
    assert audit["void_pass"]


def test_fixed_caps_do_not_depend_on_current_design() -> None:
    assert np.array_equal(stage_caps(2), [1.25e-3, 3.00e-5])
    assert np.array_equal(stage_caps(4), [1.00e-3, 2.50e-5])
    assert np.array_equal(stage_caps(8), [7.50e-4, 2.00e-5])
    assert np.array_equal(stage_caps(16), [5.00e-4, 1.50e-5])
    assert np.array_equal(stage_caps(32), [2.50e-4, 1.00e-5])
    assert np.array_equal(stage_caps(64), [1.00e-4, 7.50e-6])
    assert np.array_equal(stage_caps(128), [5.00e-5, 5.00e-6])
    assert np.array_equal(stage_caps(256), [2.50e-5, 2.50e-6])
    assert np.array_equal(stage_caps(512), [1.50e-5, 1.50e-6])
    assert np.array_equal(stage_caps(1024), [1.00e-5, 1.00e-6])


def test_acceptance_balances_fom_and_fixed_constraint_feasibility() -> None:
    caps = np.array([0.04, 0.04])
    feasible = np.array([0.03, 0.03])
    assert candidate_acceptance(1.0, 1.01, feasible, feasible, caps)["accepted"]
    assert not candidate_acceptance(1.0, 1.02, feasible, [0.05, 0.03], caps)["accepted"]
    infeasible = np.array([0.06, 0.04])
    assert candidate_acceptance(1.0, 0.99, infeasible, [0.055, 0.04], caps)["accepted"]
    assert not candidate_acceptance(1.0, 0.90, infeasible, [0.055, 0.04], caps)["accepted"]


def test_exact_bad_cell_total_is_a_fail_closed_step_gate() -> None:
    caps = np.array([0.002, 0.002])
    current = np.array([0.000956, 0.003050])
    candidate = np.array([0.002101, 0.002820])
    # Smooth violation improves, but the exact total worsens 530 -> 630.
    without_exact = candidate_acceptance(1.0, 1.0, current, candidate, caps)
    assert without_exact["accepted"]
    with_exact = candidate_acceptance(
        1.0, 1.0, current, candidate, caps,
        current_exact_bad_counts=np.array([105, 425]),
        candidate_exact_bad_counts=np.array([254, 376]),
    )
    assert not with_exact["accepted"]
    assert with_exact["exact_DRC_gate_enabled"]
    assert not with_exact["exact_total_nonincreasing"]


def test_exact_gate_allows_topology_tradeoff_when_total_does_not_increase() -> None:
    caps = np.array([0.002, 0.002])
    decision = candidate_acceptance(
        1.0, 1.0,
        np.array([0.0010, 0.0030]), np.array([0.0011, 0.0028]), caps,
        current_exact_bad_counts=np.array([105, 425]),
        candidate_exact_bad_counts=np.array([120, 400]),
    )
    assert decision["accepted"]
    assert decision["candidate_exact_bad_total"] == 520


def test_stage_never_converges_after_one_update() -> None:
    row = {
        "beta": 4.0,
        "role": "accepted_mma",
        "constraints_feasible": True,
        "relative_fom_change": 0.0,
        "rho_rms_change": 0.0,
        "rho_max_change": 0.0,
    }
    assert not stage_convergence([row], 4.0).converged


def test_stage_requires_recent_plateau_and_minimum_updates() -> None:
    history = []
    for _ in range(8):
        history.append({
            "beta": 4.0,
            "role": "accepted_mma",
            "constraints_feasible": True,
            "relative_fom_change": 0.001,
            "rho_rms_change": 0.001,
            "rho_max_change": 0.010,
        })
    assert stage_convergence(history, 4.0).converged
    history[-1]["relative_fom_change"] = 0.01
    assert not stage_convergence(history, 4.0).converged


def test_disk_constraint_is_active_from_beta2() -> None:
    assert "disk_opening_500nm_from_iteration_zero" in constraint_contract(2.0)
    assert constraint_contract(2.0) == constraint_contract(32.0)


def test_adaptive_move_reduces_only_after_four_solver_backed_plateau_updates() -> None:
    history = []
    for _ in range(8):
        history.append({
            "beta": 4.0,
            "role": "accepted_mma",
            "constraints_feasible": True,
            "relative_fom_change": 0.001,
            "rho_rms_change": 0.006,
            "rho_max_change": 0.04,
        })
    assert adaptive_move_ceiling(history, 4.0, [0.01] * 8) == 0.005
    assert adaptive_move_ceiling(history, 4.0, [0.01] * 7 + [0.005]) == 0.005
    assert adaptive_move_ceiling(history, 4.0, [0.01] * 4 + [0.005] * 4) == 0.0025
    history[-1]["relative_fom_change"] = 0.01
    assert adaptive_move_ceiling(history, 4.0, [0.01] * 8) == 0.01


def test_metrics_use_entire_373_by_373_design() -> None:
    mapping = ProductionDensityMapping()
    audit = exact_binary_audit(np.full(mapping.shape, 0.5), mapping.spacing_m)
    assert audit["design_pixel_count"] == 373 * 373


def test_only_explicit_hpc_checkout_errors_are_retryable() -> None:
    assert transient_license_failure({
        "passed": False,
        "error": "Unable to checkout the requested HPC license. This operation requires 9 licenses for feature FDTD_Solutions_engine.",
    })
    assert transient_license_failure({
        "passed": False,
        "error": (
            "Failed to start messaging, check licenses. Failed to set up "
            "Ansys license sharing. ANSYSLI exited or could not read server "
            "port; Could not bind socket on port 43903. Address already in use."
        ),
    })
    assert not transient_license_failure({
        "passed": False,
        "error": "thermal residual exceeded its gate",
    })
    assert not transient_license_failure({
        "passed": True,
        "error": "Unable to checkout the requested HPC license",
    })


def test_filter_projection_vjp_is_finite_at_closed_box_bounds() -> None:
    mapping = ProductionDensityMapping(shape=(41, 39), spacing_m=50e-9, radius_m=500e-9)
    latent = np.zeros(mapping.shape)
    latent[:, mapping.shape[1] // 2:] = 1.0
    rho = mapping.physical(latent, beta=2.0)
    gradient = mapping.vjp(latent, np.ones(mapping.shape), beta=2.0)
    assert np.all(np.isfinite(rho))
    assert np.all(np.isfinite(gradient))
    assert np.min(rho) >= 0.0 and np.max(rho) <= 1.0
    assert np.linalg.norm(gradient) > 0.0
