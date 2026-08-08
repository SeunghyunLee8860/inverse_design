from __future__ import annotations

import json
import numpy as np
import pytest

from optimization_support import (
    ProductionDensityMapping,
    MMA_CAP_CONTRACT,
    adaptive_move_ceiling,
    candidate_acceptance,
    catastrophic_exact_growth,
    constraint_contract,
    constraint_values_and_gradients,
    design_metrics,
    exact_binary_audit,
    mma_effective_constraint_caps,
    stage_caps,
    stage_convergence,
    smooth_feasibility_move_retries,
    transient_license_failure,
    two_consecutive_subthreshold_moves,
)


def test_effective_caps_preserve_each_phase_after_beta2(tmp_path, monkeypatch):
    registry = tmp_path / "caps.json"
    registry.write_text(json.dumps({
        "contract": MMA_CAP_CONTRACT,
        "stages": {"32": {"caps": [2.0e-4, 2.0e-5]}},
    }))
    monkeypatch.setenv("RUN005_STAGE_CAPS_FILE", str(registry))
    values = np.asarray([1.0e-4, 5.0e-6])
    np.testing.assert_allclose(
        mma_effective_constraint_caps(values, 2.0),
        stage_caps(2.0),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        mma_effective_constraint_caps(values, 32.0), values,
        rtol=0.0, atol=0.0,
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


def test_beta2_cap_is_fixed_and_uncalibrated_later_beta_fails_closed(
    tmp_path, monkeypatch
) -> None:
    registry = tmp_path / "caps.json"
    registry.write_text(json.dumps({"contract": MMA_CAP_CONTRACT, "stages": {}}))
    monkeypatch.setenv("RUN005_STAGE_CAPS_FILE", str(registry))
    assert np.array_equal(stage_caps(2), [1.00e-3, 1.00e-4])
    with pytest.raises(RuntimeError, match="no checkpoint-calibrated fixed cap"):
        stage_caps(4)


def test_smooth_move_retries_never_enter_microrepair_regime() -> None:
    assert smooth_feasibility_move_retries(0.01) == (0.01, 0.005, 0.0025)
    assert smooth_feasibility_move_retries(0.005) == (0.005, 0.0025)
    assert smooth_feasibility_move_retries(0.0025) == (0.0025,)


def test_acceptance_balances_fom_and_fixed_constraint_feasibility() -> None:
    caps = np.array([0.04, 0.04])
    feasible = np.array([0.03, 0.03])
    assert candidate_acceptance(1.0, 1.01, feasible, feasible, caps)["accepted"]
    assert not candidate_acceptance(1.0, 1.02, feasible, [0.05, 0.03], caps)["accepted"]
    assert candidate_acceptance(1.0, 0.998, feasible, feasible, caps)["accepted"]
    assert not candidate_acceptance(1.0, 0.9979, feasible, feasible, caps)["accepted"]
    infeasible = np.array([0.06, 0.04])
    assert not candidate_acceptance(1.0, 1.10, infeasible, [0.055, 0.04], caps)["accepted"]


def test_low_beta_exact_guard_is_only_catastrophic() -> None:
    assert catastrophic_exact_growth(158, 200) == (False, 237)
    assert catastrophic_exact_growth(158, 237) == (False, 237)
    assert catastrophic_exact_growth(158, 238) == (True, 237)
    assert catastrophic_exact_growth(0, 25) == (False, 25)
    assert catastrophic_exact_growth(0, 26) == (True, 25)


def test_two_consecutive_minimum_moves_halt() -> None:
    assert not two_consecutive_subthreshold_moves([0.01, 0.0025])
    assert two_consecutive_subthreshold_moves([0.001, 0.0025])
    assert two_consecutive_subthreshold_moves([0.0025, 0.0025])
    assert two_consecutive_subthreshold_moves([0.01, 0.002, 0.001])


def test_exact_bad_cell_total_is_a_fail_closed_step_gate() -> None:
    caps = np.array([0.004, 0.004])
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
    caps = np.array([0.004, 0.004])
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
