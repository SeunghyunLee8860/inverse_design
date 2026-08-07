from __future__ import annotations

import numpy as np

from optimization_support import (
    ProductionDensityMapping,
    candidate_acceptance,
    constraint_values_and_gradients,
    design_metrics,
    exact_binary_audit,
    stage_caps,
    stage_convergence,
    transient_license_failure,
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


def test_exact_audit_reports_kernel_and_domain_counts_separately() -> None:
    rho = np.zeros((21, 23))
    audit = exact_binary_audit(rho, 50e-9)
    assert audit["structuring_element_pixel_count"] == 81
    assert audit["design_pixel_count"] == 21 * 23
    assert audit["void_pass"]


def test_fixed_caps_do_not_depend_on_current_design() -> None:
    assert np.array_equal(stage_caps(2), [0.04, 0.04])
    assert np.array_equal(stage_caps(4), [0.03, 0.03])
    assert np.array_equal(stage_caps(8), [0.02, 0.02])
    assert np.array_equal(stage_caps(16), [0.008, 0.008])
    assert np.array_equal(stage_caps(32), [0.002, 0.002])
    assert np.array_equal(stage_caps(64), [0.0001, 0.0001])
    assert np.array_equal(stage_caps(512), [0.0001, 0.0001])


def test_acceptance_balances_fom_and_fixed_constraint_feasibility() -> None:
    caps = np.array([0.04, 0.04])
    feasible = np.array([0.03, 0.03])
    assert candidate_acceptance(1.0, 1.01, feasible, feasible, caps)["accepted"]
    assert not candidate_acceptance(1.0, 1.02, feasible, [0.05, 0.03], caps)["accepted"]
    infeasible = np.array([0.06, 0.04])
    assert candidate_acceptance(1.0, 0.97, infeasible, [0.055, 0.04], caps)["accepted"]
    assert not candidate_acceptance(1.0, 0.90, infeasible, [0.055, 0.04], caps)["accepted"]


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


def test_metrics_use_entire_373_by_373_design() -> None:
    mapping = ProductionDensityMapping()
    latent = np.full(mapping.shape, 0.5)
    metrics, _ = design_metrics(latent, 2.0, mapping)
    assert metrics["exact_binary_audit"]["design_pixel_count"] == 373 * 373


def test_only_explicit_hpc_checkout_errors_are_retryable() -> None:
    assert transient_license_failure({
        "passed": False,
        "error": "Unable to checkout the requested HPC license. This operation requires 9 licenses for feature FDTD_Solutions_engine.",
    })
    assert not transient_license_failure({
        "passed": False,
        "error": "thermal residual exceeded its gate",
    })
    assert not transient_license_failure({
        "passed": True,
        "error": "Unable to checkout the requested HPC license",
    })
