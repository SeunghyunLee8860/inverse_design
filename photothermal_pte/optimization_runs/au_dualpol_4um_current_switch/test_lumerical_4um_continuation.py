from __future__ import annotations

from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_continuation import (
    BETA_SCHEDULE,
    MAXIMUM_CONTINUATION_EVALUATIONS,
    MINIMUM_CONTINUATION_EVALUATIONS,
    STAGE_FTOL_REL,
    STAGE_XTOL_REL,
    ContinuationEpigraphProblem,
    active_design_constraint_names,
    grayness_value_gradient,
    linearized_maximin_box_warm_start,
    stage_design_caps,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    calibrated_lumerical_250nm_dfm_caps,
    smooth_lumerical_250nm_constraints,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_optimizer import (
    uniform_initial_latent_density,
)


def _latent() -> np.ndarray:
    x = np.linspace(-1.0, 1.0, CONTRACT.design_node_shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, CONTRACT.design_node_shape[1])[None, :]
    return 0.5 + 0.12 * np.sin(0.7 * np.pi * x) * np.cos(0.9 * np.pi * y)


def _fake_evaluation(latent: np.ndarray) -> dict[str, object]:
    value = np.asarray(latent, dtype=np.float64)
    return {
        "passed": True,
        "currents_A": {
            "Ea": float(-8.0e-9 + 1.0e-12 * np.sum(value - 0.5)),
            "Eb": float(-16.0e-9 - 2.0e-12 * np.sum(value - 0.5)),
        },
        "gradient_Ea_projected_A": np.full(CONTRACT.design_node_shape, 1.0e-12),
        "gradient_Eb_projected_A": np.full(CONTRACT.design_node_shape, -2.0e-12),
    }


def test_production_start_is_exact_uniform_rho_half() -> None:
    latent = uniform_initial_latent_density()
    assert latent.shape == CONTRACT.design_node_shape
    assert np.array_equal(latent, np.full(CONTRACT.design_node_shape, 0.5))


def test_production_stages_do_not_stop_on_tiny_balancing_step() -> None:
    assert STAGE_FTOL_REL == 0.0
    assert STAGE_XTOL_REL == 0.0


def test_continuation_evaluation_budget_is_explicit() -> None:
    assert MINIMUM_CONTINUATION_EVALUATIONS == 54
    assert MAXIMUM_CONTINUATION_EVALUATIONS == 188


def test_continuation_contract_requires_optical_lateral_and_pde_convergence() -> None:
    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_continuation import (
        continuation_contract,
    )

    required = continuation_contract()["final_promotion_requires"]
    assert any("100-to-50-nm optical lateral" in item for item in required)
    assert any("adaptive custom-CUDA PDE convergence" in item for item in required)


def test_linearized_maximin_warm_start_improves_worst_utility_plane() -> None:
    latent = np.full(CONTRACT.design_node_shape, 0.5)
    gradient_a = np.zeros_like(latent)
    gradient_b = np.zeros_like(latent)
    gradient_a[0, 0] = 2.0
    gradient_a[0, 1] = -1.0
    gradient_b[0, 0] = 1.0
    gradient_b[0, 1] = -2.0
    result = linearized_maximin_box_warm_start(
        latent=latent,
        current_a_A=-2.0,
        current_b_A=3.0,
        gradient_a_latent_A=gradient_a,
        gradient_b_latent_A=gradient_b,
        maximum_change=0.1,
    )

    assert result["method"] == "exact_linearized_two_utility_box_dual_v1"
    assert result["maximum_abs_change"] == 0.1
    assert result["predicted_balanced_utility_A"] > -3.0
    assert result["predicted_improvement_A"] > 0.0
    assert np.all(np.asarray(result["latent"]) >= 0.0)
    assert np.all(np.asarray(result["latent"]) <= 1.0)


def test_design_constraints_activate_zero_one_two_three() -> None:
    counts = [len(active_design_constraint_names(beta)) for beta in BETA_SCHEDULE]
    assert counts == [0, 0, 1, 2, 3, 3, 3, 3]
    assert "grayness" in active_design_constraint_names(16.0)[-1]
    assert all(
        "conductance" not in name
        for beta in BETA_SCHEDULE
        for name in active_design_constraint_names(beta)
    )


def test_grayness_gradient_matches_directional_finite_difference() -> None:
    rng = np.random.default_rng(20260825)
    latent = _latent()
    direction = rng.standard_normal(CONTRACT.design_node_shape)
    direction /= np.max(np.abs(direction))
    beta = 16.0
    value, gradient = grayness_value_gradient(latent, beta)
    step = 1.0e-6
    plus = grayness_value_gradient(latent + step * direction, beta)[0]
    minus = grayness_value_gradient(latent - step * direction, beta)[0]
    finite_difference = (plus - minus) / (2.0 * step)
    adjoint = float(np.vdot(gradient, direction))
    assert 0.0 < value <= 1.0
    assert (
        abs(adjoint - finite_difference)
        / max(abs(adjoint), abs(finite_difference), 1.0e-14)
        < 1.0e-6
    )


def test_stage_caps_are_monotone_and_final_caps_are_calibrated() -> None:
    latent = _latent()
    previous = np.full(2, np.inf)
    for beta in BETA_SCHEDULE:
        baseline = smooth_lumerical_250nm_constraints(latent, beta)[0]
        record = stage_design_caps(
            beta=beta,
            baseline_dfm_values=baseline,
            previous_dfm_caps=previous,
        )
        current = np.asarray(record["DFM_caps"])
        assert np.all(current <= previous)
        previous = current
    calibrated, _ = calibrated_lumerical_250nm_dfm_caps()
    assert np.allclose(previous, calibrated, rtol=0.0, atol=0.0)


def test_high_beta_problem_has_two_epigraph_plus_three_design_constraints() -> None:
    latent = _latent()
    baseline = smooth_lumerical_250nm_constraints(latent, 16.0)[0]
    caps = stage_design_caps(
        beta=16.0,
        baseline_dfm_values=baseline,
        previous_dfm_caps=None,
    )
    problem = ContinuationEpigraphProblem(
        _fake_evaluation,
        beta=16.0,
        dfm_caps=np.asarray(caps["DFM_caps"]),
        grayness_cap=float(caps["grayness_cap"]),
    )
    vector = np.r_[latent.ravel(), -8.0]
    values = np.empty(problem.total_constraint_count)
    gradients = np.empty((problem.total_constraint_count, vector.size))
    problem.constraints(values, vector, gradients)
    assert problem.design_constraint_count == 3
    assert problem.total_constraint_count == 5
    assert values.shape == (5,)
    assert gradients.shape == (5, vector.size)
    assert np.all(gradients[2:, -1] == 0.0)


def test_stage_caps_are_checkpointed_before_first_maxwell_evaluation() -> None:
    source = (
        Path(__file__)
        .with_name("41_optimize_lumerical_4um_dualpol_continuation.py")
        .read_text(encoding="utf-8")
    )
    cap_setup = source.index('if int(state["attempt"]) == 0:')
    checkpoint = source.index("_save_checkpoint(checkpoint_path, **state)", cap_setup)
    first_maxwell = source.index("initial_physics = driver.evaluate(latent_initial)")
    assert cap_setup < checkpoint < first_maxwell
