import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.mma import (
    initialize_mma_state,
    mma_step,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.run_ansys_dfm_ld_mma_optimization import (
    EXACT_CLEANUP_MAXIMUM_GRAY_FRACTION,
    cleanup_objective_preserved,
)


def test_exact_cleanup_is_late_and_failed_objective_is_rejected() -> None:
    assert EXACT_CLEANUP_MAXIMUM_GRAY_FRACTION == 0.02
    failed = {
        "selected": "solid_first",
        "candidates": {"solid_first": {"result": {"passed": False}}},
    }
    passed = {
        "selected": "void_first",
        "candidates": {"void_first": {"result": {"passed": True}}},
    }
    assert not cleanup_objective_preserved(failed)
    assert cleanup_objective_preserved(passed)


def test_mma_maximizes_linear_objective_under_mean_constraint() -> None:
    x = np.full(40, 0.25)
    state = initialize_mma_state(x)
    for _ in range(20):
        objective_gradient = -np.ones_like(x) / x.size
        constraints = np.asarray([np.mean(x) / 0.6 - 1.0])
        jacobian = np.ones((1, x.size)) / (0.6 * x.size)
        x, state, diagnostics = mma_step(
            x,
            objective_gradient,
            constraints,
            jacobian,
            state,
            move_limit=0.1,
        )
    assert diagnostics["used_adam"] is False
    assert abs(float(np.mean(x)) - 0.6) < 2.0e-3
    assert np.all((x >= 0.0) & (x <= 1.0))


def test_mma_move_is_only_an_upper_bound() -> None:
    x = np.full(12, 0.5)
    state = initialize_mma_state(x)
    candidate, _, diagnostics = mma_step(
        x,
        np.linspace(-1.0, 1.0, x.size),
        np.asarray([-0.5]),
        np.zeros((1, x.size)),
        state,
        move_limit=0.07,
    )
    step = np.abs(candidate - x)
    assert np.max(step) <= 0.07 + 1e-14
    assert diagnostics["hard_clip_after_update"] is False
    stationary, _, _ = mma_step(
        x,
        np.zeros_like(x),
        np.asarray([-0.5]),
        np.zeros((1, x.size)),
        state,
        move_limit=0.07,
    )
    assert np.max(np.abs(stationary - x)) < 1e-14
