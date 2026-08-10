from photothermal_pte.optimization_runs.tairte4_flake_topology import run_optimization


def test_adaptive_beta_schedule_is_gradual_and_not_fixed_to_36_updates():
    betas = run_optimization.BETA_SCHEDULE
    assert betas == (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0)
    assert max(b / a for a, b in zip(betas, betas[1:])) <= 2.0
    assert not hasattr(run_optimization, "SAFETY_MAX_UPDATES")
    assert sum(run_optimization.MIN_UPDATES.values()) > 36


def test_low_beta_is_objective_led_and_constraints_ramp_gradually():
    weights = run_optimization.MORPHOLOGY_WEIGHT
    assert weights[1.0] == 0.0
    assert weights[2.0] == 0.0
    assert all(weights[a] <= weights[b] for a, b in zip(run_optimization.BETA_SCHEDULE, run_optimization.BETA_SCHEDULE[1:]))
    assert weights[8.0] <= 0.10
    assert weights[64.0] == 1.0
