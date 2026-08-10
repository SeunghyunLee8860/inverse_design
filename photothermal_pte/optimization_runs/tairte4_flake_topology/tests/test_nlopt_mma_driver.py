from pathlib import Path

import nlopt
import numpy as np


def test_nlopt_ld_mma_solves_constraint_without_manual_move() -> None:
    size = 20
    optimizer = nlopt.opt(nlopt.LD_MMA, size)
    optimizer.set_lower_bounds(np.zeros(size))
    optimizer.set_upper_bounds(np.ones(size))

    def objective(x: np.ndarray, gradient: np.ndarray) -> float:
        if gradient.size:
            gradient[:] = -1.0 / size
        return -float(np.mean(x))

    def constraint(x: np.ndarray, gradient: np.ndarray) -> float:
        if gradient.size:
            gradient[:] = 1.0 / (0.6 * size)
        return float(np.mean(x) / 0.6 - 1.0)

    optimizer.set_min_objective(objective)
    optimizer.add_inequality_constraint(constraint, 1.0e-8)
    optimizer.set_ftol_rel(1.0e-9)
    optimizer.set_xtol_rel(1.0e-9)
    optimizer.set_maxeval(100)
    optimum = optimizer.optimize(np.full(size, 0.25))
    assert optimizer.last_optimize_result() > 0
    assert abs(float(np.mean(optimum)) - 0.6) < 1.0e-5


def test_nlopt_vector_constraint_callback_shape_matches_production() -> None:
    optimizer = nlopt.opt(nlopt.LD_MMA, 4)
    optimizer.set_lower_bounds(np.zeros(4))
    optimizer.set_upper_bounds(np.ones(4))

    def objective(x: np.ndarray, gradient: np.ndarray) -> float:
        if gradient.size:
            gradient[:] = -np.asarray([1.0, 1.0, 0.1, 0.1])
        return -float(x[0] + x[1] + 0.1 * x[2] + 0.1 * x[3])

    def constraints(result: np.ndarray, x: np.ndarray, gradient: np.ndarray) -> None:
        result[:] = [x[0] + x[1] - 1.0, x[2] + x[3] - 1.0]
        if gradient.size:
            gradient[:, :] = [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]

    optimizer.set_min_objective(objective)
    optimizer.add_inequality_mconstraint(constraints, np.full(2, 1.0e-8))
    optimizer.set_ftol_rel(1.0e-9)
    optimizer.set_maxeval(100)
    optimum = optimizer.optimize(np.full(4, 0.25))
    assert optimizer.last_optimize_result() > 0
    assert optimum[0] + optimum[1] <= 1.0 + 1.0e-7
    assert optimum[2] + optimum[3] <= 1.0 + 1.0e-7


def test_production_nlopt_driver_has_no_custom_update_or_move_limit() -> None:
    source = Path(__file__).parents[1] / "run_nlopt_mma_optimization.py"
    text = source.read_text()
    assert "nlopt.LD_MMA" in text
    assert "mma_step" not in text
    assert "MOVE_LIMIT" not in text
    assert "move_limit=MOVE_LIMIT" not in text
    assert '"manual_move_limit": None' in text
    assert "NLOPT_XTOL_REL = 1.0e-7" in text


def test_pure_current_driver_uses_ld_mma_without_connectivity_constraint() -> None:
    source = Path(__file__).parents[1] / "run_pure_current_ld_mma_optimization.py"
    text = source.read_text()
    assert "NLopt LD_MMA" in text
    assert "--connectivity-fraction" not in text
    assert "include_terminal_conductance_constraint=False" in text
    assert "constraint_count = 0 if beta < MORPHOLOGY_START_BETA else 2" in text


def test_nlopt_ld_mma_allows_the_unconstrained_low_beta_contract() -> None:
    optimizer = nlopt.opt(nlopt.LD_MMA, 2)
    optimizer.set_lower_bounds(np.zeros(2))
    optimizer.set_upper_bounds(np.ones(2))

    def objective(x: np.ndarray, gradient: np.ndarray) -> float:
        if gradient.size:
            gradient[:] = 2.0 * (x - np.asarray([0.75, 0.25]))
        return float(np.sum((x - np.asarray([0.75, 0.25])) ** 2))

    optimizer.set_min_objective(objective)
    optimizer.set_xtol_rel(1.0e-10)
    optimizer.set_maxeval(100)
    optimum = optimizer.optimize(np.asarray([0.5, 0.5]))
    assert optimizer.last_optimize_result() > 0
    assert np.allclose(optimum, [0.75, 0.25], atol=1.0e-5)


def test_native_ld_mma_scale_controls_reach_nlopt() -> None:
    optimizer = nlopt.opt(nlopt.LD_MMA, 3)
    optimizer.set_lower_bounds(np.zeros(3))
    optimizer.set_upper_bounds(np.ones(3))
    optimizer.set_initial_step(0.025)
    optimizer.set_param("rho_init", 10.0)
    optimizer.set_param("always_improve", 1)
    optimizer.set_param("inner_gradients", 1)
    assert np.allclose(optimizer.get_initial_step(np.full(3, 0.5)), 0.025)
    assert optimizer.get_param("rho_init", -1.0) == 10.0
    assert optimizer.get_param("always_improve", -1.0) == 1.0
    assert optimizer.get_param("inner_gradients", -1.0) == 1.0
