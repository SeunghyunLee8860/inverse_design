from pathlib import Path

import nlopt
import numpy as np


def test_auglag_lbfgs_solves_bounded_nonlinear_constraint() -> None:
    size = 12
    local = nlopt.opt(nlopt.LD_LBFGS, size)
    local.set_lower_bounds(np.zeros(size))
    local.set_upper_bounds(np.ones(size))
    local.set_vector_storage(5)
    local.set_ftol_rel(1e-10)
    outer = nlopt.opt(nlopt.LD_AUGLAG, size)
    outer.set_lower_bounds(np.zeros(size))
    outer.set_upper_bounds(np.ones(size))

    def objective(x: np.ndarray, gradient: np.ndarray) -> float:
        if gradient.size:
            gradient[:] = 2.0 * (x - 0.8) / size
        return float(np.mean((x - 0.8) ** 2))

    def constraint(x: np.ndarray, gradient: np.ndarray) -> float:
        if gradient.size:
            gradient[:] = 1.0 / (0.6 * size)
        return float(np.mean(x) / 0.6 - 1.0)

    outer.set_min_objective(objective)
    outer.add_inequality_constraint(constraint, 1e-9)
    outer.set_local_optimizer(local)
    outer.set_ftol_rel(1e-9)
    outer.set_maxeval(300)
    optimum = outer.optimize(np.full(size, 0.5))
    assert outer.last_optimize_result() > 0
    assert abs(float(np.mean(optimum)) - 0.6) < 1e-5


def test_production_driver_has_no_custom_move_or_mma_update() -> None:
    source = Path(__file__).parents[1] / "run_auglag_lbfgs_optimization.py"
    text = source.read_text()
    assert "nlopt.LD_AUGLAG" in text
    assert "nlopt.LD_LBFGS" in text
    assert "mma_step" not in text
    assert "MOVE_LIMIT" not in text
    assert "set_initial_step" not in text
    assert '"manual_move_limit": None' in text
