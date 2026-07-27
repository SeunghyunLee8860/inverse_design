"""CCSA warm-up: rho_init knob + accelerating-steps stall guard.

Measured chain of evidence (connected r1/r2 + standalone MMA probe on the
real seed/constraints/measured gradient, nlopt 2.11.0):
  * default rho_init=1.0 -> step RMS 4.45e-6 -> 4.45e-5 -> 6.5e-4 (10x/eval);
    real movement only from evaluation ~4-5 (each one a 915 s FDTD solve);
  * obj_scale=1e6 changes NOTHING (bit-identical trajectory) -- nlopt
    normalises per-function, so rho_init is the effective knob;
  * r1/r2 aborted stage_stalled_infeasible at eval 3 -- one evaluation
    before real movement.  Growing steps must never be classified "quiet".
"""

import importlib
from pathlib import Path

import numpy as np
import pytest

from adaptive_stage import (
    CONTINUE,
    REASON_STALLED_INFEASIBLE,
    AdaptiveConfig,
    StageController,
)

RUNNER = Path(__file__).resolve().parents[1] / "run_constrained_inverse_design.py"
CFG = AdaptiveConfig()


def test_warmup_growing_steps_do_not_stall():
    """The r1/r2 regression: plateau + infeasible + tiny-but-10x-growing steps
    must CONTINUE (optimizer is waking up), not abort."""
    controller = StageController(CFG, beta=2.0)
    latent = np.full(64, 0.5)
    step = 4.45e-6
    decision = None
    for k, objective in enumerate([1.0, 1.0001, 1.0002, 1.0001]):
        latent = latent + step
        step *= 10.0                      # measured CCSA warm-up growth
        decision = controller.record(objective, 2e-2, 2e-2, latent)
        assert decision == CONTINUE, f"aborted during warm-up at eval {k+1}"
    assert controller.stop_reason is None


def test_constant_tiny_steps_still_stall_infeasible():
    """A genuine stall (flat steps) keeps the honest abort behaviour."""
    controller = StageController(CFG, beta=32.0)   # gated regime
    latent = np.full(64, 0.5)
    decision = None
    for objective in [1.0, 1.0001, 1.0002]:
        latent = latent + 1e-5            # constant -> not accelerating
        decision = controller.record(objective, 2e-2, 2e-2, latent)
    assert decision == REASON_STALLED_INFEASIBLE


def test_metrics_expose_acceleration_flag():
    controller = StageController(CFG, beta=2.0)
    latent = np.full(64, 0.5)
    step = 1e-6
    for objective in [1.0, 1.0, 1.0]:
        latent = latent + step
        step *= 10.0
        controller.record(objective, 2e-2, 2e-2, latent)
    assert controller.window_metrics()["latent_step_accelerating"] is True


def test_rho_init_in_contract_before_config_hash():
    src = RUNNER.read_text()
    key = src.find('"nlopt_rho_init"')
    hash_pos = src.find('contract["config_hash"] = _config_hash(contract)')
    assert 0 < key < hash_pos
    # fail-closed readback of the param must exist
    assert 'opt.set_param("rho_init"' in src
    assert 'opt.get_param("rho_init"' in src


def test_rho_init_changes_config_hash():
    runner = importlib.import_module("run_constrained_inverse_design")
    a = runner._config_hash({"nlopt_rho_init": 1e-2})
    b = runner._config_hash({"nlopt_rho_init": 1.0})
    assert a != b


def test_nlopt_accepts_rho_init_param():
    nlopt = pytest.importorskip("nlopt")
    opt = nlopt.opt(nlopt.LD_MMA, 4)
    opt.set_param("rho_init", 1e-2)
    assert opt.get_param("rho_init", -1.0) == pytest.approx(1e-2)


def test_rho_init_actually_changes_mma_trajectory():
    """End-to-end proof on a miniature constrained MMA problem: rho_init
    shrinks the warm-up (a larger first step), unlike obj_scale."""
    nlopt = pytest.importorskip("nlopt")

    def first_step(rho_init):
        opt = nlopt.opt(nlopt.LD_MMA, 8)
        opt.set_lower_bounds(np.zeros(8))
        opt.set_upper_bounds(np.ones(8))
        opt.set_maxeval(2)
        if rho_init is not None:
            opt.set_param("rho_init", rho_init)
        seen = []

        def objective(x, grad):
            seen.append(np.array(x, copy=True))
            if grad.size:
                grad[:] = -1e-6 * np.ones(8)   # production-scale gradient
            return -1e-6 * float(np.sum(x))

        def constraint(x, grad):
            if grad.size:
                grad[:] = 1e-7 * np.ones(8)
            return 2e-2 + 1e-7 * float(np.sum(x))

        opt.set_min_objective(objective)
        opt.add_inequality_constraint(constraint, 0.0)
        try:
            opt.optimize(np.full(8, 0.5))
        except Exception:  # noqa: BLE001 - maxeval end is fine
            pass
        assert len(seen) >= 2
        return float(np.sqrt(np.mean((seen[1] - seen[0]) ** 2)))

    default_step = first_step(None)
    fast_step = first_step(1e-2)
    assert fast_step > 10.0 * default_step, (default_step, fast_step)
