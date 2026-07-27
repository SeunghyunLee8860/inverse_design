"""Beta feasibility gate: measured basis + controller semantics.

Measured 2026-07-27 (probe on the real constraints; pinned below as a test):
independently-DRC-PASSING designs are constraint-INFEASIBLE at low beta --
the Zhou penalty charges the intrinsic grayness of the tanh projection:

    design            b=2       b=4       b=8       b=16      b=32
    3um stripe        +1.3e-2   +3.8e-3   +1.7e-4   -9.8e-6   -1.0e-5
    525nm bar (gv)    +2.2e-2   +1.0e-2   +1.1e-3   -1.1e-6   -1.0e-5

So an infeasibility abort below beta=16 would abort EVERY run unconditionally
(connected r3 was walking into exactly that at beta=2).  Pre-gate stages
advance on plateau/maxeval; the abort semantics apply from the gate upward,
and the final DRC + exact-FDTD SUCCESS gates are unchanged.
"""

import importlib

import numpy as np

from adaptive_stage import (
    CONTINUE,
    REASON_MAXEVAL_INFEASIBLE,
    REASON_MAXEVAL_PREGATE,
    REASON_PLATEAU_PREGATE,
    REASON_STALLED_INFEASIBLE,
    ABORT_REASONS,
    ADVANCE_REASONS,
    AdaptiveConfig,
    StageController,
)

CFG = AdaptiveConfig()


def _feed_plateau_infeasible(controller, n=3, step=1e-5):
    latent = np.full(64, 0.5)
    decision = CONTINUE
    for objective in [1.0 + 1e-4 * k for k in range(n)]:
        latent = latent + step            # constant step: quiet, not accelerating
        decision = controller.record(objective, 2e-2, 2e-2, latent)
    return decision


# --- measured structural fact (the gate's justification) -------------------
def test_drc_passing_designs_are_infeasible_below_beta16():
    from periodic_constrained_mapping import PeriodicConstrainedMapping
    from geometric_constraints import LengthScaleConstraints

    mapping = PeriodicConstrainedMapping(
        241, 241, 13, period_um=6.0, filter_radius_um=0.5)
    constraints = LengthScaleConstraints(mapping)
    stripe = np.zeros((240, 240))
    stripe[:120, :] = 0.95
    stripe[120:, :] = 0.05                # 3 um stripe: independent DRC PASS
    latent = stripe.reshape(-1)
    for beta in (2.0, 4.0, 8.0):
        gs, gv = constraints.residuals(latent, beta)
        assert gs > 0 and gv > 0, (beta, gs, gv)
    for beta in (16.0, 32.0, 64.0):
        gs, gv = constraints.residuals(latent, beta)
        assert gs <= 0 and gv <= 0, (beta, gs, gv)


# --- pre-gate semantics -----------------------------------------------------
def test_pregate_plateau_infeasible_advances():
    controller = StageController(CFG, beta=2.0)
    decision = _feed_plateau_infeasible(controller)
    assert decision == REASON_PLATEAU_PREGATE
    assert decision in ADVANCE_REASONS and decision not in ABORT_REASONS


def test_pregate_maxeval_infeasible_advances():
    cfg = AdaptiveConfig(max_evals_per_stage=4, objective_rel_tol=1e-9,
                         latent_rms_tol=1e-12, latent_max_tol=1e-12)
    controller = StageController(cfg, beta=8.0)   # 8 < gate 16
    latent = np.full(64, 0.5)
    decision = CONTINUE
    for k in range(4):
        latent = latent + 0.05
        decision = controller.record(1.0 + k, 2e-2, 2e-2, latent)
    assert decision == REASON_MAXEVAL_PREGATE
    assert decision in ADVANCE_REASONS


# --- gated semantics unchanged ----------------------------------------------
def test_gated_plateau_infeasible_still_aborts():
    controller = StageController(CFG, beta=16.0)  # gate boundary is gated
    decision = _feed_plateau_infeasible(controller)
    assert decision == REASON_STALLED_INFEASIBLE


def test_gated_maxeval_infeasible_still_aborts():
    cfg = AdaptiveConfig(max_evals_per_stage=4, objective_rel_tol=1e-9,
                         latent_rms_tol=1e-12, latent_max_tol=1e-12)
    controller = StageController(cfg, beta=32.0)
    latent = np.full(64, 0.5)
    decision = CONTINUE
    for k in range(4):
        latent = latent + 0.05
        decision = controller.record(1.0 + k, 2e-2, 2e-2, latent)
    assert decision == REASON_MAXEVAL_INFEASIBLE


# --- config identity ----------------------------------------------------------
def test_gate_in_adaptive_dict_and_config_hash():
    assert CFG.to_dict()["feasibility_gate_beta"] == 16.0
    runner = importlib.import_module("run_constrained_inverse_design")
    a = runner._config_hash({"adaptive": CFG.to_dict()})
    b = runner._config_hash({"adaptive": AdaptiveConfig(
        feasibility_gate_beta=32.0).to_dict()})
    assert a != b


def test_gate_validation():
    import pytest
    with pytest.raises(ValueError):
        AdaptiveConfig(feasibility_gate_beta=0.0).validate()
