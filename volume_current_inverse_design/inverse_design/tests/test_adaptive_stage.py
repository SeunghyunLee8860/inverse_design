"""FDTD-free regression tests for the adaptive beta-stage controller.

Required behaviours (user spec):
  1. feasible plateau            -> early advance
  2. improving feasible          -> continue
  3. plateau while infeasible    -> stop the run, never advance
  4. maxeval reached, feasible   -> advance
  5. maxeval reached, infeasible -> stop the run
  6. adaptive config is hash-sensitive (part of config_hash)
  7. a changed adaptive config refuses to resume older attempts
  8. an intentional adaptive stop is never classified as a numerical failure
"""

import importlib

import numpy as np
import pytest

from adaptive_stage import (
    ABORT_REASONS,
    ADVANCE_REASONS,
    CONTINUE,
    REASON_CONVERGED,
    REASON_MAXEVAL_FEASIBLE,
    REASON_MAXEVAL_INFEASIBLE,
    REASON_STALLED_INFEASIBLE,
    AdaptiveConfig,
    StageController,
    is_intentional_stop,
)

CFG = AdaptiveConfig(
    min_evals_per_stage=3, max_evals_per_stage=12, convergence_window=3,
    objective_rel_tol=5e-3, latent_rms_tol=1e-3, latent_max_tol=1e-2,
)
N = 64


def _controller(cfg=CFG, beta=32.0):
    # beta=32 >= feasibility_gate_beta: the gated (abort-capable) regime.
    return StageController(cfg, beta=beta)


def _feed(controller, objectives, g_solid, g_void, step=0.0):
    """Feed evals with a latent that drifts by `step` per eval; last decision."""
    latent = np.full(N, 0.5)
    decision = CONTINUE
    for index, (obj, gs, gv) in enumerate(zip(objectives, g_solid, g_void)):
        latent = latent + step
        decision = controller.record(obj, gs, gv, latent)
        if decision != CONTINUE:
            return decision, index + 1
    return decision, len(objectives)


# 1. feasible plateau -> early advance ------------------------------------
def test_feasible_plateau_advances_early():
    controller = _controller()
    objectives = [1.0, 1.0001, 1.0002, 1.0001]
    decision, evals = _feed(
        controller, objectives, [-1e-6] * 4, [-1e-6] * 4, step=1e-5)
    assert decision == REASON_CONVERGED
    assert decision in ADVANCE_REASONS
    assert evals == 3          # min_evals == window == 3: earliest legal stop
    assert controller.stop_reason == REASON_CONVERGED


# 2. improving feasible -> continue ----------------------------------------
def test_improving_feasible_continues():
    controller = _controller()
    objectives = [1.0, 1.1, 1.25, 1.4, 1.6]   # >> objective_rel_tol per window
    decision, evals = _feed(
        controller, objectives, [-1e-6] * 5, [-1e-6] * 5, step=1e-5)
    assert decision == CONTINUE
    assert controller.stop_reason is None
    assert evals == 5


# 3. plateau while infeasible -> abort, never advance ----------------------
def test_plateau_infeasible_stalls_and_never_advances():
    controller = _controller()
    objectives = [1.0, 1.0001, 1.0002]
    decision, _ = _feed(
        controller, objectives, [2e-3] * 3, [1e-3] * 3, step=1e-5)
    assert decision == REASON_STALLED_INFEASIBLE
    assert decision in ABORT_REASONS
    assert decision not in ADVANCE_REASONS


def test_latent_still_moving_blocks_early_advance():
    # objective flat + feasible, but latent moves above tolerance -> continue
    controller = _controller()
    objectives = [1.0, 1.0001, 1.0002, 1.0001]
    decision, _ = _feed(
        controller, objectives, [-1e-6] * 4, [-1e-6] * 4, step=0.05)
    assert decision == CONTINUE


# 4./5. maxeval classification ---------------------------------------------
def test_maxeval_feasible_advances():
    cfg = AdaptiveConfig(min_evals_per_stage=3, max_evals_per_stage=5,
                         convergence_window=3, objective_rel_tol=1e-9,
                         latent_rms_tol=1e-12, latent_max_tol=1e-12)
    controller = StageController(cfg, beta=32.0)
    objectives = [1.0, 2.0, 3.0, 4.0, 5.0]     # never a plateau
    decision, evals = _feed(
        controller, objectives, [-1e-6] * 5, [-1e-6] * 5, step=0.1)
    assert decision == REASON_MAXEVAL_FEASIBLE
    assert decision in ADVANCE_REASONS
    assert evals == 5


def test_maxeval_infeasible_aborts():
    cfg = AdaptiveConfig(min_evals_per_stage=3, max_evals_per_stage=5,
                         convergence_window=3, objective_rel_tol=1e-9,
                         latent_rms_tol=1e-12, latent_max_tol=1e-12)
    controller = StageController(cfg, beta=32.0)
    objectives = [1.0, 2.0, 3.0, 4.0, 5.0]
    decision, _ = _feed(
        controller, objectives, [1e-3] * 5, [1e-3] * 5, step=0.1)
    assert decision == REASON_MAXEVAL_INFEASIBLE
    assert decision in ABORT_REASONS


def test_maxeval_feasible_counts_any_feasible_eval_in_stage():
    # one feasible eval early, infeasible at the end -> still advance (the
    # global best-feasible checkpoint exists), documented behaviour.
    cfg = AdaptiveConfig(min_evals_per_stage=3, max_evals_per_stage=4,
                         convergence_window=3, objective_rel_tol=1e-9,
                         latent_rms_tol=1e-12, latent_max_tol=1e-12)
    controller = StageController(cfg, beta=32.0)
    decision, _ = _feed(controller, [1.0, 2.0, 3.0, 4.0],
                        [-1e-6, 1e-3, 1e-3, 1e-3],
                        [-1e-6, 1e-3, 1e-3, 1e-3], step=0.1)
    assert decision == REASON_MAXEVAL_FEASIBLE


# 6. adaptive config is hash-sensitive --------------------------------------
def test_adaptive_config_changes_config_hash():
    runner = importlib.import_module("run_constrained_inverse_design")
    base = {"mfs_um": 0.5, "adaptive": CFG.to_dict()}
    h0 = runner._config_hash(base)
    for change in (
        {"objective_rel_tol": 1e-2},
        {"latent_rms_tol": 5e-4},
        {"min_evals_per_stage": 4},
        {"convergence_window": 5},
    ):
        mutated = dict(base)
        mutated["adaptive"] = AdaptiveConfig(
            **{**{f: getattr(CFG, f) for f in (
                "min_evals_per_stage", "max_evals_per_stage",
                "convergence_window", "objective_rel_tol",
                "latent_rms_tol", "latent_max_tol")}, **change}
        ).to_dict()
        assert runner._config_hash(mutated) != h0, change


# 7. resume mismatch rejection ----------------------------------------------
def test_resume_refuses_changed_adaptive_config():
    """The resume guard compares config_hash; adaptive params are inside it."""
    runner = importlib.import_module("run_constrained_inverse_design")
    source = open(runner.__file__).read()
    # the guard itself
    assert 'prev.get("config_hash") != contract["config_hash"]' in source
    assert "resume refused" in source
    # adaptive settings enter the contract BEFORE the hash is computed
    contract_pos = source.find('"adaptive": adaptive_cfg.to_dict()')
    hash_pos = source.find('contract["config_hash"] = _config_hash(contract)')
    assert 0 < contract_pos < hash_pos
    # and therefore a changed tolerance yields a different hash (mechanism)
    a = runner._config_hash({"adaptive": CFG.to_dict()})
    b = runner._config_hash({"adaptive": AdaptiveConfig(
        min_evals_per_stage=3, max_evals_per_stage=12, convergence_window=3,
        objective_rel_tol=1e-2, latent_rms_tol=1e-3, latent_max_tol=1e-2,
    ).to_dict()})
    assert a != b


# 8. intentional stop is never a numerical failure ---------------------------
def test_intentional_stop_not_classified_as_failure():
    nlopt = pytest.importorskip("nlopt")
    controller = _controller()
    # drive to a converged stop
    _feed(controller, [1.0, 1.0001, 1.0002],
          [-1e-6] * 3, [-1e-6] * 3, step=1e-5)
    assert controller.stop_reason == REASON_CONVERGED
    exc = nlopt.ForcedStop("nlopt forced stop")
    assert is_intentional_stop(exc, controller) is True
    # same exception WITHOUT a controller decision -> genuine failure path
    fresh = _controller()
    assert is_intentional_stop(exc, fresh) is False
    # a non-ForcedStop exception is never intentional even with a decision
    assert is_intentional_stop(RuntimeError("solver blew up"), controller) is False
    # and the runner consults is_intentional_stop BEFORE the classifier
    runner = importlib.import_module("run_constrained_inverse_design")
    source = open(runner.__file__).read()
    guard = source.find("is_intentional_stop(exc, controller)")
    classify = source.find("stage_failure = _classify(exc)")
    assert 0 < guard < classify


def test_forced_stop_end_to_end_with_real_nlopt():
    """Miniature nlopt run: controller force-stops and preserves the latent."""
    nlopt = pytest.importorskip("nlopt")
    cfg = AdaptiveConfig(min_evals_per_stage=2, max_evals_per_stage=4,
                         convergence_window=2, objective_rel_tol=1e-9,
                         latent_rms_tol=1e-12, latent_max_tol=1e-12)
    controller = StageController(cfg, beta=32.0)
    opt = nlopt.opt(nlopt.LD_MMA, 4)
    opt.set_lower_bounds(np.zeros(4))
    opt.set_upper_bounds(np.ones(4))
    opt.set_maxeval(10)

    def objective(x, grad):
        value = float(np.sum((x - 0.3) ** 2))
        decision = controller.record(value, 1e-3, 1e-3, x)  # infeasible stage
        if decision != CONTINUE:
            opt.force_stop()
        if grad.size:
            grad[:] = 2.0 * (x - 0.3)
        return value

    opt.set_min_objective(objective)
    try:
        opt.optimize(np.full(4, 0.6))
        raised = None
    except Exception as exc:  # noqa: BLE001
        raised = exc
    # maxeval(=4) infeasible -> abort decision, delivered via ForcedStop
    assert controller.stop_reason == REASON_MAXEVAL_INFEASIBLE
    assert controller.evaluations == 4
    assert controller.last_latent is not None
    assert np.all(np.isfinite(controller.last_latent))
    if raised is not None:
        assert is_intentional_stop(raised, controller) is True


# config validation ----------------------------------------------------------
def test_config_validation_rejects_nonsense():
    with pytest.raises(ValueError):
        AdaptiveConfig(min_evals_per_stage=0).validate()
    with pytest.raises(ValueError):
        AdaptiveConfig(max_evals_per_stage=2, min_evals_per_stage=3).validate()
    with pytest.raises(ValueError):
        AdaptiveConfig(convergence_window=1).validate()
    with pytest.raises(ValueError):
        AdaptiveConfig(objective_rel_tol=0.0).validate()
