"""FDTD-free tests for the 2026-08-07 stage-scheduling rework.

Covers the three user-reported failure modes:
  1. beta=2 got the same tiny budget as every other stage -> per-stage
     ``beta[:maxeval[:min_evals]]`` budgets, big first stage.
  2. enforcing the length-scale constraints on a gray low-beta field throttled
     the FOM -> constraint warm-up stages (constraints logged, not enforced,
     and NEVER able to abort the run).
  3. runs ended visibly gray -> gray-fraction metric + binarization polish
     stages appended after the ladder.
"""

import importlib

import numpy as np
import pytest

from adaptive_stage import (
    ABORT_REASONS,
    ADVANCE_REASONS,
    CONTINUE,
    REASON_MAXEVAL_WARMUP,
    REASON_STALLED_INFEASIBLE,
    REASON_WARMUP_CONVERGED,
    AdaptiveConfig,
    StageController,
)

runner = importlib.import_module("run_constrained_inverse_design")

N = 64


def _feed(controller, objectives, g_solid, g_void, step=0.0):
    latent = np.full(N, 0.5)
    decision = CONTINUE
    for obj, gs, gv in zip(objectives, g_solid, g_void):
        latent = latent + step
        decision = controller.record(obj, gs, gv, latent)
        if decision != CONTINUE:
            return decision
    return decision


# --- schedule parsing -------------------------------------------------------
def test_parse_plain_schedule_matches_legacy():
    assert runner._beta_stages("2,4,8,16") == [2.0, 4.0, 8.0, 16.0]


def test_parse_extended_entries_and_defaults():
    st = runner.parse_beta_schedule("2:40:12,4:16,8", 12, 3)
    assert st[0] == {"beta": 2.0, "maxeval": 40, "min_evals": 12}
    assert st[1] == {"beta": 4.0, "maxeval": 16, "min_evals": 3}
    assert st[2] == {"beta": 8.0, "maxeval": 12, "min_evals": 3}


def test_parse_clamps_default_min_to_stage_maxeval():
    # smoke schedules use MAXEVAL=2 with the global default min_evals=3; the
    # DEFAULT min must clamp instead of erroring out.
    st = runner.parse_beta_schedule("2,4", 2, 3)
    assert all(s["min_evals"] <= s["maxeval"] for s in st)


def test_parse_rejects_bad_entries():
    for bad in ("", "0", "2:0", "2:4:9", "4,2", "2:a", "2:4:3:1"):
        with pytest.raises(SystemExit):
            runner.parse_beta_schedule(bad, 12, 3)


# --- warm-up controller -----------------------------------------------------
CFG = AdaptiveConfig(
    min_evals_per_stage=3, max_evals_per_stage=12, convergence_window=3,
    objective_rel_tol=5e-3, latent_rms_tol=1e-3, latent_max_tol=1e-2,
)


def test_warmup_plateau_advances_even_if_infeasible():
    controller = StageController(CFG, beta=2.0, constraints_active=False)
    decision = _feed(controller, [1.0, 1.0001, 1.0002],
                     [2e-3] * 3, [1e-3] * 3, step=1e-5)
    assert decision == REASON_WARMUP_CONVERGED
    assert decision in ADVANCE_REASONS
    assert decision not in ABORT_REASONS


def test_warmup_maxeval_never_aborts():
    cfg = AdaptiveConfig(min_evals_per_stage=2, max_evals_per_stage=3,
                         convergence_window=2, objective_rel_tol=1e-9,
                         latent_rms_tol=1e-12, latent_max_tol=1e-12)
    controller = StageController(cfg, beta=2.0, constraints_active=False)
    decision = _feed(controller, [1.0, 2.0, 3.0],
                     [1e-3] * 3, [1e-3] * 3, step=0.1)
    assert decision == REASON_MAXEVAL_WARMUP
    assert decision in ADVANCE_REASONS


def test_constrained_stage_behaviour_unchanged_by_default():
    controller = StageController(CFG, beta=8.0)
    assert controller.constraints_active is True
    decision = _feed(controller, [1.0, 1.0001, 1.0002],
                     [2e-3] * 3, [1e-3] * 3, step=1e-5)
    assert decision == REASON_STALLED_INFEASIBLE   # infeasible plateau aborts


def test_constraints_active_recorded_in_summary_and_hash_sensitive():
    warm = StageController(CFG, beta=2.0, constraints_active=False)
    assert warm.summary("x")["constraints_active"] is False
    # warm-up threshold and per-stage budgets are configuration identity
    a = runner._config_hash({"constraint_start_beta": 8.0})
    b = runner._config_hash({"constraint_start_beta": 4.0})
    assert a != b


# --- binarization metric ----------------------------------------------------
def test_gray_fraction_drops_with_beta():
    from periodic_constrained_mapping import PeriodicConstrainedMapping
    mapping = PeriodicConstrainedMapping(41, 41, 2, 6.0, 0.5)
    latent = np.zeros((40, 40))
    latent[:, :20] = 1.0                     # periodic stripe, sharp edges
    g_low = runner.gray_fraction_unique(mapping, latent.reshape(-1), 2.0)
    g_high = runner.gray_fraction_unique(mapping, latent.reshape(-1), 256.0)
    assert g_high < g_low
    assert g_low >= 0.15                      # beta=2: filter band stays gray
    assert g_high < 0.05                      # beta=256 is near-binary


def test_gray_fraction_uniform_field_is_all_gray():
    from periodic_constrained_mapping import PeriodicConstrainedMapping
    mapping = PeriodicConstrainedMapping(41, 41, 2, 6.0, 0.5)
    latent = np.full(40 * 40, 0.5)
    assert runner.gray_fraction_unique(mapping, latent, 64.0) == 1.0


# --- runner wiring (source-level guards, same style as launcher tests) ------
def test_constraints_only_added_when_active():
    source = open(runner.__file__).read()
    gate = source.find("if constraints_active:")
    add = source.find("opt.add_inequality_constraint(c_solid, 0.0)")
    assert 0 < gate < add


def test_polish_loop_present_and_capped():
    source = open(runner.__file__).read()
    assert "beta * 2.0 <= args.beta_cap" in source
    assert "polish_stage_appended" in source
    # stage budgets come from the schedule entry, not the global maxeval
    assert 'opt.set_maxeval(int(stage_spec["maxeval"]))' in source


def test_stage_termination_owned_by_controller_not_nlopt():
    # nlopt xtol would end a stage after one tiny step, bypassing
    # min_evals_per_stage; it must be disabled (runtime-opt lesson).
    source = open(runner.__file__).read()
    assert "opt.set_xtol_rel(0.0)" in source
    assert "opt.set_xtol_rel(1e-4)" not in source


def test_mma_inner_penalty_warm_start_applied_and_verified():
    # CCSA rho_init=1.0 wastes the first ~5 FDTD evals of every stage on
    # step RMS ~1e-6..1e-4 (measured); the runner must set AND verify it.
    source = open(runner.__file__).read()
    assert 'opt.set_param("rho_init", float(args.rho_init))' in source
    assert 'opt.get_param("rho_init", -1.0)' in source
    assert '"nlopt_rho_init"' in source   # part of the config identity


def test_schedule_and_warmup_enter_config_hash_before_freeze():
    source = open(runner.__file__).read()
    hash_pos = source.find('contract["config_hash"] = _config_hash(contract)')
    for field in ('"beta_schedule": [dict(s) for s in stages]',
                  '"constraint_start_beta"', '"binarization_policy"'):
        pos = source.find(field)
        assert 0 < pos < hash_pos, field
