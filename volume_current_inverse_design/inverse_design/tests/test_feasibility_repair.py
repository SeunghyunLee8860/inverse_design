"""FDTD-free feasibility repair + seed-npz plumbing.

Measured basis (r4 + offline probes, 2026-07-27): at beta=16 the gated stage
burned 12 GPU evaluations (~3 h) moving g by -3% and aborted
maxeval_infeasible, while a constraint-only CPU solve moved the SAME latent
-99.4% in ~6 minutes; the epigraph form then closes the slow g_void tail that
a plain-sum objective leaves creeping (gs parks at the -1e-5 floor).
"""

import importlib
from pathlib import Path

import numpy as np
import pytest

from feasibility_repair import repair_to_feasible
from geometric_constraints import LengthScaleConstraints
from periodic_constrained_mapping import PeriodicConstrainedMapping

RUNNER = Path(__file__).resolve().parents[1] / "run_constrained_inverse_design.py"
BETA = 16.0


@pytest.fixture(scope="module")
def small_problem():
    """61-node production-like mapping: constraint evals are milliseconds."""
    mapping = PeriodicConstrainedMapping(
        61, 61, 13, period_um=6.0, filter_radius_um=0.5)
    constraints = LengthScaleConstraints(mapping)
    stripe = np.zeros((60, 60))
    stripe[:30, :] = 0.95
    stripe[30:, :] = 0.05                  # 3 um stripe: feasible at beta>=16
    return constraints, stripe.reshape(-1)


def test_already_feasible_short_circuits(small_problem):
    constraints, stripe = small_problem
    gs, gv = constraints.residuals(stripe, BETA)
    assert gs <= 0 and gv <= 0
    result = repair_to_feasible(constraints, stripe, BETA)
    assert result["feasible"] is True
    assert result["evaluations"] == 0
    assert np.array_equal(result["latent"], stripe)


def _infeasible_fixture(stripe):
    """Stripe + a 300 nm solid bar in the void half.

    Measured on this 61-node mapping: gs=+2.10e-2, gv=+1.33e-2 at beta=16
    (a per-pixel noise perturbation does NOT work -- the 0.5 um conic filter
    smooths it back to feasible, which is why this fixture is geometric).
    """
    perturbed = stripe.reshape(60, 60).copy()
    perturbed[44:47, :] = 0.95
    return perturbed.reshape(-1)


def test_repair_restores_feasibility_from_violating_design(small_problem):
    constraints, stripe = small_problem
    perturbed = _infeasible_fixture(stripe)
    gs, gv = constraints.residuals(perturbed, BETA)
    assert gs > 0 and gv > 0, "fixture must start infeasible"
    result = repair_to_feasible(
        constraints, perturbed, BETA, margin=1e-6, maxeval=2000)
    assert result["feasible"] is True, result["g_after"]
    assert result["g_after"][0] <= -1e-6 and result["g_after"][1] <= -1e-6
    # measured 18 evals / 2.1 s on this fixture -- generous headroom only
    assert 0 < result["evaluations"] <= 2000
    latent = result["latent"]
    assert latent.shape == perturbed.shape
    assert np.min(latent) >= 0.0 and np.max(latent) <= 1.0


def test_failed_repair_reports_honestly(small_problem):
    """A tiny eval budget must return feasible=False, never a fake success."""
    constraints, stripe = small_problem
    perturbed = _infeasible_fixture(stripe)
    result = repair_to_feasible(
        constraints, perturbed, BETA, margin=1e-6, maxeval=4)
    assert result["feasible"] is False
    assert result["evaluations"] >= 1
    # the returned latent is never worse than the input contract-wise
    assert result["latent"].shape == perturbed.shape


# --- runner wiring (source-level, same pattern as the other guards) ---------
def test_runner_repairs_before_gated_stage():
    src = RUNNER.read_text()
    repair_pos = src.find("repair_to_feasible(")
    controller_pos = src.find("controller = StageController(adaptive_cfg, beta)")
    assert 0 < repair_pos < controller_pos
    assert "beta >= adaptive_cfg.feasibility_gate_beta" in src
    assert '"repair_infeasible"' in src           # fail-closed abort category


def test_repair_and_seed_in_contract_before_config_hash():
    src = RUNNER.read_text()
    hash_pos = src.find('contract["config_hash"] = _config_hash(contract)')
    for key in ('"feasibility_repair"', '"seed"'):
        assert 0 < src.find(key) < hash_pos, key


def test_repair_and_seed_change_config_hash():
    runner = importlib.import_module("run_constrained_inverse_design")
    base = {"feasibility_repair": {"enabled": True, "maxeval": 60000,
                                   "margin": 1e-6},
            "seed": {"path": None, "source": "model.x0"}}
    h0 = runner._config_hash(base)
    changed = dict(base)
    changed["feasibility_repair"] = {"enabled": True, "maxeval": 60000,
                                     "margin": 1e-5}
    assert runner._config_hash(changed) != h0
    reseeded = dict(base)
    reseeded["seed"] = {"path": "/x/final_design.npz", "sha256": "abcd"}
    assert runner._config_hash(reseeded) != h0
