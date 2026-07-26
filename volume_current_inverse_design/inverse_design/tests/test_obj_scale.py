"""nlopt-only objective scaling (CCSA rho warm-up fix, connected r1 post-mortem).

Measured failure being guarded: with the physical objective at ~1e-6, nlopt
CCSA's O(1) initial rho throttles steps to |grad|/rho, growing only ~10x per
outer iteration (4.45e-6 -> 4.45e-5 in run r1) -- the stall detector then
fires during warm-up and aborts an otherwise healthy run.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest

RUNNER = Path(__file__).resolve().parents[1] / "run_constrained_inverse_design.py"


def test_value_and_gradient_scaled_as_a_pair():
    src = RUNNER.read_text()
    assert "return -f * args.obj_scale" in src
    assert "grad[:] = -dlat * args.obj_scale" in src
    # physical values must keep flowing to controller/history/checkpoints:
    # the controller records f, not the scaled value.
    assert "_controller.record(f, gs, gv, x)" in src
    assert 'np.savez_compressed(tmp, latent=x, beta=np.array(_beta),\n' \
           '                                    objective=np.array(f))' in src


def test_obj_scale_in_contract_before_config_hash():
    src = RUNNER.read_text()
    key = src.find('"objective_scale_nlopt"')
    hash_pos = src.find('contract["config_hash"] = _config_hash(contract)')
    assert 0 < key < hash_pos


def test_obj_scale_changes_config_hash():
    runner = importlib.import_module("run_constrained_inverse_design")
    a = runner._config_hash({"objective_scale_nlopt": 1e6})
    b = runner._config_hash({"objective_scale_nlopt": 1.0})
    assert a != b


def test_scaling_preserves_optimum_and_direction():
    """Scaling (f, grad) by c>0 must not move the optimum of a toy problem."""
    nlopt = pytest.importorskip("nlopt")
    solutions = {}
    for scale in (1.0, 1e6):
        opt = nlopt.opt(nlopt.LD_MMA, 2)
        opt.set_lower_bounds([0.0, 0.0])
        opt.set_upper_bounds([1.0, 1.0])
        opt.set_maxeval(80)
        opt.set_xtol_rel(1e-9)

        def objective(x, grad, _c=scale):
            value = 1e-6 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2)
            if grad.size:
                grad[:] = _c * 1e-6 * 2.0 * (x - np.array([0.3, 0.7]))
            return _c * value

        opt.set_min_objective(objective)
        solutions[scale] = opt.optimize(np.array([0.9, 0.1]))
    # the O(1)-scaled problem must find the optimum accurately; the unscaled
    # 1e-6 problem is exactly the rho-warm-up pathology (few useful steps in
    # 80 evals), which is WHY production scales the objective.
    assert np.allclose(solutions[1e6], [0.3, 0.7], atol=1e-3), solutions[1e6]


def test_default_scale_env_override():
    import os
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, "-c", (
            "import os; os.environ['VC_OBJ_SCALE']='123.5';"
            "import argparse; ap=argparse.ArgumentParser();"
            "ap.add_argument('--obj-scale', type=float,"
            " default=float(os.environ.get('VC_OBJ_SCALE','1e6')));"
            "print(ap.parse_args([]).obj_scale)")],
        capture_output=True, text=True, env=dict(os.environ))
    assert proc.stdout.strip() == "123.5"
