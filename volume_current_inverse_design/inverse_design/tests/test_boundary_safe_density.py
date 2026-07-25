"""Solver-safe affine density layer: probe stays in [0,1], chain factor exact.

Pure-arithmetic test of the affine map used inside
VolumeCurrentEvaluator.value_and_gradient(density_mode='probe_safe'); the full
near-rail AD/FD chain requires Lumerical and is a Go/No-Go gate, not a unit test.
"""

import numpy as np

RHO_STEP = 0.001
SLACK = max(1e-6, 100 * np.finfo(float).eps)
DELTA = RHO_STEP + SLACK
CHAIN = 1.0 - 2.0 * DELTA

RAILS = np.array([0.0, 1e-8, RHO_STEP, 0.5, 1 - RHO_STEP, 0.999, 1 - 1e-8, 1.0])


def rho_solver(rho_geom):
    return DELTA + CHAIN * rho_geom


def test_probe_inside_unit_interval():
    rs = rho_solver(RAILS)
    assert np.all(rs - RHO_STEP >= 0.0)
    assert np.all(rs + RHO_STEP <= 1.0)


def test_endpoints_map_into_open_interval():
    assert rho_solver(0.0) > 0.0
    assert rho_solver(1.0) < 1.0
    assert rho_solver(0.0) == DELTA
    assert abs(rho_solver(1.0) - (1.0 - DELTA)) < 1e-15


def test_chain_factor_consistency():
    # dF/drho_geom = chain * dF/drho_solver ; verify d(rho_solver)/d(rho_geom)=chain
    h = 1e-7
    num = (rho_solver(0.5 + h) - rho_solver(0.5 - h)) / (2 * h)
    assert abs(num - CHAIN) < 1e-9


def test_layer_present_in_evaluator_source():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "volume_current_evaluator.py").read_text()
    assert "solver_safe_affine" in src
    assert "delta + chain * rho_geom" in src
