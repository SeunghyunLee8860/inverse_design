from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "39_validate_3d_drude_nanostructure_adfd.py"
SPEC = importlib.util.spec_from_file_location("au_3d_control", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_drude_endpoint_exact_and_passive():
    fit = MODULE.drude_fit_from_nk(MODULE.WAVELENGTH_M, MODULE.AU_N, MODULE.AU_K, MODULE.EPS_INF)
    assert fit["omega_p_rad_s"] > 0.0
    assert fit["gamma_rad_s"] > 0.0
    assert fit["epsilon_real"] < 0.0
    assert fit["epsilon_imag"] > 0.0
    assert fit["fit_relative_error"] < 1e-14


def test_baseline_leaves_unclipped_fd_margin():
    rho = MODULE._baseline_density(10, 10)
    assert np.all(rho > 0.15)
    assert np.all(rho < 0.85)


def test_all_direction_vectors_are_unit_norm():
    directions = MODULE._direction_set(10, 10, 20260821)
    assert set(directions) == {
        "uniform",
        "smooth_asymmetric",
        "central_localized",
        "design_edge_localized",
        "fixed_seed_random",
    }
    for value in directions.values():
        assert np.isclose(np.linalg.norm(value), 1.0, rtol=0, atol=1e-12)
