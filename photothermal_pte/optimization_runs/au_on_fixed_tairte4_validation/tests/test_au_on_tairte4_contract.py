from __future__ import annotations

import cmath
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "41_validate_au_on_fixed_tairte4_optical_adfd.py"
SPEC = importlib.util.spec_from_file_location("au_on_tairte4_control", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _discrete_epsilon(fit: dict[str, float], omega: float, dt: float) -> complex:
    c1, c2, c3 = MODULE._coefficient_triplet(fit, dt)
    theta = omega * dt
    return MODULE.EPS_INF + c3 / (cmath.exp(-1j * theta) - c1 - c2 * cmath.exp(1j * theta))


def test_repository_axis_mapping_and_out_of_plane_closure():
    epsilon = MODULE._load_tairte4_epsilon()
    assert epsilon["c"] == epsilon["b"]
    assert epsilon["a"].real < 0.0
    assert epsilon["a"].imag > 0.0
    assert epsilon["b"].real > 0.0
    assert epsilon["b"].imag > 0.0


def test_finite_dt_causal_fits_reproduce_all_endpoints():
    omega = 2.0 * math.pi * MODULE.C0_M_PER_S / MODULE.WAVELENGTH_M
    dt = 4.8146e-17
    epsilon = MODULE._load_tairte4_epsilon()
    targets_and_fits = (
        (complex(MODULE.AU_N, MODULE.AU_K) ** 2, MODULE._drude_fit(complex(MODULE.AU_N, MODULE.AU_K) ** 2, omega, dt)),
        (epsilon["a"], MODULE._drude_fit(epsilon["a"], omega, dt)),
        (epsilon["b"], MODULE._lorentz_fit(epsilon["b"], omega, dt)),
        (epsilon["c"], MODULE._lorentz_fit(epsilon["c"], omega, dt)),
    )
    for target, fit in targets_and_fits:
        realized = _discrete_epsilon(fit, omega, dt)
        assert fit["gamma_rad_s"] > 0.0
        assert fit["coupling_sq_rad2_s2"] > 0.0
        assert fit["gamma_rad_s"] * dt < 2.0
        assert abs(realized - target) / abs(target) < 1e-10


def test_baseline_and_directions_are_unclipped_and_normalized():
    rho = MODULE._baseline_density(10, 10)
    assert np.all(rho > 0.15)
    assert np.all(rho < 0.85)
    directions = MODULE._direction_set(10, 10, 20260821)
    assert set(directions) == {
        "uniform",
        "smooth_asymmetric",
        "central_localized",
        "design_edge_localized",
        "fixed_seed_random",
    }
    for direction in directions.values():
        assert np.isclose(np.linalg.norm(direction), 1.0, rtol=0.0, atol=1e-12)
