"""Constraint DECISION power (not just gradient): worst-case not diluted."""

import numpy as np

from periodic_constrained_mapping import PeriodicConstrainedMapping
from geometric_constraints import LengthScaleConstraints


def _con():
    m = PeriodicConstrainedMapping(241, 241, 13, period_um=6.0, filter_radius_um=0.5)
    return m, LengthScaleConstraints(m)


def test_power_mean_not_diluted_by_grid_size():
    _, con = _con()
    field = np.zeros(240 * 240)
    field[12345] = 1e-3
    pm = float(con._pmean(field))
    mn = float(np.mean(field))
    # arithmetic mean dilutes a single node ~1/57600; power mean must not.
    assert pm / mn > 1000.0


def _lowfreq(k, rng):
    sp = np.zeros((240, 240), complex)
    for kx in range(-k, k + 1):
        for ky in range(-k, k + 1):
            if kx or ky:
                sp[kx % 240, ky % 240] = rng.standard_normal() + 1j * rng.standard_normal()
    fld = np.fft.ifft2(sp).real
    fld /= fld.std() + 1e-12
    return np.clip(0.5 + 0.35 * fld, 0.02, 0.98).reshape(-1)


def test_thin_design_penalised_more_than_smooth():
    _, con = _con()
    rng = np.random.default_rng(0)
    thin = _lowfreq(30, rng)   # many sub-length-scale features
    thick = _lowfreq(3, rng)   # few large features
    beta = 16.0
    assert float(con.solid_penalty(thin, beta)) > float(con.solid_penalty(thick, beta))
    assert float(con.void_penalty(thin, beta)) > float(con.void_penalty(thick, beta))
