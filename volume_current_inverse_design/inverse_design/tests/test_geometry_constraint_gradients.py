"""Solid/void length-scale constraint values and autograd gradients."""

import numpy as np
import pytest

from periodic_constrained_mapping import PeriodicConstrainedMapping
from geometric_constraints import LengthScaleConstraints, suggested_filter_radius_um


def _setup():
    m = PeriodicConstrainedMapping(241, 241, 13, period_um=6.0, filter_radius_um=0.5)
    return m, LengthScaleConstraints(m)


def test_constraint_values_finite_nonnegative():
    m, con = _setup()
    lat = np.clip(0.5 + 0.2 * np.random.default_rng(0).standard_normal(m.Nux * m.Nuy),
                  0.05, 0.95)
    gs = float(con.solid_penalty(lat, 8.0))
    gv = float(con.void_penalty(lat, 8.0))
    assert np.isfinite(gs) and gs >= 0.0
    assert np.isfinite(gv) and gv >= 0.0


@pytest.mark.parametrize("which", ["solid", "void"])
def test_constraint_gradient_matches_fd(which):
    m, con = _setup()
    rng = np.random.default_rng(1)
    lat = np.clip(0.5 + 0.2 * rng.standard_normal(m.Nux * m.Nuy), 0.05, 0.95)
    beta = 8.0
    fn = con.solid_penalty if which == "solid" else con.void_penalty
    rg = con.solid_residual_and_grad if which == "solid" else con.void_residual_and_grad
    _, g = rg(lat, beta)
    d = rng.standard_normal(lat.size); d /= np.linalg.norm(d)
    ad = float(np.dot(g, d))
    h = 1e-6
    fd = (float(fn(lat + h * d, beta)) - float(fn(lat - h * d, beta))) / (2 * h)
    assert abs(ad - fd) / max(abs(ad), abs(fd), 1e-300) < 1e-4


def test_suggested_radius_positive():
    assert suggested_filter_radius_um(0.5, 0.75) > 0
