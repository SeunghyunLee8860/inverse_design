"""Periodic conic filter + PeriodicConstrainedMapping: shape, seam, VJP."""

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import grad

from periodic_filter import periodic_conic_filter_unique, measured_kernel
from periodic_constrained_mapping import PeriodicConstrainedMapping

N = 240
DX = DY = 0.025
R = 0.5


def test_filter_shape_and_constant():
    x = np.random.default_rng(0).random((N, N))
    y = periodic_conic_filter_unique(x, R, DX, DY)
    assert y.shape == (N, N)
    c = 0.37 * np.ones((N, N))
    assert np.max(np.abs(periodic_conic_filter_unique(c, R, DX, DY) - 0.37)) < 1e-12


@pytest.mark.parametrize("axis", [0, 1])
def test_filter_roll_equivariance(axis):
    x = np.random.default_rng(1).random((N, N))
    y = periodic_conic_filter_unique(x, R, DX, DY)
    lhs = periodic_conic_filter_unique(np.roll(x, 7, axis=axis), R, DX, DY)
    assert np.max(np.abs(lhs - np.roll(y, 7, axis=axis))) < 1e-12


def test_filter_impulse_is_kernel():
    imp = np.zeros((N, N))
    imp[0, 0] = 1.0
    y = periodic_conic_filter_unique(imp, R, DX, DY)
    assert np.max(np.abs(y - measured_kernel(N, N, R, DX, DY))) < 1e-12


def test_filter_vjp():
    rng = np.random.default_rng(2)
    x = rng.random((N, N))
    w = rng.normal(size=(N, N))
    d = rng.normal(size=(N, N)); d /= np.linalg.norm(d)
    f = lambda v: anp.sum(periodic_conic_filter_unique(v, R, DX, DY) * w)
    ad = float(np.sum(np.asarray(grad(f)(x)) * d))
    h = 1e-5
    fd = (f(x + h * d) - f(x - h * d)) / (2 * h)
    assert abs(ad - fd) / max(abs(ad), abs(fd)) < 1e-5


def _mapping():
    return PeriodicConstrainedMapping(241, 241, 13, period_um=6.0, filter_radius_um=R)


def test_mapping_shapes_and_seam():
    m = _mapping()
    lat = np.random.default_rng(3).random(m.Nux * m.Nuy)
    phys = np.asarray(m(lat, 8.0)).reshape(241, 241, 13)
    assert phys.shape == (241, 241, 13)
    assert np.max(np.abs(phys[-1] - phys[0])) == 0.0
    assert np.max(np.abs(phys[:, -1] - phys[:, 0])) == 0.0
    assert np.max(np.abs(phys - phys[:, :, :1])) == 0.0
    assert 0.0 <= phys.min() and phys.max() <= 1.0


def test_mapping_uniform_no_bias():
    m = _mapping()
    uni = 0.5 * np.ones(m.Nux * m.Nuy)
    phys = np.asarray(m(uni, 8.0))
    assert abs(float(phys.mean()) - 0.5) < 1e-9
    assert float(phys.min()) == pytest.approx(0.5, abs=1e-9)


def test_mapping_vjp():
    m = _mapping()
    rng = np.random.default_rng(4)
    lat = rng.random(m.Nux * m.Nuy)
    w = rng.normal(size=241 * 241 * 13)
    d = rng.normal(size=lat.size); d /= np.linalg.norm(d)
    f = lambda v: anp.sum(m(v, 8.0) * w)
    ad = float(np.sum(np.asarray(grad(f)(lat)) * d))
    h = 1e-6
    fd = (f(lat + h * d) - f(lat - h * d)) / (2 * h)
    assert abs(ad - fd) / max(abs(ad), abs(fd)) < 1e-5
