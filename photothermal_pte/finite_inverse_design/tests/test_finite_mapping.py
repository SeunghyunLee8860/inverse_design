from __future__ import annotations

import numpy as np

from photothermal_pte.finite_inverse_design.finite_mapping import (
    FiniteDensityMapping,
)


def test_finite_filter_is_constant_preserving_and_nonperiodic():
    mapping = FiniteDensityMapping(nx=11, ny=9, nz=3, dx_m=25e-9, dy_m=25e-9, filter_radius_m=100e-9)
    constant = np.full(mapping.latent_shape, 0.37)
    assert np.allclose(mapping.filtered(constant), 0.37, atol=1e-15)
    impulse = np.zeros(mapping.latent_shape)
    impulse[0, 0] = 1.0
    filtered = mapping.filtered(impulse)
    assert filtered[-1, -1] == 0.0


def test_finite_mapping_jvp_vjp_identity():
    mapping = FiniteDensityMapping(nx=9, ny=8, nz=4, dx_m=25e-9, dy_m=25e-9, filter_radius_m=100e-9)
    rng = np.random.default_rng(2026072611)
    latent = rng.uniform(0.2, 0.8, size=mapping.latent_shape)
    direction = rng.normal(size=mapping.latent_shape)
    sensitivity = rng.normal(size=mapping.physical_shape)
    left = float(np.sum(mapping.jvp(latent, direction) * sensitivity))
    right = float(np.sum(direction * mapping.vjp(latent, sensitivity)))
    assert np.isclose(left, right, rtol=2e-13, atol=1e-12)


def test_finite_mapping_jvp_matches_centered_fd():
    mapping = FiniteDensityMapping(nx=8, ny=7, nz=2, dx_m=25e-9, dy_m=25e-9, filter_radius_m=100e-9)
    rng = np.random.default_rng(2026072612)
    latent = rng.uniform(0.3, 0.7, size=mapping.latent_shape)
    direction = rng.normal(size=mapping.latent_shape)
    direction /= np.max(np.abs(direction))
    analytic = mapping.jvp(latent, direction)
    step = 1e-6
    finite_difference = (
        mapping.physical(latent + step * direction)
        - mapping.physical(latent - step * direction)
    ) / (2.0 * step)
    assert np.allclose(analytic, finite_difference, rtol=2e-8, atol=2e-10)
