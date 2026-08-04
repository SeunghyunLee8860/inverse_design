from __future__ import annotations

import numpy as np

from photothermal_pte.finite_inverse_design.finite_mapping import (
    FiniteDensityMapping,
)


def test_finite_filter_is_constant_preserving_and_nonperiodic():
    mapping = FiniteDensityMapping(
        nx=11,
        ny=9,
        nz=3,
        dx_m=25e-9,
        dy_m=25e-9,
        filter_radius_m=100e-9,
    )
    constant = np.full(mapping.latent_shape, 0.37)
    assert np.allclose(mapping.filtered(constant), 0.37, atol=1e-15)
    impulse = np.zeros(mapping.latent_shape)
    impulse[0, 0] = 1.0
    filtered = mapping.filtered(impulse)
    assert filtered[-1, -1] == 0.0


def test_finite_mapping_jvp_vjp_identity():
    mapping = FiniteDensityMapping(
        nx=9,
        ny=8,
        nz=4,
        dx_m=25e-9,
        dy_m=25e-9,
        filter_radius_m=100e-9,
    )
    rng = np.random.default_rng(2026072611)
    latent = rng.uniform(0.2, 0.8, size=mapping.latent_shape)
    direction = rng.normal(size=mapping.latent_shape)
    sensitivity = rng.normal(size=mapping.physical_shape)
    left = float(np.sum(mapping.jvp(latent, direction) * sensitivity))
    right = float(np.sum(direction * mapping.vjp(latent, sensitivity)))
    assert np.isclose(left, right, rtol=2e-13, atol=1e-12)


def test_finite_mapping_jvp_matches_centered_fd():
    mapping = FiniteDensityMapping(
        nx=8,
        ny=7,
        nz=2,
        dx_m=25e-9,
        dy_m=25e-9,
        filter_radius_m=100e-9,
    )
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


def test_production_81x81_filter_constant_roundoff_is_accepted():
    mapping = FiniteDensityMapping()
    constant = np.full(mapping.latent_shape, 0.5)
    filtered = mapping.filtered(constant)
    assert np.max(np.abs(filtered - 0.5)) < 1e-14


def test_2d_mapping_jvp_vjp_and_centered_fd():
    mapping = FiniteDensityMapping(
        nx=13,
        ny=11,
        nz=5,
        dx_m=25e-9,
        dy_m=25e-9,
        filter_radius_m=100e-9,
    )
    rng = np.random.default_rng(20260728)
    latent = 0.35 + 0.3 * rng.random(mapping.latent_shape)
    direction = rng.normal(size=mapping.latent_shape)
    cotangent = rng.normal(size=mapping.latent_shape)
    jvp = mapping.jvp_2d(latent, direction)
    vjp = mapping.vjp_2d(latent, cotangent)
    left = float(np.vdot(cotangent, jvp).real)
    right = float(np.vdot(vjp, direction).real)
    assert abs(left - right) / max(abs(left), abs(right)) < 1e-12

    step = 1e-6
    finite_difference = (
        mapping.physical_2d(latent + step * direction)
        - mapping.physical_2d(latent - step * direction)
    ) / (2.0 * step)
    assert np.linalg.norm(finite_difference - jvp) / np.linalg.norm(jvp) < 1e-8


def test_3d_vjp_matches_single_extruded_2d_vjp():
    mapping = FiniteDensityMapping(
        nx=9,
        ny=7,
        nz=3,
        dx_m=25e-9,
        dy_m=25e-9,
        filter_radius_m=100e-9,
    )
    rng = np.random.default_rng(20260729)
    latent = 0.25 + 0.5 * rng.random(mapping.latent_shape)
    sensitivity_2d = rng.normal(size=mapping.latent_shape)
    sensitivity_3d = np.repeat(
        (sensitivity_2d / mapping.nz)[:, :, None],
        mapping.nz,
        axis=2,
    )
    assert np.allclose(
        mapping.vjp(latent, sensitivity_3d),
        mapping.vjp_2d(latent, sensitivity_2d),
        rtol=1e-13,
        atol=1e-13,
    )
