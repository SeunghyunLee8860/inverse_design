from __future__ import annotations

import numpy as np

from photothermal_pte.finite_inverse_design.finite_mapping import (
    _finite_conic_filter,
)
from photothermal_pte.optimization_runs.legacy_v261_optical_support.production_density_mapping import (
    ProductionDensityMapping,
)


def test_convolution_matches_sparse_finite_filter_and_transpose() -> None:
    shape = (23, 19)
    spacing = 50.0e-9
    radius = 250.0e-9
    mapping = ProductionDensityMapping(shape=shape, spacing_m=spacing, radius_m=radius)
    sparse = _finite_conic_filter(shape[0], shape[1], spacing, spacing, radius)
    rng = np.random.default_rng(8821)
    latent = rng.normal(size=shape)
    cotangent = rng.normal(size=shape)
    expected_forward = np.asarray(sparse @ latent.reshape(-1)).reshape(shape)
    expected_transpose = np.asarray(sparse.T @ cotangent.reshape(-1)).reshape(shape)
    np.testing.assert_allclose(mapping.filtered(latent), expected_forward, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(mapping.filter_transpose(cotangent), expected_transpose, rtol=2e-15, atol=2e-15)


def test_filter_has_no_opposite_edge_wrap_and_preserves_constants() -> None:
    mapping = ProductionDensityMapping(shape=(51, 47), spacing_m=50e-9, radius_m=500e-9)
    impulse = np.zeros(mapping.shape)
    impulse[0, mapping.shape[1] // 2] = 1.0
    filtered = mapping.filtered(impulse)
    assert np.max(np.abs(filtered[-10:, :])) == 0.0
    assert np.max(np.abs(mapping.filtered(np.ones(mapping.shape)) - 1.0)) < 1e-14


def test_projection_jvp_vjp_dot_identity() -> None:
    mapping = ProductionDensityMapping(shape=(43, 39), spacing_m=50e-9, radius_m=500e-9)
    rng = np.random.default_rng(91027)
    latent = 0.35 + 0.3 * rng.random(mapping.shape)
    direction = rng.normal(size=mapping.shape)
    cotangent = rng.normal(size=mapping.shape)
    jvp = mapping.jvp(latent, direction, beta=8.0)
    vjp = mapping.vjp(latent, cotangent, beta=8.0)
    left = float(np.vdot(jvp, cotangent))
    right = float(np.vdot(direction, vjp))
    scale = max(np.linalg.norm(jvp) * np.linalg.norm(cotangent), 1e-300)
    assert abs(left - right) / scale < 1e-12
