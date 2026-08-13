from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.build_exact_feasible_latent_seed import (
    inverse_filter_objective,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import MAPPING


def test_inverse_filter_gradient_dot_product() -> None:
    rng = np.random.default_rng(20260813)
    latent = rng.uniform(0.15, 0.85, size=MAPPING.shape)
    target = rng.uniform(size=MAPPING.shape) > 0.5
    direction = rng.normal(size=MAPPING.shape)
    direction /= np.linalg.norm(direction)
    step = 1.0e-6
    value, gradient = inverse_filter_objective(
        latent.ravel(), target=target, margin=0.1, regularization=1.0e-5
    )
    plus, _ = inverse_filter_objective(
        (latent + step * direction).ravel(),
        target=target,
        margin=0.1,
        regularization=1.0e-5,
    )
    minus, _ = inverse_filter_objective(
        (latent - step * direction).ravel(),
        target=target,
        margin=0.1,
        regularization=1.0e-5,
    )
    finite_difference = (plus - minus) / (2.0 * step)
    adjoint = float(np.vdot(gradient, direction.ravel()))
    assert np.isfinite(value)
    assert abs(adjoint - finite_difference) <= 1.0e-7 * max(1.0, abs(finite_difference))
