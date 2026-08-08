#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from beta_continuation_support import (
    design_metrics,
    initialize_mma_state,
    mma_step,
)
from production_density_mapping import ProductionDensityMapping


def test_constraint_gradients_directional_fd() -> None:
    mapping = ProductionDensityMapping(shape=(41, 39), spacing_m=50e-9, radius_m=500e-9)
    rng = np.random.default_rng(17)
    latent = np.clip(0.5 + 0.12 * rng.standard_normal(mapping.shape), 0.1, 0.9)
    direction = rng.standard_normal(mapping.shape)
    direction /= np.linalg.norm(direction)
    metrics, arrays = design_metrics(latent, 8.0, mapping)
    for name, key in (
        ("solid", "smooth_solid_constraint"),
        ("void", "smooth_void_constraint"),
        ("gray", "binarization_metric_mean_4rho1mrho"),
    ):
        h = 1.0e-4
        plus, _ = design_metrics(latent + h * direction, 8.0, mapping)
        minus, _ = design_metrics(latent - h * direction, 8.0, mapping)
        finite_difference = (float(plus[key]) - float(minus[key])) / (2.0 * h)
        adjoint = float(np.sum(arrays[f"gradient_{name}"] * direction))
        assert abs(adjoint - finite_difference) / max(abs(finite_difference), 1e-12) < 2e-4


def test_stateful_mma_respects_bounds_and_constraint() -> None:
    x = np.asarray([0.8, 0.8])
    state = initialize_mma_state(x)
    # Minimize x0+x1 subject to 1-x0-x1 <= 0.
    candidate, state, diagnostics = mma_step(
        x,
        np.asarray([1.0, 1.0]),
        np.asarray([1.0 - np.sum(x)]),
        np.asarray([[-1.0, -1.0]]),
        state,
        move=0.1,
    )
    assert state.iteration == 1
    assert np.all(candidate >= 0.0) and np.all(candidate <= 1.0)
    assert np.max(np.abs(candidate - x)) <= 0.1000001
    assert diagnostics["dual_success"]

