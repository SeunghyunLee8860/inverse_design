from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.tairte4_flake_topology.dual_polarization import (
    smooth_min_and_gradient,
)


def test_smooth_min_gradient_matches_finite_difference() -> None:
    currents = {"Ea": 4.0, "Eb": 9.0}
    gradients = {
        "Ea": np.asarray((1.5, -0.2)),
        "Eb": np.asarray((-0.3, 0.8)),
    }
    objective, gradient, weights = smooth_min_and_gradient(
        currents, gradients, temperature_A=2.0
    )
    assert currents["Ea"] < objective < 0.5 * (currents["Ea"] + currents["Eb"])
    assert weights["Ea"] > weights["Eb"]

    direction = np.asarray((0.4, -0.7))
    step = 1.0e-6
    plus = {
        name: currents[name] + step * float(np.dot(gradients[name], direction))
        for name in currents
    }
    minus = {
        name: currents[name] - step * float(np.dot(gradients[name], direction))
        for name in currents
    }
    zero = {name: np.zeros_like(direction) for name in currents}
    plus_value, _, _ = smooth_min_and_gradient(plus, zero, temperature_A=2.0)
    minus_value, _, _ = smooth_min_and_gradient(minus, zero, temperature_A=2.0)
    finite_difference = (plus_value - minus_value) / (2.0 * step)
    assert finite_difference == pytest.approx(float(np.dot(gradient, direction)), rel=1e-8)


def test_smooth_min_accepts_signed_current_and_rejects_bad_temperature() -> None:
    gradients = {"Ea": np.ones(2), "Eb": np.ones(2)}
    objective, gradient, _ = smooth_min_and_gradient(
        {"Ea": -1.0, "Eb": 0.0}, gradients, temperature_A=0.2
    )
    assert np.isfinite(objective)
    assert np.all(np.isfinite(gradient))
    with pytest.raises(ValueError, match="temperature"):
        smooth_min_and_gradient(
            {"Ea": 1.0, "Eb": 0.0}, gradients, temperature_A=0.0
        )
