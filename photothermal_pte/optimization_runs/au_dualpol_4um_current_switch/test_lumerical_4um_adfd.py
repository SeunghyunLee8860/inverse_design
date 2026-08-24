from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_adfd import (
    centered_adfd_metrics,
    centered_density_pair,
    centered_pair_reconstruction_metrics,
    independent_smooth_direction,
)


def test_independent_direction_is_deterministic_smooth_and_normalized() -> None:
    first = independent_smooth_direction((81, 81))
    second = independent_smooth_direction((81, 81))
    assert np.array_equal(first, second)
    assert np.max(np.abs(first)) == 1.0
    assert np.max(np.abs(np.diff(first, axis=0))) < 0.1
    assert np.max(np.abs(np.diff(first, axis=1))) < 0.1


def test_centered_density_pair_is_exact_and_feasible() -> None:
    baseline = np.full((11, 9), 0.5)
    direction, plus, minus = centered_density_pair(baseline, step=0.0025)
    assert np.allclose(0.5 * (plus + minus), baseline, rtol=0.0, atol=1.0e-16)
    assert np.allclose((plus - minus) / 0.005, direction, rtol=0.0, atol=2.0e-14)
    with pytest.raises(ValueError, match="leaves"):
        centered_density_pair(np.zeros((11, 9)), step=0.0025)


def test_pair_reconstruction_uses_step_scaled_float64_roundoff() -> None:
    x = np.linspace(0.29, 0.71, 81, dtype=np.float64)
    baseline = np.broadcast_to(x[:, None], (81, 81)).copy()
    direction, plus, minus = centered_density_pair(baseline, step=0.0025)
    metrics = centered_pair_reconstruction_metrics(
        baseline=baseline,
        direction=direction,
        plus=plus,
        minus=minus,
        step=0.0025,
    )
    assert metrics["within_float64_roundoff"] is True
    assert metrics["midpoint_max_abs_error"] <= metrics[
        "midpoint_float64_roundoff_tolerance"
    ]
    assert metrics["direction_max_abs_error"] <= metrics[
        "direction_float64_roundoff_tolerance"
    ]


def test_centered_metrics_recovers_quadratic_directional_derivative() -> None:
    direction = independent_smooth_direction((9, 7))
    gradient = np.arange(direction.size, dtype=float).reshape(direction.shape) * 1.0e-12
    derivative = float(np.sum(gradient * direction))
    baseline = -5.0e-9
    step = 0.0025
    curvature = 3.0e-9
    plus = baseline + step * derivative + curvature * step**2
    minus = baseline - step * derivative + curvature * step**2
    metrics = centered_adfd_metrics(
        gradient=gradient,
        direction=direction,
        step=step,
        baseline_current_A=baseline,
        plus_current_A=plus,
        minus_current_A=minus,
    )
    assert metrics["relative_error"] < 2.0e-11
    assert metrics["same_nonzero_sign"] is True
    assert np.isclose(metrics["centered_midpoint_minus_baseline_A"], curvature * step**2)
