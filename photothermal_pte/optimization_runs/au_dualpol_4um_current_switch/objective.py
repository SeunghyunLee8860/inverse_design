"""Balanced opposite-current objective for the two input polarizations."""

from __future__ import annotations

import numpy as np


def useful_currents(current_a_A: float, current_b_A: float) -> tuple[float, float]:
    """Return positive utilities for the requested current directions.

    The repository sign convention is +I from x_min (left) to x_max (right).
    Therefore E||a is useful when I_a is negative and E||b when I_b is positive.
    """

    return -float(current_a_A), float(current_b_A)


def smooth_minimum(
    current_a_A: float,
    current_b_A: float,
    scale_A: float,
    sharpness: float = 12.0,
) -> tuple[float, tuple[float, float]]:
    """Differentiable diagnostic approximation to min(-I_a, I_b).

    Production MMA uses the exact epigraph inequalities.  This function is
    used only by smoke tests and plots, and returns dF/d(I_a,I_b).
    """

    if scale_A <= 0.0 or sharpness <= 0.0:
        raise ValueError("scale_A and sharpness must be positive")
    ua, ub = useful_currents(current_a_A, current_b_A)
    values = np.asarray((ua, ub), dtype=np.float64) / scale_A
    shifted = -sharpness * values
    pivot = float(np.max(shifted))
    weights = np.exp(shifted - pivot)
    weights /= np.sum(weights)
    result = -scale_A * (pivot + np.log(np.sum(np.exp(shifted - pivot)))) / sharpness
    # dF/dIa = -dF/dua; dF/dIb = dF/dub.
    return float(result), (-float(weights[0]), float(weights[1]))


def epigraph_constraints(
    current_a_A: float, current_b_A: float, epigraph_A: float
) -> np.ndarray:
    """NLopt convention g<=0 for t-(-Ia) and t-Ib."""

    ua, ub = useful_currents(current_a_A, current_b_A)
    return np.asarray((epigraph_A - ua, epigraph_A - ub), dtype=np.float64)

