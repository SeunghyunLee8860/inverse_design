"""Balanced opposite-current objective for the two input polarizations."""

from __future__ import annotations

import numpy as np


PTE_CURRENT_SIGN_CONVENTION = (
    "I=integral((-sigma*S*grad(T)).grad(psi))dA; "
    "psi(x_min)=0, psi(x_max)=1; positive I is conventional current along +x"
)


def useful_currents(current_a_A: float, current_b_A: float) -> tuple[float, float]:
    """Return positive utilities for the requested current directions.

    With psi=0 at x_min and psi=1 at x_max, positive ``I=integral(J.grad(psi))``
    is the +x component of internal conventional current (left to right).
    """

    return float(current_a_A), -float(current_b_A)


def opposite_current_switching_achieved(
    current_a_A: float, current_b_A: float
) -> bool:
    """Return whether both requested strict current directions are present."""

    currents = np.asarray((current_a_A, current_b_A), dtype=np.float64)
    return bool(
        np.all(np.isfinite(currents))
        and float(current_a_A) > 0.0
        and float(current_b_A) < 0.0
    )


def exact_binary_promotion_passed(
    numerical_gates_passed: bool,
    current_a_A: float,
    current_b_A: float,
) -> bool:
    """Require numerical health and the target signs for binary promotion."""

    return bool(
        numerical_gates_passed
        and opposite_current_switching_achieved(current_a_A, current_b_A)
    )


def smooth_minimum(
    current_a_A: float,
    current_b_A: float,
    scale_A: float,
    sharpness: float = 12.0,
) -> tuple[float, tuple[float, float]]:
    """Differentiable diagnostic approximation to min(I_a, -I_b).

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
    # dF/dIa = dF/dua; dF/dIb = -dF/dub.
    return float(result), (float(weights[0]), -float(weights[1]))


def epigraph_constraints(
    current_a_A: float, current_b_A: float, epigraph_A: float
) -> np.ndarray:
    """NLopt convention g<=0 for t-Ia and t-(-Ib)."""

    ua, ub = useful_currents(current_a_A, current_b_A)
    return np.asarray((epigraph_A - ua, epigraph_A - ub), dtype=np.float64)
