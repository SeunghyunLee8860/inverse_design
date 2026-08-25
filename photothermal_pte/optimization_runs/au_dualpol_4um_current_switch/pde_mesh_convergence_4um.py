"""Fail-closed adaptive custom-PDE lateral convergence metrics.

The optical Yee heat source is frozen while the thermal/electrical core grid
is refined by aligned 2x2 cell pairs, so no interpolator or smoothing
operation can hide a discretization change.
"""

from __future__ import annotations

from typing import Any

import numpy as np


PDE_STEPS_M = (100.0e-9, 50.0e-9, 25.0e-9, 12.5e-9)
COARSE_PDE_STEP_M = PDE_STEPS_M[0]
FINE_PDE_STEP_M = PDE_STEPS_M[1]
PDE_RELATIVE_GATE = 5.0e-3


def _finite_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def fine_to_coarse_cell_average(
    fine_field: np.ndarray, *, coarse_shape: tuple[int, int]
) -> np.ndarray:
    """Average one aligned square fine field onto its coarse cell grid."""

    fine = np.asarray(fine_field, dtype=np.float64)
    if (
        fine.ndim != 2
        or not np.all(np.isfinite(fine))
        or coarse_shape[0] <= 0
        or coarse_shape[1] <= 0
    ):
        raise ValueError("fine field and coarse shape must be finite 2-D grids")
    if fine.shape[0] % coarse_shape[0] or fine.shape[1] % coarse_shape[1]:
        raise ValueError("fine field is not exactly aligned with the coarse grid")
    factors = (
        fine.shape[0] // coarse_shape[0],
        fine.shape[1] // coarse_shape[1],
    )
    if factors != (2, 2):
        raise ValueError("the approved PDE comparison requires exact 2x2 refinement")
    return fine.reshape(coarse_shape[0], factors[0], coarse_shape[1], factors[1]).mean(
        axis=(1, 3)
    )


def _relative_change(coarse: float, fine: float) -> float:
    return abs(coarse - fine) / max(abs(fine), np.finfo(float).tiny)


def pde_mesh_convergence_audit(
    *,
    coarse_current_A: float,
    fine_current_A: float,
    coarse_ta_temperature_K: np.ndarray,
    fine_ta_temperature_K: np.ndarray,
    coarse_peak_temperature_K: float,
    fine_peak_temperature_K: float,
    coarse_step_m: float = COARSE_PDE_STEP_M,
    fine_step_m: float = FINE_PDE_STEP_M,
    relative_gate: float = PDE_RELATIVE_GATE,
) -> dict[str, Any]:
    """Compare one aligned 2x PDE pair using the fine grid as reference."""

    gate = _finite_scalar(relative_gate, name="relative_gate")
    if gate <= 0.0:
        raise ValueError("relative_gate must be positive")
    coarse_step = _finite_scalar(coarse_step_m, name="coarse_step_m")
    fine_step = _finite_scalar(fine_step_m, name="fine_step_m")
    approved = all(
        any(np.isclose(step, item, rtol=0.0, atol=1.0e-18) for item in PDE_STEPS_M)
        for step in (coarse_step, fine_step)
    )
    if not approved or not np.isclose(
        coarse_step, 2.0 * fine_step, rtol=0.0, atol=1.0e-18
    ):
        raise ValueError("PDE comparison must be one approved aligned 2x pair")
    coarse_current = _finite_scalar(coarse_current_A, name="coarse_current_A")
    fine_current = _finite_scalar(fine_current_A, name="fine_current_A")
    coarse_peak = _finite_scalar(
        coarse_peak_temperature_K, name="coarse_peak_temperature_K"
    )
    fine_peak = _finite_scalar(fine_peak_temperature_K, name="fine_peak_temperature_K")
    coarse_temperature = np.asarray(coarse_ta_temperature_K, dtype=np.float64)
    fine_temperature = np.asarray(fine_ta_temperature_K, dtype=np.float64)
    if coarse_temperature.ndim != 2 or not np.all(np.isfinite(coarse_temperature)):
        raise ValueError("coarse TaIrTe4 temperature must be a finite 2-D field")
    fine_on_coarse = fine_to_coarse_cell_average(
        fine_temperature, coarse_shape=coarse_temperature.shape
    )
    difference = coarse_temperature - fine_on_coarse
    field_nrmse = float(
        np.linalg.norm(difference.ravel())
        / max(np.linalg.norm(fine_on_coarse.ravel()), np.finfo(float).tiny)
    )
    coarse_mean = float(np.mean(coarse_temperature))
    fine_mean = float(np.mean(fine_temperature))
    metrics = {
        "current_relative_change": _relative_change(coarse_current, fine_current),
        "ta_temperature_field_nrmse": field_nrmse,
        "ta_mean_temperature_relative_change": _relative_change(coarse_mean, fine_mean),
        "peak_temperature_relative_change": _relative_change(coarse_peak, fine_peak),
    }
    gates = {
        "coarse_and_fine_current_nonzero": coarse_current != 0.0
        and fine_current != 0.0,
        "current_sign_preserved": bool(
            np.signbit(coarse_current) == np.signbit(fine_current)
        ),
        "current_relative_change_lt_0p5pct": (
            metrics["current_relative_change"] < gate
        ),
        "ta_temperature_field_nrmse_lt_0p5pct": (
            metrics["ta_temperature_field_nrmse"] < gate
        ),
        "ta_mean_temperature_relative_change_lt_0p5pct": (
            metrics["ta_mean_temperature_relative_change"] < gate
        ),
        "peak_temperature_relative_change_lt_0p5pct": (
            metrics["peak_temperature_relative_change"] < gate
        ),
    }
    return {
        "method": "same_raw_Yee_Q_aligned_2x_custom_PDE_pair_v2",
        "coarse_step_m": coarse_step,
        "fine_step_m": fine_step,
        "relative_gate": gate,
        "coarse_current_A": coarse_current,
        "fine_current_A": fine_current,
        "coarse_ta_mean_temperature_K": coarse_mean,
        "fine_ta_mean_temperature_K": fine_mean,
        "coarse_peak_temperature_K": coarse_peak,
        "fine_peak_temperature_K": fine_peak,
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
    }
