"""Pure helpers for one bounded Lumerical projected-density AD--FD gate."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


SMOOTH_DIRECTION_COEFFICIENTS = (
    (0.73, 0.41, 0.17, 0.31, -0.67, -0.09),
    (0.23, -0.33, 0.04, -0.85, 0.49, 0.04),
    (0.70, -0.66, 0.10, -0.60, -0.81, -0.25),
    (0.40, 0.37, 0.12, 0.52, 0.72, 0.19),
)


def array_sha256(value: np.ndarray, *, label: str) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(label.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def smooth_direction_definition(direction_index: int) -> str:
    """Return the auditable analytic definition for one direction."""

    if not isinstance(direction_index, (int, np.integer)):
        raise ValueError("direction index must be an integer")
    index = int(direction_index)
    if not 0 <= index < len(SMOOTH_DIRECTION_COEFFICIENTS):
        raise ValueError(
            f"direction index must lie in [0,{len(SMOOTH_DIRECTION_COEFFICIENTS) - 1}]"
        )
    a, b, c, d, e, f = SMOOTH_DIRECTION_COEFFICIENTS[index]
    return (
        f"sin(pi*({a:g}*x{b:+g}*y{c:+g}))*"
        f"cos(pi*({d:g}*x{e:+g}*y{f:+g})); "
        "x,y are independent normalized nodal coordinates; L_inf normalized"
    )


def independent_smooth_direction(
    shape: tuple[int, int], direction_index: int = 0
) -> np.ndarray:
    """Return one fixed low-frequency direction independent of all AD data."""

    if shape[0] < 2 or shape[1] < 2:
        raise ValueError("AD-FD direction requires at least a 2x2 density grid")
    # Validate and expose the exact formula through the same coefficient table.
    smooth_direction_definition(direction_index)
    a, b, c, d, e, f = SMOOTH_DIRECTION_COEFFICIENTS[int(direction_index)]
    x = np.linspace(-1.0, 1.0, shape[0], dtype=np.float64)[:, None]
    y = np.linspace(-1.0, 1.0, shape[1], dtype=np.float64)[None, :]
    direction = np.sin(np.pi * (a * x + b * y + c)) * np.cos(
        np.pi * (d * x + e * y + f)
    )
    direction /= np.max(np.abs(direction))
    return np.ascontiguousarray(direction)


def centered_density_pair(
    baseline: np.ndarray, *, step: float, direction_index: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = np.asarray(baseline, dtype=np.float64)
    if rho.ndim != 2 or not np.all(np.isfinite(rho)):
        raise ValueError("baseline density must be one finite 2-D array")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("centered AD-FD step must be finite and positive")
    direction = independent_smooth_direction(rho.shape, direction_index)
    plus = rho + step * direction
    minus = rho - step * direction
    if np.min(plus) < 0.0 or np.max(plus) > 1.0:
        raise ValueError("positive perturbation leaves [0,1]")
    if np.min(minus) < 0.0 or np.max(minus) > 1.0:
        raise ValueError("negative perturbation leaves [0,1]")
    return direction, np.ascontiguousarray(plus), np.ascontiguousarray(minus)


def centered_pair_reconstruction_metrics(
    *,
    baseline: np.ndarray,
    direction: np.ndarray,
    plus: np.ndarray,
    minus: np.ndarray,
    step: float,
) -> dict[str, Any]:
    """Check a saved float64 pair against bounds derived from roundoff."""

    arrays = [np.asarray(value, dtype=np.float64) for value in (baseline, direction, plus, minus)]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("centered-pair array shapes differ")
    if any(value.ndim != 2 or not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("centered-pair arrays must be finite and 2-D")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("centered-pair step must be finite and positive")
    rho, vector, rho_plus, rho_minus = arrays
    midpoint_error = float(np.max(np.abs(0.5 * (rho_plus + rho_minus) - rho)))
    direction_error = float(
        np.max(np.abs((rho_plus - rho_minus) / (2.0 * step) - vector))
    )
    density_scale = max(
        1.0,
        float(np.max(np.abs(rho))),
        float(np.max(np.abs(rho_plus))),
        float(np.max(np.abs(rho_minus))),
    )
    epsilon = float(np.finfo(np.float64).eps)
    # The direction reconstruction subtracts two O(rho) values and divides by
    # 2h. Its absolute roundoff therefore scales as eps*|rho|/h, unlike the
    # midpoint reconstruction. Constants include the saved-input and arithmetic
    # rounding operations without relaxing either physical AD--FD gate.
    midpoint_tolerance = 8.0 * epsilon * density_scale
    direction_tolerance = 16.0 * epsilon * density_scale / step
    return {
        "midpoint_max_abs_error": midpoint_error,
        "direction_max_abs_error": direction_error,
        "midpoint_float64_roundoff_tolerance": midpoint_tolerance,
        "direction_float64_roundoff_tolerance": direction_tolerance,
        "within_float64_roundoff": bool(
            midpoint_error <= midpoint_tolerance
            and direction_error <= direction_tolerance
        ),
    }


def centered_adfd_metrics(
    *,
    gradient: np.ndarray,
    direction: np.ndarray,
    step: float,
    baseline_current_A: float,
    plus_current_A: float,
    minus_current_A: float,
) -> dict[str, Any]:
    grad = np.asarray(gradient, dtype=np.float64)
    vector = np.asarray(direction, dtype=np.float64)
    if grad.shape != vector.shape:
        raise ValueError("gradient and direction shapes differ")
    scalars = np.asarray(
        [step, baseline_current_A, plus_current_A, minus_current_A], float
    )
    if not np.all(np.isfinite(grad)) or not np.all(np.isfinite(vector)):
        raise ValueError("gradient or direction contains NaN/Inf")
    if not np.all(np.isfinite(scalars)) or step <= 0.0:
        raise ValueError("invalid centered AD-FD scalar")
    adjoint = float(np.sum(grad * vector))
    finite_difference = float((plus_current_A - minus_current_A) / (2.0 * step))
    scale = max(abs(adjoint), abs(finite_difference), np.finfo(float).tiny)
    relative_error = abs(adjoint - finite_difference) / scale
    midpoint = 0.5 * (plus_current_A + minus_current_A)
    signal = abs(plus_current_A - minus_current_A)
    return {
        "adjoint_directional_A_per_unit_rho": adjoint,
        "centered_FD_directional_A_per_unit_rho": finite_difference,
        "relative_error": relative_error,
        "same_nonzero_sign": bool(adjoint * finite_difference > 0.0),
        "plus_minus_current_difference_A": signal,
        "plus_minus_signal_relative_to_current": signal
        / max(abs(plus_current_A), abs(minus_current_A), np.finfo(float).tiny),
        "centered_midpoint_current_A": midpoint,
        "centered_midpoint_minus_baseline_A": midpoint - baseline_current_A,
        "centered_midpoint_curvature_over_linear_signal": abs(
            midpoint - baseline_current_A
        )
        / max(0.5 * signal, np.finfo(float).tiny),
    }
