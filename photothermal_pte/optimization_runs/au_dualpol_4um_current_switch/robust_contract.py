"""Fail-closed scenario contract for robust dual-polarization optimization."""

from __future__ import annotations

import numpy as np


ROBUST_ETAS = (0.35, 0.50, 0.65)
POLARIZATIONS = ("Ea", "Eb")


def eta_key(eta: float) -> str:
    return f"eta_{eta:.2f}"


def scenario_key(eta: float, polarization: str) -> str:
    if polarization not in POLARIZATIONS:
        raise ValueError(f"unknown polarization {polarization!r}")
    return f"{eta_key(eta)}_{polarization}"


def current_constraint_keys() -> tuple[str, ...]:
    return tuple(
        scenario_key(eta, polarization)
        for eta in ROBUST_ETAS
        for polarization in POLARIZATIONS
    )


def gray_constraint_keys() -> tuple[str, ...]:
    return tuple(f"{eta_key(eta)}_grayness" for eta in ROBUST_ETAS)


def constraint_labels() -> tuple[str, ...]:
    return current_constraint_keys() + gray_constraint_keys()


def grayness(rho: np.ndarray) -> float:
    density = np.asarray(rho, dtype=np.float64)
    return float(np.mean(4.0 * density * (1.0 - density)))


def grayness_cotangent(rho: np.ndarray) -> np.ndarray:
    density = np.asarray(rho, dtype=np.float64)
    return 4.0 * (1.0 - 2.0 * density) / density.size


def audit() -> dict[str, object]:
    return {
        "etas": list(ROBUST_ETAS),
        "polarizations": list(POLARIZATIONS),
        "current_constraints": list(current_constraint_keys()),
        "grayness_constraints": list(gray_constraint_keys()),
        "constraint_count": len(constraint_labels()),
        "nominal_eta_in_objective": 0.50 in ROBUST_ETAS,
        "grayness_constrained_for_every_eta": True,
    }
