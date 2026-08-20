"""Frozen material definitions and differentiable Au/air interpolation laws.

The gold endpoint is the tabulated Ordal et al. value at exactly 10 um.  The
nonlinear law follows the refractive-index interpolation described by Zeng,
Venuthurumilli, and Xu for density-based plasmonic FDTD optimization:

    n(rho) = (1-rho) n_air + rho n_Au
    epsilon(rho) = n(rho)**2

Gray values are a numerical relaxation, not a physical effective medium.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


WAVELENGTH_M = 10.0e-6
TEMPERATURE_K = 300.0
N_AIR = 1.0 + 0.0j
N_AU_ORDAL_10UM = 12.1 + 69.2j
EPSILON_AIR = N_AIR**2
EPSILON_AU_ORDAL_10UM = N_AU_ORDAL_10UM**2

# Bulk reference values.  These are not promoted as exact thin-film values.
AU_BULK_ELECTRICAL_CONDUCTIVITY_S_M = 1.0 / (2.43e-8)
AU_BULK_THERMAL_CONDUCTIVITY_W_MK = 317.0
AU_BULK_SEEBECK_V_K = 1.94e-6
LORENZ_NUMBER_W_OHM_K2 = 2.44e-8


@dataclass(frozen=True)
class DensityPath:
    name: str
    epsilon: np.ndarray
    derivative: np.ndarray


def linear_epsilon_path(rho: np.ndarray) -> DensityPath:
    """Legacy diagnostic: direct complex-epsilon interpolation."""

    density = np.asarray(rho, dtype=np.float64)
    epsilon = EPSILON_AIR + density * (EPSILON_AU_ORDAL_10UM - EPSILON_AIR)
    derivative = np.full(
        density.shape,
        EPSILON_AU_ORDAL_10UM - EPSILON_AIR,
        dtype=np.complex128,
    )
    return DensityPath("linear_epsilon_legacy_diagnostic", epsilon, derivative)


def nonlinear_index_path(rho: np.ndarray) -> DensityPath:
    """Production candidate: interpolate complex n, then square it."""

    density = np.asarray(rho, dtype=np.float64)
    delta_n = N_AU_ORDAL_10UM - N_AIR
    index = N_AIR + density * delta_n
    epsilon = index**2
    derivative = 2.0 * index * delta_n
    return DensityPath("linear_n_then_square_candidate", epsilon, derivative)


def passive_index(epsilon: np.ndarray) -> np.ndarray:
    """Return the square-root branch with nonnegative imaginary index."""

    index = np.sqrt(np.asarray(epsilon, dtype=np.complex128))
    return np.where(index.imag < 0.0, -index, index)


def wiedemann_franz_thermal_conductivity(
    conductivity_s_m: float,
    temperature_K: float = TEMPERATURE_K,
) -> float:
    return float(LORENZ_NUMBER_W_OHM_K2 * conductivity_s_m * temperature_K)
