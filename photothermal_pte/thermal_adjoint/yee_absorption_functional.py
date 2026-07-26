"""Weighted absorbed-power functional on the native periodic Yee grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import epsilon_0


@dataclass(frozen=True)
class YeeAbsorptionEvaluation:
    weighted_value: float
    power_component_W: np.ndarray
    power_total_W: float
    density_component_W_m3: np.ndarray
    wirtinger_source: np.ndarray


def weighted_yee_absorption_and_wirtinger(
    *,
    electric_field_V_m: np.ndarray,
    frequency_Hz: float,
    epsilon_imaginary: np.ndarray,
    component_volume_m3: dict[int, np.ndarray],
    density_weight: np.ndarray | None = None,
) -> YeeAbsorptionEvaluation:
    """Return weighted absorption and ``q_E=dF/dE*``.

    ``density_weight[..., c]`` multiplies the native component absorption
    density before quadrature.  It is normally the exact pullback of the
    thermal adjoint through the conservative optical-to-thermal remap.
    Component Yee volumes own the optical quadrature exactly once.
    """
    field = np.asarray(electric_field_V_m, np.complex128)
    if field.ndim == 4 and field.shape[-1] == 3:
        field = field[..., None, :]
    if field.ndim != 5 or field.shape[-2:] != (1, 3):
        raise ValueError("electric field must have shape (nx,ny,nz,1,3)")
    frequency = float(frequency_Hz)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency must be positive")
    loss = np.asarray(epsilon_imaginary, float).reshape(-1)
    if loss.size != 3 or not np.all(np.isfinite(loss)) or np.any(loss < 0.0):
        raise ValueError("epsilon imaginary parts must be three nonnegative values")
    spatial_shape = field.shape[:3]
    if density_weight is None:
        weight = np.ones((*spatial_shape, 3), float)
    else:
        weight = np.asarray(density_weight, float)
        if weight.shape != (*spatial_shape, 3):
            raise ValueError(
                f"density weight {weight.shape} != {(*spatial_shape, 3)}"
            )
        if not np.all(np.isfinite(weight)):
            raise ValueError("density weight contains NaN or Inf")
    prefactor = 0.5 * (2.0 * np.pi * frequency) * epsilon_0
    density = prefactor * loss[None, None, None, :] * np.abs(field[..., 0, :]) ** 2
    q = np.zeros_like(field)
    component_power = np.zeros(3, float)
    weighted_value = 0.0
    for component in range(3):
        if component not in component_volume_m3:
            raise ValueError(f"missing Yee volume for component {component}")
        volume = np.asarray(component_volume_m3[component], float)
        if volume.shape != spatial_shape:
            raise ValueError(
                f"component-{component} volume {volume.shape} != {spatial_shape}"
            )
        if not np.all(np.isfinite(volume)) or np.any(volume < 0.0):
            raise ValueError("Yee volume must be finite and nonnegative")
        component_power[component] = float(
            np.sum(volume * density[..., component])
        )
        weighted_value += float(
            np.sum(volume * weight[..., component] * density[..., component])
        )
        q[..., 0, component] = (
            prefactor
            * loss[component]
            * volume
            * weight[..., component]
            * field[..., 0, component]
        )
    return YeeAbsorptionEvaluation(
        weighted_value=float(weighted_value),
        power_component_W=component_power,
        power_total_W=float(np.sum(component_power)),
        density_component_W_m3=density,
        wirtinger_source=q,
    )
