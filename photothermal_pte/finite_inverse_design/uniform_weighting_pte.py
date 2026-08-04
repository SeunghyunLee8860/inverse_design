"""Finite-flake PTE functional for a uniform 45-degree weighting field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from photothermal_pte.thermal_adjoint.local_pte_functional import (
    LocalPTEFunctional,
    build_local_pte_functional,
)

from .contract import (
    WEIGHTING_FIELD_X_M_INV,
    WEIGHTING_FIELD_Y_M_INV,
)


@dataclass(frozen=True)
class UniformWeightingPTE:
    base: LocalPTEFunctional
    temperature_source_A_K: np.ndarray
    weighting_x_m_inv: float
    weighting_y_m_inv: float

    def evaluate_active(self, temperature_active_K: np.ndarray) -> float:
        temperature = np.asarray(temperature_active_K, float).reshape(-1)
        return float(np.dot(self.temperature_source_A_K, temperature))


def build_uniform_45deg_weighting_pte(
    *,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    z_edges_m: np.ndarray,
    active_mask: np.ndarray,
    active_ids: np.ndarray,
    flake_mask: np.ndarray,
) -> UniformWeightingPTE:
    base = build_local_pte_functional(
        x_edges_m=x_edges_m,
        y_edges_m=y_edges_m,
        z_edges_m=z_edges_m,
        active_mask=active_mask,
        active_ids=active_ids,
        fom_mask=flake_mask,
        periodic_axes=(),
    )
    dx_term = (
        base.sigma_a_S_m
        * base.seebeck_a_V_K
        * (base.derivative_x_m_inv.T @ base.integration_volume_m3)
    )
    dy_term = (
        base.sigma_b_S_m
        * base.seebeck_b_V_K
        * (base.derivative_y_m_inv.T @ base.integration_volume_m3)
    )
    source = -(
        WEIGHTING_FIELD_X_M_INV * dx_term
        + WEIGHTING_FIELD_Y_M_INV * dy_term
    )
    return UniformWeightingPTE(
        base=base,
        temperature_source_A_K=np.asarray(source, float).reshape(-1),
        weighting_x_m_inv=WEIGHTING_FIELD_X_M_INV,
        weighting_y_m_inv=WEIGHTING_FIELD_Y_M_INV,
    )
