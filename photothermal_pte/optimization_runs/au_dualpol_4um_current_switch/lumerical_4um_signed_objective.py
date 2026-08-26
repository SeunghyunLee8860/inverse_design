"""Signed Ea/Eb epigraph point for the Lumerical nodal design carrier."""

from __future__ import annotations

from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    LumericalNodalDesignMapping,
    OPTIMIZER_250NM_MAPPING,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.objective import (
    epigraph_constraints,
    useful_currents,
)


def _finite_projected_gradient(
    value: np.ndarray,
    *,
    label: str,
    mapping: LumericalNodalDesignMapping,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != mapping.shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite with shape {mapping.shape}")
    return result


def signed_dual_objective_point(
    *,
    latent: np.ndarray,
    beta: float,
    current_a_A: float,
    current_b_A: float,
    gradient_a_projected_A: np.ndarray,
    gradient_b_projected_A: np.ndarray,
    epigraph_A: float,
    mapping: LumericalNodalDesignMapping = OPTIMIZER_250NM_MAPPING,
) -> dict[str, Any]:
    """Pull both current gradients back and form exact epigraph constraints.

    The optimizer maximizes ``t`` subject to ``t-I_Ea <= 0`` and
    ``t+I_Eb <= 0``.  The returned constraint gradients are with respect to
    latent rho; their epigraph-coordinate derivatives are both exactly one.
    The balanced-minimum gradient is diagnostic only and is omitted at a tie,
    where the exact minimum is nondifferentiable but the epigraph remains
    well-defined.
    """

    latent_value = np.asarray(latent, dtype=np.float64)
    if latent_value.shape != mapping.shape or not np.all(np.isfinite(latent_value)):
        raise ValueError(f"latent must be finite with shape {mapping.shape}")
    if np.min(latent_value) < 0.0 or np.max(latent_value) > 1.0:
        raise ValueError("latent must lie inside [0,1]")
    scalars = np.asarray((current_a_A, current_b_A, epigraph_A), dtype=np.float64)
    if not np.all(np.isfinite(scalars)):
        raise ValueError("currents and epigraph must be finite")

    projected_a = _finite_projected_gradient(
        gradient_a_projected_A, label="Ea projected gradient", mapping=mapping
    )
    projected_b = _finite_projected_gradient(
        gradient_b_projected_A, label="Eb projected gradient", mapping=mapping
    )
    latent_a = mapping.vjp(latent_value, projected_a, beta)
    latent_b = mapping.vjp(latent_value, projected_b, beta)
    utility_a, utility_b = useful_currents(current_a_A, current_b_A)
    tie = bool(utility_a == utility_b)
    if tie:
        active = "tie"
        balanced_gradient = None
    elif utility_a < utility_b:
        active = "Ea"
        balanced_gradient = latent_a
    else:
        active = "Eb"
        balanced_gradient = -latent_b

    return {
        "current_a_A": float(current_a_A),
        "current_b_A": float(current_b_A),
        "utility_a_A": utility_a,
        "utility_b_A": utility_b,
        "balanced_utility_A": min(utility_a, utility_b),
        "active_polarization": active,
        "balanced_minimum_nondifferentiable_tie": tie,
        "epigraph_A": float(epigraph_A),
        "epigraph_constraints_A": epigraph_constraints(
            current_a_A, current_b_A, epigraph_A
        ),
        "gradient_a_latent_A": latent_a,
        "gradient_b_latent_A": latent_b,
        "constraint_gradients_latent_A": np.stack((-latent_a, latent_b)),
        "constraint_gradients_epigraph": np.ones(2, dtype=np.float64),
        "balanced_gradient_latent_A": balanced_gradient,
    }
