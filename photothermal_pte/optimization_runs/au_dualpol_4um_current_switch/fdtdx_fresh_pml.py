"""Explicit per-face 4-um CPML parameters for the fresh FDTDX route."""

from __future__ import annotations

import math
from typing import Any

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
    pml_parameters,
)


PML_FACES = ("minx", "maxx", "miny", "maxy", "minz", "maxz")
SOLVER_PARAMETER_NAMES = (
    "alpha_start",
    "alpha_end",
    "alpha_order",
    "kappa_start",
    "kappa_end",
    "kappa_order",
    "sigma_start",
    "sigma_end",
    "sigma_order",
)


def face_parameters(
    spec: MeshSpec,
    *,
    alpha_scale: float = 1.0,
    target_reflection: float = 1.0e-6,
) -> dict[str, dict[str, float]]:
    """Return complete CPML parameters with face-specific physical thickness."""

    return {
        face: pml_parameters(
            (
                spec.lateral_pml_thickness_m
                if face in ("minx", "maxx", "miny", "maxy")
                else spec.z_pml_thickness_m
            ),
            alpha_scale=alpha_scale,
            target_reflection=target_reflection,
        )
        for face in PML_FACES
    }


def solver_parameters(
    profiles: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Strip audit metadata and reject incomplete/non-finite solver profiles."""

    if set(profiles) != set(PML_FACES):
        raise ValueError(f"PML profiles must contain exactly {PML_FACES}")
    result: dict[str, dict[str, float]] = {}
    for face in PML_FACES:
        profile = profiles[face]
        missing = set(SOLVER_PARAMETER_NAMES) - set(profile)
        if missing:
            raise ValueError(f"PML face {face} is missing {sorted(missing)}")
        values = {name: float(profile[name]) for name in SOLVER_PARAMETER_NAMES}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError(f"PML face {face} has non-finite parameters")
        if values["sigma_end"] <= 0.0 or values["alpha_start"] < 0.0:
            raise ValueError(f"PML face {face} has invalid loss parameters")
        result[face] = values
    return result


def boundary_config_kwargs(
    profiles: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Expand per-face profiles to FDTDX BoundaryConfig keyword names."""

    clean = solver_parameters(profiles)
    return {
        f"{parameter}_{face}": value
        for face, profile in clean.items()
        for parameter, value in profile.items()
    }
