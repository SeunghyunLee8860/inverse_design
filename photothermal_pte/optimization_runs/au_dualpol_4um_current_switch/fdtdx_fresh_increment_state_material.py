"""Physical one-pole materials for the opt-in FDTDX increment-state path.

This adapter deliberately keeps physical pole parameters independent of the
mesh.  The patched FDTDX fork generates cancellation-resistant ``A/C/B``
coefficients from those poles for the realized time step.  It is not a gray
material interpolation law and does not authorize optimization.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_increment_state_precision import (
    physical_pole_from_target,
)


VERSION = "fdtdx-fresh-physical-one-pole-increment-state-v1"
MATERIAL_NAMES = ("au", "a", "b", "c")
DISCRETE_EPSILON_RELATIVE_ERROR_LIMIT = 1.0e-4


def _fdtdx_pole(fdtdx: Any, pole: Mapping[str, Any]) -> Any:
    kind = pole["kind"]
    coupling_sq = float(pole["coupling_sq_rad_s2"])
    if kind == "Drude":
        return fdtdx.DrudePole(
            plasma_frequency=math.sqrt(coupling_sq),
            damping=float(pole["gamma_rad_s"]),
        )
    if kind == "Lorentz":
        omega_0 = float(pole["omega_0_rad_s"])
        return fdtdx.LorentzPole(
            resonance_frequency=omega_0,
            damping=float(pole["gamma_rad_s"]),
            delta_epsilon=coupling_sq / omega_0**2,
        )
    raise RuntimeError(f"unsupported physical pole kind {kind!r}")


def physical_increment_material_data(
    fdtdx: Any,
    *,
    dt_s: float,
    omega_rad_s: float,
    wavelength_m: float,
    epsilon_au: complex,
    epsilon_ta: Mapping[str, complex],
) -> dict[str, Any]:
    """Return locked physical poles and realized float32 ``A/C/B`` endpoints."""

    if dt_s <= 0.0 or omega_rad_s <= 0.0 or wavelength_m <= 0.0:
        raise ValueError("dt, omega, and wavelength must be positive")
    targets = {"au": complex(epsilon_au), **dict(epsilon_ta)}
    if set(targets) != set(MATERIAL_NAMES):
        raise ValueError("material targets must contain exactly au/a/b/c")

    from fdtdx.increment_state import (
        compute_increment_state_coefficients_per_axis,
        susceptibility_from_increment_coefficients,
    )

    poles: dict[str, tuple[Any, ...]] = {}
    endpoints: dict[str, tuple[tuple[float, float, float], ...]] = {}
    fits: dict[str, Any] = {}
    susceptibilities: dict[str, complex] = {}
    for name in MATERIAL_NAMES:
        target = targets[name]
        physical = physical_pole_from_target(
            name,
            target,
            wavelength_m=wavelength_m,
        )
        fdtdx_pole = _fdtdx_pole(fdtdx, physical)
        coeff_a, coeff_c, coeff_b = compute_increment_state_coefficients_per_axis(
            (fdtdx_pole,), dt_s
        )
        coefficient_columns = tuple(
            np.asarray(value, dtype=np.float32)
            for value in (coeff_a, coeff_c, coeff_b)
        )
        if any(value.shape != (1, 3) for value in coefficient_columns):
            raise RuntimeError(f"unexpected increment coefficients for {name!r}")
        if any(not np.all(value == value[:, :1]) for value in coefficient_columns):
            raise RuntimeError(f"physical pole for {name!r} is not isotropic")
        endpoint = tuple(float(value[0, 0]) for value in coefficient_columns)
        realized = complex(
            np.asarray(
                susceptibility_from_increment_coefficients(
                    *(value[:, 0] for value in coefficient_columns),
                    omega=omega_rad_s,
                    dt=dt_s,
                )
            )
        )
        realized_epsilon = 1.0 + realized
        relative_error = abs(realized_epsilon - target) / abs(target)
        dynamic_stable = (
            -1.0 < endpoint[0] < 1.0
            if physical["kind"] == "Drude"
            else float(physical["omega_0_rad_s"]) * dt_s < 2.0
        )
        checks = {
            "physical_pole_passive_at_carrier": physical["passive_at_carrier"] is True,
            "continuum_endpoint_exact": physical["continuum_target_relative_error"]
            <= np.finfo(np.float64).eps * 16.0,
            "increment_state_dynamic_stable": bool(dynamic_stable),
            "realized_float32_epsilon_passive": realized_epsilon.imag > 0.0,
            "realized_float32_epsilon_matches_target": relative_error
            <= DISCRETE_EPSILON_RELATIVE_ERROR_LIMIT,
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"physical increment-state material gate failed for {name!r}: {checks}"
            )
        poles[name] = (fdtdx_pole,)
        endpoints[name] = (endpoint,)
        susceptibilities[name] = realized
        fits[name] = {
            "version": VERSION,
            "kind": physical["kind"],
            "fit_basis": "mesh-independent passive physical pole; FDTDX-generated float32 A/C/B",
            "fit_relative_error": float(relative_error),
            "physical_pole": physical,
            "increment_coefficients": {
                "A_float32": endpoint[0],
                "C_float32": endpoint[1],
                "B_float32": endpoint[2],
            },
            "checks": checks,
        }

    return {
        "version": VERSION,
        "poles": poles,
        "coefficient_endpoints": endpoints,
        "fits": fits,
        "discrete_susceptibility": susceptibilities,
        "optimizer_start_allowed": False,
        "gray_material_law_defined": False,
    }
