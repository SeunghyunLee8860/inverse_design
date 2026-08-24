"""Fail-closed candidate material normalization for fresh FDTDX model builds."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np


VERSION = "fdtdx-fresh-stable-two-pole-material-v1"
AXIS_MAP = {"b": "x", "a": "y", "c": "z"}
MATERIAL_NAMES = ("au", "a", "b", "c")


def realized_discrete_susceptibility(
    coefficient_triplet: tuple[float, float, float],
    omega_rad_s: float,
    dt_s: float,
) -> complex:
    """Evaluate one realized float32 ADE recurrence at the carrier."""

    c1, c2, c3 = (np.float32(value) for value in coefficient_triplet)
    theta = np.float32(omega_rad_s * dt_s)
    z_minus = np.exp(np.complex64(-1j * theta))
    z_plus = np.exp(np.complex64(1j * theta))
    return complex(np.complex64(c3) / (z_minus - c1 - c2 * z_plus))


def fdtdx_pole(fdtdx: Any, item: Mapping[str, Any]) -> Any:
    """Instantiate one physical pole without changing its declared kind."""

    if item.get("kind") == "Drude":
        return fdtdx.DrudePole(
            plasma_frequency=float(item["omega_p_rad_s"]),
            damping=float(item["gamma_rad_s"]),
        )
    if item.get("kind") == "Lorentz":
        return fdtdx.LorentzPole(
            resonance_frequency=float(item["omega_0_rad_s"]),
            damping=float(item["gamma_rad_s"]),
            delta_epsilon=float(item["delta_epsilon"]),
        )
    raise RuntimeError(f"unsupported candidate pole kind {item.get('kind')!r}")


def candidate_material_data(
    fdtdx: Any,
    material_law_contract: Mapping[str, Any],
    *,
    dt_s: float,
    omega_rad_s: float,
    epsilon_au: complex,
    epsilon_ta: Mapping[str, complex],
) -> dict[str, Any]:
    """Validate an opt-in law and return exact two-pole placement data."""

    law = dict(material_law_contract)
    unhashed_law = dict(law)
    supplied_contract_sha256 = unhashed_law.pop(
        "material_law_contract_sha256", None
    )
    computed_contract_sha256 = hashlib.sha256(
        json.dumps(
            unhashed_law,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    promotion = law.get("promotion")
    checks = law.get("checks")
    expected_targets = {"au": epsilon_au, **dict(epsilon_ta)}
    try:
        recorded_dt = float(
            law["case_binding"]["realized_float32_cfl"]["time_step_s"]
        )
        recorded_targets = law["material_binding"]["target_epsilon"]
        axis_map = law["material_binding"]["tairte4_crystal_to_solver_axis"]
        material_axes = law["material_axes"]
        contract_sha256 = str(law["material_law_contract_sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"invalid two-pole material-law structure: {error}") from error

    structural_checks = {
        "version_exact": law.get("version") == VERSION,
        "candidate_only": isinstance(promotion, Mapping)
        and promotion.get("candidate_only") is True,
        "optimizer_forbidden": isinstance(promotion, Mapping)
        and promotion.get("optimizer_start_allowed") is False,
        "contract_checks_passed": isinstance(checks, Mapping)
        and bool(checks)
        and all(value is True for value in checks.values()),
        "time_step_exact": recorded_dt == float(dt_s),
        "axis_map_exact": axis_map == AXIS_MAP,
        "material_axes_exact": isinstance(material_axes, Mapping)
        and set(material_axes) == set(MATERIAL_NAMES),
        "target_axes_exact": isinstance(recorded_targets, Mapping)
        and set(recorded_targets) == set(MATERIAL_NAMES),
        "target_values_exact": isinstance(recorded_targets, Mapping)
        and all(
            recorded_targets.get(name) == [value.real, value.imag]
            for name, value in expected_targets.items()
        ),
        "contract_sha256_exact": (
            supplied_contract_sha256 == computed_contract_sha256 == contract_sha256
        ),
    }
    if not all(structural_checks.values()):
        raise RuntimeError(
            "two-pole material-law contract does not match this model: "
            f"{structural_checks}"
        )

    from fdtdx.dispersion import compute_pole_coefficients_per_axis

    poles: dict[str, tuple[Any, ...]] = {}
    endpoints: dict[str, tuple[tuple[float, float, float], ...]] = {}
    fits: dict[str, Any] = {}
    susceptibilities: dict[str, complex] = {}
    for name in MATERIAL_NAMES:
        axis = material_axes[name]
        candidate = axis.get("candidate")
        if not isinstance(candidate, Mapping):
            raise RuntimeError(f"candidate block is absent for material axis {name!r}")
        pole_items = candidate.get("poles")
        if not isinstance(pole_items, list) or len(pole_items) != 2:
            raise RuntimeError(f"material axis {name!r} must contain exactly two poles")
        if candidate.get("found") is not True or candidate.get("fit_gate_passed") is not True:
            raise RuntimeError(f"material axis {name!r} did not pass its fit gate")
        if any(item.get("kind") != axis.get("pole_kind") for item in pole_items):
            raise RuntimeError(f"mixed or mismatched pole kind for material axis {name!r}")

        axis_poles = tuple(fdtdx_pole(fdtdx, item) for item in pole_items)
        c1, c2, c3, c4 = compute_pole_coefficients_per_axis(axis_poles, dt_s)
        expected = np.asarray(
            [[item["c1"], item["c2"], item["c3"]] for item in pole_items],
            dtype=np.float32,
        )
        generated_by_component = tuple(
            np.stack(
                (c1[:, component], c2[:, component], c3[:, component]), axis=1
            ).astype(np.float32)
            for component in range(3)
        )
        exact = (
            expected.shape == (2, 3)
            and np.all(np.isfinite(expected))
            and all(np.array_equal(value, expected) for value in generated_by_component)
            and np.all(np.asarray(c4, dtype=np.float32) == 0.0)
        )
        if not exact:
            raise RuntimeError(
                f"pinned FDTDX coefficients do not reproduce material axis {name!r}"
            )

        endpoint_rows = tuple(
            tuple(float(value) for value in row) for row in expected
        )
        poles[name] = axis_poles
        endpoints[name] = endpoint_rows
        fits[name] = {
            "kind": "candidate-two-pole",
            "fit_basis": (
                "contract-pinned realized-float32 one-frequency ADE response"
            ),
            "fit_relative_error": float(candidate["fit_relative_error"]),
            "poles": [dict(item) for item in pole_items],
        }
        susceptibilities[name] = sum(
            (
                realized_discrete_susceptibility(row, omega_rad_s, dt_s)
                for row in endpoint_rows
            ),
            0.0j,
        )

    return {
        "poles": poles,
        "coefficient_endpoints": endpoints,
        "fits": fits,
        "discrete_susceptibility": susceptibilities,
        "contract_sha256": contract_sha256,
        "structural_checks": structural_checks,
    }
