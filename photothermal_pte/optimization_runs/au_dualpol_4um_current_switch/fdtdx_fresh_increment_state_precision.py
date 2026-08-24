#!/usr/bin/env python3
"""CPU gate for a cancellation-resistant dispersive state representation.

FDTDX currently advances a Lorentz/Drude polarization with the second-order
recurrence

    P[n+1] = c1 P[n] + c2 P[n-1] + c3 E[n].

On the fine 4-um z meshes, ``c1`` and ``c2`` are both close to +/-1 and the
carrier denominator is assembled by subtracting nearly equal float32 values.
This module evaluates an algebraically equivalent state representation,

    V[n+1] = A V[n] - C P[n] + B E[n]
    P[n+1] = P[n] + V[n+1],

where ``V[n] = P[n] - P[n-1]``.  The small resonance coefficient ``C`` is
stored directly instead of being lost in ``c1 = 1 + A - C``.  For a Drude
pole, ``C = 0`` and only the damped increment/current state feeds Maxwell's
electric-field update.

The diagnostic is solver-free and CPU-only.  It does not patch FDTDX, run a
field solve, certify a mesh, validate an adjoint, or authorize optimization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_ade_precision_diagnostic import (
    C0_M_PER_S,
    MATERIAL_CONTRACT,
    WAVELENGTH_M,
    load_material_epsilon,
    realized_float32_cfl,
)


VERSION = "fdtdx-fresh-increment-state-precision-v1"
HERE = Path(__file__).resolve().parent
ORDAL_TABLE = (
    HERE.parent
    / "au_on_fixed_tairte4_validation/data/au_ordal_1987_nk.csv"
)
MATERIAL_AXES = ("au", "a", "b", "c")
NUM_WINDOWS = 4
DEFAULT_TOTAL_PERIODS = 32
DEFAULT_STARTUP_PERIODS = 4
DEFAULT_WINDOW_PERIODS = 4
LORENTZ_RESONANCE_RATIO = 2.0

# These gates are tighter than the full-field stationarity gate.  The carrier
# tolerance includes float32 coefficient quantization of one fixed physical
# pole; no per-mesh material parameter refit is performed.
CARRIER_RELATIVE_ERROR_LIMIT = 1.0e-4
FLOAT64_WINDOW_CHANGE_LIMIT = 1.0e-5
FLOAT32_WINDOW_CHANGE_LIMIT = 5.0e-3
FLOAT32_VS_FLOAT64_LIMIT = 5.0e-3
AU_ORDAL_2_TO_8_MAX_RELATIVE_ERROR_LIMIT = 4.0e-2
AU_ORDAL_2_TO_8_RMS_RELATIVE_ERROR_LIMIT = 2.0e-2
AU_ORDAL_3_TO_6_MAX_RELATIVE_ERROR_LIMIT = 2.0e-2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_payload_sha256(payload: Mapping[str, Any], hash_key: str) -> str:
    unhashed = dict(payload)
    unhashed.pop(hash_key, None)
    encoded = json.dumps(
        unhashed,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def _relative_change(current: complex, previous: complex) -> float:
    return float(
        abs(current - previous)
        / max(abs(current), np.finfo(np.float64).tiny)
    )


def physical_pole_from_target(
    material_axis: str,
    target_epsilon: complex,
    *,
    wavelength_m: float = WAVELENGTH_M,
) -> dict[str, Any]:
    """Return one mesh-independent passive pole matching the 4-um endpoint."""

    if material_axis not in MATERIAL_AXES:
        raise ValueError(f"unknown material axis {material_axis!r}")
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive")
    target = complex(target_epsilon)
    susceptibility = target - 1.0
    if susceptibility.imag <= 0.0:
        raise ValueError("target material must be passive at the carrier")

    omega = 2.0 * math.pi * C0_M_PER_S / wavelength_m
    if susceptibility.real < 0.0:
        kind = "Drude"
        omega_0 = 0.0
        gamma = omega * susceptibility.imag / (-susceptibility.real)
        coupling_sq = (-susceptibility.real) * (
            omega**2 + gamma**2
        )
    elif susceptibility.real > 0.0:
        kind = "Lorentz"
        omega_0 = LORENTZ_RESONANCE_RATIO * omega
        detuning = omega_0**2 - omega**2
        gamma = (
            susceptibility.imag
            / susceptibility.real
            * detuning
            / omega
        )
        coupling_sq = (
            susceptibility.real
            * (detuning**2 + (gamma * omega) ** 2)
            / detuning
        )
    else:
        raise ValueError("purely imaginary target susceptibility is unsupported")

    continuum = coupling_sq / (
        omega_0**2 - omega**2 - 1j * gamma * omega
    )
    relative_error = abs(continuum - susceptibility) / abs(susceptibility)
    return {
        "material_axis": material_axis,
        "kind": kind,
        "epsilon_infinity": 1.0,
        "omega_0_rad_s": omega_0,
        "gamma_rad_s": gamma,
        "coupling_sq_rad_s2": coupling_sq,
        "target_epsilon": _complex_pair(target),
        "target_susceptibility": _complex_pair(susceptibility),
        "continuum_susceptibility": _complex_pair(continuum),
        "continuum_target_relative_error": float(relative_error),
        "passive_at_carrier": bool(
            gamma > 0.0
            and coupling_sq > 0.0
            and continuum.imag > 0.0
        ),
    }


def increment_state_coefficients(
    pole: Mapping[str, Any],
    dt_s: float,
) -> dict[str, Any]:
    """Build locked float32 ``A, C, B`` coefficients for ``(P, delta-P)``."""

    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    gamma = float(pole["gamma_rad_s"])
    omega_0 = float(pole["omega_0_rad_s"])
    coupling_sq = float(pole["coupling_sq_rad_s2"])
    denominator = 1.0 + 0.5 * gamma * dt_s
    a = np.float32((1.0 - 0.5 * gamma * dt_s) / denominator)
    c = np.float32(omega_0**2 * dt_s**2 / denominator)
    b = np.float32(coupling_sq * dt_s**2 / denominator)
    matrix = np.asarray(
        [[1.0 - float(c), float(a)], [-float(c), float(a)]],
        dtype=np.float64,
    )
    root_radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    drude_current_stable = (
        pole["kind"] == "Drude"
        and c == 0.0
        and -1.0 < float(a) < 1.0
    )
    lorentz_state_stable = (
        pole["kind"] == "Lorentz" and root_radius < 1.0
    )
    return {
        "A_float32": float(a),
        "C_float32": float(c),
        "B_float32": float(b),
        "state_matrix_root_radius": root_radius,
        "drude_unit_P_root_is_decoupled_from_Maxwell": bool(
            pole["kind"] == "Drude" and c == 0.0
        ),
        "dynamic_state_stable": bool(
            drude_current_stable or lorentz_state_stable
        ),
    }


def discrete_susceptibility(
    coefficients: Mapping[str, Any],
    omega_rad_s: float,
    dt_s: float,
) -> complex:
    """Exact carrier response of the locked increment-state recurrence."""

    a = float(coefficients["A_float32"])
    c = float(coefficients["C_float32"])
    b = float(coefficients["B_float32"])
    z_minus = np.exp(-1j * omega_rad_s * dt_s)
    denominator = (
        (z_minus - 1.0) * (z_minus - a) + c * z_minus
    )
    return complex(b * z_minus / denominator)


def second_order_equivalent_susceptibility(
    coefficients: Mapping[str, Any],
    omega_rad_s: float,
    dt_s: float,
) -> complex:
    """Evaluate the same recurrence after symbolic elimination of ``V``."""

    a = float(coefficients["A_float32"])
    c = float(coefficients["C_float32"])
    b = float(coefficients["B_float32"])
    c1 = 1.0 + a - c
    c2 = -a
    z_minus = np.exp(-1j * omega_rad_s * dt_s)
    z_plus = np.exp(1j * omega_rad_s * dt_s)
    return complex(b / (z_minus - c1 - c2 * z_plus))


def _simulate_response(
    drive: np.ndarray,
    coefficients: Mapping[str, Any],
    dtype: type[np.float32] | type[np.float64],
) -> np.ndarray:
    """Advance the two states with explicit dtype rounding at every operation."""

    a = dtype(coefficients["A_float32"])
    c = dtype(coefficients["C_float32"])
    b = dtype(coefficients["B_float32"])
    polarization = dtype(0.0)
    increment = dtype(0.0)
    output = np.empty(drive.size, dtype=dtype)
    for index, field in enumerate(drive):
        increment = dtype(
            dtype(a * increment)
            - dtype(c * polarization)
            + dtype(b * dtype(field))
        )
        polarization = dtype(polarization + increment)
        output[index] = polarization
    return output


def simulate_axis(
    pole: Mapping[str, Any],
    coefficients: Mapping[str, Any],
    *,
    dt_s: float,
    wavelength_m: float = WAVELENGTH_M,
    total_periods: int = DEFAULT_TOTAL_PERIODS,
    startup_periods: int = DEFAULT_STARTUP_PERIODS,
    window_periods: int = DEFAULT_WINDOW_PERIODS,
) -> dict[str, Any]:
    """Compare locked-coefficient float32 and float64 state updates."""

    if dt_s <= 0.0 or wavelength_m <= 0.0:
        raise ValueError("dt_s and wavelength_m must be positive")
    if startup_periods <= 0 or window_periods <= 0:
        raise ValueError("startup_periods and window_periods must be positive")
    if total_periods < NUM_WINDOWS * window_periods:
        raise ValueError("total_periods must contain four complete late windows")

    period_s = wavelength_m / C0_M_PER_S
    omega = 2.0 * math.pi / period_s
    time_steps_total = int(math.ceil(total_periods * period_s / dt_s))
    time_s = np.arange(time_steps_total, dtype=np.float64) * dt_s
    ramp = np.clip(time_s / (startup_periods * period_s), 0.0, 1.0)
    drive = np.asarray(ramp * np.cos(omega * time_s), dtype=np.float32)
    phase = np.exp(1j * omega * time_s)

    first_period = total_periods - NUM_WINDOWS * window_periods
    window_bounds = [
        (
            first_period + index * window_periods,
            first_period + (index + 1) * window_periods,
        )
        for index in range(NUM_WINDOWS)
    ]
    index_bounds = [
        (
            int(np.searchsorted(time_s, lower * period_s)),
            int(np.searchsorted(time_s, upper * period_s)),
        )
        for lower, upper in window_bounds
    ]
    drive_phasors = [
        np.sum(drive[lower:upper] * phase[lower:upper], dtype=np.complex128)
        for lower, upper in index_bounds
    ]
    if any(abs(value) <= np.finfo(float).tiny for value in drive_phasors):
        raise RuntimeError("late-window drive phasor is zero")

    precision: dict[str, Any] = {}
    for name, dtype in (("float32", np.float32), ("float64", np.float64)):
        response = _simulate_response(drive, coefficients, dtype)
        susceptibility = [
            np.sum(
                response[lower:upper] * phase[lower:upper],
                dtype=np.complex128,
            )
            / drive_phasors[index]
            for index, (lower, upper) in enumerate(index_bounds)
        ]
        changes = [
            _relative_change(current, previous)
            for previous, current in zip(
                susceptibility[:-1], susceptibility[1:], strict=True
            )
        ]
        precision[name] = {
            "susceptibility_windows": [
                _complex_pair(value) for value in susceptibility
            ],
            "relative_window_changes": changes,
            "last_relative_window_change": changes[-1],
            "maximum_relative_window_change": max(changes),
        }

    float32_late = complex(*precision["float32"]["susceptibility_windows"][-1])
    float64_late = complex(*precision["float64"]["susceptibility_windows"][-1])
    disagreement = _relative_change(float64_late, float32_late)
    target = complex(*pole["target_susceptibility"])
    discrete = discrete_susceptibility(coefficients, omega, dt_s)
    second_order = second_order_equivalent_susceptibility(
        coefficients, omega, dt_s
    )
    carrier_error = abs(discrete - target) / abs(target)
    equivalence_error = abs(discrete - second_order) / max(
        abs(discrete), np.finfo(float).tiny
    )
    gates = {
        "continuum_target_exact": (
            float(pole["continuum_target_relative_error"]) <= 1.0e-12
        ),
        "pole_passive_at_carrier": bool(pole["passive_at_carrier"]),
        "dynamic_state_stable": bool(coefficients["dynamic_state_stable"]),
        "increment_and_second_order_algebraically_equivalent": (
            equivalence_error <= 1.0e-8
        ),
        "float32_discrete_carrier_fit": (
            carrier_error <= CARRIER_RELATIVE_ERROR_LIMIT
        ),
        "float64_reference_last_window_settled": (
            precision["float64"]["last_relative_window_change"]
            <= FLOAT64_WINDOW_CHANGE_LIMIT
        ),
        "float32_last_window_settled": (
            precision["float32"]["last_relative_window_change"]
            <= FLOAT32_WINDOW_CHANGE_LIMIT
        ),
        "float32_matches_float64_late_response": (
            disagreement <= FLOAT32_VS_FLOAT64_LIMIT
        ),
    }
    return {
        "time_steps_total": time_steps_total,
        "window_bounds_periods": [list(value) for value in window_bounds],
        "window_sample_counts": [
            upper - lower for lower, upper in index_bounds
        ],
        "discrete_susceptibility": _complex_pair(discrete),
        "second_order_equivalent_susceptibility": _complex_pair(second_order),
        "increment_second_order_relative_difference": float(equivalence_error),
        "discrete_carrier_relative_error": float(carrier_error),
        "precision": precision,
        "float32_vs_float64_late_relative_difference": disagreement,
        "gates": gates,
        "ready": all(gates.values()),
    }


def audit_au_ordal_band(pole: Mapping[str, Any]) -> dict[str, Any]:
    """Check that the fixed Au pole remains a reasonable IR Drude model."""

    if pole["material_axis"] != "au" or pole["kind"] != "Drude":
        raise ValueError("Ordal audit requires the Au Drude pole")
    if not ORDAL_TABLE.is_file():
        raise RuntimeError(f"missing Ordal table {ORDAL_TABLE}")
    rows = []
    with ORDAL_TABLE.open(newline="", encoding="utf-8") as stream:
        for item in csv.DictReader(stream):
            wavelength_um = float(item["wavelength_um"])
            measured = complex(float(item["n"]), float(item["k"])) ** 2
            omega = 2.0 * math.pi * C0_M_PER_S / (wavelength_um * 1.0e-6)
            predicted = 1.0 + float(pole["coupling_sq_rad_s2"]) / (
                float(pole["omega_0_rad_s"]) ** 2
                - omega**2
                - 1j * float(pole["gamma_rad_s"]) * omega
            )
            relative_error = abs(predicted - measured) / abs(measured)
            rows.append(
                {
                    "wavelength_um": wavelength_um,
                    "measured_epsilon": _complex_pair(measured),
                    "predicted_epsilon": _complex_pair(predicted),
                    "relative_error": float(relative_error),
                }
            )

    def summarize(lower: float, upper: float) -> dict[str, float | int]:
        selected = [
            row["relative_error"]
            for row in rows
            if lower <= row["wavelength_um"] <= upper
        ]
        if not selected:
            raise RuntimeError(f"Ordal table has no rows in {lower:g}-{upper:g} um")
        return {
            "sample_count": len(selected),
            "maximum_relative_error": float(max(selected)),
            "rms_relative_error": float(
                math.sqrt(np.mean(np.square(selected, dtype=np.float64)))
            ),
        }

    two_to_eight = summarize(2.0, 8.0)
    three_to_six = summarize(3.0, 6.0)
    gates = {
        "ordal_table_contains_exact_4um_row": any(
            row["wavelength_um"] == 4.0
            and row["relative_error"] <= 1.0e-12
            for row in rows
        ),
        "two_to_eight_um_max_error_below_limit": (
            two_to_eight["maximum_relative_error"]
            <= AU_ORDAL_2_TO_8_MAX_RELATIVE_ERROR_LIMIT
        ),
        "two_to_eight_um_rms_error_below_limit": (
            two_to_eight["rms_relative_error"]
            <= AU_ORDAL_2_TO_8_RMS_RELATIVE_ERROR_LIMIT
        ),
        "three_to_six_um_max_error_below_limit": (
            three_to_six["maximum_relative_error"]
            <= AU_ORDAL_3_TO_6_MAX_RELATIVE_ERROR_LIMIT
        ),
    }
    return {
        "source": str(ORDAL_TABLE.resolve()),
        "source_sha256": file_sha256(ORDAL_TABLE),
        "scope_note": (
            "diagnostic agreement of a 4-um-anchored single Drude pole; not a "
            "replacement for a certified multi-pole sampled-data fit"
        ),
        "two_to_eight_um": two_to_eight,
        "three_to_six_um": three_to_six,
        "gates": gates,
        "ready": all(gates.values()),
    }


def build_report(
    *,
    z_factors: Sequence[int] = (8, 16, 32),
    total_periods: int = DEFAULT_TOTAL_PERIODS,
    startup_periods: int = DEFAULT_STARTUP_PERIODS,
    window_periods: int = DEFAULT_WINDOW_PERIODS,
) -> dict[str, Any]:
    material_epsilon = load_material_epsilon()
    if tuple(material_epsilon) != MATERIAL_AXES:
        raise RuntimeError("material-axis order or membership changed")
    poles = {
        name: physical_pole_from_target(name, material_epsilon[name])
        for name in MATERIAL_AXES
    }
    levels: dict[str, Any] = {}
    for z_factor in z_factors:
        cfl = realized_float32_cfl(int(z_factor))
        dt_s = float(cfl["time_step_s"])
        axes = {}
        for name in MATERIAL_AXES:
            coefficients = increment_state_coefficients(poles[name], dt_s)
            transient = simulate_axis(
                poles[name],
                coefficients,
                dt_s=dt_s,
                total_periods=total_periods,
                startup_periods=startup_periods,
                window_periods=window_periods,
            )
            axes[name] = {
                "pole": poles[name],
                "coefficients": coefficients,
                "transient": transient,
                "ready": transient["ready"],
            }
        levels[str(int(z_factor))] = {
            "cfl": cfl,
            "material_axes": axes,
            "ready": all(item["ready"] for item in axes.values()),
        }

    au_band = audit_au_ordal_band(poles["au"])
    gates = {
        "requested_z_factors_are_unique_positive": (
            len(set(int(value) for value in z_factors)) == len(z_factors)
            and all(int(value) > 0 for value in z_factors)
        ),
        "all_material_axes_pass_every_level": all(
            level["ready"] for level in levels.values()
        ),
        "au_ordal_ir_sanity_passes": au_band["ready"],
        "optimizer_remains_forbidden": True,
    }
    ready = all(gates.values())
    report: dict[str, Any] = {
        "version": VERSION,
        "status": (
            "VALIDATED_CPU_INCREMENT_STATE_PRECISION_CANDIDATE"
            if ready
            else "BLOCKED_CPU_INCREMENT_STATE_PRECISION_CANDIDATE"
        ),
        "ready": ready,
        "scope": (
            "solver-free CPU scalar recurrence; no FDTDX patch, 3-D field "
            "solve, mesh certificate, adjoint, PTE, or optimization"
        ),
        "state_equations": {
            "increment": "V[n+1] = A*V[n] - C*P[n] + B*E[n]",
            "polarization": "P[n+1] = P[n] + V[n+1]",
            "maxwell_correction": "E[n+1] += -inv_eps*sum(V[n+1])",
            "coefficient_map": (
                "A=(1-gamma*dt/2)/(1+gamma*dt/2), "
                "C=omega0^2*dt^2/(1+gamma*dt/2), "
                "B=K*dt^2/(1+gamma*dt/2)"
            ),
            "second_order_identity": "c1=1+A-C, c2=-A, c3=B",
        },
        "physical_material_policy": {
            "mesh_independent_poles": True,
            "negative_real_susceptibility": (
                "one passive Drude pole fixed by the 4-um complex endpoint"
            ),
            "positive_real_susceptibility": (
                "one passive Lorentz pole with fixed omega0/omega=2, then "
                "gamma and strength fixed by the 4-um complex endpoint"
            ),
            "no_ccpr_one_point_active_fit": True,
            "no_per_mesh_physical_pole_refit": True,
            "no_gray_material_law_claim": True,
        },
        "thresholds": {
            "float32_discrete_carrier_relative_error": (
                CARRIER_RELATIVE_ERROR_LIMIT
            ),
            "float64_last_window_relative_change": (
                FLOAT64_WINDOW_CHANGE_LIMIT
            ),
            "float32_last_window_relative_change": (
                FLOAT32_WINDOW_CHANGE_LIMIT
            ),
            "float32_vs_float64_late_relative_difference": (
                FLOAT32_VS_FLOAT64_LIMIT
            ),
            "au_ordal_2_to_8_max_relative_error": (
                AU_ORDAL_2_TO_8_MAX_RELATIVE_ERROR_LIMIT
            ),
            "au_ordal_2_to_8_rms_relative_error": (
                AU_ORDAL_2_TO_8_RMS_RELATIVE_ERROR_LIMIT
            ),
            "au_ordal_3_to_6_max_relative_error": (
                AU_ORDAL_3_TO_6_MAX_RELATIVE_ERROR_LIMIT
            ),
        },
        "time_spec": {
            "total_periods": total_periods,
            "source_startup_periods": startup_periods,
            "window_periods": window_periods,
            "late_window_count": NUM_WINDOWS,
        },
        "au_ordal_band_sanity": au_band,
        "levels": levels,
        "gates": gates,
        "promotion": {
            "candidate_only": True,
            "is_fdtdx_implementation_certificate": False,
            "is_material_certificate": False,
            "is_mesh_certificate": False,
            "is_adjoint_certificate": False,
            "optimizer_start_allowed": False,
            "next_allowed_step": (
                "implement the same state equations in an isolated FDTDX fork; "
                "pass small forward and checkpointed AD-FD controls before any "
                "long or fine 3-D solve"
            ),
        },
        "provenance": {
            "diagnostic_script_sha256": file_sha256(Path(__file__).resolve()),
            "material_contract": str(MATERIAL_CONTRACT.resolve()),
            "material_contract_sha256": file_sha256(MATERIAL_CONTRACT),
            "ordal_table": str(ORDAL_TABLE.resolve()),
            "ordal_table_sha256": file_sha256(ORDAL_TABLE),
            "cpu_rounding_note": (
                "explicit NumPy scalar rounding after each state operation; a "
                "separate FDTDX/JAX kernel gate remains mandatory"
            ),
        },
    }
    report["increment_state_report_sha256"] = canonical_payload_sha256(
        report, "increment_state_report_sha256"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--z-factors", type=int, nargs="+", default=[8, 16, 32]
    )
    args = parser.parse_args()
    output = args.output.expanduser()
    if not output.is_absolute():
        parser.error("--output must be absolute")
    output = output.resolve()
    if not output.parent.is_dir() or output.exists():
        parser.error("output parent must exist and output must not exist")

    started = time.perf_counter()
    payload = build_report(z_factors=tuple(args.z_factors))
    payload["cpu_wall_time_s"] = time.perf_counter() - started
    payload["increment_state_report_sha256"] = canonical_payload_sha256(
        payload, "increment_state_report_sha256"
    )
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "ready": payload["ready"],
                "output": str(output),
                "file_sha256": file_sha256(output),
                "cpu_wall_time_s": payload["cpu_wall_time_s"],
                "optimizer_start_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
