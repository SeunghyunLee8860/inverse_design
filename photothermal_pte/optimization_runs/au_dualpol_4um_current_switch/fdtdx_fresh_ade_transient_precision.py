#!/usr/bin/env python3
"""Fail-closed long-time float32 ADE recurrence diagnostic.

The carrier-fit audit checks an algebraic transfer function at one frequency.
This module checks the actual repeated recurrence under the pinned four-period
linear source ramp.  It uses the same locked float32 coefficients for both a
float32 state and a float64 reference state, so their difference isolates
state-update precision rather than a different material fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import lfilter


VERSION = "fdtdx-fresh-ade-transient-precision-v1"
LAW_VERSION = "fdtdx-fresh-stable-two-pole-material-v1"
C0_M_PER_S = 299_792_458.0
MATERIAL_AXES = ("au", "a", "b", "c")
NUM_WINDOWS = 4
FLOAT64_WINDOW_CHANGE_LIMIT = 1.0e-5
FLOAT32_WINDOW_CHANGE_LIMIT = 5.0e-3
FLOAT32_VS_FLOAT64_LIMIT = 5.0e-3


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_payload_sha256(payload: Mapping[str, Any], hash_key: str) -> str:
    unhashed = dict(payload)
    unhashed.pop(hash_key, None)
    return hashlib.sha256(
        json.dumps(
            unhashed,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def _relative_change(current: complex, previous: complex) -> float:
    return float(
        abs(current - previous)
        / max(abs(current), np.finfo(np.float64).tiny)
    )


def _condition_audit(
    coefficients: np.ndarray,
    omega_rad_s: float,
    dt_s: float,
) -> list[dict[str, float]]:
    theta = np.float32(omega_rad_s * dt_s)
    z_minus = np.exp(np.complex64(-1j * theta))
    z_plus = np.exp(np.complex64(1j * theta))
    rows = []
    for c1, c2, _ in np.asarray(coefficients, dtype=np.float32):
        denominator = np.complex64(z_minus - c1 - c2 * z_plus)
        scale = float(abs(z_minus) + abs(c1) + abs(c2 * z_plus))
        condition = scale / max(float(abs(denominator)), np.finfo(float).tiny)
        rows.append(
            {
                "carrier_denominator_abs": float(abs(denominator)),
                "cancellation_condition_estimate": condition,
                "condition_times_float32_epsilon": float(
                    condition * np.finfo(np.float32).eps
                ),
            }
        )
    return rows


def simulate_axis(
    coefficients: Sequence[Sequence[float]],
    *,
    dt_s: float,
    wavelength_m: float,
    total_periods: int,
    startup_periods: int,
    window_periods: int,
) -> dict[str, Any]:
    """Integrate one two-pole scalar recurrence in float32 and float64."""

    coefficient_array = np.asarray(coefficients, dtype=np.float32)
    if coefficient_array.shape != (2, 3):
        raise ValueError("coefficients must have shape (2, 3)")
    if not np.all(np.isfinite(coefficient_array)):
        raise ValueError("coefficients must be finite")
    if dt_s <= 0.0 or wavelength_m <= 0.0:
        raise ValueError("dt_s and wavelength_m must be positive")
    if startup_periods <= 0 or window_periods <= 0:
        raise ValueError("startup_periods and window_periods must be positive")
    if total_periods < NUM_WINDOWS * window_periods:
        raise ValueError("total_periods must contain four complete late windows")

    period_s = wavelength_m / C0_M_PER_S
    omega_rad_s = 2.0 * math.pi / period_s
    time_steps_total = int(math.ceil(total_periods * period_s / dt_s))
    time_s = np.arange(time_steps_total, dtype=np.float64) * dt_s
    ramp = np.clip(time_s / (startup_periods * period_s), 0.0, 1.0)
    drive = np.asarray(ramp * np.cos(omega_rad_s * time_s), dtype=np.float32)
    phase = np.exp(1j * omega_rad_s * time_s)

    first_period = total_periods - NUM_WINDOWS * window_periods
    window_bounds = [
        (first_period + index * window_periods,
         first_period + (index + 1) * window_periods)
        for index in range(NUM_WINDOWS)
    ]
    masks = [
        (time_s >= lower * period_s) & (time_s < upper * period_s)
        for lower, upper in window_bounds
    ]
    drive_phasors = [
        np.sum(drive[mask] * phase[mask], dtype=np.complex128)
        for mask in masks
    ]
    if any(abs(value) <= np.finfo(float).tiny for value in drive_phasors):
        raise RuntimeError("late-window drive phasor is zero")

    precision: dict[str, Any] = {}
    for name, dtype in (("float32", np.float32), ("float64", np.float64)):
        pole_outputs = []
        for c1, c2, c3 in coefficient_array.astype(dtype):
            numerator = np.asarray([c3], dtype=dtype)
            denominator = np.asarray([1.0, -c1, -c2], dtype=dtype)
            pole_outputs.append(
                lfilter(numerator, denominator, drive.astype(dtype, copy=False))
            )
        response = np.sum(np.stack(pole_outputs), axis=0, dtype=dtype)
        response_phasors = [
            np.sum(response[mask] * phase[mask], dtype=np.complex128)
            for mask in masks
        ]
        susceptibility = [
            response_value / drive_value
            for response_value, drive_value in zip(
                response_phasors, drive_phasors, strict=True
            )
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
            "maximum_relative_window_change": max(changes),
            "last_relative_window_change": changes[-1],
        }

    float32_late = complex(*precision["float32"]["susceptibility_windows"][-1])
    float64_late = complex(*precision["float64"]["susceptibility_windows"][-1])
    disagreement = _relative_change(float64_late, float32_late)
    gates = {
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
        "coefficients_float32": coefficient_array.astype(float).tolist(),
        "time_steps_total": time_steps_total,
        "window_bounds_periods": [list(value) for value in window_bounds],
        "window_sample_counts": [int(np.count_nonzero(mask)) for mask in masks],
        "carrier_conditioning": _condition_audit(
            coefficient_array, omega_rad_s, dt_s
        ),
        "precision": precision,
        "float32_vs_float64_late_relative_difference": disagreement,
        "gates": gates,
        "ready": all(gates.values()),
    }


def analyze_material_law(law: Mapping[str, Any]) -> dict[str, Any]:
    try:
        version = law["version"]
        algorithm = law["algorithm"]
        case_binding = law["case_binding"]
        material_axes = law["material_axes"]
        promotion = law["promotion"]
        dt_s = float(case_binding["realized_float32_cfl"]["time_step_s"])
        time_spec = case_binding["time_spec"]
        wavelength_m = float(algorithm["target_wavelength_m"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"invalid material-law structure: {error}") from error

    structure_gates = {
        "material_law_version_exact": version == LAW_VERSION,
        "material_axes_exact": set(material_axes) == set(MATERIAL_AXES),
        "candidate_only": promotion.get("candidate_only") is True,
        "optimizer_already_forbidden": (
            promotion.get("optimizer_start_allowed") is False
        ),
        "linear_ramp_startup_is_positive": int(
            time_spec["source_startup_periods"]
        ) > 0,
    }
    if not all(structure_gates.values()):
        raise RuntimeError(f"material-law structure gates failed: {structure_gates}")

    axes: dict[str, Any] = {}
    for name in MATERIAL_AXES:
        try:
            pole_items = material_axes[name]["candidate"]["poles"]
            coefficients = [
                [item["c1"], item["c2"], item["c3"]]
                for item in pole_items
            ]
        except (KeyError, TypeError) as error:
            raise RuntimeError(f"invalid candidate poles for {name}: {error}") from error
        axes[name] = simulate_axis(
            coefficients,
            dt_s=dt_s,
            wavelength_m=wavelength_m,
            total_periods=int(time_spec["total_periods"]),
            startup_periods=int(time_spec["source_startup_periods"]),
            window_periods=int(time_spec["window_periods"]),
        )

    numerical_gates = {
        "float64_reference_all_settled": all(
            item["gates"]["float64_reference_last_window_settled"]
            for item in axes.values()
        ),
        "float32_recurrence_all_settled": all(
            item["gates"]["float32_last_window_settled"]
            for item in axes.values()
        ),
        "float32_all_match_float64_late_response": all(
            item["gates"]["float32_matches_float64_late_response"]
            for item in axes.values()
        ),
    }
    gates = {**structure_gates, **numerical_gates}
    failed_gates = [name for name, value in gates.items() if not value]
    ready = not failed_gates
    return {
        "status": (
            "VALIDATED_FDTDX_FRESH_ADE_TRANSIENT_PRECISION"
            if ready
            else "BLOCKED_FDTDX_FRESH_ADE_TRANSIENT_PRECISION"
        ),
        "ready": ready,
        "gates": gates,
        "failed_gates": failed_gates,
        "thresholds": {
            "float64_last_window_relative_change": (
                FLOAT64_WINDOW_CHANGE_LIMIT
            ),
            "float32_last_window_relative_change": (
                FLOAT32_WINDOW_CHANGE_LIMIT
            ),
            "float32_vs_float64_late_relative_difference": (
                FLOAT32_VS_FLOAT64_LIMIT
            ),
        },
        "source_emulation": {
            "profile": "SingleFrequencyProfile linear ramp",
            "drive_samples_identical_between_precisions": True,
            "dt_s": dt_s,
            "wavelength_m": wavelength_m,
            "time_spec": dict(time_spec),
        },
        "material_axes": axes,
        "optimizer_start_allowed": False,
        "next_allowed_step": (
            "replace or reformulate the ill-conditioned float32 ADE recurrence "
            "and rerun this CPU transient gate before any longer FDTD, finer "
            "mesh, adjoint, thermal/electrical, or optimization run"
        ),
    }


def audit_material_law_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    actual_sha256 = file_sha256(resolved) if resolved.is_file() else None
    expected_is_hex = (
        len(expected_sha256) == 64
        and expected_sha256 == expected_sha256.lower()
        and all(character in "0123456789abcdef" for character in expected_sha256)
    )
    if not resolved.is_absolute() or not resolved.is_file():
        raise RuntimeError("material-law path must be an existing absolute file")
    if not expected_is_hex or actual_sha256 != expected_sha256:
        raise RuntimeError("material-law file SHA-256 mismatch")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    internal = payload.get("material_law_contract_sha256")
    computed = canonical_payload_sha256(payload, "material_law_contract_sha256")
    if internal != computed:
        raise RuntimeError("material-law internal contract SHA-256 mismatch")
    return {
        "path": str(resolved),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "material_law_contract_sha256": internal,
        "ready": True,
    }, payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    if not resolved.is_absolute() or not resolved.parent.is_dir():
        raise RuntimeError("output must have an existing absolute parent directory")
    if resolved.exists():
        raise RuntimeError("refusing to overwrite an existing diagnostic")
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-law", type=Path, required=True)
    parser.add_argument("--material-law-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit, law = audit_material_law_file(
        args.material_law, args.material_law_sha256
    )
    result = analyze_material_law(law)
    payload = {
        "version": VERSION,
        "material_law_file_audit": audit,
        **result,
        "scope": (
            "CPU scalar recurrence precision audit only; no FDTD, material "
            "certificate, mesh certificate, adjoint, or optimizer"
        ),
        "implementation_sha256": file_sha256(Path(__file__).resolve()),
    }
    payload["diagnostic_payload_sha256"] = canonical_payload_sha256(
        payload, "diagnostic_payload_sha256"
    )
    _atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "ready": payload["ready"],
                "failed_gates": payload["failed_gates"],
                "output": str(args.output.expanduser().resolve()),
                "diagnostic_payload_sha256": payload[
                    "diagnostic_payload_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
