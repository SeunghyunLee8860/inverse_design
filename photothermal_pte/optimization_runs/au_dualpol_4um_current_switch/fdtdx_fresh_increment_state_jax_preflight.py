#!/usr/bin/env python3
"""Fail-closed JAX-kernel preflight for the isolated increment-state fork.

This remains a zero-dimensional material-state test.  It imports one exact,
clean FDTDX fork commit, executes its JIT-compatible ``(P, delta-P)`` kernel
for the complete late-time z ladder, and compares locked-coefficient float32
and float64 states.  It never constructs or runs a 3-D FDTD model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_ade_precision_diagnostic import (
    C0_M_PER_S,
    WAVELENGTH_M,
    load_material_epsilon,
    realized_float32_cfl,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_increment_state_precision import (
    CARRIER_RELATIVE_ERROR_LIMIT,
    FLOAT32_VS_FLOAT64_LIMIT,
    FLOAT32_WINDOW_CHANGE_LIMIT,
    FLOAT64_WINDOW_CHANGE_LIMIT,
    MATERIAL_AXES,
    physical_pole_from_target,
)


VERSION = "fdtdx-fresh-increment-state-jax-preflight-v1"
NUM_WINDOWS = 4
TOTAL_PERIODS = 32
STARTUP_PERIODS = 4
WINDOW_PERIODS = 4


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


def _git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def audit_fork(source: Path, expected_commit: str) -> dict[str, Any]:
    root = source.expanduser()
    if not root.is_absolute() or not root.is_dir():
        raise RuntimeError("--fdtdx-source must be an existing absolute directory")
    root = root.resolve()
    expected = expected_commit.strip().lower()
    if len(expected) != 40 or any(character not in "0123456789abcdef" for character in expected):
        raise RuntimeError("--fdtdx-commit must be a full lowercase Git SHA")
    observed = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain", "--untracked-files=all")
    module = root / "src/fdtdx/increment_state.py"
    test = root / "tests/unit/test_increment_state.py"
    checks = {
        "head_matches_expected_commit": observed == expected,
        "fork_worktree_clean": status == "",
        "increment_state_module_exists": module.is_file(),
        "increment_state_test_exists": test.is_file(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"FDTDX fork audit failed: {checks}")
    return {
        "root": str(root),
        "expected_commit": expected,
        "observed_commit": observed,
        "increment_state_module_sha256": file_sha256(module),
        "increment_state_test_sha256": file_sha256(test),
        "checks": checks,
        "ready": True,
    }


def import_fork(source: Path):
    expected_src = source.resolve() / "src"
    if str(expected_src) not in sys.path:
        sys.path.insert(0, str(expected_src))
    fdtdx = importlib.import_module("fdtdx")
    increment_state = importlib.import_module("fdtdx.increment_state")
    imported_fdtdx = Path(fdtdx.__file__).resolve()
    imported_increment = Path(increment_state.__file__).resolve()
    if expected_src not in imported_fdtdx.parents:
        raise RuntimeError(f"unpinned fdtdx import: {imported_fdtdx}")
    if expected_src not in imported_increment.parents:
        raise RuntimeError(f"unpinned increment-state import: {imported_increment}")
    return fdtdx, increment_state, imported_fdtdx, imported_increment


def late_window_bounds(
    time_s: np.ndarray,
    period_s: float,
    *,
    total_periods: int = TOTAL_PERIODS,
    window_periods: int = WINDOW_PERIODS,
) -> np.ndarray:
    if time_s.ndim != 1 or time_s.size == 0:
        raise ValueError("time_s must be one non-empty vector")
    if period_s <= 0.0 or window_periods <= 0:
        raise ValueError("period_s and window_periods must be positive")
    if total_periods < NUM_WINDOWS * window_periods:
        raise ValueError("total_periods must contain four complete windows")
    first = total_periods - NUM_WINDOWS * window_periods
    return np.asarray(
        [
            (
                np.searchsorted(time_s, (first + index * window_periods) * period_s),
                np.searchsorted(time_s, (first + (index + 1) * window_periods) * period_s),
            )
            for index in range(NUM_WINDOWS)
        ],
        dtype=np.int32,
    )


def _relative_change(current: complex, previous: complex) -> float:
    return float(abs(current - previous) / max(abs(current), np.finfo(float).tiny))


def summarize_precision(float32_values: np.ndarray, float64_values: np.ndarray) -> dict[str, Any]:
    float32_values = np.asarray(float32_values, dtype=np.complex128)
    float64_values = np.asarray(float64_values, dtype=np.complex128)
    if float32_values.shape != (NUM_WINDOWS,) or float64_values.shape != (NUM_WINDOWS,):
        raise ValueError("precision inputs must contain exactly four windows")
    float32_changes = [
        _relative_change(current, previous)
        for previous, current in zip(float32_values[:-1], float32_values[1:], strict=True)
    ]
    float64_changes = [
        _relative_change(current, previous)
        for previous, current in zip(float64_values[:-1], float64_values[1:], strict=True)
    ]
    disagreement = _relative_change(float64_values[-1], float32_values[-1])
    gates = {
        "float64_reference_last_window_settled": float64_changes[-1] <= FLOAT64_WINDOW_CHANGE_LIMIT,
        "float32_last_window_settled": float32_changes[-1] <= FLOAT32_WINDOW_CHANGE_LIMIT,
        "float32_matches_float64_late_response": disagreement <= FLOAT32_VS_FLOAT64_LIMIT,
    }
    return {
        "float32_susceptibility_windows": [
            [float(value.real), float(value.imag)] for value in float32_values
        ],
        "float64_susceptibility_windows": [
            [float(value.real), float(value.imag)] for value in float64_values
        ],
        "float32_relative_window_changes": float32_changes,
        "float64_relative_window_changes": float64_changes,
        "float32_vs_float64_late_relative_difference": disagreement,
        "gates": gates,
        "ready": all(gates.values()),
    }


def _jax_runner(update_kernel, *, steps: int, state_dtype):
    @jax.jit
    def run(coeff_a, coeff_c, coeff_b, drive, phase, bounds):
        coeff_a = coeff_a.astype(state_dtype)
        coeff_c = coeff_c.astype(state_dtype)
        coeff_b = coeff_b.astype(state_dtype)

        def body(index, state):
            polarization, increment, p_sum, e_sum = state
            polarization, increment = update_kernel(
                polarization,
                increment,
                drive[index].astype(state_dtype),
                coeff_a,
                coeff_c,
                coeff_b,
            )
            weights = (
                (index >= bounds[:, 0]) & (index < bounds[:, 1])
            ).astype(jnp.float64)
            phasor = phase[index]
            p_sum = p_sum + weights[:, None] * (
                polarization.astype(jnp.float64)[None, :] * phasor
            )
            e_sum = e_sum + weights * (
                drive[index].astype(jnp.float64) * phasor
            )
            return polarization, increment, p_sum, e_sum

        initial = (
            jnp.zeros(len(MATERIAL_AXES), dtype=state_dtype),
            jnp.zeros(len(MATERIAL_AXES), dtype=state_dtype),
            jnp.zeros((NUM_WINDOWS, len(MATERIAL_AXES)), dtype=jnp.complex128),
            jnp.zeros(NUM_WINDOWS, dtype=jnp.complex128),
        )
        final = jax.lax.fori_loop(0, steps, body, initial)
        return final[2] / final[3][:, None]

    return run


def analyze_level(fdtdx, increment_state, z_factor: int) -> dict[str, Any]:
    cfl = realized_float32_cfl(z_factor)
    dt_s = float(cfl["time_step_s"])
    period_s = WAVELENGTH_M / C0_M_PER_S
    omega = 2.0 * math.pi / period_s
    steps = int(math.ceil(TOTAL_PERIODS * period_s / dt_s))
    time_s = np.arange(steps, dtype=np.float64) * dt_s
    drive = np.asarray(
        np.clip(time_s / (STARTUP_PERIODS * period_s), 0.0, 1.0)
        * np.cos(omega * time_s),
        dtype=np.float32,
    )
    phase = np.exp(1j * omega * time_s)
    bounds = late_window_bounds(time_s, period_s)

    epsilon = load_material_epsilon()
    rows = []
    poles = {}
    for name in MATERIAL_AXES:
        physical = physical_pole_from_target(name, epsilon[name])
        if physical["kind"] == "Drude":
            pole = fdtdx.DrudePole(
                plasma_frequency=math.sqrt(physical["coupling_sq_rad_s2"]),
                damping=physical["gamma_rad_s"],
            )
        else:
            pole = fdtdx.LorentzPole(
                resonance_frequency=physical["omega_0_rad_s"],
                damping=physical["gamma_rad_s"],
                delta_epsilon=(
                    physical["coupling_sq_rad_s2"] / physical["omega_0_rad_s"] ** 2
                ),
            )
        coeff_a, coeff_c, coeff_b = increment_state.compute_increment_state_coefficients_per_axis((pole,), dt_s)
        row = np.asarray([coeff_a[0, 0], coeff_c[0, 0], coeff_b[0, 0]], dtype=np.float32)
        rows.append(row)
        poles[name] = physical
    coefficients = np.stack(rows)

    observed = {}
    timings = {}
    for label, dtype in (("float32", jnp.float32), ("float64", jnp.float64)):
        started = time.perf_counter()
        runner = _jax_runner(
            increment_state.update_dispersive_increment_state,
            steps=steps,
            state_dtype=dtype,
        )
        values = runner(
            jnp.asarray(coefficients[:, 0]),
            jnp.asarray(coefficients[:, 1]),
            jnp.asarray(coefficients[:, 2]),
            jnp.asarray(drive),
            jnp.asarray(phase),
            jnp.asarray(bounds),
        )
        observed[label] = np.asarray(values)
        timings[label] = time.perf_counter() - started

    axes = {}
    for axis, name in enumerate(MATERIAL_AXES):
        summary = summarize_precision(observed["float32"][:, axis], observed["float64"][:, axis])
        target = complex(*poles[name]["target_susceptibility"])
        discrete = complex(
            np.asarray(
                increment_state.susceptibility_from_increment_coefficients(
                    jnp.asarray(coefficients[axis : axis + 1, 0], dtype=jnp.float64),
                    jnp.asarray(coefficients[axis : axis + 1, 1], dtype=jnp.float64),
                    jnp.asarray(coefficients[axis : axis + 1, 2], dtype=jnp.float64),
                    omega,
                    dt_s,
                )
            )
        )
        carrier_error = abs(discrete - target) / abs(target)
        summary["coefficients_float32"] = {
            "A": float(coefficients[axis, 0]),
            "C": float(coefficients[axis, 1]),
            "B": float(coefficients[axis, 2]),
        }
        summary["discrete_carrier_relative_error"] = float(carrier_error)
        summary["gates"]["float32_discrete_carrier_fit"] = carrier_error <= CARRIER_RELATIVE_ERROR_LIMIT
        summary["ready"] = all(summary["gates"].values())
        axes[name] = summary
    return {
        "z_factor": z_factor,
        "cfl": cfl,
        "time_steps_total": steps,
        "window_bounds_indices": bounds.astype(int).tolist(),
        "jax_compile_and_execute_wall_s": timings,
        "material_axes": axes,
        "ready": all(axis["ready"] for axis in axes.values()),
    }


def build_report(source: Path, expected_commit: str, z_factors: Sequence[int]) -> dict[str, Any]:
    fork_audit = audit_fork(source, expected_commit)
    fdtdx, increment_state, imported_fdtdx, imported_increment = import_fork(source)
    environment_gates = {
        "jax_backend_is_cpu": jax.default_backend() == "cpu",
        "jax_x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "z_factors_unique_positive": len(set(z_factors)) == len(z_factors) and all(value > 0 for value in z_factors),
    }
    if not all(environment_gates.values()):
        raise RuntimeError(f"JAX preflight environment failed: {environment_gates}")
    levels = {str(value): analyze_level(fdtdx, increment_state, value) for value in z_factors}
    gates = {
        **environment_gates,
        "fork_audit_ready": fork_audit["ready"],
        "all_levels_ready": all(level["ready"] for level in levels.values()),
        "optimizer_remains_forbidden": True,
    }
    ready = all(gates.values())
    payload: dict[str, Any] = {
        "version": VERSION,
        "status": "VALIDATED_INCREMENT_STATE_JAX_KERNEL" if ready else "BLOCKED_INCREMENT_STATE_JAX_KERNEL",
        "ready": ready,
        "scope": "CPU JAX material-state kernel only; no grid, FDTD field solve, mesh, adjoint, PTE, or optimizer",
        "fork_audit": fork_audit,
        "imports": {
            "fdtdx": str(imported_fdtdx),
            "increment_state": str(imported_increment),
        },
        "jax": {
            "backend": jax.default_backend(),
            "x64_enabled": bool(jax.config.read("jax_enable_x64")),
            "devices": [str(device) for device in jax.devices()],
        },
        "time_spec": {
            "total_periods": TOTAL_PERIODS,
            "source_startup_periods": STARTUP_PERIODS,
            "window_periods": WINDOW_PERIODS,
        },
        "thresholds": {
            "carrier_relative_error": CARRIER_RELATIVE_ERROR_LIMIT,
            "float64_last_window_relative_change": FLOAT64_WINDOW_CHANGE_LIMIT,
            "float32_last_window_relative_change": FLOAT32_WINDOW_CHANGE_LIMIT,
            "float32_vs_float64_late_relative_difference": FLOAT32_VS_FLOAT64_LIMIT,
        },
        "levels": levels,
        "gates": gates,
        "promotion": {
            "candidate_only": True,
            "is_full_fdtdx_implementation_certificate": False,
            "is_mesh_certificate": False,
            "is_adjoint_certificate": False,
            "optimizer_start_allowed": False,
            "next_allowed_step": "wire the isolated state into an opt-in FDTDX E update and pass small forward plus checkpointed AD-FD controls",
        },
        "provenance": {
            "preflight_script_sha256": file_sha256(Path(__file__).resolve()),
        },
    }
    payload["jax_preflight_payload_sha256"] = canonical_payload_sha256(payload, "jax_preflight_payload_sha256")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdtdx-source", type=Path, required=True)
    parser.add_argument("--fdtdx-commit", required=True)
    parser.add_argument("--z-factors", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser()
    if not output.is_absolute():
        parser.error("--output must be absolute")
    output = output.resolve()
    if not output.parent.is_dir() or output.exists():
        parser.error("output parent must exist and output must not exist")
    started = time.perf_counter()
    payload = build_report(args.fdtdx_source, args.fdtdx_commit, tuple(args.z_factors))
    payload["total_cpu_wall_s"] = time.perf_counter() - started
    payload["jax_preflight_payload_sha256"] = canonical_payload_sha256(payload, "jax_preflight_payload_sha256")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "ready": payload["ready"],
                "output": str(output),
                "file_sha256": file_sha256(output),
                "total_cpu_wall_s": payload["total_cpu_wall_s"],
                "optimizer_start_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
