#!/usr/bin/env python3
"""Fail-closed float32 ADE precision diagnostic for the fresh z ladder.

This module performs no FDTD solve.  It reproduces the carrier-frequency
single-Drude refit on the *realized float32 rectilinear grid*, checks a much
wider damping interval, and constructs (but does not promote) a two-Drude
candidate using only recurrence pairs whose DC root does not exceed one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
    grid_edges,
)


HERE = Path(__file__).resolve().parent
MATERIAL_CONTRACT = HERE / "results_materials_4um/4um_material_contract.json"
OPTICAL_MODEL = HERE / "fdtdx_4um_model.py"
C0_M_PER_S = 299_792_458.0
WAVELENGTH_M = 4.0e-6
COURANT_FACTOR = 0.25
FIT_RELATIVE_TOLERANCE = 1.0e-5
LOCAL_RATIO_RANGE = (0.8, 1.2)
LOCAL_POINTS = 400_001
WIDE_RATIO_RANGE = (0.01, 10.0)
WIDE_POINTS = 400_001
Z_FACTORS = (2, 4, 8, 16, 32)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_material_epsilon() -> dict[str, complex]:
    payload = json.loads(MATERIAL_CONTRACT.read_text(encoding="utf-8"))
    if payload.get("status") != "VALIDATED_4UM_SINGLE_FREQUENCY_MATERIAL_READBACK":
        raise RuntimeError("4 um material contract is not validated")

    def value(item: dict[str, float]) -> complex:
        return complex(float(item["real"]), float(item["imag"]))

    result = {"au": value(payload["materials"]["Au"]["epsilon"])}
    result.update(
        {
            axis: value(payload["materials"]["TaIrTe4"][axis]["epsilon"])
            for axis in ("a", "b", "c")
        }
    )
    if result["b"] != result["c"]:
        raise RuntimeError("TaIrTe4 b/c material contract changed")
    if any(item.imag <= 0.0 for item in result.values()):
        raise RuntimeError(f"material contract contains active data: {result!r}")
    return result


def load_au_epsilon() -> complex:
    return load_material_epsilon()["au"]


def realized_float32_cfl(z_factor: int) -> dict[str, Any]:
    """Emulate JAX-x64-disabled RectilinearGrid edge realization and CFL."""

    axes = grid_edges(MeshSpec(z_factor=z_factor))
    minimum_spacings = tuple(
        float(np.min(np.diff(np.asarray(axis, dtype=np.float32))))
        for axis in axes
    )
    inverse_metric = sum(value**-2 for value in minimum_spacings)
    dt_s = COURANT_FACTOR / (C0_M_PER_S * math.sqrt(inverse_metric))
    return {
        "z_factor": z_factor,
        "realized_edge_dtype": "float32",
        "minimum_spacing_xyz_m": list(minimum_spacings),
        "courant_factor": COURANT_FACTOR,
        "time_step_s": dt_s,
    }


def drude_gamma_seed(target_chi: complex, omega_rad_s: float, dt_s: float) -> float:
    theta = omega_rad_s * dt_s
    omega_d_sq = (2.0 * math.sin(0.5 * theta) / dt_s) ** 2
    omega_s = math.sin(theta) / dt_s
    return omega_d_sq * target_chi.imag / ((-target_chi.real) * omega_s)


def lorentz_gamma_seed(
    target_chi: complex,
    omega_rad_s: float,
    dt_s: float,
    *,
    resonance_ratio: float = 2.0,
) -> tuple[float, float]:
    if not (target_chi.real > 0.0 and target_chi.imag > 0.0):
        raise ValueError("Lorentz target susceptibility must have positive real/loss")
    theta = omega_rad_s * dt_s
    omega_d_sq = (2.0 * math.sin(0.5 * theta) / dt_s) ** 2
    omega_s = math.sin(theta) / dt_s
    omega_0 = resonance_ratio * omega_rad_s
    detuning = omega_0**2 - omega_d_sq
    gamma = (target_chi.imag / target_chi.real) * detuning / omega_s
    return gamma, omega_0


def _coefficient_states(
    target_chi: complex,
    omega_rad_s: float,
    dt_s: float,
    ratios: np.ndarray,
    *,
    gamma_seed: float | None = None,
    omega_0_rad_s: float = 0.0,
) -> dict[str, np.ndarray | np.complex64]:
    seed = (
        drude_gamma_seed(target_chi, omega_rad_s, dt_s)
        if gamma_seed is None
        else gamma_seed
    )
    gamma = seed * ratios
    denominator = 1.0 + 0.5 * gamma * dt_s
    c1 = np.asarray(
        (2.0 - omega_0_rad_s**2 * dt_s**2) / denominator,
        dtype=np.float32,
    )
    c2 = np.asarray(
        -(1.0 - 0.5 * gamma * dt_s) / denominator,
        dtype=np.float32,
    )
    theta = np.float32(omega_rad_s * dt_s)
    z_minus = np.exp(np.complex64(-1j * theta))
    z_plus = np.exp(np.complex64(1j * theta))
    ade_denominator = (
        z_minus
        - c1.astype(np.complex64)
        - c2.astype(np.complex64) * z_plus
    )
    required_c3 = np.complex64(target_chi) * ade_denominator
    c3 = np.asarray(required_c3.real, dtype=np.float32)
    realized = c3.astype(np.complex64) / ade_denominator
    error = np.asarray(
        np.abs(realized.astype(np.complex128) - target_chi) / abs(target_chi),
        dtype=np.float64,
    )
    phase_score = np.abs(required_c3.imag) / np.maximum(
        np.abs(required_c3.real), np.finfo(np.float32).tiny
    )
    return {
        "gamma_seed": np.asarray(seed, dtype=np.float64),
        "omega_0_rad_s": np.asarray(omega_0_rad_s, dtype=np.float64),
        "gamma": gamma,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "denominator": ade_denominator,
        "required_c3": required_c3,
        "realized": realized,
        "error": error,
        "phase_score": phase_score,
        "z_minus": z_minus,
    }


def _single_row(
    states: dict[str, np.ndarray | np.complex64],
    ratios: np.ndarray,
    index: int,
) -> dict[str, Any]:
    realized = np.asarray(states["realized"])[index]
    c1 = np.asarray(states["c1"])[index]
    c2 = np.asarray(states["c2"])[index]
    return {
        "gamma_ratio": float(ratios[index]),
        "gamma_rad_s": float(np.asarray(states["gamma"])[index]),
        "c1": float(c1),
        "c2": float(c2),
        "c3": float(np.asarray(states["c3"])[index]),
        "c1_plus_c2": float(np.float32(c1) + np.float32(c2)),
        "realized_susceptibility": [
            float(realized.real),
            float(realized.imag),
        ],
        "fit_relative_error": float(np.asarray(states["error"])[index]),
        "fit_gate_passed": bool(
            np.asarray(states["error"])[index] < FIT_RELATIVE_TOLERANCE
        ),
    }


def single_pole_scan(
    target_chi: complex,
    omega_rad_s: float,
    dt_s: float,
    *,
    ratio_range: tuple[float, float],
    points: int,
    selection: str,
    gamma_seed: float | None = None,
    omega_0_rad_s: float = 0.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray | np.complex64], np.ndarray]:
    ratios = np.linspace(*ratio_range, points, dtype=np.float64)
    states = _coefficient_states(
        target_chi,
        omega_rad_s,
        dt_s,
        ratios,
        gamma_seed=gamma_seed,
        omega_0_rad_s=omega_0_rad_s,
    )
    if selection == "current_phase_score":
        index = int(np.argmin(np.asarray(states["phase_score"])))
    elif selection == "minimum_realized_error":
        index = int(np.argmin(np.asarray(states["error"])))
    else:
        raise ValueError(f"unknown selection {selection!r}")
    row = _single_row(states, ratios, index)
    row.update(
        ratio_range=list(ratio_range),
        sampled_points=points,
        selection=selection,
    )
    return row, states, ratios


def stable_two_drude_candidate(
    target_chi: complex,
    dt_s: float,
    states: dict[str, np.ndarray | np.complex64],
    ratios: np.ndarray,
    *,
    pole_kind: str = "Drude",
) -> dict[str, Any]:
    """Fit two positive strengths on stable float32 recurrence pairs."""

    if pole_kind not in ("Drude", "Lorentz"):
        raise ValueError(f"unsupported pole kind {pole_kind!r}")

    c1_all = np.asarray(states["c1"])
    c2_all = np.asarray(states["c2"])
    keys = np.stack((c1_all.view(np.uint32), c2_all.view(np.uint32)), axis=1)
    _, first = np.unique(keys, axis=0, return_index=True)
    indices = np.sort(first)
    selected_c1 = c1_all[indices].astype(np.float64)
    selected_c2 = c2_all[indices].astype(np.float64)
    discriminant = selected_c1**2 + 4.0 * selected_c2
    root_offset = np.sqrt(discriminant.astype(np.complex128))
    root_radius = np.maximum(
        np.abs(0.5 * (selected_c1 + root_offset)),
        np.abs(0.5 * (selected_c1 - root_offset)),
    )
    stable = root_radius <= 1.0 + 4.0 * np.finfo(np.float64).eps
    indices = indices[stable]
    root_radius = root_radius[stable]
    denominator = np.asarray(states["denominator"])[indices]
    response_per_c3 = np.complex64(1.0) / denominator
    target32 = np.complex64(target_chi)
    cross = (
        response_per_c3.real * target32.imag
        - response_per_c3.imag * target32.real
    )
    below = np.flatnonzero(cross <= 0.0)
    above = np.flatnonzero(cross >= 0.0)
    if below.size == 0 or above.size == 0:
        return {
            "found": False,
            "reason": "stable coefficient phases do not bracket the target",
        }
    below = below[np.argsort(np.abs(cross[below]))[:64]]
    above = above[np.argsort(np.abs(cross[above]))[:64]]
    best: tuple[float, tuple[int, int, np.ndarray, np.complex64]] | None = None
    target_vector = np.asarray([target32.real, target32.imag], dtype=np.float64)
    for left in below:
        for right in above:
            matrix = np.asarray(
                [
                    [response_per_c3[left].real, response_per_c3[right].real],
                    [response_per_c3[left].imag, response_per_c3[right].imag],
                ],
                dtype=np.float64,
            )
            if abs(float(np.linalg.det(matrix))) < np.finfo(float).tiny:
                continue
            weights = np.linalg.solve(matrix, target_vector)
            if np.any(weights <= 0.0):
                continue
            weights32 = weights.astype(np.float32)
            realized = (
                np.complex64(weights32[0]) * response_per_c3[left]
                + np.complex64(weights32[1]) * response_per_c3[right]
            )
            error = abs(complex(realized) - target_chi) / abs(target_chi)
            if best is None or error < best[0]:
                best = (float(error), (left, right, weights32, realized))
    if best is None:
        return {"found": False, "reason": "no positive two-pole weights found"}
    error, (left, right, weights32, realized) = best
    poles = []
    omega_0 = float(np.asarray(states["omega_0_rad_s"]))
    for local_index, weight in zip((left, right), weights32, strict=True):
        source_index = int(indices[local_index])
        gamma = float(np.asarray(states["gamma"])[source_index])
        recurrence_denominator = 1.0 + 0.5 * gamma * dt_s
        coupling = math.sqrt(float(weight) * recurrence_denominator / dt_s**2)
        reconstructed_c3 = np.float32(
            coupling**2 * dt_s**2 / recurrence_denominator
        )
        c1 = c1_all[source_index]
        c2 = c2_all[source_index]
        pole = {
            "kind": pole_kind,
            "gamma_ratio": float(ratios[source_index]),
            "gamma_rad_s": gamma,
            "omega_0_rad_s": omega_0,
            "coupling_rad_s": coupling,
            "c1": float(c1),
            "c2": float(c2),
            "c3": float(weight),
            "reconstructed_float32_c3": float(reconstructed_c3),
            "c1_plus_c2": float(np.float32(c1) + np.float32(c2)),
            "recurrence_root_radius": float(root_radius[local_index]),
            "positive_strength": bool(weight > 0.0),
            "recurrence_roots_not_above_one": bool(
                root_radius[local_index]
                <= 1.0 + 4.0 * np.finfo(np.float64).eps
            ),
            "dc_root_not_above_one": bool(
                pole_kind != "Drude"
                or float(np.float32(c1) + np.float32(c2)) <= 1.0
            ),
        }
        if pole_kind == "Drude":
            pole["omega_p_rad_s"] = coupling
        else:
            pole["delta_epsilon"] = coupling**2 / omega_0**2
        poles.append(pole)
    return {
        "found": True,
        "poles": poles,
        "realized_susceptibility": [float(realized.real), float(realized.imag)],
        "fit_relative_error": error,
        "fit_gate_passed": bool(error < FIT_RELATIVE_TOLERANCE),
        "candidate_only": True,
        "promotion_forbidden_until_same_law_time_and_z_validation": True,
    }


def analyze_material_axis(
    z_factor: int,
    material_axis: str,
    target_epsilon: complex,
) -> dict[str, Any]:
    cfl = realized_float32_cfl(z_factor)
    dt_s = float(cfl["time_step_s"])
    omega = 2.0 * math.pi * C0_M_PER_S / WAVELENGTH_M
    target_chi = target_epsilon - 1.0
    if target_chi.real < 0.0 and target_chi.imag > 0.0:
        pole_kind = "Drude"
        gamma_seed = drude_gamma_seed(target_chi, omega, dt_s)
        omega_0 = 0.0
    elif target_chi.real > 0.0 and target_chi.imag > 0.0:
        pole_kind = "Lorentz"
        gamma_seed, omega_0 = lorentz_gamma_seed(target_chi, omega, dt_s)
    else:
        raise RuntimeError(
            f"unsupported passive one-frequency target for {material_axis}: "
            f"{target_epsilon!r}"
        )
    local, _, _ = single_pole_scan(
        target_chi,
        omega,
        dt_s,
        ratio_range=LOCAL_RATIO_RANGE,
        points=LOCAL_POINTS,
        selection="current_phase_score",
        gamma_seed=gamma_seed,
        omega_0_rad_s=omega_0,
    )
    wide, states, ratios = single_pole_scan(
        target_chi,
        omega,
        dt_s,
        ratio_range=WIDE_RATIO_RANGE,
        points=WIDE_POINTS,
        selection="minimum_realized_error",
        gamma_seed=gamma_seed,
        omega_0_rad_s=omega_0,
    )
    candidate = stable_two_drude_candidate(
        target_chi,
        dt_s,
        states,
        ratios,
        pole_kind=pole_kind,
    )
    return {
        "material_axis": material_axis,
        "target_epsilon": [target_epsilon.real, target_epsilon.imag],
        "pole_kind": pole_kind,
        "omega_0_rad_s": omega_0,
        "current_single_pole_refit": local,
        "wide_single_pole_scan": wide,
        "stable_two_pole_candidate": candidate,
    }


def analyze_z_factor(z_factor: int, target_epsilon: complex) -> dict[str, Any]:
    """Backward-compatible Au view used by the first diagnostic/tests."""

    analysis = analyze_material_axis(z_factor, "au", target_epsilon)
    return {
        "z_factor": z_factor,
        "cfl": realized_float32_cfl(z_factor),
        "current_single_drude_refit": analysis["current_single_pole_refit"],
        "wide_single_drude_scan": analysis["wide_single_pole_scan"],
        "stable_two_drude_candidate": analysis["stable_two_pole_candidate"],
    }


def _load_bound_input(path: Path, expected_sha256: str | None) -> tuple[dict, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"missing input {resolved}")
    observed_sha = file_sha256(resolved)
    if expected_sha256 is not None and observed_sha != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {resolved}: {observed_sha} != {expected_sha256}"
        )
    return json.loads(resolved.read_text(encoding="utf-8")), observed_sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--z16-case-contract", type=Path, required=True)
    parser.add_argument("--z16-case-contract-sha256")
    parser.add_argument("--z16-failure-json", type=Path, required=True)
    parser.add_argument("--z16-failure-json-sha256")
    parser.add_argument("--fdtdx-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.expanduser()
    if not output.is_absolute():
        parser.error("--output must be absolute")
    output = output.resolve()
    if not output.parent.is_dir() or output.exists():
        parser.error("output parent must exist and output must not exist")

    case, case_sha = _load_bound_input(
        args.z16_case_contract, args.z16_case_contract_sha256
    )
    failure, failure_sha = _load_bound_input(
        args.z16_failure_json, args.z16_failure_json_sha256
    )
    if case.get("mesh_spec", {}).get("z_factor") != 16:
        raise RuntimeError("supplied case is not the z16 full-domain-z contract")
    if case.get("time_spec", {}).get("courant_factor") != COURANT_FACTOR:
        raise RuntimeError("supplied z16 case does not use Courant 0.25")
    if failure.get("status") != "BLOCKED_FDTDX_FRESH_SOURCE_ONLY_EXCEPTION":
        raise RuntimeError("supplied report is not a blocked source-only exception")
    if failure.get("case_contract_expected_sha256") != case_sha:
        raise RuntimeError("z16 failure is not bound to the supplied case-file bytes")
    expected_error = "realized float32 ADE refit error"
    if expected_error not in str(failure.get("error")):
        raise RuntimeError("z16 failure is not the expected float32 ADE blocker")

    fdtdx_source = args.fdtdx_source.expanduser().resolve()
    update = fdtdx_source / "src/fdtdx/fdtd/update.py"
    dispersion = fdtdx_source / "src/fdtdx/dispersion.py"
    if not update.is_file() or not dispersion.is_file():
        raise RuntimeError("--fdtdx-source is not a complete pinned FDTDX tree")

    material_epsilon = load_material_epsilon()
    levels = [
        analyze_z_factor(factor, material_epsilon["au"])
        for factor in Z_FACTORS
    ]
    material_levels = [
        {
            "z_factor": factor,
            "cfl": realized_float32_cfl(factor),
            "materials": {
                name: analyze_material_axis(factor, name, epsilon)
                for name, epsilon in material_epsilon.items()
            },
        }
        for factor in Z_FACTORS
    ]
    by_factor = {int(item["z_factor"]): item for item in material_levels}
    z8_materials = by_factor[8]["materials"]
    z16_materials = by_factor[16]["materials"]
    z32_materials = by_factor[32]["materials"]
    checks = {
        "z8_all_current_single_poles_pass": all(
            item["current_single_pole_refit"]["fit_gate_passed"]
            for item in z8_materials.values()
        ),
        "z16_current_Au_single_drude_fails": not bool(
            z16_materials["au"]["current_single_pole_refit"]["fit_gate_passed"]
        ),
        "z16_current_Ta_a_single_drude_fails": not bool(
            z16_materials["a"]["current_single_pole_refit"]["fit_gate_passed"]
        ),
        "z16_current_Ta_b_c_single_lorentz_pass": all(
            z16_materials[axis]["current_single_pole_refit"]["fit_gate_passed"]
            for axis in ("b", "c")
        ),
        "z32_all_current_single_poles_fail": all(
            not item["current_single_pole_refit"]["fit_gate_passed"]
            for item in z32_materials.values()
        ),
        "z8_z16_z32_all_stable_two_pole_candidates_pass": all(
            item["stable_two_pole_candidate"].get("fit_gate_passed", False)
            for level in material_levels
            if level["z_factor"] in (8, 16, 32)
            for item in level["materials"].values()
        ),
        "field_solve_never_started": failure.get("ready") is False,
    }
    ready = all(checks.values())
    payload = {
        "status": (
            "DIAGNOSED_Z16_FULL_MATERIAL_FLOAT32_ADE_BLOCKER"
            if ready
            else "BLOCKED_INCOMPLETE_Z16_ADE_DIAGNOSIS"
        ),
        "ready": ready,
        "scope": "solver-free carrier ADE precision diagnostic; no FDTD solve",
        "target": {
            "wavelength_m": WAVELENGTH_M,
            "material_epsilon": {
                name: [epsilon.real, epsilon.imag]
                for name, epsilon in material_epsilon.items()
            },
            "fit_relative_tolerance": FIT_RELATIVE_TOLERANCE,
        },
        "bound_inputs": {
            "z16_case_contract": {
                "path": str(args.z16_case_contract.expanduser().resolve()),
                "sha256": case_sha,
                "canonical_sha256": case.get("case_contract_sha256"),
            },
            "z16_failure_json": {
                "path": str(args.z16_failure_json.expanduser().resolve()),
                "sha256": failure_sha,
                "recorded_error": failure.get("error"),
            },
        },
        "provenance": {
            "diagnostic_script_sha256": file_sha256(Path(__file__).resolve()),
            "optical_model_sha256": file_sha256(OPTICAL_MODEL),
            "material_contract_sha256": file_sha256(MATERIAL_CONTRACT),
            "fdtdx_update_sha256": file_sha256(update),
            "fdtdx_dispersion_sha256": file_sha256(dispersion),
        },
        "checks": checks,
        "levels": levels,
        "material_levels": material_levels,
        "inference": {
            "do_not_relax_material_fit_gate": True,
            "do_not_retry_z16_with_current_single_pole_model": True,
            "two_pole_representation_is_only_a_numerical_candidate": True,
            "material_law_change_invalidates_direct_old_z8_new_z16_comparison": True,
            "required_next_validation": [
                "implement candidate behind a separately hashed material-law contract",
                "prove exact float32 c1/c2/c3 readback for both poles on every material axis",
                "rerun source-only and time/stationarity gates",
                "rerun z8, z16, and z32 with the identical two-pole algorithm",
                "do not issue a mesh certificate before a successive same-law pair passes",
            ],
        },
        "promotion": {
            "is_material_certificate": False,
            "is_mesh_certificate": False,
            "optimizer_start_allowed": False,
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
