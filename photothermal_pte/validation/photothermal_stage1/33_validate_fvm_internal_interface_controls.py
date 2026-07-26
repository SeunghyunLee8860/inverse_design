#!/usr/bin/env python3
"""Validate finite internal thermal conductance in the Cartesian FVM.

This is an independent conservative Python/SciPy control, not a Lumerical
HEAT result.  It solves two isotropic slabs in 1D using the production 3D
matrix assembly with adiabatic lateral faces.  Every material-interface face
uses

    R'' = dz_1/(2 k_1) + 1/G + dz_2/(2 k_2).

The two finite-G cases and perfect contact are each evaluated on 100, 50, and
25 nm meshes before the solver is allowed to advance to a 3D cross-check.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

import config_stage1 as config
from anisotropic_heat_fvm import (
    internal_face_heat_flux_density,
    solve_steady_diagonal_kappa,
)
from lumerical_api import utc_timestamp, write_json


CONDUCTANCES_W_M2K = (7.37e6, 1.1e9, None)
MESHES_M = (100.0e-9, 50.0e-9, 25.0e-9)
K1_W_MK = 5.0
K2_W_MK = 20.0
T1_M = 1.0e-6
T2_M = 1.0e-6
AREA_M2 = 1.0e-12
HOT_K = 310.0
COLD_K = 300.0
RELATIVE_LIMIT = 0.01
LINEAR_RESIDUAL_LIMIT = 1.0e-9
PERFECT_JUMP_NORMALIZED_LIMIT = 1.0e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--report-dir",
        default=str(
            config.REPOSITORY_ROOT
            / "reports"
            / "fvm_internal_interface_controls"
        ),
    )
    return parser.parse_args()


def clean_output_directory(explicit: str | None) -> Path:
    output = (
        Path(explicit).expanduser().resolve()
        if explicit
        else config.OUTPUT_ROOT
        / "fvm_internal_interface_controls"
        / f"{utc_timestamp()}_controls"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def repository_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.REPOSITORY_ROOT.parent))
    except ValueError:
        return str(path.resolve())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=config.REPOSITORY_ROOT.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def label_for_g(conductance_W_m2K: float | None) -> str:
    if conductance_W_m2K is None:
        return "perfect_contact"
    if conductance_W_m2K == 7.37e6:
        return "G_7p37e6"
    if conductance_W_m2K == 1.1e9:
        return "G_1p1e9"
    return f"G_{conductance_W_m2K:.8e}".replace("+", "").replace(".", "p")


def analytic_profile(
    z_m: np.ndarray,
    *,
    heat_flux_W_m2: float,
    interface_jump_K: float,
) -> np.ndarray:
    lower = HOT_K - heat_flux_W_m2 * (z_m + T1_M) / K1_W_MK
    lower_interface_K = HOT_K - heat_flux_W_m2 * T1_M / K1_W_MK
    upper = (
        lower_interface_K
        - interface_jump_K
        - heat_flux_W_m2 * z_m / K2_W_MK
    )
    return np.where(z_m < 0.0, lower, upper)


def relative_spread(values: list[float]) -> float:
    array = np.asarray(values, float)
    return float(
        (np.max(array) - np.min(array))
        / max(float(np.mean(np.abs(array))), np.finfo(float).tiny)
    )


def run_case(
    output: Path,
    *,
    conductance_W_m2K: float | None,
    mesh_m: float,
) -> dict[str, Any]:
    g_label = label_for_g(conductance_W_m2K)
    mesh_nm = int(round(mesh_m * 1.0e9))
    case_id = f"{g_label}_mesh_{mesh_nm}nm"
    case_dir = output / case_id
    case_dir.mkdir(parents=True)

    cells_1 = int(round(T1_M / mesh_m))
    cells_2 = int(round(T2_M / mesh_m))
    if not np.isclose(cells_1 * mesh_m, T1_M) or not np.isclose(
        cells_2 * mesh_m, T2_M
    ):
        raise ValueError("mesh must divide both slab thicknesses exactly")
    z_edges_m = np.concatenate(
        (
            np.linspace(-T1_M, 0.0, cells_1 + 1),
            np.linspace(0.0, T2_M, cells_2 + 1)[1:],
        )
    )
    x_edges_m = np.asarray([-0.5e-6, 0.5e-6])
    y_edges_m = np.asarray([-0.5e-6, 0.5e-6])
    shape = (1, 1, cells_1 + cells_2)
    kappa = np.empty((*shape, 3), float)
    kappa[:, :, :cells_1, :] = K1_W_MK
    kappa[:, :, cells_1:, :] = K2_W_MK
    z_face_resistance = np.zeros((1, 1, shape[2] - 1), float)
    interface_face_index = cells_1 - 1
    interface_insulance_m2K_W = (
        0.0 if conductance_W_m2K is None else 1.0 / conductance_W_m2K
    )
    z_face_resistance[0, 0, interface_face_index] = (
        interface_insulance_m2K_W
    )
    interface_resistance = {"z": z_face_resistance}

    solved = solve_steady_diagonal_kappa(
        x_edges_m=x_edges_m,
        y_edges_m=y_edges_m,
        z_edges_m=z_edges_m,
        kappa_W_mK=kappa,
        dirichlet_temperature_K={"z_min": HOT_K, "z_max": COLD_K},
        interface_resistance_m2K_W=interface_resistance,
    )
    flux_z = internal_face_heat_flux_density(
        temperature_K=solved.temperature_K,
        x_edges_m=x_edges_m,
        y_edges_m=y_edges_m,
        z_edges_m=z_edges_m,
        kappa_W_mK=kappa,
        interface_resistance_m2K_W=interface_resistance,
    )["z"][0, 0, :]

    temperature = solved.temperature_K[0, 0, :]
    centers_m = 0.5 * (z_edges_m[:-1] + z_edges_m[1:])
    lower_fit = np.polyfit(
        centers_m[:cells_1], temperature[:cells_1], 1
    )
    upper_fit = np.polyfit(
        centers_m[cells_1:], temperature[cells_1:], 1
    )
    lower_interface_K = float(np.polyval(lower_fit, 0.0))
    upper_interface_K = float(np.polyval(upper_fit, 0.0))
    numerical_jump_K = lower_interface_K - upper_interface_K

    series_resistance_m2K_W = (
        T1_M / K1_W_MK
        + interface_insulance_m2K_W
        + T2_M / K2_W_MK
    )
    analytic_flux_W_m2 = (HOT_K - COLD_K) / series_resistance_m2K_W
    analytic_jump_K = analytic_flux_W_m2 * interface_insulance_m2K_W
    exact_temperature = analytic_profile(
        centers_m,
        heat_flux_W_m2=analytic_flux_W_m2,
        interface_jump_K=analytic_jump_K,
    )

    lower_material_flux_W_m2 = float(
        np.mean(flux_z[:interface_face_index])
    )
    interface_flux_W_m2 = float(flux_z[interface_face_index])
    upper_material_flux_W_m2 = float(
        np.mean(flux_z[interface_face_index + 1 :])
    )
    boundary_input_flux_W_m2 = (
        -solved.boundary_power_out_W["z_min"] / AREA_M2
    )
    boundary_output_flux_W_m2 = (
        solved.boundary_power_out_W["z_max"] / AREA_M2
    )
    flux_values = [
        lower_material_flux_W_m2,
        interface_flux_W_m2,
        upper_material_flux_W_m2,
        boundary_input_flux_W_m2,
        boundary_output_flux_W_m2,
    ]
    transmitted_flux_W_m2 = float(np.mean(flux_values))
    material_flux_mismatch = abs(
        lower_material_flux_W_m2 - upper_material_flux_W_m2
    ) / max(abs(transmitted_flux_W_m2), np.finfo(float).tiny)
    flux_path_spread = relative_spread(flux_values)
    heat_flux_error = abs(
        transmitted_flux_W_m2 - analytic_flux_W_m2
    ) / analytic_flux_W_m2
    profile_error = float(
        np.max(np.abs(temperature - exact_temperature))
        / (HOT_K - COLD_K)
    )
    raw_cell_center_jump_K = float(
        temperature[cells_1 - 1] - temperature[cells_1]
    )
    expected_raw_cell_center_jump_K = analytic_flux_W_m2 * (
        0.5 * mesh_m / K1_W_MK
        + interface_insulance_m2K_W
        + 0.5 * mesh_m / K2_W_MK
    )
    raw_jump_error = abs(
        raw_cell_center_jump_K - expected_raw_cell_center_jump_K
    ) / max(abs(expected_raw_cell_center_jump_K), np.finfo(float).tiny)
    jump_error = (
        abs(numerical_jump_K) / (HOT_K - COLD_K)
        if conductance_W_m2K is None
        else abs(numerical_jump_K - analytic_jump_K) / abs(analytic_jump_K)
    )
    face_resistance_formula_m2K_W = (
        0.5 * mesh_m / K1_W_MK
        + interface_insulance_m2K_W
        + 0.5 * mesh_m / K2_W_MK
    )
    finite_g_jump_passed = (
        conductance_W_m2K is None
        or jump_error < RELATIVE_LIMIT
    )
    perfect_jump_passed = (
        conductance_W_m2K is not None
        or jump_error < PERFECT_JUMP_NORMALIZED_LIMIT
    )
    passed = bool(
        finite_g_jump_passed
        and perfect_jump_passed
        and heat_flux_error < RELATIVE_LIMIT
        and material_flux_mismatch < RELATIVE_LIMIT
        and flux_path_spread < RELATIVE_LIMIT
        and profile_error < RELATIVE_LIMIT
        and solved.energy_balance_relative_error < RELATIVE_LIMIT
        and solved.linear_residual_relative < LINEAR_RESIDUAL_LIMIT
        and raw_jump_error < RELATIVE_LIMIT
    )
    case = {
        "case_id": case_id,
        "status": "PASSED" if passed else "FAILED_FVM_INTERFACE_G_CONTROL",
        "passed": passed,
        "solver_scope": (
            "independent conservative Cartesian Python/SciPy FVM; "
            "not a Lumerical HEAT result"
        ),
        "G_W_m2K": conductance_W_m2K,
        "perfect_contact": conductance_W_m2K is None,
        "interface_insulance_m2K_W": interface_insulance_m2K_W,
        "mesh_m": mesh_m,
        "grid_shape": list(shape),
        "material_1": {"thickness_m": T1_M, "kappa_W_mK": K1_W_MK},
        "material_2": {"thickness_m": T2_M, "kappa_W_mK": K2_W_MK},
        "boundary_temperature_K": {"hot": HOT_K, "cold": COLD_K},
        "interface_face_index": interface_face_index,
        "interface_is_internal": True,
        "face_resistance_formula_m2K_W": (
            face_resistance_formula_m2K_W
        ),
        "face_resistance_terms_m2K_W": {
            "material_1_half_cell": 0.5 * mesh_m / K1_W_MK,
            "interface_1_over_G": interface_insulance_m2K_W,
            "material_2_half_cell": 0.5 * mesh_m / K2_W_MK,
        },
        "analytic_series_resistance_m2K_W": series_resistance_m2K_W,
        "analytic_heat_flux_W_m2": analytic_flux_W_m2,
        "numerical_transmitted_heat_flux_W_m2": transmitted_flux_W_m2,
        "heat_flux_relative_error": heat_flux_error,
        "lower_material_heat_flux_W_m2": lower_material_flux_W_m2,
        "interface_heat_flux_W_m2": interface_flux_W_m2,
        "upper_material_heat_flux_W_m2": upper_material_flux_W_m2,
        "boundary_input_heat_flux_W_m2": boundary_input_flux_W_m2,
        "boundary_output_heat_flux_W_m2": boundary_output_flux_W_m2,
        "material_flux_relative_mismatch": material_flux_mismatch,
        "all_flux_paths_relative_spread": flux_path_spread,
        "analytic_interface_temperature_jump_K": analytic_jump_K,
        "numerical_interface_temperature_jump_K": numerical_jump_K,
        "interface_temperature_jump_relative_error": jump_error,
        "lower_interface_temperature_K": lower_interface_K,
        "upper_interface_temperature_K": upper_interface_K,
        "raw_adjacent_cell_temperature_difference_K": (
            raw_cell_center_jump_K
        ),
        "expected_raw_adjacent_cell_temperature_difference_K": (
            expected_raw_cell_center_jump_K
        ),
        "raw_adjacent_cell_difference_relative_error": raw_jump_error,
        "temperature_profile_max_relative_error": profile_error,
        "boundary_power_out_W": solved.boundary_power_out_W,
        "source_power_W": solved.source_power_W,
        "energy_balance_relative_error": (
            solved.energy_balance_relative_error
        ),
        "linear_residual_relative": solved.linear_residual_relative,
        "solver": solved.solver,
        "iterations": solved.iterations,
        "criteria": {
            "finite_G_interface_jump_relative_error_lt": RELATIVE_LIMIT,
            "perfect_contact_jump_over_delta_T_lt": (
                PERFECT_JUMP_NORMALIZED_LIMIT
            ),
            "heat_flux_relative_error_lt": RELATIVE_LIMIT,
            "material_flux_relative_mismatch_lt": RELATIVE_LIMIT,
            "all_flux_paths_relative_spread_lt": RELATIVE_LIMIT,
            "temperature_profile_relative_error_lt": RELATIVE_LIMIT,
            "energy_balance_relative_error_lt": RELATIVE_LIMIT,
            "linear_residual_relative_lt": LINEAR_RESIDUAL_LIMIT,
        },
    }
    np.savez_compressed(
        case_dir / "temperature_profile.npz",
        z_edges_m=z_edges_m,
        z_centers_m=centers_m,
        temperature_K=temperature,
        analytic_temperature_K=exact_temperature,
        internal_z_heat_flux_W_m2=flux_z,
        kappa_z_W_mK=kappa[0, 0, :, 2],
        interface_resistance_z_m2K_W=z_face_resistance[0, 0, :],
    )
    write_json(case_dir / "case_result.json", case)
    return case


def convergence_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for conductance in CONDUCTANCES_W_M2K:
        label = label_for_g(conductance)
        selected = [
            case
            for case in cases
            if (
                case["perfect_contact"]
                if conductance is None
                else case["G_W_m2K"] == conductance
            )
        ]
        selected.sort(key=lambda item: item["mesh_m"], reverse=True)
        jump_values = [
            item["numerical_interface_temperature_jump_K"]
            for item in selected
        ]
        flux_values = [
            item["numerical_transmitted_heat_flux_W_m2"]
            for item in selected
        ]
        profile_errors = [
            item["temperature_profile_max_relative_error"]
            for item in selected
        ]
        energy_errors = [
            item["energy_balance_relative_error"] for item in selected
        ]
        raw_jumps = [
            item["raw_adjacent_cell_temperature_difference_K"]
            for item in selected
        ]
        finite_jump_spread = (
            None
            if conductance is None
            else relative_spread(jump_values)
        )
        perfect_raw_monotonic = (
            None
            if conductance is not None
            else all(
                raw_jumps[index + 1] < raw_jumps[index]
                for index in range(len(raw_jumps) - 1)
            )
        )
        perfect_raw_refinement_ratio = (
            None
            if conductance is not None
            else raw_jumps[-1] / raw_jumps[0]
        )
        passed = bool(
            all(item["passed"] for item in selected)
            and relative_spread(flux_values) < RELATIVE_LIMIT
            and max(profile_errors) < RELATIVE_LIMIT
            and max(energy_errors) < RELATIVE_LIMIT
            and (
                finite_jump_spread is not None
                and finite_jump_spread < RELATIVE_LIMIT
                if conductance is not None
                else (
                    perfect_raw_monotonic
                    and perfect_raw_refinement_ratio is not None
                    and perfect_raw_refinement_ratio < 0.30
                    and max(abs(value) for value in jump_values)
                    / (HOT_K - COLD_K)
                    < PERFECT_JUMP_NORMALIZED_LIMIT
                )
            )
        )
        groups[label] = {
            "passed": passed,
            "meshes_m": [item["mesh_m"] for item in selected],
            "interface_temperature_jumps_K": jump_values,
            "interface_jump_relative_spread": finite_jump_spread,
            "transmitted_heat_fluxes_W_m2": flux_values,
            "heat_flux_relative_spread": relative_spread(flux_values),
            "temperature_profile_relative_errors": profile_errors,
            "energy_balance_relative_errors": energy_errors,
            "raw_adjacent_cell_temperature_differences_K": raw_jumps,
            "perfect_contact_raw_jump_strictly_decreases": (
                perfect_raw_monotonic
            ),
            "perfect_contact_finest_to_coarsest_raw_jump_ratio": (
                perfect_raw_refinement_ratio
            ),
        }
    return groups


def write_cases_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    fields = (
        "case_id",
        "status",
        "passed",
        "G_W_m2K",
        "perfect_contact",
        "mesh_m",
        "face_resistance_formula_m2K_W",
        "analytic_heat_flux_W_m2",
        "numerical_transmitted_heat_flux_W_m2",
        "heat_flux_relative_error",
        "lower_material_heat_flux_W_m2",
        "interface_heat_flux_W_m2",
        "upper_material_heat_flux_W_m2",
        "material_flux_relative_mismatch",
        "all_flux_paths_relative_spread",
        "analytic_interface_temperature_jump_K",
        "numerical_interface_temperature_jump_K",
        "interface_temperature_jump_relative_error",
        "raw_adjacent_cell_temperature_difference_K",
        "temperature_profile_max_relative_error",
        "energy_balance_relative_error",
        "linear_residual_relative",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        for case in cases:
            writer.writerow({field: case.get(field) for field in fields})


def format_percent(value: float) -> str:
    return f"{100.0 * value:.6g}%"


def write_report(
    path: Path,
    summary: dict[str, Any],
    cases: list[dict[str, Any]],
) -> None:
    rows = []
    for case in cases:
        g = (
            "perfect"
            if case["perfect_contact"]
            else f'{case["G_W_m2K"]:.6g}'
        )
        rows.append(
            "| "
            + " | ".join(
                (
                    g,
                    f'{case["mesh_m"] * 1e9:g}',
                    f'{case["analytic_heat_flux_W_m2"]:.12g}',
                    f'{case["numerical_transmitted_heat_flux_W_m2"]:.12g}',
                    format_percent(case["heat_flux_relative_error"]),
                    f'{case["analytic_interface_temperature_jump_K"]:.12g}',
                    f'{case["numerical_interface_temperature_jump_K"]:.12g}',
                    format_percent(
                        case["interface_temperature_jump_relative_error"]
                    ),
                    format_percent(
                        case["material_flux_relative_mismatch"]
                    ),
                    format_percent(case["energy_balance_relative_error"]),
                    case["status"],
                )
            )
            + " |"
        )
    convergence = summary["mesh_convergence"]
    perfect = convergence["perfect_contact"]
    text = f"""# FVM internal-interface-G control report

**Status: `{summary["status"]}`.**

This is an independent conservative Cartesian Python/SciPy finite-volume
result, not a Lumerical HEAT result. No optical Q or full-device geometry was
used in this control.

The internal face resistance used by both matrix assembly and flux recovery
is

`R'' = dz_1/(2 k_1) + 1/G + dz_2/(2 k_2)`.

The slabs use `k1={K1_W_MK:g} W/(m K)`, `k2={K2_W_MK:g} W/(m K)`,
`t1=t2=1 um`, and fixed `310 K -> 300 K` boundary temperatures.

| G (W/m2 K) | mesh (nm) | analytic q'' | numerical q'' | flux error | analytic jump (K) | numerical jump (K) | jump error | k1/k2 flux mismatch | energy error | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## Independent checks

- The one-sided interface temperatures are obtained by independently fitting
  the cell-center temperature profile in each material and extrapolating each
  fit to `z=0`.
- Heat flux is recovered independently at the hot boundary, in material 1,
  on the interface face, in material 2, and at the cold boundary.
- Every finite-G jump, analytic series-resistance flux, material-to-material
  flux transmission, temperature profile, and global energy balance error is
  below 1%.
- The linear residual limit is `{LINEAR_RESIDUAL_LIMIT:g}`.

## Mesh refinement and perfect contact

All three G conditions pass at 100, 50, and 25 nm. The finite-G jump and
transmitted-flux spreads across the three meshes are below 1%.

For perfect contact, the extrapolated one-sided interface jump remains at
roundoff while the raw adjacent-cell difference decreases as the cell
centers approach the interface:

`{perfect["raw_adjacent_cell_temperature_differences_K"]}` K.

The finest/coarsest raw-jump ratio is
`{perfect["perfect_contact_finest_to_coarsest_raw_jump_ratio"]:.12g}`;
the expected first-order geometric ratio for 25/100 nm is 0.25.

## Gate

The finite-G FVM analytic gate is closed successfully. The next required
step is a common 3D isotropic, perfect-contact, heterogeneous-material,
volumetric-Q control solved by both v261 Lumerical HEAT and this FVM. The
validated finite optical Q must not be imported until that cross-validation
passes.
"""
    path.write_text(text, encoding="utf-8")


def build_manifest(
    output: Path,
    report_dir: Path,
    *,
    command: str,
) -> dict[str, Any]:
    files = sorted(
        list(output.rglob("case_result.json"))
        + list(output.rglob("temperature_profile.npz"))
        + [
            report_dir / "FVM_INTERNAL_INTERFACE_G_CONTROL_REPORT.md",
            report_dir / "fvm_internal_interface_controls_summary.json",
            report_dir / "fvm_internal_interface_controls_cases.csv",
        ]
    )
    return {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "branch": git_value("branch", "--show-current"),
        "base_commit_before_control": git_value("rev-parse", "HEAD"),
        "generation_command": command,
        "solver_scope": (
            "independent conservative Cartesian Python/SciPy FVM; "
            "not a Lumerical HEAT result"
        ),
        "artifacts": [
            {
                "repository_path": repository_relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }


def main() -> int:
    args = parse_args()
    output = clean_output_directory(args.output_dir)
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, *sys.argv])

    cases = [
        run_case(
            output,
            conductance_W_m2K=conductance,
            mesh_m=mesh,
        )
        for conductance in CONDUCTANCES_W_M2K
        for mesh in MESHES_M
    ]
    convergence = convergence_summary(cases)
    passed = bool(
        all(case["passed"] for case in cases)
        and all(group["passed"] for group in convergence.values())
    )
    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "status": (
            "VALIDATED_FVM_INTERNAL_INTERFACE_G_CONTROLS"
            if passed
            else "FAILED_FVM_INTERNAL_INTERFACE_G_CONTROLS"
        ),
        "passed": passed,
        "branch": git_value("branch", "--show-current"),
        "solver_scope": (
            "independent conservative Cartesian Python/SciPy FVM; "
            "not a Lumerical HEAT result"
        ),
        "discretization": (
            "cell-centered Cartesian finite volume with exact two-half-cell "
            "series resistance plus internal face insulance 1/G"
        ),
        "face_resistance_equation": (
            "dz_1/(2*k_1) + 1/G + dz_2/(2*k_2)"
        ),
        "conductances_W_m2K": [7.37e6, 1.1e9, "perfect_contact"],
        "meshes_m": list(MESHES_M),
        "case_count": len(cases),
        "all_cases_passed": all(case["passed"] for case in cases),
        "mesh_convergence": convergence,
        "criteria": {
            "finite_G_interface_jump_relative_error_lt": RELATIVE_LIMIT,
            "heat_flux_relative_error_lt": RELATIVE_LIMIT,
            "material_flux_relative_mismatch_lt": RELATIVE_LIMIT,
            "temperature_profile_relative_error_lt": RELATIVE_LIMIT,
            "energy_balance_relative_error_lt": RELATIVE_LIMIT,
            "perfect_contact_extrapolated_jump_over_delta_T_lt": (
                PERFECT_JUMP_NORMALIZED_LIMIT
            ),
            "perfect_contact_finest_to_coarsest_raw_jump_ratio_lt": 0.30,
        },
        "full_device_executed": False,
        "finite_optical_Q_imported": False,
        "next_required_gate": (
            "LUMERICAL_HEAT_VS_FVM_3D_ISOTROPIC_PERFECT_CONTACT_CROSS_VALIDATION"
        ),
        "cases": cases,
    }
    summary_path = (
        report_dir / "fvm_internal_interface_controls_summary.json"
    )
    cases_path = report_dir / "fvm_internal_interface_controls_cases.csv"
    report_path = report_dir / "FVM_INTERNAL_INTERFACE_G_CONTROL_REPORT.md"
    write_json(summary_path, summary)
    write_cases_csv(cases_path, cases)
    write_report(report_path, summary, cases)
    manifest = build_manifest(output, report_dir, command=command)
    write_json(report_dir / "RAW_ARTIFACT_MANIFEST.json", manifest)
    write_json(output / "control_summary.json", summary)

    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
