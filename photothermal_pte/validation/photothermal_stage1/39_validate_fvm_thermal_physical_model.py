#!/usr/bin/env python3
"""Validate physical-model scenarios around the converged FVM checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import config_stage1 as config
from lumerical_api import utc_timestamp, write_json


SCRIPT = Path(__file__).with_name("36_run_fvm_multimaterial_thermal.py")
EXPECTED_OPTICAL_SHA256 = (
    "7a63f82842751e7623e895701bac4ce92558679ed71bf13a3d404695a150e794"
)
EXPECTED_POWER_W = 2.56071371086521e-12
REFERENCE_CASE_ID = "scenario_control_Gtop_7p37e6_kz_1"


@dataclass(frozen=True)
class Case:
    case_id: str
    family: str
    physical_basis: str
    arguments: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-artifact",
        default=str(
            config.OUTPUT_ROOT
            / "fvm_finite_q_import"
            / "import_v4"
            / "finite_q_exact_flake_source.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            config.OUTPUT_ROOT / "fvm_thermal_physical_model" / "sweep_v1"
        ),
    )
    parser.add_argument(
        "--phase", choices=("material", "boundary", "all"), default="all"
    )
    parser.add_argument("--report-dir")
    return parser.parse_args()


def common_arguments() -> tuple[str, ...]:
    return (
        "--lateral-domain-um",
        "32",
        "--si-depth-um",
        "20",
        "--G-bottom",
        "7.37e6",
        "--G-oxide-si",
        "1.1e9",
        "--far-xy-boundary",
        "fixed",
        "--top-disk-support",
        "suspended-overhang",
        "--exposed-h-W-m2K",
        "0",
    )


def cases() -> list[Case]:
    common = common_arguments()
    return [
        Case(
            REFERENCE_CASE_ID,
            "G_top",
            (
                "PR #4 numerical-convergence checkpoint parameter set; "
                "not promoted as uniquely correct physics"
            ),
            (
                *common,
                "--G-top",
                "7.37e6",
                "--tairte4-kz-W-mK",
                "1",
            ),
        ),
        Case(
            "scenario_evaporated_SiO2_estimate_Gtop_7p37e4",
            "G_top",
            (
                "earlier contract label: evaporated-SiO2 estimate; "
                "repository does not contain a traceable literature source"
            ),
            (
                *common,
                "--G-top",
                "7.37e4",
                "--tairte4-kz-W-mK",
                "1",
            ),
        ),
        Case(
            "scenario_kz_0p5",
            "TaIrTe4_kz",
            (
                "numerical scenario, not a confidence interval; no "
                "repository source establishes a kz range"
            ),
            (
                *common,
                "--G-top",
                "7.37e6",
                "--tairte4-kz-W-mK",
                "0.5",
            ),
        ),
        Case(
            "scenario_kz_2p0",
            "TaIrTe4_kz",
            (
                "numerical scenario, not a confidence interval; no "
                "repository source establishes a kz range"
            ),
            (
                *common,
                "--G-top",
                "7.37e6",
                "--tairte4-kz-W-mK",
                "2",
            ),
        ),
        Case(
            "scenario_far_xy_adiabatic_bottom_fixed",
            "far_xy_boundary",
            (
                "boundary-robustness scenario; lateral boundary flux is a "
                "numerical truncation flux, not a physical heat-path fraction"
            ),
            (
                *common,
                "--G-top",
                "7.37e6",
                "--tairte4-kz-W-mK",
                "1",
                "--far-xy-boundary",
                "adiabatic",
            ),
        ),
        *[
            Case(
                f"scenario_exposed_convection_h{h}",
                "exposed_convection",
                "exposed-surface Robin boundary robustness scenario",
                (
                    *common,
                    "--G-top",
                    "7.37e6",
                    "--tairte4-kz-W-mK",
                    "1",
                    "--exposed-h-W-m2K",
                    str(h),
                ),
            )
            for h in (5, 10, 20)
        ],
        Case(
            "scenario_geometry_oxide_supported_overhang",
            "top_disk_support",
            (
                "alternative thermal geometry B: SiO2 fills the 100 nm "
                "annulus below the disk outside the flake; fabrication is "
                "not confirmed by repository evidence"
            ),
            (
                *common,
                "--G-top",
                "7.37e6",
                "--tairte4-kz-W-mK",
                "1",
                "--top-disk-support",
                "oxide-supported-overhang",
            ),
        ),
    ]


def selected_cases(phase: str) -> list[Case]:
    definitions = cases()
    if phase == "material":
        return [
            item
            for item in definitions
            if item.family in ("G_top", "TaIrTe4_kz")
        ]
    if phase == "boundary":
        return [
            item
            for item in definitions
            if item.case_id == REFERENCE_CASE_ID
            or item.family
            in ("far_xy_boundary", "exposed_convection", "top_disk_support")
        ]
    return definitions


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def execute_case(
    case: Case,
    *,
    source_artifact: Path,
    output_root: Path,
    extra_arguments: tuple[str, ...] = (),
    case_id: str | None = None,
) -> dict[str, Any]:
    actual_case_id = case.case_id if case_id is None else case_id
    output = output_root / actual_case_id
    result_path = output / "case_result.json"
    if result_path.is_file():
        return load_json(result_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"partial non-empty case directory: {output}")
    command = [
        sys.executable,
        str(SCRIPT),
        "--source-artifact",
        str(source_artifact),
        "--output-dir",
        str(output),
        "--case-id",
        actual_case_id,
        "--physical-scenario-label",
        actual_case_id,
        *case.arguments,
        *extra_arguments,
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{actual_case_id} failed ({completed.returncode})\n"
            f"{completed.stdout}"
        )
    return load_json(result_path)


def probe_field(result: dict[str, Any]) -> np.ndarray:
    with np.load(result["raw_field_path"], allow_pickle=False) as raw:
        edges = tuple(
            np.asarray(raw[key], float)
            for key in ("x_edges_m", "y_edges_m", "z_edges_m")
        )
        temperature = np.asarray(raw["delta_T_K_per_W_m2"], float)
    centers = tuple(0.5 * (item[:-1] + item[1:]) for item in edges)
    interpolation = RegularGridInterpolator(
        centers, temperature, method="linear", bounds_error=True
    )
    axes = (
        np.linspace(-0.85e-6, 0.85e-6, 11),
        np.linspace(-0.85e-6, 0.85e-6, 11),
        np.linspace(-90e-9, -10e-9, 7),
    )
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([item.reshape(-1) for item in mesh])
    values = interpolation(points)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("common flake probe field contains NaN or Inf")
    return values


def nrmse(result: dict[str, Any], reference: dict[str, Any]) -> float:
    values = probe_field(result)
    reference_values = probe_field(reference)
    return float(
        np.sqrt(np.mean((values - reference_values) ** 2))
        / max(np.max(np.abs(reference_values)), np.finfo(float).tiny)
    )


def response(result: dict[str, Any]) -> tuple[float, float]:
    values = result["temperature_response"]
    return (
        float(values["DeltaT_max_K_per_W_m2"]),
        float(values["TaIrTe4_volume_average_DeltaT_K_per_W_m2"]),
    )


def relative_change(value: float, reference: float) -> float:
    return (value - reference) / reference


def build_row(
    definition: Case,
    result: dict[str, Any],
    reference: dict[str, Any],
    *,
    mesh_role: str = "native",
) -> dict[str, Any]:
    tmax, average = response(result)
    reference_tmax, reference_average = response(reference)
    top_interface = result["interfaces"]["TaIrTe4_top"]
    hotspot = result["temperature_response"]["hotspot_location_m"]
    truncation = result["power_balance"][
        "numerical_truncation_boundary_flux"
    ]
    return {
        "case_id": result["case_id"],
        "family": definition.family,
        "mesh_role": mesh_role,
        "physical_basis": definition.physical_basis,
        "Tmax_K_per_W_m2": tmax,
        "TaIrTe4_average_K_per_W_m2": average,
        "Tmax_change_vs_control": relative_change(tmax, reference_tmax),
        "average_change_vs_control": relative_change(
            average, reference_average
        ),
        "common_flake_3d_NRMSE_vs_control": nrmse(result, reference),
        "hotspot_x_m": hotspot["x"],
        "hotspot_y_m": hotspot["y"],
        "hotspot_z_m": hotspot["z"],
        "top_interface_mean_jump_K": top_interface[
            "area_weighted_mean_temperature_jump_K"
        ],
        "top_interface_max_jump_K": top_interface[
            "maximum_temperature_jump_K"
        ],
        "bottom_numerical_boundary_flux_fraction": truncation[
            "bottom_fraction_of_generated"
        ],
        "lateral_numerical_boundary_flux_fraction": truncation[
            "lateral_fraction_of_generated"
        ],
        "energy_balance_relative_error": result["power_balance"][
            "relative_error"
        ],
        "linear_residual_relative": result["linear_solver"][
            "relative_residual"
        ],
        "Q_mapping_relative_error": result["source"][
            "mapping_relative_error"
        ],
        "optical_source_SHA256": result["source"][
            "optical_source_artifact_sha256"
        ],
        "mapped_source_power_W": result["source"]["mapped_source_power_W"],
        "source_mesh_mode": result["source"]["thermal_source_mesh_mode"],
        "active_solid_cells": result["grid"]["active_solid_cell_count"],
        "G_top_W_m2K": top_interface["G_W_m2K"],
        "TaIrTe4_kz_W_mK": result["materials_W_mK"][
            "TaIrTe4_diagonal"
        ][2],
        "far_xy_boundary": result["boundary_conditions"][
            "far_x_y_boundary_mode"
        ],
        "exposed_h_W_m2K": result["boundary_conditions"][
            "exposed_heat_transfer_W_m2K"
        ],
        "top_disk_support": result["geometry"]["design"][
            "thermal_support_scenario"
        ],
        "passed": result["passed"],
        "raw_result_path": str(
            Path(result["raw_field_path"]).with_name("case_result.json")
        ),
        "raw_field_path": result["raw_field_path"],
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    source_artifact = Path(args.source_artifact).expanduser().resolve()
    if not source_artifact.is_file():
        raise FileNotFoundError(source_artifact)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    definitions = selected_cases(args.phase)
    results: dict[str, dict[str, Any]] = {}
    for index, definition in enumerate(definitions, start=1):
        print(f"[{index}/{len(definitions)}] {definition.case_id}", flush=True)
        results[definition.case_id] = execute_case(
            definition,
            source_artifact=source_artifact,
            output_root=output_root,
        )
    reference = results[REFERENCE_CASE_ID]
    rows = [
        build_row(item, results[item.case_id], reference)
        for item in definitions
    ]

    refined = None
    mesh_comparison = None
    maximum_definition = None
    if args.phase == "all":
        nonreference = [
            item for item in definitions if item.case_id != REFERENCE_CASE_ID
        ]
        maximum_definition = max(
            nonreference,
            key=lambda item: abs(
                relative_change(
                    response(results[item.case_id])[0],
                    response(reference)[0],
                )
            ),
        )
        refined_case_id = f"{maximum_definition.case_id}_refined"
        print(f"[refined] {refined_case_id}", flush=True)
        refined = execute_case(
            maximum_definition,
            source_artifact=source_artifact,
            output_root=output_root,
            extra_arguments=(
                "--source-refinement-factor-z",
                "2",
                "--near-lateral-step-nm",
                "25",
                "--oxide-cells",
                "38",
                "--design-step-nm",
                "25",
                "--max-outer-step-um",
                "0.5",
                "--max-si-step-um",
                "0.25",
            ),
            case_id=refined_case_id,
        )
        native = results[maximum_definition.case_id]
        native_tmax, native_average = response(native)
        refined_tmax, refined_average = response(refined)
        mesh_comparison = {
            "scenario_case_id": maximum_definition.case_id,
            "native_case_id": native["case_id"],
            "refined_case_id": refined["case_id"],
            "native_to_refined_Tmax_relative_change": abs(
                relative_change(native_tmax, refined_tmax)
            ),
            "native_to_refined_average_relative_change": abs(
                relative_change(native_average, refined_average)
            ),
            "native_to_refined_common_flake_3d_NRMSE": nrmse(
                native, refined
            ),
            "physical_Tmax_change_vs_control_native": relative_change(
                native_tmax, response(reference)[0]
            ),
            "numerical_error_separated_from_physical_variation": True,
        }
        rows.append(
            build_row(
                maximum_definition,
                refined,
                reference,
                mesh_role="refined_maximum_variation_scenario",
            )
        )

    expected_source = all(
        row["optical_source_SHA256"] == EXPECTED_OPTICAL_SHA256
        and abs(row["mapped_source_power_W"] - EXPECTED_POWER_W)
        / EXPECTED_POWER_W
        < 0.005
        for row in rows
    )
    gates_pass = all(
        row["passed"]
        and row["energy_balance_relative_error"] < 0.01
        and row["linear_residual_relative"] < 1e-8
        and row["Q_mapping_relative_error"] < 0.005
        for row in rows
    )
    if refined is not None:
        gates_pass = gates_pass and bool(refined["passed"])
        expected_source = expected_source and (
            refined["source"]["optical_source_artifact_sha256"]
            == EXPECTED_OPTICAL_SHA256
        )
    passed = gates_pass and expected_source
    csv_path = output_root / f"physical_model_{args.phase}_cases.csv"
    write_rows(csv_path, rows)
    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "phase": args.phase,
        "status": (
            "VALIDATED_FVM_THERMAL_PHYSICAL_MODEL_SCENARIOS"
            if passed
            else "FAILED_FVM_THERMAL_PHYSICAL_MODEL_SCENARIOS"
        ),
        "passed": passed,
        "case_count": len(rows),
        "reference_case_id": REFERENCE_CASE_ID,
        "reference_is_unique_physical_truth": False,
        "G_top_scenarios": {
            "numerical_convergence_checkpoint_W_m2K": 7.37e6,
            "earlier_evaporated_SiO2_estimate_W_m2K": 7.37e4,
            "neither_promoted_as_uniquely_correct": True,
        },
        "TaIrTe4_kz_scenarios_W_mK": [0.5, 1.0, 2.0],
        "TaIrTe4_kz_basis": (
            "numerical scenarios, not a confidence interval; repository "
            "does not provide a traceable source for a physical range"
        ),
        "fabrication_geometry_status": (
            "BLOCKED_FABRICATION_GEOMETRY_UNCONFIRMED"
        ),
        "geometry_scenarios": {
            "A": "suspended/overhanging disk outside the flake",
            "B": (
                "100 nm SiO2 support annulus connects the disk overhang "
                "to the surrounding bottom oxide"
            ),
            "optical_geometry_or_Q_modified": False,
        },
        "PR3_dependency": {
            "commit": "053260da6fd0caec28ce155221bd18f683a0e5e7",
            "included_in_PR4_ancestry": False,
            "required_artifact_SHA256": EXPECTED_OPTICAL_SHA256,
        },
        "numerical_boundary_flux_interpretation": (
            "artificial truncation-boundary flux; not a physical heat-path "
            "fraction"
        ),
        "maximum_physical_variation_case": (
            None if maximum_definition is None else maximum_definition.case_id
        ),
        "maximum_case_mesh_comparison": mesh_comparison,
        "cases": rows,
        "refined_case_result": refined,
        "criteria": {
            "energy_balance_relative_error_lt": 0.01,
            "linear_residual_relative_lt": 1e-8,
            "Q_mapping_relative_error_lt": 0.005,
        },
        "prohibited_operations": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "global_rescaling": False,
            "tiling": False,
            "source_deletion": False,
        },
        "transient_PTE_adjoint_gradient_optimization_executed": False,
    }
    summary_path = output_root / f"physical_model_{args.phase}_summary.json"
    write_json(summary_path, summary)
    if args.report_dir:
        report_dir = Path(args.report_dir).expanduser().resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        write_rows(
            report_dir / f"physical_model_{args.phase}_cases.csv", rows
        )
        write_json(
            report_dir / f"physical_model_{args.phase}_summary.json",
            summary,
        )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
