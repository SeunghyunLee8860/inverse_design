#!/usr/bin/env python3
"""Run and summarize multi-material FVM production sensitivity cases."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import config_stage1 as config
from lumerical_api import utc_timestamp, write_json


SCRIPT = Path(__file__).with_name("36_run_fvm_multimaterial_thermal.py")
CONVERGENCE_LIMIT = 0.01


@dataclass(frozen=True)
class Case:
    case_id: str
    family: str
    value: str
    arguments: tuple[str, ...] = field(default_factory=tuple)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--baseline-result",
        default=str(
            config.OUTPUT_ROOT
            / "fvm_multimaterial_thermal"
            / "baseline_v2"
            / "case_result.json"
        ),
    )
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def output_directory(explicit: str | None) -> Path:
    output = (
        Path(explicit).expanduser().resolve()
        if explicit
        else config.OUTPUT_ROOT
        / "fvm_multimaterial_sensitivity"
        / utc_timestamp()
    )
    output.mkdir(parents=True, exist_ok=True)
    return output


def case_definitions() -> list[Case]:
    common_final = (
        "--lateral-domain-um",
        "32",
        "--si-depth-um",
        "20",
    )
    return [
        Case("domain_L4", "lateral_domain_um", "4", ("--lateral-domain-um", "4")),
        Case("domain_L16", "lateral_domain_um", "16", ("--lateral-domain-um", "16")),
        Case("domain_L32", "lateral_domain_um", "32", ("--lateral-domain-um", "32")),
        Case(
            "depth_D2_L32",
            "Si_depth_um",
            "2",
            ("--lateral-domain-um", "32", "--si-depth-um", "2"),
        ),
        Case(
            "depth_D5_L32",
            "Si_depth_um",
            "5",
            ("--lateral-domain-um", "32", "--si-depth-um", "5"),
        ),
        Case(
            "depth_D10_L32",
            "Si_depth_um",
            "10",
            ("--lateral-domain-um", "32", "--si-depth-um", "10"),
        ),
        Case(
            "final_native",
            "thermal_mesh",
            "native",
            common_final,
        ),
        Case(
            "mesh_coarse",
            "thermal_mesh",
            "coarse",
            (
                *common_final,
                "--source-coarsening-factor",
                "2",
                "--near-lateral-step-nm",
                "100",
                "--oxide-cells",
                "10",
                "--design-step-nm",
                "100",
                "--max-outer-step-um",
                "2",
                "--max-si-step-um",
                "1",
            ),
        ),
        Case(
            "mesh_refined",
            "thermal_mesh",
            "refined",
            (
                *common_final,
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
        ),
        *[
            Case(
                f"Gbottom_{label}",
                "G_bottom_W_m2K",
                label,
                (*common_final, "--G-bottom", value),
            )
            for label, value in (
                ("1e6", "1e6"),
                ("3e6", "3e6"),
                ("1p5e7", "1.5e7"),
                ("3e7", "3e7"),
                ("1e8", "1e8"),
                ("perfect", "perfect"),
            )
        ],
        *[
            Case(
                f"Gtop_{label}",
                "G_top_W_m2K",
                label,
                (*common_final, "--G-top", value),
            )
            for label, value in (
                ("7p37e4", "7.37e4"),
                ("7p37e5", "7.37e5"),
                ("7p37e7", "7.37e7"),
                ("perfect", "perfect"),
            )
        ],
        Case(
            "oxide_si_perfect",
            "G_oxide_Si_W_m2K",
            "perfect",
            (*common_final, "--G-oxide-si", "perfect"),
        ),
        Case(
            "convection_h10",
            "exposed_convection_W_m2K",
            "10",
            (*common_final, "--exposed-h-W-m2K", "10"),
        ),
    ]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def execute_case(case: Case, root: Path, rerun: bool) -> dict[str, Any]:
    directory = root / case.case_id
    result_path = directory / "case_result.json"
    if result_path.is_file() and not rerun:
        return load_json(result_path)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"cannot rerun non-empty case directory without removing it: {directory}"
        )
    command = [
        sys.executable,
        str(SCRIPT),
        "--output-dir",
        str(directory),
        "--case-id",
        case.case_id,
        *case.arguments,
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
            f"{case.case_id} failed with code {completed.returncode}\n"
            f"{completed.stdout}"
        )
    return load_json(result_path)


def temperature_metrics(result: dict[str, Any]) -> tuple[float, float]:
    response = result["temperature_response"]
    return (
        float(response["DeltaT_max_K_per_W_m2"]),
        float(response["TaIrTe4_volume_average_DeltaT_K_per_W_m2"]),
    )


def relative_change(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), np.finfo(float).tiny)


def flake_probe_field(result: dict[str, Any]) -> np.ndarray:
    raw_path = Path(result["raw_field_path"])
    with np.load(raw_path, allow_pickle=False) as raw:
        x_edges = np.asarray(raw["x_edges_m"], float)
        y_edges = np.asarray(raw["y_edges_m"], float)
        z_edges = np.asarray(raw["z_edges_m"], float)
        temperature = np.asarray(raw["delta_T_K_per_W_m2"], float)
    centers = tuple(
        0.5 * (edges[:-1] + edges[1:])
        for edges in (x_edges, y_edges, z_edges)
    )
    interpolator = RegularGridInterpolator(
        centers,
        temperature,
        method="linear",
        bounds_error=True,
    )
    probe_axes = (
        np.linspace(-0.85e-6, 0.85e-6, 11),
        np.linspace(-0.85e-6, 0.85e-6, 11),
        np.linspace(-90.0e-9, -10.0e-9, 7),
    )
    probe_grid = np.meshgrid(*probe_axes, indexing="ij")
    points = np.column_stack([item.reshape(-1) for item in probe_grid])
    values = interpolator(points)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("flake probe interpolation contains NaN or Inf")
    return values


def probe_nrmse(result: dict[str, Any], reference: dict[str, Any]) -> float:
    values = flake_probe_field(result)
    reference_values = flake_probe_field(reference)
    scale = max(float(np.max(np.abs(reference_values))), np.finfo(float).tiny)
    return float(np.sqrt(np.mean((values - reference_values) ** 2)) / scale)


def add_comparison(
    comparisons: list[dict[str, Any]],
    *,
    family: str,
    lower: dict[str, Any],
    upper: dict[str, Any],
) -> None:
    lower_metrics = temperature_metrics(lower)
    upper_metrics = temperature_metrics(upper)
    comparisons.append(
        {
            "family": family,
            "from_case": lower["case_id"],
            "to_case": upper["case_id"],
            "Tmax_relative_change": relative_change(
                lower_metrics[0], upper_metrics[0]
            ),
            "flake_average_relative_change": relative_change(
                lower_metrics[1], upper_metrics[1]
            ),
            "flake_probe_3d_NRMSE": probe_nrmse(lower, upper),
        }
    )


def main() -> int:
    args = parse_args()
    root = output_directory(args.output_dir)
    baseline = load_json(Path(args.baseline_result).expanduser().resolve())
    if not baseline["passed"]:
        raise RuntimeError("baseline result is not passing")
    results: dict[str, dict[str, Any]] = {"baseline_L8um_Si5um": baseline}
    definitions = case_definitions()
    for index, case in enumerate(definitions, start=1):
        print(f"[{index}/{len(definitions)}] {case.case_id}", flush=True)
        results[case.case_id] = execute_case(case, root, args.rerun)

    comparisons: list[dict[str, Any]] = []
    for lower, upper in (
        ("domain_L4", "baseline_L8um_Si5um"),
        ("baseline_L8um_Si5um", "domain_L16"),
        ("domain_L16", "domain_L32"),
    ):
        add_comparison(
            comparisons,
            family="lateral_domain_um",
            lower=results[lower],
            upper=results[upper],
        )
    for lower, upper in (
        ("depth_D2_L32", "depth_D5_L32"),
        ("depth_D5_L32", "depth_D10_L32"),
        ("depth_D10_L32", "final_native"),
    ):
        add_comparison(
            comparisons,
            family="Si_depth_um",
            lower=results[lower],
            upper=results[upper],
        )
    for lower, upper in (
        ("mesh_coarse", "final_native"),
        ("final_native", "mesh_refined"),
    ):
        add_comparison(
            comparisons,
            family="thermal_mesh",
            lower=results[lower],
            upper=results[upper],
        )

    final_comparisons = {
        family: [
            item for item in comparisons if item["family"] == family
        ][-1]
        for family in ("lateral_domain_um", "Si_depth_um", "thermal_mesh")
    }
    all_cases_pass = all(bool(item["passed"]) for item in results.values())
    convergence_pass = all(
        max(
            item["Tmax_relative_change"],
            item["flake_average_relative_change"],
            item["flake_probe_3d_NRMSE"],
        )
        < CONVERGENCE_LIMIT
        for item in final_comparisons.values()
    )
    passed = all_cases_pass and convergence_pass
    ordered_results = [baseline] + [results[item.case_id] for item in definitions]
    rows = []
    for result in ordered_results:
        tmax, average = temperature_metrics(result)
        rows.append(
            {
                "case_id": result["case_id"],
                "passed": result["passed"],
                "lateral_domain_um": result["geometry"]["lateral_domain_m"] * 1e6,
                "Si_depth_um": result["geometry"]["Si_depth_m"] * 1e6,
                "active_cells": result["grid"]["active_solid_cell_count"],
                "Tmax_K_per_W_m2": tmax,
                "flake_average_K_per_W_m2": average,
                "energy_balance_relative_error": result["power_balance"]["relative_error"],
                "Q_mapping_relative_error": result["source"]["mapping_relative_error"],
                "source_mesh_mode": result["source"]["thermal_source_mesh_mode"],
                "G_bottom_W_m2K": result["interfaces"]["TaIrTe4_bottom"]["G_W_m2K"],
                "G_top_W_m2K": result["interfaces"]["TaIrTe4_top"]["G_W_m2K"],
                "G_oxide_Si_W_m2K": result["interfaces"]["oxide_Si"]["G_W_m2K"],
                "exposed_h_W_m2K": result["boundary_conditions"][
                    "exposed_heat_transfer_W_m2K"
                ],
            }
        )
    with (root / "sensitivity_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (root / "convergence_comparisons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "status": (
            "VALIDATED_MULTIMATERIAL_FVM_PRODUCTION_CONVERGENCE"
            if passed
            else "BLOCKED_PRODUCTION_FVM_CONVERGENCE_UNVERIFIED"
        ),
        "passed": passed,
        "solver_attribution": (
            "independent conservative Cartesian Python/SciPy FVM; "
            "not a Lumerical HEAT result"
        ),
        "case_count": len(ordered_results),
        "all_case_equations_converged_and_conserved": all_cases_pass,
        "convergence_limit": CONVERGENCE_LIMIT,
        "final_pair_comparisons": final_comparisons,
        "comparisons": comparisons,
        "reference_case": results["final_native"],
        "mesh_refined_case": results["mesh_refined"],
        "output_directory": str(root),
    }
    write_json(root / "sensitivity_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
