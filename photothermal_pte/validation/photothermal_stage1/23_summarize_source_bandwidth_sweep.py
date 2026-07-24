#!/usr/bin/env python3
"""Select the narrowest passing source range and write sweep/regression reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


RANGES = [
    ("3.6-4.4", "range_3p6_4p4_flat_y", "range_3p6_4p4_disk_x"),
    ("3-6", "range_3_6_flat_y", "range_3_6_disk_x"),
    ("2.67-8", "range_2p67_8_flat_y", "range_2p67_8_disk_x"),
    ("3-12", "range_3_12_flat_y", "range_3_12_disk_x"),
]
LIMIT = 0.005


def read_result(root: Path, name: str) -> dict[str, Any]:
    path = root / name / "case_result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def solver_converged(result: dict[str, Any]) -> bool:
    if "solver_run" in result:
        return bool(result["solver_run"]["converged"])
    project = Path(result["project"])
    log = project.with_name(project.stem + "_p0.log")
    text = log.read_text(errors="replace") if log.is_file() else ""
    return "electromagnetic fields are diverging" not in text.lower()


def fitted(result: dict[str, Any], axis: str) -> complex:
    for row in result["epsilon_contract"]["axes"]:
        if row["axis"] == axis:
            value = row["fitted_epsilon"]
            return complex(value["real"], value["imag"])
    raise KeyError(axis)


def mesh_signature(result: dict[str, Any]) -> dict[str, Any]:
    mesh = result["mesh_contract"]
    return {
        "dt_s": mesh["solver"]["dt"],
        "mesh_type": mesh["solver"]["mesh type"],
        "mesh_refinement": mesh["solver"]["mesh refinement"],
        "mesh_accuracy": mesh["solver"]["mesh accuracy"],
        "x_coordinates_m": mesh["axes"]["x"]["coordinates_m"],
        "y_coordinates_m": mesh["axes"]["y"]["coordinates_m"],
        "z_coordinates_m": mesh["axes"]["z"]["coordinates_m"],
        "global_uniform_mesh_exists": mesh["mesh_objects"]["global_uniform_mesh"]["exists"],
        "flake_dz_min_m": mesh["TaIrTe4_internal_dz_min_m"],
        "flake_dz_max_m": mesh["TaIrTe4_internal_dz_max_m"],
    }


def relative_change(a: complex, b: complex) -> float:
    return float(abs(a - b) / max(abs(b), np.finfo(float).tiny))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--regression-root")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    all_results: dict[str, dict[str, Any]] = {}
    for label, flat_name, disk_name in RANGES:
        flat = read_result(root, flat_name)
        disk = read_result(root, disk_name)
        all_results[f"{label}:flat_y"] = flat
        all_results[f"{label}:disk_x"] = disk
        flat_abs = flat["absorption"]
        disk_abs = disk["absorption"]
        flat_closure = float(flat_abs["delta_closure_pabs_adv"])
        tmm_error = float(flat_abs["delta_TMM"])
        disk_closure = float(disk_abs["delta_closure_pabs_adv"])
        flat_converged = solver_converged(flat)
        disk_converged = solver_converged(disk)
        passed = (
            flat_converged and disk_converged
            and flat_closure < LIMIT and tmm_error < LIMIT and disk_closure < LIMIT
        )
        rows.append(
            {
                "source_range_um": label,
                "pulse_type": flat["source_contract_post_run"]["pulse type"],
                "flat_y_closure": flat_closure,
                "flat_y_TMM_error": tmm_error,
                "disk_closure": disk_closure,
                "flat_solver": "CONVERGED" if flat_converged else "DIVERGED",
                "disk_solver": "CONVERGED" if disk_converged else "DIVERGED",
                "result": "PASS" if passed else "FAIL",
            }
        )
        for case_label, result in (("flat_y", flat), ("disk_x", disk)):
            absorption = result["absorption"]
            case_rows.append(
                {
                    "source_range_um": label,
                    "case": case_label,
                    "polarization_angle_deg": result["polarization_angle_deg"],
                    "pulse_type": result["source_contract_post_run"]["pulse type"],
                    "pulselength_s": result["source_contract_post_run"]["pulselength"],
                    "pulse_offset_s": result["source_contract_post_run"]["offset"],
                    "sourcenorm_magnitude_at_4um": result["source_contract_post_run"]["sourcenorm_magnitude_at_4um"],
                    "R": absorption["R"],
                    "T": absorption["T"],
                    "A_global": absorption["A_global"],
                    "A_local_flux": absorption["A_local_flux"],
                    "A_six_face": absorption["A_six_face"],
                    "A_Qx": absorption["A_Qx_native"],
                    "A_Qy": absorption["A_Qy_native"],
                    "A_Qz": absorption["A_Qz_native"],
                    "A_Q": absorption["A_Q_pabs_adv"],
                    "delta_closure": absorption["delta_closure_pabs_adv"],
                    "A_TMM": None if absorption["TMM"] is None else absorption["TMM"]["polarization_weighted_A"],
                    "delta_TMM": absorption["delta_TMM"],
                    "dt_s": result["mesh_contract"]["solver"]["dt"],
                    "eps_x_fitted": fitted(result, "x"),
                    "eps_y_fitted": fitted(result, "y"),
                    "eps_z_fitted": fitted(result, "z"),
                    "result_json": str(root / (flat_name if case_label == "flat_y" else disk_name) / "case_result.json"),
                    "solver_converged": solver_converged(result),
                }
            )

    passing = [row for row in rows if row["result"] == "PASS"]
    if not passing:
        raise RuntimeError("no source range passed all criteria")
    selected = passing[0]["source_range_um"]

    with (output / "source_bandwidth_selection.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "source_bandwidth_cases.csv").open("w", newline="") as handle:
        fieldnames = list(case_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in case_rows:
            encoded = dict(row)
            for key in ("eps_x_fitted", "eps_y_fitted", "eps_z_fitted"):
                encoded[key] = repr(encoded[key])
            writer.writerow(encoded)

    reference = all_results["3-12:disk_x"]
    variation: dict[str, Any] = {}
    for case_kind in ("flat_y", "disk_x"):
        reference_case = all_results[f"3-12:{case_kind}"]
        variation[case_kind] = {}
        for label, _, _ in RANGES:
            result = all_results[f"{label}:{case_kind}"]
            variation[case_kind][label] = {
                f"epsilon_{axis}_relative_change_vs_3_12": relative_change(
                    fitted(result, axis), fitted(reference_case, axis)
                )
                for axis in "xyz"
            }
            variation[case_kind][label]["dt_relative_change_vs_3_12"] = float(
                abs(result["mesh_contract"]["solver"]["dt"] - reference_case["mesh_contract"]["solver"]["dt"])
                / reference_case["mesh_contract"]["solver"]["dt"]
            )
            variation[case_kind][label]["mesh_coordinates_identical_to_3_12"] = (
                mesh_signature(result)["x_coordinates_m"] == mesh_signature(reference_case)["x_coordinates_m"]
                and mesh_signature(result)["y_coordinates_m"] == mesh_signature(reference_case)["y_coordinates_m"]
                and mesh_signature(result)["z_coordinates_m"] == mesh_signature(reference_case)["z_coordinates_m"]
            )

    regression = None
    if args.regression_root:
        regression_root = Path(args.regression_root).expanduser().resolve()
        regression = {}
        for name in ("flat_x", "flat_y", "flat_45", "disk_x"):
            result = read_result(regression_root, name)
            absorption = result["absorption"]
            converged = solver_converged(result)
            regression[name] = {
                "source_range_um": result["source_range_um"],
                "A_global": absorption["A_global"],
                "A_local_flux": absorption["A_local_flux"],
                "A_six_face": absorption["A_six_face"],
                "A_Qx": absorption["A_Qx_native"],
                "A_Qy": absorption["A_Qy_native"],
                "A_Qz": absorption["A_Qz_native"],
                "A_Q": absorption["A_Q_pabs_adv"],
                "delta_closure": absorption["delta_closure_pabs_adv"],
                "A_TMM": None if absorption["TMM"] is None else absorption["TMM"]["polarization_weighted_A"],
                "delta_TMM": absorption["delta_TMM"],
                "solver_converged": converged,
                "pass_closure": converged and absorption["delta_closure_pabs_adv"] < LIMIT,
                "pass_TMM": converged and (
                    absorption["delta_TMM"] is None or absorption["delta_TMM"] < LIMIT
                ),
            }

    summary = {
        "selection_limit": LIMIT,
        "selection_rows": rows,
        "selected_narrowest_source_range_um": selected,
        "epsilon_dt_mesh_variation": variation,
        "regression": regression,
        "positive_control_disk_3_12": {
            "A_Q": reference["absorption"]["A_Q_pabs_adv"],
            "A_local_flux": reference["absorption"]["A_local_flux"],
            "delta_closure": reference["absorption"]["delta_closure_pabs_adv"],
        },
        "selected_disk_repeatability": {
            key: abs(
                all_results[f"{selected}:disk_x"]["absorption"][key]
                - regression["disk_x"][
                    {
                        "A_Q_pabs_adv": "A_Q",
                        "A_local_flux": "A_local_flux",
                        "delta_closure_pabs_adv": "delta_closure",
                    }[key]
                ]
            )
            for key in (
                "A_Q_pabs_adv", "A_local_flux", "delta_closure_pabs_adv"
            )
        } if regression is not None else None,
        "production_contract": {
            "source_range_um": selected,
            "material_sampled_range_um": [2.7, 13.2],
            "analysis_monitor_wavelength_um": 4.0,
            "mesh_type": "auto non-uniform",
            "mesh_refinement": "conformal variant 1",
            "mesh_accuracy": 5,
            "global_uniform_mesh": "absent",
            "TaIrTe4_dz_nm": 5.0,
        },
        "heat_run": False,
        "Qy_deleted": False,
        "Q_clipped": False,
        "flux_gain": False,
        "Q_rescaled": False,
    }
    (output / "source_bandwidth_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# TaIrTe4 source-bandwidth validation",
        "",
        "HEAT was not run. No Q channel was deleted; no clipping, gain, or rescaling was applied.",
        "",
        "| Source range | Pulse type | Flat-y closure | Flat-y TMM error | Disk closure | Solver state | Result |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        flat_text = (
            f"{100*row['flat_y_closure']:.6f}%"
            if row["flat_solver"] == "CONVERGED" else "DIVERGED (invalid)"
        )
        disk_text = (
            f"{100*row['disk_closure']:.6f}%"
            if row["disk_solver"] == "CONVERGED" else "DIVERGED (invalid)"
        )
        lines.append(
            f"| {row['source_range_um']} µm | {row['pulse_type']} | "
            f"{flat_text} | {100*row['flat_y_TMM_error']:.6f}% | "
            f"{disk_text} | {row['flat_solver']}/{row['disk_solver']} | {row['result']} |"
        )
    lines.extend(
        [
            "",
            "The 2.67–8 µm disk run terminated after the solver divergence marker; its post-run flux/Q values are invalid and were excluded from selection.",
            "The 3.6–4.4 µm source used Lumerical's standard-pulse branch and failed both flat and patterned closure tests.",
            "",
            f"## Selection",
            "",
            f"The narrowest range passing all three 0.5% criteria is **{selected} µm**.",
            "",
            "The per-case component absorption, pulse properties, fitted epsilon, dt, and mesh coordinates are in `source_bandwidth_cases.csv` and each case JSON.",
            "",
            "## Production contract (proposal only)",
            "",
            f"- Source: {selected} µm broadband; analysis remains a single point at 4 µm.",
            "- TaIrTe4 sampled material: 2.7–13.2 µm.",
            "- Mesh: auto non-uniform, conformal variant 1, accuracy 5; no global_uniform_mesh.",
            "- TaIrTe4 z override: 5 nm.",
            "- Production code was not modified by this report.",
        ]
    )
    if regression is not None:
        lines.extend(["", "## Selected-range regression", ""])
        for name, row in regression.items():
            lines.append(
                f"- {name}: solver={'CONVERGED' if row['solver_converged'] else 'DIVERGED'}, "
                f"closure={100*row['delta_closure']:.6f}%"
                + ("" if row["delta_TMM"] is None else f", TMM error={100*row['delta_TMM']:.6f}%")
            )
    (output / "SOURCE_BANDWIDTH_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
