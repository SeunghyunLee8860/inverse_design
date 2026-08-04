#!/usr/bin/env python3
"""Certify uniform rho=0.5 scalar/imported optical equivalence."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .summarize_imported_permittivity_endpoints import (
    LOW_POWER_FRACTION,
    METRIC_LIMIT,
    artifact,
    compare_npz,
    power_metrics,
)


STATUS_PASS = "VALIDATED_RHO05_IMPORTED_PERMITTIVITY_EQUIVALENCE"
STATUS_FAIL = "FAILED_RHO05_IMPORTED_PERMITTIVITY_EQUIVALENCE"
BOUNDS_LIMIT_M = 2.0e-18
EXPECTED_BOUNDS_M = {
    "x": [-1.0e-6, 1.0e-6],
    "y": [-1.0e-6, 1.0e-6],
    "z": [0.0, 600.0e-9],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar-result", required=True)
    parser.add_argument("--scalar-npz", required=True)
    parser.add_argument("--imported-result", required=True)
    parser.add_argument("--imported-npz", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--generation-command", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def selected_contract(result: dict[str, object]) -> dict[str, object]:
    geometry = result["geometry"]
    return {
        "fdtd_outer_input_bounds_m": geometry[
            "fdtd_outer_input_bounds_m"
        ],
        "tfsf_bounds_m": geometry["tfsf_bounds_m"],
        "design_bounds_m": geometry["design_bounds_m"],
        "omega_q_and_six_face_bounds_m": geometry[
            "omega_q_and_six_face_bounds_m"
        ],
        "flake_dz_m": geometry["flake_dz_m"],
        "transverse_mesh_m": geometry["transverse_mesh_m"],
        "pml": result["pml"],
        "simulation_time_ps": result["simulation_time_ps"],
        "source_readback_before_run": result[
            "source_readback_before_run"
        ],
        "source_normalization": result["source_normalization"],
    }


def main() -> int:
    args = parse_args()
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "scalar_result": args.scalar_result,
            "scalar_npz": args.scalar_npz,
            "imported_result": args.imported_result,
            "imported_npz": args.imported_npz,
        }.items()
    }
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError(paths)
    scalar = load_json(paths["scalar_result"])
    imported = load_json(paths["imported_result"])
    if not scalar.get("passed") or not imported.get("passed"):
        raise RuntimeError("both forward solves must first pass optical closure")
    if scalar.get("case") != "gray" or imported.get("case") != "gray":
        raise RuntimeError("both cases must be gray")
    if float(scalar.get("gray_rho", -1.0)) != 0.5:
        raise RuntimeError("scalar case is not rho=0.5")
    if float(imported.get("gray_rho", -1.0)) != 0.5:
        raise RuntimeError("imported case is not rho=0.5")
    if (
        imported.get("design_representation_requested")
        != "imported-permittivity"
    ):
        raise RuntimeError("imported case is mislabeled")
    imported_representation = imported["design_representation"]
    if imported_representation["sample_shape"] != [81, 81, 13]:
        raise RuntimeError("imported sample shape is not 81x81x13")
    expected_epsilon = 1.0 + 0.5 * (1.38**2 - 1.0)
    epsilon_error = abs(
        float(imported_representation["epsilon"]) - expected_epsilon
    )
    bounds = imported_representation["object_bounds_readback_m"]
    bounds_error = max(
        abs(float(bounds[axis][side]) - EXPECTED_BOUNDS_M[axis][side])
        for axis in "xyz"
        for side in (0, 1)
    )
    scalar_contract = selected_contract(scalar)
    imported_contract = selected_contract(imported)
    contract_equal = scalar_contract == imported_contract

    spatial = compare_npz(paths["scalar_npz"], paths["imported_npz"])
    power = power_metrics(scalar, imported)
    major_component_errors = [
        item["relative_difference"]
        for item in power["Q_components"].values()
        if not item["low_power_diagnostic"]
    ]
    gated = {
        "P_Q": power["P_Q_relative_difference"],
        "P_six": power["P_six_relative_difference"],
        "complex_field": spatial["combined_complex_field_NRMSE"],
        "spatial_Q": spatial["combined_spatial_Q_NRMSE"],
        "index": spatial["combined_index_NRMSE"],
        "major_Q_component": max(major_component_errors, default=0.0),
    }
    worst = max(gated.values())
    passed = bool(
        contract_equal
        and bounds_error < BOUNDS_LIMIT_M
        and epsilon_error < 1.0e-15
        and worst < METRIC_LIMIT
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "status": STATUS_PASS if passed else STATUS_FAIL,
        "passed": passed,
        "generated_at_utc": generated_at,
        "scope": (
            "uniform rho=0.5 scalar-index versus 81x81x13 "
            "imported-permittivity v261 CPU-TFSF equivalence"
        ),
        "rho": 0.5,
        "epsilon_law": "epsilon=1+rho*(1.38^2-1)",
        "expected_epsilon": expected_epsilon,
        "imported_epsilon_absolute_error": epsilon_error,
        "imported_sample_shape": imported_representation["sample_shape"],
        "imported_object_bounds_readback_m": bounds,
        "imported_bounds_max_abs_error_m": bounds_error,
        "exact_forward_contract_match": contract_equal,
        "scalar_representation_provenance": (
            "legacy matched-rho0.5 checkpoint predating the explicit "
            "design_representation_requested metadata field; its case, rho, "
            "geometry, source, mesh, PML, closure, native NPZ, FSP SHA and "
            "generation manifest are retained"
        ),
        "power": power,
        "spatial": spatial,
        "gated_metrics": gated,
        "maximum_gated_metric": worst,
        "gates": {
            "metric_limit": METRIC_LIMIT,
            "bounds_limit_m": BOUNDS_LIMIT_M,
            "contract_must_match_exactly": True,
        },
        "next_gate": (
            "COMPONENT_WISE_YEE_MATERIAL_JACOBIAN_AND_COLLOCATION"
            if passed
            else "STOP_FAIL_CLOSED"
        ),
        "thermal_run": False,
        "adjoint_run": False,
        "gradient_run": False,
        "optimization_run": False,
    }
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        report_dir / "rho05_imported_permittivity_equivalence_summary.json"
    )
    csv_path = (
        report_dir / "rho05_imported_permittivity_equivalence_cases.csv"
    )
    report_path = (
        report_dir / "RHO05_IMPORTED_PERMITTIVITY_EQUIVALENCE_REPORT.md"
    )
    manifest_path = (
        report_dir
        / "RHO05_IMPORTED_PERMITTIVITY_EQUIVALENCE_RAW_ARTIFACT_MANIFEST.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    row = {
        "rho": 0.5,
        **gated,
        "maximum_gated_metric": worst,
        "bounds_error_m": bounds_error,
        "contract_equal": contract_equal,
        "passed": passed,
    }
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(row), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(row)
    report_path.write_text(
        f"""# Uniform rho=0.5 imported-permittivity equivalence

Status: `{summary['status']}`

This is the missing gray-state representation check. It compares the
previously certified scalar-index rho=0.5 CPU-TFSF solve with a new,
otherwise matched v261 solve using 81×81×13 imported `n` samples. There is
no bitwise-equality requirement and no phase fitting, empirical
normalization, gradient rescaling, thermal solve, or optimization.

The imported chain is
`epsilon=1+rho*(1.38^2-1)`, `n=sqrt(epsilon)`, with rho=0.5. The object
bounds are x,y=[-1,1] µm and z=[0,600] nm. The scalar and imported records
have an exact match for the selected geometry, source, mesh, PML,
simulation-time, and incident-normalization contract: `{contract_equal}`.

| metric | relative difference |
| --- | ---: |
| P_Q | {gated['P_Q']:.9e} |
| P_six | {gated['P_six']:.9e} |
| complex field NRMSE | {gated['complex_field']:.9e} |
| spatial Q NRMSE | {gated['spatial_Q']:.9e} |
| index NRMSE | {gated['index']:.9e} |
| major Q-component power | {gated['major_Q_component']:.9e} |

Worst metric: `{worst:.9e}`; required: `<{METRIC_LIMIT:.9e}`.
Maximum imported-object bounds error: `{bounds_error:.9e} m`; required:
`<{BOUNDS_LIMIT_M:.9e} m`.

Raw FSP and NPZ artifacts remain outside Git. Their paths, sizes, and
SHA-256 values are recorded in the manifest.
"""
    )
    artifacts = []
    for representation in ("scalar", "imported"):
        result = scalar if representation == "scalar" else imported
        for kind in ("result", "npz"):
            artifacts.append(
                {
                    "role": f"rho05_{representation}_{kind}",
                    **artifact(paths[f"{representation}_{kind}"]),
                }
            )
        artifacts.append(
            {
                "role": f"rho05_{representation}_FSP",
                **result["artifacts"]["fsp"],
            }
        )
    manifest = {
        "status": summary["status"],
        "generated_at_utc": generated_at,
        "generation_command": args.generation_command,
        "artifacts": artifacts,
        "git_policy": (
            "Raw FSP and native field/Q NPZ artifacts stay outside Git; "
            "only paths, sizes, and SHA-256 values are committed"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
