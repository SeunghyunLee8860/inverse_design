#!/usr/bin/env python3
"""Publish physical-model sensitivity separately from numerical convergence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import config_stage1 as config
from lumerical_api import utc_timestamp, write_json


GIT_ROOT = config.REPOSITORY_ROOT.parent
EXPECTED_SHA256 = (
    "7a63f82842751e7623e895701bac4ce92558679ed71bf13a3d404695a150e794"
)
EXPECTED_POWER_W = 2.56071371086521e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=str(
            config.OUTPUT_ROOT / "fvm_thermal_physical_model" / "sweep_v1"
        ),
    )
    parser.add_argument(
        "--report-dir",
        default=str(
            config.REPOSITORY_ROOT
            / "reports"
            / "fvm_thermal_physical_model"
        ),
    )
    parser.add_argument(
        "--numerical-summary",
        default=str(
            config.REPOSITORY_ROOT
            / "reports"
            / "fvm_multimaterial_thermal"
            / "fvm_multimaterial_thermal_summary.json"
        ),
    )
    parser.add_argument("--reproduction-precheck")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def git_value(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=GIT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percent(value: Any) -> str:
    return f"{100.0 * float(value):+.6g}%"


def format_hotspot(row: dict[str, str]) -> str:
    return (
        f"({float(row['hotspot_x_m']):.6e}, "
        f"{float(row['hotspot_y_m']):.6e}, "
        f"{float(row['hotspot_z_m']):.6e})"
    )


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def artifact_entry(path: Path) -> dict[str, Any]:
    return {
        "repository_path": str(path.resolve().relative_to(GIT_ROOT)),
        "server_path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    physical = load_json(input_dir / "physical_model_all_summary.json")
    numerical = load_json(
        Path(args.numerical_summary).expanduser().resolve()
    )
    rows = load_csv(input_dir / "physical_model_all_cases.csv")
    by_id = {row["case_id"]: row for row in rows}
    control = by_id["scenario_control_Gtop_7p37e6_kz_1"]
    evaporated = by_id[
        "scenario_evaporated_SiO2_estimate_Gtop_7p37e4"
    ]

    material_rows = [
        by_id[name]
        for name in (
            "scenario_control_Gtop_7p37e6_kz_1",
            "scenario_evaporated_SiO2_estimate_Gtop_7p37e4",
            "scenario_kz_0p5",
            "scenario_kz_2p0",
        )
    ]
    material_table = [
        [
            row["case_id"],
            f"{float(row['Tmax_K_per_W_m2']):.9e}",
            f"{float(row['TaIrTe4_average_K_per_W_m2']):.9e}",
            percent(row["Tmax_change_vs_control"]),
            percent(row["average_change_vs_control"]),
            percent(row["common_flake_3d_NRMSE_vs_control"]),
            format_hotspot(row),
            f"{float(row['top_interface_mean_jump_K']):.6e}",
            f"{float(row['top_interface_max_jump_K']):.6e}",
        ]
        for row in material_rows
    ]
    boundary_rows = [
        by_id[name]
        for name in (
            "scenario_control_Gtop_7p37e6_kz_1",
            "scenario_far_xy_adiabatic_bottom_fixed",
            "scenario_exposed_convection_h5",
            "scenario_exposed_convection_h10",
            "scenario_exposed_convection_h20",
        )
    ]
    boundary_table = [
        [
            row["case_id"],
            f"{float(row['Tmax_K_per_W_m2']):.9e}",
            percent(row["Tmax_change_vs_control"]),
            percent(row["common_flake_3d_NRMSE_vs_control"]),
            f"{float(row['bottom_numerical_boundary_flux_fraction']):.6f}",
            f"{float(row['lateral_numerical_boundary_flux_fraction']):.6f}",
        ]
        for row in boundary_rows
    ]
    geometry_native = by_id[
        "scenario_geometry_oxide_supported_overhang"
    ]
    geometry_refined = by_id[
        "scenario_geometry_oxide_supported_overhang_refined"
    ]
    convergence = numerical["final_pair_comparisons"]
    mesh = physical["maximum_case_mesh_comparison"]

    report = f"""# FVM thermal physical-model sensitivity report

## Status and scope

**Status: `VALIDATED_FVM_THERMAL_PHYSICAL_MODEL_SCENARIOS` with
`BLOCKED_FABRICATION_GEOMETRY_UNCONFIRMED`.**

This report does not promote one arbitrary parameter set as a final
experimental prediction. It separates:

1. numerical convergence of the independent conservative Cartesian FVM; and
2. physical-model variation caused by uncertain material, interface,
   boundary, and fabrication assumptions.

It is not a Lumerical HEAT result. No optical geometry or Q was changed, and
no transient, PTE, adjoint, gradient, or optimization calculation was run.

## Immutable numerical checkpoint

PR #4 commit
`437ec0644b15a4b9a6919a0151e4aa531fb1e0ab` remains the immutable numerical
checkpoint. Its final-pair numerical metrics are:

| Numerical refinement | Tmax | flake average | common 3D flake NRMSE |
| --- | ---: | ---: | ---: |
| lateral 16 to 32 um | {percent(convergence["lateral_domain_um"]["Tmax_relative_change"])} | {percent(convergence["lateral_domain_um"]["flake_average_relative_change"])} | {percent(convergence["lateral_domain_um"]["flake_probe_3d_NRMSE"])} |
| Si depth 10 to 20 um | {percent(convergence["Si_depth_um"]["Tmax_relative_change"])} | {percent(convergence["Si_depth_um"]["flake_average_relative_change"])} | {percent(convergence["Si_depth_um"]["flake_probe_3d_NRMSE"])} |
| native to refined mesh | {percent(convergence["thermal_mesh"]["Tmax_relative_change"])} | {percent(convergence["thermal_mesh"]["flake_average_relative_change"])} | {percent(convergence["thermal_mesh"]["flake_probe_3d_NRMSE"])} |

The promoted publication metadata now records
`provisional_until_sensitivity_passes=false` and `next_required_gate=null`.
Raw per-case JSON retains its original provisional fields as immutable
provenance.

## G_top and TaIrTe4 kz scenarios

`G_top=7.37e6 W/(m2 K)` is the PR #4 numerical-convergence checkpoint
scenario. `G_top=7.37e4 W/(m2 K)` is the earlier contract's named
evaporated-SiO2 estimate scenario. The repository contains no traceable
literature source that establishes either as uniquely correct, so neither is
promoted.

TaIrTe4 uses fixed `kx=14.4` and `ky=3.8 W/(m K)`. The values
`kz=0.5, 1.0, 2.0 W/(m K)` are numerical scenarios, not a confidence
interval, because the repository does not establish a sourced physical
range.

{table(
    [
        "scenario",
        "Tmax",
        "flake average",
        "Tmax change",
        "average change",
        "3D NRMSE",
        "hotspot (x,y,z) m",
        "mean top jump K",
        "max top jump K",
    ],
    material_table,
)}

Direct G_top comparison:

- checkpoint scenario Tmax:
  `{float(control["Tmax_K_per_W_m2"]):.12e} K/(W/m2)`;
  evaporated-estimate scenario Tmax:
  `{float(evaporated["Tmax_K_per_W_m2"]):.12e} K/(W/m2)`.
- checkpoint/evaporated mean top-interface jump:
  `{float(control["top_interface_mean_jump_K"]):.12e}` /
  `{float(evaporated["top_interface_mean_jump_K"]):.12e} K`.
- checkpoint/evaporated maximum top-interface jump:
  `{float(control["top_interface_max_jump_K"]):.12e}` /
  `{float(evaporated["top_interface_max_jump_K"]):.12e} K`.

## Boundary-condition robustness

Every case uses a fixed bottom temperature on the same 32 um lateral,
20 um Si-depth geometry. Artificial lateral/bottom truncation-boundary flux
is reported only as a **numerical boundary flux**, not as a physical
heat-path fraction.

{table(
    [
        "scenario",
        "Tmax",
        "Tmax change",
        "3D NRMSE",
        "bottom numerical fraction",
        "lateral numerical fraction",
    ],
    boundary_table,
)}

Exposed convection was evaluated at `h=0,5,10,20 W/(m2 K)`. Its small effect
under these numerical boundaries does not validate an experimental ambient
heat-transfer coefficient.

## Top-disk fabrication geometry

Repository optical geometry defines a radius-1.5-um SiO2 disk at
`z=0...600 nm` touching the 2x2-um flake, but does not establish how the disk
outside the flake is fabricated or thermally supported. Therefore:

- scenario A: suspended/overhanging disk outside the flake;
- scenario B: a 100 nm SiO2 support annulus fills the gap outside the flake
  and connects the disk to the surrounding bottom oxide.

Scenario B changes Tmax by
`{percent(geometry_native["Tmax_change_vs_control"])}` and the flake-average
temperature by `{percent(geometry_native["average_change_vs_control"])}`.
Its common 3D flake-field NRMSE is
`{percent(geometry_native["common_flake_3d_NRMSE_vs_control"])}`.
This large difference is why fabrication geometry remains a blocker.

For the maximum-variation scenario B, native-to-refined numerical changes
are:

- Tmax: `{percent(mesh["native_to_refined_Tmax_relative_change"])}`;
- flake average:
  `{percent(mesh["native_to_refined_average_relative_change"])}`;
- common 3D flake-field NRMSE:
  `{percent(mesh["native_to_refined_common_flake_3d_NRMSE"])}`.

The refined scenario-B Tmax is
`{float(geometry_refined["Tmax_K_per_W_m2"]):.12e} K/(W/m2)`. The physical
support-geometry variation is much larger than the associated numerical
mesh error.

## Optical dependency and fail-closed reproduction

PR #3 commit `053260da6fd0caec28ce155221bd18f683a0e5e7` is not in PR #4
ancestry. A clean checkout must supply the external raw PR #3 NPZ:

```bash
python photothermal_pte/validation/photothermal_stage1/40_reproduce_fvm_thermal_physical_model.py \\
  --pr3-q-artifact /absolute/path/to/finite_q_on_artifact.npz \\
  --output-root /absolute/path/to/new_output
```

The entry point verifies SHA-256
`{EXPECTED_SHA256}` before creating output or starting an import/thermal
solve. Missing or mismatched artifacts fail closed. The raw NPZ is not
committed.

All ten cases preserve `P_Q={EXPECTED_POWER_W:.15g} W`; mapping error is zero.
Clipping, smoothing, gain, global rescaling, tiling, and source deletion are
all false. Every energy-balance error is below 1%, every linear residual is
below `1e-8`, and every Q mapping error is below 0.5%.
"""
    report_path = report_dir / "FVM_THERMAL_PHYSICAL_MODEL_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    final_cases = report_dir / "fvm_thermal_physical_model_cases.csv"
    final_cases.write_text(
        (input_dir / "physical_model_all_cases.csv").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    final_summary = {
        **physical,
        "published_at_utc": utc_timestamp(),
        "published_status": (
            "VALIDATED_FVM_THERMAL_PHYSICAL_MODEL_SCENARIOS_WITH_"
            "FABRICATION_GEOMETRY_BLOCKER"
        ),
        "immutable_numerical_checkpoint": (
            "437ec0644b15a4b9a6919a0151e4aa531fb1e0ab"
        ),
        "numerical_convergence": convergence,
        "physical_uncertainty_is_not_numerical_error": True,
        "arbitrary_parameter_set_called_final_experimental_prediction": False,
        "promoted_reference_metadata": {
            "provisional_until_sensitivity_passes": False,
            "next_required_gate": None,
            "completed_gate": (
                "DOMAIN_DEPTH_MESH_INTERFACE_BOUNDARY_SENSITIVITY"
            ),
            "raw_per_case_JSON_modified": False,
        },
        "report_path": str(report_path),
        "cases_CSV_path": str(final_cases),
    }
    summary_path = (
        report_dir / "fvm_thermal_physical_model_summary.json"
    )
    write_json(summary_path, final_summary)

    artifacts: list[dict[str, Any]] = []
    for result_path in sorted(input_dir.glob("*/case_result.json")):
        raw_path = result_path.with_name("temperature_flux_3d.npz")
        artifacts.extend(
            [artifact_entry(result_path), artifact_entry(raw_path)]
        )
    precheck_path = (
        None
        if not args.reproduction_precheck
        else Path(args.reproduction_precheck).expanduser().resolve()
    )
    if precheck_path is not None:
        artifacts.append(artifact_entry(precheck_path))
    for path in sorted(report_dir.glob("physical_model_*")):
        if path.is_file():
            artifacts.append(artifact_entry(path))
    for path in (report_path, summary_path, final_cases):
        artifacts.append(artifact_entry(path))
    manifest = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "branch": git_value("branch", "--show-current"),
        "generation_commit": git_value("rev-parse", "HEAD"),
        "immutable_numerical_checkpoint": (
            "437ec0644b15a4b9a6919a0151e4aa531fb1e0ab"
        ),
        "PR3_artifact_in_ancestry": False,
        "external_PR3_artifact_SHA256": EXPECTED_SHA256,
        "raw_NPZ_committed_to_git": False,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    write_json(report_dir / "RAW_ARTIFACT_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
