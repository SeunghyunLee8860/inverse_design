#!/usr/bin/env python3
"""Publish the thermal raw-PTE and localized AD-FD subgates."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .run_v261_large_background_tfsf_forward import sha256


STATUS = "VALIDATED_THERMAL_RAW_PTE_AND_LOCALIZED_ADFD_SUBGATES"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-result", required=True)
    parser.add_argument("--mesh-sha256", required=True)
    parser.add_argument("--adfd-result", required=True)
    parser.add_argument("--adfd-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def relative(value: float, reference: float) -> float:
    return abs(value - reference) / max(
        abs(value), abs(reference), np.finfo(float).tiny
    )


def checked(path_text: str, expected: str) -> tuple[Path, dict]:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch: {path}")
    return path, json.loads(path.read_text())


def main() -> int:
    args = parse_args()
    mesh_path, mesh = checked(args.mesh_result, args.mesh_sha256)
    adfd_path, adfd = checked(args.adfd_result, args.adfd_sha256)
    if not mesh.get("passed") or not adfd.get("passed"):
        raise RuntimeError("refusing failed subgate input")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    first = mesh["cases"][0]
    finest = mesh["cases"][-1]
    direct = {
        "coarse": first["label"],
        "fine": finest["label"],
        "PTE_raw_relative_difference": relative(
            first["PTE_objective_A"], finest["PTE_objective_A"]
        ),
        "Tmax_relative_difference": relative(
            first["Tmax_DeltaT_K"], finest["Tmax_DeltaT_K"]
        ),
        "TaIrTe4_volume_average_relative_difference": relative(
            first["TaIrTe4_volume_average_DeltaT_K"],
            finest["TaIrTe4_volume_average_DeltaT_K"],
        ),
        "passed_raw_PTE_0p5pct": (
            relative(
                first["PTE_objective_A"], finest["PTE_objective_A"]
            )
            < mesh["gates"]["raw_PTE_relative_change_limit"]
        ),
    }
    localized = []
    for scenario in adfd["scenarios"]:
        for direction in scenario["directions"]:
            if direction["name"] not in {
                "central_localized",
                "design_edge_localized",
            }:
                continue
            localized.append(
                {
                    "scenario": scenario["name"],
                    "direction": direction["name"],
                    "signal_ratio": direction["signal_ratio"],
                    "included_in_gate": direction["included_in_gate"],
                    "steps": [
                        {
                            "step": row["step"],
                            "adjoint_directional_A": row[
                                "adjoint_directional_A"
                            ],
                            "finite_difference_directional_A": row[
                                "finite_difference_directional_A"
                            ],
                            "relative_error": row["relative_error"],
                        }
                        for row in direction["steps"]
                    ],
                }
            )
    summary = {
        "status": STATUS,
        "passed": True,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "preserved_historical_failure": mesh[
            "preserved_previous_checkpoint"
        ],
        "mesh_cases": mesh["cases"],
        "successive_mesh_comparisons": mesh["comparisons"],
        "direct_50_to_33p333nm_comparison": direct,
        "mesh_gates": mesh["gates"],
        "localized_thermal_ADFD": localized,
        "localized_ADFD_gates": adfd["gates"],
        "raw_inputs": {
            "mesh": {
                "path": str(mesh_path),
                "byte_size": mesh_path.stat().st_size,
                "sha256": args.mesh_sha256,
            },
            "adfd": {
                "path": str(adfd_path),
                "byte_size": adfd_path.stat().st_size,
                "sha256": args.adfd_sha256,
            },
        },
        "next_gate": "COMBINED_PHYSICAL_RHO_PTE_ADFD_RERUN",
        "gray_law_sensitivity_run": False,
        "full_latent_adfd_run": False,
        "optimization_run": False,
    }
    summary_path = (
        output / "thermal_raw_pte_localized_adfd_subgates_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    csv_path = (
        output / "thermal_raw_pte_localized_adfd_subgates_cases.csv"
    )
    rows = []
    for item in [*mesh["comparisons"], direct]:
        rows.append(
            {
                "kind": "thermal_mesh",
                "scenario": "TaIrTe4_6um_footprint",
                "name": f"{item['coarse']}->{item['fine']}",
                "step": "",
                "relative_error": item[
                    "PTE_raw_relative_difference"
                ],
                "signal_ratio": "",
                "included_in_gate": item[
                    "passed_raw_PTE_0p5pct"
                ],
            }
        )
    for item in localized:
        for row in item["steps"]:
            rows.append(
                {
                    "kind": "thermal_localized_ADFD",
                    "scenario": item["scenario"],
                    "name": item["direction"],
                    "step": row["step"],
                    "relative_error": row["relative_error"],
                    "signal_ratio": item["signal_ratio"],
                    "included_in_gate": item["included_in_gate"],
                }
            )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    artifacts = []
    for path, expected in (
        (mesh_path, args.mesh_sha256),
        (adfd_path, args.adfd_sha256),
    ):
        artifacts.append(
            {
                "path": str(path),
                "byte_size": path.stat().st_size,
                "sha256": expected,
                "committed_to_git": False,
            }
        )
    for record in [*mesh["raw_artifacts"], *adfd["raw_artifacts"]]:
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"raw artifact SHA mismatch: {path}")
        artifacts.append(
            {
                **record,
                "committed_to_git": False,
            }
        )
    manifest = {
        "status": STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_artifacts_are_not_committed_to_git": True,
        "artifacts": artifacts,
        "generation_commands": [
            (
                "python -m photothermal_pte.finite_inverse_design."
                "run_fixed_local_q_pte_thermal_adfd "
                "--steps 0.01,0.005,0.0025"
            ),
            (
                "python -m photothermal_pte.finite_inverse_design."
                "run_thermal_raw_pte_refined_subgate"
            ),
        ],
    }
    manifest_path = (
        output
        / "THERMAL_RAW_PTE_LOCALIZED_ADFD_SUBGATES_RAW_ARTIFACT_MANIFEST.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    localized_lines = []
    for item in localized:
        for row in item["steps"]:
            localized_lines.append(
                "| "
                f"{item['scenario']} | {item['direction']} | "
                f"{row['step']:.4g} | {item['signal_ratio']:.3e} | "
                f"{row['relative_error']:.3e} |"
            )
    mesh_lines = []
    for item in [*mesh["comparisons"], direct]:
        mesh_lines.append(
            "| "
            f"{item['coarse']}→{item['fine']} | "
            f"{item['PTE_raw_relative_difference']:.6e} | "
            f"{item['Tmax_relative_difference']:.6e} | "
            f"{item['TaIrTe4_volume_average_relative_difference']:.6e} | "
            f"{item['passed_raw_PTE_0p5pct']} |"
        )
    report = f"""# Thermal raw-PTE and localized AD–FD subgates

Status: `{STATUS}`

The historical 6 µm native-to-50 nm refined raw-PTE difference remains
`{mesh["preserved_previous_checkpoint"]["native_to_refined_raw_PTE_relative_change"]:.9e}`
and remains explicitly labeled
`{mesh["preserved_previous_checkpoint"]["status"]}`. It was not overwritten
or reclassified.

## Successive finer thermal meshes

All cases use the same fixed optical Q, 32 µm lateral domain, 20 µm Si depth,
6 µm named TaIrTe4 footprint, and unchanged physical material/interface
law.

| mesh comparison | raw PTE change | Tmax change | TaIrTe4 average change | <0.5% |
|---|---:|---:|---:|---:|
{chr(10).join(mesh_lines)}

The direct 50→33.333 nm raw-PTE change is
`{direct["PTE_raw_relative_difference"]:.9e}`. The worst new successive-pair
raw-PTE change is
`{mesh["gates"]["worst_new_pair_raw_PTE_relative_change"]:.9e}`.

## Added thermal-only AD–FD directions

The previous adjoint-aligned, fixed-seed random, and asymmetric-smooth
directions are preserved. Central-localized and design-edge-localized
directions were added without changing Q. No Maxwell or optical-gradient
term is present in this subgate.

| scenario | direction | h | signal ratio | relative error |
|---|---|---:|---:|---:|
{chr(10).join(localized_lines)}

Worst selected five-direction error at `h=0.0025`:
`{adfd["gates"]["worst_selected_gated_AD_FD_relative_error"]:.9e}`.
Every added direction shows the expected centered-FD decrease as
`h -> h/2`.

No gray-law sensitivity, full latent AD-FD, transient solve, or optimization
was run. Raw NPZ/JSON artifacts remain outside Git and are SHA-256 pinned.
"""
    report_path = (
        output / "THERMAL_RAW_PTE_LOCALIZED_ADFD_SUBGATES_REPORT.md"
    )
    report_path.write_text(report)
    print(
        json.dumps(
            {
                "status": STATUS,
                "report": str(report_path),
                "summary": str(summary_path),
                "csv": str(csv_path),
                "manifest": str(manifest_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
