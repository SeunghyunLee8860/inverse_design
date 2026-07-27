#!/usr/bin/env python3
"""Publish the optical-dz downstream convergence certificate."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .run_v261_large_background_tfsf_forward import sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-result", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
        "committed_to_git": False,
    }


def add_manifest(
    records: dict[str, dict[str, object]], candidate: dict[str, object]
) -> None:
    path = Path(str(candidate["path"])).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = artifact(path)
    expected = candidate.get("sha256")
    if expected is not None and actual["sha256"] != expected:
        raise RuntimeError(f"artifact SHA mismatch: {path}")
    records[str(path)] = actual


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_result).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    actual_sha = sha256(input_path)
    if actual_sha != args.expected_input_sha256:
        raise RuntimeError("input result SHA-256 mismatch")
    result = json.loads(input_path.read_text())
    if not result.get("passed"):
        raise RuntimeError("refusing to publish a failed convergence result")

    summary = {
        **result,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_result": {
            "path": str(input_path),
            "byte_size": input_path.stat().st_size,
            "sha256": actual_sha,
            "committed_to_git": False,
        },
        "next_gate": (
            "THERMAL_RAW_PTE_CONVERGENCE_AND_CENTRAL_EDGE_ADFD"
        ),
    }
    summary_path = (
        output
        / "optical_dz_downstream_pte_gradient_convergence_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    csv_path = (
        output
        / "optical_dz_downstream_pte_gradient_convergence_cases.csv"
    )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "coarse_dz_nm",
                "fine_dz_nm",
                "scenario",
                "P_Q_relative_difference",
                "remapped_Q_field_NRMSE",
                "Tmax_relative_difference",
                "TaIrTe4_temperature_field_NRMSE",
                "PTE_objective_relative_difference",
                "optical_directional_gradient_relative_difference",
                "combined_directional_gradient_relative_difference",
                "decisive_maximum",
                "passed_0p5pct",
            ],
        )
        writer.writeheader()
        for comparison in result["comparisons"]:
            for scenario, metrics in comparison["scenarios"].items():
                writer.writerow(
                    {
                        "coarse_dz_nm": comparison["coarse_dz_nm"],
                        "fine_dz_nm": comparison["fine_dz_nm"],
                        "scenario": scenario,
                        "P_Q_relative_difference": comparison[
                            "P_Q_relative_difference"
                        ],
                        **metrics,
                    }
                )

    raw_records: dict[str, dict[str, object]] = {}
    add_manifest(raw_records, artifact(input_path))
    for record in result["records"].values():
        add_manifest(raw_records, record["case_result"])
        add_manifest(raw_records, record["jacobian_result"])
        case_result = json.loads(
            Path(record["case_result"]["path"]).read_text()
        )
        add_manifest(raw_records, case_result["forward_FSP"])
        for candidate in case_result["raw_artifacts"]:
            add_manifest(raw_records, candidate)
        jacobian_result = json.loads(
            Path(record["jacobian_result"]["path"]).read_text()
        )
        add_manifest(
            raw_records, jacobian_result["artifacts"]["forward_FSP"]
        )
        add_manifest(
            raw_records, jacobian_result["artifacts"]["adjoint_FSP"]
        )
        for candidate in jacobian_result["artifacts"][
            "component_J"
        ].values():
            add_manifest(raw_records, candidate)
        add_manifest(
            raw_records, jacobian_result["artifacts"]["coordinates"]
        )
    for candidate in result["gradient_artifacts"]:
        add_manifest(raw_records, candidate)
    manifest = {
        "status": result["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_artifacts_are_not_committed_to_git": True,
        "artifacts": list(raw_records.values()),
        "generation_commands": [
            (
                "python -m photothermal_pte.finite_inverse_design."
                "run_nonuniform_optical_dz_forward"
            ),
            (
                "python -m photothermal_pte.finite_inverse_design."
                "run_optical_dz_downstream_case"
            ),
            (
                "python -m photothermal_pte.finite_inverse_design."
                "build_layout_component_yee_jacobian"
            ),
            (
                "python -m photothermal_pte.finite_inverse_design."
                "summarize_optical_dz_downstream_convergence"
            ),
        ],
    }
    manifest_path = (
        output
        / "OPTICAL_DZ_DOWNSTREAM_PTE_GRADIENT_CONVERGENCE_RAW_ARTIFACT_MANIFEST.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    rows = []
    for dz, record in result["records"].items():
        for scenario, value in record["scenarios"].items():
            rows.append(
                "| "
                f"{dz} | {scenario} | {record['P_Q_W']:.9e} | "
                f"{record['six_face_closure_relative_error']:.3e} | "
                f"{value['PTE_objective_A']:.9e} | "
                f"{value['gradient']['optical_directional_gradient_A']:.9e} | "
                f"{value['gradient']['combined_directional_gradient_A']:.9e} |"
            )
    comparisons = []
    for item in result["comparisons"]:
        for scenario, value in item["scenarios"].items():
            comparisons.append(
                "| "
                f"{item['coarse_dz_nm']:g}→{item['fine_dz_nm']:g} | "
                f"{scenario} | {value['remapped_Q_field_NRMSE']:.3e} | "
                f"{value['TaIrTe4_temperature_field_NRMSE']:.3e} | "
                f"{value['PTE_objective_relative_difference']:.3e} | "
                f"{value['optical_directional_gradient_relative_difference']:.3e} | "
                f"{value['combined_directional_gradient_relative_difference']:.3e} |"
            )
    report = f"""# Optical dz downstream PTE/gradient convergence

Status: `{result["status"]}`

The nonuniform 81×81 physical-density forward source was solved at
`dz = 2.5, 1.25, 0.625 nm`. Each source was conservatively remapped into the
same explicit 4 µm and 6 µm thermal scenarios. The spatially weighted Maxwell
adjoint used

`dI_PTE/dQ_thermal -> R_Q^T -> native PABS Yee vector source`.

The optical density gradient used explicit component operators
`J_c = d epsilon_Yee,c / d rho_81x81`. Forward field, adjoint field,
`epsilon_c`, and clipped `dV_c` were paired on the same component-specific
PABS coordinates. The removed DESIGN-monitor same-index path was not used.

## Per-mesh values

| optical dz | thermal scenario | P_Q (W) | six-face closure | PTE (A) | optical directional gradient (A) | combined directional gradient (A) |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Convergence

| dz comparison (nm) | scenario | remapped-Q NRMSE | TaIrTe4 T-field NRMSE | raw PTE relative change | optical gradient relative change | combined gradient relative change |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(comparisons)}

The production optical mesh is therefore:

`flake_dz_nm = {result["production_flake_dz_nm"]:g}`

This is the coarsest mesh whose raw PTE, optical directional gradient, and
combined directional gradient are all within 0.5% of the 0.625 nm reference
for both named thermal footprints. No empirical normalization, gradient
rescaling, clipping, smoothing, gain, global Q rescaling, tiling, or Q-source
deletion was used.

Worst layout JVP/VJP dot error:
`{result["gates"]["worst_JVP_VJP_dot_relative_error"]:.9e}`.

Raw FSP/NPZ/J artifacts remain outside Git and are SHA-256 pinned in the
manifest. This checkpoint does not run gray-law sensitivity, latent AD-FD, or
optimization.
"""
    report_path = (
        output
        / "OPTICAL_DZ_DOWNSTREAM_PTE_GRADIENT_CONVERGENCE_REPORT.md"
    )
    report_path.write_text(report)
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": str(summary_path),
                "csv": str(csv_path),
                "manifest": str(manifest_path),
                "report": str(report_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
