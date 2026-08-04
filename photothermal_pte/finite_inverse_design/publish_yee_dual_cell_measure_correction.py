#!/usr/bin/env python3
"""Publish the full-Yee-dual-cell gradient-measure correction checkpoint."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .run_v261_large_background_tfsf_forward import sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-result", required=True)
    parser.add_argument("--raw-sha256", required=True)
    parser.add_argument("--failed-stage10", required=True)
    parser.add_argument("--failed-stage10-sha256", required=True)
    parser.add_argument("--failed-strong", required=True)
    parser.add_argument("--failed-strong-sha256", required=True)
    parser.add_argument("--report-dir", required=True)
    return parser.parse_args()


def checked(path_text: str, expected: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"missing or SHA-mismatched artifact: {path}")
    return path


def artifact(path: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path),
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    args = parse_args()
    raw_path = checked(args.raw_result, args.raw_sha256)
    stage10_path = checked(
        args.failed_stage10, args.failed_stage10_sha256
    )
    strong_path = checked(
        args.failed_strong, args.failed_strong_sha256
    )
    raw = json.loads(raw_path.read_text())
    stage10 = json.loads(stage10_path.read_text())
    strong = json.loads(strong_path.read_text())
    if (
        raw["status"] != "VALIDATED_FULL_YEE_DUAL_CELL_GRADIENT_MEASURE"
        or not raw["passed"]
    ):
        raise RuntimeError("raw correction result is not validated")
    if stage10.get("passed") or strong.get("passed"):
        raise RuntimeError("diagnostic failure checkpoint was overwritten")

    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "yee_dual_cell_measure_correction_summary.json"
    cases_path = report_dir / "yee_dual_cell_measure_correction_cases.csv"
    report_path = report_dir / "YEE_DUAL_CELL_GRADIENT_MEASURE_CORRECTION.md"
    manifest_path = (
        report_dir
        / "YEE_DUAL_CELL_GRADIENT_MEASURE_CORRECTION_RAW_ARTIFACT_MANIFEST.json"
    )
    rows = []
    for scenario, value in raw["scenarios"].items():
        for row in value["rows"]:
            rows.append(
                {
                    "scenario": scenario,
                    "step": row["step"],
                    "clipped_optical_error": row[
                        "clipped_design_box"
                    ]["optical_relative_error"],
                    "clipped_combined_error": row[
                        "clipped_design_box"
                    ]["combined_relative_error"],
                    "full_optical_error": row[
                        "full_yee_dual_cell"
                    ]["optical_relative_error"],
                    "full_combined_error": row[
                        "full_yee_dual_cell"
                    ]["combined_relative_error"],
                }
            )
    with cases_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": raw["status"],
        "passed": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "solver-free correction of the Maxwell material-gradient "
            "integration measure using SHA-pinned completed FSPs"
        ),
        "root_cause": raw["root_cause"],
        "correct_measure": raw["correct_measure"],
        "gates": raw["gates"],
        "component_J_support": raw["component_J_support"],
        "stage10_preserved": {
            "status": stage10["status"],
            "passed": stage10["passed"],
            "reported_relative_error_range": [0.0215, 0.0289],
        },
        "inverse_collocation_strong_diagnostic_preserved": {
            "status": strong["status"],
            "passed": strong["passed"],
            "worst_strong_direction_relative_error": strong["gates"][
                "worst_strong_direction_relative_error"
            ],
        },
        "rho05_representation_equivalence_checkpoint": (
            "already validated at commit 6a73dea; not rerun"
        ),
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "gray_law_sensitivity_run": False,
        "full_latent_adfd_run": False,
        "optimization_run": False,
        "combined_multidirection_gate_complete": False,
        "next_gate": (
            "corrected strong and five-direction physical-rho combined "
            "AD-FD using full Yee dual-cell volumes"
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "status": raw["status"],
        "generation_command": (
            "python -m photothermal_pte.finite_inverse_design."
            "diagnose_yee_dual_cell_measure_correction "
            "--base-forward <SHA-pinned FSP> --component-split "
            "<SHA-pinned JSON> --jacobian-dir <SHA-pinned J directory> "
            "--adjoint-4um <SHA-pinned FSP> --adjoint-6um "
            "<SHA-pinned FSP> --output-dir <external>"
        ),
        "raw_artifacts": [
            artifact(raw_path, "full_yee_measure_raw_result"),
            artifact(stage10_path, "immutable_failed_stage10"),
            artifact(strong_path, "failed_inverse_collocation_strong_gate"),
            raw["base_forward"],
            raw["component_split"],
            *[
                raw["scenarios"][scenario]["adjoint_FSP"]
                | {"role": f"official_fieldregion_adjoint_{scenario}"}
                for scenario in ("4um", "6um")
            ],
        ],
        "raw_NPZ_or_FSP_committed_to_git": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    table = "\n".join(
        "| {scenario} | {step:g} | {clipped_combined_error:.6e} | "
        "{full_combined_error:.6e} |".format(**row)
        for row in rows
    )
    report_path.write_text(
        f"""# Yee dual-cell gradient-measure correction

Status: `{raw['status']}`

The failed Stage 10 result remains an immutable diagnostic. No empirical
normalization, fitted factor, or gradient rescaling was used.

## Root cause

`J_c = d epsilon_Yee,c / d rho` already includes the conformal material fill
and exact design-support intersection. The old combined path multiplied the
forward/adjoint product by a Yee volume clipped again to the nominal
`[-1,1] um` design box. That applied the support fraction twice. The corrected
bilinear form uses the complete component-specific Yee dual-cell volume.

| thermal scenario | FD step | old clipped combined error | corrected full-Yee combined error |
|---|---:|---:|---:|
{table}

The worst corrected optical/combined directional error is
`{raw['gates']['worst_full_measure_ADFD_relative_error']:.6e}`. The old
clipped computation is reproduced with relative error
`{raw['gates']['clipped_path_reproduction_relative_error']:.3e}`, so the
change is isolated to the integration measure.

## Support audit

The explicit `J_x`, `J_y`, and `J_z` operators remain unchanged. Their active
row counts and the number of active rows incorrectly reduced by clipping are:

- x: `{raw['component_J_support']['x']['active_J_row_count']}` active,
  `{raw['component_J_support']['x']['active_rows_changed_by_clipping']}` changed.
- y: `{raw['component_J_support']['y']['active_J_row_count']}` active,
  `{raw['component_J_support']['y']['active_rows_changed_by_clipping']}` changed.
- z: `{raw['component_J_support']['z']['active_J_row_count']}` active,
  `{raw['component_J_support']['z']['active_rows_changed_by_clipping']}` changed.

This checkpoint validates the corrected measure for the existing smooth
direction at both 4 and 6 um thermal scenarios. It does not claim completion
of the strong/five-direction combined physical-density gate. Gray-law
sensitivity, latent AD-FD, and optimization remain blocked.
"""
    )
    print(
        json.dumps(
            {
                "status": raw["status"],
                "summary": str(summary_path),
                "cases": str(cases_path),
                "report": str(report_path),
                "manifest": str(manifest_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
