#!/usr/bin/env python3
"""Publish the 81x81 nodal optical/thermal coupling certificate."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-summary", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--generation-command", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    raw_path = Path(args.raw_summary).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    raw = json.loads(raw_path.read_text())
    if not raw.get("passed"):
        raise RuntimeError(f"refusing failed mapping: {raw['status']}")
    report_path = report_dir / "NODAL_PHYSICAL_COUPLING_REPORT.md"
    summary_path = report_dir / "nodal_physical_coupling_summary.json"
    csv_path = report_dir / "nodal_physical_coupling_cases.csv"
    manifest_path = (
        report_dir
        / "NODAL_PHYSICAL_COUPLING_RAW_ARTIFACT_MANIFEST.json"
    )
    rows = []
    for record in raw["optical_mapping"]["directions"]:
        rows.append(
            {
                "target": "optical_81x81x13_nodes",
                "resolution_nm": 25.0,
                **record,
            }
        )
    for mapping in raw["thermal_mappings"]:
        for record in mapping["directions"]:
            rows.append(
                {
                    "target": "thermal_cell_average",
                    "resolution_nm": mapping["core_xy_cell_size_nm"],
                    **record,
                }
            )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    published = {
        "status": raw["status"],
        "passed": raw["passed"],
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": raw["scope"],
        "physical_density_contract": raw["physical_density_contract"],
        "optical_mapping": raw["optical_mapping"],
        "thermal_mappings": raw["thermal_mappings"],
        "gates": raw["gates"],
        "next_gate": raw["next_gate"],
        "not_claimed": [
            "filter/projection",
            "permittivity import equivalence",
            "Maxwell solve",
            "thermal solve",
            "combined AD-FD",
            "optimization",
        ],
    }
    summary_path.write_text(json.dumps(published, indent=2) + "\n")

    mapping_lines = []
    for mapping in raw["thermal_mappings"]:
        mapping_lines.append(
            "| "
            + " | ".join(
                [
                    f"{mapping['core_xy_cell_size_nm']:g}",
                    "×".join(str(value) for value in mapping["shape"]),
                    f"{mapping['rho1_max_abs_error']:.3e}",
                    f"{mapping['affine_cell_average_max_abs_error']:.3e}",
                    f"{mapping['area_integral_relative_error']:.3e}",
                    f"{mapping['corner_to_opposite_corner_value']:.3e}",
                ]
            )
            + " |"
        )
    direction_lines = [
        "| "
        + " | ".join(
            [
                row["target"],
                f"{row['resolution_nm']:g}",
                row["direction"],
                f"{row['JVP_centered_FD_relative_error']:.6e}",
                f"{row['JVP_VJP_dot_relative_error']:.6e}",
            ]
        )
        + " |"
        for row in rows
    ]
    gates = raw["gates"]
    report = rf"""# 81×81 nodal physical-density optical/thermal coupling

Status: `{raw['status']}`

## Coordinate contract

The physical design variable is exactly 81×81 **nodes** on
\([-1,1]\) µm × \([-1,1]\) µm at 25 nm spacing.  It is not 81
finite-width pixels and has no periodic fencepost or wrap.

The optical map is identity on those x-y nodes and exact repetition on 13
z nodes from 0 to 600 nm at 50 nm spacing.  Its VJP is the literal sum over
the same z copies.

The thermal map is the exact area average of the nonperiodic
piecewise-bilinear nodal interpolant over each Cartesian control volume.
The transpose is the literal sparse-matrix transpose.  Target and source
bounds must match exactly, so cropping, padding, gain, or tiling fail closed.

## Endpoint, affine, conservation, and non-wrap controls

Optical rho=0 / rho=1 / z-extrusion maximum errors:
`{raw['optical_mapping']['rho0_max_abs_error']:.3e} /
{raw['optical_mapping']['rho1_max_abs_error']:.3e} /
{raw['optical_mapping']['z_extrusion_max_abs_error']:.3e}`.

| thermal cell (nm) | shape | rho=1 error | affine-average error | area-integral error | opposite-corner leakage |
| ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(mapping_lines)}

## JVP, centered FD, and VJP

| target | resolution (nm) | direction | JVP–FD relative error | JVP–VJP dot error |
| --- | ---: | --- | ---: | ---: |
{chr(10).join(direction_lines)}

## Gates

- Worst JVP–FD error:
  `{gates['worst_JVP_centered_FD_relative_error']:.6e}`
  (limit `{gates['JVP_centered_FD_limit']:.6e}`).
- Worst JVP–VJP transpose error:
  `{gates['worst_JVP_VJP_dot_relative_error']:.6e}`
  (limit `{gates['transpose_limit']:.6e}`).
- Worst endpoint constant error:
  `{gates['worst_endpoint_constant_error']:.6e}`.
- Worst area-integral error:
  `{gates['worst_area_integral_relative_error']:.6e}`.
- Opposite-boundary leakage:
  `{gates['worst_opposite_boundary_leakage']:.6e}`.

This is a solver-free coupling certificate.  Imported-permittivity endpoint
equivalence is the next fail-closed gate.
"""
    report_path.write_text(report)
    manifest = {
        "status": raw["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_command": args.generation_command,
        "raw_summary": {
            "path": str(raw_path),
            "byte_size": raw_path.stat().st_size,
            "sha256": sha256(raw_path),
        },
        "raw_artifact": raw["raw_artifact"],
        "git_policy": (
            "raw NPZ remains outside Git; path, size, and SHA-256 are "
            "recorded here"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": raw["status"],
                "report": str(report_path),
                "summary": str(summary_path),
                "csv": str(csv_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
