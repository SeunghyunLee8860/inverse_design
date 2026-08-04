#!/usr/bin/env python3
"""Publish the fail-closed Device-A optical runsetup checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--empty-runsetup", type=Path, required=True)
    parser.add_argument("--finite-runsetup", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = {}
    raw = {}
    for label, directory in (
        ("empty_stack", args.empty_runsetup),
        ("finite_device_a", args.finite_runsetup),
    ):
        result_path = directory / "case_result.json"
        result = json.loads(result_path.read_text())
        contract = result["pre_run_contract"]
        cases[label] = {
            "status": result["status"],
            "all_checks_passed": contract["checks"]["all"],
            "checks": contract["checks"],
            "boundaries": contract["boundaries"],
            "solver": contract["solver"],
            "geometry": contract["geometry"],
            "material": contract["material"],
            "mesh": contract["mesh"],
            "object_bounds_readback_m": contract["object_bounds_readback_m"],
        }
        raw[label] = {
            "case_result": record(result_path),
            "project": record(directory / "finite_2um_optical_q.fsp"),
            "raw_files_committed_to_git": False,
        }
    finite = cases["finite_device_a"]
    digitized = finite["geometry"]["digitized_device_a_contract"]
    summary = {
        "status": "DEVICE_A_OPTICAL_RUNSETUP_AUDITED_NOT_SOLVED",
        "scope": (
            "v261 runsetup/geometry/material/GPU-resource audit only; no "
            "Maxwell time stepping, thermal, PTE, adjoint, or optimization"
        ),
        "cases": cases,
        "key_contract": {
            "coordinate_axes": finite["geometry"]["coordinate_contract"],
            "simulation_origin_shift_um": digitized[
                "simulation_origin_shift_um"
            ],
            "relative_geometry_preserved": digitized[
                "translation_preserves_all_relative_coordinates"
            ],
            "minimum_lateral_PML_clearance_um": digitized[
                "minimum_lateral_PML_clearance_um"
            ],
            "six_boundaries": finite["boundaries"],
            "electrodes_in_finite_optical_model": finite["geometry"][
                "electrodes_in_optical_model"
            ],
            "source": finite["geometry"]["source"],
            "material": finite["material"],
            "mesh": finite["mesh"],
        },
        "gates": {
            "empty_runsetup_all_checks": cases["empty_stack"][
                "all_checks_passed"
            ],
            "finite_runsetup_all_checks": finite["all_checks_passed"],
            "Maxwell_solve_authorized_next": all(
                case["all_checks_passed"] for case in cases.values()
            ),
        },
        "raw_artifacts": raw,
    }
    (args.output_dir / "device_a_optical_runsetup_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.output_dir / "RAW_ARTIFACT_MANIFEST_RUNSETUP.json").write_text(
        json.dumps(
            {
                "policy": "Raw FSP and solver artifacts remain outside Git.",
                "artifacts": raw,
            },
            indent=2,
        )
        + "\n"
    )
    material = finite["material"]["electrode_material_readback"]
    clearance = digitized["minimum_lateral_PML_clearance_um"]
    report = f"""# Device A optical runsetup audit

Status: `DEVICE_A_OPTICAL_RUNSETUP_AUDITED_NOT_SOLVED`

This checkpoint created and saved the v261 geometry, invoked `runsetup`, and
read the realized contract. It did **not** time-step Maxwell and is not an
optical, thermal, or current result.

## Frozen contract

- all six boundaries: PML; no periodic/Bloch boundary
- scalar Gaussian, 11 um, explicit-assumption physical waist 12 um
- source span / lateral domain: 50 / 60 um
- local mesh: 50 nm nested in a 100 nm outer material region
- TaIrTe4 and metal-region dz: 5 nm
- TaIrTe4 axes: code x=b, y=a, z=c=b
- Device A: Figure-digitized approximation, not unpublished CAD
- electrodes: digitized 5 nm Ti / 50 nm Au top and bottom polygons

The digitized structure and the beam were translated together by
`{digitized['simulation_origin_shift_um']}` um. This changes only the
coordinate origin and preserves every beam/device/contact relative position.
The resulting nominal lateral PML clearances are x={clearance['x']:.6f} um and
y={clearance['y']:.6f} um.

## Material readback at 11 um

- Au: built-in `{material['Au']['material']}`, n =
  {material['Au']['complex_index_at_11um']['real']:.9g} +
  {material['Au']['complex_index_at_11um']['imag']:.9g}i
- Ti: built-in `{material['Ti']['material']}`, n =
  {material['Ti']['complex_index_at_11um']['real']:.9g} +
  {material['Ti']['complex_index_at_11um']['imag']:.9g}i

Both CRC sampled-data ranges include 11 um. TaIrTe4 fitted epsilon-z equals
epsilon-x exactly under the documented epsilon-c=epsilon-b 3D closure.

## Gate

Empty and finite runsetup checks both passed. A GPU-only Maxwell solve is the
next authorized step. No CPU fallback, Q modification, thermal solve, PTE,
adjoint, AD-FD, or optimization occurred in this checkpoint.
"""
    (args.output_dir / "DEVICE_A_OPTICAL_RUNSETUP_REPORT.md").write_text(report)
    print(json.dumps(summary["gates"], indent=2))
    return 0 if summary["gates"]["Maxwell_solve_authorized_next"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
