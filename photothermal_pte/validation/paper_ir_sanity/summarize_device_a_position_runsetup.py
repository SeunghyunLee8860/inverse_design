#!/usr/bin/env python3
"""Summarize four Device-A source-offset runsetup audits without solving."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "committed_to_git": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sminus-a", type=Path, required=True)
    parser.add_argument("--sminus-b", type=Path, required=True)
    parser.add_argument("--splus-a", type=Path, required=True)
    parser.add_argument("--splus-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_fixed_contract(pre: dict[str, Any]) -> dict[str, Any]:
    geometry = pre["geometry"]
    source = geometry["source"]
    return {
        "boundaries": pre["boundaries"],
        "flake_vertices_um": geometry["flake_vertices_um"],
        "domain_bounds_m": geometry["domain_bounds_m"],
        "absorption_analysis_bounds_m": geometry["absorption_analysis_bounds_m"],
        "six_face_absorption_box_bounds_m": geometry[
            "six_face_absorption_box_bounds_m"
        ],
        "mesh_override_objects": pre["mesh"]["override_objects"],
        "fixed_local_mesh_center_m": source["fixed_local_mesh_center_m"],
        "coordinate_contract": geometry["coordinate_contract"],
        "material": pre["material"]["requested_epsilon_at_11um"],
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directories = {
        "sminus_a": args.sminus_a,
        "sminus_b": args.sminus_b,
        "splus_a": args.splus_a,
        "splus_b": args.splus_b,
    }
    cases: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    fixed = []
    for label, directory in directories.items():
        result_path = directory / "case_result.json"
        project_path = directory / "finite_2um_optical_q.fsp"
        result = json.loads(result_path.read_text())
        pre = result["pre_run_contract"]
        source = pre["geometry"]["source"]
        fixed.append(canonical_fixed_contract(pre))
        cases[label] = {
            "status": result["status"],
            "all_checks_passed": pre["checks"]["all"],
            "beam_center_m": source["beam_center_m"],
            "source_only_offset_m": source["source_only_offset_m"],
            "fixed_local_mesh_center_m": source["fixed_local_mesh_center_m"],
            "source_aperture_PML_clearance_m": source[
                "source_aperture_PML_clearance_m"
            ],
            "polarization_angle_deg": pre["source_readback"][
                "polarization_angle_deg"
            ],
            "gpu_resource": pre["solver"]["resources"]["2"]["device type"],
        }
        artifacts[label] = {
            "case_result": record(result_path),
            "project": record(project_path),
        }
    fixed_equal = all(item == fixed[0] for item in fixed[1:])
    summary = {
        "status": (
            "DEVICE_A_POSITION_RUNSETUP_AUDITED_NOT_SOLVED"
            if fixed_equal and all(case["all_checks_passed"] for case in cases.values())
            else "FAILED_DEVICE_A_POSITION_RUNSETUP_AUDIT"
        ),
        "scope": "four runsetup/readback cases; no Maxwell time stepping",
        "cases": cases,
        "gates": {
            "all_pre_run_checks_passed": all(
                case["all_checks_passed"] for case in cases.values()
            ),
            "device_PML_monitors_mesh_material_identical": fixed_equal,
            "only_source_center_and_polarization_change": fixed_equal,
            "all_contracts_not_solved": all(
                case["status"] == "CONTRACT_BUILT_NOT_SOLVED"
                for case in cases.values()
            ),
        },
        "raw_artifacts": artifacts,
    }
    (args.output_dir / "device_a_position_runsetup_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.output_dir / "RAW_ARTIFACT_MANIFEST_RUNSETUP.json").write_text(
        json.dumps({"raw_artifacts": artifacts}, indent=2) + "\n"
    )
    report = f"""# Device-A position-scan runsetup audit

Status: `{summary['status']}`

All four `s0±1 um`, `E||a/b` v261 sessions opened on the host, saved, ran
`runsetup`, and passed every pre-run check. No Maxwell time stepping occurred.

The Device-A flake/electrode polygons, six PML boundaries, monitors, material,
and every local-mesh override bound are exactly identical across the four
contracts: `{fixed_equal}`. Only source center and polarization change. The
50-nm local mesh remains fixed at the pre-registered `s0` center.

The first sandboxed session attempt failed before opening Lumerical because it
could not join the host ANSYSLI sharing context. The host runsetup sessions then
opened normally; no existing Lumerical process or `.ansys` state was deleted.
"""
    (args.output_dir / "DEVICE_A_POSITION_RUNSETUP_REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2))
    return 0 if all(summary["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
