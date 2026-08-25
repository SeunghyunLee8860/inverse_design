#!/usr/bin/env python3
"""Byte-bound z2-to-z4 tail certificate for the user-balanced track."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    TimeSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_full_z_certificate import (
    compare_full_z_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_exact_binary_control import (
    _json_default,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_source_pair import (
    sha256,
    validate_source_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_certificate import (
    POLARIZATIONS,
    STATUS_BLOCKED,
    STATUS_CONVERGED,
    STATUS_INVALID,
    VERSION,
    _git,
    _material_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_refinement import (
    case_contract,
    mesh_audit,
)


TAIL_VERSION = "fdtdx-user-balanced-z2-z4-tail-certificate-v1"
LEVELS = ("z2", "z4")
FACTORS = {"z2": 2, "z4": 4}
CERTIFICATE_NAME = "FDTDX_USER_BALANCED_Z2_TO_Z4_CERTIFICATE.json"


def build_tail_certificate(
    source_pair_paths: dict[str, Path],
    source_pair_hashes: dict[str, str],
    report_paths: dict[str, dict[str, Path]],
    report_hashes: dict[str, dict[str, str]],
) -> dict:
    repository = Path(__file__).resolve().parents[3]
    time = TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5)
    source_pairs = {}
    source_audits = {}
    for level in LEVELS:
        factor = FACTORS[level]
        source_pairs[level], source_audits[level] = validate_source_pair(
            source_pair_paths[level],
            source_pair_hashes[level],
            time,
            expected_case_contract=case_contract(time, factor),
            expected_mesh=mesh_audit(factor),
        )

    payloads = {level: {} for level in LEVELS}
    case_audits = {level: {} for level in LEVELS}
    snapshots = {level: {} for level in LEVELS}
    for level in LEVELS:
        for polarization in POLARIZATIONS:
            payload, audit, snapshot = _material_audit(
                report_paths[level][polarization],
                report_hashes[level][polarization],
                level,
                polarization,
                source_audits[level],
                repository,
            )
            payloads[level][polarization] = payload
            case_audits[level][polarization] = audit
            snapshots[level][polarization] = snapshot

    artifact_checks = {
        "repository_clean_while_certifying": _git(
            repository, "status", "--porcelain", "--untracked-files=all"
        )
        == "",
        "both_source_pairs_revalidate": all(
            audit["ready"] for audit in source_audits.values()
        ),
        "all_four_material_cases_revalidate": all(
            case_audits[level][polarization]["ready"]
            for level in LEVELS
            for polarization in POLARIZATIONS
        ),
    }
    comparison = None
    if all(artifact_checks.values()):
        comparison = compare_full_z_pair(
            "z2",
            "z4",
            snapshots,
            payloads,
            source_pairs,
            z_factors=FACTORS,
            successive_pairs=(("z2", "z4"),),
        )
    certificate_valid = all(artifact_checks.values()) and comparison is not None
    convergence_pass = certificate_valid and comparison["pass"] is True
    status = (
        STATUS_CONVERGED
        if convergence_pass
        else STATUS_BLOCKED
        if certificate_valid
        else STATUS_INVALID
    )
    generator = Path(__file__).resolve()
    return {
        "version": TAIL_VERSION,
        "comparison_engine_version": VERSION,
        "status": status,
        "certificate_valid": certificate_valid,
        "convergence_pass": convergence_pass,
        "mesh_selected": "z2" if convergence_pass else None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_checks": artifact_checks,
        "failed_artifact_checks": [
            name for name, passed in artifact_checks.items() if not passed
        ],
        "source_pairs": source_audits,
        "material_cases": case_audits,
        "comparison": comparison,
        "optimizer_start_allowed": False,
        "provenance": {
            "repository_commit": _git(repository, "rev-parse", "HEAD"),
            "repository_dirty_porcelain": _git(
                repository, "status", "--porcelain", "--untracked-files=all"
            ),
            "generator_path": str(generator),
            "generator_sha256": sha256(generator),
            "lumerical_used": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for level in LEVELS:
        parser.add_argument(f"--{level}-source-pair", type=Path, required=True)
        parser.add_argument(f"--{level}-source-pair-sha256", required=True)
        for polarization in POLARIZATIONS:
            parser.add_argument(
                f"--{level}-{polarization.lower()}-report", type=Path, required=True
            )
            parser.add_argument(
                f"--{level}-{polarization.lower()}-report-sha256", required=True
            )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser()
    if not output.is_absolute() or not output.parent.is_dir() or output.exists():
        parser.error("--output must be a new absolute file under an existing directory")
    source_paths = {level: getattr(args, f"{level}_source_pair") for level in LEVELS}
    source_hashes = {
        level: getattr(args, f"{level}_source_pair_sha256") for level in LEVELS
    }
    report_paths = {
        level: {
            polarization: getattr(args, f"{level}_{polarization.lower()}_report")
            for polarization in POLARIZATIONS
        }
        for level in LEVELS
    }
    report_hashes = {
        level: {
            polarization: getattr(args, f"{level}_{polarization.lower()}_report_sha256")
            for polarization in POLARIZATIONS
        }
        for level in LEVELS
    }
    payload = build_tail_certificate(
        source_paths, source_hashes, report_paths, report_hashes
    )
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "status": payload["status"],
                "certificate_valid": payload["certificate_valid"],
                "convergence_pass": payload["convergence_pass"],
                "failed_artifact_checks": payload["failed_artifact_checks"],
                "failed_comparison_checks": (
                    [
                        name
                        for name, passed in payload["comparison"]["checks"].items()
                        if not passed
                    ]
                    if payload["comparison"] is not None
                    else None
                ),
            },
            default=_json_default,
        )
    )
    return 0 if payload["certificate_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
