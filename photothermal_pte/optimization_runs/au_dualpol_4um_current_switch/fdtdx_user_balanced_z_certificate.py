#!/usr/bin/env python3
"""Byte-bound baseline-to-z2 comparison for the user-balanced FDTDX track."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    TimeSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_full_z_certificate import (
    compare_full_z_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_time_settling_certificate import (
    fixed_probe_and_weights,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_exact_binary_control import (
    _json_default,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_exact_binary import (
    DEFAULT_REFERENCE,
    STATUS_READY as MATERIAL_STATUS_READY,
    VERSION as MATERIAL_VERSION,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_mesh import (
    mesh_audit as baseline_mesh_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_source_only import (
    balanced_case_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_source_pair import (
    sha256,
    validate_source_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_refinement import (
    case_contract as z_case_contract,
    mesh_audit as z_mesh_audit,
)


VERSION = "fdtdx-user-balanced-z-certificate-v1"
STATUS_CONVERGED = "VALIDATED_FDTDX_USER_BALANCED_Z_CONVERGENCE"
STATUS_BLOCKED = "VALIDATED_BLOCKED_FDTDX_USER_BALANCED_Z_CONVERGENCE"
STATUS_INVALID = "INVALID_FDTDX_USER_BALANCED_Z_CERTIFICATE"
CERTIFICATE_NAME = "FDTDX_USER_BALANCED_Z1_TO_Z2_CERTIFICATE.json"
POLARIZATIONS = ("Ea", "Eb")
LEVELS = ("z1", "z2")
EXPECTED_FACTOR = {"z1": 1, "z2": 2, "z4": 4}
REQUIRED_RAW_ARRAYS = {
    "au_late",
    "tairte4_late",
    "target",
    "design_mask",
    "solver_mask",
    "q_au_late_W_m3",
    "q_tairte4_late_W_m3",
    "electric_dual_volume_au_m3",
    "electric_dual_volume_tairte4_m3",
    "grid_x_edges_m",
    "grid_y_edges_m",
    "grid_z_edges_m",
}


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_audit(path: Path, expected_sha256: str) -> dict[str, Any]:
    supplied = path.expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_file()
    actual = sha256(resolved) if exists else None
    checks = {
        "path_is_absolute": supplied.is_absolute(),
        "file_exists": exists,
        "expected_sha256_is_hex": _is_sha256(expected_sha256),
        "sha256_matches": actual == expected_sha256,
    }
    return {
        "path": str(resolved),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "checks": checks,
        "ready": all(checks.values()),
    }


def _runner_blob_audit(payload: dict[str, Any], repository: Path) -> dict[str, Any]:
    provenance = payload.get("provenance", {})
    commit = provenance.get("repository_commit")
    runner = Path(provenance.get("runner_path", "")).expanduser().resolve()
    recorded = provenance.get("runner_sha256")
    relative = None
    blob = None
    error = None
    try:
        relative = runner.relative_to(repository)
        blob = subprocess.run(
            ("git", "-C", str(repository), "show", f"{commit}:{relative}"),
            check=True,
            capture_output=True,
        ).stdout
    except (ValueError, subprocess.CalledProcessError, TypeError) as exception:
        error = repr(exception)
    actual = hashlib.sha256(blob).hexdigest() if blob is not None else None
    checks = {
        "commit_is_sha": _is_sha256(commit)
        or (
            isinstance(commit, str)
            and 7 <= len(commit) <= 40
            and all(character in "0123456789abcdef" for character in commit)
        ),
        "runner_is_under_repository": relative is not None,
        "historical_blob_exists": blob is not None,
        "recorded_runner_sha256_is_hex": _is_sha256(recorded),
        "historical_blob_sha256_matches": actual == recorded,
    }
    return {
        "repository_commit": commit,
        "runner_path": str(runner),
        "runner_relative_path": str(relative) if relative is not None else None,
        "recorded_sha256": recorded,
        "historical_blob_sha256": actual,
        "error": error,
        "checks": checks,
        "ready": all(checks.values()),
    }


def _load_raw_snapshot(
    payload: dict[str, Any], raw_audit: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    declared = payload.get("raw", {}).get("arrays", {})
    checks: dict[str, bool] = {
        "raw_file_audit_ready": raw_audit["ready"],
        "required_arrays_declared": REQUIRED_RAW_ARRAYS.issubset(declared),
    }
    snapshot = None
    error = None
    if all(checks.values()):
        try:
            with np.load(raw_audit["path"], allow_pickle=False) as archive:
                checks["declared_schema_exact"] = set(archive.files) == set(declared)
                checks["declared_shapes_exact"] = checks[
                    "declared_schema_exact"
                ] and all(
                    list(np.asarray(archive[name]).shape) == declared[name]
                    for name in archive.files
                )
                checks["all_raw_values_finite"] = checks[
                    "declared_schema_exact"
                ] and all(
                    bool(np.all(np.isfinite(np.asarray(archive[name]))))
                    for name in archive.files
                )
                edges = tuple(
                    np.asarray(archive[f"grid_{axis}_edges_m"]) for axis in "xyz"
                )
                probe, probe_weights, _ = fixed_probe_and_weights(
                    np.asarray(archive["target"]), edges, payload["placement"]
                )
                snapshot = {
                    "grid_edges": edges,
                    "probe": probe,
                    "probe_weights": probe_weights,
                    "fields_late": {
                        "au": np.asarray(archive["au_late"]),
                        "tairte4": np.asarray(archive["tairte4_late"]),
                    },
                    "q_late": {
                        "au": np.asarray(archive["q_au_late_W_m3"]),
                        "tairte4": np.asarray(archive["q_tairte4_late_W_m3"]),
                    },
                    "volumes": {
                        "au": np.asarray(archive["electric_dual_volume_au_m3"]),
                        "tairte4": np.asarray(
                            archive["electric_dual_volume_tairte4_m3"]
                        ),
                    },
                    "power_late": payload["evaluation"]["Q"]["late"],
                    "design_mask": np.asarray(archive["design_mask"]),
                    "solver_mask": np.asarray(archive["solver_mask"]),
                }
                checks["fixed_probe_finite"] = bool(
                    np.all(np.isfinite(probe))
                    and np.all(np.isfinite(probe_weights))
                    and np.all(probe_weights > 0.0)
                )
        except (KeyError, OSError, TypeError, ValueError) as exception:
            error = repr(exception)
    checks["snapshot_built"] = snapshot is not None
    audit = {
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "error": error,
        "ready": bool(checks) and all(checks.values()),
    }
    return snapshot if audit["ready"] else None, audit


def _material_audit(
    report_path: Path,
    expected_sha256: str,
    level: str,
    polarization: str,
    source_pair_audit: dict[str, Any],
    repository: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    report = _file_audit(report_path, expected_sha256)
    payload = (
        json.loads(Path(report["path"]).read_text(encoding="utf-8"))
        if report["ready"]
        else {}
    )
    factor = EXPECTED_FACTOR[level]
    expected_time = TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5)
    expected_case = (
        balanced_case_contract(expected_time)
        if factor == 1
        else z_case_contract(expected_time, factor)
    )
    expected_mesh = baseline_mesh_audit() if factor == 1 else z_mesh_audit(factor)
    raw_record = payload.get("raw", {})
    raw = _file_audit(Path(raw_record.get("path", "")), raw_record.get("sha256", ""))
    snapshot, raw_snapshot = _load_raw_snapshot(payload, raw)
    gates = payload.get("evaluation", {}).get("gates", {})
    provenance_checks = payload.get("provenance_checks", {})
    material_checks = payload.get("material", {}).get("checks", {})
    runner_blob = _runner_blob_audit(payload, repository)
    recorded_pair = payload.get("source_pair", {})
    checks = {
        "report_ready": report["ready"],
        "version_status_ready": payload.get("version") == MATERIAL_VERSION
        and payload.get("status") == MATERIAL_STATUS_READY
        and payload.get("ready") is True,
        "labels_exact": payload.get("polarization") == polarization
        and payload.get("reference") == DEFAULT_REFERENCE
        and payload.get("full_domain_z_factor", 1) == factor,
        "numerical_case_exact": payload.get("numerical_case_contract") == expected_case,
        "mesh_exact": payload.get("mesh") == expected_mesh,
        "source_pair_revalidation_ready": source_pair_audit["ready"],
        "recorded_source_pair_binding_exact": recorded_pair.get("path")
        == source_pair_audit["path"]
        and recorded_pair.get("actual_sha256") == source_pair_audit["actual_sha256"]
        and recorded_pair.get("ready") is True,
        "evaluation_gates_all_true": bool(gates)
        and all(value is True for value in gates.values())
        and payload.get("evaluation", {}).get("failed_gates") == [],
        "provenance_checks_all_true": bool(provenance_checks)
        and all(value is True for value in provenance_checks.values()),
        "material_checks_all_true": bool(material_checks)
        and all(value is True for value in material_checks.values()),
        "exact_binary_no_gray": payload.get("material", {})
        .get("exact_binary_au", {})
        .get("gray_density_allowed")
        is False,
        "repository_clean_before_after": payload.get("provenance", {}).get(
            "repository_dirty_porcelain_before"
        )
        == payload.get("provenance", {}).get("repository_dirty_porcelain_after")
        == "",
        "historical_runner_blob_ready": runner_blob["ready"],
        "raw_snapshot_ready": raw_snapshot["ready"],
    }
    audit = {
        "report": report,
        "raw": raw,
        "raw_snapshot": raw_snapshot,
        "historical_runner_blob": runner_blob,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
    }
    return payload, audit, snapshot if audit["ready"] else None


def build_certificate(
    source_pair_paths: dict[str, Path],
    source_pair_hashes: dict[str, str],
    report_paths: dict[str, dict[str, Path]],
    report_hashes: dict[str, dict[str, str]],
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    time = TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5)
    source_pairs: dict[str, Any] = {}
    source_audits: dict[str, Any] = {}
    for level in LEVELS:
        factor = EXPECTED_FACTOR[level]
        kwargs = (
            {}
            if factor == 1
            else {
                "expected_case_contract": z_case_contract(time, factor),
                "expected_mesh": z_mesh_audit(factor),
            }
        )
        source_pairs[level], source_audits[level] = validate_source_pair(
            source_pair_paths[level], source_pair_hashes[level], time, **kwargs
        )

    payloads: dict[str, Any] = {level: {} for level in LEVELS}
    case_audits: dict[str, Any] = {level: {} for level in LEVELS}
    snapshots: dict[str, Any] = {level: {} for level in LEVELS}
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
            "z1",
            "z2",
            snapshots,
            payloads,
            source_pairs,
            z_factors=EXPECTED_FACTOR,
            successive_pairs=(("z1", "z2"),),
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
        "version": VERSION,
        "status": status,
        "certificate_valid": certificate_valid,
        "convergence_pass": convergence_pass,
        "mesh_selected": "z1" if convergence_pass else None,
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
    payload = build_certificate(
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
