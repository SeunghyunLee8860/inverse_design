#!/usr/bin/env python3
"""Fail-closed full-domain-z certificate for increment-state exact-binary data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any, Mapping

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    ANCHOR_CASE,
    FreshCaseSpec,
    TimeSpec,
    case_contract,
    case_for_axis,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_full_z_certificate import (
    LEVELS,
    POLARIZATIONS,
    SUCCESSIVE_PAIRS,
    Z_FACTOR,
    compare_full_z_pair,
    full_z_selection_gates,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_time_settling_certificate import (
    audit_raw_case,
    sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_exact_binary_control import (
    EXPECTED_FDTDX_COMMIT,
    _json_default,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_exact_binary_mesh_case import (
    DEFAULT_REFERENCE,
    REPORT_NAME as CASE_REPORT_NAME,
    SCOPE as CASE_SCOPE,
    STATUS_READY as CASE_STATUS_READY,
    VERSION as CASE_VERSION,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_source_pair_validation import (
    validate_source_pair,
)


VERSION = "fdtdx-increment-state-full-domain-z-certificate-v1"
STATUS_READY = "VALIDATED_FDTDX_INCREMENT_STATE_FULL_DOMAIN_Z_CONVERGENCE"
STATUS_BLOCKED = "BLOCKED_FDTDX_INCREMENT_STATE_FULL_DOMAIN_Z_CONVERGENCE"
STATUS_EXCEPTION = "BLOCKED_FDTDX_INCREMENT_STATE_FULL_DOMAIN_Z_EXCEPTION"
CERTIFICATE_NAME = "FDTDX_INCREMENT_STATE_FULL_DOMAIN_Z_CERTIFICATE.json"
TOTAL_PERIODS = 24
WINDOW_PERIODS = 4
COURANT_FACTOR = 0.5
POWER_MATCH_RTOL = 5.0e-13
POWER_MATCH_ATOL_W = 1.0e-30


def expected_full_z_case(level: str) -> FreshCaseSpec:
    if level not in LEVELS:
        raise ValueError(f"full-z level must be one of {LEVELS}")
    spec = case_for_axis(
        "full_domain_z",
        LEVELS.index(level),
        time=TimeSpec(
            total_periods=TOTAL_PERIODS,
            window_periods=WINDOW_PERIODS,
            courant_factor=COURANT_FACTOR,
        ),
        pml_alpha_scale=ANCHOR_CASE.pml_alpha_scale,
        pml_target_reflection=ANCHOR_CASE.pml_target_reflection,
    )
    if spec.mesh.z_factor != Z_FACTOR[level]:
        raise RuntimeError("full-z level label and mesh factor disagree")
    return spec


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _all_true(values: Mapping[str, Any]) -> bool:
    return bool(values) and all(value is True for value in values.values())


def _report_path(root: Path, polarization: str) -> Path:
    return (root / polarization / CASE_REPORT_NAME).resolve()


def _runner_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance", {})
    recorded_path = provenance.get("runner_path")
    if not isinstance(recorded_path, str):
        return {
            "path": None,
            "expected_sha256": provenance.get("runner_sha256"),
            "actual_sha256": None,
            "checks": {
                "path_is_absolute": False,
                "file_exists": False,
                "sha256_matches": False,
            },
            "ready": False,
        }
    supplied = Path(recorded_path).expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_file()
    actual = sha256(resolved) if exists else None
    expected = provenance.get("runner_sha256")
    checks = {
        "path_is_absolute": supplied.is_absolute(),
        "file_exists": exists,
        "sha256_matches": actual == expected,
    }
    return {
        "path": str(resolved),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "checks": checks,
        "ready": all(checks.values()),
    }


def _normalization_checks(
    payload: Mapping[str, Any],
    source_pair: Mapping[str, Any],
    snapshot: Mapping[str, Any] | None,
) -> dict[str, bool]:
    if snapshot is None:
        return {"raw_snapshot_available": False}
    try:
        pair_normalization = source_pair["common_normalization"]
        report_normalization = payload["normalization_policy"]
        reporting = payload["evaluation"]["common_285uW_reporting"]
        late = snapshot["power_late"]
    except (KeyError, TypeError):
        return {"normalization_inputs_available": False}
    scale = float(pair_normalization["common_power_scale"])
    expected = {
        "late_total_Q_W": float(late["total_W"]) * scale,
        "late_Au_Q_W": float(late["by_material"]["au"]["total_W"]) * scale,
        "late_TaIrTe4_Q_W": float(late["by_material"]["tairte4"]["total_W"]) * scale,
    }
    return {
        "raw_snapshot_available": True,
        "raw_fields_and_Q_unscaled": report_normalization.get(
            "raw_fields_and_Q_are_unscaled"
        )
        is True,
        "per_polarization_matching_forbidden": report_normalization.get(
            "per_polarization_power_matching_forbidden"
        )
        is True,
        "common_power_scale_exact": report_normalization.get("common_power_scale")
        == pair_normalization["common_power_scale"],
        "common_field_scale_exact": report_normalization.get(
            "common_field_amplitude_scale"
        )
        == pair_normalization["common_field_amplitude_scale"],
        "source_reference_exact": payload["evaluation"]["flux"].get(
            "source_reference_all_air_unscaled_W"
        )
        == source_pair["comparison"]["mean_unscaled_incident_power_W"],
        "scaled_Q_recomputes_from_raw": all(
            math.isclose(
                float(reporting[name]),
                value,
                rel_tol=POWER_MATCH_RTOL,
                abs_tol=POWER_MATCH_ATOL_W,
            )
            for name, value in expected.items()
        ),
    }


def audit_material_case(
    report_path: Path,
    expected_report_sha256: str,
    polarization: str,
    level: str,
    spec: FreshCaseSpec,
    source_pair: Mapping[str, Any],
    source_pair_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    supplied = report_path.expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_file()
    actual_sha = sha256(resolved) if exists else None
    payload = json.loads(resolved.read_text(encoding="utf-8")) if exists else {}
    raw_audit: dict[str, Any] = {"ready": False, "checks": {}}
    snapshot = None
    raw_error = None
    if exists:
        try:
            raw_audit, snapshot = audit_raw_case(payload, spec)
        except Exception as error:
            raw_error = repr(error)
    runner = _runner_audit(payload) if exists else {"ready": False, "checks": {}}
    evaluation = payload.get("evaluation", {})
    evaluation_gates = evaluation.get("gates", {})
    material = payload.get("material", {})
    exact_binary = material.get("exact_binary_au", {})
    provenance = payload.get("provenance", {})
    source_pair_record = payload.get("source_pair", {})
    expected_vector = [0.0, 1.0, 0.0] if polarization == "Ea" else [1.0, 0.0, 0.0]
    normalization_checks = _normalization_checks(payload, source_pair, snapshot)
    checks = {
        "report_path_is_absolute": supplied.is_absolute(),
        "report_exists": exists,
        "expected_report_sha256_is_hex": len(expected_report_sha256) == 64
        and all(
            character in "0123456789abcdef" for character in expected_report_sha256
        ),
        "report_sha256_matches": actual_sha == expected_report_sha256,
        "version_status_scope_ready": payload.get("version") == CASE_VERSION
        and payload.get("status") == CASE_STATUS_READY
        and payload.get("scope") == CASE_SCOPE
        and payload.get("ready") is True,
        "labels_exact": payload.get("polarization") == polarization
        and payload.get("reference") == DEFAULT_REFERENCE
        and payload.get("mesh_axis") == "full_domain_z"
        and payload.get("mesh_level") == LEVELS.index(level),
        "canonical_case_exact": payload.get("numerical_case_contract")
        == case_contract(spec),
        "mesh_exact": payload.get("mesh") == case_contract(spec)["resolved_mesh"],
        "polarization_vector_exact": payload.get("source_contract", {}).get(
            "fixed_E_polarization_vector"
        )
        == expected_vector,
        "source_pair_binding_exact": source_pair_audit.get("ready") is True
        and source_pair_record.get("path") == source_pair_audit.get("path")
        and source_pair_record.get("expected_sha256")
        == source_pair_audit.get("expected_sha256")
        and source_pair_record.get("actual_sha256")
        == source_pair_audit.get("actual_sha256"),
        "source_pair_contract_checks_all_true": _all_true(
            payload.get("source_pair_contract_checks", {})
        ),
        "evaluation_ready_and_gates_all_true": evaluation.get("ready") is True
        and _all_true(evaluation_gates)
        and evaluation.get("failed_gates") == [],
        "material_readback_ready": material.get("ready") is True
        and _all_true(material.get("checks", {}))
        and material.get("failed_checks") == [],
        "exact_binary_no_gray_law": exact_binary.get("ready") is True
        and exact_binary.get("gray_density_allowed") is False
        and exact_binary.get("rho_power") is None,
        "raw_artifact_revalidated": raw_audit.get("ready") is True,
        "runner_bytes_revalidated": runner.get("ready") is True,
        "repository_was_clean": provenance.get("repository_dirty_porcelain_before")
        == ""
        and provenance.get("repository_dirty_porcelain_after") == "",
        "fdtdx_commit_and_source_clean": provenance.get("fdtdx_source", {}).get(
            "commit"
        )
        == EXPECTED_FDTDX_COMMIT
        and provenance.get("fdtdx_source", {}).get("dirty_porcelain") == "",
        "optimizer_was_forbidden": payload.get("optimizer_start_allowed") is False,
        "normalization_revalidated": _all_true(normalization_checks),
    }
    audit = {
        "path": str(resolved),
        "expected_sha256": expected_report_sha256,
        "actual_sha256": actual_sha,
        "raw": raw_audit,
        "raw_audit_error": raw_error,
        "runner": runner,
        "normalization_checks": normalization_checks,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
    }
    return payload, audit, snapshot if audit["ready"] else None


def build_certificate(
    case_roots: Mapping[str, Path],
    source_pair_paths: Mapping[str, Path],
    source_pair_sha256s: Mapping[str, str],
    report_sha256s: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    expected_levels = set(LEVELS)
    if any(
        set(mapping) != expected_levels
        for mapping in (
            case_roots,
            source_pair_paths,
            source_pair_sha256s,
            report_sha256s,
        )
    ):
        raise ValueError(f"all mappings must contain exactly {LEVELS}")
    if any(set(report_sha256s[level]) != set(POLARIZATIONS) for level in LEVELS):
        raise ValueError("each report SHA mapping must contain exactly Ea and Eb")

    specs = {level: expected_full_z_case(level) for level in LEVELS}
    source_pairs: dict[str, Any] = {}
    source_audits: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {}
    case_audits: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, dict[str, Any] | None] = {}
    for level in LEVELS:
        source_payload, source_audit = validate_source_pair(
            source_pair_paths[level], source_pair_sha256s[level], specs[level]
        )
        source_pairs[level] = source_payload
        source_audits[level] = source_audit
        payloads[level] = {}
        case_audits[level] = {}
        snapshots[level] = {}
        for polarization in POLARIZATIONS:
            payload, audit, snapshot = audit_material_case(
                _report_path(case_roots[level], polarization),
                report_sha256s[level][polarization],
                polarization,
                level,
                specs[level],
                source_payload,
                source_audit,
            )
            payloads[level][polarization] = payload
            case_audits[level][polarization] = audit
            snapshots[level][polarization] = snapshot

    pair_comparisons = {
        f"{coarse}_to_{fine}": compare_full_z_pair(
            coarse, fine, snapshots, payloads, source_pairs
        )
        for coarse, fine in SUCCESSIVE_PAIRS
    }
    case_ready = {
        level: {
            polarization: case_audits[level][polarization]["ready"]
            for polarization in POLARIZATIONS
        }
        for level in LEVELS
    }
    selection = full_z_selection_gates(
        case_ready,
        {
            pair: pair_comparisons[f"{pair[0]}_to_{pair[1]}"]["pass"]
            for pair in SUCCESSIVE_PAIRS
        },
    )
    flat_payloads = [
        payloads[level][polarization]
        for level in LEVELS
        for polarization in POLARIZATIONS
    ]
    source_commits = {
        source_pairs.get(level, {})
        .get("source_case_contracts", {})
        .get("fdtdx_source", {})
        .get("commit")
        for level in LEVELS
    }
    repository_commits = {
        payload.get("provenance", {}).get("repository_commit")
        for payload in flat_payloads
    }
    runner_hashes = {
        payload.get("provenance", {}).get("runner_sha256") for payload in flat_payloads
    }
    base_mesh = dict(specs[LEVELS[0]].mesh.__dict__)
    base_mesh.pop("z_factor")
    global_checks = {
        "all_source_certificates_revalidated": all(
            source_audits[level]["ready"] is True for level in LEVELS
        ),
        "all_six_material_cases_revalidated": all(
            case_audits[level][polarization]["ready"] is True
            for level in LEVELS
            for polarization in POLARIZATIONS
        ),
        "only_full_domain_z_factor_changes": all(
            {
                name: value
                for name, value in spec.mesh.__dict__.items()
                if name != "z_factor"
            }
            == base_mesh
            and spec.mesh.z_factor == Z_FACTOR[level]
            for level, spec in specs.items()
        ),
        "all_cases_share_one_repository_commit": len(repository_commits) == 1
        and None not in repository_commits,
        "all_cases_share_one_runner_hash": len(runner_hashes) == 1
        and None not in runner_hashes,
        "all_sources_use_expected_fdtdx_commit": source_commits
        == {EXPECTED_FDTDX_COMMIT},
        "both_successive_pairs_were_evaluated": set(pair_comparisons)
        == {"z2_to_z4", "z4_to_z8"},
        "optimizer_remains_forbidden": all(
            payload.get("optimizer_start_allowed") is False for payload in flat_payloads
        ),
    }
    ready = all(global_checks.values()) and all(selection.values())
    return {
        "version": VERSION,
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "scope": (
            "fixed L500 optical full-domain-z convergence only; no thermal, "
            "electrical, adjoint, gray material law, or optimizer"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "levels": list(LEVELS),
        "successive_pairs": [list(pair) for pair in SUCCESSIVE_PAIRS],
        "case_roots": {level: str(case_roots[level].resolve()) for level in LEVELS},
        "source_pair_audits": source_audits,
        "case_audits": case_audits,
        "pair_comparisons": pair_comparisons,
        "selection_gates": selection,
        "failed_selection_gates": [
            name for name, passed in selection.items() if not passed
        ],
        "global_checks": global_checks,
        "failed_global_checks": [
            name for name, passed in global_checks.items() if not passed
        ],
        "repository_commits_in_material_cases": sorted(
            value for value in repository_commits if value is not None
        ),
        "runner_hashes_in_material_cases": sorted(
            value for value in runner_hashes if value is not None
        ),
        "optimizer_start_allowed": False,
        "promotion": {
            "full_domain_z_converged": ready,
            "selected_mesh_level": "z8" if ready else None,
            "next_required_action": (
                "continue to independent non-z mesh axes"
                if ready
                else "extend full-domain-z refinement before any optimizer"
            ),
        },
    }


def _parse_level_paths(values: list[str], label: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        level, separator, path = value.partition("=")
        if separator != "=" or level not in LEVELS or level in parsed or not path:
            raise ValueError(f"invalid {label} entry {value!r}")
        supplied = Path(path).expanduser()
        if not supplied.is_absolute():
            raise ValueError(f"{label} paths must be absolute")
        parsed[level] = supplied.resolve()
    if set(parsed) != set(LEVELS):
        raise ValueError(f"{label} entries must contain exactly {LEVELS}")
    return parsed


def _parse_level_hashes(values: list[str], label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        level, separator, digest = value.partition("=")
        valid = len(digest) == 64 and all(
            character in "0123456789abcdef" for character in digest
        )
        if separator != "=" or level not in LEVELS or level in parsed or not valid:
            raise ValueError(f"invalid {label} entry {value!r}")
        parsed[level] = digest
    if set(parsed) != set(LEVELS):
        raise ValueError(f"{label} entries must contain exactly {LEVELS}")
    return parsed


def _parse_report_hashes(values: list[str]) -> dict[str, dict[str, str]]:
    parsed = {level: {} for level in LEVELS}
    for value in values:
        key, separator, digest = value.partition("=")
        level, colon, polarization = key.partition(":")
        valid = len(digest) == 64 and all(
            character in "0123456789abcdef" for character in digest
        )
        if (
            separator != "="
            or colon != ":"
            or level not in LEVELS
            or polarization not in POLARIZATIONS
            or polarization in parsed[level]
            or not valid
        ):
            raise ValueError(f"invalid report SHA entry {value!r}")
        parsed[level][polarization] = digest
    if any(set(parsed[level]) != set(POLARIZATIONS) for level in LEVELS):
        raise ValueError("report SHA entries must contain every level:polarization")
    return parsed


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
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
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--case-root", action="append", default=[])
    parser.add_argument("--source-pair", action="append", default=[])
    parser.add_argument("--source-pair-sha256", action="append", default=[])
    parser.add_argument("--report-sha256", action="append", default=[])
    args = parser.parse_args()
    output = args.output_directory.expanduser().resolve()
    try:
        if not args.output_directory.expanduser().is_absolute():
            raise RuntimeError("output directory must be absolute")
        if not output.is_dir() or any(output.iterdir()):
            raise RuntimeError("output directory must be existing and empty")
        repository = Path(__file__).resolve().parents[3]
        dirty = _git(repository, "status", "--porcelain", "--untracked-files=all")
        if dirty != "":
            raise RuntimeError("repository must be clean before certification")
        payload = build_certificate(
            _parse_level_paths(args.case_root, "case root"),
            _parse_level_paths(args.source_pair, "source pair"),
            _parse_level_hashes(args.source_pair_sha256, "source-pair SHA"),
            _parse_report_hashes(args.report_sha256),
        )
        payload["certificate_provenance"] = {
            "repository_commit": _git(repository, "rev-parse", "HEAD"),
            "repository_dirty_porcelain": dirty,
            "generator_path": str(Path(__file__).resolve()),
            "generator_sha256": sha256(Path(__file__).resolve()),
        }
    except Exception as error:
        payload = {
            "version": VERSION,
            "status": STATUS_EXCEPTION,
            "ready": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "optimizer_start_allowed": False,
        }
    _atomic_json(output / CERTIFICATE_NAME, payload)
    print(
        json.dumps(
            {
                "certificate": str(output / CERTIFICATE_NAME),
                "status": payload["status"],
                "ready": payload["ready"],
            }
        )
    )
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
