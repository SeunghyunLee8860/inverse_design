"""Revalidate an external increment-state source-pair certificate by bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    FreshCaseSpec,
    case_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_exact_binary_control import (
    EXPECTED_FDTDX_COMMIT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_source_pair import (
    STATUS_READY,
    VERSION,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hex_digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _recorded_file_audit(record: dict[str, Any]) -> dict[str, Any]:
    recorded_path = record.get("path")
    path = Path(recorded_path).expanduser() if isinstance(recorded_path, str) else None
    if path is None:
        return {
            "path": None,
            "actual_sha256": None,
            "recorded_sha256": record.get("actual_sha256"),
            "checks": {
                "path_is_absolute": False,
                "file_exists": False,
                "recorded_sha256_is_hex": False,
                "sha256_matches_record": False,
            },
            "ready": False,
        }
    resolved = path.resolve()
    exists = resolved.is_file()
    actual = sha256(resolved) if exists else None
    recorded = record.get("actual_sha256")
    checks = {
        "path_is_absolute": path.is_absolute(),
        "file_exists": exists,
        "recorded_sha256_is_hex": isinstance(recorded, str) and _hex_digest(recorded),
        "sha256_matches_record": actual == recorded,
    }
    return {
        "path": str(resolved),
        "actual_sha256": actual,
        "recorded_sha256": recorded,
        "checks": checks,
        "ready": all(checks.values()),
    }


def validate_source_pair(
    path: Path,
    expected_sha256: str,
    expected_case: FreshCaseSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(expected_case, FreshCaseSpec):
        raise TypeError("expected_case must be a FreshCaseSpec")
    supplied = path.expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_file()
    actual_sha256 = sha256(resolved) if exists else None
    payload = json.loads(resolved.read_text(encoding="utf-8")) if exists else {}
    pair_gates = payload.get("gates", {})
    expected_contract = case_contract(expected_case)
    source_contracts = payload.get("source_case_contracts", {})
    source = source_contracts.get("fdtdx_source", {})

    artifact_audits: dict[str, Any] = {}
    for polarization in ("Ea", "Eb"):
        case = payload.get("cases", {}).get(polarization, {})
        raw = case.get("raw", {})
        artifact_audits[polarization] = {
            "report": _recorded_file_audit(case),
            "raw": _recorded_file_audit(raw),
        }
    artifact_checks = {
        f"{polarization}_{kind}_bytes_match": audit["ready"]
        for polarization, artifacts in artifact_audits.items()
        for kind, audit in artifacts.items()
    }
    checks = {
        "certificate_path_is_absolute": supplied.is_absolute(),
        "certificate_exists": exists,
        "expected_sha256_is_hex": _hex_digest(expected_sha256),
        "certificate_sha256_matches": actual_sha256 == expected_sha256,
        "version_exact": payload.get("version") == VERSION,
        "status_and_ready": payload.get("status") == STATUS_READY
        and payload.get("ready") is True,
        "pair_gates_all_true": bool(pair_gates)
        and all(value is True for value in pair_gates.values()),
        "failed_gates_empty": payload.get("failed_gates") == [],
        "numerical_case_exact": source_contracts.get("numerical_case_contract")
        == expected_contract,
        "mesh_exact": source_contracts.get("mesh")
        == expected_contract["resolved_mesh"],
        "pml_exact": source_contracts.get("pml_face_parameters")
        == expected_contract["resolved_pml_face_parameters"],
        "time_request_exact": all(
            source_contracts.get("time_contract", {}).get(name) == value
            for name, value in expected_contract["time_spec"].items()
        ),
        "fdtdx_commit_exact": source.get("commit") == EXPECTED_FDTDX_COMMIT,
        "fdtdx_source_clean": source.get("dirty_porcelain") == "",
        "per_polarization_scaling_forbidden": payload.get(
            "normalization_policy", {}
        ).get("per_polarization_power_matching_forbidden")
        is True,
        "common_scale_finite_positive": float(
            payload.get("common_normalization", {}).get("common_power_scale", 0.0)
        )
        > 0.0,
        **artifact_checks,
    }
    audit = {
        "path": str(resolved),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "artifacts": artifact_audits,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
    }
    return payload, audit
