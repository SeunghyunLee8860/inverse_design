#!/usr/bin/env python3
"""Certify one common Ea/Eb normalization on the user-balanced mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    TimeSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_pair import (
    _common_source_contract,
    _raw_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_exact_binary_control import (
    EXPECTED_FDTDX_COMMIT,
    _json_default,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_mesh import (
    mesh_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_source_only import (
    STATUS_READY as CASE_STATUS_READY,
    VERSION as CASE_VERSION,
    balanced_case_contract,
)


VERSION = "fdtdx-user-balanced-source-pair-v1"
STATUS_READY = "VALIDATED_FDTDX_USER_BALANCED_SOURCE_PAIR"
STATUS_BLOCKED = "BLOCKED_FDTDX_USER_BALANCED_SOURCE_PAIR"
CASE_SCOPE = "all-air source-only on the requested balanced FDTDX mesh"
POWER_MISMATCH_RELATIVE_LIMIT = 5.0e-3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _recorded_file_audit(record: dict[str, Any]) -> dict[str, Any]:
    recorded_path = record.get("path")
    supplied = (
        Path(recorded_path).expanduser() if isinstance(recorded_path, str) else None
    )
    resolved = supplied.resolve() if supplied is not None else None
    exists = resolved.is_file() if resolved is not None else False
    actual = sha256(resolved) if exists and resolved is not None else None
    recorded = record.get("actual_sha256")
    checks = {
        "path_is_absolute": supplied is not None and supplied.is_absolute(),
        "file_exists": exists,
        "recorded_sha256_is_hex": _is_sha256(recorded),
        "sha256_matches_record": actual == recorded,
    }
    return {
        "path": str(resolved) if resolved is not None else None,
        "actual_sha256": actual,
        "recorded_sha256": recorded,
        "checks": checks,
        "ready": all(checks.values()),
    }


def _load_case(
    path: Path, expected_sha256: str, polarization: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = path.expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_file()
    actual_sha256 = sha256(resolved) if exists else None
    payload = json.loads(resolved.read_text(encoding="utf-8")) if exists else {}
    evaluation = payload.get("evaluation", {})
    gates = evaluation.get("gates", {})
    checks_payload = payload.get("checks", {})
    provenance = payload.get("provenance", {})
    source = provenance.get("fdtdx_source", {})
    try:
        raw = _raw_audit(payload) if exists else {"ready": False}
    except (KeyError, TypeError, ValueError):
        raw = {"ready": False}
    checks = {
        "path_is_absolute": supplied.is_absolute(),
        "report_exists": exists,
        "expected_sha256_is_hex": _is_sha256(expected_sha256),
        "report_sha256_matches": actual_sha256 == expected_sha256,
        "version_exact": payload.get("version") == CASE_VERSION,
        "status_and_ready": payload.get("status") == CASE_STATUS_READY
        and payload.get("ready") is True,
        "failed_checks_empty": payload.get("failed_checks") == [],
        "scope_exact": payload.get("scope") == CASE_SCOPE,
        "polarization_exact": payload.get("polarization") == polarization,
        "evaluation_ready": evaluation.get("ready") is True,
        "evaluation_gates_all_true": bool(gates)
        and all(value is True for value in gates.values()),
        "all_air_readback_ready": payload.get("all_air_material_readback", {}).get(
            "ready"
        )
        is True,
        "source_checks_all_true": bool(checks_payload)
        and all(value is True for value in checks_payload.values()),
        "increment_state_exact": checks_payload.get("increment_state_selected") is True,
        "per_case_scaling_not_applied": checks_payload.get(
            "per_case_scaling_not_applied"
        )
        is True,
        "repository_clean_before_after": provenance.get(
            "repository_dirty_porcelain_before"
        )
        == provenance.get("repository_dirty_porcelain_after")
        == "",
        "fdtdx_commit_exact": source.get("commit") == EXPECTED_FDTDX_COMMIT,
        "fdtdx_source_clean": source.get("dirty_porcelain") == "",
        "raw_ready": raw.get("ready") is True,
    }
    return payload, {
        "path": str(resolved),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "polarization": polarization,
        "incident_power_W": evaluation.get("flux", {}).get("incident_plane_signed_W"),
        "raw": raw,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
    }


def build_pair(
    ea_path: Path,
    ea_sha256: str,
    eb_path: Path,
    eb_sha256: str,
) -> dict[str, Any]:
    ea, ea_audit = _load_case(ea_path, ea_sha256, "Ea")
    eb, eb_audit = _load_case(eb_path, eb_sha256, "Eb")
    cases = {"Ea": ea_audit, "Eb": eb_audit}
    powers = {
        name: float(audit["incident_power_W"])
        for name, audit in cases.items()
        if audit["incident_power_W"] is not None
    }
    powers_ready = set(powers) == {"Ea", "Eb"} and all(
        math.isfinite(value) and value > 0.0 for value in powers.values()
    )
    mean_power = sum(powers.values()) / 2.0 if powers_ready else math.nan
    mismatch = (
        abs(powers["Ea"] - powers["Eb"]) / mean_power if powers_ready else math.inf
    )
    target_ea = ea.get("reporting_incident_power_W")
    target_eb = eb.get("reporting_incident_power_W")
    common_target_ready = (
        isinstance(target_ea, (int, float))
        and isinstance(target_eb, (int, float))
        and math.isfinite(float(target_ea))
        and float(target_ea) > 0.0
        and target_ea == target_eb
    )
    common_target = float(target_ea) if common_target_ready else math.nan
    common_power_scale = (
        common_target / mean_power if powers_ready and common_target_ready else math.nan
    )
    common_field_scale = (
        math.sqrt(common_power_scale)
        if math.isfinite(common_power_scale) and common_power_scale > 0.0
        else math.nan
    )

    expected_time = TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5)
    expected_case = balanced_case_contract(expected_time)
    reports = (ea, eb)
    numerical_case = ea.get("numerical_case_contract")
    source_vectors_exact = ea.get("source_contract", {}).get(
        "fixed_E_polarization_vector"
    ) == [0.0, 1.0, 0.0] and eb.get("source_contract", {}).get(
        "fixed_E_polarization_vector"
    ) == [1.0, 0.0, 0.0]
    gates = {
        "case_audits_ready": all(audit["ready"] for audit in cases.values()),
        "report_paths_distinct": ea_audit["path"] != eb_audit["path"],
        "raw_paths_distinct": ea_audit["raw"].get("path")
        != eb_audit["raw"].get("path"),
        "numerical_case_exact_canonical_24_4": numerical_case == expected_case
        and eb.get("numerical_case_contract") == expected_case,
        "mesh_matches_requested_contract": all(
            report.get("mesh") == expected_case["mesh"] for report in reports
        ),
        "time_request_matches_contract": all(
            all(
                report.get("time_contract", {}).get(name) == value
                for name, value in expected_case["time"].items()
            )
            for report in reports
        ),
        "mesh_identical": ea.get("mesh") == eb.get("mesh"),
        "time_contract_identical": ea.get("time_contract") == eb.get("time_contract"),
        "pml_identical": ea.get("pml_face_parameters") == eb.get("pml_face_parameters"),
        "placement_identical": ea.get("placement") == eb.get("placement"),
        "all_air_readback_identical": ea.get("all_air_material_readback")
        == eb.get("all_air_material_readback"),
        "common_source_contract_identical": _common_source_contract(ea)
        == _common_source_contract(eb),
        "polarization_vectors_exact": source_vectors_exact,
        "repository_commit_identical": ea.get("provenance", {}).get("repository_commit")
        == eb.get("provenance", {}).get("repository_commit"),
        "fdtdx_source_identical": ea.get("provenance", {}).get("fdtdx_source")
        == eb.get("provenance", {}).get("fdtdx_source"),
        "runtime_lock_identical": ea.get("runtime_lock") == eb.get("runtime_lock"),
        "runner_sha256_identical": ea.get("provenance", {}).get("runner_sha256")
        == eb.get("provenance", {}).get("runner_sha256"),
        "common_target_identical_finite_positive": common_target_ready,
        "source_powers_finite_positive": powers_ready,
        "source_power_relative_mismatch": mismatch <= POWER_MISMATCH_RELATIVE_LIMIT,
    }
    ready = all(gates.values())
    repository = Path(__file__).resolve().parents[3]
    runner = Path(__file__).resolve()
    return {
        "version": VERSION,
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "scope": "dual-polarization source normalization on user-balanced mesh",
        "cases": cases,
        "comparison": {
            "unscaled_incident_power_W": powers,
            "mean_unscaled_incident_power_W": mean_power,
            "relative_power_mismatch": mismatch,
            "relative_power_mismatch_limit": POWER_MISMATCH_RELATIVE_LIMIT,
        },
        "normalization_policy": {
            "per_polarization_power_matching_forbidden": True,
            "common_reference": ("arithmetic mean of unscaled Ea/Eb incident powers"),
        },
        "common_normalization": {
            "reporting_target_incident_power_W": common_target,
            "common_power_scale": common_power_scale,
            "common_field_amplitude_scale": common_field_scale,
            "scaled_incident_power_W": {
                name: value * common_power_scale for name, value in powers.items()
            },
        },
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "source_case_contracts": {
            "numerical_case_contract": numerical_case,
            "mesh": ea.get("mesh"),
            "time_contract": ea.get("time_contract"),
            "pml_face_parameters": ea.get("pml_face_parameters"),
            "placement": ea.get("placement"),
            "common_source_contract": _common_source_contract(ea),
            "source_contracts": {
                "Ea": ea.get("source_contract"),
                "Eb": eb.get("source_contract"),
            },
            "all_air_material_readback": ea.get("all_air_material_readback"),
            "fdtdx_source": ea.get("provenance", {}).get("fdtdx_source"),
            "runtime_lock": ea.get("runtime_lock"),
        },
        "provenance": {
            "repository_commit": _git(repository, "rev-parse", "HEAD"),
            "repository_dirty_porcelain": _git(
                repository, "status", "--porcelain", "--untracked-files=all"
            ),
            "generator_path": str(runner),
            "generator_sha256": sha256(runner),
            "lumerical_used": False,
        },
        "optimizer_start_allowed": False,
    }


def validate_source_pair(
    path: Path,
    expected_sha256: str,
    expected_time: TimeSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(expected_time, TimeSpec):
        raise TypeError("expected_time must be a TimeSpec")
    supplied = path.expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_file()
    actual_sha256 = sha256(resolved) if exists else None
    payload = json.loads(resolved.read_text(encoding="utf-8")) if exists else {}
    contracts = payload.get("source_case_contracts", {})
    pair_gates = payload.get("gates", {})
    artifacts: dict[str, Any] = {}
    for polarization in ("Ea", "Eb"):
        case = payload.get("cases", {}).get(polarization, {})
        artifacts[polarization] = {
            "report": _recorded_file_audit(case),
            "raw": _recorded_file_audit(case.get("raw", {})),
        }
    artifact_checks = {
        f"{polarization}_{kind}_bytes_match": audit["ready"]
        for polarization, records in artifacts.items()
        for kind, audit in records.items()
    }
    source = contracts.get("fdtdx_source", {})
    expected_case = balanced_case_contract(expected_time)
    checks = {
        "certificate_path_is_absolute": supplied.is_absolute(),
        "certificate_exists": exists,
        "expected_sha256_is_hex": _is_sha256(expected_sha256),
        "certificate_sha256_matches": actual_sha256 == expected_sha256,
        "version_exact": payload.get("version") == VERSION,
        "status_and_ready": payload.get("status") == STATUS_READY
        and payload.get("ready") is True,
        "pair_gates_all_true": bool(pair_gates)
        and all(value is True for value in pair_gates.values()),
        "failed_gates_empty": payload.get("failed_gates") == [],
        "numerical_case_exact": contracts.get("numerical_case_contract")
        == expected_case,
        "mesh_exact": contracts.get("mesh") == mesh_audit(),
        "time_request_exact": all(
            contracts.get("time_contract", {}).get(name) == value
            for name, value in expected_case["time"].items()
        ),
        "fdtdx_commit_exact": source.get("commit") == EXPECTED_FDTDX_COMMIT,
        "fdtdx_source_clean": source.get("dirty_porcelain") == "",
        "per_polarization_scaling_forbidden": payload.get(
            "normalization_policy", {}
        ).get("per_polarization_power_matching_forbidden")
        is True,
        "common_scale_finite_positive": math.isfinite(
            float(
                payload.get("common_normalization", {}).get("common_power_scale", 0.0)
            )
        )
        and float(
            payload.get("common_normalization", {}).get("common_power_scale", 0.0)
        )
        > 0.0,
        **artifact_checks,
    }
    return payload, {
        "path": str(resolved),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "artifacts": artifacts,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ea-report", type=Path, required=True)
    parser.add_argument("--ea-report-sha256", required=True)
    parser.add_argument("--eb-report", type=Path, required=True)
    parser.add_argument("--eb-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser()
    if not output.is_absolute() or not output.parent.is_dir() or output.exists():
        parser.error("--output must be a new absolute file under an existing directory")
    payload = build_pair(
        args.ea_report,
        args.ea_report_sha256,
        args.eb_report,
        args.eb_report_sha256,
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
                "ready": payload["ready"],
                "failed_gates": payload["failed_gates"],
                "relative_power_mismatch": payload["comparison"][
                    "relative_power_mismatch"
                ],
            },
            default=_json_default,
        )
    )
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
