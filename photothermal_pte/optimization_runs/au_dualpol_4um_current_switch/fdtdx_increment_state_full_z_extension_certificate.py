#!/usr/bin/env python3
"""Certify the increment-state z8-to-z16 exact-binary extension."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any, Mapping

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    FreshCaseSpec,
    case_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_full_z_certificate import (
    POLARIZATIONS,
    compare_full_z_pair,
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
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_full_z_certificate import (
    STATUS_BLOCKED as PRIOR_STATUS,
    VERSION as PRIOR_VERSION,
    _all_true,
    _normalization_checks,
    expected_full_z_case,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_full_z_extension_case import (
    expected_extension_case,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_source_pair_validation import (
    validate_source_pair,
)


VERSION = "fdtdx-increment-state-full-z16-extension-certificate-v1"
STATUS_READY = "VALIDATED_FDTDX_INCREMENT_STATE_Z16_EXTENSION_PENDING_Z32"
STATUS_BLOCKED = "BLOCKED_FDTDX_INCREMENT_STATE_Z16_EXTENSION"
STATUS_EXCEPTION = "BLOCKED_FDTDX_INCREMENT_STATE_Z16_EXTENSION_EXCEPTION"
CERTIFICATE_NAME = "FDTDX_INCREMENT_STATE_FULL_Z16_EXTENSION_CERTIFICATE.json"
LEVELS = ("z8", "z16")
Z_FACTORS = {"z8": 8, "z16": 16}
SUCCESSIVE_PAIRS = (("z8", "z16"),)
CASE_VERSION = {
    "z8": "fdtdx-increment-state-exact-binary-mesh-case-v1",
    "z16": "fdtdx-increment-state-exact-binary-mesh-case-v2",
    "z32": "fdtdx-increment-state-exact-binary-mesh-case-v2",
}


def expected_case(level: str) -> FreshCaseSpec:
    if level == "z8":
        return expected_full_z_case("z8")
    if level == "z16":
        return expected_extension_case("z16")
    raise ValueError(f"extension level must be one of {LEVELS}")


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def audit_prior_certificate(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = path.expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_file()
    actual = sha256(resolved) if exists else None
    payload = json.loads(resolved.read_text(encoding="utf-8")) if exists else {}
    case_audits = payload.get("case_audits", {})
    source_audits = payload.get("source_pair_audits", {})
    pair_comparisons = payload.get("pair_comparisons", {})
    provenance = payload.get("certificate_provenance", {})
    checks = {
        "path_is_absolute": supplied.is_absolute(),
        "file_exists": exists,
        "sha256_matches": actual == expected_sha256,
        "prior_version_status_exact": payload.get("version") == PRIOR_VERSION
        and payload.get("status") == PRIOR_STATUS
        and payload.get("ready") is False,
        "prior_global_checks_all_true": _all_true(payload.get("global_checks", {}))
        and payload.get("failed_global_checks") == [],
        "all_prior_source_artifacts_ready": all(
            source_audits.get(level, {}).get("ready") is True
            for level in ("z2", "z4", "z8")
        ),
        "all_prior_material_artifacts_ready": all(
            case_audits.get(level, {}).get(polarization, {}).get("ready") is True
            for level in ("z2", "z4", "z8")
            for polarization in POLARIZATIONS
        ),
        "both_prior_pairs_retained_and_failed": set(pair_comparisons)
        == {"z2_to_z4", "z4_to_z8"}
        and all(item.get("pass") is False for item in pair_comparisons.values()),
        "prior_selection_remained_blocked": payload.get("promotion", {}).get(
            "full_domain_z_converged"
        )
        is False
        and payload.get("optimizer_start_allowed") is False,
        "prior_generator_provenance_clean": provenance.get("repository_dirty_porcelain")
        == ""
        and isinstance(provenance.get("repository_commit"), str),
    }
    audit = {
        "path": str(resolved),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
    }
    return payload, audit


def _case_labels(payload: Mapping[str, Any], level: str, polarization: str) -> bool:
    common = (
        payload.get("polarization") == polarization
        and payload.get("reference") == DEFAULT_REFERENCE
    )
    if level == "z8":
        return (
            common
            and payload.get("mesh_axis") == "full_domain_z"
            and payload.get("mesh_level") == 2
            and payload.get("full_z_extension") is None
        )
    if level not in ("z16", "z32"):
        raise ValueError("extension level must be z8, z16, or z32")
    return (
        common
        and payload.get("mesh_axis") == "anchor"
        and payload.get("mesh_level") == 0
        and payload.get("full_z_extension") == level
    )


def audit_case(
    report_path: Path,
    expected_report_sha256: str,
    level: str,
    polarization: str,
    spec: FreshCaseSpec,
    source_pair: Mapping[str, Any],
    source_pair_audit: Mapping[str, Any],
    prior_case_audit: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    supplied = report_path.expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_file()
    actual = sha256(resolved) if exists else None
    payload = json.loads(resolved.read_text(encoding="utf-8")) if exists else {}
    raw_audit: dict[str, Any] = {"ready": False, "checks": {}}
    snapshot = None
    raw_error = None
    if exists:
        try:
            raw_audit, snapshot = audit_raw_case(payload, spec)
        except Exception as error:
            raw_error = repr(error)
    material = payload.get("material", {})
    exact_binary = material.get("exact_binary_au", {})
    evaluation = payload.get("evaluation", {})
    provenance = payload.get("provenance", {})
    source_record = payload.get("source_pair", {})
    expected_vector = [0.0, 1.0, 0.0] if polarization == "Ea" else [1.0, 0.0, 0.0]
    normalization = _normalization_checks(payload, source_pair, snapshot)
    prior_binding = True
    if level == "z8":
        prior_binding = (
            prior_case_audit is not None
            and prior_case_audit.get("ready") is True
            and prior_case_audit.get("path") == str(resolved)
            and prior_case_audit.get("actual_sha256") == actual
            and prior_case_audit.get("raw", {}).get("actual_sha256")
            == raw_audit.get("actual_sha256")
        )
    runner_current = True
    if level in ("z16", "z32"):
        runner_path = Path(provenance.get("runner_path", "")).expanduser()
        runner_current = (
            runner_path.is_absolute()
            and runner_path.resolve().is_file()
            and sha256(runner_path.resolve()) == provenance.get("runner_sha256")
        )
    checks = {
        "report_path_is_absolute": supplied.is_absolute(),
        "report_exists_and_sha256_matches": exists and actual == expected_report_sha256,
        "version_status_scope_ready": payload.get("version") == CASE_VERSION[level]
        and payload.get("status") == CASE_STATUS_READY
        and payload.get("scope") == CASE_SCOPE
        and payload.get("ready") is True,
        "case_labels_exact": _case_labels(payload, level, polarization),
        "canonical_case_exact": payload.get("numerical_case_contract")
        == case_contract(spec),
        "polarization_vector_exact": payload.get("source_contract", {}).get(
            "fixed_E_polarization_vector"
        )
        == expected_vector,
        "source_pair_binding_exact": source_pair_audit.get("ready") is True
        and source_record.get("path") == source_pair_audit.get("path")
        and source_record.get("actual_sha256") == source_pair_audit.get("actual_sha256")
        and source_record.get("expected_sha256")
        == source_pair_audit.get("expected_sha256"),
        "source_pair_contract_checks_all_true": _all_true(
            payload.get("source_pair_contract_checks", {})
        ),
        "evaluation_ready_and_gates_all_true": evaluation.get("ready") is True
        and _all_true(evaluation.get("gates", {}))
        and evaluation.get("failed_gates") == [],
        "material_readback_ready": material.get("ready") is True
        and _all_true(material.get("checks", {}))
        and material.get("failed_checks") == [],
        "exact_binary_no_gray_law": exact_binary.get("ready") is True
        and exact_binary.get("gray_density_allowed") is False
        and exact_binary.get("rho_power") is None,
        "raw_artifact_revalidated": raw_audit.get("ready") is True,
        "normalization_revalidated": _all_true(normalization),
        "prior_certificate_binds_z8_or_not_required": prior_binding,
        "current_runner_binds_z16_or_not_required": runner_current,
        "repository_was_clean": provenance.get("repository_dirty_porcelain_before")
        == ""
        and provenance.get("repository_dirty_porcelain_after") == "",
        "fdtdx_commit_and_source_clean": provenance.get("fdtdx_source", {}).get(
            "commit"
        )
        == EXPECTED_FDTDX_COMMIT
        and provenance.get("fdtdx_source", {}).get("dirty_porcelain") == "",
        "optimizer_was_forbidden": payload.get("optimizer_start_allowed") is False,
    }
    audit = {
        "path": str(resolved),
        "expected_sha256": expected_report_sha256,
        "actual_sha256": actual,
        "raw": raw_audit,
        "raw_audit_error": raw_error,
        "normalization_checks": normalization,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
    }
    return payload, audit, snapshot if audit["ready"] else None


def build_certificate(
    prior_certificate_path: Path,
    prior_certificate_sha256: str,
    source_pair_paths: Mapping[str, Path],
    source_pair_sha256s: Mapping[str, str],
    z16_root: Path,
    z16_report_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    if set(source_pair_paths) != set(LEVELS) or set(source_pair_sha256s) != set(LEVELS):
        raise ValueError("source-pair mappings must contain exactly z8 and z16")
    if set(z16_report_sha256s) != set(POLARIZATIONS):
        raise ValueError("z16 report SHA mapping must contain exactly Ea and Eb")
    prior, prior_audit = audit_prior_certificate(
        prior_certificate_path, prior_certificate_sha256
    )
    specs = {level: expected_case(level) for level in LEVELS}
    sources: dict[str, Any] = {}
    source_audits: dict[str, Any] = {}
    for level in LEVELS:
        sources[level], source_audits[level] = validate_source_pair(
            source_pair_paths[level], source_pair_sha256s[level], specs[level]
        )

    payloads: dict[str, dict[str, Any]] = {level: {} for level in LEVELS}
    audits: dict[str, dict[str, Any]] = {level: {} for level in LEVELS}
    snapshots: dict[str, dict[str, Any] | None] = {level: {} for level in LEVELS}
    for polarization in POLARIZATIONS:
        prior_case = prior.get("case_audits", {}).get("z8", {}).get(polarization)
        z8_report = Path(prior_case.get("path", "")) if prior_case else Path("")
        z8_sha = prior_case.get("actual_sha256", "") if prior_case else ""
        (
            payloads["z8"][polarization],
            audits["z8"][polarization],
            snapshots["z8"][polarization],
        ) = audit_case(
            z8_report,
            z8_sha,
            "z8",
            polarization,
            specs["z8"],
            sources["z8"],
            source_audits["z8"],
            prior_case,
        )
        z16_report = (z16_root / polarization / CASE_REPORT_NAME).resolve()
        (
            payloads["z16"][polarization],
            audits["z16"][polarization],
            snapshots["z16"][polarization],
        ) = audit_case(
            z16_report,
            z16_report_sha256s[polarization],
            "z16",
            polarization,
            specs["z16"],
            sources["z16"],
            source_audits["z16"],
            None,
        )

    comparison = compare_full_z_pair(
        "z8",
        "z16",
        snapshots,
        payloads,
        sources,
        z_factors=Z_FACTORS,
        successive_pairs=SUCCESSIVE_PAIRS,
    )
    global_checks = {
        "prior_certificate_revalidated": prior_audit["ready"] is True,
        "both_source_pairs_revalidated": all(
            source_audits[level]["ready"] is True for level in LEVELS
        ),
        "all_four_material_artifacts_revalidated": all(
            audits[level][polarization]["ready"] is True
            for level in LEVELS
            for polarization in POLARIZATIONS
        ),
        "only_z_factor_changes_between_cases": {
            name: value
            for name, value in specs["z8"].mesh.__dict__.items()
            if name != "z_factor"
        }
        == {
            name: value
            for name, value in specs["z16"].mesh.__dict__.items()
            if name != "z_factor"
        },
        "z8_to_z16_comparison_evaluated": comparison.get("error") is None,
        "optimizer_remains_forbidden": all(
            payloads[level][polarization].get("optimizer_start_allowed") is False
            for level in LEVELS
            for polarization in POLARIZATIONS
        ),
    }
    ready = all(global_checks.values()) and comparison["pass"] is True
    return {
        "version": VERSION,
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "scope": (
            "z8-to-z16 fixed-L500 optical extension only; no thermal, electrical, "
            "adjoint, gray material law, optimizer, or production-mesh selection"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prior_certificate": prior_audit,
        "source_pair_audits": source_audits,
        "case_audits": audits,
        "z8_to_z16_comparison": comparison,
        "global_checks": global_checks,
        "failed_global_checks": [
            name for name, passed in global_checks.items() if not passed
        ],
        "optimizer_start_allowed": False,
        "promotion": {
            "z8_to_z16_pass": comparison["pass"],
            "full_domain_z_converged": False,
            "selected_mesh_level": None,
            "requires_two_successive_passing_tail_pairs": True,
            "next_required_action": (
                "run z32 confirmation; z16 is not selected from one passing pair"
                if ready
                else "diagnose failed z8-to-z16 metrics before deciding on z32"
            ),
        },
    }


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
    parser.add_argument("--prior-certificate", type=Path, required=True)
    parser.add_argument("--prior-certificate-sha256", required=True)
    parser.add_argument("--z8-source-pair", type=Path, required=True)
    parser.add_argument("--z8-source-pair-sha256", required=True)
    parser.add_argument("--z16-source-pair", type=Path, required=True)
    parser.add_argument("--z16-source-pair-sha256", required=True)
    parser.add_argument("--z16-root", type=Path, required=True)
    parser.add_argument("--z16-ea-report-sha256", required=True)
    parser.add_argument("--z16-eb-report-sha256", required=True)
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
            args.prior_certificate,
            args.prior_certificate_sha256,
            {"z8": args.z8_source_pair, "z16": args.z16_source_pair},
            {
                "z8": args.z8_source_pair_sha256,
                "z16": args.z16_source_pair_sha256,
            },
            args.z16_root,
            {
                "Ea": args.z16_ea_report_sha256,
                "Eb": args.z16_eb_report_sha256,
            },
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
