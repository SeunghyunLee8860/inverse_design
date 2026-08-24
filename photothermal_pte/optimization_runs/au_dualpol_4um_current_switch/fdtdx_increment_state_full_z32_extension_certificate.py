#!/usr/bin/env python3
"""Certify the increment-state z16-to-z32 exact-binary extension."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback
from typing import Any, Mapping

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    FreshCaseSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_full_z_certificate import (
    POLARIZATIONS,
    compare_full_z_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_time_settling_certificate import (
    sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_exact_binary_control import (
    _json_default,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_exact_binary_mesh_case import (
    REPORT_NAME as CASE_REPORT_NAME,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_full_z_certificate import (
    _all_true,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_full_z_extension_case import (
    expected_extension_case,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_full_z_extension_certificate import (
    STATUS_BLOCKED as PRIOR_STATUS,
    VERSION as PRIOR_VERSION,
    _git,
    audit_case,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_source_pair_validation import (
    validate_source_pair,
)


VERSION = "fdtdx-increment-state-full-z32-extension-certificate-v1"
STATUS_READY = "VALIDATED_FDTDX_INCREMENT_STATE_Z16_TO_Z32_NO_MESH_SELECTION"
STATUS_BLOCKED = "BLOCKED_FDTDX_INCREMENT_STATE_Z16_TO_Z32"
STATUS_EXCEPTION = "BLOCKED_FDTDX_INCREMENT_STATE_Z16_TO_Z32_EXCEPTION"
CERTIFICATE_NAME = "FDTDX_INCREMENT_STATE_FULL_Z32_EXTENSION_CERTIFICATE.json"
LEVELS = ("z16", "z32")
Z_FACTORS = {"z16": 16, "z32": 32}
SUCCESSIVE_PAIRS = (("z16", "z32"),)


def expected_case(level: str) -> FreshCaseSpec:
    if level not in LEVELS:
        raise ValueError(f"extension level must be one of {LEVELS}")
    return expected_extension_case(level)


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
    comparison = payload.get("z8_to_z16_comparison", {})
    promotion = payload.get("promotion", {})
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
        "both_prior_source_artifacts_ready": all(
            source_audits.get(level, {}).get("ready") is True
            for level in ("z8", "z16")
        ),
        "all_prior_material_artifacts_ready": all(
            case_audits.get(level, {}).get(polarization, {}).get("ready") is True
            for level in ("z8", "z16")
            for polarization in POLARIZATIONS
        ),
        "z8_to_z16_was_evaluated_and_failed": comparison.get("error") is None
        and comparison.get("pass") is False,
        "prior_selection_remained_blocked": promotion.get("z8_to_z16_pass")
        is False
        and promotion.get("full_domain_z_converged") is False
        and promotion.get("selected_mesh_level") is None
        and promotion.get("requires_two_successive_passing_tail_pairs") is True
        and payload.get("optimizer_start_allowed") is False,
        "prior_generator_provenance_clean": provenance.get(
            "repository_dirty_porcelain"
        )
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


def _case_rebound_to_prior(
    prior: Mapping[str, Any], audits: Mapping[str, Mapping[str, Any]]
) -> bool:
    for polarization in POLARIZATIONS:
        old = prior.get("case_audits", {}).get("z16", {}).get(polarization, {})
        new = audits.get("z16", {}).get(polarization, {})
        if not (
            old.get("ready") is True
            and new.get("ready") is True
            and old.get("path") == new.get("path")
            and old.get("actual_sha256") == new.get("actual_sha256")
            and old.get("raw", {}).get("actual_sha256")
            == new.get("raw", {}).get("actual_sha256")
        ):
            return False
    return True


def promotion(comparison_pass: bool) -> dict[str, Any]:
    result = "passed" if comparison_pass else "failed"
    return {
        "z8_to_z16_pass": False,
        "z16_to_z32_pass": comparison_pass,
        "full_domain_z_converged": False,
        "selected_mesh_level": None,
        "requires_two_successive_passing_tail_pairs": True,
        "z_only_ladder_terminated": True,
        "z64_run_allowed": False,
        "optimizer_start_allowed": False,
        "next_required_action": (
            f"z16-to-z32 {result}; do not run z64 or select a mesh because "
            "z8-to-z16 failed and z32 is already impractical for optimization. "
            "Use the frozen artifacts to diagnose a balanced spatial-mesh strategy."
        ),
    }


def build_certificate(
    prior_certificate_path: Path,
    prior_certificate_sha256: str,
    source_pair_paths: Mapping[str, Path],
    source_pair_sha256s: Mapping[str, str],
    z32_root: Path,
    z32_report_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    if set(source_pair_paths) != set(LEVELS) or set(source_pair_sha256s) != set(
        LEVELS
    ):
        raise ValueError("source-pair mappings must contain exactly z16 and z32")
    if set(z32_report_sha256s) != set(POLARIZATIONS):
        raise ValueError("z32 report SHA mapping must contain exactly Ea and Eb")

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
    snapshots: dict[str, dict[str, Any]] = {level: {} for level in LEVELS}
    for polarization in POLARIZATIONS:
        prior_case = prior.get("case_audits", {}).get("z16", {}).get(
            polarization, {}
        )
        z16_report = Path(prior_case.get("path", ""))
        z16_sha = prior_case.get("actual_sha256", "")
        (
            payloads["z16"][polarization],
            audits["z16"][polarization],
            snapshots["z16"][polarization],
        ) = audit_case(
            z16_report,
            z16_sha,
            "z16",
            polarization,
            specs["z16"],
            sources["z16"],
            source_audits["z16"],
            None,
        )
        z32_report = (z32_root / polarization / CASE_REPORT_NAME).resolve()
        (
            payloads["z32"][polarization],
            audits["z32"][polarization],
            snapshots["z32"][polarization],
        ) = audit_case(
            z32_report,
            z32_report_sha256s[polarization],
            "z32",
            polarization,
            specs["z32"],
            sources["z32"],
            source_audits["z32"],
            None,
        )

    comparison = compare_full_z_pair(
        "z16",
        "z32",
        snapshots,
        payloads,
        sources,
        z_factors=Z_FACTORS,
        successive_pairs=SUCCESSIVE_PAIRS,
    )
    global_checks = {
        "prior_z16_certificate_revalidated": prior_audit["ready"] is True,
        "both_source_pairs_revalidated": all(
            source_audits[level]["ready"] is True for level in LEVELS
        ),
        "all_four_material_artifacts_revalidated": all(
            audits[level][polarization]["ready"] is True
            for level in LEVELS
            for polarization in POLARIZATIONS
        ),
        "z16_case_bytes_rebound_to_prior_certificate": _case_rebound_to_prior(
            prior, audits
        ),
        "only_z_factor_changes_between_cases": {
            name: value
            for name, value in specs["z16"].mesh.__dict__.items()
            if name != "z_factor"
        }
        == {
            name: value
            for name, value in specs["z32"].mesh.__dict__.items()
            if name != "z_factor"
        },
        "z16_to_z32_comparison_evaluated": comparison.get("error") is None,
        "optimizer_remains_forbidden": all(
            payloads[level][polarization].get("optimizer_start_allowed") is False
            for level in LEVELS
            for polarization in POLARIZATIONS
        ),
    }
    ready = all(global_checks.values()) and comparison.get("pass") is True
    return {
        "version": VERSION,
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "scope": (
            "z16-to-z32 fixed-L500 optical extension only; no thermal, electrical, "
            "adjoint, gray material law, optimizer, z64 run, or production-mesh selection"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prior_certificate": prior_audit,
        "source_pair_audits": source_audits,
        "case_audits": audits,
        "z16_to_z32_comparison": comparison,
        "global_checks": global_checks,
        "failed_global_checks": [
            name for name, passed in global_checks.items() if not passed
        ],
        "optimizer_start_allowed": False,
        "promotion": promotion(comparison.get("pass") is True),
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
    parser.add_argument("--z16-source-pair", type=Path, required=True)
    parser.add_argument("--z16-source-pair-sha256", required=True)
    parser.add_argument("--z32-source-pair", type=Path, required=True)
    parser.add_argument("--z32-source-pair-sha256", required=True)
    parser.add_argument("--z32-root", type=Path, required=True)
    parser.add_argument("--z32-ea-report-sha256", required=True)
    parser.add_argument("--z32-eb-report-sha256", required=True)
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
            {"z16": args.z16_source_pair, "z32": args.z32_source_pair},
            {
                "z16": args.z16_source_pair_sha256,
                "z32": args.z32_source_pair_sha256,
            },
            args.z32_root,
            {
                "Ea": args.z32_ea_report_sha256,
                "Eb": args.z32_eb_report_sha256,
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
