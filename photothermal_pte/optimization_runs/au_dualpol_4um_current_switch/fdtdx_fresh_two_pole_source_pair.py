#!/usr/bin/env python3
"""Pair candidate-bound Ea/Eb all-air source reports without per-case scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
from typing import Any, Mapping

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    FreshCaseSpec,
    case_from_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
    validate_source_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_pair import (
    _atomic_json,
    _git,
    _load_json,
    build_pair_certificate,
    sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_material_contract import (
    material_law_from_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_source_only import (
    CASE_STATUS,
    SCOPE,
)


PAIR_STATUS = "VALIDATED_FDTDX_FRESH_TWO_POLE_SOURCE_ONLY_PAIR"
BLOCKED_STATUS = "BLOCKED_FDTDX_FRESH_TWO_POLE_SOURCE_ONLY_PAIR"
EXCEPTION_STATUS = "BLOCKED_FDTDX_FRESH_TWO_POLE_SOURCE_ONLY_PAIR_EXCEPTION"
CERTIFICATE_NAME = "FDTDX_FRESH_TWO_POLE_SOURCE_ONLY_PAIR.json"


def candidate_law_pair_audit(
    ea: dict[str, Any], eb: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Reconstruct the shared canonical law and reject any source-side drift."""

    reports = (ea, eb)
    laws = [report.get("candidate_material_law_contract") for report in reports]
    law_audits = [
        report.get("candidate_material_law_file_audit") for report in reports
    ]
    model_audits = [report.get("candidate_source_model_audit") for report in reports]
    pre_solve = [report.get("pre_solve_checks") for report in reports]
    law_identical = isinstance(laws[0], Mapping) and laws[0] == laws[1]
    file_audits_ready = all(
        isinstance(audit, Mapping)
        and audit.get("ready") is True
        and audit.get("actual_sha256") == audit.get("expected_sha256")
        for audit in law_audits
    )
    file_audits_identical = file_audits_ready and law_audits[0] == law_audits[1]
    law_case_file_hash_matches_both = law_identical and all(
        report["numerical_case_file_audit"].get("actual_sha256")
        == laws[0]["case_binding"]["case_file_sha256"]
        for report in reports
    )
    law_internal_hash_matches_file_audits = law_identical and all(
        audit.get("material_law_contract_sha256")
        == laws[0]["material_law_contract_sha256"]
        for audit in law_audits
        if isinstance(audit, Mapping)
    ) and len(law_audits) == 2
    model_audits_ready = all(
        isinstance(audit, Mapping)
        and audit.get("ready") is True
        and bool(audit.get("checks"))
        and all(value is True for value in audit["checks"].values())
        for audit in model_audits
    )
    pre_solve_ready = all(
        isinstance(checks, Mapping)
        and bool(checks)
        and all(value is True for value in checks.values())
        for checks in pre_solve
    )
    canonical = False
    reconstruction_error = None
    reconstructed = None
    if law_identical:
        try:
            numerical_case = ea["numerical_case_contract"]
            case_spec = case_from_contract(numerical_case)
            case_sha = ea["numerical_case_file_audit"]["actual_sha256"]
            fdtdx_source = Path(ea["provenance"]["fdtdx_source"]["path"])
            reconstructed = material_law_from_contract(
                laws[0], case_spec, numerical_case, case_sha, fdtdx_source
            )
            canonical = reconstructed == laws[0]
        except Exception as error:
            reconstruction_error = repr(error)
    checks = {
        "candidate_law_contracts_identical": law_identical,
        "candidate_law_file_audits_ready": file_audits_ready,
        "candidate_law_file_audits_identical": file_audits_identical,
        "candidate_law_case_file_hash_matches_both": (
            law_case_file_hash_matches_both
        ),
        "candidate_law_internal_hash_matches_file_audits": (
            law_internal_hash_matches_file_audits
        ),
        "candidate_law_canonical_reconstruction": canonical,
        "candidate_source_model_audits_ready": model_audits_ready,
        "candidate_pre_solve_checks_ready": pre_solve_ready,
        "candidate_runner_provenance_identical": (
            ea["provenance"].get("runner_sha256")
            == eb["provenance"].get("runner_sha256")
            and ea["provenance"].get("implementation_sha256")
            == eb["provenance"].get("implementation_sha256")
        ),
        "optimizer_forbidden_in_both_reports": all(
            report.get("optimizer_start_allowed") is False for report in reports
        ),
    }
    return reconstructed, {
        "ready": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "reconstruction_error": reconstruction_error,
        "material_law_contract_sha256": (
            reconstructed["material_law_contract_sha256"]
            if reconstructed is not None
            else None
        ),
        "material_law_file_audit": law_audits[0] if file_audits_identical else None,
    }


def validate_candidate_source_pair(
    path: Path,
    expected_sha256: str,
    expected_case: FreshCaseSpec,
    expected_material_law: dict[str, Any],
    expected_material_law_file_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate a candidate pair and its exact law before a material solve."""

    payload, audit = validate_source_pair(
        path,
        expected_sha256,
        expected_case,
        expected_pair_status=PAIR_STATUS,
    )
    contracts = payload.get("source_case_contracts", {})
    recorded_law = contracts.get("candidate_material_law_contract")
    recorded_file_audit = contracts.get("candidate_material_law_file_audit")
    candidate_audit = payload.get("candidate_material_law_audit")
    extra_checks = {
        "candidate_pair_scope_exact": payload.get("scope")
        == (
            "dual-polarization all-air source normalization pair bound to one "
            "candidate two-pole material law"
        ),
        "candidate_pair_optimizer_forbidden": (
            payload.get("optimizer_start_allowed") is False
        ),
        "candidate_material_law_exact": recorded_law == expected_material_law,
        "candidate_material_law_file_sha256_exact": (
            isinstance(recorded_file_audit, Mapping)
            and recorded_file_audit.get("actual_sha256")
            == expected_material_law_file_audit.get("actual_sha256")
        ),
        "candidate_material_law_internal_sha256_exact": (
            isinstance(recorded_file_audit, Mapping)
            and recorded_file_audit.get("material_law_contract_sha256")
            == expected_material_law["material_law_contract_sha256"]
        ),
        "candidate_material_law_audit_ready": (
            isinstance(candidate_audit, Mapping)
            and candidate_audit.get("ready") is True
            and candidate_audit.get("failed_checks") == []
            and bool(candidate_audit.get("checks"))
            and all(value is True for value in candidate_audit["checks"].values())
        ),
    }
    audit["checks"].update(extra_checks)
    audit["failed_checks"] = [
        name for name, passed in audit["checks"].items() if not passed
    ]
    audit["ready"] = all(audit["checks"].values())
    return payload, audit


def build_candidate_pair_certificate(
    ea_report: Path, eb_report: Path
) -> dict[str, Any]:
    result = build_pair_certificate(
        ea_report,
        eb_report,
        expected_case_status=CASE_STATUS,
        expected_scope=SCOPE,
        pair_status=PAIR_STATUS,
        blocked_status=BLOCKED_STATUS,
    )
    ea = _load_json(ea_report)
    eb = _load_json(eb_report)
    law, law_audit = candidate_law_pair_audit(ea, eb)
    result["candidate_material_law_audit"] = law_audit
    result["gates"]["candidate_material_law_audit_ready"] = law_audit["ready"]
    if law is not None:
        result["source_case_contracts"]["candidate_material_law_contract"] = law
        result["source_case_contracts"]["candidate_material_law_file_audit"] = (
            law_audit["material_law_file_audit"]
        )
        result["source_case_contracts"][
            "candidate_source_implementation_sha256"
        ] = ea["provenance"]["implementation_sha256"]
    result["ready"] = all(result["gates"].values())
    result["status"] = PAIR_STATUS if result["ready"] else BLOCKED_STATUS
    result["failed_gates"] = [
        name for name, passed in result["gates"].items() if not passed
    ]
    result["scope"] = (
        "dual-polarization all-air source normalization pair bound to one "
        "candidate two-pole material law"
    )
    result["optimizer_start_allowed"] = False
    return result


def write_candidate_pair_certificate(
    ea_report: Path, eb_report: Path, output_directory: Path
) -> dict[str, Any]:
    output = output_directory.expanduser().resolve()
    if not output.is_absolute() or not output.is_dir():
        raise RuntimeError("output directory must be an existing absolute directory")
    if any(output.iterdir()):
        raise RuntimeError("output directory must be empty before pair certification")
    result = build_candidate_pair_certificate(ea_report, eb_report)
    repository = Path(__file__).resolve().parents[3]
    dirty = _git(repository, "status", "--porcelain", "--untracked-files=all")
    result["provenance"] = {
        "certificate_repository_commit": _git(repository, "rev-parse", "HEAD"),
        "certificate_repository_dirty_porcelain": dirty,
        "certificate_generator_path": str(Path(__file__).resolve()),
        "certificate_generator_sha256": sha256(Path(__file__).resolve()),
    }
    result["gates"]["certificate_repository_clean"] = dirty == ""
    result["ready"] = all(result["gates"].values())
    result["status"] = PAIR_STATUS if result["ready"] else BLOCKED_STATUS
    result["failed_gates"] = [
        name for name, passed in result["gates"].items() if not passed
    ]
    _atomic_json(output / CERTIFICATE_NAME, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ea", type=Path, required=True)
    parser.add_argument("--eb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = write_candidate_pair_certificate(
            args.ea, args.eb, args.output_dir
        )
    except Exception as error:
        failure = {
            "status": EXCEPTION_STATUS,
            "ready": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "optimizer_start_allowed": False,
        }
        output = args.output_dir.expanduser().resolve()
        if output.is_dir() and not any(output.iterdir()):
            _atomic_json(output / CERTIFICATE_NAME, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "ready": result["ready"],
                "failed_gates": result["failed_gates"],
                "comparison": result["comparison"],
                "common_normalization": result["common_normalization"],
                "material_law_contract_sha256": result[
                    "candidate_material_law_audit"
                ]["material_law_contract_sha256"],
                "report": str(args.output_dir.resolve() / CERTIFICATE_NAME),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
