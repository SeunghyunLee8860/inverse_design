#!/usr/bin/env python3
"""Byte-bound downstream PTE comparison for the user-balanced z2/z4 tail."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_frozen_q_thermal_z_case import (
    _git,
    sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_pte_tail_case import (
    POLARIZATIONS,
    RAW_NAME,
    REPORT_NAME,
    STATUS_READY as CASE_STATUS_READY,
    VERSION as CASE_VERSION,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_tail_certificate import (
    LEVELS,
    TAIL_VERSION,
)


VERSION = "fdtdx-user-balanced-pte-tail-certificate-v1"
STATUS_PASS = "VALIDATED_DIAGNOSTIC_FDTDX_USER_BALANCED_PTE_TAIL_PASS"
STATUS_BLOCKED = "VALIDATED_BLOCKED_FDTDX_USER_BALANCED_PTE_TAIL"
STATUS_INVALID = "INVALID_FDTDX_USER_BALANCED_PTE_TAIL_CERTIFICATE"
CERTIFICATE_NAME = "FDTDX_USER_BALANCED_PTE_TAIL_CERTIFICATE.json"

# Numerical diagnostic thresholds, not experimental-accuracy claims.
LIMITS = {
    "mapped_absorbed_power_relative_change": 0.02,
    "mapped_source_xy_NRMSE": 0.05,
    "ta_max_temperature_relative_change": 0.02,
    "ta_mean_temperature_relative_change": 0.02,
    "ta_temperature_NRMSE": 0.02,
    "ta_gradient_l2_relative_change": 0.05,
    "ta_gradient_vector_NRMSE": 0.05,
    "pte_current_relative_change": 0.05,
    "pte_current_density_NRMSE": 0.05,
}


def _all_true(values: Mapping[str, Any]) -> bool:
    return bool(values) and all(value is True for value in values.values())


def _relative_change(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)


def _nrmse(coarse: np.ndarray, fine: np.ndarray) -> float:
    left = np.asarray(coarse)
    right = np.asarray(fine)
    if left.shape != right.shape or left.size == 0:
        return float("inf")
    return float(
        np.linalg.norm(left - right)
        / max(np.linalg.norm(left), np.linalg.norm(right), np.finfo(float).tiny)
    )


def _same_nonzero_sign(left: float, right: float) -> bool:
    return bool(left != 0.0 and right != 0.0 and np.signbit(left) == np.signbit(right))


def compare_pair(
    coarse_report: Mapping[str, Any],
    fine_report: Mapping[str, Any],
    coarse_fields: Mapping[str, np.ndarray],
    fine_fields: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Compare z2 and z4 after their Q fields use one identical PTE mesh."""

    coarse_thermal = coarse_report["thermal_solution"]
    fine_thermal = fine_report["thermal_solution"]
    coarse_current = float(coarse_report["pte_solution"]["signed_current_A"])
    fine_current = float(fine_report["pte_solution"]["signed_current_A"])
    coarse_gradient_l2 = float(
        coarse_thermal["ta_gradient_combined_l2_K_m"]
    )
    fine_gradient_l2 = float(fine_thermal["ta_gradient_combined_l2_K_m"])
    coarse_gradient = np.stack(
        (
            coarse_fields["ta_gradient_x_K_m"],
            coarse_fields["ta_gradient_y_K_m"],
        )
    )
    fine_gradient = np.stack(
        (
            fine_fields["ta_gradient_x_K_m"],
            fine_fields["ta_gradient_y_K_m"],
        )
    )
    metrics = {
        "mapped_absorbed_power_relative_change": _relative_change(
            float(
                coarse_report["normalization"]["mapped_scaled_absorbed_power_W"]
            ),
            float(
                fine_report["normalization"]["mapped_scaled_absorbed_power_W"]
            ),
        ),
        "mapped_source_xy_NRMSE": _nrmse(
            coarse_fields["source_power_xy_W"],
            fine_fields["source_power_xy_W"],
        ),
        "ta_max_temperature_relative_change": _relative_change(
            float(coarse_thermal["ta_max_temperature_rise_K"]),
            float(fine_thermal["ta_max_temperature_rise_K"]),
        ),
        "ta_mean_temperature_relative_change": _relative_change(
            float(coarse_thermal["ta_mean_temperature_rise_K"]),
            float(fine_thermal["ta_mean_temperature_rise_K"]),
        ),
        "ta_temperature_NRMSE": _nrmse(
            coarse_fields["ta_temperature_rise_K"],
            fine_fields["ta_temperature_rise_K"],
        ),
        "ta_gradient_l2_relative_change": _relative_change(
            coarse_gradient_l2, fine_gradient_l2
        ),
        "ta_gradient_vector_NRMSE": _nrmse(coarse_gradient, fine_gradient),
        "pte_current_relative_change": _relative_change(
            coarse_current, fine_current
        ),
        "pte_current_density_NRMSE": _nrmse(
            coarse_fields["pte_current_density_A_m2"],
            fine_fields["pte_current_density_A_m2"],
        ),
    }
    threshold_checks = {
        name: value <= LIMITS[name] for name, value in metrics.items()
    }
    invariant_checks = {
        "temperature_coordinates_identical": np.array_equal(
            coarse_fields["ta_x_centers_m"], fine_fields["ta_x_centers_m"]
        )
        and np.array_equal(
            coarse_fields["ta_y_centers_m"], fine_fields["ta_y_centers_m"]
        ),
        "electrical_weighting_identical_to_roundoff": _nrmse(
            coarse_fields["ta_electrical_weighting_V"],
            fine_fields["ta_electrical_weighting_V"],
        )
        <= 2.0e-12,
        "pte_current_nonzero_sign_stable": _same_nonzero_sign(
            coarse_current, fine_current
        ),
    }
    passed = all(threshold_checks.values()) and all(invariant_checks.values())
    return {
        "coarse_level": "z2",
        "fine_level": "z4",
        "limits": LIMITS,
        "metrics": metrics,
        "threshold_checks": threshold_checks,
        "failed_threshold_checks": [
            name for name, value in threshold_checks.items() if not value
        ],
        "invariant_checks": invariant_checks,
        "failed_invariant_checks": [
            name for name, value in invariant_checks.items() if not value
        ],
        "signed_current_A": {"z2": coarse_current, "z4": fine_current},
        "pass": passed,
    }


def audit_case(
    report_path: Path,
    expected_report_sha256: str,
    level: str,
    polarization: str,
    expected_optical_certificate_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    supplied = report_path.expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_file()
    actual_report_sha = sha256(resolved) if exists else None
    report = (
        json.loads(resolved.read_text(encoding="utf-8")) if exists else {}
    )
    raw_supplied = Path(report.get("raw", {}).get("path", "")).expanduser()
    raw_path = raw_supplied.resolve()
    raw_exists = raw_path.is_file()
    actual_raw_sha = sha256(raw_path) if raw_exists else None
    fields: dict[str, np.ndarray] = {}
    raw_load_error = None
    if raw_exists:
        try:
            with np.load(raw_path, allow_pickle=False) as archive:
                fields = {name: np.asarray(archive[name]) for name in archive.files}
        except Exception as error:  # pragma: no cover - artifact failure path
            raw_load_error = repr(error)
    required = {
        "ta_temperature_rise_K",
        "ta_gradient_x_K_m",
        "ta_gradient_y_K_m",
        "ta_x_centers_m",
        "ta_y_centers_m",
        "source_power_xy_W",
        "pte_current_density_A_m2",
        "ta_electrical_weighting_V",
    }
    declared = report.get("raw", {}).get("arrays", {})
    provenance = report.get("provenance", {})
    checks = {
        "report_path_is_absolute": supplied.is_absolute(),
        "report_exists": exists,
        "report_sha256_matches": actual_report_sha == expected_report_sha256,
        "report_version_status_ready": report.get("version") == CASE_VERSION
        and report.get("status") == CASE_STATUS_READY
        and report.get("ready") is True,
        "labels_exact": report.get("optical_z_level") == level
        and report.get("polarization") == polarization,
        "optical_certificate_rebound": report.get("optical_input_audit", {}).get(
            "actual_certificate_sha256"
        )
        == expected_optical_certificate_sha256,
        "case_provenance_checks_all_true": _all_true(
            report.get("provenance_checks", {})
        ),
        "repository_was_clean": provenance.get(
            "repository_dirty_porcelain_before"
        )
        == provenance.get("repository_dirty_porcelain_after")
        == "",
        "runner_sha256_is_hex": isinstance(provenance.get("runner_sha256"), str)
        and len(provenance.get("runner_sha256")) == 64,
        "raw_path_is_absolute": raw_supplied.is_absolute(),
        "raw_exists": raw_exists,
        "raw_sha256_matches_report": actual_raw_sha
        == report.get("raw", {}).get("sha256"),
        "raw_load_succeeded": raw_load_error is None,
        "required_arrays_exact": set(fields) == required,
        "raw_shapes_match_report": all(
            list(value.shape) == declared.get(name)
            for name, value in fields.items()
        ),
        "raw_values_all_finite": bool(fields)
        and all(np.all(np.isfinite(value)) for value in fields.values()),
        "production_and_optimizer_remained_blocked": report.get(
            "production_mesh_selected"
        )
        is False
        and report.get("optimizer_start_allowed") is False,
        "actual_electrodes_not_claimed": report.get(
            "actual_electrodes_validated"
        )
        is False,
    }
    audit = {
        "path": str(resolved),
        "expected_sha256": expected_report_sha256,
        "actual_sha256": actual_report_sha,
        "raw_path": str(raw_path),
        "actual_raw_sha256": actual_raw_sha,
        "raw_load_error": raw_load_error,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
    }
    return report, fields, audit


def build_certificate(
    optical_certificate_path: Path,
    expected_optical_certificate_sha256: str,
    report_paths: Mapping[str, Mapping[str, Path]],
    report_hashes: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    optical_supplied = optical_certificate_path.expanduser()
    optical_resolved = optical_supplied.resolve()
    optical_exists = optical_resolved.is_file()
    optical_actual_sha = sha256(optical_resolved) if optical_exists else None
    optical = (
        json.loads(optical_resolved.read_text(encoding="utf-8"))
        if optical_exists
        else {}
    )
    optical_checks = {
        "path_is_absolute": optical_supplied.is_absolute(),
        "file_exists": optical_exists,
        "sha256_matches": optical_actual_sha
        == expected_optical_certificate_sha256,
        "tail_version_exact": optical.get("version") == TAIL_VERSION,
        "tail_artifact_certificate_valid": optical.get("certificate_valid")
        is True,
        "strict_optical_convergence_failed": optical.get("convergence_pass")
        is False
        and optical.get("mesh_selected") is None,
        "optimizer_remained_forbidden": optical.get("optimizer_start_allowed")
        is False,
    }
    reports = {level: {} for level in LEVELS}
    fields = {level: {} for level in LEVELS}
    audits = {level: {} for level in LEVELS}
    for level in LEVELS:
        for polarization in POLARIZATIONS:
            report, snapshot, audit = audit_case(
                report_paths[level][polarization],
                report_hashes[level][polarization],
                level,
                polarization,
                expected_optical_certificate_sha256,
            )
            reports[level][polarization] = report
            fields[level][polarization] = snapshot
            audits[level][polarization] = audit

    artifact_checks = {
        "repository_clean_while_certifying": _git(
            repository, "status", "--porcelain", "--untracked-files=all"
        )
        == "",
        "optical_tail_certificate_revalidates": all(optical_checks.values()),
        "all_four_pte_cases_revalidate": all(
            audits[level][polarization]["ready"]
            for level in LEVELS
            for polarization in POLARIZATIONS
        ),
        "all_cases_share_one_runner_commit": len(
            {
                reports[level][polarization]
                .get("provenance", {})
                .get("repository_commit")
                for level in LEVELS
                for polarization in POLARIZATIONS
            }
        )
        == 1,
        "all_cases_share_identical_thermal_mesh": len(
            {
                json.dumps(
                    reports[level][polarization].get("thermal_mesh", {}),
                    sort_keys=True,
                )
                for level in LEVELS
                for polarization in POLARIZATIONS
            }
        )
        == 1,
    }
    comparisons: dict[str, Any] = {}
    if all(artifact_checks.values()):
        comparisons = {
            polarization: compare_pair(
                reports["z2"][polarization],
                reports["z4"][polarization],
                fields["z2"][polarization],
                fields["z4"][polarization],
            )
            for polarization in POLARIZATIONS
        }
    diagnostic_tail_pass = bool(comparisons) and all(
        comparison["pass"] for comparison in comparisons.values()
    )
    opposite_sign_by_level = {}
    if all(artifact_checks.values()):
        for level in LEVELS:
            current_a = float(reports[level]["Ea"]["pte_solution"]["signed_current_A"])
            current_b = float(reports[level]["Eb"]["pte_solution"]["signed_current_A"])
            opposite_sign_by_level[level] = bool(
                current_a != 0.0
                and current_b != 0.0
                and np.signbit(current_a) != np.signbit(current_b)
            )
    certificate_valid = all(artifact_checks.values()) and bool(comparisons)
    status = (
        STATUS_PASS
        if certificate_valid and diagnostic_tail_pass
        else STATUS_BLOCKED
        if certificate_valid
        else STATUS_INVALID
    )
    generator = Path(__file__).resolve()
    return {
        "version": VERSION,
        "status": status,
        "certificate_valid": certificate_valid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "numerical z2-to-z4 stability of frozen-Q temperature, gradient, "
            "and floating-Au left/right-edge PTE current; not strict optical "
            "field convergence, actual-electrode validation, experimental "
            "accuracy, physical-parameter closure, or optimizer permission"
        ),
        "optical_tail_certificate": {
            "path": str(optical_resolved),
            "expected_sha256": expected_optical_certificate_sha256,
            "actual_sha256": optical_actual_sha,
            "checks": optical_checks,
        },
        "case_audits": audits,
        "artifact_checks": artifact_checks,
        "failed_artifact_checks": [
            name for name, passed in artifact_checks.items() if not passed
        ],
        "comparisons": comparisons,
        "diagnostic_pte_observable_tail_pass": diagnostic_tail_pass,
        "selected_diagnostic_optical_z_level_for_pte_observables": (
            "z2" if diagnostic_tail_pass else None
        ),
        "opposite_signed_diagnostic_current_by_level": opposite_sign_by_level,
        "strict_optical_mesh_converged": False,
        "strict_optical_mesh_selected": None,
        "actual_electrodes_validated": False,
        "electrical_mesh_converged": False,
        "thermal_physical_parameters_converged": False,
        "production_multiphysics_mesh_selected": False,
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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optical-tail-certificate", type=Path, required=True)
    parser.add_argument("--optical-tail-certificate-sha256", required=True)
    for level in LEVELS:
        for polarization in POLARIZATIONS:
            label = f"{level}-{polarization.lower()}"
            parser.add_argument(f"--{label}-report", type=Path, required=True)
            parser.add_argument(f"--{label}-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser()
    if not output.is_absolute() or not output.parent.is_dir() or output.exists():
        parser.error("--output must be a new absolute file under an existing directory")
    report_paths = {
        level: {
            polarization: getattr(args, f"{level}_{polarization.lower()}_report")
            for polarization in POLARIZATIONS
        }
        for level in LEVELS
    }
    report_hashes = {
        level: {
            polarization: getattr(
                args, f"{level}_{polarization.lower()}_report_sha256"
            )
            for polarization in POLARIZATIONS
        }
        for level in LEVELS
    }
    payload = build_certificate(
        args.optical_tail_certificate,
        args.optical_tail_certificate_sha256,
        report_paths,
        report_hashes,
    )
    _atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "status": payload["status"],
                "certificate_valid": payload["certificate_valid"],
                "diagnostic_pte_observable_tail_pass": payload[
                    "diagnostic_pte_observable_tail_pass"
                ],
                "selected_diagnostic_optical_z_level_for_pte_observables": payload[
                    "selected_diagnostic_optical_z_level_for_pte_observables"
                ],
                "optimizer_start_allowed": False,
            }
        )
    )
    return 0 if payload["certificate_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
