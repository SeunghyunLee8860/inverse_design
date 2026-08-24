#!/usr/bin/env python3
"""Certify frozen-Q thermal x/y convergence at diagnostic thermal z factor 2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any, Mapping

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_frozen_q_thermal_xy_case import (
    REPORT_NAME,
    STATUS_READY as CASE_STATUS_READY,
    THERMAL_Z_REFINEMENT_FACTOR,
    VERSION as CASE_VERSION,
    sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_frozen_q_thermal_z_certificate import (
    GRADIENT_COMBINED_NRMSE_LIMIT,
    STATUS_READY as PRIOR_STATUS_READY,
    TA_MAX_RELATIVE_LIMIT,
    TA_MEAN_RELATIVE_LIMIT,
    TEMPERATURE_MAP_NRMSE_LIMIT,
    VERSION as PRIOR_VERSION,
)


VERSION = "fdtdx-frozen-q-thermal-xy-certificate-v1"
STATUS_READY = "VALIDATED_DIAGNOSTIC_FDTDX_FROZEN_Q_THERMAL_XY_CONVERGENCE"
STATUS_BLOCKED = "BLOCKED_FDTDX_FROZEN_Q_THERMAL_XY_CONVERGENCE"
STATUS_EXCEPTION = "BLOCKED_FDTDX_FROZEN_Q_THERMAL_XY_CERTIFICATE_EXCEPTION"
CERTIFICATE_NAME = "FDTDX_FROZEN_Q_THERMAL_XY_CERTIFICATE.json"
LEVELS = (1, 2, 4)
POLARIZATIONS = ("Ea", "Eb")
SUCCESSIVE_PAIRS = ((1, 2), (2, 4))
BASE_FIELD_NAMES = (
    "ta_temperature_rise_K",
    "ta_gradient_x_K_m",
    "ta_gradient_y_K_m",
    "ta_x_centers_m",
    "ta_y_centers_m",
    "source_power_xy_W",
)
BASE_COORDINATE_ATOL_M = 2.0e-18


def _all_true(values: Mapping[str, Any]) -> bool:
    return bool(values) and all(value is True for value in values.values())


def _relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)


def _nrmse(coarse: np.ndarray, fine: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(fine) - np.asarray(coarse))
        / max(float(np.linalg.norm(fine)), np.finfo(float).tiny)
    )


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit_prior_z_certificate(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = path.expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_file()
    actual = sha256(resolved) if exists else None
    payload = json.loads(resolved.read_text(encoding="utf-8")) if exists else {}
    selection = payload.get("selection", {})
    provenance = payload.get("certificate_provenance", {})
    checks = {
        "path_is_absolute": supplied.is_absolute(),
        "file_exists": exists,
        "sha256_matches": actual == expected_sha256,
        "version_status_ready": payload.get("version") == PRIOR_VERSION
        and payload.get("status") == PRIOR_STATUS_READY
        and payload.get("ready") is True,
        "global_checks_all_true": _all_true(payload.get("global_checks", {}))
        and payload.get("failed_global_checks") == [],
        "diagnostic_z_factor2_selected": selection.get(
            "selected_diagnostic_frozen_q_thermal_z_factor"
        )
        == THERMAL_Z_REFINEMENT_FACTOR
        and selection.get("thermal_z_diagnostic_converged") is True,
        "xy_and_production_remained_unselected": selection.get(
            "thermal_xy_converged"
        )
        is False
        and selection.get("production_multiphysics_mesh_selected") is False
        and payload.get("production_multiphysics_mesh_selected") is False,
        "optimizer_remained_forbidden": selection.get("optimizer_start_allowed")
        is False
        and payload.get("optimizer_start_allowed") is False,
        "generator_repository_was_clean": provenance.get(
            "repository_dirty_porcelain"
        )
        == "",
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


def _raw_snapshot(path: Path) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    required = {
        *BASE_FIELD_NAMES,
        "ta_temperature_native_K",
        "ta_x_native_centers_m",
        "ta_y_native_centers_m",
        "source_power_native_xy_W",
        "thermal_x_native_centers_m",
        "thermal_y_native_centers_m",
        "thermal_z_centers_m",
        "center_temperature_rise_K",
    }
    with np.load(path, allow_pickle=False) as archive:
        exact = set(archive.files) == required
        snapshot = {name: np.asarray(archive[name]) for name in required}
    checks = {
        "arrays_declared_exactly": exact,
        "all_arrays_finite": all(
            np.all(np.isfinite(value)) for value in snapshot.values()
        ),
        "base_ta_fields_are_160x160": all(
            snapshot[name].shape == (160, 160)
            for name in (
                "ta_temperature_rise_K",
                "ta_gradient_x_K_m",
                "ta_gradient_y_K_m",
            )
        ),
        "base_source_is_266x266": snapshot["source_power_xy_W"].shape
        == (266, 266),
        "base_coordinates_are_exact_shapes": snapshot["ta_x_centers_m"].shape
        == (160,)
        and snapshot["ta_y_centers_m"].shape == (160,),
        "base_temperature_nonnegative": bool(
            np.min(snapshot["ta_temperature_rise_K"]) >= 0.0
        ),
        "native_source_nonnegative": bool(
            np.min(snapshot["source_power_native_xy_W"]) >= 0.0
        ),
    }
    return snapshot, checks


def audit_case(
    case_root: Path,
    factor: int,
    polarization: str,
    expected_report_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray] | None]:
    supplied_root = case_root.expanduser()
    report_path = (
        supplied_root.resolve() / f"factor{factor}" / polarization / REPORT_NAME
    )
    report_exists = report_path.is_file()
    actual_report_sha = sha256(report_path) if report_exists else None
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_exists
        else {}
    )
    raw_record = report.get("raw", {})
    raw_supplied = Path(raw_record.get("path", "")).expanduser()
    raw_path = raw_supplied.resolve()
    raw_exists = raw_path.is_file()
    actual_raw_sha = sha256(raw_path) if raw_exists else None
    snapshot = None
    raw_error = None
    raw_checks: dict[str, bool] = {}
    if raw_exists:
        try:
            snapshot, raw_checks = _raw_snapshot(raw_path)
        except Exception as error:
            raw_error = repr(error)
    provenance = report.get("provenance", {})
    runner_supplied = Path(provenance.get("runner_path", "")).expanduser()
    runner_path = runner_supplied.resolve()
    runner_exists = runner_path.is_file()
    runner_sha = sha256(runner_path) if runner_exists else None
    mesh = report.get("thermal_mesh", {})
    expected_shape = [266 * factor, 266 * factor, 66]
    checks = {
        "case_root_is_absolute": supplied_root.is_absolute(),
        "report_exists": report_exists,
        "expected_report_sha256_is_hex": len(expected_report_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_report_sha256),
        "report_sha256_matches": actual_report_sha == expected_report_sha256,
        "version_status_ready": report.get("version") == CASE_VERSION
        and report.get("status") == CASE_STATUS_READY
        and report.get("ready") is True,
        "factor_polarization_and_z_exact": report.get(
            "thermal_xy_refinement_factor"
        )
        == factor
        and report.get("thermal_z_refinement_factor")
        == THERMAL_Z_REFINEMENT_FACTOR
        and report.get("polarization") == polarization,
        "diagnostic_scope_and_blocks_exact": report.get("diagnostic_only") is True
        and report.get("optical_mesh_blocked") is True
        and report.get("thermal_domain_converged") is False
        and report.get("electrical_mesh_converged") is False
        and report.get("production_mesh_selected") is False
        and report.get("optimizer_start_allowed") is False,
        "all_embedded_audits_ready": report.get("input_audit", {}).get("ready")
        is True
        and report.get("raw_field_audit", {}).get("ready") is True
        and _all_true(report.get("mapping_checks", {}))
        and _all_true(report.get("mesh_checks", {}))
        and _all_true(report.get("solver_checks", {}))
        and _all_true(report.get("provenance_checks", {})),
        "mesh_shape_and_unknowns_exact": mesh.get("shape") == expected_shape
        and mesh.get("unknowns") == 4_669_896 * factor**2,
        "raw_path_is_absolute": raw_supplied.is_absolute(),
        "raw_exists": raw_exists,
        "raw_sha256_matches_report": actual_raw_sha == raw_record.get("sha256"),
        "raw_snapshot_checks_all_true": _all_true(raw_checks),
        "native_shapes_scale_with_factor": snapshot is not None
        and snapshot["ta_temperature_native_K"].shape
        == (160 * factor, 160 * factor)
        and snapshot["source_power_native_xy_W"].shape
        == (266 * factor, 266 * factor),
        "runner_path_is_absolute": runner_supplied.is_absolute(),
        "runner_exists": runner_exists,
        "runner_sha256_matches": runner_sha == provenance.get("runner_sha256"),
        "repository_was_clean": provenance.get(
            "repository_dirty_porcelain_before"
        )
        == ""
        and provenance.get("repository_dirty_porcelain_after") == "",
        "lumerical_was_not_used": provenance.get("lumerical_used") is False,
    }
    audit = {
        "factor": factor,
        "polarization": polarization,
        "report_path": str(report_path),
        "expected_report_sha256": expected_report_sha256,
        "actual_report_sha256": actual_report_sha,
        "raw_path": str(raw_path),
        "actual_raw_sha256": actual_raw_sha,
        "runner_path": str(runner_path),
        "runner_actual_sha256": runner_sha,
        "raw_error": raw_error,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
    }
    return report, audit, snapshot if audit["ready"] else None


def compare_pair(
    coarse_factor: int,
    fine_factor: int,
    polarization: str,
    reports: Mapping[int, Mapping[str, Any]],
    snapshots: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    coarse = snapshots[coarse_factor]
    fine = snapshots[fine_factor]
    coarse_solution = reports[coarse_factor]["thermal_solution"]
    fine_solution = reports[fine_factor]["thermal_solution"]
    temperature_nrmse = _nrmse(
        coarse["ta_temperature_rise_K"], fine["ta_temperature_rise_K"]
    )
    maximum_relative = _relative(
        float(coarse_solution["ta_base_max_temperature_rise_K"]),
        float(fine_solution["ta_base_max_temperature_rise_K"]),
    )
    mean_relative = _relative(
        float(coarse_solution["ta_base_mean_temperature_rise_K"]),
        float(fine_solution["ta_base_mean_temperature_rise_K"]),
    )
    gradient_numerator = np.sqrt(
        np.sum(
            (fine["ta_gradient_x_K_m"] - coarse["ta_gradient_x_K_m"]) ** 2
        )
        + np.sum(
            (fine["ta_gradient_y_K_m"] - coarse["ta_gradient_y_K_m"]) ** 2
        )
    )
    gradient_denominator = np.sqrt(
        np.sum(fine["ta_gradient_x_K_m"] ** 2)
        + np.sum(fine["ta_gradient_y_K_m"] ** 2)
    )
    gradient_nrmse = float(
        gradient_numerator
        / max(float(gradient_denominator), np.finfo(float).tiny)
    )
    source_nrmse = _nrmse(
        coarse["source_power_xy_W"], fine["source_power_xy_W"]
    )
    gates = {
        "base_coordinates_reconstructed_within_2e-18_m": np.allclose(
            coarse["ta_x_centers_m"],
            fine["ta_x_centers_m"],
            rtol=0.0,
            atol=BASE_COORDINATE_ATOL_M,
        )
        and np.allclose(
            coarse["ta_y_centers_m"],
            fine["ta_y_centers_m"],
            rtol=0.0,
            atol=BASE_COORDINATE_ATOL_M,
        ),
        "source_xy_distribution_exact_to_roundoff": source_nrmse <= 5e-12,
        "ta_temperature_map_nrmse_within_2pct": temperature_nrmse
        <= TEMPERATURE_MAP_NRMSE_LIMIT,
        "ta_max_temperature_relative_within_2pct": maximum_relative
        <= TA_MAX_RELATIVE_LIMIT,
        "ta_mean_temperature_relative_within_2pct": mean_relative
        <= TA_MEAN_RELATIVE_LIMIT,
        "ta_combined_gradient_nrmse_within_5pct": gradient_nrmse
        <= GRADIENT_COMBINED_NRMSE_LIMIT,
    }
    return {
        "polarization": polarization,
        "coarse_factor": coarse_factor,
        "fine_factor": fine_factor,
        "metrics": {
            "ta_temperature_map_nrmse": temperature_nrmse,
            "ta_max_temperature_relative": maximum_relative,
            "ta_mean_temperature_relative": mean_relative,
            "ta_combined_gradient_nrmse": gradient_nrmse,
            "source_xy_power_nrmse": source_nrmse,
        },
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
    }


def crosscheck_factor1_to_prior_z2(
    prior: Mapping[str, Any], snapshots: Mapping[str, Mapping[str, np.ndarray]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for polarization in POLARIZATIONS:
        prior_audit = prior.get("case_audits", {}).get("2", {}).get(
            polarization, {}
        )
        raw_path = Path(prior_audit.get("raw_path", "")).expanduser().resolve()
        exists = raw_path.is_file()
        actual_sha = sha256(raw_path) if exists else None
        arrays_equal = {}
        if exists:
            with np.load(raw_path, allow_pickle=False) as archive:
                arrays_equal = {
                    name: np.array_equal(
                        np.asarray(archive[name]), snapshots[polarization][name]
                    )
                    for name in BASE_FIELD_NAMES
                }
        checks = {
            "prior_factor2_case_was_ready": prior_audit.get("ready") is True,
            "prior_raw_exists": exists,
            "prior_raw_sha256_revalidated": actual_sha
            == prior_audit.get("actual_raw_sha256"),
            "all_base_arrays_exactly_equal": bool(arrays_equal)
            and all(arrays_equal.values()),
        }
        result[polarization] = {
            "prior_raw_path": str(raw_path),
            "prior_raw_actual_sha256": actual_sha,
            "array_equal": arrays_equal,
            "checks": checks,
            "ready": all(checks.values()),
        }
    return result


def selection(two_pairs_pass: bool, baseline_rebound: bool) -> dict[str, Any]:
    converged = two_pairs_pass and baseline_rebound
    return {
        "factor1_exactly_rebound_to_selected_z2_baseline": baseline_rebound,
        "two_successive_thermal_xy_tail_pairs_pass": two_pairs_pass,
        "selected_diagnostic_frozen_q_thermal_z_factor": 2 if converged else None,
        "selected_diagnostic_frozen_q_thermal_xy_factor": 2 if converged else None,
        "selected_diagnostic_thermal_shape": [532, 532, 66] if converged else None,
        "thermal_xy_z_diagnostic_converged": converged,
        "thermal_domain_and_boundary_converged": False,
        "optical_mesh_converged": False,
        "electrical_mesh_converged": False,
        "production_multiphysics_mesh_selected": False,
        "optimizer_start_allowed": False,
    }


def build_certificate(
    prior_path: Path,
    prior_sha256: str,
    case_root: Path,
    report_sha256s: Mapping[int, Mapping[str, str]],
) -> dict[str, Any]:
    if set(report_sha256s) != set(LEVELS) or any(
        set(report_sha256s[factor]) != set(POLARIZATIONS) for factor in LEVELS
    ):
        raise ValueError("report SHA mapping must contain factors 1/2/4 and Ea/Eb")
    prior, prior_audit = audit_prior_z_certificate(prior_path, prior_sha256)
    reports: dict[int, dict[str, Any]] = {factor: {} for factor in LEVELS}
    audits: dict[int, dict[str, Any]] = {factor: {} for factor in LEVELS}
    snapshots: dict[int, dict[str, Any]] = {factor: {} for factor in LEVELS}
    for factor in LEVELS:
        for polarization in POLARIZATIONS:
            report, audit, snapshot = audit_case(
                case_root,
                factor,
                polarization,
                report_sha256s[factor][polarization],
            )
            reports[factor][polarization] = report
            audits[factor][polarization] = audit
            snapshots[factor][polarization] = snapshot
    all_cases_ready = all(
        audits[factor][polarization]["ready"]
        for factor in LEVELS
        for polarization in POLARIZATIONS
    )
    comparisons: dict[str, Any] = {}
    if all_cases_ready:
        for coarse, fine in SUCCESSIVE_PAIRS:
            comparisons[f"factor{coarse}_to_factor{fine}"] = {
                polarization: compare_pair(
                    coarse,
                    fine,
                    polarization,
                    {factor: reports[factor][polarization] for factor in LEVELS},
                    {
                        factor: snapshots[factor][polarization]
                        for factor in LEVELS
                    },
                )
                for polarization in POLARIZATIONS
            }
    all_pairs_pass = all_cases_ready and all(
        comparisons[f"factor{coarse}_to_factor{fine}"][polarization]["pass"]
        for coarse, fine in SUCCESSIVE_PAIRS
        for polarization in POLARIZATIONS
    )
    baseline = (
        crosscheck_factor1_to_prior_z2(prior, snapshots[1])
        if all_cases_ready and prior_audit["ready"]
        else {}
    )
    baseline_ready = bool(baseline) and all(
        baseline[polarization]["ready"] for polarization in POLARIZATIONS
    )
    input_hashes = {
        reports[factor][polarization]
        .get("input_audit", {})
        .get("actual_certificate_sha256")
        for factor in LEVELS
        for polarization in POLARIZATIONS
    }
    commits = {
        reports[factor][polarization]
        .get("provenance", {})
        .get("repository_commit")
        for factor in LEVELS
        for polarization in POLARIZATIONS
    }
    global_checks = {
        "prior_thermal_z_certificate_revalidated": prior_audit["ready"],
        "all_six_xy_case_artifacts_revalidated": all_cases_ready,
        "xy_factor1_exactly_rebounds_to_prior_z_factor2": baseline_ready,
        "all_cases_bind_one_optical_z32_certificate_hash": len(input_hashes) == 1
        and None not in input_hashes,
        "all_cases_bind_one_clean_runner_commit": len(commits) == 1
        and None not in commits,
        "both_successive_xy_pairs_pass_for_both_polarizations": all_pairs_pass,
        "production_and_optimizer_remain_forbidden": all(
            reports[factor][polarization].get("production_mesh_selected") is False
            and reports[factor][polarization].get("optimizer_start_allowed") is False
            for factor in LEVELS
            for polarization in POLARIZATIONS
        ),
    }
    ready = all(global_checks.values())
    return {
        "version": VERSION,
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "scope": (
            "diagnostic-only frozen-Q thermal x/y convergence at prior-selected "
            "thermal z factor 2; no domain/boundary, optical, electrical, "
            "adjoint, optimizer, or production-mesh certification"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prior_thermal_z_certificate": prior_audit,
        "case_root": str(case_root.expanduser().resolve()),
        "case_audits": audits,
        "factor1_prior_z2_crosscheck": baseline,
        "comparisons": comparisons,
        "global_checks": global_checks,
        "failed_global_checks": [
            name for name, passed in global_checks.items() if not passed
        ],
        "selection": selection(all_pairs_pass, baseline_ready),
        "production_multiphysics_mesh_selected": False,
        "optimizer_start_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--prior-thermal-z-certificate", type=Path, required=True)
    parser.add_argument("--prior-thermal-z-certificate-sha256", required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    for factor in LEVELS:
        for polarization in POLARIZATIONS:
            parser.add_argument(
                f"--factor{factor}-{polarization.lower()}-report-sha256",
                required=True,
            )
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
        hashes = {
            factor: {
                polarization: getattr(
                    args, f"factor{factor}_{polarization.lower()}_report_sha256"
                )
                for polarization in POLARIZATIONS
            }
            for factor in LEVELS
        }
        payload = build_certificate(
            args.prior_thermal_z_certificate,
            args.prior_thermal_z_certificate_sha256,
            args.case_root,
            hashes,
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
            "production_multiphysics_mesh_selected": False,
            "optimizer_start_allowed": False,
        }
    _atomic_json(output / CERTIFICATE_NAME, payload)
    print(
        json.dumps(
            {
                "certificate": str(output / CERTIFICATE_NAME),
                "status": payload["status"],
                "ready": payload["ready"],
                "selection": payload.get("selection"),
            }
        )
    )
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
