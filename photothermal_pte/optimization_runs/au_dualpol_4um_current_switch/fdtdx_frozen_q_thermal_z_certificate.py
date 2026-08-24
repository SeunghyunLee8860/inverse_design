#!/usr/bin/env python3
"""Certify two successive frozen-Q thermal z-refinement tail pairs."""

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

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_frozen_q_thermal_z_case import (
    REPORT_NAME,
    STATUS_READY as CASE_STATUS_READY,
    VERSION as CASE_VERSION,
    sha256,
)


VERSION = "fdtdx-frozen-q-thermal-z-certificate-v1"
STATUS_READY = "VALIDATED_DIAGNOSTIC_FDTDX_FROZEN_Q_THERMAL_Z_CONVERGENCE"
STATUS_BLOCKED = "BLOCKED_FDTDX_FROZEN_Q_THERMAL_Z_CONVERGENCE"
STATUS_EXCEPTION = "BLOCKED_FDTDX_FROZEN_Q_THERMAL_Z_CERTIFICATE_EXCEPTION"
CERTIFICATE_NAME = "FDTDX_FROZEN_Q_THERMAL_Z_CERTIFICATE.json"
LEVELS = (1, 2, 4)
POLARIZATIONS = ("Ea", "Eb")
SUCCESSIVE_PAIRS = ((1, 2), (2, 4))
TEMPERATURE_MAP_NRMSE_LIMIT = 0.02
TA_MAX_RELATIVE_LIMIT = 0.02
TA_MEAN_RELATIVE_LIMIT = 0.02
GRADIENT_COMBINED_NRMSE_LIMIT = 0.05


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


def _raw_snapshot(path: Path) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    required = {
        "ta_temperature_rise_K",
        "ta_gradient_x_K_m",
        "ta_gradient_y_K_m",
        "ta_x_centers_m",
        "ta_y_centers_m",
        "thermal_z_centers_m",
        "center_temperature_rise_K",
        "source_power_xy_W",
        "thermal_x_centers_m",
        "thermal_y_centers_m",
    }
    with np.load(path, allow_pickle=False) as archive:
        arrays_exact = set(archive.files) == required
        snapshot = {name: np.asarray(archive[name]) for name in required}
    checks = {
        "arrays_declared_exactly": arrays_exact,
        "all_arrays_finite": all(
            np.all(np.isfinite(value)) for value in snapshot.values()
        ),
        "ta_field_shapes_exact": all(
            snapshot[name].shape == (160, 160)
            for name in (
                "ta_temperature_rise_K",
                "ta_gradient_x_K_m",
                "ta_gradient_y_K_m",
            )
        ),
        "ta_coordinates_exact": snapshot["ta_x_centers_m"].shape == (160,)
        and snapshot["ta_y_centers_m"].shape == (160,),
        "temperature_nonnegative": bool(
            np.min(snapshot["ta_temperature_rise_K"]) >= 0.0
        ),
        "source_power_nonnegative": bool(
            np.min(snapshot["source_power_xy_W"]) >= 0.0
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
    resolved_root = supplied_root.resolve()
    report_path = (resolved_root / f"factor{factor}" / polarization / REPORT_NAME)
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
    raw_error = None
    snapshot = None
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
    runner_actual_sha = sha256(runner_path) if runner_exists else None
    mesh = report.get("thermal_mesh", {})
    checks = {
        "case_root_is_absolute": supplied_root.is_absolute(),
        "report_exists": report_exists,
        "expected_report_sha256_is_hex": len(expected_report_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_report_sha256),
        "report_sha256_matches": actual_report_sha == expected_report_sha256,
        "version_status_ready": report.get("version") == CASE_VERSION
        and report.get("status") == CASE_STATUS_READY
        and report.get("ready") is True,
        "factor_and_polarization_exact": report.get(
            "thermal_z_refinement_factor"
        )
        == factor
        and report.get("polarization") == polarization,
        "diagnostic_scope_and_blocks_exact": report.get("diagnostic_only") is True
        and report.get("optical_mesh_blocked") is True
        and report.get("production_mesh_selected") is False
        and report.get("optimizer_start_allowed") is False,
        "input_artifact_audit_ready": report.get("input_audit", {}).get("ready")
        is True
        and _all_true(report.get("input_audit", {}).get("checks", {})),
        "raw_field_audit_ready": report.get("raw_field_audit", {}).get("ready")
        is True
        and _all_true(report.get("raw_field_audit", {}).get("checks", {})),
        "mapping_solver_provenance_checks_all_true": _all_true(
            report.get("mapping_checks", {})
        )
        and _all_true(report.get("solver_checks", {}))
        and _all_true(report.get("provenance_checks", {})),
        "mesh_shape_scales_only_in_z": mesh.get("shape")
        == [266, 266, 33 * factor]
        and mesh.get("unknowns") == 2_334_948 * factor,
        "raw_path_is_absolute": raw_supplied.is_absolute(),
        "raw_exists": raw_exists,
        "raw_sha256_matches_report": actual_raw_sha == raw_record.get("sha256"),
        "raw_snapshot_checks_all_true": _all_true(raw_checks),
        "runner_path_is_absolute": runner_supplied.is_absolute(),
        "runner_exists": runner_exists,
        "runner_sha256_matches": runner_actual_sha == provenance.get("runner_sha256"),
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
        "runner_actual_sha256": runner_actual_sha,
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
    coarse_report = reports[coarse_factor]
    fine_report = reports[fine_factor]
    temperature_nrmse = _nrmse(
        coarse["ta_temperature_rise_K"], fine["ta_temperature_rise_K"]
    )
    maximum_relative = _relative(
        float(coarse_report["thermal_solution"]["ta_max_temperature_rise_K"]),
        float(fine_report["thermal_solution"]["ta_max_temperature_rise_K"]),
    )
    mean_relative = _relative(
        float(coarse_report["thermal_solution"]["ta_mean_temperature_rise_K"]),
        float(fine_report["thermal_solution"]["ta_mean_temperature_rise_K"]),
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
    coordinates_exact = all(
        np.array_equal(coarse[name], fine[name])
        for name in (
            "ta_x_centers_m",
            "ta_y_centers_m",
            "thermal_x_centers_m",
            "thermal_y_centers_m",
        )
    )
    source_xy_nrmse = _nrmse(
        coarse["source_power_xy_W"], fine["source_power_xy_W"]
    )
    gates = {
        "lateral_coordinates_exact": coordinates_exact,
        "source_xy_power_distribution_exact_to_roundoff": source_xy_nrmse
        <= 5.0e-12,
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
            "source_xy_power_nrmse": source_xy_nrmse,
        },
        "limits": {
            "ta_temperature_map_nrmse": TEMPERATURE_MAP_NRMSE_LIMIT,
            "ta_max_temperature_relative": TA_MAX_RELATIVE_LIMIT,
            "ta_mean_temperature_relative": TA_MEAN_RELATIVE_LIMIT,
            "ta_combined_gradient_nrmse": GRADIENT_COMBINED_NRMSE_LIMIT,
        },
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
    }


def selection(two_successive_pairs_pass: bool) -> dict[str, Any]:
    return {
        "two_successive_thermal_z_tail_pairs_pass": two_successive_pairs_pass,
        "selected_diagnostic_frozen_q_thermal_z_factor": (
            2 if two_successive_pairs_pass else None
        ),
        "selected_diagnostic_thermal_shape": (
            [266, 266, 66] if two_successive_pairs_pass else None
        ),
        "thermal_z_diagnostic_converged": two_successive_pairs_pass,
        "thermal_xy_converged": False,
        "optical_mesh_converged": False,
        "electrical_mesh_converged": False,
        "production_multiphysics_mesh_selected": False,
        "optimizer_start_allowed": False,
        "interpretation": (
            "factor 2 is selected only for the frozen-z32-Q diagnostic because "
            "factor1-to-2 and factor2-to-4 both pass for Ea and Eb; the blocked "
            "optical mesh, unconverged thermal x/y mesh, and unconverged actual "
            "electrical geometry still forbid production promotion"
        ),
    }


def build_certificate(
    case_root: Path, report_sha256s: Mapping[int, Mapping[str, str]]
) -> dict[str, Any]:
    if set(report_sha256s) != set(LEVELS) or any(
        set(report_sha256s[factor]) != set(POLARIZATIONS) for factor in LEVELS
    ):
        raise ValueError("report SHA mapping must contain factors 1/2/4 and Ea/Eb")
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
            key = f"factor{coarse}_to_factor{fine}"
            comparisons[key] = {
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
    all_comparisons_pass = all_cases_ready and all(
        comparisons[f"factor{coarse}_to_factor{fine}"][polarization]["pass"]
        for coarse, fine in SUCCESSIVE_PAIRS
        for polarization in POLARIZATIONS
    )
    common_input_hashes = {
        reports[factor][polarization]
        .get("input_audit", {})
        .get("actual_certificate_sha256")
        for factor in LEVELS
        for polarization in POLARIZATIONS
    }
    runner_commits = {
        reports[factor][polarization]
        .get("provenance", {})
        .get("repository_commit")
        for factor in LEVELS
        for polarization in POLARIZATIONS
    }
    global_checks = {
        "all_six_case_artifacts_revalidated": all_cases_ready,
        "all_cases_bind_one_z32_certificate_hash": len(common_input_hashes) == 1
        and None not in common_input_hashes,
        "all_cases_bind_one_clean_runner_commit": len(runner_commits) == 1
        and None not in runner_commits,
        "both_successive_pairs_evaluated": len(comparisons) == 2,
        "both_successive_pairs_pass_for_both_polarizations": all_comparisons_pass,
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
            "diagnostic-only thermal z convergence under one frozen blocked-z32 "
            "exact-binary FDTDX Q field per polarization; no thermal x/y, optical, "
            "electrical, adjoint, optimizer, or production-mesh certification"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_root": str(case_root.expanduser().resolve()),
        "case_audits": audits,
        "comparisons": comparisons,
        "global_checks": global_checks,
        "failed_global_checks": [
            name for name, passed in global_checks.items() if not passed
        ],
        "selection": selection(all_comparisons_pass),
        "production_multiphysics_mesh_selected": False,
        "optimizer_start_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
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
        report_sha256s = {
            factor: {
                polarization: getattr(
                    args, f"factor{factor}_{polarization.lower()}_report_sha256"
                )
                for polarization in POLARIZATIONS
            }
            for factor in LEVELS
        }
        payload = build_certificate(args.case_root, report_sha256s)
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
