#!/usr/bin/env python3
"""Diagnose where the blocked z8-to-z16 field/Q changes are concentrated."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_full_z_certificate import (
    _interpolate_fine_to_coarse_z,
    _z_coordinates,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_time_settling_certificate import (
    sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_exact_binary_control import (
    _json_default,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_full_z_extension_certificate import (
    STATUS_BLOCKED as EXPECTED_STATUS,
    VERSION as EXPECTED_VERSION,
)


VERSION = "fdtdx-increment-state-z16-interface-diagnostic-v1"
STATUS_READY = "VALIDATED_FDTDX_INCREMENT_STATE_Z16_INTERFACE_DIAGNOSTIC"
STATUS_BLOCKED = "BLOCKED_FDTDX_INCREMENT_STATE_Z16_INTERFACE_DIAGNOSTIC"
REPORT_NAME = "FDTDX_INCREMENT_STATE_Z16_INTERFACE_DIAGNOSTIC.json"


def weighted_error_diagnostic(
    coarse: np.ndarray,
    fine_on_coarse: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    left = np.asarray(coarse)
    right = np.asarray(fine_on_coarse)
    weight = np.asarray(weights, dtype=np.float64)
    if left.shape != right.shape or left.shape != weight.shape or left.ndim != 3:
        raise ValueError("field and weights must share one (x,y,z) shape")
    if np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    difference = right - left
    plane_error = np.sum(np.abs(difference) ** 2 * weight, axis=(0, 1))
    plane_norm = np.sum(np.abs(right) ** 2 * weight, axis=(0, 1))
    error = float(np.sum(plane_error))
    norm = float(np.sum(plane_norm))
    inner = np.sum(np.conj(right) * left * weight)
    right_norm = float(np.sum(np.abs(right) ** 2 * weight))
    scale = inner / max(right_norm, np.finfo(float).tiny)
    aligned_error = float(np.sum(np.abs(scale * right - left) ** 2 * weight))

    boundary: dict[str, Any] = {}
    trimmed: dict[str, Any] = {}
    for count in (1, 2):
        indices = np.r_[0:count, left.shape[-1] - count : left.shape[-1]]
        indices = np.unique(indices)
        boundary[str(count)] = {
            "planes_each_side": count,
            "error_fraction": float(np.sum(plane_error[indices]))
            / max(error, np.finfo(float).tiny),
            "fine_norm_fraction": float(np.sum(plane_norm[indices]))
            / max(norm, np.finfo(float).tiny),
        }
        if left.shape[-1] > 2 * count:
            interior = slice(count, -count)
            interior_error = float(np.sum(plane_error[interior]))
            interior_norm = float(np.sum(plane_norm[interior]))
            trimmed[str(count)] = math.sqrt(
                interior_error / max(interior_norm, np.finfo(float).tiny)
            )
    return {
        "complex_E_NRMSE": math.sqrt(error / max(norm, np.finfo(float).tiny)),
        "best_global_complex_scale_fine_to_coarse": [
            float(scale.real),
            float(scale.imag),
        ],
        "best_scale_amplitude": float(abs(scale)),
        "best_scale_phase_rad": float(np.angle(scale)),
        "scale_aligned_complex_E_NRMSE": math.sqrt(
            aligned_error / max(norm, np.finfo(float).tiny)
        ),
        "boundary_concentration": boundary,
        "trimmed_boundary_complex_E_NRMSE": trimmed,
        "weighted_error_numerator": error,
        "weighted_fine_norm_denominator": norm,
    }


def _certificate_audit(path: Path, expected_sha256: str) -> tuple[dict, dict]:
    supplied = path.expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_file()
    actual = sha256(resolved) if exists else None
    payload = json.loads(resolved.read_text(encoding="utf-8")) if exists else {}
    cases = payload.get("case_audits", {})
    comparison = payload.get("z8_to_z16_comparison", {})
    checks = {
        "path_is_absolute": supplied.is_absolute(),
        "file_exists": exists,
        "sha256_matches": actual == expected_sha256,
        "version_status_exact_blocked": payload.get("version") == EXPECTED_VERSION
        and payload.get("status") == EXPECTED_STATUS
        and payload.get("ready") is False,
        "global_checks_all_true": bool(payload.get("global_checks"))
        and all(value is True for value in payload["global_checks"].values()),
        "all_case_artifacts_ready": all(
            cases.get(level, {}).get(polarization, {}).get("ready") is True
            for level in ("z8", "z16")
            for polarization in ("Ea", "Eb")
        ),
        "comparison_failed_after_evaluation": comparison.get("pass") is False
        and isinstance(comparison.get("metrics"), dict)
        and not comparison.get("error"),
        "optimizer_forbidden": payload.get("optimizer_start_allowed") is False,
    }
    return payload, {
        "path": str(resolved),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
    }


def _load_case(
    certificate: dict,
    level: str,
    polarization: str,
) -> tuple[dict, dict[str, np.ndarray], dict]:
    audit = certificate["case_audits"][level][polarization]
    report_path = Path(audit["path"]).resolve()
    raw_path = Path(audit["raw"]["path"]).resolve()
    report_sha = sha256(report_path)
    raw_sha = sha256(raw_path)
    checks = {
        "report_sha256_matches_certificate": report_sha == audit["actual_sha256"],
        "raw_sha256_matches_certificate": raw_sha == audit["raw"]["actual_sha256"],
    }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = {
        "au_late",
        "tairte4_late",
        "electric_dual_volume_au_m3",
        "electric_dual_volume_tairte4_m3",
        "grid_z_edges_m",
        "solver_mask",
        "target",
    }
    with np.load(raw_path, allow_pickle=False) as archive:
        checks["required_arrays_present"] = required.issubset(archive.files)
        arrays = {
            name: np.asarray(archive[name]) for name in required if name in archive
        }
    checks["required_arrays_finite"] = set(arrays) == required and all(
        np.all(np.isfinite(value)) for value in arrays.values()
    )
    return (
        report,
        arrays,
        {
            "report_path": str(report_path),
            "report_sha256": report_sha,
            "raw_path": str(raw_path),
            "raw_sha256": raw_sha,
            "checks": checks,
            "ready": all(checks.values()),
        },
    )


def _power_shares(report: dict) -> dict[str, Any]:
    late = report["evaluation"]["Q"]["late"]
    total = float(late["total_W"])
    result: dict[str, Any] = {}
    for material in ("au", "tairte4"):
        record = late["by_material"][material]
        material_total = float(record["total_W"])
        result[material] = {
            "total_W_unscaled": material_total,
            "fraction_of_all_absorption": material_total / total,
            "components": {
                axis: {
                    "power_W_unscaled": float(record["component_W"][axis]),
                    "fraction_of_material_absorption": float(
                        record["component_W"][axis]
                    )
                    / material_total,
                    "fraction_of_all_absorption": float(record["component_W"][axis])
                    / total,
                }
                for axis in ("x", "y", "z")
            },
        }
    return result


def build_diagnostic(certificate_path: Path, certificate_sha256: str) -> dict:
    certificate, certificate_audit = _certificate_audit(
        certificate_path, certificate_sha256
    )
    if not certificate_audit["ready"]:
        raise RuntimeError(f"extension certificate audit failed: {certificate_audit}")
    reports: dict[str, dict[str, dict]] = {level: {} for level in ("z8", "z16")}
    arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {
        level: {} for level in ("z8", "z16")
    }
    artifact_audits: dict[str, dict[str, dict]] = {level: {} for level in ("z8", "z16")}
    for level in ("z8", "z16"):
        for polarization in ("Ea", "Eb"):
            report, raw, audit = _load_case(certificate, level, polarization)
            reports[level][polarization] = report
            arrays[level][polarization] = raw
            artifact_audits[level][polarization] = audit
            if not audit["ready"]:
                raise RuntimeError(
                    f"case artifact audit failed: {level}/{polarization}"
                )

    diagnostics: dict[str, Any] = {}
    maximum_component_change = 0.0
    maximum_component_share = 0.0
    for polarization in ("Ea", "Eb"):
        coarse = arrays["z8"][polarization]
        fine = arrays["z16"][polarization]
        coarse_report = reports["z8"][polarization]
        fine_report = reports["z16"][polarization]
        diagnostics[polarization] = {
            "power_shares_z8": _power_shares(coarse_report),
            "power_shares_z16": _power_shares(fine_report),
            "material_fields": {},
            "target_components": {},
        }
        component_changes = certificate["z8_to_z16_comparison"]["per_polarization"][
            polarization
        ]["material_component_Q_relative_change"]
        for material in ("au", "tairte4"):
            diagnostics[polarization]["material_fields"][material] = {}
            key = "au_design" if material == "au" else "fixed_tairte4"
            coarse_bounds = tuple(coarse_report["placement"][key][2])
            fine_bounds = tuple(fine_report["placement"][key][2])
            mask = coarse["solver_mask"] if material == "au" else None
            for component, axis in enumerate(("x", "y", "z")):
                coarse_z = _z_coordinates(
                    coarse["grid_z_edges_m"], coarse_bounds, component
                )
                fine_z = _z_coordinates(fine["grid_z_edges_m"], fine_bounds, component)
                fine_on_coarse = _interpolate_fine_to_coarse_z(
                    fine[f"{material}_late"][component], fine_z, coarse_z
                )
                weights = coarse[f"electric_dual_volume_{material}_m3"][component]
                if mask is not None:
                    weights = weights * mask[:, :, None]
                field = weighted_error_diagnostic(
                    coarse[f"{material}_late"][component],
                    fine_on_coarse,
                    weights,
                )
                z16_share = diagnostics[polarization]["power_shares_z16"][material][
                    "components"
                ][axis]["fraction_of_all_absorption"]
                field["Q_relative_change"] = component_changes[material][axis]
                field["z16_fraction_of_all_absorption"] = z16_share
                diagnostics[polarization]["material_fields"][material][axis] = field
                if component_changes[material][axis] > maximum_component_change:
                    maximum_component_change = component_changes[material][axis]
                    maximum_component_share = z16_share

        for component, axis in enumerate(("x", "y")):
            shape = coarse["target"][component].shape
            diagnostics[polarization]["target_components"][axis] = (
                weighted_error_diagnostic(
                    coarse["target"][component],
                    fine["target"][component],
                    np.ones(shape, dtype=np.float64),
                )
            )

    ta_max = max(
        certificate["z8_to_z16_comparison"]["per_polarization"][polarization][
            "material_region_complex_E_NRMSE_after_fine_to_coarse_z_interpolation"
        ]["tairte4"]
        for polarization in ("Ea", "Eb")
    )
    au_max = max(
        certificate["z8_to_z16_comparison"]["per_polarization"][polarization][
            "material_region_complex_E_NRMSE_after_fine_to_coarse_z_interpolation"
        ]["au"]
        for polarization in ("Ea", "Eb")
    )
    checks = {
        "certificate_revalidated": certificate_audit["ready"],
        "all_case_artifacts_revalidated": all(
            artifact_audits[level][polarization]["ready"]
            for level in ("z8", "z16")
            for polarization in ("Ea", "Eb")
        ),
        "diagnostic_values_finite": bool(
            math.isfinite(ta_max)
            and math.isfinite(au_max)
            and math.isfinite(maximum_component_share)
        ),
    }
    ready = all(checks.values())
    return {
        "version": VERSION,
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "scope": (
            "diagnostic decomposition only; convergence gates are unchanged and "
            "no mesh or optimizer is promoted"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "certificate": certificate_audit,
        "artifact_audits": artifact_audits,
        "diagnostics": diagnostics,
        "summary": {
            "maximum_Au_region_complex_E_NRMSE": au_max,
            "maximum_TaIrTe4_region_complex_E_NRMSE": ta_max,
            "maximum_component_Q_relative_change": maximum_component_change,
            "fraction_of_all_absorption_carried_by_max_change_component": (
                maximum_component_share
            ),
            "interpretation_guard": (
                "localization or low power share does not waive any declared "
                "convergence gate"
            ),
        },
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "optimizer_start_allowed": False,
        "z32_start_allowed_by_this_diagnostic": False,
    }


def _atomic_json(path: Path, payload: dict) -> None:
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
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--certificate-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory.expanduser().resolve()
    if not args.output_directory.expanduser().is_absolute():
        parser.error("output directory must be absolute")
    if not output.is_dir() or any(output.iterdir()):
        parser.error("output directory must be existing and empty")
    try:
        payload = build_diagnostic(args.certificate, args.certificate_sha256)
    except Exception as error:
        payload = {
            "version": VERSION,
            "status": STATUS_BLOCKED,
            "ready": False,
            "error": repr(error),
            "optimizer_start_allowed": False,
            "z32_start_allowed_by_this_diagnostic": False,
        }
    _atomic_json(output / REPORT_NAME, payload)
    print(
        json.dumps(
            {
                "report": str(output / REPORT_NAME),
                "status": payload["status"],
                "ready": payload["ready"],
            }
        )
    )
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
