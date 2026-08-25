#!/usr/bin/env python3
"""Frozen-Q thermal/electrical diagnostic for one z2/z4 optical case.

The optical solve is never rerun here.  A byte-bound exact-binary Q artifact is
conservatively mapped to the same selected diagnostic thermal mesh, followed by
the same floating-Au electrical weighting solve.  This isolates the effect of
the FDTDX full-domain z refinement on downstream PTE observables.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any, Mapping

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_frozen_q_thermal_sensitivity_case import (
    audit_prior_domain_certificate,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_frozen_q_thermal_z_case import (
    ENERGY_BALANCE_LIMIT,
    MAPPING_RTOL,
    POWER_RTOL,
    THERMAL_RESIDUAL_LIMIT,
    FrozenGrid,
    _atomic_json,
    _atomic_npz,
    _environment_manifest,
    _git,
    _output_directory,
    _relative_error,
    material_slices,
    require_exclusive_physical_gpu,
    sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_exact_binary import (
    STATUS_READY as MATERIAL_STATUS_READY,
    VERSION as MATERIAL_VERSION,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_certificate import (
    STATUS_BLOCKED as OPTICAL_STATUS_BLOCKED,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_tail_certificate import (
    FACTORS,
    LEVELS,
    TAIL_VERSION,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    STEP_M,
    build_electrical_system,
    build_thermal_state,
    current_integrand,
    map_native_q_to_thermal,
    solve_electrical,
    solve_thermal,
    tairte4_temperature,
    thermal_edges,
)


VERSION = "fdtdx-user-balanced-pte-tail-case-v1"
STATUS_READY = "VALIDATED_DIAGNOSTIC_FDTDX_USER_BALANCED_PTE_TAIL_CASE"
STATUS_EXCEPTION = "BLOCKED_FDTDX_USER_BALANCED_PTE_TAIL_CASE_EXCEPTION"
REPORT_NAME = "FDTDX_USER_BALANCED_PTE_TAIL_CASE.json"
RAW_NAME = "FDTDX_USER_BALANCED_PTE_TAIL_CASE_FIELDS.npz"
POLARIZATIONS = ("Ea", "Eb")
THERMAL_XY_REFINEMENT_FACTOR = 2
THERMAL_Z_REFINEMENT_FACTOR = 2
THERMAL_DOMAIN = {
    "lateral_half_span_um": 48,
    "substrate_depth_um": 30,
    "top_air_height_um": 3.0,
}
EXPECTED_THERMAL_SHAPE = (548, 548, 72)
EXPECTED_SOLID_CELLS = 375
BASE_TA_CELLS = 160
BASE_THERMAL_CELLS = 266
ELECTRICAL_RESIDUAL_LIMIT = 2.0e-8
ELECTRICAL_BALANCE_LIMIT = 2.0e-8


def _all_true(values: Mapping[str, Any]) -> bool:
    return bool(values) and all(value is True for value in values.values())


def _is_hex_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _restrict_blocks(
    array: np.ndarray, factor: int, *, reduction: str
) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] % factor or value.shape[1] % factor:
        raise ValueError("refined xy array is incompatible with factor")
    blocked = value.reshape(
        value.shape[0] // factor,
        factor,
        value.shape[1] // factor,
        factor,
    )
    if reduction == "mean":
        return blocked.mean(axis=(1, 3))
    if reduction == "sum":
        return blocked.sum(axis=(1, 3))
    raise ValueError("block reduction must be mean or sum")


def _base_centers(refined_centers: np.ndarray, factor: int) -> np.ndarray:
    value = np.asarray(refined_centers, dtype=np.float64)
    if value.ndim != 1 or value.size % factor:
        raise ValueError("refined centers are incompatible with factor")
    return value.reshape(-1, factor).mean(axis=1)


def audit_optical_input(
    certificate_path: Path,
    expected_certificate_sha256: str,
    level: str,
    polarization: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    """Rebind the blocked tail certificate, report, and raw Q by bytes."""

    supplied = certificate_path.expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_file()
    actual_certificate_sha = sha256(resolved) if exists else None
    certificate = (
        json.loads(resolved.read_text(encoding="utf-8")) if exists else {}
    )
    case = certificate.get("material_cases", {}).get(level, {}).get(
        polarization, {}
    )
    report_record = case.get("report", {})
    report_supplied = Path(report_record.get("path", "")).expanduser()
    report_path = report_supplied.resolve()
    report_exists = report_path.is_file()
    actual_report_sha = sha256(report_path) if report_exists else None
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_exists
        else {}
    )
    raw_record = case.get("raw", {})
    raw_supplied = Path(raw_record.get("path", "")).expanduser()
    raw_path = raw_supplied.resolve()
    raw_exists = raw_path.is_file()
    actual_raw_sha = sha256(raw_path) if raw_exists else None
    exact_binary = report.get("material", {}).get("exact_binary_au", {})
    exact_checks = exact_binary.get("checks", {})
    normalization = report.get("normalization_policy", {})
    mesh = report.get("mesh", {})
    checks = {
        "certificate_path_is_absolute": supplied.is_absolute(),
        "certificate_exists": exists,
        "expected_certificate_sha256_is_hex": _is_hex_sha256(
            expected_certificate_sha256
        ),
        "certificate_sha256_matches": actual_certificate_sha
        == expected_certificate_sha256,
        "blocked_tail_certificate_valid": certificate.get("version")
        == TAIL_VERSION
        and certificate.get("status") == OPTICAL_STATUS_BLOCKED
        and certificate.get("certificate_valid") is True
        and certificate.get("convergence_pass") is False
        and certificate.get("mesh_selected") is None,
        "tail_artifact_checks_all_true": _all_true(
            certificate.get("artifact_checks", {})
        )
        and certificate.get("failed_artifact_checks") == [],
        "tail_optimizer_remains_forbidden": certificate.get(
            "optimizer_start_allowed"
        )
        is False,
        "certified_case_ready": case.get("ready") is True
        and _all_true(case.get("checks", {}))
        and case.get("failed_checks") == [],
        "report_path_is_absolute": report_supplied.is_absolute(),
        "report_exists": report_exists,
        "report_sha256_rebound_to_certificate": actual_report_sha
        == report_record.get("actual_sha256")
        == report_record.get("expected_sha256"),
        "raw_path_is_absolute": raw_supplied.is_absolute(),
        "raw_exists": raw_exists,
        "raw_sha256_rebound_to_certificate": actual_raw_sha
        == raw_record.get("actual_sha256")
        == raw_record.get("expected_sha256"),
        "report_version_status_ready": report.get("version") == MATERIAL_VERSION
        and report.get("status") == MATERIAL_STATUS_READY
        and report.get("ready") is True,
        "polarization_exact": report.get("polarization") == polarization,
        "full_domain_z_level_exact": mesh.get("axis") == "full_domain_z"
        and mesh.get("factor_from_user_baseline") == FACTORS[level]
        and mesh.get("grid_shape_xyz", [None, None, None])[2]
        == 150 * FACTORS[level],
        "exact_binary_no_gray": exact_binary.get("gray_density_allowed")
        is False
        and exact_checks.get("no_gray_material_law") is True
        and exact_checks.get("solver_mask_remains_binary") is True,
        "common_normalization_policy_exact": normalization.get(
            "raw_fields_and_Q_are_unscaled"
        )
        is True
        and normalization.get("per_polarization_power_matching_forbidden")
        is True
        and float(normalization.get("common_power_scale", 0.0)) > 0.0,
        "source_optimizer_was_forbidden": report.get("optimizer_start_allowed")
        is False,
    }
    audit = {
        "certificate_path": str(resolved),
        "expected_certificate_sha256": expected_certificate_sha256,
        "actual_certificate_sha256": actual_certificate_sha,
        "report_path": str(report_path),
        "actual_report_sha256": actual_report_sha,
        "raw_path": str(raw_path),
        "actual_raw_sha256": actual_raw_sha,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
    }
    if not audit["ready"]:
        raise RuntimeError(f"frozen optical input audit failed: {audit}")
    return certificate, report, raw_path, audit


def load_frozen_fields(
    raw_path: Path, report: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "design_mask",
        "solver_mask",
        "grid_x_edges_m",
        "grid_y_edges_m",
        "grid_z_edges_m",
        "q_au_late_W_m3",
        "q_tairte4_late_W_m3",
        "electric_dual_volume_au_m3",
        "electric_dual_volume_tairte4_m3",
    }
    with np.load(raw_path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"raw artifact is missing arrays: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    declared = report.get("raw", {}).get("arrays", {})
    mask = arrays["design_mask"]
    solver_mask = arrays["solver_mask"]
    q_fields = {
        "au": np.asarray(arrays["q_au_late_W_m3"], dtype=np.float64),
        "tairte4": np.asarray(
            arrays["q_tairte4_late_W_m3"], dtype=np.float64
        ),
    }
    volumes = {
        "au": np.asarray(
            arrays["electric_dual_volume_au_m3"], dtype=np.float64
        ),
        "tairte4": np.asarray(
            arrays["electric_dual_volume_tairte4_m3"], dtype=np.float64
        ),
    }
    edges = tuple(
        np.asarray(arrays[f"grid_{axis}_edges_m"], dtype=np.float64)
        for axis in "xyz"
    )
    checks = {
        "all_required_shapes_match_report_schema": all(
            list(arrays[name].shape) == declared.get(name) for name in required
        ),
        "design_mask_integer_binary": np.issubdtype(mask.dtype, np.integer)
        and set(np.unique(mask).tolist()) <= {0, 1},
        "solver_mask_equals_design_mask": np.array_equal(mask, solver_mask),
        "design_mask_shape_exact": mask.shape == CONTRACT.design_shape,
        "design_mask_solid_count_exact": int(np.count_nonzero(mask))
        == EXPECTED_SOLID_CELLS,
        "grid_edges_finite_strictly_increasing": all(
            edge.ndim == 1
            and edge.size >= 2
            and np.all(np.isfinite(edge))
            and np.all(np.diff(edge) > 0.0)
            for edge in edges
        ),
    }
    raw_power: dict[str, float] = {}
    for material in ("au", "tairte4"):
        q_value = q_fields[material]
        volume = volumes[material]
        checks[f"{material}_Q_volume_shape_equal"] = q_value.shape == volume.shape
        checks[f"{material}_Q_finite_nonnegative"] = bool(
            np.all(np.isfinite(q_value)) and np.all(q_value >= 0.0)
        )
        checks[f"{material}_dual_volume_finite_positive"] = bool(
            np.all(np.isfinite(volume)) and np.all(volume > 0.0)
        )
        raw_power[material] = float(np.sum(q_value * volume))
    checks["Au_Q_exactly_zero_outside_binary_mask"] = bool(
        np.all(q_fields["au"][:, mask == 0, :] == 0.0)
    )
    reported = report.get("evaluation", {}).get("Q", {}).get("late", {})
    for material in ("au", "tairte4"):
        checks[f"{material}_raw_power_matches_report"] = _relative_error(
            raw_power[material],
            float(
                reported.get("by_material", {})
                .get(material, {})
                .get("total_W", 0.0)
            ),
        ) <= POWER_RTOL
    total = sum(raw_power.values())
    checks["total_raw_power_matches_report"] = _relative_error(
        total, float(reported.get("total_W", 0.0))
    ) <= POWER_RTOL
    audit = {
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
        "raw_power_W": {**raw_power, "total": total},
    }
    if not audit["ready"]:
        raise RuntimeError(f"frozen raw array audit failed: {audit}")
    return {
        "mask": mask.astype(np.uint8, copy=False),
        "q_fields_W_m3": q_fields,
        "dual_volumes_m3": volumes,
        "grid": FrozenGrid(edges),
        "edges": edges,
        "audit": audit,
    }


def run(
    output_directory: Path,
    optical_certificate_path: Path,
    expected_optical_certificate_sha256: str,
    prior_thermal_domain_certificate_path: Path,
    expected_prior_thermal_domain_certificate_sha256: str,
    level: str,
    polarization: str,
    expected_physical_gpu: int,
) -> dict[str, Any]:
    started_total = time.perf_counter()
    output = _output_directory(output_directory)
    if level not in LEVELS:
        raise ValueError(f"optical z level must be one of {LEVELS}")
    if polarization not in POLARIZATIONS:
        raise ValueError(f"polarization must be one of {POLARIZATIONS}")
    gpu_before = require_exclusive_physical_gpu(
        expected_physical_gpu, allow_current_process=False
    )
    repository = Path(__file__).resolve().parents[3]
    dirty_before = _git(
        repository, "status", "--porcelain", "--untracked-files=all"
    )
    if dirty_before != "":
        raise RuntimeError("repository must be clean before diagnostic solve")

    _, prior_thermal_audit = audit_prior_domain_certificate(
        prior_thermal_domain_certificate_path,
        expected_prior_thermal_domain_certificate_sha256,
    )
    if not prior_thermal_audit["ready"]:
        raise RuntimeError(
            f"prior thermal-domain certificate failed: {prior_thermal_audit}"
        )
    _, report, raw_path, optical_input_audit = audit_optical_input(
        optical_certificate_path,
        expected_optical_certificate_sha256,
        level,
        polarization,
    )

    started_load = time.perf_counter()
    frozen = load_frozen_fields(raw_path, report)
    load_runtime_s = time.perf_counter() - started_load

    started_build = time.perf_counter()
    state = build_thermal_state(
        frozen["mask"].astype(np.float64),
        z_refinement_factor=THERMAL_Z_REFINEMENT_FACTOR,
        xy_refinement_factor=THERMAL_XY_REFINEMENT_FACTOR,
        **THERMAL_DOMAIN,
    )
    build_runtime_s = time.perf_counter() - started_build
    if state.system.shape != EXPECTED_THERMAL_SHAPE:
        raise RuntimeError(f"unexpected thermal shape: {state.system.shape}")

    started_remap = time.perf_counter()
    source_unscaled_W, mapping, _ = map_native_q_to_thermal(
        state,
        q_fields_W_m3=frozen["q_fields_W_m3"],
        dual_volumes_m3=frozen["dual_volumes_m3"],
        material_slices=material_slices(report["placement"]),
        realized_grid=frozen["grid"],
    )
    common_power_scale = float(
        report["normalization_policy"]["common_power_scale"]
    )
    source_power_W = source_unscaled_W * common_power_scale
    remap_runtime_s = time.perf_counter() - started_remap
    expected_unscaled_W = float(report["evaluation"]["Q"]["late"]["total_W"])
    expected_scaled_W = expected_unscaled_W * common_power_scale
    mapping_checks = {
        "both_material_maps_conservative": all(
            value["relative_error"] <= MAPPING_RTOL for value in mapping.values()
        ),
        "mapped_unscaled_total_matches_certified_Q": _relative_error(
            float(np.sum(source_unscaled_W)), expected_unscaled_W
        )
        <= POWER_RTOL,
        "common_scaled_total_recomputes": _relative_error(
            float(np.sum(source_power_W)), expected_scaled_W
        )
        <= POWER_RTOL,
        "mapped_source_finite_nonnegative": bool(
            np.all(np.isfinite(source_power_W))
            and np.all(source_power_W >= 0.0)
        ),
    }
    if not all(mapping_checks.values()):
        raise RuntimeError(f"thermal source mapping failed: {mapping_checks}")

    environment = _environment_manifest()
    if not (
        environment["cuda_available"]
        and environment["visible_cuda_device_count"] == 1
    ):
        raise RuntimeError("thermal environment must expose exactly one CUDA GPU")
    started_thermal = time.perf_counter()
    temperature_K, thermal_solver = solve_thermal(
        state, source_power_W, cuda_device=0
    )
    thermal_runtime_s = time.perf_counter() - started_thermal

    ta_native_K = tairte4_temperature(state, temperature_K)
    ta_base_K = _restrict_blocks(
        ta_native_K, THERMAL_XY_REFINEMENT_FACTOR, reduction="mean"
    )
    if ta_base_K.shape != (BASE_TA_CELLS, BASE_TA_CELLS):
        raise RuntimeError("restricted Ta temperature is not 160x160")
    ta_native_x = state.centers[0][
        (state.centers[0] >= -8e-6) & (state.centers[0] < 8e-6)
    ]
    ta_native_y = state.centers[1][
        (state.centers[1] >= -8e-6) & (state.centers[1] < 8e-6)
    ]
    ta_x = _base_centers(ta_native_x, THERMAL_XY_REFINEMENT_FACTOR)
    ta_y = _base_centers(ta_native_y, THERMAL_XY_REFINEMENT_FACTOR)
    gradient_x_K_m, gradient_y_K_m = np.gradient(
        ta_base_K, ta_x, ta_y, edge_order=2
    )

    started_electrical = time.perf_counter()
    electrical = build_electrical_system(
        frozen["mask"].astype(np.float64), ta_base_K
    )
    weighting, pte_current_A, electrical_solver = solve_electrical(
        electrical, cuda_device=0
    )
    electrical_runtime_s = time.perf_counter() - started_electrical
    current_density_A_m2 = current_integrand(ta_base_K, weighting)
    integrated_current_A = float(np.sum(current_density_A_m2) * STEP_M**2)

    source_native_xy_W = np.sum(source_power_W, axis=2)
    base_x_indices = np.flatnonzero(
        (state.centers[0] >= -32e-6) & (state.centers[0] < 32e-6)
    )
    base_y_indices = np.flatnonzero(
        (state.centers[1] >= -32e-6) & (state.centers[1] < 32e-6)
    )
    source_base_xy_W = _restrict_blocks(
        source_native_xy_W[np.ix_(base_x_indices, base_y_indices)],
        THERMAL_XY_REFINEMENT_FACTOR,
        reduction="sum",
    )
    if source_base_xy_W.shape != (BASE_THERMAL_CELLS, BASE_THERMAL_CELLS):
        raise RuntimeError("restricted thermal source is not 266x266")
    ta_weighting = weighting[: BASE_TA_CELLS * BASE_TA_CELLS].reshape(
        BASE_TA_CELLS, BASE_TA_CELLS
    )
    raw_arrays = {
        "ta_temperature_rise_K": ta_base_K,
        "ta_gradient_x_K_m": gradient_x_K_m,
        "ta_gradient_y_K_m": gradient_y_K_m,
        "ta_x_centers_m": ta_x,
        "ta_y_centers_m": ta_y,
        "source_power_xy_W": source_base_xy_W,
        "pte_current_density_A_m2": current_density_A_m2,
        "ta_electrical_weighting_V": ta_weighting,
    }
    raw_output = output / RAW_NAME
    _atomic_npz(raw_output, raw_arrays)

    matrix = state.system.matrix_W_K
    difference = matrix - matrix.T
    matrix_scale = max(float(np.max(np.abs(matrix.data))), np.finfo(float).tiny)
    asymmetry = (
        0.0
        if difference.nnz == 0
        else float(np.max(np.abs(difference.data))) / matrix_scale
    )
    expected_edges = thermal_edges(
        THERMAL_Z_REFINEMENT_FACTOR,
        xy_refinement_factor=THERMAL_XY_REFINEMENT_FACTOR,
        **THERMAL_DOMAIN,
    )
    mesh_checks = {
        "selected_thermal_edges_exact": all(
            np.array_equal(actual, expected)
            for actual, expected in zip(state.edges, expected_edges)
        ),
        "selected_thermal_shape_exact": state.system.shape
        == EXPECTED_THERMAL_SHAPE,
        "restricted_source_conserves_total": _relative_error(
            float(np.sum(source_base_xy_W)), float(np.sum(source_power_W))
        )
        <= 5.0e-14,
        "ta_coordinates_are_100nm": np.allclose(
            np.diff(ta_x), STEP_M, rtol=0.0, atol=2.0e-18
        )
        and np.allclose(np.diff(ta_y), STEP_M, rtol=0.0, atol=2.0e-18),
    }
    thermal_checks = {
        "matrix_symmetric_to_roundoff": asymmetry <= 1.0e-13,
        "temperature_finite": bool(np.all(np.isfinite(temperature_K))),
        "temperature_nonnegative_to_solver_tolerance": float(
            np.min(temperature_K)
        )
        >= -1.0e-8 * max(float(np.max(temperature_K)), 1.0),
        "relative_residual_within_limit": float(
            thermal_solver["relative_residual"]
        )
        <= THERMAL_RESIDUAL_LIMIT,
        "energy_balance_within_limit": float(
            thermal_solver["energy_balance_relative"]
        )
        <= ENERGY_BALANCE_LIMIT,
    }
    electrical_checks = {
        "weighting_and_current_finite": bool(
            np.all(np.isfinite(weighting)) and np.isfinite(pte_current_A)
        ),
        "relative_residual_within_limit": float(
            electrical_solver["relative_residual"]
        )
        <= ELECTRICAL_RESIDUAL_LIMIT,
        "explicit_free_residual_within_limit": float(
            electrical_solver["explicit_free_residual"]
        )
        <= ELECTRICAL_RESIDUAL_LIMIT,
        "terminal_balance_within_limit": float(
            electrical_solver["terminal_balance_relative"]
        )
        <= ELECTRICAL_BALANCE_LIMIT,
        "current_density_integrates_to_objective": _relative_error(
            integrated_current_A, pte_current_A
        )
        <= 5.0e-12,
    }
    dirty_after = _git(
        repository, "status", "--porcelain", "--untracked-files=all"
    )
    gpu_after = require_exclusive_physical_gpu(
        expected_physical_gpu, allow_current_process=True
    )
    provenance_checks = {
        "repository_clean_before_and_after": dirty_before == dirty_after == "",
        "blocked_optical_certificate_revalidated": optical_input_audit["ready"]
        is True,
        "thermal_domain_certificate_revalidated": prior_thermal_audit["ready"]
        is True,
        "raw_fields_revalidated": frozen["audit"]["ready"] is True,
        "mapping_checks_all_true": all(mapping_checks.values()),
        "mesh_checks_all_true": all(mesh_checks.values()),
        "thermal_checks_all_true": all(thermal_checks.values()),
        "electrical_checks_all_true": all(electrical_checks.values()),
        "one_exclusive_visible_gpu_before_and_after": gpu_before["exclusive"]
        is True
        and gpu_after["exclusive"] is True,
        "optimizer_not_run": True,
        "lumerical_not_used": True,
    }
    ready = all(provenance_checks.values())
    payload: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS_READY if ready else STATUS_EXCEPTION,
        "ready": ready,
        "scope": (
            "diagnostic-only frozen-Q thermal and floating-Au electrical solve "
            "for one byte-bound z2/z4 FDTDX case; no optical rerun, actual "
            "electrode validation, physical-parameter closure, adjoint, "
            "optimizer, or production promotion"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "optical_z_level": level,
        "optical_z_factor": FACTORS[level],
        "polarization": polarization,
        "optical_input_audit": optical_input_audit,
        "prior_thermal_domain_certificate": prior_thermal_audit,
        "raw_field_audit": frozen["audit"],
        "normalization": {
            "reporting_incident_power_W": CONTRACT.reporting_incident_power_W,
            "common_power_scale": common_power_scale,
            "mapped_scaled_absorbed_power_W": float(np.sum(source_power_W)),
            "per_polarization_power_matching_forbidden": True,
        },
        "mapping": mapping,
        "mapping_checks": mapping_checks,
        "thermal_mesh": {
            "shape": list(state.system.shape),
            "unknowns": int(matrix.shape[0]),
            "matrix_nonzeros": int(matrix.nnz),
            "matrix_relative_asymmetry": asymmetry,
            "xy_refinement_factor": THERMAL_XY_REFINEMENT_FACTOR,
            "z_refinement_factor": THERMAL_Z_REFINEMENT_FACTOR,
            "domain": THERMAL_DOMAIN,
        },
        "mesh_checks": mesh_checks,
        "thermal_solution": {
            "solver": thermal_solver,
            "global_max_temperature_rise_K": float(np.max(temperature_K)),
            "ta_max_temperature_rise_K": float(np.max(ta_base_K)),
            "ta_mean_temperature_rise_K": float(np.mean(ta_base_K)),
            "ta_gradient_combined_l2_K_m": float(
                np.sqrt(
                    np.sum(gradient_x_K_m**2) + np.sum(gradient_y_K_m**2)
                )
            ),
        },
        "thermal_checks": thermal_checks,
        "electrical_model": {
            "kind": "floating_Au_left_right_flake_edge_weighting_diagnostic",
            "actual_metal_electrodes_present": False,
            "electrical_mesh_pitch_m": STEP_M,
            "electrical_mesh_converged": False,
        },
        "pte_solution": {
            "signed_current_A": pte_current_A,
            "integrated_current_density_A": integrated_current_A,
            "solver": electrical_solver,
        },
        "electrical_checks": electrical_checks,
        "runtime": {
            "raw_load_and_audit_s": load_runtime_s,
            "thermal_assembly_s": build_runtime_s,
            "conservative_remap_s": remap_runtime_s,
            "thermal_cuda_pcg_s": thermal_runtime_s,
            "electrical_assembly_and_cuda_pcg_s": electrical_runtime_s,
            "total_s": time.perf_counter() - started_total,
        },
        "environment": environment,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "raw": {
            "path": str(raw_output),
            "sha256": sha256(raw_output),
            "arrays": {
                name: list(value.shape) for name, value in raw_arrays.items()
            },
        },
        "provenance": {
            "repository_commit": _git(repository, "rev-parse", "HEAD"),
            "repository_dirty_porcelain_before": dirty_before,
            "repository_dirty_porcelain_after": dirty_after,
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "lumerical_used": False,
        },
        "provenance_checks": provenance_checks,
        "diagnostic_only": True,
        "strict_optical_mesh_converged": False,
        "thermal_physical_parameters_converged": False,
        "electrical_mesh_converged": False,
        "actual_electrodes_validated": False,
        "production_mesh_selected": False,
        "optimizer_start_allowed": False,
    }
    _atomic_json(output / REPORT_NAME, payload)
    print(
        json.dumps(
            {
                "report": str(output / REPORT_NAME),
                "ready": ready,
                "optical_z_level": level,
                "polarization": polarization,
                "signed_current_A": pte_current_A,
                **payload["runtime"],
            }
        )
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--optical-tail-certificate", type=Path, required=True)
    parser.add_argument("--optical-tail-certificate-sha256", required=True)
    parser.add_argument(
        "--prior-thermal-domain-certificate", type=Path, required=True
    )
    parser.add_argument(
        "--prior-thermal-domain-certificate-sha256", required=True
    )
    parser.add_argument("--optical-z-level", choices=LEVELS, required=True)
    parser.add_argument("--polarization", choices=POLARIZATIONS, required=True)
    parser.add_argument("--expected-physical-gpu", type=int, required=True)
    args = parser.parse_args()
    output = args.output_directory.expanduser().resolve()
    try:
        payload = run(
            args.output_directory,
            args.optical_tail_certificate,
            args.optical_tail_certificate_sha256,
            args.prior_thermal_domain_certificate,
            args.prior_thermal_domain_certificate_sha256,
            args.optical_z_level,
            args.polarization,
            args.expected_physical_gpu,
        )
    except Exception as error:
        payload = {
            "version": VERSION,
            "status": STATUS_EXCEPTION,
            "ready": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "optical_z_level": args.optical_z_level,
            "polarization": args.polarization,
            "diagnostic_only": True,
            "strict_optical_mesh_converged": False,
            "thermal_physical_parameters_converged": False,
            "electrical_mesh_converged": False,
            "actual_electrodes_validated": False,
            "production_mesh_selected": False,
            "optimizer_start_allowed": False,
        }
        if output.is_dir() and not (output / REPORT_NAME).exists():
            _atomic_json(output / REPORT_NAME, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
