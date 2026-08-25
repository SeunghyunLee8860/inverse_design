#!/usr/bin/env python3
"""Fail-closed 100-nm to 50-nm lateral FDTDX comparison certificate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    OPTICAL_PAIR_GATES,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    TimeSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
    relative_difference,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_full_z_certificate import (
    MATERIAL_REGION_COMPLEX_E_NRMSE_LIMIT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_metrics import (
    weighted_complex_nrmse,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_exact_binary_control import (
    _json_default,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_exact_binary import (
    DEFAULT_REFERENCE,
    STATUS_READY as MATERIAL_STATUS_READY,
    VERSION as MATERIAL_VERSION,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_lateral_refinement import (
    case_contract as lateral_case_contract,
    mesh_audit as lateral_mesh_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_source_pair import (
    sha256,
    validate_source_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_certificate import (
    _file_audit,
    _load_raw_snapshot,
    _material_audit as z_material_audit,
    _runner_blob_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_refinement import (
    case_contract as z_case_contract,
    mesh_audit as z_mesh_audit,
)


VERSION = "fdtdx-user-balanced-lateral-certificate-v1"
STATUS_PAIR_PASS = "VALIDATED_FDTDX_USER_BALANCED_LATERAL_PAIR_PASS_DIAGNOSTIC"
STATUS_BLOCKED = "VALIDATED_BLOCKED_FDTDX_USER_BALANCED_LATERAL_CONVERGENCE"
STATUS_INVALID = "INVALID_FDTDX_USER_BALANCED_LATERAL_CERTIFICATE"
CERTIFICATE_NAME = "FDTDX_USER_BALANCED_XY100_TO_XY50_CERTIFICATE.json"
POLARIZATIONS = ("Ea", "Eb")
LEVELS = ("xy100", "xy50")
COMMON_REPORTING_POWER_W = 285.0e-6
SOURCE_POWER_RELATIVE_CHANGE_LIMIT = OPTICAL_PAIR_GATES[
    "source_power_relative_change"
]
MINIMUM_COMMON_XY_SUPPORT_FRACTION = 0.99
REMAP_CONSERVATION_RTOL = 5.0e-13
COORDINATE_ATOL_M = 2.0e-12


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _lateral_material_audit(
    report_path: Path,
    expected_sha256: str,
    polarization: str,
    source_pair_audit: dict[str, Any],
    repository: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    report = _file_audit(report_path, expected_sha256)
    payload = (
        json.loads(Path(report["path"]).read_text(encoding="utf-8"))
        if report["ready"]
        else {}
    )
    time = TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5)
    expected_case = lateral_case_contract(time)
    expected_mesh = lateral_mesh_audit()
    raw_record = payload.get("raw", {})
    raw = _file_audit(Path(raw_record.get("path", "")), raw_record.get("sha256", ""))
    snapshot, raw_snapshot = _load_raw_snapshot(payload, raw)
    gates = payload.get("evaluation", {}).get("gates", {})
    provenance_checks = payload.get("provenance_checks", {})
    material_checks = payload.get("material", {}).get("checks", {})
    runner_blob = _runner_blob_audit(payload, repository)
    recorded_pair = payload.get("source_pair", {})
    exact_binary = payload.get("material", {}).get("exact_binary_au", {})
    mesh_invariants = payload.get("mesh", {}).get("invariants", {})
    checks = {
        "report_ready": report["ready"],
        "version_status_ready": payload.get("version") == MATERIAL_VERSION
        and payload.get("status") == MATERIAL_STATUS_READY
        and payload.get("ready") is True,
        "labels_exact": payload.get("polarization") == polarization
        and payload.get("reference") == DEFAULT_REFERENCE
        and payload.get("full_domain_z_factor") == 2
        and payload.get("design_flake_xy_factor") == 2,
        "numerical_case_exact": payload.get("numerical_case_contract")
        == expected_case,
        "mesh_exact": payload.get("mesh") == expected_mesh,
        "mesh_invariants_all_true": bool(mesh_invariants)
        and all(value is True for value in mesh_invariants.values()),
        "source_pair_revalidation_ready": source_pair_audit["ready"],
        "recorded_source_pair_binding_exact": recorded_pair.get("path")
        == source_pair_audit["path"]
        and recorded_pair.get("actual_sha256")
        == source_pair_audit["actual_sha256"]
        and recorded_pair.get("ready") is True,
        "evaluation_gates_all_true": bool(gates)
        and all(value is True for value in gates.values())
        and payload.get("evaluation", {}).get("failed_gates") == [],
        "provenance_checks_all_true": bool(provenance_checks)
        and all(value is True for value in provenance_checks.values()),
        "material_checks_all_true": bool(material_checks)
        and all(value is True for value in material_checks.values()),
        "exact_binary_no_gray": exact_binary.get("gray_density_allowed") is False
        and exact_binary.get("rho_power") is None
        and exact_binary.get("design_xy_factor") == 2,
        "repository_clean_before_after": payload.get("provenance", {}).get(
            "repository_dirty_porcelain_before"
        )
        == payload.get("provenance", {}).get("repository_dirty_porcelain_after")
        == "",
        "historical_runner_blob_ready": runner_blob["ready"],
        "raw_snapshot_ready": raw_snapshot["ready"],
    }
    audit = {
        "report": report,
        "raw": raw,
        "raw_snapshot": raw_snapshot,
        "historical_runner_blob": runner_blob,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
    }
    return payload, audit, snapshot if audit["ready"] else None


def _edge_index(edges: np.ndarray, coordinate_m: float) -> int:
    matches = np.flatnonzero(
        np.isclose(edges, coordinate_m, rtol=0.0, atol=COORDINATE_ATOL_M)
    )
    if matches.size != 1:
        raise ValueError(f"required physical edge {coordinate_m:.9e} m is absent")
    return int(matches[0])


def _yee_coordinates(
    edges: np.ndarray,
    bounds: tuple[int, int],
    component: int,
    axis: int,
) -> np.ndarray:
    value = np.asarray(edges, dtype=np.float64)
    lower, upper = bounds
    indices = np.arange(lower, upper)
    coordinates = (
        0.5 * (value[indices] + value[indices + 1])
        if component == axis
        else value[indices]
    )
    if coordinates.size < 2 or np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("Yee coordinates must be strictly increasing")
    return coordinates


def _interpolate_axis(
    value: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    axis: int,
) -> np.ndarray:
    array = np.asarray(value)
    source_coordinate = np.asarray(source, dtype=np.float64)
    target_coordinate = np.asarray(target, dtype=np.float64)
    if array.shape[axis] != source_coordinate.size:
        raise ValueError("field shape does not match source coordinates")
    if (
        target_coordinate[0] < source_coordinate[0] - COORDINATE_ATOL_M
        or target_coordinate[-1] > source_coordinate[-1] + COORDINATE_ATOL_M
    ):
        raise ValueError("fine-to-coarse lateral interpolation would extrapolate")
    right = np.searchsorted(source_coordinate, target_coordinate, side="left")
    right = np.clip(right, 1, source_coordinate.size - 1)
    left = right - 1
    fraction = (target_coordinate - source_coordinate[left]) / (
        source_coordinate[right] - source_coordinate[left]
    )
    shape = [1] * array.ndim
    shape[axis] = target_coordinate.size
    fraction = fraction.reshape(shape)
    return np.take(array, left, axis=axis) * (1.0 - fraction) + np.take(
        array, right, axis=axis
    ) * fraction


def _interpolate_vector_fine_to_coarse_xy(
    fine_field: np.ndarray,
    fine_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    fine_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    coarse_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    coarse_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    value = np.asarray(fine_field)
    expected = (3, *(upper - lower for lower, upper in fine_bounds))
    if value.shape != expected:
        raise ValueError(f"fine field shape {value.shape} != placement {expected}")
    result = []
    for component in range(3):
        interpolated = _interpolate_axis(
            value[component],
            _yee_coordinates(fine_edges[0], fine_bounds[0], component, 0),
            _yee_coordinates(coarse_edges[0], coarse_bounds[0], component, 0),
            axis=0,
        )
        interpolated = _interpolate_axis(
            interpolated,
            _yee_coordinates(fine_edges[1], fine_bounds[1], component, 1),
            _yee_coordinates(coarse_edges[1], coarse_bounds[1], component, 1),
            axis=1,
        )
        result.append(interpolated)
    return np.stack(result)


def _control_intervals(
    edges: np.ndarray,
    bounds: tuple[int, int],
    component: int,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(edges, dtype=np.float64)
    lower, upper = bounds
    indices = np.arange(lower, upper)
    if component == axis:
        left, right = value[indices], value[indices + 1]
    else:
        if lower < 1 or upper >= value.size:
            raise ValueError("edge-centered Yee controls require neighboring cells")
        left = 0.5 * (value[indices - 1] + value[indices])
        right = 0.5 * (value[indices] + value[indices + 1])
    if np.any(right <= left):
        raise ValueError("Yee control intervals must have positive width")
    return left, right


def _overlap_matrix(
    coarse_left: np.ndarray,
    coarse_right: np.ndarray,
    fine_left: np.ndarray,
    fine_right: np.ndarray,
) -> np.ndarray:
    return np.maximum(
        0.0,
        np.minimum(coarse_right[:, None], fine_right[None, :])
        - np.maximum(coarse_left[:, None], fine_left[None, :]),
    )


def _restrict_component_q_to_coarse_xy(
    coarse_q: np.ndarray,
    fine_q: np.ndarray,
    coarse_volume: np.ndarray,
    fine_volume: np.ndarray,
    coarse_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    fine_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    coarse_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    fine_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    component: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    coarse_value = np.asarray(coarse_q, dtype=np.float64)
    fine_value = np.asarray(fine_q, dtype=np.float64)
    coarse_weight = np.asarray(coarse_volume, dtype=np.float64)
    fine_weight = np.asarray(fine_volume, dtype=np.float64)
    if coarse_value.shape != coarse_weight.shape or fine_value.shape != fine_weight.shape:
        raise ValueError("Q and dual-volume shapes must match")
    if coarse_value.shape[2] != fine_value.shape[2]:
        raise ValueError("lateral comparison requires an identical z grid")

    controls = []
    for axis in (0, 1):
        coarse_left, coarse_right = _control_intervals(
            coarse_edges[axis], coarse_bounds[axis], component, axis
        )
        fine_left, fine_right = _control_intervals(
            fine_edges[axis], fine_bounds[axis], component, axis
        )
        common_lower = max(float(coarse_left[0]), float(fine_left[0]))
        common_upper = min(float(coarse_right[-1]), float(fine_right[-1]))
        if common_upper <= common_lower:
            raise ValueError("coarse and fine controls have no common support")
        clipped_left = np.maximum(coarse_left, common_lower)
        clipped_right = np.minimum(coarse_right, common_upper)
        common_width = np.maximum(clipped_right - clipped_left, 0.0)
        if np.any(common_width <= 0.0):
            raise ValueError("each coarse control must retain common support")
        overlap = _overlap_matrix(
            clipped_left, clipped_right, fine_left, fine_right
        )
        controls.append(
            {
                "coarse_native_width": coarse_right - coarse_left,
                "fine_native_width": fine_right - fine_left,
                "common_width": common_width,
                "overlap": overlap,
                "fine_coverage": np.sum(overlap, axis=0),
                "row_sums_match": np.allclose(
                    np.sum(overlap, axis=1),
                    common_width,
                    rtol=1.0e-12,
                    atol=2.0e-18,
                ),
                "support_fraction": min(
                    (common_upper - common_lower)
                    / (float(coarse_right[-1]) - float(coarse_left[0])),
                    (common_upper - common_lower)
                    / (float(fine_right[-1]) - float(fine_left[0])),
                ),
                "common_bounds_m": [common_lower, common_upper],
            }
        )

    x_control, y_control = controls
    fine_on_coarse = np.einsum(
        "if,jg,fgz->ijz",
        x_control["overlap"],
        y_control["overlap"],
        fine_value,
        optimize=True,
    ) / (
        x_control["common_width"][:, None, None]
        * y_control["common_width"][None, :, None]
    )
    coarse_z_measure = coarse_weight / (
        x_control["coarse_native_width"][:, None, None]
        * y_control["coarse_native_width"][None, :, None]
    )
    fine_z_measure = fine_weight / (
        x_control["fine_native_width"][:, None, None]
        * y_control["fine_native_width"][None, :, None]
    )
    coarse_z_invariant = np.allclose(
        coarse_z_measure, coarse_z_measure[:1, :1, :], rtol=1.0e-12, atol=0.0
    )
    fine_z_invariant = np.allclose(
        fine_z_measure, fine_z_measure[:1, :1, :], rtol=1.0e-12, atol=0.0
    )
    z_measures_match = np.allclose(
        coarse_z_measure[:1, :1, :],
        fine_z_measure[:1, :1, :],
        rtol=1.0e-12,
        atol=0.0,
    )
    common_weight = (
        coarse_z_measure[:1, :1, :]
        * x_control["common_width"][:, None, None]
        * y_control["common_width"][None, :, None]
    )
    mapped_power = float(np.sum(fine_on_coarse * common_weight))
    direct_fine_common_power = float(
        np.sum(
            fine_value
            * fine_z_measure[:1, :1, :]
            * x_control["fine_coverage"][:, None, None]
            * y_control["fine_coverage"][None, :, None]
        )
    )
    conservation_error = abs(mapped_power - direct_fine_common_power) / max(
        abs(direct_fine_common_power), np.finfo(float).tiny
    )
    minimum_support = min(item["support_fraction"] for item in controls)
    checks = {
        "coarse_z_measure_is_xy_invariant": bool(coarse_z_invariant),
        "fine_z_measure_is_xy_invariant": bool(fine_z_invariant),
        "coarse_and_fine_z_measures_match": bool(z_measures_match),
        "overlap_rows_cover_common_coarse_controls": all(
            bool(item["row_sums_match"]) for item in controls
        ),
        "common_xy_support_fraction_sufficient": minimum_support
        >= MINIMUM_COMMON_XY_SUPPORT_FRACTION,
        "fine_restriction_conserves_common_support_power": conservation_error
        <= REMAP_CONSERVATION_RTOL,
    }
    return coarse_value, fine_on_coarse, common_weight, {
        "common_x_bounds_m": x_control["common_bounds_m"],
        "common_y_bounds_m": y_control["common_bounds_m"],
        "minimum_common_xy_support_fraction": minimum_support,
        "required_common_xy_support_fraction": MINIMUM_COMMON_XY_SUPPORT_FRACTION,
        "fine_restriction_relative_power_error": conservation_error,
        "checks": checks,
        "ready": all(checks.values()),
    }


def _material_field_comparison(
    coarse: Mapping[str, Any],
    fine: Mapping[str, Any],
    coarse_payload: Mapping[str, Any],
    fine_payload: Mapping[str, Any],
    coarse_field_scale: float,
    fine_field_scale: float,
) -> tuple[dict[str, float], dict[str, bool]]:
    placement_key = {"au": "au_design", "tairte4": "fixed_tairte4"}
    result: dict[str, float] = {}
    checks: dict[str, bool] = {}
    for material, key in placement_key.items():
        coarse_bounds = tuple(tuple(axis) for axis in coarse_payload["placement"][key])
        fine_bounds = tuple(tuple(axis) for axis in fine_payload["placement"][key])
        coarse_field = np.asarray(coarse["fields_late"][material]) * coarse_field_scale
        fine_on_coarse = _interpolate_vector_fine_to_coarse_xy(
            np.asarray(fine["fields_late"][material]) * fine_field_scale,
            fine["grid_edges"],
            fine_bounds,
            coarse["grid_edges"],
            coarse_bounds,
        )
        weights = np.asarray(coarse["volumes"][material], dtype=np.float64)
        if material == "au":
            coarse_mask = np.asarray(coarse["solver_mask"])
            fine_mask = np.asarray(fine["solver_mask"])
            replicated = np.repeat(np.repeat(coarse_mask, 2, axis=0), 2, axis=1)
            checks["Au_solver_mask_exact_2x2_replication"] = np.array_equal(
                fine_mask, replicated
            )
            checks["Au_solver_masks_binary"] = bool(
                np.all((coarse_mask == 0) | (coarse_mask == 1))
                and np.all((fine_mask == 0) | (fine_mask == 1))
            )
            weights = weights * coarse_mask[None, :, :, None]
        result[material] = weighted_complex_nrmse(
            fine_on_coarse, coarse_field, weights
        )
    return result, checks


def _conservative_q_comparison(
    coarse: Mapping[str, Any],
    fine: Mapping[str, Any],
    coarse_payload: Mapping[str, Any],
    fine_payload: Mapping[str, Any],
    coarse_q_scale: float,
    fine_q_scale: float,
) -> dict[str, Any]:
    placement_key = {"au": "au_design", "tairte4": "fixed_tairte4"}
    numerator = 0.0
    denominator = 0.0
    details: dict[str, Any] = {}
    audits = []
    for material, key in placement_key.items():
        details[material] = {}
        coarse_bounds = tuple(tuple(axis) for axis in coarse_payload["placement"][key])
        fine_bounds = tuple(tuple(axis) for axis in fine_payload["placement"][key])
        for component, axis in enumerate("xyz"):
            coarse_q, fine_q, weights, audit = _restrict_component_q_to_coarse_xy(
                np.asarray(coarse["q_late"][material][component]) * coarse_q_scale,
                np.asarray(fine["q_late"][material][component]) * fine_q_scale,
                coarse["volumes"][material][component],
                fine["volumes"][material][component],
                coarse["grid_edges"],
                fine["grid_edges"],
                coarse_bounds,
                fine_bounds,
                component,
            )
            component_numerator = float(np.sum((fine_q - coarse_q) ** 2 * weights))
            component_denominator = float(np.sum(fine_q**2 * weights))
            numerator += component_numerator
            denominator += component_denominator
            details[material][axis] = {
                **audit,
                "Q_volume_NRMSE": math.sqrt(
                    component_numerator
                    / max(component_denominator, np.finfo(float).tiny)
                ),
            }
            audits.append(audit)
    checks = {
        "all_component_restrictions_ready": bool(audits)
        and all(item["ready"] is True for item in audits),
        "all_component_checks_pass": bool(audits)
        and all(all(item["checks"].values()) for item in audits),
    }
    return {
        "combined_Q_volume_L2_NRMSE": math.sqrt(
            numerator / max(denominator, np.finfo(float).tiny)
        ),
        "minimum_common_xy_support_fraction": min(
            item["minimum_common_xy_support_fraction"] for item in audits
        ),
        "maximum_fine_restriction_relative_power_error": max(
            item["fine_restriction_relative_power_error"] for item in audits
        ),
        "details": details,
        "checks": checks,
        "ready": all(checks.values()),
    }


def _incident_power(payload: Mapping[str, Any]) -> float:
    value = float(
        payload["evaluation"]["flux"]["source_reference_all_air_unscaled_W"]
    )
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("source reference power must be finite and positive")
    return value


def compare_lateral_pair(
    snapshots: Mapping[str, Mapping[str, Mapping[str, Any] | None]],
    payloads: Mapping[str, Mapping[str, Mapping[str, Any]]],
    source_pairs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if any(
        snapshots[level][polarization] is None
        for level in LEVELS
        for polarization in POLARIZATIONS
    ):
        return {"pass": False, "error": "one or more raw snapshots are invalid", "checks": {}}
    source_power = {
        level: {
            polarization: float(
                source_pairs[level]["comparison"]["unscaled_incident_power_W"][
                    polarization
                ]
            )
            for polarization in POLARIZATIONS
        }
        for level in LEVELS
    }
    source_change = {
        polarization: relative_difference(
            source_power["xy100"][polarization],
            source_power["xy50"][polarization],
        )
        for polarization in POLARIZATIONS
    }
    source_change["mean"] = relative_difference(
        source_pairs["xy100"]["comparison"]["mean_unscaled_incident_power_W"],
        source_pairs["xy50"]["comparison"]["mean_unscaled_incident_power_W"],
    )
    observed_ratio = (
        source_pairs["xy50"]["comparison"]["mean_unscaled_incident_power_W"]
        / source_pairs["xy100"]["comparison"]["mean_unscaled_incident_power_W"]
    )

    per_polarization: dict[str, Any] = {}
    structural_checks: dict[str, bool] = {}
    for polarization in POLARIZATIONS:
        coarse = snapshots["xy100"][polarization]
        fine = snapshots["xy50"][polarization]
        coarse_payload = payloads["xy100"][polarization]
        fine_payload = payloads["xy50"][polarization]
        coarse_power = _incident_power(coarse_payload)
        fine_power = _incident_power(fine_payload)
        coarse_field_scale = math.sqrt(COMMON_REPORTING_POWER_W / coarse_power)
        fine_field_scale = math.sqrt(COMMON_REPORTING_POWER_W / fine_power)
        coarse_q_scale = COMMON_REPORTING_POWER_W / coarse_power
        fine_q_scale = COMMON_REPORTING_POWER_W / fine_power

        coarse_edges = coarse["grid_edges"]
        fine_edges = fine["grid_edges"]
        coarse_probe_bounds = (
            (_edge_index(coarse_edges[0], -4.0e-6), _edge_index(coarse_edges[0], 4.0e-6)),
            (_edge_index(coarse_edges[1], -4.0e-6), _edge_index(coarse_edges[1], 4.0e-6)),
            tuple(coarse_payload["placement"]["target_field"][2]),
        )
        fine_probe_bounds = (
            (_edge_index(fine_edges[0], -4.0e-6), _edge_index(fine_edges[0], 4.0e-6)),
            (_edge_index(fine_edges[1], -4.0e-6), _edge_index(fine_edges[1], 4.0e-6)),
            tuple(fine_payload["placement"]["target_field"][2]),
        )
        fine_probe_on_coarse = _interpolate_vector_fine_to_coarse_xy(
            np.asarray(fine["probe"]) * fine_field_scale,
            fine_edges,
            fine_probe_bounds,
            coarse_edges,
            coarse_probe_bounds,
        )
        probe_nrmse = weighted_complex_nrmse(
            fine_probe_on_coarse[:2],
            np.asarray(coarse["probe"][:2]) * coarse_field_scale,
            coarse["probe_weights"][:2],
        )
        material_field, field_checks = _material_field_comparison(
            coarse,
            fine,
            coarse_payload,
            fine_payload,
            coarse_field_scale,
            fine_field_scale,
        )
        structural_checks.update(
            {f"{polarization}_{name}": value for name, value in field_checks.items()}
        )
        q_comparison = _conservative_q_comparison(
            coarse,
            fine,
            coarse_payload,
            fine_payload,
            coarse_q_scale,
            fine_q_scale,
        )
        structural_checks[f"{polarization}_conservative_Q_restriction_ready"] = (
            q_comparison["ready"] is True
        )
        material_component_q: dict[str, Any] = {}
        component_changes = []
        normalized_power: dict[str, Any] = {"xy100": {}, "xy50": {}}
        for material in ("au", "tairte4"):
            normalized_power["xy100"][material] = {
                "component_W": {
                    axis: coarse_payload["evaluation"]["Q"]["late"]["by_material"][
                        material
                    ]["component_W"][axis]
                    * coarse_q_scale
                    for axis in "xyz"
                },
                "total_W": coarse_payload["evaluation"]["Q"]["late"][
                    "by_material"
                ][material]["total_W"]
                * coarse_q_scale,
            }
            normalized_power["xy50"][material] = {
                "component_W": {
                    axis: fine_payload["evaluation"]["Q"]["late"]["by_material"][
                        material
                    ]["component_W"][axis]
                    * fine_q_scale
                    for axis in "xyz"
                },
                "total_W": fine_payload["evaluation"]["Q"]["late"][
                    "by_material"
                ][material]["total_W"]
                * fine_q_scale,
            }
            material_component_q[material] = {
                axis: relative_difference(
                    normalized_power["xy100"][material]["component_W"][axis],
                    normalized_power["xy50"][material]["component_W"][axis],
                )
                for axis in "xyz"
            }
            material_component_q[material]["total"] = relative_difference(
                normalized_power["xy100"][material]["total_W"],
                normalized_power["xy50"][material]["total_W"],
            )
            component_changes.extend(material_component_q[material].values())
        normalized_total = {
            "xy100": coarse_payload["evaluation"]["Q"]["late"]["total_W"]
            * coarse_q_scale,
            "xy50": fine_payload["evaluation"]["Q"]["late"]["total_W"]
            * fine_q_scale,
        }
        per_polarization[polarization] = {
            "incident_power_W": {"xy100": coarse_power, "xy50": fine_power},
            "common_285uW_field_scales": {
                "xy100": coarse_field_scale,
                "xy50": fine_field_scale,
            },
            "common_285uW_Q_scales": {
                "xy100": coarse_q_scale,
                "xy50": fine_q_scale,
            },
            "common_285uW_total_Q_W": normalized_total,
            "common_285uW_material_Q_W": normalized_power,
            "total_Q_relative_change": relative_difference(
                normalized_total["xy100"], normalized_total["xy50"]
            ),
            "material_component_Q_relative_change": material_component_q,
            "material_component_Q_max_relative_change": max(component_changes),
            "tangential_complex_E_fixed_probe_NRMSE": probe_nrmse,
            "material_region_complex_E_NRMSE_after_fine_to_coarse_xy_interpolation": material_field,
            "material_region_complex_E_max_NRMSE": max(material_field.values()),
            "conservative_Q": q_comparison,
        }

    closure_values = [
        float(payloads[level][polarization]["evaluation"]["flux"][name])
        for level in LEVELS
        for polarization in POLARIZATIONS
        for name in (
            "Q_vs_closed_phasor_symmetric_relative",
            "Q_vs_closed_td_symmetric_relative",
        )
    ]
    metrics = {
        "raw_source_power_relative_change": max(source_change.values()),
        "q_closed_flux_relative": max(closure_values),
        "stationarity_complex_E_NRMSE": max(
            float(
                payloads["xy50"][polarization]["evaluation"]["field_stationarity"][
                    "maximum_complex_E_NRMSE"
                ]
            )
            for polarization in POLARIZATIONS
        ),
        "common_power_total_Q_relative_change": max(
            item["total_Q_relative_change"] for item in per_polarization.values()
        ),
        "common_power_material_component_Q_max_relative_change": max(
            item["material_component_Q_max_relative_change"]
            for item in per_polarization.values()
        ),
        "common_power_complex_E_fixed_probe_NRMSE": max(
            item["tangential_complex_E_fixed_probe_NRMSE"]
            for item in per_polarization.values()
        ),
        "common_power_conservative_Q_volume_L2_NRMSE": max(
            item["conservative_Q"]["combined_Q_volume_L2_NRMSE"]
            for item in per_polarization.values()
        ),
        "common_power_material_region_complex_E_max_NRMSE": max(
            item["material_region_complex_E_max_NRMSE"]
            for item in per_polarization.values()
        ),
    }
    limits = {
        "raw_source_power_relative_change": SOURCE_POWER_RELATIVE_CHANGE_LIMIT,
        "q_closed_flux_relative": OPTICAL_PAIR_GATES["q_closed_flux_relative"],
        "stationarity_complex_E_NRMSE": OPTICAL_PAIR_GATES[
            "stationarity_complex_E_NRMSE"
        ],
        "common_power_total_Q_relative_change": OPTICAL_PAIR_GATES[
            "total_Q_relative_change"
        ],
        "common_power_material_component_Q_max_relative_change": OPTICAL_PAIR_GATES[
            "material_component_Q_max_relative_change"
        ],
        "common_power_complex_E_fixed_probe_NRMSE": OPTICAL_PAIR_GATES[
            "complex_E_fixed_probe_NRMSE"
        ],
        "common_power_conservative_Q_volume_L2_NRMSE": OPTICAL_PAIR_GATES[
            "conservative_Q_volume_L2_NRMSE"
        ],
        "common_power_material_region_complex_E_max_NRMSE": (
            MATERIAL_REGION_COMPLEX_E_NRMSE_LIMIT
        ),
    }
    metric_checks = {name: metrics[name] <= limit for name, limit in limits.items()}
    checks = {**metric_checks, **structural_checks}
    return {
        "coarse_level": "xy100",
        "fine_level": "xy50",
        "same_z_and_physical_geometry_only_design_complete_flake_xy_refined": True,
        "normalization_method": (
            "each mesh is scaled once by its two-polarization mean all-air source "
            "calibration to the same 285 uW; no per-polarization scaling"
        ),
        "field_method": (
            "common-power complex fine field linearly interpolated at physical "
            "component-specific coarse Yee x/y coordinates; extrapolation forbidden"
        ),
        "Q_method": (
            "common-power component Q density restricted through exact physical "
            "x/y Yee control-volume overlaps; conservation checked independently"
        ),
        "source_power_relative_change": source_change,
        "source_cell_area_scaling_diagnostic": {
            "observed_fine_to_coarse_power_ratio": observed_ratio,
            "expected_ratio_if_power_scales_with_cell_area": 0.25,
            "relative_error_from_cell_area_ratio": abs(observed_ratio - 0.25) / 0.25,
            "interpretation": (
                "diagnostic evidence that the unscaled FDTDX plane-source amplitude "
                "convention changes with lateral sampling; this is not a physics "
                "convergence pass"
            ),
        },
        "per_polarization": per_polarization,
        "metrics": metrics,
        "limits": limits,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "pass": all(checks.values()),
    }


def build_certificate(
    source_pair_paths: Mapping[str, Path],
    source_pair_hashes: Mapping[str, str],
    report_paths: Mapping[str, Mapping[str, Path]],
    report_hashes: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    time = TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5)
    expected_source = {
        "xy100": (z_case_contract(time, 2), z_mesh_audit(2)),
        "xy50": (lateral_case_contract(time), lateral_mesh_audit()),
    }
    source_pairs: dict[str, Any] = {}
    source_audits: dict[str, Any] = {}
    for level in LEVELS:
        expected_case, expected_mesh = expected_source[level]
        source_pairs[level], source_audits[level] = validate_source_pair(
            source_pair_paths[level],
            source_pair_hashes[level],
            time,
            expected_case_contract=expected_case,
            expected_mesh=expected_mesh,
        )

    payloads: dict[str, dict[str, Any]] = {level: {} for level in LEVELS}
    case_audits: dict[str, dict[str, Any]] = {level: {} for level in LEVELS}
    snapshots: dict[str, dict[str, Any]] = {level: {} for level in LEVELS}
    for polarization in POLARIZATIONS:
        payload, audit, snapshot = z_material_audit(
            report_paths["xy100"][polarization],
            report_hashes["xy100"][polarization],
            "z2",
            polarization,
            source_audits["xy100"],
            repository,
        )
        payloads["xy100"][polarization] = payload
        case_audits["xy100"][polarization] = audit
        snapshots["xy100"][polarization] = snapshot
        payload, audit, snapshot = _lateral_material_audit(
            report_paths["xy50"][polarization],
            report_hashes["xy50"][polarization],
            polarization,
            source_audits["xy50"],
            repository,
        )
        payloads["xy50"][polarization] = payload
        case_audits["xy50"][polarization] = audit
        snapshots["xy50"][polarization] = snapshot

    binary_checks: dict[str, bool] = {}
    for polarization in POLARIZATIONS:
        coarse = snapshots["xy100"][polarization]
        fine = snapshots["xy50"][polarization]
        if coarse is None or fine is None:
            binary_checks[f"{polarization}_mask_contract"] = False
            binary_checks[f"{polarization}_z_edges_byte_exact"] = False
        else:
            binary_checks[f"{polarization}_design_mask_identical"] = np.array_equal(
                coarse["design_mask"], fine["design_mask"]
            )
            binary_checks[f"{polarization}_solver_mask_exact_2x2"] = np.array_equal(
                fine["solver_mask"],
                np.repeat(np.repeat(coarse["solver_mask"], 2, axis=0), 2, axis=1),
            )
            binary_checks[f"{polarization}_z_edges_byte_exact"] = np.array_equal(
                coarse["grid_edges"][2], fine["grid_edges"][2]
            )
    artifact_checks = {
        "repository_clean_while_certifying": _git(
            repository, "status", "--porcelain", "--untracked-files=all"
        )
        == "",
        "both_source_pairs_revalidate": all(
            audit["ready"] for audit in source_audits.values()
        ),
        "all_four_material_cases_revalidate": all(
            case_audits[level][polarization]["ready"]
            for level in LEVELS
            for polarization in POLARIZATIONS
        ),
        **binary_checks,
    }
    comparison = (
        compare_lateral_pair(snapshots, payloads, source_pairs)
        if all(artifact_checks.values())
        else None
    )
    certificate_valid = all(artifact_checks.values()) and comparison is not None
    pair_pass = certificate_valid and comparison["pass"] is True
    status = (
        STATUS_PAIR_PASS if pair_pass else STATUS_BLOCKED if certificate_valid else STATUS_INVALID
    )
    generator = Path(__file__).resolve()
    return {
        "version": VERSION,
        "status": status,
        "certificate_valid": certificate_valid,
        "lateral_pair_pass": pair_pass,
        "production_mesh_selected": False,
        "mesh_selected": None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_checks": artifact_checks,
        "failed_artifact_checks": [
            name for name, passed in artifact_checks.items() if not passed
        ],
        "source_pairs": source_audits,
        "material_cases": case_audits,
        "comparison": comparison,
        "promotion_rules": {
            "one_100nm_to_50nm_pair_selects_production_mesh": False,
            "50nm_to_25nm_confirmation_pair_required": True,
            "source_discretization_blocker_must_be_closed": True,
            "other_independent_mesh_axes_remain_unselected": True,
            "optimizer_start_allowed": False,
        },
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


def main() -> int:
    parser = argparse.ArgumentParser()
    for level in LEVELS:
        parser.add_argument(f"--{level}-source-pair", type=Path, required=True)
        parser.add_argument(f"--{level}-source-pair-sha256", required=True)
        for polarization in POLARIZATIONS:
            parser.add_argument(
                f"--{level}-{polarization.lower()}-report", type=Path, required=True
            )
            parser.add_argument(
                f"--{level}-{polarization.lower()}-report-sha256", required=True
            )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser()
    if not output.is_absolute() or not output.parent.is_dir() or output.exists():
        parser.error("--output must be a new absolute file under an existing directory")
    source_paths = {level: getattr(args, f"{level}_source_pair") for level in LEVELS}
    source_hashes = {
        level: getattr(args, f"{level}_source_pair_sha256") for level in LEVELS
    }
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
        source_paths, source_hashes, report_paths, report_hashes
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
                "status": payload["status"],
                "certificate_valid": payload["certificate_valid"],
                "lateral_pair_pass": payload["lateral_pair_pass"],
                "failed_artifact_checks": payload["failed_artifact_checks"],
                "failed_comparison_checks": (
                    payload["comparison"]["failed_checks"]
                    if payload["comparison"] is not None
                    else None
                ),
            },
            default=_json_default,
        )
    )
    return 0 if payload["certificate_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
