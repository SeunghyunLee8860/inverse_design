"""Fail-closed certificate for the 24-period FDTDX full-domain-z ladder."""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any, Mapping

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    C0_M_PER_S,
    OPTICAL_PAIR_GATES,
    MeshSpec,
    axis_levels,
    evaluate_pair,
    grid_edges as contract_grid_edges,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_anchor_placement import (
    expected_placement,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    FreshCaseSpec,
    TimeSpec,
    case_contract,
    load_case_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
    relative_difference,
    validate_source_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_metrics import (
    weighted_complex_nrmse,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_time_settling_certificate import (
    POLARIZATIONS,
    REFERENCE_NAME,
    _all_true,
    _material_case_audit,
    sha256,
)


CERTIFICATE_NAME = "FDTDX_FRESH_FULL_Z_CERTIFICATE.json"
STATUS_READY = "VALIDATED_FDTDX_FRESH_L500_FULL_DOMAIN_Z_CONVERGENCE"
STATUS_BLOCKED = "BLOCKED_FDTDX_FRESH_L500_FULL_DOMAIN_Z_CONVERGENCE"
LEVELS = ("z2", "z4", "z8")
Z_FACTOR = {"z2": 2, "z4": 4, "z8": 8}
SUCCESSIVE_PAIRS = (("z2", "z4"), ("z4", "z8"))
SELECTED_LEVEL = "z4"
CONFIRMATION_LEVEL = "z8"
TOTAL_PERIODS = 24
WINDOW_PERIODS = 4
COURANT_FACTOR = 0.25
CFL_DT_RTOL = 5.0e-7
MINIMUM_COMMON_Z_SUPPORT_FRACTION = 0.90
COMMON_Z_SUPPORT_NUMERICAL_ATOL = 5.0e-7
REMAP_CONSERVATION_RTOL = 5.0e-13
MATERIAL_REGION_COMPLEX_E_NRMSE_LIMIT = 5.0e-2


def expected_full_z_case(level: str) -> FreshCaseSpec:
    if level not in LEVELS:
        raise ValueError(f"full-z level must be one of {LEVELS}")
    meshes = axis_levels("full_domain_z", MeshSpec())
    mesh = meshes[LEVELS.index(level)]
    if mesh.z_factor != Z_FACTOR[level]:
        raise RuntimeError("full-z ladder and level labels disagree")
    return FreshCaseSpec(
        mesh=mesh,
        time=TimeSpec(
            total_periods=TOTAL_PERIODS,
            window_periods=WINDOW_PERIODS,
            courant_factor=COURANT_FACTOR,
        ),
    )


def _z_coordinates(
    edges: np.ndarray, bounds: tuple[int, int], component: int
) -> np.ndarray:
    value = np.asarray(edges, dtype=np.float64)
    lower, upper = bounds
    indices = np.arange(lower, upper)
    if component == 2:
        coordinates = 0.5 * (value[indices] + value[indices + 1])
    else:
        coordinates = value[indices]
    if coordinates.size < 2 or np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("material Yee z coordinates must be strictly increasing")
    return coordinates


def _interpolate_fine_to_coarse_z(
    fine: np.ndarray, fine_z: np.ndarray, coarse_z: np.ndarray
) -> np.ndarray:
    source = np.asarray(fine_z, dtype=np.float64)
    target = np.asarray(coarse_z, dtype=np.float64)
    value = np.asarray(fine)
    tolerance = 2.0e-15
    if value.shape[-1] != source.size:
        raise ValueError("fine field z shape does not match its Yee coordinates")
    if target[0] < source[0] - tolerance or target[-1] > source[-1] + tolerance:
        raise ValueError("fine-to-coarse field interpolation would extrapolate")
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 1, source.size - 1)
    left = right - 1
    fraction = (target - source[left]) / (source[right] - source[left])
    return (
        value[..., left] * (1.0 - fraction)
        + value[..., right] * fraction
    )


def _material_field_comparison(
    coarse: Mapping[str, Any],
    fine: Mapping[str, Any],
    coarse_payload: Mapping[str, Any],
    fine_payload: Mapping[str, Any],
) -> dict[str, float]:
    placement_key = {"au": "au_design", "tairte4": "fixed_tairte4"}
    result: dict[str, float] = {}
    for material, key in placement_key.items():
        coarse_bounds = tuple(coarse_payload["placement"][key][2])
        fine_bounds = tuple(fine_payload["placement"][key][2])
        weights = np.asarray(coarse["volumes"][material], dtype=np.float64)
        if material == "au":
            coarse_mask = np.asarray(coarse["solver_mask"])
            fine_mask = np.asarray(fine["solver_mask"])
            if (
                coarse_mask.shape != weights.shape[1:3]
                or fine_mask.shape != weights.shape[1:3]
                or not np.array_equal(coarse_mask, fine_mask)
                or not np.all((coarse_mask == 0) | (coarse_mask == 1))
            ):
                raise ValueError("Au field comparison requires one identical binary mask")
            weights = weights * coarse_mask[None, :, :, None]
        interpolated = []
        for component in range(3):
            coarse_z = _z_coordinates(
                coarse["grid_edges"][2], coarse_bounds, component
            )
            fine_z = _z_coordinates(
                fine["grid_edges"][2], fine_bounds, component
            )
            interpolated.append(
                _interpolate_fine_to_coarse_z(
                    fine["fields_late"][material][component],
                    fine_z,
                    coarse_z,
                )
            )
        fine_on_coarse = np.stack(interpolated)
        result[material] = weighted_complex_nrmse(
            fine_on_coarse,
            coarse["fields_late"][material],
            weights,
        )
    return result


def _z_control_intervals(
    edges: np.ndarray, bounds: tuple[int, int], component: int
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(edges, dtype=np.float64)
    lower, upper = bounds
    indices = np.arange(lower, upper)
    if component == 2:
        left = value[indices]
        right = value[indices + 1]
    else:
        if lower < 1 or upper >= value.size:
            raise ValueError("edge-centered material controls require neighboring cells")
        left = 0.5 * (value[indices - 1] + value[indices])
        right = 0.5 * (value[indices] + value[indices + 1])
    if np.any(right <= left):
        raise ValueError("Yee control intervals must have positive width")
    return left, right


def _restrict_component_q_to_coarse_z(
    coarse_q: np.ndarray,
    fine_q: np.ndarray,
    coarse_volume: np.ndarray,
    fine_volume: np.ndarray,
    coarse_edges: np.ndarray,
    fine_edges: np.ndarray,
    coarse_bounds: tuple[int, int],
    fine_bounds: tuple[int, int],
    component: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    coarse_value = np.asarray(coarse_q, dtype=np.float64)
    fine_value = np.asarray(fine_q, dtype=np.float64)
    coarse_weight = np.asarray(coarse_volume, dtype=np.float64)
    fine_weight = np.asarray(fine_volume, dtype=np.float64)
    coarse_left, coarse_right = _z_control_intervals(
        coarse_edges, coarse_bounds, component
    )
    fine_left, fine_right = _z_control_intervals(
        fine_edges, fine_bounds, component
    )
    coarse_native_width = coarse_right - coarse_left
    fine_native_width = fine_right - fine_left
    if coarse_value.shape != coarse_weight.shape or fine_value.shape != fine_weight.shape:
        raise ValueError("Q and volume component shapes must match")
    if coarse_value.shape[:2] != fine_value.shape[:2]:
        raise ValueError("full-z comparison requires identical x/y material grids")
    if coarse_value.shape[-1] != coarse_native_width.size:
        raise ValueError("coarse Q z shape does not match Yee controls")
    if fine_value.shape[-1] != fine_native_width.size:
        raise ValueError("fine Q z shape does not match Yee controls")

    coarse_area = coarse_weight / coarse_native_width[None, None, :]
    fine_area = fine_weight / fine_native_width[None, None, :]
    coarse_area_z_invariant = np.allclose(
        coarse_area, coarse_area[..., :1], rtol=1.0e-12, atol=0.0
    )
    fine_area_z_invariant = np.allclose(
        fine_area, fine_area[..., :1], rtol=1.0e-12, atol=0.0
    )
    xy_areas_match = np.allclose(
        coarse_area[..., 0], fine_area[..., 0], rtol=1.0e-12, atol=0.0
    )

    common_lower = max(float(coarse_left[0]), float(fine_left[0]))
    common_upper = min(float(coarse_right[-1]), float(fine_right[-1]))
    if common_upper <= common_lower:
        raise ValueError("coarse and fine Yee controls have no common z support")
    clipped_left = np.maximum(coarse_left, common_lower)
    clipped_right = np.minimum(coarse_right, common_upper)
    common_width = np.maximum(clipped_right - clipped_left, 0.0)
    if np.any(common_width <= 0.0):
        raise ValueError("every coarse material control must retain common support")
    overlap = np.maximum(
        0.0,
        np.minimum(clipped_right[:, None], fine_right[None, :])
        - np.maximum(clipped_left[:, None], fine_left[None, :]),
    )
    row_sums_match = np.allclose(
        np.sum(overlap, axis=1), common_width, rtol=1.0e-12, atol=1.0e-18
    )
    fine_coverage = np.sum(overlap, axis=0)
    fine_on_coarse = np.einsum(
        "xyf,cf->xyc", fine_value, overlap, optimize=True
    ) / common_width[None, None, :]
    common_weight = coarse_area[..., 0, None] * common_width[None, None, :]
    mapped_power = float(np.sum(fine_on_coarse * common_weight))
    direct_fine_common_power = float(
        np.sum(
            fine_value
            * fine_area[..., :1]
            * fine_coverage[None, None, :]
        )
    )
    conservation_error = abs(mapped_power - direct_fine_common_power) / max(
        abs(direct_fine_common_power), np.finfo(float).tiny
    )
    common_span = common_upper - common_lower
    support_fraction = min(
        common_span / (float(coarse_right[-1]) - float(coarse_left[0])),
        common_span / (float(fine_right[-1]) - float(fine_left[0])),
    )
    checks = {
        "coarse_xy_area_is_z_invariant": bool(coarse_area_z_invariant),
        "fine_xy_area_is_z_invariant": bool(fine_area_z_invariant),
        "coarse_and_fine_xy_areas_match": bool(xy_areas_match),
        "overlap_rows_cover_common_coarse_controls": bool(row_sums_match),
        "common_z_support_fraction_sufficient": support_fraction
        >= MINIMUM_COMMON_Z_SUPPORT_FRACTION - COMMON_Z_SUPPORT_NUMERICAL_ATOL,
        "fine_restriction_conserves_common_support_power": conservation_error
        <= REMAP_CONSERVATION_RTOL,
    }
    return coarse_value, fine_on_coarse, common_weight, {
        "common_z_bounds_m": [common_lower, common_upper],
        "common_z_support_fraction": support_fraction,
        "minimum_required_common_z_support_fraction": (
            MINIMUM_COMMON_Z_SUPPORT_FRACTION
        ),
        "common_z_support_numerical_tolerance": COMMON_Z_SUPPORT_NUMERICAL_ATOL,
        "fine_restriction_relative_power_error": conservation_error,
        "checks": checks,
        "ready": all(checks.values()),
    }


def _conservative_q_comparison(
    coarse: Mapping[str, Any],
    fine: Mapping[str, Any],
    coarse_payload: Mapping[str, Any],
    fine_payload: Mapping[str, Any],
) -> dict[str, Any]:
    placement_key = {"au": "au_design", "tairte4": "fixed_tairte4"}
    numerator = 0.0
    denominator = 0.0
    details: dict[str, Any] = {}
    flat_audits = []
    for material, key in placement_key.items():
        details[material] = {}
        coarse_bounds = tuple(coarse_payload["placement"][key][2])
        fine_bounds = tuple(fine_payload["placement"][key][2])
        for component, axis in enumerate(("x", "y", "z")):
            coarse_q, fine_q, weights, audit = _restrict_component_q_to_coarse_z(
                coarse["q_late"][material][component],
                fine["q_late"][material][component],
                coarse["volumes"][material][component],
                fine["volumes"][material][component],
                coarse["grid_edges"][2],
                fine["grid_edges"][2],
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
            flat_audits.append(audit)
    checks = {
        "all_component_restrictions_ready": bool(flat_audits)
        and all(item["ready"] is True for item in flat_audits),
        "all_component_checks_pass": bool(flat_audits)
        and all(_all_true(item["checks"]) for item in flat_audits),
    }
    return {
        "combined_Q_volume_L2_NRMSE": math.sqrt(
            numerator / max(denominator, np.finfo(float).tiny)
        ),
        "minimum_common_z_support_fraction": min(
            item["common_z_support_fraction"] for item in flat_audits
        ),
        "maximum_fine_restriction_relative_power_error": max(
            item["fine_restriction_relative_power_error"] for item in flat_audits
        ),
        "details": details,
        "checks": checks,
        "ready": all(checks.values()),
    }


def compare_full_z_pair(
    coarse_level: str,
    fine_level: str,
    snapshots: Mapping[str, Mapping[str, Mapping[str, Any] | None]],
    payloads: Mapping[str, Mapping[str, Mapping[str, Any]]],
    source_pairs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if (coarse_level, fine_level) not in SUCCESSIVE_PAIRS:
        raise ValueError("only declared successive full-z pairs may be compared")
    if any(
        snapshots[level][polarization] is None
        for level in (coarse_level, fine_level)
        for polarization in POLARIZATIONS
    ):
        return {
            "coarse_level": coarse_level,
            "fine_level": fine_level,
            "pass": False,
            "error": "one or more raw snapshots failed artifact validation",
            "checks": {},
        }
    if any(
        not isinstance(source_pairs.get(level, {}).get("comparison"), Mapping)
        for level in (coarse_level, fine_level)
    ):
        return {
            "coarse_level": coarse_level,
            "fine_level": fine_level,
            "pass": False,
            "error": "one or more source-pair payloads failed validation",
            "checks": {},
        }

    source_change = {
        polarization: relative_difference(
            source_pairs[coarse_level]["comparison"]["unscaled_incident_power_W"][polarization],
            source_pairs[fine_level]["comparison"]["unscaled_incident_power_W"][polarization],
        )
        for polarization in POLARIZATIONS
    }
    source_change["mean"] = relative_difference(
        source_pairs[coarse_level]["comparison"]["mean_unscaled_incident_power_W"],
        source_pairs[fine_level]["comparison"]["mean_unscaled_incident_power_W"],
    )

    per_polarization: dict[str, Any] = {}
    comparison_checks: dict[str, bool] = {}
    for polarization in POLARIZATIONS:
        coarse = snapshots[coarse_level][polarization]
        fine = snapshots[fine_level][polarization]
        coarse_payload = payloads[coarse_level][polarization]
        fine_payload = payloads[fine_level][polarization]
        probe_weights_identical = np.array_equal(
            coarse["probe_weights"][:2], fine["probe_weights"][:2]
        )
        probe_nrmse = weighted_complex_nrmse(
            fine["probe"][:2], coarse["probe"][:2], fine["probe_weights"][:2]
        )
        material_field = _material_field_comparison(
            coarse, fine, coarse_payload, fine_payload
        )
        q_comparison = _conservative_q_comparison(
            coarse, fine, coarse_payload, fine_payload
        )
        material_component_q: dict[str, Any] = {}
        flat_component_changes: list[float] = []
        for material in ("au", "tairte4"):
            material_component_q[material] = {
                axis: relative_difference(
                    coarse["power_late"]["by_material"][material]["component_W"][axis],
                    fine["power_late"]["by_material"][material]["component_W"][axis],
                )
                for axis in ("x", "y", "z")
            }
            material_component_q[material]["total"] = relative_difference(
                coarse["power_late"]["by_material"][material]["total_W"],
                fine["power_late"]["by_material"][material]["total_W"],
            )
            flat_component_changes.extend(material_component_q[material].values())
        per_polarization[polarization] = {
            "total_Q_relative_change": relative_difference(
                coarse["power_late"]["total_W"], fine["power_late"]["total_W"]
            ),
            "material_component_Q_relative_change": material_component_q,
            "material_component_Q_max_relative_change": max(flat_component_changes),
            "tangential_complex_E_fixed_probe_NRMSE": probe_nrmse,
            "material_region_complex_E_NRMSE_after_fine_to_coarse_z_interpolation": material_field,
            "material_region_complex_E_max_NRMSE": max(material_field.values()),
            "conservative_Q": q_comparison,
        }
        comparison_checks[f"{polarization}_tangential_probe_weights_identical"] = bool(
            probe_weights_identical
        )
        comparison_checks[f"{polarization}_conservative_Q_restriction_ready"] = (
            q_comparison["ready"] is True
        )

    closure_values = [
        float(payloads[level][polarization]["evaluation"]["flux"][name])
        for level in (coarse_level, fine_level)
        for polarization in POLARIZATIONS
        for name in (
            "Q_vs_closed_phasor_symmetric_relative",
            "Q_vs_closed_td_symmetric_relative",
        )
    ]
    metrics = {
        "source_power_relative_change": max(source_change.values()),
        "q_closed_flux_relative": max(closure_values),
        "stationarity_complex_E_NRMSE": max(
            float(payloads[fine_level][polarization]["evaluation"]["field_stationarity"]["maximum_complex_E_NRMSE"])
            for polarization in POLARIZATIONS
        ),
        "total_Q_relative_change": max(
            item["total_Q_relative_change"] for item in per_polarization.values()
        ),
        "material_component_Q_max_relative_change": max(
            item["material_component_Q_max_relative_change"]
            for item in per_polarization.values()
        ),
        "complex_E_fixed_probe_NRMSE": max(
            item["tangential_complex_E_fixed_probe_NRMSE"]
            for item in per_polarization.values()
        ),
        "conservative_Q_volume_L2_NRMSE": max(
            item["conservative_Q"]["combined_Q_volume_L2_NRMSE"]
            for item in per_polarization.values()
        ),
        "material_region_complex_E_max_NRMSE": max(
            item["material_region_complex_E_max_NRMSE"]
            for item in per_polarization.values()
        ),
    }
    evaluated = evaluate_pair(metrics)
    comparison_checks["material_region_complex_E_max_NRMSE"] = (
        metrics["material_region_complex_E_max_NRMSE"]
        <= MATERIAL_REGION_COMPLEX_E_NRMSE_LIMIT
    )
    checks = {**evaluated["checks"], **comparison_checks}
    return {
        "coarse_level": coarse_level,
        "coarse_z_factor": Z_FACTOR[coarse_level],
        "fine_level": fine_level,
        "fine_z_factor": Z_FACTOR[fine_level],
        "same_xy_grid_different_z_grid_comparison": True,
        "fixed_probe_method": (
            "exact common physical [-4,+4] um x/y cells at z=0.250 um; "
            "tangential Ex/Ey only because stored Ez lies at a grid-dependent "
            "half-cell z offset; component-specific Yee area weights"
        ),
        "material_field_method": (
            "component-wise complex linear interpolation of the fine Yee field "
            "onto coarse physical z coordinates; exact-binary Au solver mask "
            "applied so design-window air is excluded; extrapolation forbidden"
        ),
        "Q_method": (
            "component-Yee power density restricted by exact physical z-control "
            "overlaps onto common coarse control volumes; x/y grids unchanged"
        ),
        "source_power_relative_change": source_change,
        "per_polarization": per_polarization,
        "metrics": metrics,
        "limits": {
            **OPTICAL_PAIR_GATES,
            "material_region_complex_E_max_NRMSE": (
                MATERIAL_REGION_COMPLEX_E_NRMSE_LIMIT
            ),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def full_z_selection_gates(
    case_ready: Mapping[str, Mapping[str, bool]],
    pair_pass: Mapping[tuple[str, str], bool],
) -> dict[str, bool]:
    return {
        "all_three_full_z_levels_internally_ready": all(
            case_ready[level][polarization] is True
            for level in LEVELS
            for polarization in POLARIZATIONS
        ),
        "z2_to_z4_cross_comparison_passes": pair_pass[("z2", "z4")] is True,
        "z4_to_z8_cross_comparison_passes": pair_pass[("z4", "z8")] is True,
    }


def source_raw_grid_audit(
    source_pair: Mapping[str, Any], spec: FreshCaseSpec, campaign_root: Path
) -> dict[str, Any]:
    expected_edges = tuple(
        np.asarray(axis, dtype=np.float64)
        for axis in contract_grid_edges(spec.mesh)
    )
    checks: dict[str, bool] = {}
    records: dict[str, Any] = {}
    for polarization in POLARIZATIONS:
        report_input = Path(
            source_pair["cases"][polarization]["report_path"]
        ).expanduser()
        report_path = report_input.resolve()
        checks[f"{polarization}_report_path_is_absolute"] = report_input.is_absolute()
        checks[f"{polarization}_report_is_under_campaign_root"] = (
            report_path.is_relative_to(campaign_root)
        )
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        runner_input = Path(
            payload["provenance"]["runner_path"]
        ).expanduser()
        runner_path = runner_input.resolve()
        runner_exists = runner_path.is_file()
        checks[f"{polarization}_source_runner_exists_and_matches"] = (
            runner_exists
            and runner_input.is_absolute()
            and sha256(runner_path) == payload["provenance"]["runner_sha256"]
        )
        raw_input = Path(payload["raw"]["path"]).expanduser()
        raw_path = raw_input.resolve()
        checks[f"{polarization}_raw_path_is_absolute"] = raw_input.is_absolute()
        checks[f"{polarization}_raw_is_under_campaign_root"] = (
            raw_path.is_relative_to(campaign_root)
        )
        declared = payload["raw"]["arrays"]
        required = {f"grid_{axis}_edges_m" for axis in "xyz"}
        checks[f"{polarization}_grid_arrays_declared"] = required.issubset(declared)
        checks[f"{polarization}_placement_matches_contract"] = (
            payload["placement"]
            == source_pair["source_case_contracts"]["placement"]
            == expected_placement(spec.mesh)
        )
        axis_records: dict[str, Any] = {}
        if checks[f"{polarization}_grid_arrays_declared"]:
            with np.load(raw_path, allow_pickle=False) as archive:
                for index, axis in enumerate("xyz"):
                    name = f"grid_{axis}_edges_m"
                    actual = np.asarray(archive[name])
                    expected = expected_edges[index]
                    exact = np.array_equal(actual, expected.astype(actual.dtype))
                    error = float(
                        np.max(np.abs(actual.astype(np.float64) - expected))
                    )
                    checks[f"{polarization}_{axis}_grid_exact_after_dtype_cast"] = exact
                    checks[f"{polarization}_{axis}_grid_error_below_tolerance"] = (
                        error <= 1.0e-12
                    )
                    axis_records[axis] = {
                        "dtype": str(actual.dtype),
                        "shape": list(actual.shape),
                        "max_absolute_error_m": error,
                    }
        records[polarization] = {
            "report_path": str(report_path),
            "runner_path": str(runner_path),
            "runner_sha256": payload["provenance"]["runner_sha256"],
            "raw_path": str(raw_path),
            "grid_edges": axis_records,
        }
    return {
        "records": records,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": bool(checks) and all(checks.values()),
    }


def build_full_z_certificate(
    root: Path,
    contract_sha256s: Mapping[str, str],
    source_pair_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    campaign_root = root.expanduser().resolve()
    if not root.expanduser().is_absolute() or not campaign_root.is_dir():
        raise RuntimeError("campaign root must be an existing absolute directory")
    if set(contract_sha256s) != set(LEVELS):
        raise ValueError(f"contract SHA mapping must contain exactly {LEVELS}")
    if set(source_pair_sha256s) != set(LEVELS):
        raise ValueError(f"source-pair SHA mapping must contain exactly {LEVELS}")

    contracts: dict[str, Any] = {}
    source_pairs: dict[str, Any] = {}
    source_grid_audits: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {}
    cases: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    for level in LEVELS:
        spec = expected_full_z_case(level)
        contract_path = (
            campaign_root / "contracts" / f"l500_full_z_{level}.json"
        ).resolve()
        loaded_spec, contract_payload, contract_audit = load_case_contract(
            contract_path, contract_sha256s[level]
        )
        if loaded_spec != spec:
            raise RuntimeError(f"{level} is not the exact expected full-z case")
        contracts[level] = {
            "spec": spec,
            "payload": contract_payload,
            "audit": contract_audit,
        }

        pair_path = (
            campaign_root
            / f"source_pair_full_z_{level}"
            / "FDTDX_FRESH_SOURCE_ONLY_PAIR.json"
        ).resolve()
        try:
            pair_payload, pair_audit = validate_source_pair(
                pair_path, source_pair_sha256s[level], expected_case=spec
            )
        except Exception as error:
            pair_payload = {}
            pair_audit = {
                "ready": False,
                "error": repr(error),
                "checks": {},
                "failed_checks": ["source_pair_revalidation_exception"],
            }
        source_pairs[level] = {
            "path": str(pair_path),
            "payload": pair_payload,
            "audit": pair_audit,
        }
        if pair_audit.get("ready") is True:
            try:
                source_grid_audits[level] = source_raw_grid_audit(
                    pair_payload, spec, campaign_root
                )
            except Exception as error:
                source_grid_audits[level] = {
                    "ready": False,
                    "error": repr(error),
                    "checks": {},
                    "failed_checks": ["source_raw_grid_audit_exception"],
                }
        else:
            source_grid_audits[level] = {
                "ready": False,
                "checks": {},
                "failed_checks": ["source_pair_not_ready"],
            }

        payloads[level] = {}
        cases[level] = {}
        snapshots[level] = {}
        for polarization in POLARIZATIONS:
            report = (
                campaign_root
                / f"l500_full_z_{level}_{polarization}"
                / "FDTDX_FRESH_EXACT_BINARY_PILOT.json"
            ).resolve()
            payload, audit, snapshot = _material_case_audit(
                report,
                campaign_root,
                TOTAL_PERIODS,
                polarization,
                spec,
                contract_path,
                contract_sha256s[level],
                pair_path,
                source_pair_sha256s[level],
            )
            payloads[level][polarization] = payload
            cases[level][polarization] = audit
            snapshots[level][polarization] = snapshot

    pair_results = {
        f"{coarse}_to_{fine}": compare_full_z_pair(
            coarse,
            fine,
            snapshots,
            payloads,
            {level: source_pairs[level]["payload"] for level in LEVELS},
        )
        for coarse, fine in SUCCESSIVE_PAIRS
    }
    case_ready = {
        level: {
            polarization: payloads[level][polarization]["evaluation"]["ready"]
            for polarization in POLARIZATIONS
        }
        for level in LEVELS
    }
    selection = full_z_selection_gates(
        case_ready,
        {
            pair: pair_results[f"{pair[0]}_to_{pair[1]}"]["pass"]
            for pair in SUCCESSIVE_PAIRS
        },
    )
    flat_payloads = [
        payloads[level][polarization]
        for level in LEVELS
        for polarization in POLARIZATIONS
    ]
    flat_cases = [
        cases[level][polarization]
        for level in LEVELS
        for polarization in POLARIZATIONS
    ]
    repository_commits = {
        payload["provenance"]["repository_commit"] for payload in flat_payloads
    }
    source_pair_commits = {
        item["payload"].get("provenance", {}).get("certificate_repository_commit")
        for item in source_pairs.values()
    }
    expected_cfl_dt = [
        COURANT_FACTOR
        / (
            C0_M_PER_S
            * math.sqrt(
                sum(
                    1.0
                    / float(
                        np.min(
                            np.diff(
                                snapshots[level][polarization]["grid_edges"][axis].astype(
                                    np.float64
                                )
                            )
                        )
                    )
                    ** 2
                    for axis in range(3)
                )
            )
        )
        for level in LEVELS
        for polarization in POLARIZATIONS
    ]
    realized_dt = [
        float(payloads[level][polarization]["time_contract"]["time_step_s"])
        for level in LEVELS
        for polarization in POLARIZATIONS
    ]
    dt_times_z_factor = [
        float(payloads[level][polarization]["time_contract"]["time_step_s"])
        * Z_FACTOR[level]
        for level in LEVELS
        for polarization in POLARIZATIONS
    ]
    end_times = [
        float(payloads[level][polarization]["time_contract"]["time_step_s"])
        * int(payloads[level][polarization]["time_contract"]["time_steps_total"])
        for level in LEVELS
        for polarization in POLARIZATIONS
    ]
    maximum_dt = max(
        float(payloads[level][polarization]["time_contract"]["time_step_s"])
        for level in LEVELS
        for polarization in POLARIZATIONS
    )
    reference_snapshot = snapshots[LEVELS[0]][POLARIZATIONS[0]]
    xy_grid_edges_identical = reference_snapshot is not None and all(
        snapshot is not None
        and np.array_equal(snapshot["grid_edges"][axis], reference_snapshot["grid_edges"][axis])
        for level in LEVELS
        for polarization in POLARIZATIONS
        for axis in (0, 1)
        for snapshot in (snapshots[level][polarization],)
    )
    masks_identical = reference_snapshot is not None and all(
        snapshot is not None
        and np.array_equal(snapshot["design_mask"], reference_snapshot["design_mask"])
        and np.array_equal(snapshot["solver_mask"], reference_snapshot["solver_mask"])
        for level in LEVELS
        for polarization in POLARIZATIONS
        for snapshot in (snapshots[level][polarization],)
    )
    probe_tangential_weights_identical = reference_snapshot is not None and all(
        snapshot is not None
        and np.array_equal(
            snapshot["probe_weights"][:2], reference_snapshot["probe_weights"][:2]
        )
        for level in LEVELS
        for polarization in POLARIZATIONS
        for snapshot in (snapshots[level][polarization],)
    )
    base_mesh = contracts[LEVELS[0]]["payload"]["mesh_spec"]
    source_runner_hashes = {
        record["runner_sha256"]
        for audit in source_grid_audits.values()
        for record in audit.get("records", {}).values()
        if "runner_sha256" in record
    }
    gates = {
        "all_canonical_case_contracts_revalidated": all(
            contracts[level]["audit"]["ready"] is True for level in LEVELS
        ),
        "all_source_pairs_revalidated": all(
            source_pairs[level]["audit"].get("ready") is True for level in LEVELS
        ),
        "all_source_raw_grids_and_runners_revalidated": all(
            source_grid_audits[level].get("ready") is True for level in LEVELS
        ),
        "all_material_artifacts_and_recomputed_physics_ready": all(
            case["artifact_ready"] is True for case in flat_cases
        ),
        "all_material_cases_internally_ready": all(
            case_ready[level][polarization] is True
            for level in LEVELS
            for polarization in POLARIZATIONS
        ),
        "only_full_domain_z_factor_changes_across_contracts": all(
            {name: value for name, value in contracts[level]["payload"]["mesh_spec"].items() if name != "z_factor"}
            == {name: value for name, value in base_mesh.items() if name != "z_factor"}
            and contracts[level]["payload"]["mesh_spec"]["z_factor"]
            == Z_FACTOR[level]
            and contracts[level]["spec"].time
            == contracts[LEVELS[0]]["spec"].time
            and contracts[level]["spec"].pml_alpha_scale
            == contracts[LEVELS[0]]["spec"].pml_alpha_scale
            and contracts[level]["spec"].pml_target_reflection
            == contracts[LEVELS[0]]["spec"].pml_target_reflection
            for level in LEVELS
        ),
        "realized_dt_matches_raw_grid_CFL_formula": all(
            math.isclose(actual, expected, rel_tol=CFL_DT_RTOL, abs_tol=0.0)
            for actual, expected in zip(realized_dt, expected_cfl_dt, strict=True)
        ),
        "realized_end_times_agree_within_one_coarse_step": max(end_times)
        - min(end_times)
        <= maximum_dt,
        "x_y_grid_edges_identical": bool(xy_grid_edges_identical),
        "exact_L500_design_and_solver_masks_identical": bool(masks_identical),
        "tangential_probe_weights_identical": bool(probe_tangential_weights_identical),
        "pml_physical_parameters_identical": len(
            {
                json.dumps(
                    contracts[level]["payload"]["resolved_pml_face_parameters"],
                    sort_keys=True,
                )
                for level in LEVELS
            }
        )
        == 1,
        "material_repository_commit_identical": len(repository_commits) == 1,
        "source_pair_repository_commit_identical": len(source_pair_commits) == 1,
        "source_and_material_repository_commit_identical": source_pair_commits
        == repository_commits,
        "source_runner_hash_identical": len(source_runner_hashes) == 1,
        "source_pair_generator_hash_identical": len(
            {
                item["payload"].get("provenance", {}).get(
                    "certificate_generator_sha256"
                )
                for item in source_pairs.values()
            }
        )
        == 1,
        "fdtdx_source_provenance_identical": len(
            {
                json.dumps(payload["provenance"]["fdtdx_source"], sort_keys=True)
                for payload in flat_payloads
            }
        )
        == 1,
        "runtime_lock_identical": len(
            {
                json.dumps(payload["provenance"]["runtime_lock"], sort_keys=True)
                for payload in flat_payloads
            }
        )
        == 1,
        "material_runner_hash_identical": len(
            {payload["provenance"]["runner_sha256"] for payload in flat_payloads}
        )
        == 1,
        "material_contract_hash_identical": len(
            {
                payload["provenance"]["material_contract_sha256"]
                for payload in flat_payloads
            }
        )
        == 1,
        "both_successive_full_z_comparisons_and_selection_pass": _all_true(
            selection
        ),
        "optimizer_remains_forbidden": all(
            payload.get("optimizer_start_allowed") is False
            for payload in flat_payloads
        ),
    }
    ready = all(gates.values())
    return {
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "scope": (
            "full-domain-z resolution convergence of one exact-binary L500 "
            "optical reference at 24 periods and Courant 0.25; x/y fixed"
        ),
        "campaign_root": str(campaign_root),
        "contracts": {
            level: {
                "z_factor": Z_FACTOR[level],
                "grid_shape_xyz": contracts[level]["payload"]["resolved_mesh"]["grid_shape_xyz"],
                "vertical_segments": contracts[level]["payload"]["resolved_mesh"]["vertical_segments"],
                "case_contract_sha256": contracts[level]["payload"]["case_contract_sha256"],
                "file_audit": contracts[level]["audit"],
            }
            for level in LEVELS
        },
        "source_pairs": {
            level: {
                "path": source_pairs[level]["path"],
                "audit": source_pairs[level]["audit"],
                "raw_grid_audit": source_grid_audits[level],
                "unscaled_incident_power_W": source_pairs[level]["payload"].get(
                    "comparison", {}
                ).get("unscaled_incident_power_W"),
            }
            for level in LEVELS
        },
        "cases": cases,
        "successive_comparisons": pair_results,
        "selection": {
            "selected_level": SELECTED_LEVEL if ready else None,
            "selected_z_factor": Z_FACTOR[SELECTED_LEVEL] if ready else None,
            "confirmation_level": CONFIRMATION_LEVEL if ready else None,
            "confirmation_z_factor": (
                Z_FACTOR[CONFIRMATION_LEVEL] if ready else None
            ),
            "policy": (
                "select z_factor=4 only when all z2/z4/z8 Ea/Eb cases are "
                "internally ready and both z2-to-z4 and z4-to-z8 physical-grid "
                "comparisons pass"
            ),
            "gates": selection,
            "failed_gates": [name for name, passed in selection.items() if not passed],
        },
        "realized_time_diagnostics": {
            "realized_dt_s": realized_dt,
            "expected_raw_grid_CFL_dt_s": expected_cfl_dt,
            "dt_times_z_factor_s": dt_times_z_factor,
            "end_times_s": end_times,
            "maximum_time_step_s": maximum_dt,
        },
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "is_full_domain_z_resolution_certificate": ready,
        "is_mesh_certificate": False,
        "optimizer_start_allowed": False,
        "next_allowed_step": (
            "run the exact L500 reference on the design-window x/y resolution "
            "ladder at 24 periods, Courant 0.25, and selected z_factor; do not optimize"
        ),
    }


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_full_z_certificate(
    root: Path,
    contract_sha256s: Mapping[str, str],
    source_pair_sha256s: Mapping[str, str],
    output_directory: Path,
) -> dict[str, Any]:
    output = output_directory.expanduser().resolve()
    if not output_directory.expanduser().is_absolute() or not output.is_dir():
        raise RuntimeError("output directory must be an existing absolute directory")
    if any(output.iterdir()):
        raise RuntimeError("output directory must be empty before certification")
    result = build_full_z_certificate(root, contract_sha256s, source_pair_sha256s)
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
    result["status"] = STATUS_READY if result["ready"] else STATUS_BLOCKED
    result["is_full_domain_z_resolution_certificate"] = result["ready"]
    result["failed_gates"] = [
        name for name, passed in result["gates"].items() if not passed
    ]
    if not result["ready"]:
        for name in (
            "selected_level",
            "selected_z_factor",
            "confirmation_level",
            "confirmation_z_factor",
        ):
            result["selection"][name] = None
    _atomic_json(output / CERTIFICATE_NAME, result)
    return result


def _level_sha(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        try:
            level, digest = item.split("=", 1)
        except ValueError as error:
            raise ValueError(f"{label} entries must use LEVEL=SHA256") from error
        if level in result:
            raise ValueError(f"duplicate {label} level {level}")
        normalized = digest.strip().lower()
        if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
            raise ValueError(f"{label} {level} is not a lowercase SHA256")
        result[level] = normalized
    if set(result) != set(LEVELS):
        raise ValueError(f"{label} entries must contain exactly {LEVELS}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract-sha256", action="append", default=[], metavar="LEVEL=SHA256")
    parser.add_argument("--source-pair-sha256", action="append", default=[], metavar="LEVEL=SHA256")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = write_full_z_certificate(
            args.root,
            _level_sha(args.contract_sha256, "contract"),
            _level_sha(args.source_pair_sha256, "source-pair"),
            args.output_dir,
        )
    except Exception as error:
        failure = {
            "status": "BLOCKED_FDTDX_FRESH_L500_FULL_DOMAIN_Z_EXCEPTION",
            "ready": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "is_full_domain_z_resolution_certificate": False,
            "is_mesh_certificate": False,
            "optimizer_start_allowed": False,
        }
        output = args.output_dir.expanduser().resolve()
        if output.is_dir() and not any(output.iterdir()):
            _atomic_json(output / CERTIFICATE_NAME, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    summary = {
        "status": result["status"],
        "ready": result["ready"],
        "selection": result["selection"],
        "failed_gates": result["failed_gates"],
        "successive_comparisons": {
            name: {"pass": item["pass"], "metrics": item.get("metrics")}
            for name, item in result["successive_comparisons"].items()
        },
        "report": str(args.output_dir.expanduser().resolve() / CERTIFICATE_NAME),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
