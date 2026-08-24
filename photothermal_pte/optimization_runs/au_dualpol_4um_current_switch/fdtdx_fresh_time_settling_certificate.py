"""Fail-closed certificate for the fresh FDTDX 16/24/32-period ladder.

This module certifies only time settling for one exact-binary L500 reference on
one fixed spatial grid.  It deliberately does not certify a mesh, a thermal or
electrical model, an adjoint, or an optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any, Mapping

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    DESIGN_HALF_SPAN_M,
    OPTICAL_PAIR_GATES,
    MeshSpec,
    evaluate_pair,
    grid_edges as contract_grid_edges,
    reference_mask,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_material import (
    mask_material_audit,
    solver_mask,
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
    STATUS_BLOCKED as CASE_STATUS_BLOCKED,
    STATUS_READY as CASE_STATUS_READY,
    combined_weighted_nrmse,
    component_power,
    relative_difference,
    validate_source_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_metrics import (
    weighted_complex_nrmse,
)


CERTIFICATE_NAME = "FDTDX_FRESH_TIME_SETTLING_CERTIFICATE.json"
STATUS_READY = "VALIDATED_FDTDX_FRESH_L500_TIME_SETTLING"
STATUS_BLOCKED = "BLOCKED_FDTDX_FRESH_L500_TIME_SETTLING"
REFERENCE_NAME = "l_shape_4um_with_500nm_arms"
PERIODS = (16, 24, 32)
POLARIZATIONS = ("Ea", "Eb")
SUCCESSIVE_PAIRS = ((16, 24), (24, 32))
SELECTED_PERIODS = 24
CONFIRMATION_PERIODS = 32
WINDOW_PERIODS = 4
COURANT_FACTOR = 0.5
EXPECTED_SCOPE = (
    "one fixed exact-binary optical material pilot; "
    "no thermal/electrical/adjoint/optimizer"
)
RAW_POWER_RTOL = 5.0e-13
RAW_POWER_ATOL_W = 1.0e-30
METRIC_MATCH_RTOL = 5.0e-11
COORDINATE_ATOL_M = 1.0e-12
PROBE_Z_M = 0.250e-6
PILOT_EVALUATION_GATES = frozenset((
    "all_raw_detector_and_Q_values_finite",
    "complex_field_stationarity",
    "Q_previous_late_total_change",
    "Q_previous_late_spatial_change",
    "Q_nonnegative",
    "Q_total_finite_positive",
    "closed_phasor_finite_positive",
    "closed_td_finite_positive",
    "Q_closed_phasor_closure",
    "Q_closed_td_closure",
    "closed_td_phasor_agreement",
    "absorbed_fraction_physical",
    "source_side_net_flux_finite",
    "empty_has_exact_zero_Au_Q_or_nonempty_has_positive_Au_Q",
))
MATERIAL_CHECKS = frozenset((
    "a_realized_epsilon_matches_target",
    "a_realized_epsilon_passive",
    "au_realized_epsilon_matches_target",
    "au_realized_epsilon_passive",
    "b_realized_epsilon_matches_target",
    "b_realized_epsilon_passive",
    "c_realized_epsilon_matches_target",
    "c_realized_epsilon_passive",
    "exact_binary_au_readback",
    "silicon_inverse_permittivity_readback",
    "sio2_inverse_permittivity_readback",
    "tairte4_dispersive_c1_exact",
    "tairte4_dispersive_c2_exact",
    "tairte4_dispersive_c3_exact",
    "tairte4_epsilon_infinity_inverse_is_one",
))
EXACT_BINARY_CHECKS = frozenset((
    "au_window_epsilon_infinity_inverse_is_one",
    "dispersive_c1_exact_air_au_endpoints",
    "dispersive_c2_exact_air_au_endpoints",
    "dispersive_c3_exact_air_au_endpoints",
    "no_gray_material_law",
    "solver_mask_remains_binary",
))
SOURCE_PAIR_CONTRACT_CHECKS = frozenset((
    "mesh_matches_source_pair",
    "numerical_case_matches_source_pair",
    "placement_matches_source_pair",
    "pml_matches_source_pair",
    "source_matches_polarization_case",
    "time_matches_source_pair",
))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.uint8).tobytes()).hexdigest()


def _all_true(values: Mapping[str, Any]) -> bool:
    return bool(values) and all(value is True for value in values.values())


def _load_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must contain one object: {resolved}")
    return payload


def _file_audit(path_value: str | Path, expected_sha256: str) -> dict[str, Any]:
    supplied = Path(path_value).expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_file()
    actual = sha256(resolved) if exists else None
    return {
        "path": str(resolved),
        "path_is_absolute": supplied.is_absolute(),
        "exists": exists,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "sha256_matches": exists and actual == expected_sha256,
    }


def _power_matches(recorded: Mapping[str, Any], computed: Mapping[str, Any]) -> bool:
    recorded_values = [
        float(recorded["component_W"][axis]) for axis in ("x", "y", "z")
    ] + [float(recorded["total_W"])]
    computed_values = [
        float(computed["component_W"][axis]) for axis in ("x", "y", "z")
    ] + [float(computed["total_W"])]
    return bool(
        np.allclose(
            recorded_values,
            computed_values,
            rtol=RAW_POWER_RTOL,
            atol=RAW_POWER_ATOL_W,
        )
    )


def _strictly_increasing_finite(value: np.ndarray) -> bool:
    array = np.asarray(value, dtype=np.float64)
    return bool(
        array.ndim == 1
        and array.size >= 2
        and np.all(np.isfinite(array))
        and np.all(np.diff(array) > 0.0)
    )


def _edge_index(edges: np.ndarray, coordinate_m: float) -> int:
    matches = np.flatnonzero(
        np.isclose(edges, coordinate_m, rtol=0.0, atol=COORDINATE_ATOL_M)
    )
    if matches.size != 1:
        raise ValueError(
            f"physical coordinate {coordinate_m:.17g} m is not one unique grid edge"
        )
    return int(matches[0])


def _edge_dual_widths(widths: np.ndarray) -> np.ndarray:
    return 0.5 * (np.concatenate((widths[:1], widths[:-1])) + widths)


def electric_yee_volumes_from_edges(
    grid_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    """Reconstruct component-specific electric Yee dual volumes from edges."""

    widths = tuple(np.diff(np.asarray(edges, dtype=np.float64)) for edges in grid_edges)
    dual = tuple(_edge_dual_widths(value) for value in widths)
    if any(lower < 0 or upper <= lower for lower, upper in bounds):
        raise ValueError("Yee-volume bounds must be nonempty and nonnegative")
    volumes = []
    for component in range(3):
        selected = []
        for axis, (lower, upper) in enumerate(bounds):
            metric = widths[axis] if axis == component else dual[axis]
            if upper > metric.size:
                raise ValueError("Yee-volume bounds exceed the grid")
            selected.append(metric[lower:upper])
        volumes.append(
            selected[0][:, None, None]
            * selected[1][None, :, None]
            * selected[2][None, None, :]
        )
    return np.stack(volumes)


def fixed_probe_and_weights(
    target: np.ndarray,
    grid_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    placement: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Restrict the target detector to the fixed 8x8 um physical probe."""

    value = np.asarray(target)
    if value.ndim != 4 or value.shape[0] != 3:
        raise ValueError("target field must have shape (3,x,y,z)")
    bounds = tuple(tuple(int(item) for item in axis) for axis in placement["target_field"])
    if len(bounds) != 3 or any(upper <= lower for lower, upper in bounds):
        raise ValueError("target_field placement must contain three nonempty bounds")
    expected_shape = (3, *(upper - lower for lower, upper in bounds))
    if value.shape != expected_shape:
        raise ValueError(
            f"target array shape {value.shape} does not match placement {expected_shape}"
        )

    x_edges, y_edges, z_edges = (
        np.asarray(item, dtype=np.float64) for item in grid_edges
    )
    x_lower = _edge_index(x_edges, -DESIGN_HALF_SPAN_M)
    x_upper = _edge_index(x_edges, DESIGN_HALF_SPAN_M)
    y_lower = _edge_index(y_edges, -DESIGN_HALF_SPAN_M)
    y_upper = _edge_index(y_edges, DESIGN_HALF_SPAN_M)
    if not (
        bounds[0][0] <= x_lower < x_upper <= bounds[0][1]
        and bounds[1][0] <= y_lower < y_upper <= bounds[1][1]
    ):
        raise ValueError("fixed physical probe is not contained in target detector")
    if bounds[2][1] - bounds[2][0] != 1:
        raise ValueError("target detector must be exactly one z cell thick")
    target_z = float(z_edges[bounds[2][0]])
    if not math.isclose(target_z, PROBE_Z_M, rel_tol=0.0, abs_tol=COORDINATE_ATOL_M):
        raise ValueError("target detector is not anchored at z=0.250 um")

    local_x = slice(x_lower - bounds[0][0], x_upper - bounds[0][0])
    local_y = slice(y_lower - bounds[1][0], y_upper - bounds[1][0])
    probe = np.asarray(value[:, local_x, local_y, :])
    dx = np.diff(x_edges)
    dy = np.diff(y_edges)
    dual_x = _edge_dual_widths(dx)
    dual_y = _edge_dual_widths(dy)
    component_x = (dx[x_lower:x_upper], dual_x[x_lower:x_upper])
    component_y = (dual_y[y_lower:y_upper], dy[y_lower:y_upper])
    weights = np.stack(
        (
            component_x[0][:, None] * component_y[0][None, :],
            component_x[1][:, None] * component_y[1][None, :],
            component_x[1][:, None] * component_y[0][None, :],
        )
    )[..., None]
    if probe.shape != weights.shape:
        raise RuntimeError(
            f"fixed probe and Yee area shape mismatch: {probe.shape}, {weights.shape}"
        )
    audit = {
        "physical_xy_bounds_m": [
            [-DESIGN_HALF_SPAN_M, DESIGN_HALF_SPAN_M],
            [-DESIGN_HALF_SPAN_M, DESIGN_HALF_SPAN_M],
        ],
        "physical_z_m": target_z,
        "global_cell_bounds": [
            [x_lower, x_upper],
            [y_lower, y_upper],
            [bounds[2][0], bounds[2][1]],
        ],
        "shape": list(probe.shape),
        "component_specific_Yee_area_weights": True,
    }
    return probe, weights, audit


def expected_case(total_periods: int) -> FreshCaseSpec:
    if total_periods not in PERIODS:
        raise ValueError(f"total_periods must be one of {PERIODS}")
    return FreshCaseSpec(
        mesh=MeshSpec(),
        time=TimeSpec(
            total_periods=total_periods,
            window_periods=WINDOW_PERIODS,
            courant_factor=COURANT_FACTOR,
        ),
    )


def _required_raw_arrays() -> set[str]:
    return {
        "design_mask",
        "solver_mask",
        "au_previous",
        "au_late",
        "tairte4_previous",
        "tairte4_late",
        "target",
        "q_au_previous_W_m3",
        "q_au_late_W_m3",
        "q_tairte4_previous_W_m3",
        "q_tairte4_late_W_m3",
        "electric_dual_volume_au_m3",
        "electric_dual_volume_tairte4_m3",
        "grid_x_edges_m",
        "grid_y_edges_m",
        "grid_z_edges_m",
    }


def audit_raw_case(
    payload: Mapping[str, Any],
    spec: FreshCaseSpec,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Hash, reload, and independently recompute one material-case NPZ."""

    raw_record = payload["raw"]
    file_audit = _file_audit(raw_record["path"], raw_record["sha256"])
    result: dict[str, Any] = {
        **file_audit,
        "declared_arrays": raw_record["arrays"],
        "checks": {},
        "derived": {},
        "ready": False,
    }
    initial_checks = {
        "path_is_absolute": file_audit["path_is_absolute"],
        "file_exists": file_audit["exists"],
        "sha256_matches": file_audit["sha256_matches"],
    }
    if not all(initial_checks.values()):
        result["checks"] = initial_checks
        return result, None

    exact_design = np.asarray(reference_mask(REFERENCE_NAME), dtype=np.uint8)
    exact_solver = solver_mask(exact_design, spec.mesh)
    exact_material = mask_material_audit(exact_design, spec.mesh)
    required = _required_raw_arrays()
    with np.load(file_audit["path"], allow_pickle=False) as archive:
        files = set(archive.files)
        arrays_declared_exactly = files == set(raw_record["arrays"])
        required_present = required.issubset(files)
        declared_shapes_match = all(
            name in files
            and list(np.asarray(archive[name]).shape) == declared_shape
            for name, declared_shape in raw_record["arrays"].items()
        )
        if not required_present:
            checks = {
                **initial_checks,
                "arrays_declared_exactly": arrays_declared_exactly,
                "required_arrays_present": False,
                "declared_shapes_match": declared_shapes_match,
            }
            result["checks"] = checks
            return result, None

        arrays = {name: np.asarray(archive[name]) for name in required}
        all_declared_finite = all(
            np.all(np.isfinite(np.asarray(archive[name]))) for name in archive.files
        )

    design_mask = arrays["design_mask"]
    expanded_mask = arrays["solver_mask"]
    masks_integer = design_mask.dtype.kind in "biu" and expanded_mask.dtype.kind in "biu"
    masks_binary = bool(
        np.all((design_mask == 0) | (design_mask == 1))
        and np.all((expanded_mask == 0) | (expanded_mask == 1))
    )
    grid_edges = tuple(
        np.asarray(arrays[f"grid_{axis}_edges_m"])
        for axis in ("x", "y", "z")
    )
    expected_grid_shape = tuple(case_contract(spec)["resolved_mesh"]["grid_shape_xyz"])
    expected_grid_edges = tuple(
        np.asarray(axis, dtype=np.float64) for axis in contract_grid_edges(spec.mesh)
    )
    grid_shapes_match_contract = all(
        edges.shape == (expected_grid_shape[index] + 1,)
        for index, edges in enumerate(grid_edges)
    )
    grid_edges_valid = all(_strictly_increasing_finite(item) for item in grid_edges)
    grid_edges_match_contract = all(
        np.array_equal(actual, expected.astype(actual.dtype))
        and float(np.max(np.abs(actual.astype(np.float64) - expected)))
        <= COORDINATE_ATOL_M
        for actual, expected in zip(grid_edges, expected_grid_edges, strict=True)
    )

    fields = {
        window: {
            material: arrays[f"{material}_{window}"]
            for material in ("au", "tairte4")
        }
        for window in ("previous", "late")
    }
    q = {
        window: {
            material: np.asarray(arrays[f"q_{material}_{window}_W_m3"], dtype=np.float64)
            for material in ("au", "tairte4")
        }
        for window in ("previous", "late")
    }
    volumes = {
        material: np.asarray(
            arrays[f"electric_dual_volume_{material}_m3"], dtype=np.float64
        )
        for material in ("au", "tairte4")
    }
    placement_keys = {"au": "au_design", "tairte4": "fixed_tairte4"}
    placement_bounds = {
        material: tuple(
            tuple(int(item) for item in axis)
            for axis in payload["placement"][placement_keys[material]]
        )
        for material in ("au", "tairte4")
    }
    expected_volumes = {
        material: electric_yee_volumes_from_edges(grid_edges, placement_bounds[material])
        for material in ("au", "tairte4")
    }
    field_shapes_match_placement = all(
        fields[window][material].shape
        == (
            3,
            *(upper - lower for lower, upper in placement_bounds[material]),
        )
        for window in ("previous", "late")
        for material in ("au", "tairte4")
    )
    stored_volumes_match_edges = all(
        np.array_equal(volumes[material], expected_volumes[material])
        for material in ("au", "tairte4")
    )
    au_q_respects_binary_occupancy = (
        all(q[window]["au"].shape[1:3] == expanded_mask.shape for window in ("previous", "late"))
        and all(
            np.all(q[window]["au"][:, expanded_mask == 0, :] == 0.0)
            for window in ("previous", "late")
        )
    )
    field_q_volume_shapes_match = all(
        fields[window][material].shape
        == q[window][material].shape
        == volumes[material].shape
        and fields[window][material].ndim == 4
        and fields[window][material].shape[0] == 3
        for window in ("previous", "late")
        for material in ("au", "tairte4")
    )
    finite_nonnegative_q = all(
        np.all(np.isfinite(value)) and np.all(value >= 0.0)
        for window in q.values()
        for value in window.values()
    )
    finite_positive_volumes = all(
        np.all(np.isfinite(value)) and np.all(value > 0.0)
        for value in volumes.values()
    )

    computed_power: dict[str, dict[str, Any]] = {}
    power_matches = True
    for window in ("previous", "late"):
        by_material = {
            material: component_power(q[window][material], volumes[material])
            for material in ("au", "tairte4")
        }
        total = float(sum(item["total_W"] for item in by_material.values()))
        computed_power[window] = {"by_material": by_material, "total_W": total}
        recorded = payload["evaluation"]["Q"][window]
        power_matches = power_matches and all(
            _power_matches(recorded["by_material"][material], by_material[material])
            for material in ("au", "tairte4")
        )
        power_matches = power_matches and math.isclose(
            float(recorded["total_W"]),
            total,
            rel_tol=RAW_POWER_RTOL,
            abs_tol=RAW_POWER_ATOL_W,
        )

    stationarity = {
        material: weighted_complex_nrmse(
            fields["late"][material], fields["previous"][material], volumes[material]
        )
        for material in ("au", "tairte4")
    }
    stationarity["maximum"] = max(stationarity.values())
    q_spatial = combined_weighted_nrmse(q["late"], q["previous"], volumes)
    q_total = relative_difference(
        computed_power["late"]["total_W"], computed_power["previous"]["total_W"]
    )
    recorded_stationarity = payload["evaluation"]["field_stationarity"]
    recorded_q = payload["evaluation"]["Q"]
    recomputed_metrics_match = bool(
        math.isclose(
            stationarity["au"],
            float(recorded_stationarity["au_complex_E_NRMSE"]),
            rel_tol=METRIC_MATCH_RTOL,
            abs_tol=0.0,
        )
        and math.isclose(
            stationarity["tairte4"],
            float(recorded_stationarity["tairte4_complex_E_NRMSE"]),
            rel_tol=METRIC_MATCH_RTOL,
            abs_tol=0.0,
        )
        and math.isclose(
            stationarity["maximum"],
            float(recorded_stationarity["maximum_complex_E_NRMSE"]),
            rel_tol=METRIC_MATCH_RTOL,
            abs_tol=0.0,
        )
        and math.isclose(
            q_spatial,
            float(recorded_q["previous_late_spatial_NRMSE"]),
            rel_tol=METRIC_MATCH_RTOL,
            abs_tol=0.0,
        )
        and math.isclose(
            q_total,
            float(recorded_q["previous_late_total_relative_change"]),
            rel_tol=METRIC_MATCH_RTOL,
            abs_tol=0.0,
        )
    )

    probe, probe_weights, probe_audit = fixed_probe_and_weights(
        arrays["target"], grid_edges, payload["placement"]
    )
    exact_binary_report = payload["material"]["exact_binary_au"]
    checks = {
        **initial_checks,
        "arrays_declared_exactly": arrays_declared_exactly,
        "required_arrays_present": required_present,
        "declared_shapes_match": declared_shapes_match,
        "all_declared_arrays_finite": bool(all_declared_finite),
        "masks_have_integer_dtype": masks_integer,
        "masks_are_binary": masks_binary,
        "design_mask_is_exact_L500_reference": np.array_equal(design_mask, exact_design),
        "solver_mask_is_exact_replication": np.array_equal(expanded_mask, exact_solver),
        "design_mask_has_375_solid_cells": int(np.count_nonzero(design_mask)) == 375,
        "mask_hashes_match_exact_material_report": (
            _array_sha256(design_mask) == exact_binary_report["design_mask_sha256"]
            and _array_sha256(expanded_mask) == exact_binary_report["solver_mask_sha256"]
            and exact_binary_report["design_mask_sha256"]
            == exact_material["design_mask_sha256"]
            and exact_binary_report["solver_mask_sha256"]
            == exact_material["solver_mask_sha256"]
        ),
        "field_shapes_match_recorded_placement": field_shapes_match_placement,
        "field_Q_volume_shapes_match": field_q_volume_shapes_match,
        "stored_Yee_dual_volumes_match_grid_edges_and_placement": stored_volumes_match_edges,
        "Au_Q_is_exactly_zero_outside_binary_mask": bool(au_q_respects_binary_occupancy),
        "Q_is_finite_nonnegative": bool(finite_nonnegative_q),
        "electric_dual_volumes_are_finite_positive": bool(finite_positive_volumes),
        "grid_edge_shapes_match_contract": grid_shapes_match_contract,
        "grid_edges_match_contract_coordinates": grid_edges_match_contract,
        "grid_edges_are_finite_strictly_increasing": grid_edges_valid,
        "raw_Q_integrals_match_report": power_matches,
        "raw_stationarity_metrics_match_report": recomputed_metrics_match,
        "fixed_physical_probe_is_valid": bool(
            np.all(np.isfinite(probe))
            and np.all(np.isfinite(probe_weights))
            and np.all(probe_weights > 0.0)
        ),
    }
    result["checks"] = checks
    result["derived"] = {
        "design_mask_sha256": _array_sha256(design_mask),
        "solver_mask_sha256": _array_sha256(expanded_mask),
        "design_solid_cells": int(np.count_nonzero(design_mask)),
        "solver_solid_cells": int(np.count_nonzero(expanded_mask)),
        "power_W": computed_power,
        "stationarity_complex_E_NRMSE": stationarity,
        "previous_late_spatial_Q_NRMSE": q_spatial,
        "previous_late_total_Q_relative_change": q_total,
        "fixed_probe": probe_audit,
    }
    result["ready"] = _all_true(checks)
    snapshot = None
    if result["ready"]:
        snapshot = {
            "grid_edges": grid_edges,
            "probe": probe,
            "probe_weights": probe_weights,
            "fields_late": {material: fields["late"][material] for material in fields["late"]},
            "q_late": q["late"],
            "volumes": volumes,
            "power_late": computed_power["late"],
            "design_mask": design_mask,
            "solver_mask": expanded_mask,
        }
    return result, snapshot


def _material_case_audit(
    report_path: Path,
    root: Path,
    period: int,
    polarization: str,
    spec: FreshCaseSpec,
    contract_path: Path,
    contract_sha256: str,
    source_pair_path: Path,
    source_pair_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    resolved = report_path.expanduser().resolve()
    payload = _load_json(resolved)
    raw, snapshot = audit_raw_case(payload, spec)
    evaluation = payload["evaluation"]
    gates = evaluation["gates"]
    failed = [name for name, passed in gates.items() if passed is not True]
    evaluation_consistent = (
        set(gates) == PILOT_EVALUATION_GATES
        and evaluation.get("ready") is all(value is True for value in gates.values())
        and sorted(evaluation.get("failed_gates", [])) == sorted(failed)
    )
    expected_status = CASE_STATUS_READY if evaluation.get("ready") is True else CASE_STATUS_BLOCKED
    runner = _file_audit(
        payload["provenance"]["runner_path"], payload["provenance"]["runner_sha256"]
    )
    material_contract = _file_audit(
        payload["provenance"]["material_contract_path"],
        payload["provenance"]["material_contract_sha256"],
    )
    expected_contract = case_contract(spec)
    numerical_file = payload["numerical_case_file_audit"]
    exact_binary = payload["material"]["exact_binary_au"]
    expected_vector = [0.0, 1.0, 0.0] if polarization == "Ea" else [1.0, 0.0, 0.0]
    raw_path = Path(payload["raw"]["path"]).expanduser().resolve()
    checks = {
        "report_path_is_absolute": report_path.expanduser().is_absolute(),
        "report_is_under_campaign_root": resolved.is_relative_to(root),
        "raw_is_under_campaign_root": raw_path.is_relative_to(root),
        "case_labels_exact": payload.get("reference") == REFERENCE_NAME
        and payload.get("polarization") == polarization,
        "scope_exact": payload.get("scope") == EXPECTED_SCOPE,
        "status_and_evaluation_consistent": payload.get("status") == expected_status
        and payload.get("ready") is evaluation.get("ready")
        and evaluation_consistent,
        "numerical_case_contract_exact": payload.get("numerical_case_contract")
        == expected_contract,
        "numerical_case_file_binding_exact": (
            Path(numerical_file["path"]).expanduser().resolve() == contract_path
            and numerical_file.get("expected_sha256") == contract_sha256
            and numerical_file.get("actual_sha256") == contract_sha256
            and numerical_file.get("case_contract_sha256")
            == expected_contract["case_contract_sha256"]
            and numerical_file.get("ready") is True
            and _all_true(numerical_file.get("checks", {}))
        ),
        "mesh_exact": payload.get("mesh") == expected_contract["resolved_mesh"],
        "placement_matches_solver_independent_contract": payload.get("placement")
        == expected_placement(spec.mesh),
        "time_request_and_courant_exact": all(
            payload["time_contract"].get(name) == value
            for name, value in {
                "total_periods": period,
                "window_periods": WINDOW_PERIODS,
                "source_startup_periods": spec.time.source_startup_periods,
                "courant_factor": spec.time.courant_factor,
            }.items()
        )
        and int(payload["time_contract"].get("time_steps_total", 0)) > 0
        and float(payload["time_contract"].get("time_step_s", 0.0)) > 0.0,
        "pml_exact": payload.get("pml_face_parameters")
        == expected_contract["resolved_pml_face_parameters"],
        "source_polarization_exact": payload["source_contract"].get(
            "fixed_E_polarization_vector"
        )
        == expected_vector
        and payload["source_contract"].get("polarization") == polarization,
        "source_pair_binding_exact": (
            Path(payload["source_pair"]["path"]).expanduser().resolve()
            == source_pair_path
            and payload["source_pair"].get("expected_sha256") == source_pair_sha256
            and payload["source_pair"].get("actual_sha256") == source_pair_sha256
            and payload["source_pair"].get("ready") is True
            and _all_true(payload["source_pair"].get("checks", {}))
            and payload["source_pair"].get("failed_checks") == []
            and set(payload.get("source_pair_contract_checks", {}))
            == SOURCE_PAIR_CONTRACT_CHECKS
            and _all_true(payload.get("source_pair_contract_checks", {}))
        ),
        "material_stack_ready": payload["material"].get("ready") is True
        and set(payload["material"].get("checks", {})) == MATERIAL_CHECKS
        and _all_true(payload["material"].get("checks", {}))
        and payload["material"].get("failed_checks") == [],
        "exact_binary_material_ready": exact_binary.get("ready") is True
        and set(exact_binary.get("checks", {})) == EXACT_BINARY_CHECKS
        and _all_true(exact_binary.get("checks", {}))
        and exact_binary.get("gray_density_allowed") is False
        and exact_binary.get("rho_power") is None
        and exact_binary.get("design_solid_cells") == 375,
        "raw_artifact_and_recomputed_physics_ready": raw.get("ready") is True,
        "normalization_policy_exact": payload["normalization_policy"].get(
            "raw_fields_and_Q_are_unscaled"
        )
        is True
        and payload["normalization_policy"].get(
            "per_polarization_matching_forbidden"
        )
        is True,
        "runner_exists_and_matches": runner["path_is_absolute"]
        and runner["sha256_matches"],
        "material_contract_exists_and_matches": material_contract["path_is_absolute"]
        and material_contract["sha256_matches"],
        "recorded_repositories_clean": payload["provenance"].get(
            "repository_dirty_porcelain"
        )
        == ""
        and payload["provenance"]["fdtdx_source"].get("dirty_porcelain") == "",
        "optimizer_remains_forbidden": payload.get("optimizer_start_allowed") is False,
    }
    audit = {
        "report_path": str(resolved),
        "report_sha256": sha256(resolved),
        "expected_periods": period,
        "expected_polarization": polarization,
        "recorded_status": payload.get("status"),
        "recorded_ready": payload.get("ready"),
        "evaluation_failed_gates": evaluation.get("failed_gates"),
        "maximum_complex_E_NRMSE": evaluation["field_stationarity"][
            "maximum_complex_E_NRMSE"
        ],
        "previous_late_spatial_Q_NRMSE": evaluation["Q"][
            "previous_late_spatial_NRMSE"
        ],
        "Q_closed_phasor_symmetric_relative": evaluation["flux"][
            "Q_vs_closed_phasor_symmetric_relative"
        ],
        "Q_closed_td_symmetric_relative": evaluation["flux"][
            "Q_vs_closed_td_symmetric_relative"
        ],
        "repository_commit": payload["provenance"]["repository_commit"],
        "fdtdx_source": payload["provenance"]["fdtdx_source"],
        "runtime_lock": payload["provenance"]["runtime_lock"],
        "runner": runner,
        "material_contract": material_contract,
        "raw": raw,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "artifact_ready": _all_true(checks),
    }
    return payload, audit, snapshot


def _arrays_identical(
    cases: Mapping[int, Mapping[str, Mapping[str, Any] | None]], key: str
) -> bool:
    values: list[Any] = []
    for period in PERIODS:
        for polarization in POLARIZATIONS:
            snapshot = cases[period][polarization]
            if snapshot is None:
                return False
            values.append(snapshot[key])
    if not values:
        return False
    if isinstance(values[0], tuple):
        return all(
            isinstance(value, tuple)
            and len(value) == len(values[0])
            and all(
                np.array_equal(values[0][index], value[index])
                for index in range(len(values[0]))
            )
            for value in values[1:]
        )
    return all(np.array_equal(values[0], value) for value in values[1:])


def _nested_arrays_identical(
    cases: Mapping[int, Mapping[str, Mapping[str, Any] | None]], key: str
) -> bool:
    for material in ("au", "tairte4"):
        arrays: list[np.ndarray] = []
        for period in PERIODS:
            for polarization in POLARIZATIONS:
                snapshot = cases[period][polarization]
                if snapshot is None:
                    return False
                arrays.append(snapshot[key][material])
        if not all(np.array_equal(arrays[0], item) for item in arrays[1:]):
            return False
    return True


def compare_time_pair(
    coarse_period: int,
    fine_period: int,
    snapshots: Mapping[int, Mapping[str, Mapping[str, Any] | None]],
    payloads: Mapping[int, Mapping[str, Mapping[str, Any]]],
    source_pairs: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare one same-grid time pair for both polarizations."""

    if (coarse_period, fine_period) not in SUCCESSIVE_PAIRS:
        raise ValueError("only the declared successive time pairs may be compared")
    if any(
        snapshots[period][polarization] is None
        for period in (coarse_period, fine_period)
        for polarization in POLARIZATIONS
    ):
        return {
            "coarse_periods": coarse_period,
            "fine_periods": fine_period,
            "pass": False,
            "error": "one or more raw snapshots failed artifact validation",
            "checks": {},
        }

    source_details: dict[str, float] = {}
    for polarization in POLARIZATIONS:
        source_details[polarization] = relative_difference(
            source_pairs[coarse_period]["comparison"]["unscaled_incident_power_W"][
                polarization
            ],
            source_pairs[fine_period]["comparison"]["unscaled_incident_power_W"][
                polarization
            ],
        )
    source_details["mean"] = relative_difference(
        source_pairs[coarse_period]["comparison"]["mean_unscaled_incident_power_W"],
        source_pairs[fine_period]["comparison"]["mean_unscaled_incident_power_W"],
    )

    per_polarization: dict[str, Any] = {}
    for polarization in POLARIZATIONS:
        coarse = snapshots[coarse_period][polarization]
        fine = snapshots[fine_period][polarization]
        target_nrmse = weighted_complex_nrmse(
            fine["probe"], coarse["probe"], fine["probe_weights"]
        )
        material_field = {
            material: weighted_complex_nrmse(
                fine["fields_late"][material],
                coarse["fields_late"][material],
                fine["volumes"][material],
            )
            for material in ("au", "tairte4")
        }
        total_q = relative_difference(
            coarse["power_late"]["total_W"], fine["power_late"]["total_W"]
        )
        material_component_q: dict[str, Any] = {}
        flat_material_component: list[float] = []
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
            flat_material_component.extend(material_component_q[material].values())
        q_l2 = combined_weighted_nrmse(
            fine["q_late"], coarse["q_late"], fine["volumes"]
        )
        per_polarization[polarization] = {
            "total_Q_relative_change": total_q,
            "material_component_Q_relative_change": material_component_q,
            "material_component_Q_max_relative_change": max(flat_material_component),
            "complex_E_fixed_probe_NRMSE": target_nrmse,
            "material_region_complex_E_NRMSE": material_field,
            "material_region_complex_E_max_NRMSE": max(material_field.values()),
            "conservative_Q_volume_L2_NRMSE": q_l2,
        }

    closure_values = []
    for period in (coarse_period, fine_period):
        for polarization in POLARIZATIONS:
            flux = payloads[period][polarization]["evaluation"]["flux"]
            closure_values.extend(
                (
                    float(flux["Q_vs_closed_phasor_symmetric_relative"]),
                    float(flux["Q_vs_closed_td_symmetric_relative"]),
                )
            )
    fine_stationarity = max(
        float(
            payloads[fine_period][polarization]["evaluation"]["field_stationarity"][
                "maximum_complex_E_NRMSE"
            ]
        )
        for polarization in POLARIZATIONS
    )
    metrics = {
        "source_power_relative_change": max(source_details.values()),
        "q_closed_flux_relative": max(closure_values),
        "stationarity_complex_E_NRMSE": fine_stationarity,
        "total_Q_relative_change": max(
            item["total_Q_relative_change"] for item in per_polarization.values()
        ),
        "material_component_Q_max_relative_change": max(
            item["material_component_Q_max_relative_change"]
            for item in per_polarization.values()
        ),
        "complex_E_fixed_probe_NRMSE": max(
            item["complex_E_fixed_probe_NRMSE"]
            for item in per_polarization.values()
        ),
        "conservative_Q_volume_L2_NRMSE": max(
            item["conservative_Q_volume_L2_NRMSE"]
            for item in per_polarization.values()
        ),
    }
    evaluation = evaluate_pair(metrics)
    return {
        "coarse_periods": coarse_period,
        "fine_periods": fine_period,
        "same_grid_comparison": True,
        "fixed_probe_method": (
            "exact common physical [-4,+4] um x/y cells at z=0.250 um; "
            "component-specific Yee area weights; no interpolation needed"
        ),
        "Q_method": (
            "same physical control volumes with component-specific stored Yee "
            "dual volumes; no remap needed"
        ),
        "source_power_relative_change": source_details,
        "per_polarization": per_polarization,
        "metrics": metrics,
        "limits": OPTICAL_PAIR_GATES,
        "checks": evaluation["checks"],
        "pass": evaluation["pass"],
    }


def settling_selection_gates(
    case_ready: Mapping[int, Mapping[str, bool]],
    case_failed_gates: Mapping[int, Mapping[str, list[str]]],
    pair_pass: Mapping[tuple[int, int], bool],
) -> dict[str, bool]:
    """Apply the explicit first-settled-level plus confirmation policy."""

    allowed_coarse_failures = {
        "complex_field_stationarity",
        "Q_previous_late_spatial_change",
    }
    coarse_failed_as_settling = all(
        case_ready[16][polarization] is False
        and "complex_field_stationarity" in case_failed_gates[16][polarization]
        and set(case_failed_gates[16][polarization]).issubset(allowed_coarse_failures)
        for polarization in POLARIZATIONS
    )
    return {
        "16_period_coarse_rejected_only_for_settling": coarse_failed_as_settling,
        "24_period_selected_cases_internally_ready": all(
            case_ready[24][polarization] is True for polarization in POLARIZATIONS
        ),
        "32_period_confirmation_cases_internally_ready": all(
            case_ready[32][polarization] is True for polarization in POLARIZATIONS
        ),
        "16_to_24_cross_comparison_passes": pair_pass[(16, 24)] is True,
        "24_to_32_cross_comparison_passes": pair_pass[(24, 32)] is True,
    }


def build_time_settling_certificate(
    root: Path,
    contract_sha256s: Mapping[int, str],
    source_pair_sha256s: Mapping[int, str],
) -> dict[str, Any]:
    campaign_root = root.expanduser().resolve()
    if not root.expanduser().is_absolute() or not campaign_root.is_dir():
        raise RuntimeError("campaign root must be an existing absolute directory")
    if set(contract_sha256s) != set(PERIODS):
        raise ValueError(f"contract SHA mapping must contain exactly {PERIODS}")
    if set(source_pair_sha256s) != set(PERIODS):
        raise ValueError(f"source-pair SHA mapping must contain exactly {PERIODS}")

    contracts: dict[int, Any] = {}
    source_pairs: dict[int, Any] = {}
    payloads: dict[int, dict[str, Any]] = {}
    cases: dict[int, dict[str, Any]] = {}
    snapshots: dict[int, dict[str, Any]] = {}
    for period in PERIODS:
        spec = expected_case(period)
        contract_path = (campaign_root / "contracts" / f"l500_anchor_t{period}.json").resolve()
        loaded_spec, contract_payload, contract_audit = load_case_contract(
            contract_path, contract_sha256s[period]
        )
        if loaded_spec != spec:
            raise RuntimeError(f"t{period} contract is not the exact expected time case")
        contracts[period] = {
            "payload": contract_payload,
            "audit": contract_audit,
            "spec": spec,
        }

        pair_path = (
            campaign_root
            / f"source_pair_t{period}"
            / "FDTDX_FRESH_SOURCE_ONLY_PAIR.json"
        ).resolve()
        try:
            pair_payload, pair_audit = validate_source_pair(
                pair_path, source_pair_sha256s[period], expected_case=spec
            )
        except Exception as error:
            pair_payload = {}
            pair_audit = {
                "ready": False,
                "error": repr(error),
                "checks": {},
                "failed_checks": ["source_pair_revalidation_exception"],
            }
        source_pairs[period] = {
            "path": str(pair_path),
            "payload": pair_payload,
            "audit": pair_audit,
        }

        payloads[period] = {}
        cases[period] = {}
        snapshots[period] = {}
        for polarization in POLARIZATIONS:
            report = (
                campaign_root
                / f"l500_t{period}_{polarization}"
                / "FDTDX_FRESH_EXACT_BINARY_PILOT.json"
            ).resolve()
            payload, audit, snapshot = _material_case_audit(
                report,
                campaign_root,
                period,
                polarization,
                spec,
                contract_path,
                contract_sha256s[period],
                pair_path,
                source_pair_sha256s[period],
            )
            payloads[period][polarization] = payload
            cases[period][polarization] = audit
            snapshots[period][polarization] = snapshot

    pair_results = {
        f"{coarse}_to_{fine}": compare_time_pair(
            coarse,
            fine,
            snapshots,
            payloads,
            {period: source_pairs[period]["payload"] for period in PERIODS},
        )
        for coarse, fine in SUCCESSIVE_PAIRS
    }
    case_ready = {
        period: {
            polarization: payloads[period][polarization]["evaluation"]["ready"]
            for polarization in POLARIZATIONS
        }
        for period in PERIODS
    }
    case_failed = {
        period: {
            polarization: payloads[period][polarization]["evaluation"]["failed_gates"]
            for polarization in POLARIZATIONS
        }
        for period in PERIODS
    }
    selection_gates = settling_selection_gates(
        case_ready,
        case_failed,
        {
            pair: pair_results[f"{pair[0]}_to_{pair[1]}"]["pass"]
            for pair in SUCCESSIVE_PAIRS
        },
    )

    flat_payloads = [
        payloads[period][polarization]
        for period in PERIODS
        for polarization in POLARIZATIONS
    ]
    flat_cases = [
        cases[period][polarization]
        for period in PERIODS
        for polarization in POLARIZATIONS
    ]
    repository_commits = {
        payload["provenance"]["repository_commit"] for payload in flat_payloads
    }
    source_pair_commits = {
        item["payload"].get("provenance", {}).get("certificate_repository_commit")
        for item in source_pairs.values()
    }
    gates = {
        "all_canonical_case_contracts_revalidated": all(
            contracts[period]["audit"]["ready"] is True for period in PERIODS
        ),
        "all_source_pairs_revalidated": all(
            source_pairs[period]["audit"].get("ready") is True for period in PERIODS
        ),
        "all_material_artifacts_and_recomputed_physics_ready": all(
            case["artifact_ready"] is True for case in flat_cases
        ),
        "only_total_periods_change_across_contracts": all(
            contracts[period]["spec"].mesh == contracts[16]["spec"].mesh
            and contracts[period]["spec"].pml_alpha_scale
            == contracts[16]["spec"].pml_alpha_scale
            and contracts[period]["spec"].pml_target_reflection
            == contracts[16]["spec"].pml_target_reflection
            and contracts[period]["spec"].time.window_periods == WINDOW_PERIODS
            and contracts[period]["spec"].time.courant_factor == COURANT_FACTOR
            and contracts[period]["spec"].time.total_periods == period
            for period in PERIODS
        ),
        "grid_edges_identical_for_exact_same_cell_comparison": _arrays_identical(
            snapshots, "grid_edges"
        ),
        "Yee_dual_volumes_identical": _nested_arrays_identical(snapshots, "volumes"),
        "exact_L500_masks_identical": _arrays_identical(snapshots, "design_mask")
        and _arrays_identical(snapshots, "solver_mask"),
        "material_raw_array_schema_identical": len(
            {
                json.dumps(case["raw"]["declared_arrays"], sort_keys=True)
                for case in flat_cases
            }
        )
        == 1,
        "placement_identical": len({
            json.dumps(payload["placement"], sort_keys=True)
            for payload in flat_payloads
        }) == 1,
        "same_polarization_source_contract_identical_across_time": all(
            len({
                json.dumps(payloads[period][polarization]["source_contract"], sort_keys=True)
                for period in PERIODS
            }) == 1
            for polarization in POLARIZATIONS
        ),
        "realized_time_step_identical_and_step_count_scales_with_periods": (
            len({
                payload["time_contract"]["time_step_s"]
                for payload in flat_payloads
            }) == 1
            and len({
                payloads[period][polarization]["time_contract"]["time_steps_total"] / period
                for period in PERIODS
                for polarization in POLARIZATIONS
            }) == 1
        ),
        "realized_material_response_identical": len({
            json.dumps(payload["material"]["realized_material_response"], sort_keys=True)
            for payload in flat_payloads
        }) == 1,
        "material_repository_commit_identical": len(repository_commits) == 1,
        "source_and_material_repository_commit_identical": len(source_pair_commits) == 1
        and source_pair_commits == repository_commits,
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
        "runner_hash_identical": len(
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
        "settling_selection_policy_passes": all(selection_gates.values()),
        "optimizer_remains_forbidden": all(
            payload.get("optimizer_start_allowed") is False for payload in flat_payloads
        ),
    }
    ready = all(gates.values())
    return {
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "scope": (
            "time settling of one exact-binary L500 optical reference on one "
            "fixed FDTDX spatial grid"
        ),
        "campaign_root": str(campaign_root),
        "contracts": {
            str(period): {
                "case_contract_sha256": contracts[period]["payload"][
                    "case_contract_sha256"
                ],
                "file_audit": contracts[period]["audit"],
            }
            for period in PERIODS
        },
        "source_pairs": {
            str(period): {
                "path": source_pairs[period]["path"],
                "audit": source_pairs[period]["audit"],
                "unscaled_incident_power_W": source_pairs[period]["payload"].get(
                    "comparison", {}
                ).get("unscaled_incident_power_W"),
                "mean_unscaled_incident_power_W": source_pairs[period]["payload"].get(
                    "comparison", {}
                ).get("mean_unscaled_incident_power_W"),
            }
            for period in PERIODS
        },
        "cases": {str(period): cases[period] for period in PERIODS},
        "successive_comparisons": pair_results,
        "selection": {
            "selected_total_periods": SELECTED_PERIODS if ready else None,
            "confirmation_total_periods": CONFIRMATION_PERIODS if ready else None,
            "policy": (
                "select the first internally settled level only when both declared "
                "successive cross-comparisons pass and the next level independently "
                "passes its internal gates"
            ),
            "gates": selection_gates,
            "failed_gates": [
                name for name, passed in selection_gates.items() if not passed
            ],
        },
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "is_mesh_certificate": False,
        "optimizer_start_allowed": False,
        "next_allowed_step": (
            "run the exact same L500 reference at 24 periods on the Courant "
            "ladder [0.5, 0.375, 0.25]; do not optimize"
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


def write_time_settling_certificate(
    root: Path,
    contract_sha256s: Mapping[int, str],
    source_pair_sha256s: Mapping[int, str],
    output_directory: Path,
) -> dict[str, Any]:
    output = output_directory.expanduser().resolve()
    if not output_directory.expanduser().is_absolute() or not output.is_dir():
        raise RuntimeError("output directory must be an existing absolute directory")
    if any(output.iterdir()):
        raise RuntimeError("output directory must be empty before certification")
    result = build_time_settling_certificate(
        root, contract_sha256s, source_pair_sha256s
    )
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
    result["failed_gates"] = [
        name for name, passed in result["gates"].items() if not passed
    ]
    if not result["ready"]:
        result["selection"]["selected_total_periods"] = None
        result["selection"]["confirmation_total_periods"] = None
    _atomic_json(output / CERTIFICATE_NAME, result)
    return result


def _period_sha(values: list[str], label: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for item in values:
        try:
            period_text, digest = item.split("=", 1)
            period = int(period_text)
        except ValueError as error:
            raise ValueError(f"{label} entries must use PERIOD=SHA256") from error
        if period in result:
            raise ValueError(f"duplicate {label} period {period}")
        normalized = digest.strip().lower()
        if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
            raise ValueError(f"{label} t{period} is not a lowercase SHA256")
        result[period] = normalized
    if set(result) != set(PERIODS):
        raise ValueError(f"{label} entries must contain exactly {PERIODS}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract-sha256", action="append", default=[], metavar="PERIOD=SHA256")
    parser.add_argument("--source-pair-sha256", action="append", default=[], metavar="PERIOD=SHA256")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = write_time_settling_certificate(
            args.root,
            _period_sha(args.contract_sha256, "contract"),
            _period_sha(args.source_pair_sha256, "source-pair"),
            args.output_dir,
        )
    except Exception as error:
        failure = {
            "status": "BLOCKED_FDTDX_FRESH_L500_TIME_SETTLING_EXCEPTION",
            "ready": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
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
        "selected_total_periods": result["selection"]["selected_total_periods"],
        "confirmation_total_periods": result["selection"][
            "confirmation_total_periods"
        ],
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
