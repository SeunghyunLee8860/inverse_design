"""One-case exact-binary material pilot on the validated fresh FDTDX anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_4um_model as optical_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_dependency import (
    configured_source,
    require_source,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
    REFERENCE_NAMES,
    mesh_audit,
    reference_mask,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_material import (
    arrays_for_exact_binary,
    readback_exact_binary,
    solver_mask,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    ANCHOR_CASE,
    FreshCaseSpec,
    case_contract,
    realized_time_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_mesh import (
    build_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_metrics import (
    electric_yee_dual_volumes,
    weighted_complex_nrmse,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_only import (
    extract_detector_fields,
    resolve_case_input,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_pair import (
    PAIR_STATUS,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_runtime_preflight import (
    load_runtime_lock,
)


ANCHOR_SPEC = ANCHOR_CASE.mesh
JSON_NAME = "FDTDX_FRESH_EXACT_BINARY_PILOT.json"
RAW_NAME = "FDTDX_FRESH_EXACT_BINARY_PILOT_FIELDS.npz"
STATUS_READY = "VALIDATED_FDTDX_FRESH_EXACT_BINARY_PILOT_CASE"
STATUS_BLOCKED = "BLOCKED_FDTDX_FRESH_EXACT_BINARY_PILOT_CASE"
SOURCE_PAIR_SHA256_LENGTH = 64
STATIONARITY_LIMIT = 5.0e-3
Q_WINDOW_CHANGE_LIMIT = 5.0e-3
Q_SPATIAL_CHANGE_LIMIT = 5.0e-3
Q_CLOSED_FLUX_RELATIVE_LIMIT = 2.0e-2
ABSORBED_FRACTION_MAXIMUM = 1.02
MATERIAL_RELATIVE_READBACK_LIMIT = 1.0e-5
EPS0_F_PER_M = 8.854_187_812_8e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_difference(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(
        abs(float(left)), abs(float(right)), np.finfo(float).tiny
    )


def absorption_power_density(
    field: np.ndarray,
    epsilon_imaginary: float | np.ndarray,
    physical_prefactor: float,
    occupancy_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Return component-resolved passive Q on electric Yee sample volumes."""

    value = np.asarray(field)
    if value.ndim != 4 or value.shape[0] != 3:
        raise ValueError("field must have shape (3,x,y,z)")
    loss = np.asarray(epsilon_imaginary, dtype=np.float64)
    if loss.ndim == 0:
        loss = np.full((3, 1, 1, 1), float(loss), dtype=np.float64)
    elif loss.shape == (3,):
        loss = loss[:, None, None, None]
    if loss.shape != (3, 1, 1, 1):
        raise ValueError("epsilon imaginary part must be scalar or length three")
    if np.any(~np.isfinite(loss)) or np.any(loss < 0.0):
        raise ValueError("epsilon imaginary part must be finite and passive")
    occupancy: float | np.ndarray = 1.0
    if occupancy_xy is not None:
        occupancy_value = np.asarray(occupancy_xy)
        if occupancy_value.shape != value.shape[1:3]:
            raise ValueError("occupancy must match the field x/y shape")
        if occupancy_value.dtype.kind not in "biu" or not np.all(
            (occupancy_value == 0) | (occupancy_value == 1)
        ):
            raise ValueError("occupancy must contain exact integer/bool 0/1 values")
        occupancy = occupancy_value[None, :, :, None]
    return (
        float(physical_prefactor)
        * loss
        * occupancy
        * np.abs(value).astype(np.float64) ** 2
    )


def component_power(q: np.ndarray, volumes: np.ndarray) -> dict[str, Any]:
    density = np.asarray(q, dtype=np.float64)
    weights = np.asarray(volumes, dtype=np.float64)
    if density.shape != weights.shape or density.ndim != 4 or density.shape[0] != 3:
        raise ValueError("Q and Yee volumes must have identical (3,x,y,z) shapes")
    components = {
        axis: float(np.sum(density[index] * weights[index]))
        for index, axis in enumerate(("x", "y", "z"))
    }
    return {"component_W": components, "total_W": float(sum(components.values()))}


def combined_weighted_nrmse(
    late: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
    volumes: dict[str, np.ndarray],
) -> float:
    numerator = 0.0
    denominator = 0.0
    if set(late) != set(previous) or set(late) != set(volumes):
        raise ValueError("late, previous, and volume material keys must match")
    for material in sorted(late):
        current = np.asarray(late[material], dtype=np.float64)
        prior = np.asarray(previous[material], dtype=np.float64)
        weight = np.asarray(volumes[material], dtype=np.float64)
        if current.shape != prior.shape or current.shape != weight.shape:
            raise ValueError("Q fields and Yee volumes must have identical shapes")
        numerator += float(np.sum((current - prior) ** 2 * weight))
        denominator += float(np.sum(current**2 * weight))
    return math.sqrt(numerator / max(denominator, np.finfo(float).tiny))


def _file_audit(path_value: str, expected_sha256: str) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    absolute = path.is_absolute()
    resolved = path.resolve()
    exists = resolved.is_file()
    actual = sha256(resolved) if exists else None
    return {
        "path": str(resolved),
        "path_is_absolute": absolute,
        "exists": exists,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "sha256_matches": exists and actual == expected_sha256,
    }


def validate_source_pair(
    path: Path,
    expected_sha256: str,
    expected_case: FreshCaseSpec = ANCHOR_CASE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate the certificate, artifacts, and exact numerical case."""

    if not isinstance(expected_case, FreshCaseSpec):
        raise TypeError("expected_case must be a FreshCaseSpec")

    resolved = path.expanduser().resolve()
    normalized_sha = expected_sha256.strip().lower()
    certificate_exists = resolved.is_file()
    actual_sha = sha256(resolved) if certificate_exists else None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("source pair certificate must contain one JSON object")

    case_artifacts: dict[str, Any] = {}
    for polarization in ("Ea", "Eb"):
        case = payload["cases"][polarization]
        case_artifacts[polarization] = {
            "report": _file_audit(case["report_path"], case["report_sha256"]),
            "raw": _file_audit(
                case["raw"]["path"], case["raw"]["actual_sha256"]
            ),
        }
    generator = _file_audit(
        payload["provenance"]["certificate_generator_path"],
        payload["provenance"]["certificate_generator_sha256"],
    )
    normalization = payload["common_normalization"]
    numeric = (
        normalization["reporting_target_incident_power_W"],
        normalization["common_power_scale"],
        normalization["common_field_amplitude_scale"],
        payload["comparison"]["mean_unscaled_incident_power_W"],
    )
    certificate_gates = payload["gates"]
    expected_contract = case_contract(expected_case)
    source_contracts = payload["source_case_contracts"]
    expected_time_request = {
        "total_periods": expected_case.time.total_periods,
        "window_periods": expected_case.time.window_periods,
        "source_startup_periods": expected_case.time.source_startup_periods,
        "courant_factor": expected_case.time.courant_factor,
    }
    recorded_time = source_contracts["time_contract"]
    checks = {
        "expected_sha256_is_lowercase_hex": len(normalized_sha)
        == SOURCE_PAIR_SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in normalized_sha),
        "certificate_exists": certificate_exists,
        "certificate_sha256_matches": actual_sha == normalized_sha,
        "certificate_status_and_ready": payload.get("status") == PAIR_STATUS
        and payload.get("ready") is True
        and payload.get("failed_gates") == [],
        "certificate_gates_all_true": bool(certificate_gates)
        and all(value is True for value in certificate_gates.values()),
        "per_polarization_scaling_forbidden": payload["normalization_policy"].get(
            "per_polarization_power_matching_forbidden"
        )
        is True,
        "normalization_finite_positive": all(
            math.isfinite(float(value)) and float(value) > 0.0 for value in numeric
        ),
        "numerical_case_contract_exact": source_contracts.get(
            "numerical_case_contract"
        )
        == expected_contract,
        "mesh_contract_exact": source_contracts["mesh"]
        == mesh_audit(expected_case.mesh),
        "time_request_exact": all(
            recorded_time.get(name) == value
            for name, value in expected_time_request.items()
        ),
        "pml_contract_exact": source_contracts["pml_face_parameters"]
        == expected_contract["resolved_pml_face_parameters"],
        "case_reports_exist_and_match": all(
            item["report"]["path_is_absolute"]
            and item["report"]["sha256_matches"]
            for item in case_artifacts.values()
        ),
        "case_raw_files_exist_and_match": all(
            item["raw"]["path_is_absolute"] and item["raw"]["sha256_matches"]
            for item in case_artifacts.values()
        ),
        "certificate_generator_exists_and_matches": generator["path_is_absolute"]
        and generator["sha256_matches"],
    }
    audit = {
        "path": str(resolved),
        "expected_sha256": normalized_sha,
        "actual_sha256": actual_sha,
        "case_artifacts": case_artifacts,
        "certificate_generator": generator,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
    }
    return payload, audit


def _unique_summary(value: np.ndarray, maximum: int = 8) -> list[float] | str:
    unique = np.unique(np.asarray(value))
    if unique.size > maximum:
        return f"{unique.size} unique values"
    return [float(item) for item in unique]


def material_stack_audit(
    model: dict[str, Any],
    arrays: Any,
    mask: Any,
    spec: MeshSpec = ANCHOR_SPEC,
) -> dict[str, Any]:
    """Read back all physical material regions used by the pilot."""

    exact_au = readback_exact_binary(model, arrays, mask, spec)
    inv = arrays.inv_permittivities
    checks: dict[str, bool] = {"exact_binary_au_readback": exact_au["ready"]}
    inverse_readback: dict[str, Any] = {}
    for label, object_name, epsilon_name in (
        ("silicon", "fixed_silicon_substrate", "silicon"),
        ("sio2", "fixed_285nm_sio2", "sio2"),
    ):
        region = np.asarray(inv[(slice(None), *model["slices"][object_name])])
        expected = 1.0 / float(model["epsilon"][epsilon_name].real)
        error = float(np.max(np.abs(region.astype(np.float64) - expected)))
        relative = error / abs(expected)
        checks[f"{label}_inverse_permittivity_readback"] = (
            relative <= MATERIAL_RELATIVE_READBACK_LIMIT
        )
        inverse_readback[label] = {
            "expected": expected,
            "observed_unique": _unique_summary(region),
            "maximum_relative_error": relative,
        }

    ta_slice = model["slices"]["fixed_tairte4"]
    ta_coefficients: dict[str, Any] = {}
    for array_name, coefficient_index in (
        ("dispersive_c1", 0),
        ("dispersive_c2", 1),
        ("dispersive_c3", 2),
    ):
        observed = np.asarray(getattr(arrays, array_name)[(0, slice(None), *ta_slice)])
        component_checks = []
        component_summary = []
        for component, axis in enumerate(("b", "a", "c")):
            expected = np.asarray(
                model["coefficients"][axis][coefficient_index],
                dtype=observed.dtype,
            )
            matches = np.array_equal(
                observed[component], np.full_like(observed[component], expected)
            )
            component_checks.append(matches)
            component_summary.append(
                {
                    "component": ("Ex", "Ey", "Ez")[component],
                    "crystal_axis": axis,
                    "expected": float(expected),
                    "observed_unique": _unique_summary(observed[component]),
                    "exact": matches,
                }
            )
        checks[f"tairte4_{array_name}_exact"] = all(component_checks)
        ta_coefficients[array_name] = component_summary
    ta_inv = np.asarray(inv[(slice(None), *ta_slice)])
    checks["tairte4_epsilon_infinity_inverse_is_one"] = np.array_equal(
        ta_inv, np.ones_like(ta_inv)
    )

    susceptibility: dict[str, Any] = {}
    for name, value in model["discrete_susceptibility"].items():
        epsilon = 1.0 + value
        target_name = "tairte4" if name in ("a", "b", "c") else "au"
        target = (
            model["epsilon"][target_name][name]
            if target_name == "tairte4"
            else model["epsilon"][target_name]
        )
        relative_error = abs(epsilon - target) / abs(target)
        passive = math.isfinite(epsilon.real) and math.isfinite(epsilon.imag) and epsilon.imag > 0.0
        checks[f"{name}_realized_epsilon_passive"] = passive
        checks[f"{name}_realized_epsilon_matches_target"] = (
            relative_error <= MATERIAL_RELATIVE_READBACK_LIMIT
        )
        susceptibility[name] = {
            "realized_epsilon": [float(epsilon.real), float(epsilon.imag)],
            "target_epsilon": [float(target.real), float(target.imag)],
            "relative_error": float(relative_error),
            "passive": passive,
        }
    return {
        "absorption_loss_basis": model["absorption_loss_basis"],
        "exact_binary_au": exact_au,
        "inverse_permittivity_readback": inverse_readback,
        "tairte4_inverse_permittivity_unique": _unique_summary(ta_inv),
        "tairte4_coefficient_readback": ta_coefficients,
        "realized_material_response": susceptibility,
        "finite_dt_ADE_coefficients": {
            name: [float(item) for item in values]
            for name, values in model["coefficients"].items()
        },
        "finite_dt_ADE_fits": model["fits"],
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
    }


def _power_evaluation(
    model: dict[str, Any],
    output: Any,
    mask: np.ndarray,
    source_pair: dict[str, Any],
    spec: MeshSpec = ANCHOR_SPEC,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    states = output.detector_states
    fields = extract_detector_fields(states)
    volumes = {
        "au": electric_yee_dual_volumes(
            model["grid"], model["slices"]["au_design"]
        ),
        "tairte4": electric_yee_dual_volumes(
            model["grid"], model["slices"]["fixed_tairte4"]
        ),
    }
    eta0 = float(model["fdtdx"].constants.eta0)
    prefactor = 0.5 * float(model["omega_rad_s"]) * EPS0_F_PER_M * eta0**2
    ta_loss = np.asarray(
        [model["discrete_susceptibility"][axis].imag for axis in ("b", "a", "c")],
        dtype=np.float64,
    )
    expanded_mask = solver_mask(mask, spec)
    q: dict[str, dict[str, np.ndarray]] = {"previous": {}, "late": {}}
    for window in ("previous", "late"):
        q[window]["au"] = absorption_power_density(
            fields[f"au_{window}"],
            model["discrete_susceptibility"]["au"].imag,
            prefactor,
            expanded_mask,
        )
        q[window]["tairte4"] = absorption_power_density(
            fields[f"tairte4_{window}"], ta_loss, prefactor
        )

    powers: dict[str, Any] = {}
    for window in ("previous", "late"):
        by_material = {
            material: component_power(q[window][material], volumes[material])
            for material in ("au", "tairte4")
        }
        powers[window] = {
            "by_material": by_material,
            "total_W": float(sum(item["total_W"] for item in by_material.values())),
        }
    late_total = powers["late"]["total_W"]
    previous_total = powers["previous"]["total_W"]
    field_stationarity = {
        "au_complex_E_NRMSE": weighted_complex_nrmse(
            fields["au_late"], fields["au_previous"], volumes["au"]
        ),
        "tairte4_complex_E_NRMSE": weighted_complex_nrmse(
            fields["tairte4_late"],
            fields["tairte4_previous"],
            volumes["tairte4"],
        ),
    }
    field_stationarity["maximum_complex_E_NRMSE"] = max(
        field_stationarity.values()
    )
    q_spatial_change = combined_weighted_nrmse(q["late"], q["previous"], volumes)
    q_total_change = relative_difference(late_total, previous_total)

    closed_phasor = float(
        eta0
        * np.asarray(
            model["placed"]["material_flux"].compute_net_flux(
                states["material_flux"]
            )
        )[0]
    )
    closed_td = float(eta0 * np.mean(fields["closed_td"][:, 0]))
    source_side_net = float(
        eta0
        * np.asarray(
            model["placed"]["incident_plane"].compute_poynting_flux(
                states["incident_plane"]
            )
        )[0]
    )
    source_reference = float(
        source_pair["comparison"]["mean_unscaled_incident_power_W"]
    )
    common_power_scale = float(
        source_pair["common_normalization"]["common_power_scale"]
    )
    flux = {
        "source_reference_all_air_unscaled_W": source_reference,
        "source_side_material_case_net_downward_W": source_side_net,
        "source_side_plane_is_not_incident_calibration": True,
        "closed_box_inward_phasor_signed_W": closed_phasor,
        "closed_box_inward_td_mean_signed_W": closed_td,
        "Q_vs_closed_phasor_symmetric_relative": relative_difference(
            late_total, closed_phasor
        ),
        "Q_vs_closed_td_symmetric_relative": relative_difference(
            late_total, closed_td
        ),
        "closed_td_vs_phasor_symmetric_relative": relative_difference(
            closed_td, closed_phasor
        ),
        "absorbed_fraction_of_all_air_source": late_total / source_reference,
        "normalization": "eta0 times normalized FDTDX flux; Q uses eta0 squared",
    }
    finite = all(np.all(np.isfinite(value)) for value in fields.values()) and all(
        np.all(np.isfinite(value))
        for window in q.values()
        for value in window.values()
    )
    nonnegative_q = all(
        bool(np.min(value) >= 0.0)
        for window in q.values()
        for value in window.values()
    )
    au_late_power = powers["late"]["by_material"]["au"]["total_W"]
    reference_is_empty = int(np.count_nonzero(expanded_mask)) == 0
    gates = {
        "all_raw_detector_and_Q_values_finite": bool(finite),
        "complex_field_stationarity": field_stationarity[
            "maximum_complex_E_NRMSE"
        ]
        <= STATIONARITY_LIMIT,
        "Q_previous_late_total_change": q_total_change <= Q_WINDOW_CHANGE_LIMIT,
        "Q_previous_late_spatial_change": q_spatial_change
        <= Q_SPATIAL_CHANGE_LIMIT,
        "Q_nonnegative": nonnegative_q,
        "Q_total_finite_positive": math.isfinite(late_total) and late_total > 0.0,
        "closed_phasor_finite_positive": math.isfinite(closed_phasor)
        and closed_phasor > 0.0,
        "closed_td_finite_positive": math.isfinite(closed_td) and closed_td > 0.0,
        "Q_closed_phasor_closure": flux["Q_vs_closed_phasor_symmetric_relative"]
        <= Q_CLOSED_FLUX_RELATIVE_LIMIT,
        "Q_closed_td_closure": flux["Q_vs_closed_td_symmetric_relative"]
        <= Q_CLOSED_FLUX_RELATIVE_LIMIT,
        "closed_td_phasor_agreement": flux[
            "closed_td_vs_phasor_symmetric_relative"
        ]
        <= Q_CLOSED_FLUX_RELATIVE_LIMIT,
        "absorbed_fraction_physical": math.isfinite(
            flux["absorbed_fraction_of_all_air_source"]
        )
        and 0.0 < flux["absorbed_fraction_of_all_air_source"]
        <= ABSORBED_FRACTION_MAXIMUM,
        "source_side_net_flux_finite": math.isfinite(source_side_net),
        "empty_has_exact_zero_Au_Q_or_nonempty_has_positive_Au_Q": (
            au_late_power == 0.0 if reference_is_empty else au_late_power > 0.0
        ),
    }
    common_scaled = {
        "late_total_Q_W": late_total * common_power_scale,
        "late_Au_Q_W": au_late_power * common_power_scale,
        "late_TaIrTe4_Q_W": powers["late"]["by_material"]["tairte4"][
            "total_W"
        ]
        * common_power_scale,
        "closed_phasor_W": closed_phasor * common_power_scale,
        "closed_td_W": closed_td * common_power_scale,
    }
    raw = dict(fields)
    raw.update(
        design_mask=np.asarray(mask, dtype=np.uint8),
        solver_mask=expanded_mask,
        q_au_previous_W_m3=q["previous"]["au"],
        q_au_late_W_m3=q["late"]["au"],
        q_tairte4_previous_W_m3=q["previous"]["tairte4"],
        q_tairte4_late_W_m3=q["late"]["tairte4"],
        electric_dual_volume_au_m3=volumes["au"],
        electric_dual_volume_tairte4_m3=volumes["tairte4"],
        grid_x_edges_m=np.asarray(model["grid"].edges(0)),
        grid_y_edges_m=np.asarray(model["grid"].edges(1)),
        grid_z_edges_m=np.asarray(model["grid"].edges(2)),
    )
    evaluation = {
        "finite": bool(finite),
        "field_stationarity": field_stationarity,
        "Q": {
            "physical_prefactor": prefactor,
            "formula": "0.5*omega*eps0*eta0^2*Im(realized discrete susceptibility)*abs(E_phasor)^2",
            "previous": powers["previous"],
            "late": powers["late"],
            "previous_late_total_relative_change": q_total_change,
            "previous_late_spatial_NRMSE": q_spatial_change,
        },
        "flux": flux,
        "common_285uW_reporting": common_scaled,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "ready": all(gates.values()),
    }
    return evaluation, raw


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _output_directory(value: Path) -> Path:
    path = value.expanduser().resolve()
    if not path.is_absolute() or not path.is_dir():
        raise RuntimeError("output directory must be an existing absolute directory")
    if any(path.iterdir()):
        raise RuntimeError("output directory must be empty before exact-binary pilot")
    return path


def run(
    output_directory: Path,
    source: Path,
    source_pair_path: Path,
    source_pair_sha256: str,
    reference: str,
    polarization: str,
    case_spec: FreshCaseSpec = ANCHOR_CASE,
    case_file_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = _output_directory(output_directory)
    if not isinstance(case_spec, FreshCaseSpec):
        raise TypeError("case_spec must be a FreshCaseSpec")
    numerical_case = case_contract(case_spec)
    source_audit = require_source(source)
    source_pair, pair_audit = validate_source_pair(
        source_pair_path, source_pair_sha256, case_spec
    )
    pair_checks = pair_audit["checks"]
    pair_checks["fdtdx_source_matches_current"] = (
        source_pair["source_case_contracts"]["fdtdx_source"]
        == source_audit["actual"]
    )
    pair_checks["runtime_lock_matches_current"] = (
        source_pair["source_case_contracts"]["runtime_lock"]
        == load_runtime_lock()
    )
    pair_audit["failed_checks"] = [
        name for name, passed in pair_checks.items() if not passed
    ]
    pair_audit["ready"] = all(pair_checks.values())
    if not pair_audit["ready"]:
        raise RuntimeError(f"source pair revalidation failed: {pair_audit}")

    model = build_model(
        case_spec.mesh,
        polarization,
        total_periods=case_spec.time.total_periods,
        window_periods=case_spec.time.window_periods,
        courant_factor=case_spec.time.courant_factor,
        alpha_scale=case_spec.pml_alpha_scale,
        target_reflection=case_spec.pml_target_reflection,
        include_adjoint_source=False,
        air_only_source_calibration=False,
    )
    current_time = realized_time_contract(case_spec, model)
    contract_checks = {
        "numerical_case_matches_source_pair": numerical_case
        == source_pair["source_case_contracts"]["numerical_case_contract"],
        "mesh_matches_source_pair": model["fresh_mesh_audit"]
        == source_pair["source_case_contracts"]["mesh"],
        "time_matches_source_pair": current_time
        == source_pair["source_case_contracts"]["time_contract"],
        "pml_matches_source_pair": model["pml_face_parameters"]
        == source_pair["source_case_contracts"]["pml_face_parameters"],
        "placement_matches_source_pair": model["placement"]
        == source_pair["source_case_contracts"]["placement"],
        "source_matches_polarization_case": model["source_contract"]
        == source_pair["source_case_contracts"]["source_contracts"][polarization],
    }
    if not all(contract_checks.values()):
        raise RuntimeError(f"pilot/source contract mismatch: {contract_checks}")

    mask = np.asarray(reference_mask(reference), dtype=np.uint8)
    arrays = arrays_for_exact_binary(model, mask, case_spec.mesh)
    material = material_stack_audit(model, arrays, mask, case_spec.mesh)
    if not material["ready"]:
        raise RuntimeError(f"material readback failed before FDTD: {material}")

    started = time.perf_counter()
    _, fdtd_output = model["fdtdx"].run_fdtd(
        arrays,
        model["placed"],
        model["config"],
        model["key"],
        show_progress=False,
    )
    marker = fdtd_output.detector_states["target_field"]["phasor"]
    model["jax"].block_until_ready(marker)
    solve_runtime = time.perf_counter() - started
    evaluation, raw_arrays = _power_evaluation(
        model, fdtd_output, mask, source_pair, case_spec.mesh
    )

    raw_path = output / RAW_NAME
    _atomic_npz(raw_path, raw_arrays)
    repository = Path(__file__).resolve().parents[3]
    provenance = {
        "repository_commit": _git(repository, "rev-parse", "HEAD"),
        "repository_dirty_porcelain": _git(
            repository, "status", "--porcelain", "--untracked-files=all"
        ),
        "fdtdx_source": source_audit["actual"],
        "runtime_lock": load_runtime_lock(),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "material_contract_path": str(optical_model.MATERIAL_JSON.resolve()),
        "material_contract_sha256": sha256(optical_model.MATERIAL_JSON),
    }
    pre_solve_ready = (
        pair_audit["ready"]
        and all(contract_checks.values())
        and material["ready"]
        and provenance["repository_dirty_porcelain"] == ""
    )
    ready = pre_solve_ready and evaluation["ready"]
    payload = {
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "scope": "one fixed exact-binary optical material pilot; no thermal/electrical/adjoint/optimizer",
        "reference": reference,
        "polarization": polarization,
        "numerical_case_contract": numerical_case,
        "numerical_case_file_audit": case_file_audit,
        "mesh": model["fresh_mesh_audit"],
        "time_contract": current_time,
        "source_contract": model["source_contract"],
        "pml_face_parameters": model["pml_face_parameters"],
        "placement": model["placement"],
        "source_pair": pair_audit,
        "source_pair_contract_checks": contract_checks,
        "material": material,
        "evaluation": evaluation,
        "solve_runtime_s": solve_runtime,
        "raw": {
            "path": str(raw_path),
            "sha256": sha256(raw_path),
            "arrays": {
                name: list(np.asarray(value).shape)
                for name, value in raw_arrays.items()
            },
        },
        "normalization_policy": {
            "raw_fields_and_Q_are_unscaled": True,
            "per_polarization_matching_forbidden": True,
            "common_power_scale": source_pair["common_normalization"][
                "common_power_scale"
            ],
            "common_field_amplitude_scale": source_pair[
                "common_normalization"
            ]["common_field_amplitude_scale"],
        },
        "provenance": provenance,
        "optimizer_start_allowed": False,
    }
    _atomic_json(output / JSON_NAME, payload)
    return payload


def main() -> int:
    configured_output = os.environ.get("FDTDX_FRESH_OUTPUT_DIR", "").strip()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(configured_output) if configured_output else None,
    )
    parser.add_argument("--source", type=Path, default=configured_source())
    parser.add_argument("--source-pair", type=Path, required=True)
    parser.add_argument("--source-pair-sha256", required=True)
    parser.add_argument("--reference", choices=REFERENCE_NAMES, required=True)
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--case-contract", type=Path, required=True)
    parser.add_argument("--case-contract-sha256", required=True)
    args = parser.parse_args()
    if args.output_dir is None:
        parser.error("--output-dir or FDTDX_FRESH_OUTPUT_DIR is required")
    try:
        case_spec, _, case_audit = resolve_case_input(
            args.case_contract, args.case_contract_sha256
        )
        result = run(
            args.output_dir,
            args.source,
            args.source_pair,
            args.source_pair_sha256,
            args.reference,
            args.polarization,
            case_spec,
            case_audit,
        )
    except Exception as error:
        failure = {
            "status": "BLOCKED_FDTDX_FRESH_EXACT_BINARY_PILOT_EXCEPTION",
            "ready": False,
            "reference": args.reference,
            "polarization": args.polarization,
            "case_contract_path": (
                str(args.case_contract.expanduser().resolve())
                if args.case_contract is not None
                else None
            ),
            "case_contract_expected_sha256": args.case_contract_sha256,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "optimizer_start_allowed": False,
        }
        output = args.output_dir.expanduser().resolve()
        if output.is_dir() and not any(output.iterdir()):
            _atomic_json(output / JSON_NAME, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    summary = {
        "status": result["status"],
        "ready": result["ready"],
        "reference": result["reference"],
        "polarization": result["polarization"],
        "case_contract_sha256": result["numerical_case_contract"][
            "case_contract_sha256"
        ],
        "failed_gates": result["evaluation"]["failed_gates"],
        "maximum_complex_E_NRMSE": result["evaluation"]["field_stationarity"][
            "maximum_complex_E_NRMSE"
        ],
        "unscaled_Q_W": result["evaluation"]["Q"]["late"]["total_W"],
        "Q_vs_closed_phasor_relative": result["evaluation"]["flux"][
            "Q_vs_closed_phasor_symmetric_relative"
        ],
        "Q_vs_closed_td_relative": result["evaluation"]["flux"][
            "Q_vs_closed_td_symmetric_relative"
        ],
        "solve_runtime_s": result["solve_runtime_s"],
        "report": str(args.output_dir.expanduser().resolve() / JSON_NAME),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
