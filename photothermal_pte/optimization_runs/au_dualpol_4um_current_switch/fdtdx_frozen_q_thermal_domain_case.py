#!/usr/bin/env python3
"""One frozen-Q thermal domain-size case at selected diagnostic x/y and z factors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_frozen_q_thermal_xy_certificate import (
    STATUS_READY as PRIOR_XY_STATUS_READY,
    VERSION as PRIOR_XY_VERSION,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_frozen_q_thermal_z_case import (
    ENERGY_BALANCE_LIMIT,
    MAPPING_RTOL,
    POWER_RTOL,
    THERMAL_RESIDUAL_LIMIT,
    _atomic_json,
    _atomic_npz,
    _environment_manifest,
    _git,
    _output_directory,
    _relative_error,
    audit_frozen_input,
    load_frozen_fields,
    material_slices,
    require_exclusive_physical_gpu,
    sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    build_thermal_state,
    map_native_q_to_thermal,
    solve_thermal,
    tairte4_temperature,
    thermal_edges,
)


VERSION = "fdtdx-frozen-q-thermal-domain-case-v1"
STATUS_READY = "VALIDATED_DIAGNOSTIC_FDTDX_FROZEN_Q_THERMAL_DOMAIN_CASE"
STATUS_EXCEPTION = "BLOCKED_FDTDX_FROZEN_Q_THERMAL_DOMAIN_CASE_EXCEPTION"
REPORT_NAME = "FDTDX_FROZEN_Q_THERMAL_DOMAIN_CASE.json"
RAW_NAME = "FDTDX_FROZEN_Q_THERMAL_DOMAIN_CASE_FIELDS.npz"
POLARIZATIONS = ("Ea", "Eb")
THERMAL_XY_REFINEMENT_FACTOR = 2
THERMAL_Z_REFINEMENT_FACTOR = 2
DOMAIN_LEVELS = {
    "lateral": (32.0, 48.0, 64.0),
    "substrate": (20.0, 30.0, 40.0),
    "top_air": (2.0, 3.0, 4.0),
    "combined": (1.0, 2.0),
}
BASE_THERMAL_LATERAL_CELLS = 266
BASE_TA_LATERAL_CELLS = 160


def _restrict_blocks(
    array: np.ndarray, factor: int, *, reduction: str
) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if (
        not isinstance(factor, int)
        or isinstance(factor, bool)
        or factor < 1
    ):
        raise ValueError("xy refinement factor must be a positive integer")
    if value.ndim != 2 or value.shape[0] % factor or value.shape[1] % factor:
        raise ValueError("refined xy array shape is incompatible with factor")
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
    raise ValueError("block reduction must be 'mean' or 'sum'")


def _base_centers(refined_centers: np.ndarray, factor: int) -> np.ndarray:
    value = np.asarray(refined_centers, dtype=np.float64)
    if value.ndim != 1 or value.size % factor:
        raise ValueError("refined center array is incompatible with factor")
    return value.reshape(-1, factor).mean(axis=1)


def domain_configuration(axis: str, level_um: float) -> dict[str, float | int]:
    if axis not in DOMAIN_LEVELS:
        raise ValueError(f"thermal domain axis must be one of {tuple(DOMAIN_LEVELS)}")
    if level_um not in DOMAIN_LEVELS[axis]:
        raise ValueError(
            f"thermal domain level for {axis} must be one of {DOMAIN_LEVELS[axis]}"
        )
    configuration: dict[str, float | int] = {
        "lateral_half_span_um": 32,
        "substrate_depth_um": 20,
        "top_air_height_um": 2.0,
    }
    if axis == "combined":
        return (
            {
                "lateral_half_span_um": 48,
                "substrate_depth_um": 30,
                "top_air_height_um": 3.0,
            }
            if level_um == 1.0
            else {
                "lateral_half_span_um": 64,
                "substrate_depth_um": 40,
                "top_air_height_um": 4.0,
            }
        )
    key = {
        "lateral": "lateral_half_span_um",
        "substrate": "substrate_depth_um",
        "top_air": "top_air_height_um",
    }[axis]
    configuration[key] = int(level_um) if axis != "top_air" else float(level_um)
    return configuration


def audit_prior_xy_certificate(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = path.expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_file()
    actual = sha256(resolved) if exists else None
    payload = json.loads(resolved.read_text(encoding="utf-8")) if exists else {}
    selection = payload.get("selection", {})
    provenance = payload.get("certificate_provenance", {})
    global_checks = payload.get("global_checks", {})
    checks = {
        "path_is_absolute": supplied.is_absolute(),
        "file_exists": exists,
        "expected_sha256_is_hex": len(expected_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_sha256),
        "sha256_matches": actual == expected_sha256,
        "version_status_ready": payload.get("version") == PRIOR_XY_VERSION
        and payload.get("status") == PRIOR_XY_STATUS_READY
        and payload.get("ready") is True,
        "global_checks_all_true": bool(global_checks)
        and all(value is True for value in global_checks.values())
        and payload.get("failed_global_checks") == [],
        "selected_xy_z_factors_are_two": selection.get(
            "selected_diagnostic_frozen_q_thermal_xy_factor"
        )
        == THERMAL_XY_REFINEMENT_FACTOR
        and selection.get("selected_diagnostic_frozen_q_thermal_z_factor")
        == THERMAL_Z_REFINEMENT_FACTOR
        and selection.get("thermal_xy_z_diagnostic_converged") is True,
        "domain_and_production_remained_unselected": selection.get(
            "thermal_domain_and_boundary_converged"
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


def run(
    output_directory: Path,
    prior_xy_certificate_path: Path,
    expected_prior_xy_certificate_sha256: str,
    optical_certificate_path: Path,
    expected_optical_certificate_sha256: str,
    polarization: str,
    domain_axis: str,
    domain_level_um: float,
    expected_physical_gpu: int,
) -> dict[str, Any]:
    started_total = time.perf_counter()
    output = _output_directory(output_directory)
    if polarization not in POLARIZATIONS:
        raise ValueError(f"polarization must be one of {POLARIZATIONS}")
    domain = domain_configuration(domain_axis, domain_level_um)
    gpu_before = require_exclusive_physical_gpu(
        expected_physical_gpu, allow_current_process=False
    )
    repository = Path(__file__).resolve().parents[3]
    dirty_before = _git(
        repository, "status", "--porcelain", "--untracked-files=all"
    )
    if dirty_before != "":
        raise RuntimeError("repository must be clean before diagnostic solve")
    _, prior_xy_audit = audit_prior_xy_certificate(
        prior_xy_certificate_path, expected_prior_xy_certificate_sha256
    )
    if not prior_xy_audit["ready"]:
        raise RuntimeError(f"prior thermal xy certificate failed: {prior_xy_audit}")
    _, report, raw_path, input_audit = audit_frozen_input(
        optical_certificate_path,
        expected_optical_certificate_sha256,
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
        **domain,
    )
    build_runtime_s = time.perf_counter() - started_build

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
    expected_unscaled_W = float(
        report["evaluation"]["Q"]["late"]["total_W"]
    )
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
        raise RuntimeError("thermal environment does not expose exactly one CUDA GPU")
    started_solve = time.perf_counter()
    temperature_K, solver = solve_thermal(state, source_power_W, cuda_device=0)
    solve_runtime_s = time.perf_counter() - started_solve

    ta_native_K = tairte4_temperature(state, temperature_K)
    ta_base_K = _restrict_blocks(
        ta_native_K, THERMAL_XY_REFINEMENT_FACTOR, reduction="mean"
    )
    if ta_base_K.shape != (
        BASE_TA_LATERAL_CELLS,
        BASE_TA_LATERAL_CELLS,
    ):
        raise RuntimeError("restricted Ta temperature is not 160x160")
    ta_native_x = state.centers[0][
        (state.centers[0] >= -8e-6) & (state.centers[0] < 8e-6)
    ]
    ta_native_y = state.centers[1][
        (state.centers[1] >= -8e-6) & (state.centers[1] < 8e-6)
    ]
    ta_base_x = _base_centers(ta_native_x, THERMAL_XY_REFINEMENT_FACTOR)
    ta_base_y = _base_centers(ta_native_y, THERMAL_XY_REFINEMENT_FACTOR)
    gradient_x_K_m, gradient_y_K_m = np.gradient(
        ta_base_K, ta_base_x, ta_base_y, edge_order=2
    )
    source_native_xy_W = np.sum(source_power_W, axis=2)
    base_x_indices = np.flatnonzero(
        (state.centers[0] >= -32e-6) & (state.centers[0] < 32e-6)
    )
    base_y_indices = np.flatnonzero(
        (state.centers[1] >= -32e-6) & (state.centers[1] < 32e-6)
    )
    source_native_base_window_W = source_native_xy_W[
        np.ix_(base_x_indices, base_y_indices)
    ]
    source_base_xy_W = _restrict_blocks(
        source_native_base_window_W,
        THERMAL_XY_REFINEMENT_FACTOR,
        reduction="sum",
    )
    if source_base_xy_W.shape != (
        BASE_THERMAL_LATERAL_CELLS,
        BASE_THERMAL_LATERAL_CELLS,
    ):
        raise RuntimeError("restricted base-window thermal source is not 266x266")
    center_x = int(np.argmin(np.abs(state.centers[0])))
    center_y = int(np.argmin(np.abs(state.centers[1])))
    raw_arrays = {
        "ta_temperature_rise_K": ta_base_K,
        "ta_gradient_x_K_m": gradient_x_K_m,
        "ta_gradient_y_K_m": gradient_y_K_m,
        "ta_x_centers_m": ta_base_x,
        "ta_y_centers_m": ta_base_y,
        "source_power_xy_W": source_base_xy_W,
        "ta_temperature_native_K": ta_native_K,
        "ta_x_native_centers_m": ta_native_x,
        "ta_y_native_centers_m": ta_native_y,
        "source_power_native_xy_W": source_native_xy_W,
        "thermal_x_native_centers_m": state.centers[0],
        "thermal_y_native_centers_m": state.centers[1],
        "thermal_z_centers_m": state.centers[2],
        "center_temperature_rise_K": temperature_K[center_x, center_y, :],
    }
    raw_output = output / RAW_NAME
    _atomic_npz(raw_output, raw_arrays)

    matrix = state.system.matrix_W_K
    difference = matrix - matrix.T
    matrix_scale = max(
        float(np.max(np.abs(matrix.data))), np.finfo(float).tiny
    )
    asymmetry = (
        0.0
        if difference.nnz == 0
        else float(np.max(np.abs(difference.data))) / matrix_scale
    )
    base_edges = thermal_edges(
        THERMAL_Z_REFINEMENT_FACTOR,
        xy_refinement_factor=1,
        **domain,
    )
    mesh_checks = {
        "x_original_faces_preserved": np.array_equal(
            state.edges[0][::THERMAL_XY_REFINEMENT_FACTOR], base_edges[0]
        ),
        "y_original_faces_preserved": np.array_equal(
            state.edges[1][::THERMAL_XY_REFINEMENT_FACTOR], base_edges[1]
        ),
        "selected_z_edges_exact": np.array_equal(state.edges[2], base_edges[2]),
        "declared_domain_bounds_exact": np.allclose(
            (
                state.edges[0][0],
                state.edges[0][-1],
                state.edges[1][0],
                state.edges[1][-1],
                state.edges[2][0],
                state.edges[2][-1],
            ),
            (
                -float(domain["lateral_half_span_um"]) * 1e-6,
                float(domain["lateral_half_span_um"]) * 1e-6,
                -float(domain["lateral_half_span_um"]) * 1e-6,
                float(domain["lateral_half_span_um"]) * 1e-6,
                -float(domain["substrate_depth_um"]) * 1e-6,
                float(domain["top_air_height_um"]) * 1e-6,
            ),
            rtol=0.0,
            atol=2e-18,
        ),
        "base_temperature_coordinates_reconstructed": np.allclose(
            ta_base_x,
            0.5 * (base_edges[0][:-1] + base_edges[0][1:])[
                (0.5 * (base_edges[0][:-1] + base_edges[0][1:]) >= -8e-6)
                & (0.5 * (base_edges[0][:-1] + base_edges[0][1:]) < 8e-6)
            ],
            rtol=0.0,
            atol=2e-18,
        ),
        "restricted_source_conserves_total": _relative_error(
            float(np.sum(source_base_xy_W)), float(np.sum(source_power_W))
        )
        <= 5e-14,
    }
    solver_checks = {
        "matrix_symmetric_to_roundoff": asymmetry <= 1.0e-13,
        "temperature_finite": bool(np.all(np.isfinite(temperature_K))),
        "temperature_nonnegative_to_solver_tolerance": float(
            np.min(temperature_K)
        )
        >= -1.0e-8 * max(float(np.max(temperature_K)), 1.0),
        "explicit_relative_residual_within_limit": float(
            solver["relative_residual"]
        )
        <= THERMAL_RESIDUAL_LIMIT,
        "energy_balance_within_limit": float(
            solver["energy_balance_relative"]
        )
        <= ENERGY_BALANCE_LIMIT,
    }
    dirty_after = _git(
        repository, "status", "--porcelain", "--untracked-files=all"
    )
    gpu_after = require_exclusive_physical_gpu(
        expected_physical_gpu, allow_current_process=True
    )
    provenance_checks = {
        "repository_clean_before_and_after": dirty_before == dirty_after == "",
        "prior_xy_certificate_revalidated": prior_xy_audit["ready"] is True,
        "input_artifacts_revalidated": input_audit["ready"] is True,
        "mapping_checks_all_true": all(mapping_checks.values()),
        "mesh_checks_all_true": all(mesh_checks.values()),
        "solver_checks_all_true": all(solver_checks.values()),
        "one_exclusive_visible_gpu_before_and_after": gpu_before["exclusive"]
        is True
        and gpu_after["exclusive"] is True,
        "exact_binary_geometry_used": frozen["audit"]["checks"][
            "design_mask_integer_binary"
        ]
        and frozen["audit"]["checks"]["design_mask_solid_count_exact"],
        "optimizer_not_run": True,
        "lumerical_not_used": True,
    }
    ready = all(provenance_checks.values())
    payload: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS_READY if ready else STATUS_EXCEPTION,
        "ready": ready,
        "scope": (
            "diagnostic-only frozen-Q thermal domain-size convergence at "
            "selected diagnostic x/y and z factor 2; no optical/electrical/"
            "adjoint/optimizer or production-mesh promotion"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "polarization": polarization,
        "thermal_domain_axis": domain_axis,
        "thermal_domain_level_um": domain_level_um,
        "thermal_domain": domain,
        "thermal_xy_refinement_factor": THERMAL_XY_REFINEMENT_FACTOR,
        "thermal_z_refinement_factor": THERMAL_Z_REFINEMENT_FACTOR,
        "prior_thermal_xy_certificate": prior_xy_audit,
        "input_audit": input_audit,
        "raw_field_audit": frozen["audit"],
        "normalization": {
            "reporting_incident_power_W": CONTRACT.reporting_incident_power_W,
            "common_power_scale": common_power_scale,
            "mapped_scaled_absorbed_power_W": float(np.sum(source_power_W)),
            "per_polarization_power_matching_forbidden": True,
        },
        "mapping": mapping,
        "mapping_checks": mapping_checks,
        "mesh_checks": mesh_checks,
        "thermal_mesh": {
            "shape": list(state.system.shape),
            "unknowns": int(matrix.shape[0]),
            "matrix_nonzeros": int(matrix.nnz),
            "matrix_relative_asymmetry": asymmetry,
            "x_edges_sha256": hashlib.sha256(
                state.edges[0].tobytes()
            ).hexdigest(),
            "y_edges_sha256": hashlib.sha256(
                state.edges[1].tobytes()
            ).hexdigest(),
            "z_edges_sha256": hashlib.sha256(
                state.edges[2].tobytes()
            ).hexdigest(),
        },
        "thermal_solution": {
            "solver": solver,
            "global_min_temperature_rise_K": float(np.min(temperature_K)),
            "global_max_temperature_rise_K": float(np.max(temperature_K)),
            "ta_native_max_temperature_rise_K": float(np.max(ta_native_K)),
            "ta_base_max_temperature_rise_K": float(np.max(ta_base_K)),
            "ta_base_mean_temperature_rise_K": float(np.mean(ta_base_K)),
            "ta_base_gradient_combined_l2_K_m": float(
                np.sqrt(
                    np.sum(gradient_x_K_m**2) + np.sum(gradient_y_K_m**2)
                )
            ),
        },
        "solver_checks": solver_checks,
        "runtime": {
            "raw_load_and_audit_s": load_runtime_s,
            "thermal_assembly_s": build_runtime_s,
            "conservative_remap_s": remap_runtime_s,
            "cuda_pcg_s": solve_runtime_s,
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
        "optical_mesh_blocked": True,
        "thermal_domain_converged": False,
        "electrical_mesh_converged": False,
        "production_mesh_selected": False,
        "optimizer_start_allowed": False,
    }
    _atomic_json(output / REPORT_NAME, payload)
    print(
        json.dumps(
            {
                "report": str(output / REPORT_NAME),
                "ready": ready,
                "polarization": polarization,
                "thermal_domain_axis": domain_axis,
                "thermal_domain_level_um": domain_level_um,
                "thermal_xy_refinement_factor": THERMAL_XY_REFINEMENT_FACTOR,
                "unknowns": int(matrix.shape[0]),
                **payload["runtime"],
            }
        )
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--prior-thermal-xy-certificate", type=Path, required=True)
    parser.add_argument("--prior-thermal-xy-certificate-sha256", required=True)
    parser.add_argument("--z32-certificate", type=Path, required=True)
    parser.add_argument("--z32-certificate-sha256", required=True)
    parser.add_argument("--polarization", choices=POLARIZATIONS, required=True)
    parser.add_argument(
        "--thermal-domain-axis",
        choices=tuple(DOMAIN_LEVELS),
        required=True,
    )
    parser.add_argument("--thermal-domain-level-um", type=float, required=True)
    parser.add_argument("--expected-physical-gpu", type=int, required=True)
    args = parser.parse_args()
    output = args.output_directory.expanduser().resolve()
    try:
        payload = run(
            args.output_directory,
            args.prior_thermal_xy_certificate,
            args.prior_thermal_xy_certificate_sha256,
            args.z32_certificate,
            args.z32_certificate_sha256,
            args.polarization,
            args.thermal_domain_axis,
            args.thermal_domain_level_um,
            args.expected_physical_gpu,
        )
    except Exception as error:
        payload = {
            "version": VERSION,
            "status": STATUS_EXCEPTION,
            "ready": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "polarization": args.polarization,
            "thermal_domain_axis": args.thermal_domain_axis,
            "thermal_domain_level_um": args.thermal_domain_level_um,
            "thermal_xy_refinement_factor": THERMAL_XY_REFINEMENT_FACTOR,
            "thermal_z_refinement_factor": THERMAL_Z_REFINEMENT_FACTOR,
            "diagnostic_only": True,
            "optical_mesh_blocked": True,
            "thermal_domain_converged": False,
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
