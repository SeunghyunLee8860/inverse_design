#!/usr/bin/env python3
"""Diagnostic thermal-z solve driven by one frozen, certified FDTDX z32 Q field.

This runner deliberately cannot promote an optical mesh or start optimization.
It revalidates the blocked z32 certificate and exact-binary raw artifact by
bytes, conservatively remaps the common-285-uW-scaled heat source, and changes
only the explicit thermal z discretization.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
from typing import Any, Mapping

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_exact_binary_mesh_case import (
    STATUS_READY as MATERIAL_STATUS_READY,
    VERSION as MATERIAL_VERSION,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_full_z32_extension_certificate import (
    STATUS_BLOCKED as Z32_STATUS_BLOCKED,
    VERSION as Z32_CERTIFICATE_VERSION,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    build_thermal_state,
    map_native_q_to_thermal,
    solve_thermal,
    tairte4_temperature,
)


VERSION = "fdtdx-frozen-q-thermal-z-case-v1"
STATUS_READY = "VALIDATED_DIAGNOSTIC_FDTDX_FROZEN_Q_THERMAL_Z_CASE"
STATUS_EXCEPTION = "BLOCKED_FDTDX_FROZEN_Q_THERMAL_Z_CASE_EXCEPTION"
REPORT_NAME = "FDTDX_FROZEN_Q_THERMAL_Z_CASE.json"
RAW_NAME = "FDTDX_FROZEN_Q_THERMAL_Z_CASE_FIELDS.npz"
POLARIZATIONS = ("Ea", "Eb")
ALLOWED_Z_REFINEMENT_FACTORS = (1, 2, 4)
EXPECTED_SOLID_CELLS = 375
MAPPING_RTOL = 5.0e-12
POWER_RTOL = 5.0e-12
THERMAL_RESIDUAL_LIMIT = 2.0e-8
ENERGY_BALANCE_LIMIT = 2.0e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _all_true(values: Mapping[str, Any]) -> bool:
    return bool(values) and all(value is True for value in values.values())


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)


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


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _output_directory(path: Path) -> Path:
    supplied = path.expanduser()
    resolved = supplied.resolve()
    if not supplied.is_absolute() or not resolved.is_dir():
        raise RuntimeError("output directory must be existing and absolute")
    if any(resolved.iterdir()):
        raise RuntimeError("output directory must be empty")
    return resolved


def certificate_control_checks(payload: Mapping[str, Any]) -> dict[str, bool]:
    promotion = payload.get("promotion", {})
    return {
        "blocked_z32_certificate_version_status_exact": payload.get("version")
        == Z32_CERTIFICATE_VERSION
        and payload.get("status") == Z32_STATUS_BLOCKED
        and payload.get("ready") is False,
        "certificate_global_checks_all_true": _all_true(
            payload.get("global_checks", {})
        )
        and payload.get("failed_global_checks") == [],
        "optical_mesh_selection_remains_blocked": promotion.get(
            "full_domain_z_converged"
        )
        is False
        and promotion.get("selected_mesh_level") is None
        and promotion.get("z_only_ladder_terminated") is True
        and promotion.get("z64_run_allowed") is False,
        "optimizer_remains_forbidden": payload.get("optimizer_start_allowed")
        is False
        and promotion.get("optimizer_start_allowed") is False,
    }


def audit_frozen_input(
    certificate_path: Path,
    expected_certificate_sha256: str,
    polarization: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    supplied = certificate_path.expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_file()
    actual_certificate_sha = sha256(resolved) if exists else None
    certificate = (
        json.loads(resolved.read_text(encoding="utf-8")) if exists else {}
    )
    controls = certificate_control_checks(certificate)
    case = certificate.get("case_audits", {}).get("z32", {}).get(
        polarization, {}
    )
    report_supplied = Path(case.get("path", "")).expanduser()
    report_path = report_supplied.resolve()
    report_exists = report_path.is_file()
    actual_report_sha = sha256(report_path) if report_exists else None
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_exists
        else {}
    )
    raw_supplied = Path(report.get("raw", {}).get("path", "")).expanduser()
    raw_path = raw_supplied.resolve()
    raw_exists = raw_path.is_file()
    actual_raw_sha = sha256(raw_path) if raw_exists else None
    raw_case = case.get("raw", {})
    normalization = report.get("normalization_policy", {})
    checks = {
        "certificate_path_is_absolute": supplied.is_absolute(),
        "certificate_exists": exists,
        "certificate_sha256_matches": actual_certificate_sha
        == expected_certificate_sha256,
        **controls,
        "certified_case_ready": case.get("ready") is True
        and _all_true(case.get("checks", {}))
        and case.get("failed_checks") == [],
        "report_path_is_absolute": report_supplied.is_absolute(),
        "report_exists": report_exists,
        "report_sha256_rebound_to_certificate": actual_report_sha
        == case.get("actual_sha256")
        == case.get("expected_sha256"),
        "material_report_version_status_ready": report.get("version")
        == MATERIAL_VERSION
        and report.get("status") == MATERIAL_STATUS_READY
        and report.get("ready") is True,
        "polarization_and_z32_labels_exact": report.get("polarization")
        == polarization
        and report.get("full_z_extension") == "z32",
        "raw_path_is_absolute": raw_supplied.is_absolute(),
        "raw_exists": raw_exists,
        "raw_sha256_rebound_to_certificate": actual_raw_sha
        == raw_case.get("actual_sha256")
        == report.get("raw", {}).get("sha256"),
        "certified_raw_checks_all_true": raw_case.get("ready") is True
        and _all_true(raw_case.get("checks", {})),
        "exact_binary_no_gray_law": report.get("material", {})
        .get("exact_binary_au", {})
        .get("gray_density_allowed")
        is False
        and report.get("material", {})
        .get("exact_binary_au", {})
        .get("rho_power")
        is None,
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
        raise RuntimeError(f"frozen z32 input audit failed: {audit}")
    return certificate, report, raw_path, audit


def material_slices(
    placement: Mapping[str, Any],
) -> dict[str, tuple[slice, slice, slice]]:
    result = {}
    for material, name in (("au", "au_design"), ("tairte4", "fixed_tairte4")):
        bounds = placement.get(name)
        if (
            not isinstance(bounds, list)
            or len(bounds) != 3
            or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(value, int) for value in pair)
                or pair[0] < 0
                or pair[1] <= pair[0]
                for pair in bounds
            )
        ):
            raise ValueError(f"invalid {name} placement")
        result[material] = tuple(slice(pair[0], pair[1]) for pair in bounds)
    return result


class FrozenGrid:
    def __init__(self, edges: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        self._edges = edges

    def edges(self, axis: int) -> np.ndarray:
        return self._edges[axis]


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
    expected_shapes = {
        "au": (3, 80, 80, 64),
        "tairte4": (3, 160, 160, 160),
    }
    raw_power = {}
    checks = {
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
    for material in ("au", "tairte4"):
        q_value = q_fields[material]
        volume = volumes[material]
        checks[f"{material}_Q_volume_shape_exact"] = (
            q_value.shape == expected_shapes[material] == volume.shape
        )
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
            float(reported.get("by_material", {}).get(material, {}).get("total_W", 0.0)),
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


def _nvidia_gpu_inventory() -> dict[str, Any]:
    gpu_lines = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    process_lines = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    gpus = []
    for line in gpu_lines:
        index, uuid, name = (item.strip() for item in line.split(",", 2))
        gpus.append({"index": int(index), "uuid": uuid, "name": name})
    processes = []
    for line in process_lines:
        if not line.strip():
            continue
        uuid, pid, name, memory = (item.strip() for item in line.split(",", 3))
        processes.append(
            {"uuid": uuid, "pid": int(pid), "name": name, "used_memory": memory}
        )
    return {"gpus": gpus, "compute_processes": processes}


def require_exclusive_physical_gpu(
    expected_physical_gpu: int, *, allow_current_process: bool
) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    inventory = _nvidia_gpu_inventory()
    matches = [
        item for item in inventory["gpus"] if item["index"] == expected_physical_gpu
    ]
    if visible != str(expected_physical_gpu) or len(matches) != 1:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must expose exactly the expected physical GPU"
        )
    gpu = matches[0]
    occupants = [
        item
        for item in inventory["compute_processes"]
        if item["uuid"] == gpu["uuid"]
        and (not allow_current_process or item["pid"] != os.getpid())
    ]
    if occupants:
        raise RuntimeError(
            f"physical GPU {expected_physical_gpu} has other compute processes: "
            f"{occupants}"
        )
    return {"selected": gpu, "inventory": inventory, "exclusive": True}


def _environment_manifest() -> dict[str, Any]:
    import scipy
    import torch

    distributions = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    frozen = "\n".join(distributions) + "\n"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_cuda_device_count": torch.cuda.device_count(),
        "visible_cuda_device_name": torch.cuda.get_device_name(0),
        "distribution_manifest_sha256": hashlib.sha256(frozen.encode()).hexdigest(),
        "distribution_manifest": distributions,
    }


def run(
    output_directory: Path,
    certificate_path: Path,
    expected_certificate_sha256: str,
    polarization: str,
    z_refinement_factor: int,
    expected_physical_gpu: int,
) -> dict[str, Any]:
    started_total = time.perf_counter()
    output = _output_directory(output_directory)
    if polarization not in POLARIZATIONS:
        raise ValueError(f"polarization must be one of {POLARIZATIONS}")
    if z_refinement_factor not in ALLOWED_Z_REFINEMENT_FACTORS:
        raise ValueError(
            f"thermal z refinement factor must be one of {ALLOWED_Z_REFINEMENT_FACTORS}"
        )
    gpu_before = require_exclusive_physical_gpu(
        expected_physical_gpu, allow_current_process=False
    )
    repository = Path(__file__).resolve().parents[3]
    dirty_before = _git(
        repository, "status", "--porcelain", "--untracked-files=all"
    )
    if dirty_before != "":
        raise RuntimeError("repository must be clean before diagnostic solve")
    _, report, raw_path, input_audit = audit_frozen_input(
        certificate_path, expected_certificate_sha256, polarization
    )

    started_load = time.perf_counter()
    frozen = load_frozen_fields(raw_path, report)
    load_runtime_s = time.perf_counter() - started_load

    started_build = time.perf_counter()
    state = build_thermal_state(
        frozen["mask"].astype(np.float64),
        z_refinement_factor=z_refinement_factor,
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

    ta_temperature_K = tairte4_temperature(state, temperature_K)
    ta_x = state.centers[0][
        (state.centers[0] >= -8e-6) & (state.centers[0] < 8e-6)
    ]
    ta_y = state.centers[1][
        (state.centers[1] >= -8e-6) & (state.centers[1] < 8e-6)
    ]
    gradient_x_K_m, gradient_y_K_m = np.gradient(
        ta_temperature_K, ta_x, ta_y, edge_order=2
    )
    center_x = int(np.argmin(np.abs(state.centers[0])))
    center_y = int(np.argmin(np.abs(state.centers[1])))
    source_xy_W = np.sum(source_power_W, axis=2)
    raw_arrays = {
        "ta_temperature_rise_K": ta_temperature_K,
        "ta_gradient_x_K_m": gradient_x_K_m,
        "ta_gradient_y_K_m": gradient_y_K_m,
        "ta_x_centers_m": ta_x,
        "ta_y_centers_m": ta_y,
        "thermal_z_centers_m": state.centers[2],
        "center_temperature_rise_K": temperature_K[center_x, center_y, :],
        "source_power_xy_W": source_xy_W,
        "thermal_x_centers_m": state.centers[0],
        "thermal_y_centers_m": state.centers[1],
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
        "input_artifacts_revalidated": input_audit["ready"] is True,
        "mapping_checks_all_true": all(mapping_checks.values()),
        "solver_checks_all_true": all(solver_checks.values()),
        "one_exclusive_visible_gpu_before_and_after": gpu_before["exclusive"]
        is True
        and gpu_after["exclusive"] is True,
        "exact_binary_geometry_used": frozen["audit"]["checks"][
            "design_mask_integer_binary"
        ]
        and frozen["audit"]["checks"]["design_mask_solid_count_exact"],
        "optical_mesh_still_blocked": certificate_control_checks(
            json.loads(certificate_path.resolve().read_text(encoding="utf-8"))
        )["optical_mesh_selection_remains_blocked"],
        "optimizer_not_run": True,
        "lumerical_not_used": True,
    }
    ready = all(provenance_checks.values())
    payload = {
        "version": VERSION,
        "status": STATUS_READY if ready else STATUS_EXCEPTION,
        "ready": ready,
        "scope": (
            "diagnostic-only frozen-Q thermal z refinement; optical z32 remains "
            "blocked; no electrical solve, adjoint, optimizer, or mesh promotion"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "polarization": polarization,
        "thermal_z_refinement_factor": z_refinement_factor,
        "input_audit": input_audit,
        "raw_field_audit": frozen["audit"],
        "normalization": {
            "reporting_incident_power_W": CONTRACT.reporting_incident_power_W,
            "common_power_scale": common_power_scale,
            "certified_unscaled_absorbed_power_W": expected_unscaled_W,
            "mapped_unscaled_absorbed_power_W": float(
                np.sum(source_unscaled_W)
            ),
            "mapped_scaled_absorbed_power_W": float(np.sum(source_power_W)),
            "per_polarization_power_matching_forbidden": True,
        },
        "mapping": mapping,
        "mapping_checks": mapping_checks,
        "thermal_mesh": {
            "shape": list(state.system.shape),
            "unknowns": int(matrix.shape[0]),
            "matrix_nonzeros": int(matrix.nnz),
            "x_edges_sha256": hashlib.sha256(
                state.edges[0].tobytes()
            ).hexdigest(),
            "y_edges_sha256": hashlib.sha256(
                state.edges[1].tobytes()
            ).hexdigest(),
            "z_edges_sha256": hashlib.sha256(
                state.edges[2].tobytes()
            ).hexdigest(),
            "matrix_relative_asymmetry": asymmetry,
        },
        "thermal_solution": {
            "solver": solver,
            "global_min_temperature_rise_K": float(np.min(temperature_K)),
            "global_max_temperature_rise_K": float(np.max(temperature_K)),
            "ta_min_temperature_rise_K": float(np.min(ta_temperature_K)),
            "ta_max_temperature_rise_K": float(np.max(ta_temperature_K)),
            "ta_mean_temperature_rise_K": float(np.mean(ta_temperature_K)),
            "ta_gradient_x_l2_K_m": float(np.linalg.norm(gradient_x_K_m)),
            "ta_gradient_y_l2_K_m": float(np.linalg.norm(gradient_y_K_m)),
            "ta_gradient_combined_l2_K_m": float(
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
                "thermal_z_refinement_factor": z_refinement_factor,
                "unknowns": int(matrix.shape[0]),
                **payload["runtime"],
            }
        )
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--z32-certificate", type=Path, required=True)
    parser.add_argument("--z32-certificate-sha256", required=True)
    parser.add_argument("--polarization", choices=POLARIZATIONS, required=True)
    parser.add_argument(
        "--thermal-z-refinement-factor",
        type=int,
        choices=ALLOWED_Z_REFINEMENT_FACTORS,
        required=True,
    )
    parser.add_argument("--expected-physical-gpu", type=int, required=True)
    args = parser.parse_args()
    output = args.output_directory.expanduser().resolve()
    try:
        payload = run(
            args.output_directory,
            args.z32_certificate,
            args.z32_certificate_sha256,
            args.polarization,
            args.thermal_z_refinement_factor,
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
            "thermal_z_refinement_factor": args.thermal_z_refinement_factor,
            "diagnostic_only": True,
            "optical_mesh_blocked": True,
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
