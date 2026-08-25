#!/usr/bin/env python3
"""All-air source-only pilot on the user-requested balanced FDTDX mesh."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    TimeSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_only import (
    all_air_arrays,
    evaluate_output,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_exact_binary_control import (
    EXPECTED_FDTDX_COMMIT,
    _json_default,
    _json_safe_memory_stats,
    _source_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_source_only import (
    _atomic_json,
    _atomic_npz,
    _git,
    _output_directory,
    _sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_mesh import (
    build_model,
    mesh_audit,
)


VERSION = "fdtdx-user-balanced-source-only-v1"
CASE_VERSION = "fdtdx-user-balanced-case-v1"
STATUS_READY = "VALIDATED_FDTDX_USER_BALANCED_SOURCE_ONLY"
STATUS_BLOCKED = "BLOCKED_FDTDX_USER_BALANCED_SOURCE_ONLY"
REPORT_NAME = "FDTDX_USER_BALANCED_SOURCE_ONLY.json"
RAW_NAME = "FDTDX_USER_BALANCED_SOURCE_ONLY_FIELDS.npz"


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def balanced_case_contract(time_spec: TimeSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": CASE_VERSION,
        "mesh": mesh_audit(),
        "time": {
            "total_periods": time_spec.total_periods,
            "window_periods": time_spec.window_periods,
            "courant_factor": time_spec.courant_factor,
            "source_startup_periods": time_spec.source_startup_periods,
        },
        "pml": {
            "layers_each_face_xyz": [8, 8, 8],
            "alpha_scale": 1.0,
            "target_reflection": 1e-6,
        },
        "rules": {
            "same_contract_required_for_Ea_and_Eb": True,
            "source_pair_required_before_material_case": True,
            "per_polarization_normalization_forbidden": True,
            "exact_binary_only_after_source_validation": True,
            "optimizer_start_allowed": False,
        },
    }
    payload["case_contract_sha256"] = _canonical_sha256(payload)
    return payload


def realized_time_contract(time_spec: TimeSpec, model: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_periods": time_spec.total_periods,
        "window_periods": time_spec.window_periods,
        "source_startup_periods": int(
            model["source_contract"]["num_startup_periods"]
        ),
        "courant_factor": time_spec.courant_factor,
        "time_step_s": float(model["config"].time_step_duration),
        "time_steps_total": int(model["config"].time_steps_total),
    }


def run(
    output_directory: Path,
    source: Path,
    polarization: str,
    total_periods: int,
    window_periods: int,
    courant_factor: float,
) -> dict[str, Any]:
    started_total = time.perf_counter()
    output = _output_directory(output_directory)
    source_audit = _source_audit(source)
    if not source_audit["ready"]:
        raise RuntimeError(f"patched FDTDX source audit failed: {source_audit}")
    if Path(os.environ.get("FDTDX_SOURCE_DIR", "")).resolve() != Path(
        source_audit["path"]
    ):
        raise RuntimeError("FDTDX_SOURCE_DIR does not match --source")
    if polarization not in ("Ea", "Eb"):
        raise ValueError("polarization must be Ea or Eb")
    time_spec = TimeSpec(
        total_periods=total_periods,
        window_periods=window_periods,
        courant_factor=courant_factor,
    )
    repository = Path(__file__).resolve().parents[3]
    runner_path = Path(__file__).resolve()
    dirty_before = _git(
        repository, "status", "--porcelain", "--untracked-files=all"
    )
    if dirty_before:
        raise RuntimeError("repository must be clean before balanced source solve")

    started_build = time.perf_counter()
    model = build_model(
        polarization,
        total_periods=time_spec.total_periods,
        window_periods=time_spec.window_periods,
        courant_factor=time_spec.courant_factor,
        include_adjoint_source=False,
        air_only_source_calibration=True,
        dispersive_state_representation="increment",
    )
    arrays, air_audit = all_air_arrays(model)
    build_runtime_s = time.perf_counter() - started_build
    if not air_audit["ready"]:
        raise RuntimeError(f"all-air material readback failed: {air_audit}")
    devices = model["jax"].devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("exactly one visible GPU is required")
    device_before = [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
            "memory_stats": _json_safe_memory_stats(device),
        }
        for device in devices
    ]

    started_solve = time.perf_counter()
    _, fdtd_output = model["fdtdx"].run_fdtd(
        arrays,
        model["placed"],
        model["config"],
        model["key"],
        show_progress=False,
    )
    model["jax"].block_until_ready(
        fdtd_output.detector_states["target_field"]["phasor"]
    )
    solve_runtime_s = time.perf_counter() - started_solve
    started_evaluation = time.perf_counter()
    evaluation, fields = evaluate_output(model, fdtd_output, polarization)
    evaluation_runtime_s = time.perf_counter() - started_evaluation
    fields.update(
        grid_x_edges_m=np.asarray(model["grid"].edges(0)),
        grid_y_edges_m=np.asarray(model["grid"].edges(1)),
        grid_z_edges_m=np.asarray(model["grid"].edges(2)),
    )
    raw_path = output / RAW_NAME
    _atomic_npz(raw_path, fields)

    dirty_after = _git(
        repository, "status", "--porcelain", "--untracked-files=all"
    )
    checks = {
        "repository_clean_before_and_after": dirty_before == dirty_after == "",
        "fdtdx_source_ready": source_audit["ready"],
        "patched_commit_exact": source_audit["commit"] == EXPECTED_FDTDX_COMMIT,
        "one_visible_gpu": len(devices) == 1 and devices[0].platform == "gpu",
        "increment_state_selected": (
            model["config"].dispersive_state_representation == "increment"
        ),
        "all_air_readback_ready": air_audit["ready"],
        "requested_grid_exact_except_declared_si_pitch": (
            model["fresh_mesh_audit"] == mesh_audit()
        ),
        "source_evaluation_ready": evaluation["ready"],
        "per_case_scaling_not_applied": True,
    }
    ready = all(checks.values())
    payload = {
        "version": VERSION,
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "scope": "all-air source-only on the requested balanced FDTDX mesh",
        "polarization": polarization,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "numerical_case_contract": balanced_case_contract(time_spec),
        "mesh": model["fresh_mesh_audit"],
        "time_contract": realized_time_contract(time_spec, model),
        "source_contract": model["source_contract"],
        "pml_face_parameters": model["pml_face_parameters"],
        "placement": model["placement"],
        "all_air_material_readback": air_audit,
        "evaluation": evaluation,
        "reporting_incident_power_W": CONTRACT.reporting_incident_power_W,
        "runtime": {
            "cold_build_and_array_preparation_s": build_runtime_s,
            "cold_compile_and_forward_s": solve_runtime_s,
            "host_evaluation_s": evaluation_runtime_s,
            "total_s": time.perf_counter() - started_total,
        },
        "raw": {
            "path": str(raw_path),
            "sha256": _sha256(raw_path),
            "arrays": {
                name: list(np.asarray(value).shape) for name, value in fields.items()
            },
        },
        "device_before": device_before,
        "device_after": [
            {
                "id": int(device.id),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "memory_stats": _json_safe_memory_stats(device),
            }
            for device in devices
        ],
        "runtime_lock": {
            "python": platform.python_version(),
            "jax": model["jax"].__version__,
            "fdtdx_import": str(Path(model["fdtdx"].__file__).resolve()),
        },
        "provenance": {
            "repository_commit": _git(repository, "rev-parse", "HEAD"),
            "repository_dirty_porcelain_before": dirty_before,
            "repository_dirty_porcelain_after": dirty_after,
            "fdtdx_source": source_audit,
            "runner_path": str(runner_path),
            "runner_sha256": _sha256(runner_path),
            "lumerical_used": False,
        },
        "checks": checks,
        "optimizer_start_allowed": False,
    }
    report_path = output / REPORT_NAME
    _atomic_json(report_path, payload)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "ready": ready,
                "incident_power_W": evaluation["flux"]["incident_plane_signed_W"],
                **payload["runtime"],
            },
            default=_json_default,
        )
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--total-periods", type=int, default=24)
    parser.add_argument("--window-periods", type=int, default=4)
    parser.add_argument("--courant-factor", type=float, default=0.5)
    args = parser.parse_args()
    payload = run(
        args.output_directory,
        args.source,
        args.polarization,
        args.total_periods,
        args.window_periods,
        args.courant_factor,
    )
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
