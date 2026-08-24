#!/usr/bin/env python3
"""All-air source calibration for the patched increment-state FDTDX fork."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    ANCHOR_CASE,
    FreshCaseSpec,
    TimeSpec,
    case_contract,
    realized_time_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_mesh import (
    build_model,
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


VERSION = "fdtdx-increment-state-source-only-v1"
STATUS_READY = "VALIDATED_FDTDX_INCREMENT_STATE_SOURCE_ONLY_CASE"
STATUS_BLOCKED = "BLOCKED_FDTDX_INCREMENT_STATE_SOURCE_ONLY_CASE"
SCOPE = "all-air source-only on patched increment-state FDTDX"
REPORT_NAME = "FDTDX_INCREMENT_STATE_SOURCE_ONLY.json"
RAW_NAME = "FDTDX_INCREMENT_STATE_SOURCE_ONLY_FIELDS.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _output_directory(path: Path) -> Path:
    supplied = path.expanduser()
    resolved = supplied.resolve()
    if not supplied.is_absolute() or not resolved.is_dir():
        raise RuntimeError("output directory must be existing and absolute")
    if any(resolved.iterdir()):
        raise RuntimeError("output directory must be empty")
    return resolved


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def run(
    output_directory: Path,
    source: Path,
    polarization: str,
    total_periods: int,
    window_periods: int,
) -> dict[str, Any]:
    started_total = time.perf_counter()
    output_directory = _output_directory(output_directory)
    source_audit = _source_audit(source)
    if not source_audit["ready"]:
        raise RuntimeError(f"patched FDTDX source audit failed: {source_audit}")
    configured_source = Path(os.environ.get("FDTDX_SOURCE_DIR", "")).resolve()
    if configured_source != Path(source_audit["path"]):
        raise RuntimeError("FDTDX_SOURCE_DIR does not match --source")

    case_spec = FreshCaseSpec(
        mesh=ANCHOR_CASE.mesh,
        time=TimeSpec(
            total_periods=total_periods,
            window_periods=window_periods,
            courant_factor=ANCHOR_CASE.time.courant_factor,
        ),
        pml_alpha_scale=ANCHOR_CASE.pml_alpha_scale,
        pml_target_reflection=ANCHOR_CASE.pml_target_reflection,
    )
    repository = Path(__file__).resolve().parents[3]
    runner_path = Path(__file__).resolve()
    started_build = time.perf_counter()
    model = build_model(
        case_spec.mesh,
        polarization,
        total_periods=case_spec.time.total_periods,
        window_periods=case_spec.time.window_periods,
        courant_factor=case_spec.time.courant_factor,
        alpha_scale=case_spec.pml_alpha_scale,
        target_reflection=case_spec.pml_target_reflection,
        include_adjoint_source=False,
        air_only_source_calibration=True,
        dispersive_state_representation="increment",
    )
    arrays, air_audit = all_air_arrays(model)
    build_runtime_s = time.perf_counter() - started_build
    if not air_audit["ready"]:
        raise RuntimeError(f"all-air material readback failed: {air_audit}")

    devices = model["jax"].devices()
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
    marker = fdtd_output.detector_states["target_field"]["phasor"]
    model["jax"].block_until_ready(marker)
    solve_runtime_s = time.perf_counter() - started_solve

    started_evaluation = time.perf_counter()
    evaluation, fields = evaluate_output(model, fdtd_output, polarization)
    evaluation_runtime_s = time.perf_counter() - started_evaluation
    fields.update(
        grid_x_edges_m=np.asarray(model["grid"].edges(0)),
        grid_y_edges_m=np.asarray(model["grid"].edges(1)),
        grid_z_edges_m=np.asarray(model["grid"].edges(2)),
    )
    raw_path = output_directory / RAW_NAME
    _atomic_npz(raw_path, fields)

    repository_dirty = _git(
        repository, "status", "--porcelain", "--untracked-files=all"
    )
    provenance_checks = {
        "repository_worktree_clean": repository_dirty == "",
        "fdtdx_source_ready": source_audit["ready"],
        "patched_commit_exact": source_audit["commit"] == EXPECTED_FDTDX_COMMIT,
        "one_visible_gpu": len(devices) == 1 and devices[0].platform == "gpu",
        "increment_state_selected": model["config"].dispersive_state_representation
        == "increment",
        "all_air_readback_ready": air_audit["ready"],
        "per_case_scaling_not_applied": True,
    }
    ready = all(provenance_checks.values()) and evaluation["ready"]
    payload = {
        "version": VERSION,
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "failed_provenance_checks": [
            name for name, passed in provenance_checks.items() if not passed
        ],
        "scope": SCOPE,
        "polarization": polarization,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "numerical_case_contract": case_contract(case_spec),
        "mesh": model["fresh_mesh_audit"],
        "time_contract": realized_time_contract(case_spec, model),
        "source_contract": model["source_contract"],
        "pml_face_parameters": model["pml_face_parameters"],
        "placement": model["placement"],
        "all_air_material_readback": air_audit,
        "dispersive_state_representation": model["dispersive_state_representation"],
        "evaluation": evaluation,
        "reporting_incident_power_W": CONTRACT.reporting_incident_power_W,
        "per_case_scale_not_authorized_until_pair_comparison": True,
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
            "repository_dirty_porcelain": repository_dirty,
            "fdtdx_source": source_audit,
            "runner_path": str(runner_path),
            "runner_sha256": _sha256(runner_path),
            "lumerical_used": False,
        },
        "provenance_checks": provenance_checks,
        "optimizer_start_allowed": False,
    }
    report_path = output_directory / REPORT_NAME
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
    args = parser.parse_args()
    payload = run(
        args.output_directory,
        args.source,
        args.polarization,
        args.total_periods,
        args.window_periods,
    )
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
