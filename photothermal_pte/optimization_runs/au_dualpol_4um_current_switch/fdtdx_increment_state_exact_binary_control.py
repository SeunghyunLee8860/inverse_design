#!/usr/bin/env python3
"""Cold-runtime and energy-closure control for patched increment-state FDTDX.

This is deliberately not an optimizer and not a source-normalized physics
comparison.  It runs one exact air/Au reference on the anchor mesh, checks
the realized physical one-pole ``A/C/B`` material stack, and reports the
cold build/compile/forward cost plus unnormalized Q/closed-flux closure.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    REFERENCE_NAMES,
    reference_mask,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_material import (
    arrays_for_exact_binary,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    ANCHOR_CASE,
    case_contract,
    realized_time_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
    _power_evaluation,
    material_stack_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_mesh import (
    build_model,
)


VERSION = "fdtdx-increment-state-exact-binary-control-v1"
STATUS_READY = "VALIDATED_FDTDX_INCREMENT_STATE_EXACT_BINARY_CONTROL"
STATUS_BLOCKED = "BLOCKED_FDTDX_INCREMENT_STATE_EXACT_BINARY_CONTROL"
EXPECTED_FDTDX_COMMIT = "6cc0e97252ee0b95de5016e8db1a5b414177efa4"
DEFAULT_REFERENCE = "l_shape_4um_with_500nm_arms"
REPORT_NAME = "FDTDX_INCREMENT_STATE_EXACT_BINARY_CONTROL.json"


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


def _source_audit(source: Path) -> dict[str, Any]:
    supplied = source.expanduser()
    resolved = supplied.resolve()
    exists = resolved.is_dir()
    commit = _git(resolved, "rev-parse", "HEAD") if exists else None
    dirty = (
        _git(resolved, "status", "--porcelain", "--untracked-files=all")
        if exists
        else None
    )
    checks = {
        "path_is_absolute": supplied.is_absolute(),
        "source_directory_exists": exists,
        "patched_commit_exact": commit == EXPECTED_FDTDX_COMMIT,
        "source_worktree_clean": dirty == "",
    }
    return {
        "path": str(resolved),
        "commit": commit,
        "dirty_porcelain": dirty,
        "checks": checks,
        "ready": all(checks.values()),
    }


def _output_directory(path: Path) -> Path:
    supplied = path.expanduser()
    resolved = supplied.resolve()
    if not supplied.is_absolute() or not resolved.is_dir():
        raise RuntimeError("output directory must be an existing absolute directory")
    if any(resolved.iterdir()):
        raise RuntimeError("output directory must be empty")
    return resolved


def _json_safe_memory_stats(device: Any) -> dict[str, int | float | str] | None:
    stats = device.memory_stats()
    if stats is None:
        return None
    return {
        str(key): value
        for key, value in stats.items()
        if isinstance(value, (int, float, str))
    }


def _unnormalized_closure_evaluation(
    model: dict[str, Any], output: Any, mask: np.ndarray
) -> dict[str, Any]:
    placeholder_source = {
        "comparison": {"mean_unscaled_incident_power_W": 1.0},
        "common_normalization": {"common_power_scale": 1.0},
    }
    evaluation, _ = _power_evaluation(
        model,
        output,
        mask,
        placeholder_source,
        ANCHOR_CASE.mesh,
    )
    evaluation["scope"] = (
        "unnormalized Q/closed-flux closure only; no source-only calibration"
    )
    evaluation["source_normalization_available"] = False
    evaluation.pop("common_285uW_reporting", None)
    for key in (
        "source_reference_all_air_unscaled_W",
        "absorbed_fraction_of_all_air_source",
    ):
        evaluation["flux"].pop(key, None)
    evaluation["gates"].pop("absorbed_fraction_physical", None)
    evaluation["failed_gates"] = [
        name for name, passed in evaluation["gates"].items() if not passed
    ]
    evaluation["ready"] = all(evaluation["gates"].values())
    return evaluation


def run(
    output_directory: Path,
    source: Path,
    polarization: str,
    reference: str,
) -> dict[str, Any]:
    started_total = time.perf_counter()
    output = _output_directory(output_directory)
    source_audit = _source_audit(source)
    if not source_audit["ready"]:
        raise RuntimeError(f"patched FDTDX source audit failed: {source_audit}")
    configured = Path(os.environ.get("FDTDX_SOURCE_DIR", "")).resolve()
    if configured != Path(source_audit["path"]):
        raise RuntimeError("FDTDX_SOURCE_DIR does not match --source")

    repository = Path(__file__).resolve().parents[3]
    runner_path = Path(__file__).resolve()
    mask = np.asarray(reference_mask(reference), dtype=np.uint8)
    started_build = time.perf_counter()
    model = build_model(
        ANCHOR_CASE.mesh,
        polarization,
        total_periods=ANCHOR_CASE.time.total_periods,
        window_periods=ANCHOR_CASE.time.window_periods,
        courant_factor=ANCHOR_CASE.time.courant_factor,
        alpha_scale=ANCHOR_CASE.pml_alpha_scale,
        target_reflection=ANCHOR_CASE.pml_target_reflection,
        include_adjoint_source=False,
        air_only_source_calibration=False,
        dispersive_state_representation="increment",
    )
    arrays = arrays_for_exact_binary(model, mask, ANCHOR_CASE.mesh)
    material = material_stack_audit(model, arrays, mask, ANCHOR_CASE.mesh)
    build_runtime_s = time.perf_counter() - started_build
    if not material["ready"]:
        raise RuntimeError(f"increment-state material readback failed: {material}")

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
    evaluation = _unnormalized_closure_evaluation(
        model, fdtd_output, mask
    )
    evaluation_runtime_s = time.perf_counter() - started_evaluation
    device_after = [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
            "memory_stats": _json_safe_memory_stats(device),
        }
        for device in devices
    ]

    repository_dirty = _git(
        repository, "status", "--porcelain", "--untracked-files=all"
    )
    provenance_checks = {
        "repository_worktree_clean": repository_dirty == "",
        "fdtdx_source_ready": source_audit["ready"],
        "one_visible_gpu": len(devices) == 1 and devices[0].platform == "gpu",
        "increment_state_selected": model["config"].dispersive_state_representation
        == "increment",
        "physical_one_pole_law_selected": model["material_law_mode"]
        == "physical-one-pole-increment-state",
        "source_normalization_not_claimed": evaluation[
            "source_normalization_available"
        ]
        is False,
    }
    ready = (
        all(provenance_checks.values())
        and material["ready"]
        and evaluation["ready"]
        and math.isfinite(solve_runtime_s)
        and solve_runtime_s > 0.0
    )
    payload = {
        "version": VERSION,
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "failed_provenance_checks": [
            name for name, passed in provenance_checks.items() if not passed
        ],
        "scope": (
            "one exact-binary Maxwell forward timing/closure control; no absolute "
            "Ea/Eb comparison, thermal, electrical, adjoint, or optimizer"
        ),
        "polarization": polarization,
        "reference": reference,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case": case_contract(ANCHOR_CASE),
        "time_contract": realized_time_contract(ANCHOR_CASE, model),
        "mesh": model["fresh_mesh_audit"],
        "placement": model["placement"],
        "source_contract": model["source_contract"],
        "material": material,
        "evaluation": evaluation,
        "runtime": {
            "cold_build_and_array_preparation_s": build_runtime_s,
            "cold_compile_and_forward_s": solve_runtime_s,
            "host_evaluation_s": evaluation_runtime_s,
            "total_s": time.perf_counter() - started_total,
            "interpretation": (
                "solve time includes first-call JAX compilation in this process; "
                "it is not an adjoint or optimization-iteration timing"
            ),
        },
        "device_before": device_before,
        "device_after": device_after,
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
    report = output / REPORT_NAME
    temporary = report.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, report)
    print(json.dumps({"report": str(report), **payload["runtime"], "ready": ready}))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--reference", choices=REFERENCE_NAMES, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    payload = run(
        args.output_directory,
        args.source,
        args.polarization,
        args.reference,
    )
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
