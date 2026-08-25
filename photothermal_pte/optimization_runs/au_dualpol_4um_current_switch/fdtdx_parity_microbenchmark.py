"""Bounded GPU timing gate for the fresh 4-um FDTDX parity model.

This is not a physics validation or a full forward solve.  It measures short
partial forwards in the inactive, previous-phasor, and late-detector windows,
then extrapolates their per-step slopes to the frozen 40-period schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
DEFAULT_STEPS = (64, 256, 1024)
DEFAULT_REPETITIONS = 2
MAX_FORWARD_SECONDS = 30.0 * 60.0


def _csv_rows(raw: str) -> list[list[str]]:
    return [
        [part.strip() for part in line.split(",")]
        for line in raw.splitlines()
        if line.strip()
    ]


def require_idle_gpu(
    gpu_uuid: str,
    *,
    gpu_rows: Iterable[Iterable[str]],
    process_rows: Iterable[Iterable[str]],
) -> dict[str, object]:
    """Fail closed unless the exact requested UUID is completely idle."""

    normalized_gpu_rows = [list(row) for row in gpu_rows]
    normalized_process_rows = [list(row) for row in process_rows]
    matches = [
        row for row in normalized_gpu_rows if len(row) >= 2 and row[1] == gpu_uuid
    ]
    if len(matches) != 1:
        raise RuntimeError(f"GPU UUID was not found exactly once: {gpu_uuid}")
    row = matches[0]
    if len(row) != 6:
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {row}")
    index, _, name, memory_used_mib, memory_total_mib, utilization_percent = row
    users = [
        row
        for row in normalized_process_rows
        if len(row) >= 1 and row[0] == gpu_uuid
    ]
    if users:
        raise RuntimeError(f"GPU {gpu_uuid} has compute processes: {users}")
    if int(memory_used_mib) != 0 or int(utilization_percent) != 0:
        raise RuntimeError(
            f"GPU {gpu_uuid} is not idle: memory={memory_used_mib} MiB, "
            f"utilization={utilization_percent}%"
        )
    return {
        "index": int(index),
        "uuid": gpu_uuid,
        "name": name,
        "memory_used_MiB": int(memory_used_mib),
        "memory_total_MiB": int(memory_total_mib),
        "utilization_percent": int(utilization_percent),
        "compute_processes": [],
    }


def query_and_require_idle_gpu(gpu_uuid: str) -> dict[str, object]:
    gpu_raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    process_raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return require_idle_gpu(
        gpu_uuid,
        gpu_rows=_csv_rows(gpu_raw),
        process_rows=_csv_rows(process_raw),
    )


def fit_phase(measurements: list[dict[str, object]], phase: str) -> dict[str, object]:
    step_counts = sorted(
        {int(row["steps"]) for row in measurements if row["phase"] == phase}
    )
    if len(step_counts) < 3:
        raise ValueError(f"phase {phase!r} needs at least three step counts")
    medians = {
        steps: float(
            np.median(
                [
                    float(row["seconds"])
                    for row in measurements
                    if row["phase"] == phase and int(row["steps"]) == steps
                ]
            )
        )
        for steps in step_counts
    }
    slope, intercept = np.polyfit(
        np.asarray(step_counts, dtype=np.float64),
        np.asarray([medians[steps] for steps in step_counts], dtype=np.float64),
        1,
    )
    return {
        "median_seconds": {str(key): value for key, value in medians.items()},
        "intercept_seconds": float(intercept),
        "seconds_per_step": float(slope),
    }


def estimate_full_forward_seconds(
    fits: dict[str, dict[str, object]],
    *,
    total_steps: int,
    window_steps: int,
) -> float:
    inactive_steps = total_steps - 2 * window_steps
    if inactive_steps <= 0:
        raise ValueError("two detector windows must be shorter than the full run")
    return float(
        float(fits["inactive"]["seconds_per_step"]) * inactive_steps
        + float(fits["previous_window"]["seconds_per_step"]) * window_steps
        + float(fits["late_window"]["seconds_per_step"]) * window_steps
    )


def _git_output(arguments: list[str]) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


def _write_new_external_json(output_path: Path, payload: dict[str, object]) -> None:
    output = output_path.expanduser().resolve()
    if output == REPOSITORY or REPOSITORY in output.parents:
        raise RuntimeError("raw benchmark JSON must remain outside the Git repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(serialized, encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    gpu_snapshot = query_and_require_idle_gpu(args.gpu_uuid)
    existing_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing_visibility not in {None, "", args.gpu_uuid}:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES conflicts with the requested UUID: "
            f"{existing_visibility!r}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # GPU-sensitive imports must occur only after the exact UUID is isolated.
    import jax
    import jax.numpy as jnp
    from fdtdx.fdtd.fdtd import custom_fdtd_forward

    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_contract import (
        grid_audit,
    )
    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_model import (
        arrays_for_density,
        build_model,
    )

    resources = grid_audit()["resources"]
    total_steps = int(resources["time"]["total_steps"])
    window_steps = int(resources["time"]["late_window_steps"])
    print("phase=build", flush=True)
    build_started = time.perf_counter()
    model = build_model(args.polarization, backend="gpu")
    arrays = arrays_for_density(
        model,
        jnp.full((80, 80), args.rho, dtype=jnp.float32),
    )
    jax.block_until_ready(arrays.fields.E)
    build_seconds = time.perf_counter() - build_started

    def advance(container, start_time, end_time):
        return custom_fdtd_forward(
            arrays=container,
            objects=model["placed"],
            config=model["config"],
            key=model["key"],
            reset_container=True,
            record_detectors=True,
            start_time=start_time,
            end_time=end_time,
            show_progress=False,
        )[1]

    print("phase=compile", flush=True)
    compile_started = time.perf_counter()
    executable = jax.jit(advance).lower(
        arrays,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    ).compile()
    compile_seconds = time.perf_counter() - compile_started
    warmup_started = time.perf_counter()
    warm = executable(
        arrays,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )
    jax.block_until_ready(warm.fields.E)
    warmup_seconds = time.perf_counter() - warmup_started

    phase_starts = {
        "inactive": 0,
        "previous_window": total_steps - 2 * window_steps,
        "late_window": total_steps - window_steps,
    }
    measurements: list[dict[str, object]] = []
    last = warm
    print("phase=timing", flush=True)
    for phase, start in phase_starts.items():
        for steps in args.steps:
            for repetition in range(args.repetitions):
                started = time.perf_counter()
                last = executable(
                    arrays,
                    jnp.asarray(start, dtype=jnp.int32),
                    jnp.asarray(start + steps, dtype=jnp.int32),
                )
                jax.block_until_ready(last.fields.E)
                row = {
                    "phase": phase,
                    "steps": steps,
                    "repetition": repetition,
                    "seconds": time.perf_counter() - started,
                }
                measurements.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    fits = {phase: fit_phase(measurements, phase) for phase in phase_starts}
    estimated_seconds = estimate_full_forward_seconds(
        fits,
        total_steps=total_steps,
        window_steps=window_steps,
    )
    detector_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(last.detector_states)
        if hasattr(leaf, "dtype")
    ]
    detectors_finite = all(
        np.isfinite(np.asarray(leaf)).all() for leaf in detector_leaves
    )
    field_maxima = {
        "E": float(np.max(np.abs(np.asarray(last.fields.E)))),
        "H": float(np.max(np.abs(np.asarray(last.fields.H)))),
        "P_curr": float(
            np.max(np.abs(np.asarray(last.fields.dispersive_P_curr)))
        ),
    }
    field_steps_executed = 1 + 3 * args.repetitions * sum(args.steps)
    slopes_positive = all(
        float(payload["seconds_per_step"]) > 0.0 for payload in fits.values()
    )
    numerical_gate = bool(
        detectors_finite
        and all(np.isfinite(list(field_maxima.values())))
        and slopes_positive
    )
    forward_runtime_gate = estimated_seconds < MAX_FORWARD_SECONDS
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report: dict[str, object] = {
        "schema": "fdtdx_4um_parity_microbenchmark_v1",
        "status": (
            "PASS_SHORT_TIMING_ONLY"
            if numerical_gate and forward_runtime_gate
            else "BLOCKED"
        ),
        "physics_validated": False,
        "full_forward_executed": False,
        "optimizer_enabled": False,
        "polarization": args.polarization,
        "rho": args.rho,
        "gpu_preflight": gpu_snapshot,
        "jax_devices": [str(device) for device in jax.devices()],
        "cublas_runtime_version": model["cublas_runtime_version"],
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "git_status_porcelain": _git_output(["status", "--porcelain"]),
        "script_sha256": script_sha256,
        "build_seconds": build_seconds,
        "compile_seconds": compile_seconds,
        "warmup_1_step_seconds": warmup_seconds,
        "measurements": measurements,
        "fits": fits,
        "estimated_full_forward_seconds": estimated_seconds,
        "estimated_full_forward_minutes": estimated_seconds / 60.0,
        "forward_runtime_limit_seconds": MAX_FORWARD_SECONDS,
        "forward_runtime_gate": "PASS" if forward_runtime_gate else "BLOCKED",
        "numerical_timing_gate": "PASS" if numerical_gate else "BLOCKED",
        "field_maxima_after_last_partial_run": field_maxima,
        "detector_leaf_count": len(detector_leaves),
        "detectors_finite": bool(detectors_finite),
        "field_steps_executed": field_steps_executed,
        "full_forward_steps_not_executed": total_steps,
        "raw_result_in_git": False,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _write_new_external_json(args.output_json, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--polarization", choices=("Ea", "Eb"), default="Ea")
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    args = parser.parse_args()
    if not 0.0 <= args.rho <= 1.0:
        parser.error("--rho must be in [0,1]")
    if len(set(args.steps)) < 3 or any(value <= 0 for value in args.steps):
        parser.error("--steps needs at least three distinct positive values")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    args.steps = tuple(sorted(set(args.steps)))
    return args


def main() -> int:
    report = run_benchmark(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] == "PASS_SHORT_TIMING_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
