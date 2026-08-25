#!/usr/bin/env python3
"""Bounded exact-grid GPU probe of the latent-rho-to-Maxwell VJP path.

This deliberately short run is a connectivity and resource measurement, not a
production optical gradient certificate.  The scalar loss contains no explicit
density term: it is the final electric-field energy inside the Au design slab.
Consequently a passing derivative must traverse the shared nodal mapping, Au
ADE c3 coefficients, and the checkpointed Maxwell time loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_ad_contract import (
    CHECKPOINT_CANDIDATES,
    adfd_direction_audit,
    array_sha256,
    gradient_source_audit,
    latent_directions,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_design_mapping import (
    MAPPING,
    deterministic_gray_latent,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_dynamic_checkpoint import (
    checkpoint_carry_audit,
    dynamic_checkpointed_fdtd,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_sparse_ade_checkpoint import (
    sparse_ade_checkpoint_carry_audit,
    sparse_ade_checkpointed_fdtd,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_sparse_ade_support import (
    sparse_ade_coefficient_support_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_microbenchmark import (
    _git_output,
    _write_new_external_json,
    query_and_require_idle_gpu,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_optical_controls import (
    _validate_new_external_path,
    _write_npz,
    file_sha256,
)


SCHEMA = "fdtdx_4um_parity_bounded_ad_microprobe_v1"
DEFAULT_STEPS = 4096
DEFAULT_CHECKPOINTS = 16
DEFAULT_FD_STEP = 5.0e-3
DEFAULT_DIRECTION = "uniform"
MAX_STEPS = 65_536
MAX_WALL_SECONDS = 30.0 * 60.0
MAX_DIRECTIONAL_RELATIVE_ERROR = 0.20


def symmetric_relative_error(lhs: float, rhs: float) -> float:
    return abs(lhs - rhs) / max(abs(lhs) + abs(rhs), np.finfo(np.float64).tiny)


def connectivity_gate(
    *,
    value: float,
    gradient: np.ndarray,
    ad_directional: float,
    fd_directional: float,
    value_and_grad_seconds: float,
) -> tuple[str, dict[str, bool | float]]:
    grad = np.asarray(gradient, dtype=np.float64)
    relative_error = symmetric_relative_error(ad_directional, fd_directional)
    gates: dict[str, bool | float] = {
        "finite_value": math.isfinite(value),
        "positive_field_only_value": math.isfinite(value) and value > 0.0,
        "finite_gradient": bool(np.all(np.isfinite(grad))),
        "nonzero_gradient": bool(np.max(np.abs(grad)) > 0.0),
        "finite_directionals": math.isfinite(ad_directional)
        and math.isfinite(fd_directional),
        "nonzero_directionals": abs(ad_directional) > 0.0 and abs(fd_directional) > 0.0,
        "same_directional_sign": ad_directional * fd_directional > 0.0,
        "directional_relative_error": relative_error,
        "directional_relative_error_gate": relative_error
        < MAX_DIRECTIONAL_RELATIVE_ERROR,
        "bounded_value_and_grad_runtime": math.isfinite(value_and_grad_seconds)
        and value_and_grad_seconds < MAX_WALL_SECONDS,
    }
    passed = all(
        value for key, value in gates.items() if key != "directional_relative_error"
    )
    return (
        "PASS_BOUNDED_AD_CONNECTIVITY_ONLY" if passed else "BLOCKED",
        gates,
    )


def device_memory_stats(device: Any) -> dict[str, int]:
    raw = device.memory_stats() or {}
    return {
        str(key): int(value)
        for key, value in raw.items()
        if isinstance(value, (int, np.integer))
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_json = _validate_new_external_path(args.output_json)
    output_npz = _validate_new_external_path(args.output_npz)
    if output_json == output_npz:
        raise RuntimeError("JSON and NPZ outputs must be different")

    commit_before = _git_output(["rev-parse", "HEAD"])
    status_before = _git_output(["status", "--porcelain"])
    if status_before:
        raise RuntimeError("AD microprobe requires a clean worktree")
    source_audit = gradient_source_audit()
    direction_audit = adfd_direction_audit(args.fd_step)
    if source_audit["status"] != "PASS" or direction_audit["status"] != "PASS":
        raise RuntimeError("AD source/direction prerequisite audit failed")

    gpu_snapshot = query_and_require_idle_gpu(args.gpu_uuid)
    existing_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing_visibility not in {None, "", args.gpu_uuid}:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES conflicts with requested UUID: {existing_visibility!r}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # GPU-sensitive imports follow exact UUID isolation.
    import fdtdx
    import jax
    import jax.numpy as jnp

    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_model import (
        arrays_for_density,
        build_model,
        setup_audit,
    )

    device = jax.devices("gpu")[0]
    print(
        f"phase=build polarization={args.polarization} steps={args.steps} "
        f"checkpoints={args.checkpoints}",
        flush=True,
    )
    build_started = time.perf_counter()
    model = build_model(args.polarization, backend="gpu", air_only=False)
    setup = setup_audit(model)
    if setup["status"] != "PASS":
        raise RuntimeError(f"exact physical setup audit failed: {setup}")
    dt_s = float(model["config"].time_step_duration)
    config = (
        model["config"]
        .aset("time", float(args.steps * dt_s))
        .aset(
            "gradient_config",
            fdtdx.GradientConfig(
                method="checkpointed", num_checkpoints=args.checkpoints
            ),
        )
    )
    if config.time_steps_total != args.steps:
        raise RuntimeError(
            f"requested {args.steps} steps but config resolves {config.time_steps_total}"
        )
    latent_host = np.asarray(deterministic_gray_latent(), dtype=np.float32)
    direction_host = np.asarray(latent_directions()[args.direction], dtype=np.float32)
    latent = jnp.asarray(latent_host)
    direction = jnp.asarray(direction_host)
    initial_arrays = arrays_for_density(
        model, MAPPING.jax_cell_density(latent, beta=4.0)
    )
    jax.block_until_ready(initial_arrays.fields.E)
    build_seconds = time.perf_counter() - build_started
    memory_after_build = device_memory_stats(device)
    carry_audit = checkpoint_carry_audit(initial_arrays, jax_module=jax)
    if carry_audit["status"] != "PASS":
        raise RuntimeError(f"dynamic checkpoint carry audit failed: {carry_audit}")
    sparse_regions = (
        model["slices"]["fixed_tairte4"],
        model["slices"]["au_design"],
    )
    sparse_carry_audit = None
    sparse_support_audit = None
    if args.loop_implementation == "sparse":
        sparse_carry_audit = sparse_ade_checkpoint_carry_audit(
            initial_arrays, regions=sparse_regions, jax_module=jax
        )
        sparse_support_audit = sparse_ade_coefficient_support_audit(
            initial_arrays, regions=sparse_regions, jax_module=jax
        )
        if (
            sparse_carry_audit["status"] != "PASS"
            or sparse_support_audit["status"] != "PASS"
        ):
            raise RuntimeError(
                "sparse ADE prerequisite audit failed: "
                f"carry={sparse_carry_audit}, support={sparse_support_audit}"
            )
    au_slice = model["slices"]["au_design"]

    def field_only_loss(latent_density):
        cells = MAPPING.jax_cell_density(latent_density, beta=4.0)
        arrays = arrays_for_density(model, cells)
        if args.loop_implementation == "generic":
            _, output = model["fdtdx"].run_fdtd(
                arrays=arrays,
                objects=model["placed"],
                config=config,
                key=model["key"],
                show_progress=False,
            )
        elif args.loop_implementation == "dynamic":
            _, output = dynamic_checkpointed_fdtd(
                arrays=arrays,
                objects=model["placed"],
                config=config,
                key=model["key"],
                record_detectors=True,
            )
        else:
            _, output = sparse_ade_checkpointed_fdtd(
                arrays=arrays,
                objects=model["placed"],
                config=config,
                key=model["key"],
                regions=sparse_regions,
                record_detectors=True,
            )
        e_au = output.fields.E[(slice(None),) + au_slice]
        return jnp.mean(jnp.square(e_au))

    print("phase=compile_value_and_grad", flush=True)
    compile_ad_started = time.perf_counter()
    value_and_grad_executable = (
        jax.jit(jax.value_and_grad(field_only_loss)).lower(latent).compile()
    )
    compile_value_and_grad_seconds = time.perf_counter() - compile_ad_started
    memory_after_ad_compile = device_memory_stats(device)

    print("phase=value_and_grad", flush=True)
    ad_started = time.perf_counter()
    value_device, gradient_device = value_and_grad_executable(latent)
    jax.block_until_ready(gradient_device)
    value_and_grad_seconds = time.perf_counter() - ad_started
    value = float(np.asarray(value_device))
    gradient = np.asarray(gradient_device, dtype=np.float64)
    ad_directional = float(np.sum(gradient * direction_host, dtype=np.float64))
    memory_after_value_and_grad = device_memory_stats(device)

    print("phase=compile_primal", flush=True)
    compile_primal_started = time.perf_counter()
    primal_executable = jax.jit(field_only_loss).lower(latent).compile()
    compile_primal_seconds = time.perf_counter() - compile_primal_started
    print("phase=centered_fd", flush=True)
    fd_started = time.perf_counter()
    plus_device = primal_executable(latent + args.fd_step * direction)
    minus_device = primal_executable(latent - args.fd_step * direction)
    jax.block_until_ready((plus_device, minus_device))
    centered_fd_seconds = time.perf_counter() - fd_started
    plus_value = float(np.asarray(plus_device))
    minus_value = float(np.asarray(minus_device))
    fd_directional = (plus_value - minus_value) / (2.0 * args.fd_step)
    memory_after_fd = device_memory_stats(device)

    status, gates = connectivity_gate(
        value=value,
        gradient=gradient,
        ad_directional=ad_directional,
        fd_directional=fd_directional,
        value_and_grad_seconds=value_and_grad_seconds,
    )
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    raw_arrays = {
        "latent": latent_host,
        "direction": direction_host,
        "gradient": gradient,
        "value": np.asarray(value, dtype=np.float64),
        "plus_value": np.asarray(plus_value, dtype=np.float64),
        "minus_value": np.asarray(minus_value, dtype=np.float64),
        "ad_directional": np.asarray(ad_directional, dtype=np.float64),
        "fd_directional": np.asarray(fd_directional, dtype=np.float64),
    }
    _write_npz(output_npz, raw_arrays)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "scope": "short_exact_grid_AD_connectivity_and_resource_probe_only",
        "production_gradient_validated": False,
        "full_40_period_gradient_executed": False,
        "optimizer_enabled": False,
        "polarization": args.polarization,
        "grid_shape": list(model["grid"].shape),
        "steps": args.steps,
        "simulated_periods": args.steps
        / model["plan"]["time"]["time_steps_total"]
        * 40.0,
        "checkpoint_method": "checkpointed",
        "loop_implementation": args.loop_implementation,
        "checkpoint_carry_audit": carry_audit,
        "sparse_ADE_carry_audit": sparse_carry_audit,
        "sparse_ADE_support_audit": sparse_support_audit,
        "checkpoints": args.checkpoints,
        "beta": 4.0,
        "direction": args.direction,
        "fd_step": args.fd_step,
        "loss": "mean(final_E_squared_inside_Au); no_explicit_rho_term",
        "value": value,
        "gradient_min": float(np.min(gradient)),
        "gradient_max": float(np.max(gradient)),
        "gradient_l2": float(np.linalg.norm(gradient)),
        "gradient_nonzero_count": int(np.count_nonzero(gradient)),
        "ad_directional": ad_directional,
        "fd_directional": fd_directional,
        "plus_value": plus_value,
        "minus_value": minus_value,
        "gates": gates,
        "timing_seconds": {
            "build": build_seconds,
            "compile_value_and_grad": compile_value_and_grad_seconds,
            "value_and_grad": value_and_grad_seconds,
            "compile_primal": compile_primal_seconds,
            "two_centered_FD_forwards": centered_fd_seconds,
        },
        "device_memory_stats": {
            "after_build": memory_after_build,
            "after_AD_compile": memory_after_ad_compile,
            "after_value_and_grad": memory_after_value_and_grad,
            "after_centered_FD": memory_after_fd,
        },
        "gpu_preflight": gpu_snapshot,
        "jax_devices": [str(value) for value in jax.devices()],
        "cublas_runtime_version": model["cublas_runtime_version"],
        "git_commit": commit_before,
        "git_status_porcelain_before": status_before,
        "script_sha256": script_sha256,
        "gradient_source_sha256": source_audit["source_sha256"],
        "mapping_sha256": MAPPING.coefficient_sha256(),
        "latent_sha256": array_sha256(
            latent_host, label="fdtdx-parity-ad-microprobe-latent-v1"
        ),
        "direction_sha256": array_sha256(
            direction_host,
            label=f"fdtdx-parity-adfd-{args.direction}-v1",
        ),
        "raw_npz_path": str(output_npz),
        "raw_npz_sha256": file_sha256(output_npz),
        "raw_result_in_git": False,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _write_new_external_json(output_json, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--polarization", choices=("Ea", "Eb"), default="Ea")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--checkpoints", type=int, default=DEFAULT_CHECKPOINTS)
    parser.add_argument(
        "--loop-implementation",
        choices=("generic", "dynamic", "sparse"),
        default="generic",
    )
    parser.add_argument("--fd-step", type=float, default=DEFAULT_FD_STEP)
    parser.add_argument(
        "--direction", choices=tuple(latent_directions()), default=DEFAULT_DIRECTION
    )
    args = parser.parse_args()
    if not 2 <= args.steps <= MAX_STEPS:
        parser.error(f"--steps must be in [2,{MAX_STEPS}]")
    if args.checkpoints not in CHECKPOINT_CANDIDATES:
        parser.error(f"--checkpoints must be one of {CHECKPOINT_CANDIDATES}")
    if args.checkpoints >= args.steps:
        parser.error("--checkpoints must be smaller than --steps")
    if not math.isfinite(args.fd_step) or not 0.0 < args.fd_step <= 2.0e-2:
        parser.error("--fd-step must be finite and in (0,0.02]")
    return args


def main() -> int:
    report = run_probe(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] == "PASS_BOUNDED_AD_CONNECTIVITY_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
