#!/usr/bin/env python3
"""Bounded exact-grid GPU probe of the sparse sliced reversible FDTDX VJP."""

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
    adfd_direction_audit,
    array_sha256,
    latent_directions,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_design_mapping import (
    MAPPING,
    deterministic_gray_latent,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_gradient_detectors import (
    PRODUCTION_GRADIENT_DETECTORS,
    filter_gradient_detectors,
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
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_reversible_sparse_sliced_vjp import (
    reversible_ade_cpml_phasor_sparse_sliced_fdtd,
    reversible_sparse_slice_checkpoint_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_sparse_ade_support import (
    sparse_ade_coefficient_support_audit,
)


SCHEMA = "fdtdx_4um_parity_reversible_sparse_microprobe_v1"
DEFAULT_STEPS = 4096
DEFAULT_SLICE_STEPS = 256
DEFAULT_FD_STEP = 5.0e-3
DEFAULT_DIRECTION = "uniform"
MAX_STEPS = 16_384
MAX_DIRECTIONAL_RELATIVE_ERROR = 5.0e-3
MAX_MEASURED_WALL_SECONDS = 30.0 * 60.0
FULL_STEPS = 256_163
USER_RUNTIME_GATE_SECONDS = 30.0 * 60.0
HERE = Path(__file__).resolve().parent
REVERSIBLE_SOURCE_FILES = (
    HERE / "fdtdx_parity_reversible_ade_step.py",
    HERE / "fdtdx_parity_reversible_cpml.py",
    HERE / "fdtdx_parity_reversible_sparse_sliced_vjp.py",
)


def reversible_source_audit() -> dict[str, Any]:
    """Hash and minimally identify the custom reverse implementation."""

    missing = [str(path) for path in REVERSIBLE_SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing reversible source files: {missing}")
    source = {
        path.name: path.read_text(encoding="utf-8") for path in REVERSIBLE_SOURCE_FILES
    }
    checks = {
        "ADE_previous_state_reconstructed": (
            "P_prev = jnp.where" in source["fdtdx_parity_reversible_ade_step.py"]
        ),
        "CPML_psi_reconstructed": (
            "reverse_cpml_auxiliary" in source["fdtdx_parity_reversible_cpml.py"]
        ),
        "sparse_exact_slice_reset_present": (
            "expand(P_curr_regional)"
            in source["fdtdx_parity_reversible_sparse_sliced_vjp.py"]
            and "support_audit" in source["fdtdx_parity_reversible_sparse_sliced_vjp.py"]
        ),
    }
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in REVERSIBLE_SOURCE_FILES
    }
    return {
        "schema": "fdtdx_4um_reversible_source_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_sha256": hashes,
    }


def symmetric_relative_error(lhs: float, rhs: float) -> float:
    return abs(lhs - rhs) / max(abs(lhs) + abs(rhs), np.finfo(np.float64).tiny)


def reversible_connectivity_gate(
    *,
    value: float,
    gradient: np.ndarray,
    ad_directional: float,
    fd_directional: float,
    value_and_grad_seconds: float,
) -> tuple[str, dict[str, bool | float]]:
    gradient_array = np.asarray(gradient, dtype=np.float64)
    relative_error = symmetric_relative_error(ad_directional, fd_directional)
    gates: dict[str, bool | float] = {
        "finite_positive_value": math.isfinite(value) and value > 0.0,
        "finite_gradient": bool(np.all(np.isfinite(gradient_array))),
        "nonzero_gradient": bool(np.max(np.abs(gradient_array)) > 0.0),
        "finite_directionals": math.isfinite(ad_directional)
        and math.isfinite(fd_directional),
        "nonzero_directionals": abs(ad_directional) > 0.0 and abs(fd_directional) > 0.0,
        "same_directional_sign": ad_directional * fd_directional > 0.0,
        "directional_relative_error": relative_error,
        "directional_relative_error_gate": relative_error
        < MAX_DIRECTIONAL_RELATIVE_ERROR,
        "bounded_measured_value_and_grad_runtime": (
            math.isfinite(value_and_grad_seconds)
            and value_and_grad_seconds < MAX_MEASURED_WALL_SECONDS
        ),
    }
    passed = all(
        observed
        for name, observed in gates.items()
        if name != "directional_relative_error"
    )
    return (
        "PASS_BOUNDED_REVERSIBLE_AD_CONNECTIVITY_ONLY" if passed else "BLOCKED",
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
        raise RuntimeError("reversible microprobe requires a clean worktree")
    source_audit = reversible_source_audit()
    direction_audit = adfd_direction_audit(args.fd_step)
    if source_audit["status"] != "PASS" or direction_audit["status"] != "PASS":
        raise RuntimeError(
            f"reversible source/direction prerequisite failed: "
            f"source={source_audit}, direction={direction_audit}"
        )

    gpu_snapshot = query_and_require_idle_gpu(args.gpu_uuid)
    existing_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing_visibility not in {None, "", args.gpu_uuid}:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES conflicts with requested UUID: {existing_visibility!r}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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
        f"slice_steps={args.slice_steps}",
        flush=True,
    )
    build_started = time.perf_counter()
    model = build_model(args.polarization, backend="gpu", air_only=False)
    setup = setup_audit(model)
    if setup["status"] != "PASS":
        raise RuntimeError(f"exact physical setup audit failed: {setup}")
    gradient_base, gradient_placed, detector_audit = filter_gradient_detectors(
        model["base"],
        model["placed"],
        keep_names=PRODUCTION_GRADIENT_DETECTORS,
        jax_module=jax,
    )
    model = dict(model)
    model["base"] = gradient_base
    model["placed"] = gradient_placed
    dt_s = float(model["config"].time_step_duration)
    config = (
        model["config"]
        .aset("time", float(args.steps * dt_s))
        .aset("gradient_config", None)
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
        model,
        MAPPING.jax_cell_density(latent, beta=4.0),
    )
    jax.block_until_ready(initial_arrays.fields.E)
    sparse_regions = (
        model["slices"]["fixed_tairte4"],
        model["slices"]["au_design"],
    )
    support_audit = sparse_ade_coefficient_support_audit(
        initial_arrays,
        regions=sparse_regions,
        jax_module=jax,
    )
    slice_audit = reversible_sparse_slice_checkpoint_audit(
        initial_arrays,
        regions=sparse_regions,
        jax_module=jax,
    )
    if support_audit["status"] != "PASS" or slice_audit["status"] != "PASS":
        raise RuntimeError(
            f"sparse prerequisite failed: support={support_audit}, slice={slice_audit}"
        )
    build_seconds = time.perf_counter() - build_started
    memory_after_build = device_memory_stats(device)
    au_slice = model["slices"]["au_design"]

    def field_only_loss(latent_density):
        cells = MAPPING.jax_cell_density(latent_density, beta=4.0)
        arrays = arrays_for_density(model, cells)
        _, output = reversible_ade_cpml_phasor_sparse_sliced_fdtd(
            arrays=arrays,
            objects=model["placed"],
            config=config,
            key=model["key"],
            steps_per_slice=args.slice_steps,
            regions=sparse_regions,
            support_audit=support_audit,
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

    status, gates = reversible_connectivity_gate(
        value=value,
        gradient=gradient,
        ad_directional=ad_directional,
        fd_directional=fd_directional,
        value_and_grad_seconds=value_and_grad_seconds,
    )
    projected_single_seconds = value_and_grad_seconds * FULL_STEPS / args.steps
    projected_dual_seconds = 2.0 * projected_single_seconds
    runtime_projection = {
        "method": "single_depth_linear_ratio_not_a_certificate",
        "single_polarization_seconds": projected_single_seconds,
        "single_polarization_minutes": projected_single_seconds / 60.0,
        "two_polarization_seconds": projected_dual_seconds,
        "two_polarization_minutes": projected_dual_seconds / 60.0,
        "single_polarization_under_30_minutes": (
            projected_single_seconds < USER_RUNTIME_GATE_SECONDS
        ),
        "two_polarization_under_30_minutes": (
            projected_dual_seconds < USER_RUNTIME_GATE_SECONDS
        ),
    }

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
        "scope": "short_exact_grid_reversible_AD_connectivity_and_resource_probe_only",
        "production_gradient_validated": False,
        "full_40_period_gradient_executed": False,
        "optimizer_enabled": False,
        "polarization": args.polarization,
        "grid_shape": list(model["grid"].shape),
        "steps": args.steps,
        "slice_steps": args.slice_steps,
        "num_slices": math.ceil(args.steps / args.slice_steps),
        "simulated_periods": args.steps / FULL_STEPS * 40.0,
        "detector_profile": "production_Au_and_TaIrTe4_late_phasors",
        "gradient_detector_audit": detector_audit,
        "sparse_ADE_support_audit": support_audit,
        "reversible_slice_checkpoint_audit": slice_audit,
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
        "runtime_projection": runtime_projection,
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
        "jax_devices": [str(observed) for observed in jax.devices()],
        "cublas_runtime_version": model["cublas_runtime_version"],
        "git_commit": commit_before,
        "git_status_porcelain_before": status_before,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reversible_source_audit": source_audit,
        "mapping_sha256": MAPPING.coefficient_sha256(),
        "latent_sha256": array_sha256(
            latent_host,
            label="fdtdx-parity-reversible-microprobe-latent-v1",
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
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--slice-steps", type=int, default=DEFAULT_SLICE_STEPS)
    parser.add_argument("--fd-step", type=float, default=DEFAULT_FD_STEP)
    parser.add_argument(
        "--direction",
        choices=tuple(latent_directions()),
        default=DEFAULT_DIRECTION,
    )
    args = parser.parse_args()
    if not 2 <= args.steps <= MAX_STEPS:
        parser.error(f"--steps must be in [2,{MAX_STEPS}]")
    if not 1 <= args.slice_steps <= args.steps:
        parser.error("--slice-steps must be in [1,steps]")
    if not math.isfinite(args.fd_step) or not 0.0 < args.fd_step <= 2.0e-2:
        parser.error("--fd-step must be finite and in (0,0.02]")
    return args


def main() -> int:
    report = run_probe(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
