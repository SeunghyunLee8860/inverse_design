#!/usr/bin/env python3
"""Certify Run-002 finite filter/projection JVP, VJP, and mapping-only FD."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from production_density_mapping import ProductionDensityMapping


STATUS = "VALIDATED_PRODUCTION_FINITE_FILTER_PROJECTION"
BETAS = (2.0, 4.0, 8.0, 16.0, 32.0)
FD_STEPS = (1.0e-3, 5.0e-4, 2.5e-4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def normalized(value: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(value)))
    if scale == 0.0:
        raise RuntimeError("zero validation direction")
    return np.asarray(value, float) / scale


def fields(shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    x = np.linspace(-1.0, 1.0, shape[0])
    y = np.linspace(-1.0, 1.0, shape[1])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    latent = (
        0.5
        + 0.075 * np.sin(1.3 * np.pi * xx) * np.cos(0.8 * np.pi * yy)
        + 0.045 * np.cos(0.7 * np.pi * xx + 0.9 * np.pi * yy)
        + 0.035 * np.exp(-((xx - 0.18) ** 2 + (yy + 0.11) ** 2) / 0.09)
    )
    rng = np.random.default_rng(20260806)
    directions = {
        "uniform": np.ones(shape),
        "smooth_asymmetric": np.sin(1.2 * np.pi * xx) + 0.43 * np.cos(0.7 * np.pi * yy) + 0.17 * xx * yy,
        "central_localized": np.exp(-(xx**2 + yy**2) / (2.0 * 0.075**2)),
        "design_edge_localized": np.exp(-((xx + 0.965) ** 2 + (yy - 0.24) ** 2) / (2.0 * 0.045**2)),
        "fixed_seed_random": rng.normal(size=shape),
    }
    cotangents = {
        "uniform": np.ones(shape),
        "smooth_asymmetric": 0.8 + 0.25 * np.sin(0.9 * np.pi * xx - 0.6 * np.pi * yy),
        "central_localized": 0.2 + np.exp(-((xx + 0.04) ** 2 + (yy - 0.03) ** 2) / (2.0 * 0.11**2)),
        "design_edge_localized": 0.2 + np.exp(-((xx + 0.94) ** 2 + (yy - 0.19) ** 2) / (2.0 * 0.07**2)),
        "fixed_seed_random": 0.8 + 0.1 * rng.normal(size=shape),
    }
    return latent, {key: normalized(value) for key, value in directions.items()}, cotangents


def objective(mapping: ProductionDensityMapping, latent: np.ndarray, beta: float, cotangent: np.ndarray) -> float:
    return float(np.vdot(mapping.physical(latent, beta), cotangent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing non-empty output directory")
    output.mkdir(parents=True, exist_ok=True)

    mapping = ProductionDensityMapping()
    latent, directions, cotangents = fields(mapping.shape)
    if not (float(np.min(latent)) > 0.0 and float(np.max(latent)) < 1.0):
        raise RuntimeError("validation latent field must be strictly interior")
    audit = mapping.audit()
    if audit["constant_preservation_max_abs"] >= 1.0e-14:
        raise RuntimeError("finite filter fails constant preservation")

    impulse = np.zeros(mapping.shape)
    impulse[0, mapping.shape[1] // 2] = 1.0
    impulse_filtered = mapping.filtered(impulse)
    radius_nodes = int(np.ceil(mapping.radius_m / mapping.spacing_m))
    opposite_edge_leakage = float(np.max(np.abs(impulse_filtered[-radius_nodes:, :])))
    if opposite_edge_leakage != 0.0:
        raise RuntimeError("finite filter wraps to the opposite edge")

    dot_cases: list[dict[str, object]] = []
    fd_cases: list[dict[str, object]] = []
    worst_dot = 0.0
    worst_fd_all_steps = 0.0
    for beta in BETAS:
        for name, direction in directions.items():
            cotangent = np.asarray(cotangents[name], float)
            jvp = mapping.jvp(latent, direction, beta)
            vjp = mapping.vjp(latent, cotangent, beta)
            left = float(np.vdot(jvp, cotangent))
            right = float(np.vdot(direction, vjp))
            cauchy_scale = max(
                float(np.linalg.norm(jvp) * np.linalg.norm(cotangent)),
                float(np.linalg.norm(direction) * np.linalg.norm(vjp)),
                np.finfo(float).tiny,
            )
            dot_error = abs(left - right) / cauchy_scale
            worst_dot = max(worst_dot, dot_error)
            dot_cases.append(
                {
                    "beta": beta,
                    "direction": name,
                    "jvp_dot_cotangent": left,
                    "direction_dot_vjp": right,
                    "cauchy_normalized_error": dot_error,
                }
            )
            ad = right
            for step in FD_STEPS:
                if np.min(latent - step * direction) <= 0.0 or np.max(latent + step * direction) >= 1.0:
                    raise RuntimeError("mapping-only FD would clip the latent field")
                plus = objective(mapping, latent + step * direction, beta, cotangent)
                minus = objective(mapping, latent - step * direction, beta, cotangent)
                fd = (plus - minus) / (2.0 * step)
                relative = abs(fd - ad) / max(abs(fd), abs(ad), np.finfo(float).tiny)
                worst_fd_all_steps = max(worst_fd_all_steps, relative)
                fd_cases.append(
                    {
                        "beta": beta,
                        "direction": name,
                        "step": step,
                        "adjoint_directional_derivative": ad,
                        "finite_difference_directional_derivative": fd,
                        "relative_error": relative,
                    }
                )

    finest_step = min(FD_STEPS)
    finest_errors = [
        float(case["relative_error"])
        for case in fd_cases
        if case["step"] == finest_step
    ]
    monotonic_failures: list[dict[str, object]] = []
    for beta in BETAS:
        for name in directions:
            trajectory = [
                case
                for case in fd_cases
                if case["beta"] == beta and case["direction"] == name
            ]
            errors = [float(case["relative_error"]) for case in trajectory]
            if any(right > left for left, right in zip(errors[:-1], errors[1:])):
                monotonic_failures.append(
                    {"beta": beta, "direction": name, "relative_errors": errors}
                )
    worst_fd_finest = max(finest_errors)
    gates = {
        "constant_preservation_max_abs": {"value": audit["constant_preservation_max_abs"], "limit": 1.0e-14, "passed": audit["constant_preservation_max_abs"] < 1.0e-14},
        "opposite_edge_wrap_max_abs": {"value": opposite_edge_leakage, "limit": 0.0, "passed": opposite_edge_leakage == 0.0},
        "jvp_vjp_cauchy_normalized_error": {"value": worst_dot, "limit": 1.0e-12, "passed": worst_dot < 1.0e-12},
        "mapping_only_fd_finest_step_relative_error": {"value": worst_fd_finest, "step": finest_step, "limit": 1.0e-5, "passed": worst_fd_finest < 1.0e-5},
        "mapping_only_fd_h_to_h2_monotonic": {"failure_count": len(monotonic_failures), "limit": 0, "passed": not monotonic_failures},
    }
    passed = all(value["passed"] for value in gates.values())
    npz_path = output / "production_finite_filter_projection.npz"
    np.savez_compressed(
        npz_path,
        latent=latent,
        filtered=mapping.filtered(latent),
        projected_beta2=mapping.physical(latent, 2.0),
        projected_beta8=mapping.physical(latent, 8.0),
        projected_beta32=mapping.physical(latent, 32.0),
        kernel=mapping.kernel,
        normalization=mapping.normalization,
        edge_impulse_filtered=impulse_filtered,
    )
    result = {
        "status": STATUS if passed else "FAILED_PRODUCTION_FINITE_FILTER_PROJECTION",
        "passed": passed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mapping": audit,
        "window_um": {"x": [-9.3, 9.3], "y": [-9.3, 9.3]},
        "betas": list(BETAS),
        "fd_steps": list(FD_STEPS),
        "directions": list(directions),
        "latent_range": [float(np.min(latent)), float(np.max(latent))],
        "edge_support_audit": {"impulse_index": [0, mapping.shape[1] // 2], "radius_nodes": radius_nodes, "opposite_edge_leakage_max_abs": opposite_edge_leakage},
        "gates": gates,
        "dot_cases": dot_cases,
        "fd_cases": fd_cases,
        "fd_diagnostics": {
            "worst_relative_error_over_all_steps": worst_fd_all_steps,
            "worst_relative_error_at_finest_step": worst_fd_finest,
            "monotonic_failures": monotonic_failures,
            "interpretation": "coarser centered-FD truncation error is diagnostic; the gate requires monotonic h-to-h/2 convergence and the declared finest-step tolerance",
        },
        "Maxwell_solves": 0,
        "thermal_solves": 0,
        "optimizer_started": False,
        "raw_artifact": artifact(npz_path),
    }
    result_path = output / "production_finite_filter_projection_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
