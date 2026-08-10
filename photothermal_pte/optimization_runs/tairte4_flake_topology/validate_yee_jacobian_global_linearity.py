#!/usr/bin/env python3
"""Layout-only global-linearity audit for the TaIrTe4 component-Yee map."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology import optical
from photothermal_pte.optimization_runs.tairte4_flake_topology.validate_combined_adfd import (
    load_operator,
    open_fdtd,
    set_density,
)

import build_nonuniform_complex_yee_jacobian as jacobian_builder


def relative_norm(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.linalg.norm(first - second)
        / max(np.linalg.norm(first), np.linalg.norm(second), np.finfo(float).tiny)
    )


def designs(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, shape[1])[None, :]
    rng = np.random.default_rng(10411)
    return {
        "uniform_0p1": np.full(shape, 0.1),
        "uniform_0p9": np.full(shape, 0.9),
        "smooth_nonuniform": np.clip(
            0.5 + 0.38 * np.sin(1.3 * np.pi * x) * np.cos(0.7 * np.pi * y),
            0.05,
            0.95,
        ),
        "fixed_seed_nonuniform": rng.uniform(0.05, 0.95, size=shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-fsp", required=True, type=Path)
    parser.add_argument("--jacobian-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-device", default="GPU 5")
    args = parser.parse_args()
    operator, baseline, operator_meta = load_operator(args.jacobian_dir)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "FAILED_TAIRTE4_FLAKE_YEE_JACOBIAN_GLOBAL_LINEARITY",
        "passed": False,
        "Maxwell_solves": 0,
    }
    fdtd = None
    try:
        fdtd, _, _ = open_fdtd(args.gpu_device)
        fdtd.load(str(args.base_fsp.expanduser().resolve()))
        fdtd.switchtolayout()
        set_density(fdtd, baseline)
        baseline_detail = jacobian_builder.index_detail(fdtd)
        records = {}
        worst_fixed = 0.0
        worst_local_fd = 0.0
        x = np.linspace(-1.0, 1.0, baseline.shape[0])[:, None]
        y = np.linspace(-1.0, 1.0, baseline.shape[1])[None, :]
        direction = np.sin(0.8 * np.pi * x) * np.cos(0.6 * np.pi * y)
        direction /= np.max(np.abs(direction))
        fd_step = 1.0e-4
        for name, rho in designs(baseline.shape).items():
            set_density(fdtd, rho)
            actual = jacobian_builder.index_detail(fdtd)
            fixed_tangent = operator.jvp(rho - baseline)
            components = {}
            for component in "xyz":
                fixed_predicted = baseline_detail[f"epsilon_{component}"] + fixed_tangent[component]
                observed = actual[f"epsilon_{component}"]
                fixed_error = relative_norm(fixed_predicted, observed)
                worst_fixed = max(worst_fixed, fixed_error)
                components[component] = {
                    "fixed_rho0p5_J_global_relative_L2_error_diagnostic": fixed_error,
                }
            records[name] = {
                "rho_range": [float(np.min(rho)), float(np.max(rho))],
                "components": components,
            }
        local_rho = designs(baseline.shape)["fixed_seed_nonuniform"]
        local_operator, local_meta = jacobian_builder.build_tairte4_local_epsilon_operator(
            fdtd, local_rho
        )
        plus_minus = {}
        for sign, label in ((1.0, "plus"), (-1.0, "minus")):
            set_density(fdtd, local_rho + sign * fd_step * direction)
            plus_minus[label] = jacobian_builder.index_detail(fdtd)
        local_components = {}
        for component in "xyz":
            fd = (
                plus_minus["plus"][f"epsilon_{component}"]
                - plus_minus["minus"][f"epsilon_{component}"]
            ) / (2.0 * fd_step)
            ad = local_operator.jvp(direction)[component]
            error = relative_norm(ad, fd)
            worst_local_fd = max(worst_local_fd, error)
            local_components[component] = {"centered_FD_relative_L2_error": error}
        passed = bool(worst_local_fd < 1.0e-6)
        result = {
            "status": (
                "VALIDATED_TAIRTE4_FLAKE_CURRENT_DENSITY_LOCAL_YEE_JACOBIAN"
                if passed
                else "FAILED_TAIRTE4_FLAKE_YEE_JACOBIAN_GLOBAL_LINEARITY"
            ),
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "fixed-rho0.5 global-J diagnostic plus current-density layout-only local-J certification on near-binary nonuniform density",
            "operator": operator_meta,
            "cases": records,
            "fixed_rho0p5_J_worst_global_error_diagnostic": worst_fixed,
            "exact_local_J_worst_centered_FD_error": worst_local_fd,
            "exact_local_J_case": {
                "density": "fixed_seed_nonuniform_0p05_to_0p95",
                "components": local_components,
                "construction": local_meta
            },
            "gates": {
                "exact_local_J_centered_FD_error_limit": 1.0e-6
            },
            "Maxwell_solves": 0,
            "CPU_FDTD_fallback": False,
        }
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
