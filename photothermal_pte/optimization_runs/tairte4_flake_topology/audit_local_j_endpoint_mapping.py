#!/usr/bin/env python3
"""Layout-only endpoint audit for the current-density Yee Jacobian."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.legacy_v261_optical_support import (
    build_nonuniform_complex_yee_jacobian as builder,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.validate_combined_adfd import (
    open_fdtd,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology import optical


def relative_l2(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(
        np.linalg.norm((actual - predicted).reshape(-1))
        / max(np.linalg.norm(actual.reshape(-1)), np.finfo(float).tiny)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-fsp", required=True, type=Path)
    parser.add_argument("--rho-npz", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-device", default="GPU 5")
    parser.add_argument("--step", type=float, default=1.0e-5)
    parser.add_argument(
        "--direction-class", choices=("endpoints", "centered"), default="endpoints"
    )
    args = parser.parse_args()
    with np.load(args.rho_npz) as artifact:
        rho = np.asarray(artifact["rho"], dtype=np.float64)
    lower = rho < builder.BUILD_STEP
    upper = rho > 1.0 - builder.BUILD_STEP
    centered = ~(lower | upper)
    direction = np.zeros_like(rho)
    if args.direction_class == "endpoints":
        direction[lower] = 1.0
        direction[upper] = -1.0
    else:
        direction[centered] = 1.0
    if not np.any(direction):
        raise RuntimeError("endpoint audit density has no endpoint nodes")
    plus_trial = rho + args.step * direction
    minus_trial = rho - args.step * direction
    trials = (plus_trial,) if args.direction_class == "endpoints" else (plus_trial, minus_trial)
    if any(np.any(trial < 0.0) or np.any(trial > 1.0) for trial in trials):
        raise RuntimeError("mapping-audit direction left the unit interval")

    fdtd = None
    try:
        fdtd, _, _ = open_fdtd(args.gpu_device)
        fdtd.load(str(args.base_fsp.resolve()))
        fdtd.switchtolayout()
        operator, metadata = builder.build_tairte4_local_epsilon_operator(fdtd, rho)
        density_arguments = {
            "imported_object": optical.DESIGN_OBJECT,
            "nodes": optical.design_nodes(),
        }
        builder.set_tairte4_flake_density(fdtd, rho, **density_arguments)
        baseline = builder.index_detail(fdtd)
        builder.set_tairte4_flake_density(fdtd, plus_trial, **density_arguments)
        plus = builder.index_detail(fdtd)
        if args.direction_class == "centered":
            builder.set_tairte4_flake_density(fdtd, minus_trial, **density_arguments)
            minus = builder.index_detail(fdtd)
        predicted = operator.jvp(direction)
        components = {}
        for component in "xyz":
            if args.direction_class == "endpoints":
                actual = (
                    plus[f"epsilon_{component}"]
                    - baseline[f"epsilon_{component}"]
                ) / args.step
            else:
                actual = (
                    plus[f"epsilon_{component}"]
                    - minus[f"epsilon_{component}"]
                ) / (2.0 * args.step)
            components[component] = {
                "relative_L2_error": relative_l2(actual, predicted[component]),
                "actual_norm": float(np.linalg.norm(actual.reshape(-1))),
                "predicted_norm": float(np.linalg.norm(predicted[component].reshape(-1))),
            }
    finally:
        if fdtd is not None:
            fdtd.close()
    result = {
        "status": "VALIDATED_LOCAL_J_MAPPING"
        if max(row["relative_L2_error"] for row in components.values()) < 1.0e-4
        else "FAILED_LOCAL_J_MAPPING",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rho_shape": list(rho.shape),
        "lower_endpoint_node_count": int(np.count_nonzero(lower)),
        "upper_endpoint_node_count": int(np.count_nonzero(upper)),
        "centered_node_count": int(np.count_nonzero(centered)),
        "audit_step": args.step,
        "direction_class": args.direction_class,
        "components": components,
        "local_operator": metadata,
        "Maxwell_solves": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"].startswith("VALIDATED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
