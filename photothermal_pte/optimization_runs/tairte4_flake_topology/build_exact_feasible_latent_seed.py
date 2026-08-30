#!/usr/bin/env python3
"""Construct a latent seed whose filtered threshold matches an exact DFM target.

An exact binary repair is a *physical* design.  Feeding that array directly to
the conic filter is not an inverse of the filter and can reintroduce sub-500-nm
features.  This utility solves a bounded, solver-free inverse-filter problem so
that the filtered field lies on the requested side of eta=0.5 with a finite
margin at every design node.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import optimize

from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import (
    MAPPING,
    exact_binary_audit,
)


def inverse_filter_objective(
    vector: np.ndarray,
    *,
    target: np.ndarray,
    margin: float,
    regularization: float,
) -> tuple[float, np.ndarray]:
    """Return squared filter-margin violation and its exact transpose gradient."""

    latent = np.asarray(vector, dtype=np.float64).reshape(MAPPING.shape)
    filtered = MAPPING.filtered(latent)
    signed = np.where(target, filtered - 0.5, 0.5 - filtered)
    violation = np.maximum(margin - signed, 0.0)
    normalization = float(target.size)
    value = float(np.sum(violation * violation) / normalization)
    cotangent = np.where(target, -2.0 * violation, 2.0 * violation) / normalization
    gradient = MAPPING.filter_transpose(cotangent)
    if regularization:
        delta = latent - target.astype(np.float64)
        value += float(regularization * np.mean(delta * delta))
        gradient += 2.0 * regularization * delta / normalization
    return value, gradient.ravel()


def solve_seed(
    target: np.ndarray,
    *,
    margin: float,
    regularization: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, optimize.OptimizeResult]:
    target = np.asarray(target, dtype=bool)
    if target.shape != MAPPING.shape:
        raise ValueError(f"target shape {target.shape} != mapping shape {MAPPING.shape}")
    audit, _ = exact_binary_audit(target.astype(np.float64))
    if not audit["passed"]:
        raise ValueError("target binary design does not pass the exact 500-nm audit")

    def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
        return inverse_filter_objective(
            vector,
            target=target,
            margin=margin,
            regularization=regularization,
        )

    result = optimize.minimize(
        objective,
        target.astype(np.float64).ravel(),
        method="L-BFGS-B",
        jac=True,
        bounds=optimize.Bounds(0.0, 1.0),
        options={
            "maxiter": int(maximum_iterations),
            "ftol": 1.0e-15,
            "gtol": 1.0e-10,
            "maxls": 40,
        },
    )
    return np.asarray(result.x, dtype=np.float64).reshape(MAPPING.shape), result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-rho-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--margin-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--regularization", type=float, default=1.0e-8)
    parser.add_argument("--maximum-iterations", type=int, default=500)
    args = parser.parse_args()

    if not 0.0 < args.margin < 0.5:
        raise ValueError("margin must be in (0, 0.5)")
    if args.beta <= 0.0:
        raise ValueError("beta must be positive")
    if not 0.0 <= args.margin_tolerance < args.margin:
        raise ValueError("margin tolerance must be in [0, margin)")
    with np.load(args.target_rho_npz.expanduser().resolve()) as loaded:
        if "rho" not in loaded:
            raise ValueError("target NPZ must contain rho")
        target = np.asarray(loaded["rho"], dtype=np.float64) >= 0.5
    latent, result = solve_seed(
        target,
        margin=args.margin,
        regularization=args.regularization,
        maximum_iterations=args.maximum_iterations,
    )
    filtered = MAPPING.filtered(latent)
    projected = MAPPING.physical(latent, args.beta)
    realized = projected >= 0.5
    exact, _ = exact_binary_audit(projected)
    signed_margin = np.where(target, filtered - 0.5, 0.5 - filtered)
    mismatch = int(np.count_nonzero(realized != target))
    passed = bool(
        mismatch == 0
        and exact["passed"]
        and float(np.min(signed_margin)) >= args.margin - args.margin_tolerance
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        latent=latent,
        rho=projected,
        target_binary=target,
        filtered=filtered,
        beta=np.asarray(args.beta),
        requested_filter_margin=np.asarray(args.margin),
    )
    report = {
        "schema": "exact-feasible-inverse-filter-latent-seed-v1",
        "passed": passed,
        "source_target_rho_npz": str(args.target_rho_npz.expanduser().resolve()),
        "output_npz": str(args.output_npz.expanduser().resolve()),
        "beta": float(args.beta),
        "requested_filter_margin": float(args.margin),
        "filter_margin_tolerance": float(args.margin_tolerance),
        "minimum_realized_signed_filter_margin": float(np.min(signed_margin)),
        "target_mismatch_node_count": mismatch,
        "exact_500nm_audit": exact,
        "optimizer": {
            "name": "scipy_L-BFGS-B_solver_free_inverse_filter_only",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "final_objective": float(result.fun),
            "regularization": float(args.regularization),
        },
        "note": (
            "This solve only inverts the fixed conic density filter. It does not "
            "replace Maxwell/thermal/electrical adjoints or the LD_MMA update."
        ),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
