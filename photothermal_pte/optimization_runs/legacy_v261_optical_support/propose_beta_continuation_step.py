#!/usr/bin/env python3
"""Propose one stateful constrained-MMA beta-continuation step."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from beta_continuation_support import (
    design_metrics,
    initialize_mma_state,
    load_mma_state,
    mma_step,
    save_mma_state,
)
from production_density_mapping import ProductionDensityMapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-result", type=Path, required=True)
    parser.add_argument("--evaluation-raw", type=Path, required=True)
    parser.add_argument("--mma-state", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-iteration", type=int, required=True)
    parser.add_argument("--move", type=float, default=0.02)
    parser.add_argument("--constraint-reduction", type=float, default=0.01)
    parser.add_argument("--fixed-objective-scale-W-per-A", type=float, default=1.0e8)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = args.evaluation_result.expanduser().resolve()
    raw_path = args.evaluation_raw.expanduser().resolve()
    result = json.loads(result_path.read_text())
    if not result.get("passed"):
        raise RuntimeError("input forward/adjoint evaluation did not pass")
    if sha256(raw_path) != result["raw_artifact"]["sha256"]:
        raise RuntimeError("evaluation NPZ SHA mismatch")
    data = np.load(raw_path)
    latent = np.asarray(data["latent"], float)
    gradient_A = np.asarray(data["gradient_latent_A"], float)
    beta = float(result["beta"])
    mapping = ProductionDensityMapping()
    metrics, arrays = design_metrics(latent, beta, mapping)
    names = ("solid", "void", "gray")
    current = np.asarray([
        metrics["smooth_solid_constraint"],
        metrics["smooth_void_constraint"],
        metrics["binarization_metric_mean_4rho1mrho"],
    ], float)
    reduction = float(args.constraint_reduction)
    if not 0.0 < reduction < 0.25:
        raise ValueError("constraint reduction must be in (0, 0.25)")
    caps = np.maximum(current * (1.0 - reduction), 1.0e-12)
    constraint_values = current / caps - 1.0
    constraint_gradients = np.stack(
        [arrays[f"gradient_{name}"].ravel() / cap for name, cap in zip(names, caps)]
    )
    incident = float(result["incident_power_W"])
    scale = float(args.fixed_objective_scale_W_per_A)
    objective_gradient_minimize = -scale * gradient_A.ravel() / incident
    if args.mma_state:
        state_path = args.mma_state.expanduser().resolve()
        state = load_mma_state(state_path)
        state_input = artifact(state_path)
    else:
        state = initialize_mma_state(latent)
        state_input = None
    candidate_flat, candidate_state, diagnostics = mma_step(
        latent.ravel(),
        objective_gradient_minimize,
        constraint_values,
        constraint_gradients,
        state,
        move=float(args.move),
    )
    candidate = candidate_flat.reshape(latent.shape)
    candidate_metrics, candidate_arrays = design_metrics(candidate, beta, mapping)
    current_exact = metrics["exact_binary_audit"]
    candidate_exact = candidate_metrics["exact_binary_audit"]
    current_bad = int(current_exact["solid_bad_cell_count"]) + int(current_exact["void_bad_cell_count"])
    candidate_bad = int(candidate_exact["solid_bad_cell_count"]) + int(candidate_exact["void_bad_cell_count"])
    drc_backtracking_alpha = 1.0
    if candidate_bad > current_bad:
        # Exact morphology is nondifferentiable and is not inserted into the
        # MMA Jacobian.  A deterministic line search prevents an accepted
        # proposal from increasing its total exact violation count; it does
        # not repair or threshold the continuous design.
        for alpha in (0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125):
            trial = latent + alpha * (candidate - latent)
            trial_metrics, trial_arrays = design_metrics(trial, beta, mapping)
            trial_exact = trial_metrics["exact_binary_audit"]
            trial_bad = int(trial_exact["solid_bad_cell_count"]) + int(trial_exact["void_bad_cell_count"])
            if trial_bad <= current_bad:
                candidate = trial
                candidate_metrics = trial_metrics
                candidate_arrays = trial_arrays
                candidate_bad = trial_bad
                drc_backtracking_alpha = alpha
                break
        else:
            raise RuntimeError("no non-worsening exact-DRC proposal found within the MMA direction")
    proposal = output / "beta_continuation_proposal.npz"
    np.savez_compressed(
        proposal,
        latent=candidate,
        latent_previous=latent,
        filtered=candidate_arrays["filtered"],
        rho=candidate_arrays["rho"],
        rho_previous=arrays["rho"],
        certified_gradient_latent_A=gradient_A,
        constraint_caps=caps,
    )
    state_candidate_path = output / "mma_state_candidate.npz"
    save_mma_state(state_candidate_path, candidate_state)
    result_out = {
        "status": "PROPOSED_BETA_CONTINUATION_STEP",
        "passed": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "global_iteration_proposed": args.global_iteration,
        "beta": beta,
        "algorithm": "stateful Svanberg MMA separable convex subproblem",
        "asymptote_state_persisted": True,
        "move_limit": float(args.move),
        "fixed_nondimensionalization": {
            "scale_W_per_A": scale,
            "formula": "scale * I_PTE / P_incident",
            "dynamic_gradient_rescaling": False,
        },
        "manufacturability_contract": {
            "minimum_solid_feature_nm": 500.0,
            "minimum_void_feature_nm": 500.0,
            "finite_nonperiodic_conic_filter_radius_nm": 500.0,
            "Wang_thresholds": {"eta_eroded": 0.75, "eta_dilated": 0.25},
            "smooth_constraints": "Zhou solid and void indicators plus grayness",
            "exact_binary_audit": "radius-250-nm disk opening; audit only, no repair",
        },
        "constraint_reduction_requested": reduction,
        "constraint_names": list(names),
        "constraint_current": current.tolist(),
        "constraint_caps": caps.tolist(),
        "constraint_normalized_values": constraint_values.tolist(),
        "candidate_offline_metrics": candidate_metrics,
        "exact_DRC_backtracking": {
            "applied": drc_backtracking_alpha < 1.0,
            "alpha": drc_backtracking_alpha,
            "current_total_bad_cells": current_bad,
            "candidate_total_bad_cells": candidate_bad,
            "continuous_line_search_only": True,
            "posthoc_binary_repair": False,
        },
        "mma_diagnostics": diagnostics,
        "inputs": {
            "evaluation_result": artifact(result_path),
            "evaluation_NPZ": artifact(raw_path),
            "mma_state": state_input,
        },
        "raw_artifact": artifact(proposal),
        "candidate_state": artifact(state_candidate_path),
        "actual_candidate_objective": "not evaluated",
        "Maxwell_solves": 0,
        "thermal_solves": 0,
    }
    out_path = output / "beta_continuation_proposal_result.json"
    out_path.write_text(json.dumps(result_out, indent=2) + "\n")
    print(json.dumps(result_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
