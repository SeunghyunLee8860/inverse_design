#!/usr/bin/env python3
"""Generate one bounded nominal MMA proposal from a certified latent gradient."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nlopt
import numpy as np

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
    parser.add_argument("--preparation-result", type=Path, required=True)
    parser.add_argument("--preparation-raw", type=Path, required=True)
    parser.add_argument("--preparation-raw-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--initial-step", type=float, default=0.02)
    parser.add_argument("--fixed-objective-scale-W-per-A", type=float, default=1.0e8)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    if args.iteration < 1:
        raise ValueError("iteration must be positive")
    output.mkdir(parents=True, exist_ok=True)
    prep_path = args.preparation_result.expanduser().resolve()
    prep = json.loads(prep_path.read_text())
    if prep.get("status") != "COMPLETED_SELECTED_FULL_LATENT_ADJOINT_PREPARATION" or not prep.get("passed"):
        raise RuntimeError("full-latent preparation did not pass")
    raw_path = args.preparation_raw.expanduser().resolve()
    if sha256(raw_path) != args.preparation_raw_sha256:
        raise RuntimeError("full-latent preparation raw SHA mismatch")
    raw = np.load(raw_path)
    latent0 = np.asarray(raw["latent"], float)
    gradient_A = np.asarray(raw["gradient_latent_A"], float)
    if latent0.shape != (373, 373) or gradient_A.shape != latent0.shape:
        raise RuntimeError("latent/gradient shape mismatch")
    incident_power = float(prep["incident_power_W"])
    objective0_A = float(prep["objective_A"])
    scale = float(args.fixed_objective_scale_W_per_A)
    objective0_scaled = scale * objective0_A / incident_power
    gradient_scaled = scale * gradient_A / incident_power
    x0 = latent0.ravel()
    g0 = gradient_scaled.ravel()
    if args.initial_step <= 0.0 or args.initial_step > 0.1:
        raise ValueError("initial step must be in (0, 0.1]")
    evaluations = []

    def linearized_objective(x, grad):
        if grad.size:
            grad[:] = g0
        value = objective0_scaled + float(np.dot(g0, np.asarray(x) - x0))
        evaluations.append({"linearized_scaled_objective": value, "max_abs_step": float(np.max(np.abs(np.asarray(x) - x0)))})
        return value

    opt = nlopt.opt(nlopt.LD_MMA, x0.size)
    opt.set_lower_bounds(np.zeros_like(x0))
    opt.set_upper_bounds(np.ones_like(x0))
    opt.set_initial_step(np.full_like(x0, args.initial_step))
    opt.set_maxeval(2)
    opt.set_max_objective(linearized_objective)
    candidate = np.asarray(opt.optimize(x0), float).reshape(latent0.shape)
    mapping = ProductionDensityMapping()
    beta = float(prep["beta"])
    rho0 = mapping.physical(latent0, beta)
    rho1 = mapping.physical(candidate, beta)
    if np.min(candidate) < 0.0 or np.max(candidate) > 1.0:
        raise RuntimeError("MMA candidate violates bounds")
    tag = f"{args.iteration:03d}"
    proposal_path = output / f"nominal_mma_iteration_{tag}_proposal.npz"
    np.savez_compressed(
        proposal_path,
        latent= candidate,
        latent_previous=latent0,
        filtered=mapping.filtered(candidate),
        rho=mapping.physical(candidate, beta),
        rho_previous=rho0,
        certified_gradient_latent_A=gradient_A,
    )
    result = {
        "status": f"PROPOSED_NOMINAL_MMA_ITERATION_{tag}",
        "passed": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "NLopt_LD_MMA",
        "nlopt_version": getattr(nlopt, "__version__", "unknown"),
        "nlopt_result_code": int(opt.last_optimize_result()),
        "surrogate_evaluations": len(evaluations),
        "evaluations": evaluations,
        "beta": beta,
        "fixed_nondimensionalization": {
            "formula": f"{scale:.12g} W/A * I_PTE / P_incident",
            "scale_W_per_A": scale,
            "initial_scaled_objective": objective0_scaled,
            "dynamic_gradient_rescaling": False,
        },
        "constraints": {
            "latent_bounds": [0.0, 1.0],
            "finite_nonperiodic_conic_filter_radius_nm": 500.0,
            "projection_beta": beta,
            "volume_fraction_constraint": None,
            "symmetry_constraint": None,
            "exact_binary_DRC": "not yet applied; required before fabrication promotion",
        },
        "initial_step": args.initial_step,
        "latent_previous_range": [float(np.min(latent0)), float(np.max(latent0))],
        "latent_candidate_range": [float(np.min(candidate)), float(np.max(candidate))],
        "latent_max_abs_change": float(np.max(np.abs(candidate - latent0))),
        "physical_density_previous_range": [float(np.min(rho0)), float(np.max(rho0))],
        "physical_density_candidate_range": [float(np.min(rho1)), float(np.max(rho1))],
        "physical_density_mean_previous": float(np.mean(rho0)),
        "physical_density_mean_candidate": float(np.mean(rho1)),
        "gray_fraction_previous": float(np.mean((rho0 > 0.05) & (rho0 < 0.95))),
        "gray_fraction_candidate": float(np.mean((rho1 > 0.05) & (rho1 < 0.95))),
        "actual_candidate_objective": "not evaluated by this proposal step",
        "inputs": {"preparation_result": artifact(prep_path), "preparation_NPZ": artifact(raw_path)},
        "raw_artifact": artifact(proposal_path),
        "optimizer_iteration_proposed": args.iteration,
        "optimizer_iteration_evaluated": args.iteration - 1,
        "Maxwell_solves": 0,
        "thermal_solves": 0,
        "gradient_rescaling": False,
    }
    result_path = output / f"nominal_mma_iteration_{tag}_proposal_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
