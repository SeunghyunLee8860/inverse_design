#!/usr/bin/env python3
"""Constrained PTE topology optimization with NLopt AUGLAG + L-BFGS.

This driver deliberately contains no custom MMA update, manual move limit,
gradient-direction normalization, Adam state, or post-update clipping.  The
local L-BFGS optimizer chooses its step through line-search evaluations of the
actual full-physics objective.  AUGLAG supplies the nonlinear-constraint terms.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import nlopt
import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import (
    MAPPING,
    exact_binary_audit,
    metrics,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.run_nlopt_mma_optimization import (
    StageEvaluator,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.run_true_mma_optimization import (
    BETA_SCHEDULE,
    FULL_SOLID_TERMINAL_CONDUCTANCE_S,
    MORPHOLOGY_START_BETA,
    evaluate,
    record_manifest_entry,
    sha256,
    stage_morphology_caps,
    verify_file,
    write_json,
)


MAXIMUM_STAGE_EVALUATIONS = 24
OUTER_FTOL_REL = 2.0e-3
OUTER_XTOL_REL = 1.0e-6
LOCAL_FTOL_REL = 5.0e-4
LOCAL_XTOL_REL = 1.0e-7
LOCAL_MAXIMUM_EVALUATIONS = 16
CONSTRAINT_TOLERANCE = 1.0e-5


def make_optimizer(evaluator: StageEvaluator, constraint_count: int) -> nlopt.opt:
    variable_count = int(np.prod(MAPPING.shape))
    local = nlopt.opt(nlopt.LD_LBFGS, variable_count)
    local.set_lower_bounds(np.zeros(variable_count))
    local.set_upper_bounds(np.ones(variable_count))
    local.set_vector_storage(10)
    local.set_ftol_rel(LOCAL_FTOL_REL)
    local.set_xtol_rel(LOCAL_XTOL_REL)
    local.set_maxeval(LOCAL_MAXIMUM_EVALUATIONS)

    optimizer = nlopt.opt(nlopt.LD_AUGLAG, variable_count)
    optimizer.set_lower_bounds(np.zeros(variable_count))
    optimizer.set_upper_bounds(np.ones(variable_count))
    optimizer.set_min_objective(evaluator.objective)
    optimizer.add_inequality_mconstraint(
        evaluator.constraints,
        np.full(constraint_count, CONSTRAINT_TOLERANCE),
    )
    optimizer.set_local_optimizer(local)
    optimizer.set_ftol_rel(OUTER_FTOL_REL)
    optimizer.set_xtol_rel(OUTER_XTOL_REL)
    optimizer.set_maxeval(MAXIMUM_STAGE_EVALUATIONS)
    return optimizer


def initial_manifest(base_fsp: Path, base_sha256: str, jacobian_dir: Path) -> dict[str, object]:
    certificate = jacobian_dir / "component_yee_jacobian_result.json"
    return {
        "schema": "nlopt-auglag-lbfgs-contact-anchored-artifact-manifest-v1",
        "raw_artifacts_committed_to_git": False,
        "optimizer": {
            "outer": "NLopt LD_AUGLAG",
            "local": "NLopt LD_LBFGS",
            "version": nlopt.__version__,
            "manual_move_limit": None,
            "custom_mma_update_used": False,
            "line_search_uses_full_physics_objective": True,
        },
        "base_FSP": {
            "path": str(base_fsp),
            "size_bytes": base_fsp.stat().st_size,
            "sha256": base_sha256,
        },
        "component_Yee_Jacobian": {
            "path": str(jacobian_dir),
            "certificate": str(certificate),
            "certificate_sha256": sha256(certificate),
        },
        "evaluations": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--published-dir", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--base-fsp", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--jacobian-dir", required=True, type=Path)
    parser.add_argument("--connectivity-fraction", type=float, default=0.10)
    parser.add_argument("--constraint-device", default="cuda:0")
    args = parser.parse_args()
    CONTRACT.validate()
    if CONTRACT.geometry_mode != "contact_anchored":
        raise RuntimeError("AUGLAG/L-BFGS production requires contact_anchored geometry")
    base_fsp = verify_file(args.base_fsp, args.base_sha256)
    jacobian_dir = args.jacobian_dir.expanduser().resolve()
    certificate = jacobian_dir / "component_yee_jacobian_result.json"
    if not certificate.is_file() or not json.loads(certificate.read_text()).get("passed"):
        raise RuntimeError("component-Yee Jacobian certificate is missing or failed")

    raw_root = args.raw_root.expanduser().resolve()
    published = args.published_dir.expanduser().resolve()
    if raw_root.exists() and any(raw_root.iterdir()):
        raise RuntimeError("fresh AUGLAG/L-BFGS run refuses a nonempty raw root")
    raw_root.mkdir(parents=True, exist_ok=True)
    published.mkdir(parents=True, exist_ok=True)
    events = raw_root / "events.jsonl"
    history: list[dict[str, object]] = []
    manifest = initial_manifest(base_fsp, args.base_sha256, jacobian_dir)
    latent = np.full(MAPPING.shape, 0.5, dtype=np.float64)
    fixed_source_power: float | None = None
    evaluation_counter = 0
    global_evaluation = 0
    minimum_conductance = args.connectivity_fraction * FULL_SOLID_TERMINAL_CONDUCTANCE_S
    final_point = None

    for beta in BETA_SCHEDULE:
        initial_summary, _ = metrics(latent, beta, device=args.constraint_device)
        caps = stage_morphology_caps(np.asarray([
            initial_summary["smooth_solid_constraint"],
            initial_summary["smooth_void_constraint"],
        ]), beta)
        constraint_count = 1 if beta < MORPHOLOGY_START_BETA else 3
        evaluator = StageEvaluator(
            beta=beta,
            polarization=args.polarization,
            gpu=args.gpu,
            raw_root=raw_root,
            published=published,
            events=events,
            history=history,
            manifest=manifest,
            base_fsp=base_fsp,
            base_sha256=args.base_sha256,
            jacobian_dir=jacobian_dir,
            minimum_conductance_S=minimum_conductance,
            morphology_caps=caps,
            fixed_source_power_W=fixed_source_power,
            evaluation_counter=evaluation_counter,
            global_evaluation=global_evaluation,
            constraint_device=args.constraint_device,
            algorithm_label="NLopt LD_AUGLAG + LD_LBFGS",
            output_slug="auglag_lbfgs",
        )
        optimizer = make_optimizer(evaluator, constraint_count)
        optimum = optimizer.optimize(latent.ravel()).reshape(MAPPING.shape)
        result_code = optimizer.last_optimize_result()
        final_point = evaluator.point(optimum.ravel())
        if result_code < 0:
            raise RuntimeError(f"NLopt AUGLAG/L-BFGS failed with result code {result_code}")
        if result_code == nlopt.MAXEVAL_REACHED:
            raise RuntimeError("AUGLAG/L-BFGS stage reached maxeval without convergence")
        if np.max(final_point.constraint_values) > CONSTRAINT_TOLERANCE:
            raise RuntimeError("AUGLAG/L-BFGS returned an infeasible stage point")
        previous = latent
        latent = optimum
        fixed_source_power = evaluator.fixed_source_power_W
        evaluation_counter = evaluator.evaluation_counter
        global_evaluation = evaluator.global_evaluation
        stage = {
            "passed": True,
            "status": "VALIDATED_AUGLAG_LBFGS_STAGE",
            "beta": beta,
            "nlopt_result_code": result_code,
            "full_physics_evaluations": evaluator.stage_full_physics_evaluations,
            "maximum_absolute_stage_change": float(np.max(np.abs(latent - previous))),
            "rms_stage_change": float(np.sqrt(np.mean((latent - previous) ** 2))),
            "objective_at_reference_power_A": history[-1]["objective_at_reference_power_A"],
            "maximum_constraint_value": float(np.max(final_point.constraint_values)),
            "gray_fraction_0p01_0p99": final_point.summary["gray_fraction_0p01_0p99"],
            "binarization": final_point.summary["binarization_mean_4rho1mrho"],
        }
        write_json(published / f"beta_{beta:g}_accepted_stage.json", stage)
        checkpoint = raw_root / f"beta_{beta:g}_completed_checkpoint.npz"
        np.savez_compressed(checkpoint, latent=latent, beta=np.asarray(beta))
        manifest.setdefault("stage_checkpoints", {})[f"beta_{beta:g}"] = {
            "path": str(checkpoint), "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint), **stage,
        }
        write_json(published / "RAW_ARTIFACT_MANIFEST.json", manifest)

    assert final_point is not None
    final_rho = MAPPING.physical(latent, BETA_SCHEDULE[-1])
    exact, _ = exact_binary_audit(final_rho)
    gray = float(np.mean((final_rho > 0.01) & (final_rho < 0.99)))
    if exact["total_bad_cell_count"] != 0 or gray >= 0.01:
        raise RuntimeError("final continuous design did not pass binary/500 nm gates")
    binary = (final_rho >= 0.5).astype(np.float64)
    binary_audit, _ = exact_binary_audit(binary)
    if binary_audit["total_bad_cell_count"] != 0:
        raise RuntimeError("thresholded binary failed exact 500 nm audit")
    binary_path = raw_root / "final_exact_binary_density.npz"
    np.savez_compressed(binary_path, rho=binary)
    output = raw_root / "final_exact_binary_evaluation"
    result, _, _ = evaluate(
        binary,
        polarization=args.polarization,
        output=output,
        gpu=args.gpu,
        events=events,
        base_fsp=base_fsp,
        base_sha256=args.base_sha256,
        jacobian_dir=jacobian_dir,
    )
    write_json(published / "FINAL_RESULT.json", {
        "passed": True,
        "status": "VALIDATED_AUGLAG_LBFGS_EXACT_BINARY_PTE_OPTIMIZATION",
        "polarization": args.polarization,
        "algorithm": "NLopt LD_AUGLAG + LD_LBFGS",
        "manual_move_limit": None,
        "initial_density": "uniform rho=0.5",
        "full_physics_evaluations": global_evaluation,
        "exact_binary_audit": binary_audit,
        "binary_objective_result": result,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
