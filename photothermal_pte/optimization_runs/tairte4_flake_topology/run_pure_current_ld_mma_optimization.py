#!/usr/bin/env python3
"""Pure-terminal-current, contact-anchored PTE optimization with NLopt LD_MMA.

This is deliberately a new driver rather than a modification of the
historical ``run_nlopt_mma_optimization.py``.  The older driver retains its
terminal-conductance inequality for provenance.  This driver does *not* use a
connectivity, conductance, symmetry, volume, hand-written move, Adam, or
gradient-normalization constraint.  The only active constraints are the
documented 500-nm solid/void morphology inequalities from beta=8 onward.

The electrical solve still uses its physical terminal boundary conditions:
top electrode weighting potential psi=1 and bottom electrode psi=0.  Terminal
conductance is evaluated as a diagnostic so an electrically weak/disconnected
candidate is visible in the published history, but it is not passed to NLopt.
"""

from __future__ import annotations

import argparse
import json
import os
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
    MAXIMUM_STAGE_EVALUATIONS,
    NLOPT_CONSTRAINT_TOL,
    NLOPT_FTOL_REL,
    NLOPT_XTOL_REL,
    StageEvaluator,
    emit,
    initial_manifest,
    make_optimizer,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.run_true_mma_optimization import (
    BETA_SCHEDULE,
    MORPHOLOGY_START_BETA,
    sha256,
    stage_morphology_caps,
    verify_file,
    write_json,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]


def pure_current_manifest(base_fsp: Path, base_sha256: str, jacobian_dir: Path) -> dict[str, object]:
    """Manifest with the no-connectivity optimization contract made explicit."""
    manifest = initial_manifest(base_fsp, base_sha256, jacobian_dir)
    manifest["schema"] = "pure-terminal-current-nlopt-ld-mma-raw-artifact-manifest-v1"
    manifest["optimizer"].update({
        "algorithm": "LD_MMA",
        "objective": "signed full-flake terminal PTE current",
        "terminal_conductance_constraint": False,
        "terminal_conductance_role": "diagnostic_only",
        "symmetry_constraint": False,
        "volume_constraint": False,
        "manual_move_limit": None,
        "custom_mma_update_used": False,
    })
    manifest["active_constraints"] = {
        "beta_below_8": [],
        "beta_8_and_above": ["500nm_solid_opening", "500nm_void_opening"],
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--published-dir", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--base-fsp", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--jacobian-dir", required=True, type=Path)
    parser.add_argument("--constraint-device", default="cuda:0")
    parser.add_argument("--maximum-beta-stages", type=int, default=0)
    args = parser.parse_args()

    CONTRACT.validate()
    if CONTRACT.geometry_mode != "contact_anchored":
        raise RuntimeError("pure-current LD_MMA requires contact_anchored geometry")
    base_fsp = verify_file(args.base_fsp, args.base_sha256)
    jacobian_dir = args.jacobian_dir.expanduser().resolve()
    jacobian_certificate = jacobian_dir / "component_yee_jacobian_result.json"
    if not jacobian_certificate.is_file() or not json.loads(
        jacobian_certificate.read_text()
    ).get("passed"):
        raise RuntimeError("component-Yee Jacobian certificate is missing or failed")

    raw_root = args.raw_root.expanduser().resolve()
    published = args.published_dir.expanduser().resolve()
    if (raw_root / "stage_checkpoint.npz").exists():
        raise RuntimeError(
            "NLopt internal asymptotes are not serializable; resume only from a completed "
            "beta-stage checkpoint using a dedicated restart command"
        )
    raw_root.mkdir(parents=True, exist_ok=True)
    published.mkdir(parents=True, exist_ok=True)
    events = raw_root / "events.jsonl"
    latent = np.full(MAPPING.shape, 0.5, dtype=np.float64)
    history: list[dict[str, object]] = []
    manifest = pure_current_manifest(base_fsp, args.base_sha256, jacobian_dir)
    fixed_source_power: float | None = None
    evaluation_counter = 0
    global_evaluation = 0

    for beta_index, beta in enumerate(BETA_SCHEDULE):
        if args.maximum_beta_stages and beta_index >= args.maximum_beta_stages:
            break
        stage_summary, _ = metrics(latent, beta, device=args.constraint_device)
        caps = stage_morphology_caps(
            np.asarray([
                stage_summary["smooth_solid_constraint"],
                stage_summary["smooth_void_constraint"],
            ]),
            beta,
        )
        constraint_count = 0 if beta < MORPHOLOGY_START_BETA else 2
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
            minimum_conductance_S=None,
            morphology_caps=caps,
            fixed_source_power_W=fixed_source_power,
            evaluation_counter=evaluation_counter,
            global_evaluation=global_evaluation,
            constraint_device=args.constraint_device,
            algorithm_label="NLopt LD_MMA (pure terminal current; no connectivity constraint)",
            output_slug="pure_current_ld_mma",
            include_terminal_conductance_constraint=False,
        )
        optimizer = make_optimizer(evaluator, constraint_count)
        emit(
            events,
            "nlopt_stage_start",
            beta=beta,
            algorithm="LD_MMA",
            objective="signed full-flake terminal PTE current",
            terminal_conductance_constraint=False,
            active_constraint_count=constraint_count,
            active_constraints=(
                [] if not constraint_count
                else ["500nm_solid_opening", "500nm_void_opening"]
            ),
            nlopt_version=nlopt.__version__,
            manual_move_limit=None,
            ftol_rel=NLOPT_FTOL_REL,
            xtol_rel=NLOPT_XTOL_REL,
            maxeval=MAXIMUM_STAGE_EVALUATIONS,
        )
        optimum = optimizer.optimize(latent.ravel()).reshape(MAPPING.shape)
        result_code = optimizer.last_optimize_result()
        final_point = evaluator.point(optimum.ravel())
        if (
            final_point.constraint_values.size
            and float(np.max(final_point.constraint_values)) > NLOPT_CONSTRAINT_TOL
        ):
            raise RuntimeError("NLopt stage returned an infeasible 500-nm morphology")
        if result_code == nlopt.MAXEVAL_REACHED:
            raise RuntimeError(
                "NLopt stage hit maxeval without convergence; refusing beta promotion"
            )
        if result_code < 0:
            raise RuntimeError(f"NLopt LD_MMA failed with result code {result_code}")
        latent = optimum
        fixed_source_power = evaluator.fixed_source_power_W
        evaluation_counter = evaluator.evaluation_counter
        global_evaluation = evaluator.global_evaluation
        checkpoint = raw_root / f"beta_{beta:g}_completed_checkpoint.npz"
        np.savez_compressed(checkpoint, latent=latent, beta=np.asarray(beta))
        manifest.setdefault("stage_checkpoints", {})[f"beta_{beta:g}"] = {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
            "nlopt_result_code": result_code,
            "nlopt_result_name": {
                nlopt.SUCCESS: "SUCCESS",
                nlopt.STOPVAL_REACHED: "STOPVAL_REACHED",
                nlopt.FTOL_REACHED: "FTOL_REACHED",
                nlopt.XTOL_REACHED: "XTOL_REACHED",
            }.get(result_code, str(result_code)),
            "full_physics_evaluations": evaluator.stage_full_physics_evaluations,
            "terminal_conductance_constraint": False,
        }
        write_json(published / "RAW_ARTIFACT_MANIFEST.json", manifest)
        emit(
            events,
            "nlopt_stage_complete",
            beta=beta,
            result_code=result_code,
            evaluations=evaluator.stage_full_physics_evaluations,
            final_terminal_conductance_S=float(
                final_point.result["terminal_conductance_S"]
            ),
        )

    if args.maximum_beta_stages:
        return 0
    final_rho = MAPPING.physical(latent, BETA_SCHEDULE[-1])
    exact, _ = exact_binary_audit(final_rho)
    if exact["total_bad_cell_count"] != 0 or float(
        np.mean((final_rho > 0.01) & (final_rho < 0.99))
    ) >= 0.01:
        raise RuntimeError("final continuous design did not pass binary/500-nm gates")
    binary = (final_rho >= 0.5).astype(np.float64)
    binary_audit, _ = exact_binary_audit(binary)
    if binary_audit["total_bad_cell_count"] != 0:
        raise RuntimeError("thresholded binary design failed exact 500-nm audit")
    binary_path = raw_root / "final_exact_binary_density.npz"
    np.savez_compressed(binary_path, rho=binary)
    final_output = raw_root / "final_exact_binary_evaluation"
    command = [
        sys.executable,
        "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology.evaluate_binary_objective",
        "--base-fsp", str(base_fsp),
        "--base-sha256", args.base_sha256,
        "--rho-npz", str(binary_path),
        "--output-dir", str(final_output),
        "--polarization", args.polarization,
        "--gpu-device", f"GPU {args.gpu}",
        "--cuda-device", "0",
        "--reference-objective-A", str(float(final_point.result["objective_A"])),
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    completed = subprocess.run(command, cwd=REPOSITORY, env=environment)
    final_result_path = final_output / "binary_objective_result.json"
    if completed.returncode or not final_result_path.is_file():
        raise RuntimeError("fresh exact-binary evaluation failed")
    final_result = json.loads(final_result_path.read_text())
    if not final_result.get("passed"):
        raise RuntimeError("fresh exact-binary result is not passed")
    write_json(published / "FINAL_RESULT.json", {
        "passed": True,
        "status": "VALIDATED_PURE_CURRENT_LD_MMA_EXACT_BINARY_CONTACT_ANCHORED_PTE_OPTIMIZATION",
        "polarization": args.polarization,
        "algorithm": "NLopt LD_MMA",
        "nlopt_version": nlopt.__version__,
        "objective": "signed full-flake terminal PTE current",
        "top_bottom_weighting_boundaries": {"top": 1.0, "bottom": 0.0},
        "terminal_conductance_constraint": False,
        "terminal_conductance_role": "diagnostic_only",
        "manual_move_limit": None,
        "initial_density": "uniform rho=0.5",
        "final_beta": BETA_SCHEDULE[-1],
        "full_physics_evaluations": global_evaluation,
        "binary_result": final_result,
        "exact_binary_audit": binary_audit,
        "posthoc_morphology_repair": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
