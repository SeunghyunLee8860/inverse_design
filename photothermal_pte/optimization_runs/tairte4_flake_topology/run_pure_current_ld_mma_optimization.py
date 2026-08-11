#!/usr/bin/env python3
"""Pure-terminal-current, contact-anchored PTE optimization with NLopt LD_MMA.

This is deliberately a new driver rather than a modification of the
historical ``run_nlopt_mma_optimization.py``.  The older driver retains its
terminal-conductance inequality for provenance.  This driver does *not* use a
connectivity, conductance, symmetry, volume, hand-written move, Adam, or
gradient-normalization constraint.  The only active constraints are the
documented 500-nm solid/void morphology inequalities from beta=1 onward.

The electrical solve still uses its physical terminal boundary conditions:
top electrode weighting potential psi=1 and bottom electrode psi=0.  Terminal
conductance is evaluated as a diagnostic so an electrically weak/disconnected
candidate is visible in the published history, but it is not passed to NLopt.
"""

from __future__ import annotations

import argparse
import json
from math import tanh
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
    MORPHOLOGY_TARGET_CAP,
    sha256,
    verify_file,
    write_json,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
CODE_MANIFEST = REPOSITORY / "photothermal_pte/optimization_runs/PURE_CURRENT_LD_MMA_CODE_MANIFEST.json"
TARGET_INITIAL_PHYSICAL_DENSITY_STEP = 0.025
BASE_RHO_INIT_AT_BETA1 = 10.0
PURE_CURRENT_MORPHOLOGY_START_BETA = 1.0
PURE_CURRENT_MORPHOLOGY_TARGET_CAP = {
    1.0: 8.0e-3,
    2.0: 5.0e-3,
    4.0: 3.0e-3,
    **MORPHOLOGY_TARGET_CAP,
}
# The calibrated first CCSA trial is physically meaningful but smaller than
# the shared driver's historical 1e-7 x tolerance.  This is a numerical
# termination tolerance, not a per-iteration move restriction.
PURE_CURRENT_NLOPT_XTOL_REL = 1.0e-9
STAGE_PLATEAU_WINDOW = 3
STAGE_PLATEAU_RELATIVE_CHANGE = 2.0e-3


def midpoint_projection_derivative(beta: float) -> float:
    """Return d(projected density)/d(filtered latent) at rho=0.5."""
    return beta / (2.0 * tanh(0.5 * beta))


def verify_optimizer_code_manifest() -> dict[str, object]:
    """Fail closed if the audited optimizer source bundle has changed."""
    if not CODE_MANIFEST.is_file():
        raise RuntimeError("pure-current LD_MMA code-manifest is missing")
    payload = json.loads(CODE_MANIFEST.read_text())
    if not payload.get("passed"):
        raise RuntimeError("pure-current LD_MMA code-manifest is not approved")
    expected = payload.get("source_sha256")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("pure-current LD_MMA code-manifest has no source hashes")
    actual: dict[str, str] = {}
    for relative, digest in expected.items():
        path = REPOSITORY / relative
        if not path.is_file():
            raise RuntimeError(f"audited optimizer source is missing: {relative}")
        actual[str(relative)] = sha256(path)
        if actual[str(relative)] != digest:
            raise RuntimeError(
                f"audited optimizer source SHA mismatch: {relative}; regenerate the code manifest"
            )
    return {
        "path": str(CODE_MANIFEST),
        "sha256": sha256(CODE_MANIFEST),
        "source_sha256": actual,
    }


def stage_mma_controls(beta: float) -> dict[str, object]:
    """Stage-specific native LD_MMA scales in projected-physical units.

    These initialize the first trust region and CCSA curvature only. They do
    not cap later updates: NLopt owns all subsequent asymptote adaptation. The
    projection derivative supplies the beta-to-beta scaling, so a latent step
    has the same local physical-density meaning at every beta.
    """
    reference = midpoint_projection_derivative(BETA_SCHEDULE[0])
    ratio = midpoint_projection_derivative(beta) / reference
    return {
        "initial_step": TARGET_INITIAL_PHYSICAL_DENSITY_STEP / ratio,
        "rho_init": BASE_RHO_INIT_AT_BETA1 * ratio * ratio,
        "always_improve": 1,
        "inner_gradients": 1,
        "projection_midpoint_derivative": midpoint_projection_derivative(beta),
        "projection_derivative_ratio_to_beta1": ratio,
        "target_initial_physical_density_step": TARGET_INITIAL_PHYSICAL_DENSITY_STEP,
        "base_rho_init_at_beta1": BASE_RHO_INIT_AT_BETA1,
        "fixed_per_iteration_move_limit": None,
        "xtol_rel": PURE_CURRENT_NLOPT_XTOL_REL,
    }


def feasible_stage_morphology_caps(values: np.ndarray, beta: float) -> np.ndarray:
    """Enter each morphology stage feasible, without an instantaneous 10% cut.

    The older continuation used max(target, 0.9 * current), deliberately
    placing a newly reprojected design about 11.1% outside the new cap when
    its residual exceeded the target.  Here the entry cap is the current
    residual or the absolute target, whichever is larger.  This guarantees
    that a beta change, projection change, and newly active constraints are
    not introduced as one forced-infeasible transition.
    """
    if beta < PURE_CURRENT_MORPHOLOGY_START_BETA:
        return np.asarray([np.inf, np.inf])
    return np.maximum(
        PURE_CURRENT_MORPHOLOGY_TARGET_CAP[beta], np.asarray(values, dtype=float)
    )


def physical_stage_readiness(
    history: list[dict[str, object]], beta: float, constraint_tolerance: float
) -> dict[str, object]:
    """Decide whether a numerical NLopt stop is physically ready for beta change.

    NLopt FTOL/XTOL reports only a numerical stopping condition.  A beta
    transition additionally requires a measured objective plateau over the
    last three full-physics changes and feasible active morphology constraints.
    No fixed minimum update count is imposed: four data points are the minimum
    needed to form the three measured changes in this test.
    """
    rows = [row for row in history if float(row["beta"]) == beta]
    if len(rows) < STAGE_PLATEAU_WINDOW + 1:
        return {
            "ready": False,
            "reason": "insufficient_full_physics_evaluations_for_plateau",
            "full_physics_evaluations": len(rows),
        }
    recent = rows[-(STAGE_PLATEAU_WINDOW + 1):]
    objective = np.asarray(
        [row["objective_at_reference_power_A"] for row in recent], dtype=float
    )
    denominator = np.maximum(
        np.maximum(np.abs(objective[:-1]), np.abs(objective[1:])), 1.0e-18
    )
    relative_changes = np.abs(np.diff(objective)) / denominator
    constraint_values = [
        float(row["maximum_constraint_value"])
        for row in recent
        if row["maximum_constraint_value"] is not None
    ]
    feasible = not constraint_values or max(constraint_values) <= constraint_tolerance
    plateau = bool(np.max(relative_changes) <= STAGE_PLATEAU_RELATIVE_CHANGE)
    return {
        "ready": bool(plateau and feasible),
        "reason": (
            "physical_objective_plateau_and_constraints_feasible"
            if plateau and feasible
            else "objective_not_plateaued" if not plateau
            else "active_constraints_infeasible"
        ),
        "full_physics_evaluations": len(rows),
        "relative_changes_last_three": relative_changes.tolist(),
        "maximum_relative_change_last_three": float(np.max(relative_changes)),
        "plateau_relative_change_gate": STAGE_PLATEAU_RELATIVE_CHANGE,
        "active_constraints_feasible": feasible,
    }


def pure_current_manifest(
    base_fsp: Path,
    base_sha256: str,
    jacobian_dir: Path,
    code_provenance: dict[str, object],
) -> dict[str, object]:
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
        "beta_1_and_above": ["500nm_solid_opening", "500nm_void_opening"],
    }
    manifest["mma_scale_policy"] = {
        "kind": "projection-Jacobian-scaled-native-LD_MMA-initialization",
        "target_initial_physical_density_step": TARGET_INITIAL_PHYSICAL_DENSITY_STEP,
        "base_rho_init_at_beta1": BASE_RHO_INIT_AT_BETA1,
        "fixed_per_iteration_move_limit": None,
        "note": "LD_MMA, not this driver, adapts all later asymptotes.",
    }
    manifest["beta_promotion_policy"] = {
        "NLopt_numerical_stop_alone_is_sufficient": False,
        "required_full_physics_objective_changes": STAGE_PLATEAU_WINDOW,
        "relative_plateau_gate": STAGE_PLATEAU_RELATIVE_CHANGE,
        "active_constraints_must_be_feasible": True,
    }
    manifest["optimizer_code_provenance"] = code_provenance
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
    parser.add_argument(
        "--initial-latent-npz",
        type=Path,
        help=(
            "Explicit native-LD_MMA warm-start point. This starts a fresh NLopt "
            "stage with new asymptotes; it does not claim to serialize or resume "
            "NLopt internal state."
        ),
    )
    args = parser.parse_args()

    CONTRACT.validate()
    if CONTRACT.geometry_mode != "contact_anchored":
        raise RuntimeError("pure-current LD_MMA requires contact_anchored geometry")
    base_fsp = verify_file(args.base_fsp, args.base_sha256)
    code_provenance = verify_optimizer_code_manifest()
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
    initial_provenance: dict[str, object]
    if args.initial_latent_npz is None:
        latent = np.full(MAPPING.shape, 0.5, dtype=np.float64)
        initial_provenance = {"kind": "uniform", "value": 0.5}
    else:
        initial_path = args.initial_latent_npz.expanduser().resolve()
        if not initial_path.is_file():
            raise RuntimeError(f"initial latent NPZ is missing: {initial_path}")
        with np.load(initial_path) as loaded:
            if "latent" not in loaded:
                raise RuntimeError("initial latent NPZ does not contain 'latent'")
            latent = np.asarray(loaded["latent"], dtype=np.float64)
        if latent.shape != MAPPING.shape:
            raise RuntimeError(
                f"initial latent shape {latent.shape} != expected {MAPPING.shape}"
            )
        if not np.all(np.isfinite(latent)) or np.any(latent < 0.0) or np.any(latent > 1.0):
            raise RuntimeError("initial latent design is non-finite or outside [0,1]")
        initial_provenance = {
            "kind": "native_LD_MMA_warm_restart",
            "path": str(initial_path),
            "size_bytes": int(initial_path.stat().st_size),
            "sha256": sha256(initial_path),
            "note": (
                "NLopt internal asymptotes are intentionally reset; this is a fresh "
                "native LD_MMA stage initialized at a fully evaluated physical design."
            ),
        }
    history: list[dict[str, object]] = []
    manifest = pure_current_manifest(
        base_fsp, args.base_sha256, jacobian_dir, code_provenance
    )
    manifest["initialization"] = initial_provenance
    emit(events, "native_ld_mma_initialization", **initial_provenance)
    fixed_source_power: float | None = None
    evaluation_counter = 0
    global_evaluation = 0

    for beta_index, beta in enumerate(BETA_SCHEDULE):
        if args.maximum_beta_stages and beta_index >= args.maximum_beta_stages:
            break
        stage_summary, _ = metrics(latent, beta, device=args.constraint_device)
        caps = feasible_stage_morphology_caps(
            np.asarray([
                stage_summary["smooth_solid_constraint"],
                stage_summary["smooth_void_constraint"],
            ]),
            beta,
        )
        constraint_count = 2
        controls = stage_mma_controls(beta)
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
            morphology_start_beta=PURE_CURRENT_MORPHOLOGY_START_BETA,
            optimizer_controls=controls,
        )
        optimizer = make_optimizer(
            evaluator,
            constraint_count,
            initial_step=float(controls["initial_step"]),
            rho_init=float(controls["rho_init"]),
            always_improve=int(controls["always_improve"]),
            inner_gradients=int(controls["inner_gradients"]),
            xtol_rel=float(controls["xtol_rel"]),
        )
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
            optimizer_controls=controls,
            ftol_rel=NLOPT_FTOL_REL,
            xtol_rel=controls["xtol_rel"],
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
        if result_code < 0:
            raise RuntimeError(f"NLopt LD_MMA failed with result code {result_code}")
        readiness = physical_stage_readiness(
            history, beta, NLOPT_CONSTRAINT_TOL
        )
        if not readiness["ready"]:
            raise RuntimeError(
                "NLopt numerical termination is not a physically ready beta transition: "
                f"{readiness['reason']}"
            )
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
            "physical_stage_readiness": readiness,
        }
        write_json(published / "RAW_ARTIFACT_MANIFEST.json", manifest)
        emit(
            events,
            "nlopt_stage_complete",
            beta=beta,
            result_code=result_code,
            evaluations=evaluator.stage_full_physics_evaluations,
            physical_stage_readiness=readiness,
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
        "initialization": initial_provenance,
        "final_beta": BETA_SCHEDULE[-1],
        "full_physics_evaluations": global_evaluation,
        "binary_result": final_result,
        "exact_binary_audit": binary_audit,
        "posthoc_morphology_repair": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
