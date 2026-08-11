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
# Keep NLopt's numerical FTOL tighter than the fixed per-beta continuation
# budget.  FTOL remains a normal native-optimizer early-stop criterion; raw
# callback-to-callback objective changes are not used as a beta gate.
PURE_CURRENT_NLOPT_FTOL_REL = 1.0e-6
PURE_CURRENT_BETA_FACTOR = 2.0
PURE_CURRENT_MAX_BETA = 1024.0
PURE_CURRENT_GRAYSCALE_MAX_EVALUATIONS = MAXIMUM_STAGE_EVALUATIONS
PURE_CURRENT_CONTINUATION_MAX_EVALUATIONS = 20


def lumopt_style_beta_schedule() -> tuple[float, ...]:
    """Return beta=1 followed by user-selected factor-2 continuation to 1024.

    Ansys LumOpt separates the initial grayscale phase from fixed-budget
    binarization stages and multiplies beta by a continuation factor.  Keep
    rounded decimal values so command-line recovery can identify a stage
    reproducibly instead of depending on accumulated binary roundoff.
    """
    values = [1.0]
    while values[-1] < PURE_CURRENT_MAX_BETA:
        value = min(PURE_CURRENT_MAX_BETA, values[-1] * PURE_CURRENT_BETA_FACTOR)
        values.append(float(f"{value:.12g}"))
    return tuple(values)


PURE_CURRENT_BETA_SCHEDULE = lumopt_style_beta_schedule()


def lumopt_style_beta_promotion_policy() -> dict[str, object]:
    """Return the single source of truth for initial/recovery manifests."""
    return {
        "kind": "Ansys-LumOpt-style fixed-budget continuation",
        "initial_grayscale_beta": 1.0,
        "initial_grayscale_max_evaluations": PURE_CURRENT_GRAYSCALE_MAX_EVALUATIONS,
        "beta_factor": PURE_CURRENT_BETA_FACTOR,
        "continuation_max_evaluations_per_beta": PURE_CURRENT_CONTINUATION_MAX_EVALUATIONS,
        "maximum_beta": PURE_CURRENT_MAX_BETA,
        "raw_trial_objective_plateau_used_as_gate": False,
        "normal_NLopt_early_stop_promotes_beta": True,
        "active_constraints_must_be_feasible": True,
    }


def fixed_morphology_cap_policy() -> dict[str, object]:
    """Describe the constraint contract independently of run history."""
    return {
        "kind": "fixed_absolute_smooth_opening_residual_cap",
        "minimum_feature_nm": 500.0,
        "solid_and_void_constraints": True,
        "stage_entry_may_be_infeasible": True,
        "current_residual_may_relax_cap": False,
        "beta_targets": {
            f"{beta:g}": morphology_target_cap(beta)
            for beta in PURE_CURRENT_BETA_SCHEDULE
        },
        "beta_may_advance_only_if_smooth_constraints_feasible": True,
        "final_exact_bad_nodes_required": 0,
    }


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
    reference = midpoint_projection_derivative(PURE_CURRENT_BETA_SCHEDULE[0])
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


def morphology_target_cap(beta: float) -> float:
    """Log-interpolate the existing audited caps for continuation beta stages."""
    anchors = np.asarray(sorted(PURE_CURRENT_MORPHOLOGY_TARGET_CAP), dtype=float)
    targets = np.asarray(
        [PURE_CURRENT_MORPHOLOGY_TARGET_CAP[value] for value in anchors], dtype=float
    )
    if beta <= anchors[0]:
        return float(targets[0])
    if beta >= anchors[-1]:
        # Continue the final halving-per-beta-doubling trend without allowing
        # a zero constraint scale at high beta.
        return float(max(1.0e-6, targets[-1] * anchors[-1] / beta))
    return float(
        np.exp(np.interp(np.log(beta), np.log(anchors), np.log(targets)))
    )


def fixed_stage_morphology_caps(values: np.ndarray, beta: float) -> np.ndarray:
    """Return the absolute 500-nm residual target for one beta stage.

    A cap selected as ``max(target, current)`` makes every newly projected
    design feasible by construction and therefore enforces only
    non-worsening.  That policy left the exact audit fixed at 360 bad nodes.
    Native LD_MMA is allowed to enter a stage infeasible and must reduce both
    residuals to the recorded absolute target before beta can advance.
    ``values`` is retained for input validation and audit symmetry.
    """
    values = np.asarray(values, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("morphology residuals must be two finite values")
    if beta < PURE_CURRENT_MORPHOLOGY_START_BETA:
        return np.asarray([np.inf, np.inf])
    target = morphology_target_cap(beta)
    return np.asarray([target, target], dtype=float)


def continuation_stage_completion(
    history: list[dict[str, object]], beta: float, constraint_tolerance: float,
    result_code: int, maximum_evaluations: int,
) -> dict[str, object]:
    """Record an official-style fixed-budget/normal-stop beta completion.

    NLopt callbacks include trial and repeated points, so their raw objective
    sequence is not an accepted-iterate sequence and must not be used as a
    physical plateau veto.  A positive NLopt stop with a feasible returned
    point completes the stage; MAXEVAL is the fixed continuation budget and
    FTOL/XTOL are documented normal early-stop paths.
    """
    rows = [row for row in history if float(row["beta"]) == beta]
    last_constraint = rows[-1]["maximum_constraint_value"] if rows else None
    feasible = last_constraint is None or float(last_constraint) <= constraint_tolerance
    return {
        "ready": bool(result_code > 0 and feasible),
        "reason": (
            "normal_nlopt_stop_and_constraints_feasible"
            if result_code > 0 and feasible
            else "active_constraints_infeasible" if not feasible
            else "nlopt_failure"
        ),
        "full_physics_evaluations": len(rows),
        "maximum_stage_evaluations": int(maximum_evaluations),
        "nlopt_result_code": int(result_code),
        "raw_trial_objective_plateau_used_as_gate": False,
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
    manifest["beta_promotion_policy"] = lumopt_style_beta_promotion_policy()
    manifest["morphology_cap_policy"] = fixed_morphology_cap_policy()
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
        "--start-beta",
        type=float,
        help=(
            "Explicit first beta for a warm recovery. It must be a member of the "
            "configured beta-continuation schedule."
        ),
    )
    parser.add_argument(
        "--initial-latent-npz",
        type=Path,
        help=(
            "Explicit native-LD_MMA warm-start point. This starts a fresh NLopt "
            "stage with new asymptotes; it does not claim to serialize or resume "
            "NLopt internal state."
        ),
    )
    parser.add_argument(
        "--recovery-append",
        action="store_true",
        help=(
            "Append a fresh native-LD_MMA stage to an interrupted published run. "
            "Existing evaluation history and raw-manifest provenance are retained; "
            "NLopt's non-serializable internal asymptotes are intentionally reset."
        ),
    )
    parser.add_argument(
        "--recovery-validation-result",
        type=Path,
        help=(
            "Optional already-passed raw retry result used only as recovery provenance. "
            "It is not imported as an optimizer evaluation."
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
    if args.recovery_append and args.initial_latent_npz is None:
        raise RuntimeError("--recovery-append requires --initial-latent-npz")
    if args.recovery_validation_result is not None and not args.recovery_append:
        raise RuntimeError("--recovery-validation-result requires --recovery-append")
    if (raw_root / "stage_checkpoint.npz").exists() and not args.recovery_append:
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
    if args.recovery_append:
        history_path = published / "optimization_history.json"
        manifest_path = published / "RAW_ARTIFACT_MANIFEST.json"
        if not history_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(
                "recovery append requires existing published optimization history and manifest"
            )
        history = json.loads(history_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(history, list) or not history:
            raise RuntimeError("recovery append requires a nonempty evaluation history")
        if not isinstance(manifest, dict) or not isinstance(manifest.get("evaluations"), dict):
            raise RuntimeError("recovery append requires a valid raw artifact manifest")
        prior_beta_policy = manifest.get("beta_promotion_policy")
        replacement_beta_policy = lumopt_style_beta_promotion_policy()
        if prior_beta_policy != replacement_beta_policy:
            manifest.setdefault("continuation_policy_revisions", []).append({
                "reason": (
                    "Replace the invalid raw-callback plateau veto with the approved "
                    "Ansys LumOpt-style fixed-budget beta continuation."
                ),
                "prior_policy": prior_beta_policy,
                "replacement_policy": replacement_beta_policy,
            })
        manifest["beta_promotion_policy"] = replacement_beta_policy
        prior_morphology_policy = manifest.get("morphology_cap_policy")
        replacement_morphology_policy = fixed_morphology_cap_policy()
        if prior_morphology_policy != replacement_morphology_policy:
            manifest.setdefault("morphology_policy_revisions", []).append({
                "reason": (
                    "The prior max(target,current) cap held exact violations at "
                    "360 nodes; replace it with the fixed absolute beta target."
                ),
                "prior_policy": prior_morphology_policy,
                "replacement_policy": replacement_morphology_policy,
            })
        manifest["morphology_cap_policy"] = replacement_morphology_policy
        try:
            evaluation_counter = max(int(row["evaluation_id"]) for row in history)
            global_evaluation = max(
                int(row["global_full_physics_evaluation"]) for row in history
            )
            source_powers = np.asarray(
                [float(row["fixed_source_power_W"]) for row in history], dtype=float
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("existing history cannot establish a recovery contract") from error
        fixed_source_power = float(source_powers[0])
        if (
            not np.all(np.isfinite(source_powers))
            or np.any(np.abs(source_powers - fixed_source_power) / fixed_source_power >= 0.005)
        ):
            raise RuntimeError("existing history violates the fixed-source-power recovery contract")
        if evaluation_counter != len(history) or global_evaluation != len(history):
            raise RuntimeError("existing history has non-contiguous evaluation numbering")
        recovery_validation: dict[str, object] | None = None
        if args.recovery_validation_result is not None:
            validation_path = args.recovery_validation_result.expanduser().resolve()
            if not validation_path.is_file():
                raise RuntimeError("recovery validation result is missing")
            validation = json.loads(validation_path.read_text())
            if not validation.get("passed"):
                raise RuntimeError("recovery validation result is not passed")
            recovery_validation = {
                "path": str(validation_path),
                "size_bytes": int(validation_path.stat().st_size),
                "sha256": sha256(validation_path),
                "status": validation.get("status"),
                "objective_A": validation.get("objective_A"),
            }
        recovery = {
            "kind": "native_LD_MMA_recovery_append",
            "prior_evaluations": len(history),
            "next_published_evaluation_id": evaluation_counter + 1,
            "start_beta": args.start_beta,
            "initialization": initial_provenance,
            "reason": (
                "The prior run's original stop record remains immutable in events/history. "
                "This fresh native LD_MMA stage intentionally resets unserializable internal "
                "asymptotes, reuses the last fully evaluated design, and starts the approved "
                "user-selected factor-2 beta continuation."
            ),
            "recovery_validation_raw_result": recovery_validation,
            "recovery_driver_code_provenance": code_provenance,
        }
        manifest.setdefault("recovery_chain", []).append(recovery)
        emit(events, "native_ld_mma_recovery_append", **recovery)
    else:
        history = []
        manifest = pure_current_manifest(
            base_fsp, args.base_sha256, jacobian_dir, code_provenance
        )
        manifest["initialization"] = initial_provenance
        # ``emit`` reserves its first positional argument for the event-file
        # path.  Keep a warm-start provenance ``path`` nested rather than
        # passing it as an event keyword, otherwise a valid warm start fails
        # before its first FDTD solve with ``multiple values for argument
        # 'path'``.
        emit(events, "native_ld_mma_initialization", initialization=initial_provenance)
        fixed_source_power = None
        evaluation_counter = 0
        global_evaluation = 0

    beta_schedule = PURE_CURRENT_BETA_SCHEDULE
    if args.start_beta is not None:
        matches = np.flatnonzero(
            np.isclose(beta_schedule, args.start_beta, rtol=0.0, atol=1.0e-10)
        )
        if matches.size != 1:
            raise RuntimeError(
                f"--start-beta {args.start_beta} is not in {beta_schedule}"
            )
        beta_schedule = beta_schedule[int(matches[0]):]
    completed_beta_stages = 0
    for beta in beta_schedule:
        if args.maximum_beta_stages and completed_beta_stages >= args.maximum_beta_stages:
            break
        stage_summary, _ = metrics(latent, beta, device=args.constraint_device)
        caps = fixed_stage_morphology_caps(
            np.asarray([
                stage_summary["smooth_solid_constraint"],
                stage_summary["smooth_void_constraint"],
            ]),
            beta,
        )
        constraint_count = 2
        controls = stage_mma_controls(beta)
        stage_maximum_evaluations = (
            PURE_CURRENT_GRAYSCALE_MAX_EVALUATIONS
            if np.isclose(beta, 1.0)
            else PURE_CURRENT_CONTINUATION_MAX_EVALUATIONS
        )
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
            # Preserve the interrupted raw evaluation_0012 inputs/output unchanged.
            # The public history still continues at evaluation_0012, while this
            # fresh physical solve is distinguishable in raw provenance.
            output_slug=(
                "pure_current_ld_mma_recovery2"
                if args.recovery_append
                else "pure_current_ld_mma"
            ),
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
            ftol_rel=PURE_CURRENT_NLOPT_FTOL_REL,
            maxeval=stage_maximum_evaluations,
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
            ftol_rel=PURE_CURRENT_NLOPT_FTOL_REL,
            xtol_rel=controls["xtol_rel"],
            maxeval=stage_maximum_evaluations,
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
        readiness = continuation_stage_completion(
            history, beta, NLOPT_CONSTRAINT_TOL, result_code,
            stage_maximum_evaluations,
        )
        if not readiness["ready"]:
            raise RuntimeError(
                "NLopt continuation stage did not return a feasible normal stop: "
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
        completed_beta_stages += 1

    if args.maximum_beta_stages:
        return 0
    final_beta = float(beta)
    final_rho = MAPPING.physical(latent, final_beta)
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
        "final_beta": final_beta,
        "full_physics_evaluations": global_evaluation,
        "binary_result": final_result,
        "exact_binary_audit": binary_audit,
        "posthoc_morphology_repair": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
