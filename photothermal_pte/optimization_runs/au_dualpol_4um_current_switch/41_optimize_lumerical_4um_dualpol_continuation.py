#!/usr/bin/env python3
"""Run resumable Lumerical LD_MMA beta continuation to an exact Au mask."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import nlopt
import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_continuation import (
    BETA_SCHEDULE,
    DESIGN_CONSTRAINT_TOLERANCE,
    EPIGRAPH_CONSTRAINT_TOLERANCE,
    INITIAL_MAXIMIN_WARM_STEP_FRACTION,
    MAXIMUM_STAGE_ATTEMPTS,
    MOVE_LIMIT,
    STAGE_FTOL_REL,
    STAGE_MAXEVAL,
    STAGE_XTOL_REL,
    ContinuationEpigraphProblem,
    active_design_constraint_names,
    continuation_contract,
    design_constraint_point,
    grayness_value_gradient,
    linearized_maximin_box_warm_start,
    stage_design_caps,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    OPTIMIZER_250NM_MAPPING,
    exact_binary_cell_candidate,
    smooth_lumerical_250nm_constraints,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_optimizer import (
    LumericalEvaluationDriver,
    OptimizerRuntime,
    artifact,
    uniform_initial_latent_density,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
CHECKPOINT_SCHEMA = "au-lumerical-continuation-checkpoint-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Write the solver-free continuation/runtime audit and exit.",
    )
    parser.add_argument(
        "--stop-after-beta",
        type=float,
        choices=BETA_SCHEDULE,
        help="Testing/controlled handoff gate; omit for full continuation.",
    )
    return parser.parse_args()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _new_manifest(runtime: OptimizerRuntime) -> dict[str, Any]:
    return {
        "schema": "au-lumerical-dualpol-production-continuation-v1",
        "status": "RUNNING_LUMERICAL_4UM_DUALPOL_BETA_CONTINUATION",
        "passed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "optimizer": {"library": "NLopt", "version": nlopt.__version__, "algorithm": "LD_MMA"},
        "runtime": runtime.audit(),
        "continuation_contract": continuation_contract(),
        "stages": [],
        "final": None,
        "Lumerical_HEAT_or_CHARGE_solves": 0,
        "FDTDX_Maxwell_solves": 0,
    }


def _save_checkpoint(
    path: Path,
    *,
    latent: np.ndarray,
    beta_index: int,
    attempt: int,
    dfm_caps: np.ndarray,
    grayness_cap: float,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        schema=np.asarray(CHECKPOINT_SCHEMA),
        latent=np.asarray(latent, dtype=np.float64),
        beta_index=np.asarray(beta_index, dtype=np.int64),
        attempt=np.asarray(attempt, dtype=np.int64),
        dfm_caps=np.asarray(dfm_caps, dtype=np.float64),
        grayness_cap=np.asarray(grayness_cap, dtype=np.float64),
    )
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        schema = str(np.asarray(data["schema"]).item())
        if schema != CHECKPOINT_SCHEMA:
            raise RuntimeError(f"checkpoint schema changed: {schema}")
        latent = np.asarray(data["latent"], dtype=np.float64)
        if latent.shape != CONTRACT.design_node_shape:
            raise RuntimeError("checkpoint latent shape changed")
        return {
            "latent": latent,
            "beta_index": int(np.asarray(data["beta_index"]).item()),
            "attempt": int(np.asarray(data["attempt"]).item()),
            "dfm_caps": np.asarray(data["dfm_caps"], dtype=np.float64),
            "grayness_cap": float(np.asarray(data["grayness_cap"]).item()),
        }


def _attempt_directory(root: Path, beta: float, attempt: int) -> tuple[Path, int]:
    value = int(attempt)
    while True:
        candidate = root / "stages" / f"beta_{beta:03g}_attempt_{value:02d}"
        if not candidate.exists():
            return candidate, value
        value += 1


def _stage_constraints_satisfied(point: dict[str, Any]) -> bool:
    values = np.asarray(
        point["design_constraints"]["normalized_values"], dtype=np.float64
    )
    return bool(values.size == 0 or np.max(values) <= DESIGN_CONSTRAINT_TOLERANCE)


def _run_exact_binary_evaluation(
    *,
    runtime: OptimizerRuntime,
    binary_path: Path,
    output: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(HERE / "42_evaluate_lumerical_4um_exact_binary.py"),
        "--binary-mask-npz",
        str(binary_path),
        "--output-dir",
        str(output),
        "--gpu-index",
        str(runtime.gpu_index),
        "--accelerator-policy",
        runtime.accelerator_policy,
        "--threads",
        str(runtime.threads),
        "--ea-source-calibration",
        str(runtime.source_calibration["Ea"]),
        "--eb-source-calibration",
        str(runtime.source_calibration["Eb"]),
    ]
    completed = subprocess.run(command, cwd=REPOSITORY, check=False)
    result_path = output / "exact_binary_dualpol_result.json"
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(
            f"exact-binary evaluator failed with exit {completed.returncode}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("passed") is not True:
        raise RuntimeError("exact-binary evaluator did not pass its numerical gates")
    return result


def main() -> int:
    args = _parse_args()
    started = time.monotonic()
    output: Path | None = None
    manifest: dict[str, Any] = {
        "status": "FAILED_LUMERICAL_4UM_DUALPOL_BETA_CONTINUATION",
        "passed": False,
    }
    try:
        base_runtime = OptimizerRuntime.from_environment(require_smoke_beta=False)
        output = base_runtime.output_root
        manifest_path = output / "production_manifest.json"
        checkpoint_path = output / "continuation_checkpoint.npz"
        if output.exists() and any(output.iterdir()):
            if not manifest_path.is_file() or not checkpoint_path.is_file():
                raise RuntimeError(
                    "non-empty output is not a resumable production continuation"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            state = _load_checkpoint(checkpoint_path)
            if manifest.get("git_commit") != _git_commit():
                raise RuntimeError(
                    "refusing resume under a different Git commit; use the committed run worktree"
                )
        else:
            output.mkdir(parents=True, exist_ok=True)
            (output / "stages").mkdir()
            (output / "checkpoints").mkdir()
            manifest = _new_manifest(base_runtime)
            latent = uniform_initial_latent_density()
            projected = OPTIMIZER_250NM_MAPPING.physical(latent, BETA_SCHEDULE[0])
            if not np.array_equal(projected, latent):
                raise RuntimeError("uniform rho=0.5 is not exactly preserved at beta=1")
            state = {
                "latent": latent,
                "beta_index": 0,
                "attempt": 0,
                "dfm_caps": np.full(2, np.inf, dtype=np.float64),
                "grayness_cap": np.inf,
            }
            _save_checkpoint(checkpoint_path, **state)
            _write_json(manifest_path, manifest)

        if args.preflight_only:
            manifest["status"] = "PASSED_LUMERICAL_4UM_CONTINUATION_PREFLIGHT_ONLY"
            manifest["passed"] = True
            manifest["preflight_only"] = True
            manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(manifest_path, manifest)
            print(json.dumps(manifest, indent=2, default=str))
            return 0

        manifest["status"] = "RUNNING_LUMERICAL_4UM_DUALPOL_BETA_CONTINUATION"
        manifest["passed"] = False
        manifest.pop("preflight_only", None)
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, manifest)

        while state["beta_index"] < len(BETA_SCHEDULE):
            beta_index = int(state["beta_index"])
            beta = BETA_SCHEDULE[beta_index]
            attempt_dir, attempt = _attempt_directory(
                output, beta, int(state["attempt"])
            )
            attempt_dir.mkdir(parents=True)
            runtime = replace(base_runtime, output_root=attempt_dir, beta=beta)
            driver = LumericalEvaluationDriver(
                runtime, prune_heavy_intermediates=True
            )
            latent_initial = np.asarray(state["latent"], dtype=np.float64)
            initial_dfm, _, _ = smooth_lumerical_250nm_constraints(
                latent_initial, beta
            )
            # Use the checkpoint logical attempt, not the next free artifact
            # directory suffix. If a process dies after creating attempt_00
            # but before finishing its first Maxwell evaluation, the restart
            # uses attempt_01 while the stage caps are still uninitialized.
            # Persist those caps before the expensive solve so such a restart
            # cannot silently disable DFM or grayness constraints.
            if int(state["attempt"]) == 0:
                cap_record = stage_design_caps(
                    beta=beta,
                    baseline_dfm_values=initial_dfm,
                    previous_dfm_caps=np.asarray(state["dfm_caps"], dtype=np.float64),
                )
                state["dfm_caps"] = np.asarray(
                    cap_record["DFM_caps"], dtype=np.float64
                )
                state["grayness_cap"] = float(cap_record["grayness_cap"])
                _save_checkpoint(checkpoint_path, **state)
            initial_physics = driver.evaluate(latent_initial)
            problem = ContinuationEpigraphProblem(
                driver.evaluate,
                beta=beta,
                dfm_caps=np.asarray(state["dfm_caps"], dtype=np.float64),
                grayness_cap=float(state["grayness_cap"]),
            )
            epigraph_initial_nA = 1.0e9 * min(
                float(initial_physics["currents_A"]["Ea"]),
                -float(initial_physics["currents_A"]["Eb"]),
            )
            optimizer_latent = latent_initial
            optimizer_maxeval = STAGE_MAXEVAL[beta]
            warm_start_audit: dict[str, Any] | None = None
            if (
                beta_index == 0
                and int(state["attempt"]) == 0
                and np.array_equal(latent_initial, uniform_initial_latent_density())
            ):
                initial_point = problem.point(
                    np.r_[latent_initial.ravel(), epigraph_initial_nA]
                )
                warm = linearized_maximin_box_warm_start(
                    latent=latent_initial,
                    current_a_A=float(initial_point["current_a_A"]),
                    current_b_A=float(initial_point["current_b_A"]),
                    gradient_a_latent_A=np.asarray(
                        initial_point["gradient_a_latent_A"], dtype=np.float64
                    ),
                    gradient_b_latent_A=np.asarray(
                        initial_point["gradient_b_latent_A"], dtype=np.float64
                    ),
                    maximum_change=(
                        INITIAL_MAXIMIN_WARM_STEP_FRACTION * MOVE_LIMIT[beta]
                    ),
                )
                optimizer_latent = np.asarray(warm["latent"], dtype=np.float64)
                warm_physics = driver.evaluate(optimizer_latent)
                epigraph_initial_nA = 1.0e9 * min(
                    float(warm_physics["currents_A"]["Ea"]),
                    -float(warm_physics["currents_A"]["Eb"]),
                )
                optimizer_maxeval = max(1, STAGE_MAXEVAL[beta] - 1)
                warm_start_audit = {
                    key: value
                    for key, value in warm.items()
                    if key not in {"latent", "delta"}
                }
                warm_start_audit["actual_currents_nA"] = {
                    key: 1.0e9 * float(value)
                    for key, value in warm_physics["currents_A"].items()
                }
                warm_start_audit["actual_balanced_utility_nA"] = epigraph_initial_nA
            variable_count = problem.variable_count
            optimizer = nlopt.opt(nlopt.LD_MMA, variable_count)
            lower_latent = np.maximum(0.0, latent_initial.ravel() - MOVE_LIMIT[beta])
            upper_latent = np.minimum(1.0, latent_initial.ravel() + MOVE_LIMIT[beta])
            optimizer.set_lower_bounds(np.r_[lower_latent, -100.0])
            optimizer.set_upper_bounds(np.r_[upper_latent, 1000.0])
            optimizer.set_max_objective(problem.objective)
            tolerances = np.r_[
                np.full(2, EPIGRAPH_CONSTRAINT_TOLERANCE),
                np.full(
                    problem.design_constraint_count,
                    DESIGN_CONSTRAINT_TOLERANCE,
                ),
            ]
            optimizer.add_inequality_mconstraint(problem.constraints, tolerances)
            optimizer.set_initial_step(
                np.r_[
                    np.full(variable_count - 1, 0.25 * MOVE_LIMIT[beta]),
                    0.1,
                ]
            )
            optimizer.set_ftol_rel(STAGE_FTOL_REL)
            optimizer.set_xtol_rel(STAGE_XTOL_REL)
            optimizer.set_maxeval(optimizer_maxeval)
            vector_initial = np.r_[optimizer_latent.ravel(), epigraph_initial_nA]
            vector_final = optimizer.optimize(vector_initial)
            final_point = problem.point(vector_final)
            latent_final = vector_final[:-1].reshape(CONTRACT.design_node_shape)
            projected_final = OPTIMIZER_250NM_MAPPING.physical(latent_final, beta)
            binary_mask, binary_audit = exact_binary_cell_candidate(projected_final)
            design_satisfied = _stage_constraints_satisfied(final_point)
            switching = bool(
                float(final_point["current_a_A"]) > 0.0
                and float(final_point["current_b_A"]) < 0.0
            )
            stage_result = {
                "status": "COMPLETED_LUMERICAL_4UM_BETA_ATTEMPT",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "beta": beta,
                "beta_index": beta_index,
                "attempt": attempt,
                "active_design_constraints": list(
                    active_design_constraint_names(beta)
                ),
                "NLopt_total_constraint_count": problem.total_constraint_count,
                "requested_maxeval": STAGE_MAXEVAL[beta],
                "optimizer_requested_maxeval": optimizer_maxeval,
                "uniform_baseline_outside_optimizer": warm_start_audit is not None,
                "initial_maximin_warm_start": warm_start_audit,
                "reported_numevals": optimizer.get_numevals(),
                "result_code": optimizer.last_optimize_result(),
                "unique_physics_evaluations": len(driver.history),
                "initial_currents_nA": {
                    key: 1.0e9 * float(value)
                    for key, value in initial_physics["currents_A"].items()
                },
                "final_currents_nA": {
                    "Ea": 1.0e9 * float(final_point["current_a_A"]),
                    "Eb": 1.0e9 * float(final_point["current_b_A"]),
                },
                "balanced_utility_nA": 1.0e9
                * float(final_point["balanced_utility_A"]),
                "opposite_current_switching_achieved": switching,
                "design_constraints": {
                    "names": final_point["design_constraints"]["names"],
                    "normalized_values": final_point["design_constraints"][
                        "normalized_values"
                    ].tolist(),
                    "satisfied": design_satisfied,
                    "raw_DFM_values": final_point["design_constraints"][
                        "raw_DFM_values"
                    ].tolist(),
                    "DFM_caps": np.asarray(state["dfm_caps"]).tolist(),
                    "grayness": float(
                        final_point["design_constraints"]["grayness"]
                    ),
                    "grayness_cap": float(state["grayness_cap"]),
                },
                "exact_binary_candidate_audit": {
                    key: value
                    for key, value in binary_audit.items()
                    if key not in {"binary", "bad_solid", "bad_void"}
                },
                "callback_history": problem.callback_history,
                "wall_s": float(sum(row["wall_s"] for row in driver.history)),
            }
            state_path = attempt_dir / "stage_final_state.npz"
            np.savez_compressed(
                state_path,
                latent_initial=latent_initial,
                latent_final=latent_final,
                projected_final=projected_final,
                binary_candidate_cell_mask=binary_mask,
            )
            stage_result["state_artifact"] = artifact(state_path)
            _write_json(attempt_dir / "stage_result.json", stage_result)
            manifest["stages"].append(stage_result)
            manifest["latest"] = {
                "beta": beta,
                "attempt": attempt,
                "currents_nA": stage_result["final_currents_nA"],
                "balanced_utility_nA": stage_result["balanced_utility_nA"],
                "grayness": stage_result["design_constraints"]["grayness"],
                "design_constraints_satisfied": design_satisfied,
            }
            manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(manifest_path, manifest)

            final_beta = beta == BETA_SCHEDULE[-1]
            continuous_binary_gate = bool(
                final_beta
                and design_satisfied
                and float(final_point["design_constraints"]["grayness"])
                <= float(state["grayness_cap"])
                * (1.0 + DESIGN_CONSTRAINT_TOLERANCE)
                and binary_audit["solid_pass"]
                and binary_audit["void_pass"]
            )
            if final_beta and continuous_binary_gate and switching:
                binary_path = output / "final_exact_binary_cell_mask.npz"
                np.savez_compressed(binary_path, binary_mask=binary_mask)
                exact_result = _run_exact_binary_evaluation(
                    runtime=runtime,
                    binary_path=binary_path,
                    output=output / "final_exact_binary_evaluation",
                )
                exact_switching = bool(
                    exact_result["currents_A"]["Ea"] > 0.0
                    and exact_result["currents_A"]["Eb"] < 0.0
                )
                if not exact_switching:
                    raise RuntimeError(
                        "relaxed design switched but the exact-binary reevaluation did not"
                    )
                manifest["status"] = (
                    "PASSED_LUMERICAL_4UM_DUALPOL_EXACT_BINARY_AU_OPTIMIZATION"
                )
                manifest["passed"] = True
                manifest["final"] = {
                    "beta": beta,
                    "continuous_stage": stage_result,
                    "binary_mask": artifact(binary_path),
                    "exact_binary_evaluation": exact_result,
                    "ordinary_dispersive_Au": True,
                    "posthoc_morphology_repair": False,
                }
                state["beta_index"] = len(BETA_SCHEDULE)
                state["attempt"] = 0
                state["latent"] = latent_final
                _save_checkpoint(checkpoint_path, **state)
                _write_json(manifest_path, manifest)
                print(json.dumps(manifest["final"], indent=2, default=str))
                return 0

            attempts_used = attempt + 1
            may_advance = bool(not final_beta and design_satisfied)
            if may_advance:
                state["latent"] = latent_final
                state["beta_index"] = beta_index + 1
                state["attempt"] = 0
                # A new stage computes a new grayness cap.  DFM caps remain
                # monotone through state["dfm_caps"].
                state["grayness_cap"] = np.inf
                _save_checkpoint(checkpoint_path, **state)
                checkpoint_copy = output / "checkpoints" / f"beta_{beta:03g}_completed.npz"
                _save_checkpoint(checkpoint_copy, **state)
                if args.stop_after_beta == beta:
                    manifest["status"] = "PAUSED_AFTER_REQUESTED_BETA"
                    _write_json(manifest_path, manifest)
                    return 0
                continue

            if attempts_used >= MAXIMUM_STAGE_ATTEMPTS[beta]:
                manifest["status"] = (
                    "STOPPED_UNRESOLVED_FINAL_BINARY_OR_SWITCHING_GATES"
                    if final_beta
                    else "STOPPED_UNRESOLVED_STAGE_DESIGN_CONSTRAINTS"
                )
                manifest["passed"] = False
                manifest["blocking_beta"] = beta
                manifest["blocking_attempts"] = attempts_used
                manifest["wall_s"] = time.monotonic() - started
                _write_json(manifest_path, manifest)
                _save_checkpoint(
                    checkpoint_path,
                    latent=latent_final,
                    beta_index=beta_index,
                    attempt=attempts_used,
                    dfm_caps=np.asarray(state["dfm_caps"]),
                    grayness_cap=float(state["grayness_cap"]),
                )
                return 2

            state["latent"] = latent_final
            state["attempt"] = attempts_used
            _save_checkpoint(checkpoint_path, **state)

        raise RuntimeError("continuation exited without exact-binary promotion")
    except Exception as error:
        manifest.update(
            status="FAILED_LUMERICAL_4UM_DUALPOL_BETA_CONTINUATION",
            passed=False,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            wall_s=time.monotonic() - started,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        if output is None:
            output = Path(os.environ.get("EIDL_RUN_DIR", ".")).resolve()
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / "production_manifest.json", manifest)
        print(json.dumps(manifest, indent=2, default=str))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
