#!/usr/bin/env python3
"""Bounded LD_MMA continuation with official Ansys DFM and exact repair.

This driver replaces the unbounded same-beta recovery loop.  Every beta is
visited once.  The official v261 minimum-feature objective is inactive through
beta=12 and is included, with its exact CAD gradient, from beta=16 onward.
The final thresholded design is repaired solver-free, independently audited,
and only exact-feasible candidates receive fresh Maxwell/PTE evaluations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import nlopt
import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.ansys_minimum_feature import (
    CONTRACT as DFM_CONTRACT,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import (
    MAPPING,
    exact_binary_audit,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.repair_exact_binary_candidates import (
    gradient_aware_exact_repair,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.run_nlopt_mma_optimization import (
    StageEvaluator,
    initial_manifest,
    make_optimizer,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.run_true_mma_optimization import (
    equivalent_current,
    sha256,
    verify_file,
    write_json,
)


REPOSITORY = Path(__file__).resolve().parents[3]
BETA_SCHEDULE = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
MAXIMUM_EVALUATIONS = {
    1.0: 20,
    2.0: 10,
    4.0: 8,
    8.0: 8,
    16.0: 10,
    32.0: 10,
    64.0: 8,
    128.0: 8,
}
STAGE_TRUST_RADIUS = {
    1.0: 0.20,
    2.0: 0.20,
    4.0: 0.15,
    8.0: 0.12,
    16.0: 0.10,
    32.0: 0.08,
    64.0: 0.06,
    128.0: 0.05,
}
REFERENCE_INCIDENT_POWER_W = 285.0e-6


def emit(path: Path, event: str, **values: object) -> None:
    row = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **values,
    }
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def evaluate_exact_candidate(
    *,
    rho: np.ndarray,
    rank: int,
    candidate_root: Path,
    polarization: str,
    gpu: int,
    base_fsp: Path,
    base_sha256: str,
    reference_objective_A: float,
) -> dict[str, object]:
    candidate_root.mkdir(parents=True, exist_ok=True)
    density = candidate_root / f"exact_candidate_{rank:02d}.npz"
    output = candidate_root / f"exact_candidate_{rank:02d}_physics"
    np.savez_compressed(density, rho=np.asarray(rho, dtype=np.float64))
    command = [
        sys.executable,
        "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology.evaluate_binary_objective",
        "--base-fsp",
        str(base_fsp),
        "--base-sha256",
        base_sha256,
        "--rho-npz",
        str(density),
        "--output-dir",
        str(output),
        "--polarization",
        polarization,
        "--gpu-device",
        f"GPU {gpu}",
        "--cuda-device",
        "0",
        "--reference-objective-A",
        str(reference_objective_A),
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    completed = subprocess.run(command, cwd=REPOSITORY, env=environment)
    result_path = output / "binary_objective_result.json"
    if not result_path.is_file():
        raise RuntimeError(f"exact candidate {rank} physics evaluation failed")
    result = json.loads(result_path.read_text())
    physical_gates_passed = bool(
        result.get(
            "physical_gates_passed",
            result.get("forward", {}).get("closure", np.inf) < 0.005
            and result.get("gates", {}).get("Q_mapping_error", np.inf) < 0.005
            and result.get("gates", {}).get("thermal_forward_residual", np.inf) < 1e-8
            and result.get("gates", {}).get("thermal_energy_balance", np.inf) < 0.01
            and result.get("gates", {}).get("electrical_weighting_residual", np.inf) < 1e-8
            and np.isfinite(float(result.get("objective_A", np.nan)))
            and float(result.get("terminal_conductance_S", 0.0)) > 0.0,
        )
    )
    if completed.returncode not in (0, 1) or not physical_gates_passed:
        raise RuntimeError(f"exact candidate {rank} failed physical solver gates")
    return {
        "rank": rank,
        "density": {
            "path": str(density),
            "size_bytes": density.stat().st_size,
            "sha256": sha256(density),
        },
        "result_path": str(result_path),
        "objective_A": float(result["objective_A"]),
        "physical_gates_passed": physical_gates_passed,
        "objective_gate_passed": bool(
            result.get("binary_objective_preserved_within_one_percent", False)
        ),
        "result": result,
    }


def restore_resume_state(
    raw_root: Path, published: Path
) -> tuple[list[dict[str, object]], dict[str, object], np.ndarray, float, int, int, float]:
    """Restore the last completed beta without replaying Maxwell evaluations."""

    history_path = raw_root / "history.json"
    manifest_path = published / "RAW_ARTIFACT_MANIFEST.json"
    if not history_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("resume requires history.json and the published manifest")
    history = json.loads(history_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    completed: list[tuple[float, Path]] = []
    for beta in BETA_SCHEDULE:
        path = raw_root / f"beta_{beta:g}_completed.npz"
        if path.is_file():
            completed.append((beta, path))
    if not completed:
        raise RuntimeError("resume found no completed beta checkpoint")
    last_beta, checkpoint = completed[-1]
    with np.load(checkpoint) as data:
        latent = np.asarray(data["latent"], dtype=np.float64)
    if latent.shape != MAPPING.shape:
        raise RuntimeError("resume checkpoint shape disagrees with design mapping")
    if not history:
        raise RuntimeError("resume history is empty")
    fixed_source_power = float(history[0]["fixed_source_power_W"])
    evaluation_counter = max(int(row["evaluation_id"]) for row in history)
    global_evaluation = max(
        int(row["global_full_physics_evaluation"]) for row in history
    )
    return (
        history,
        manifest,
        latent,
        fixed_source_power,
        evaluation_counter,
        global_evaluation,
        last_beta,
    )


def restore_final_point(raw_root: Path, history: list[dict[str, object]]) -> SimpleNamespace:
    """Restore the latest objective result and physical-density gradient."""

    row = max(history, key=lambda item: int(item["global_full_physics_evaluation"]))
    evaluation_id = int(row["evaluation_id"])
    matches = sorted(raw_root.glob(f"evaluation_{evaluation_id:04d}_beta*_official_ansys_dfm"))
    if len(matches) != 1:
        raise RuntimeError("resume could not resolve the latest evaluation directory")
    result = json.loads((matches[0] / "objective_gradient_result.json").read_text())
    with np.load(matches[0] / "objective_gradient.npz") as data:
        gradient = np.asarray(data["gradient_total_A"], dtype=np.float64)
    return SimpleNamespace(result=result, gradient_physical_A=gradient)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--published-dir", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--base-fsp", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--jacobian-dir", required=True, type=Path)
    parser.add_argument("--constraint-device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    CONTRACT.validate()
    DFM_CONTRACT.validate()
    if CONTRACT.geometry_mode != "contact_anchored" or CONTRACT.contact_axis != "y":
        raise RuntimeError("this production run requires top/bottom contact anchoring")
    if CONTRACT.design_node_shape != MAPPING.shape:
        raise RuntimeError("density mapping and geometry shape disagree")

    base_fsp = verify_file(args.base_fsp, args.base_sha256)
    jacobian_dir = args.jacobian_dir.expanduser().resolve()
    certificate = jacobian_dir / "component_yee_jacobian_result.json"
    if not certificate.is_file() or not json.loads(certificate.read_text()).get("passed"):
        raise RuntimeError("component-Yee Jacobian certificate is missing or failed")

    raw_root = args.raw_root.expanduser().resolve()
    published = args.published_dir.expanduser().resolve()
    if raw_root.exists() and any(raw_root.iterdir()) and not args.resume:
        raise RuntimeError("fresh bounded run requires an empty raw directory")
    raw_root.mkdir(parents=True, exist_ok=True)
    published.mkdir(parents=True, exist_ok=True)
    events = raw_root / "events.jsonl"
    if args.resume:
        (
            history,
            manifest,
            latent,
            fixed_source_power,
            evaluation_counter,
            global_evaluation,
            last_completed_beta,
        ) = restore_resume_state(raw_root, published)
        final_point = restore_final_point(raw_root, history)
        emit(
            events,
            "bounded_run_resumed",
            last_completed_beta=last_completed_beta,
            next_evaluation=evaluation_counter + 1,
        )
    else:
        history = []
        manifest = initial_manifest(base_fsp, args.base_sha256, jacobian_dir)
        manifest["schema"] = "official-ansys-dfm-exact-repair-optimization-v1"
        manifest["continuation"] = {
            "beta_schedule": list(BETA_SCHEDULE),
            "maximum_evaluations_per_beta": MAXIMUM_EVALUATIONS,
            "stage_trust_radius": STAGE_TRUST_RADIUS,
            "trust_region_interpretation": (
                "fixed bounds around each beta-stage start; not a fixed "
                "per-evaluation update or normalized-gradient step"
            ),
            "same_beta_restart_loop": False,
            "official_dfm": DFM_CONTRACT.audit(),
            "exact_final_gate": "independent 500-nm binary opening audit",
        }
        write_json(published / "RAW_ARTIFACT_MANIFEST.json", manifest)
        latent = np.full(MAPPING.shape, 0.5, dtype=np.float64)
        fixed_source_power = None
        evaluation_counter = 0
        global_evaluation = 0
        last_completed_beta = 0.0
        final_point = None

    for beta in BETA_SCHEDULE:
        if beta <= last_completed_beta:
            continue
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
            morphology_caps=np.asarray([np.inf, np.inf]),
            fixed_source_power_W=fixed_source_power,
            evaluation_counter=evaluation_counter,
            global_evaluation=global_evaluation,
            constraint_device=args.constraint_device,
            algorithm_label="NLopt LD_MMA + official Ansys v261 DFM",
            output_slug="official_ansys_dfm",
            include_terminal_conductance_constraint=False,
            morphology_start_beta=np.inf,
            morphology_penalty_weight=0.0,
            use_official_ansys_dfm=True,
            optimizer_controls={
                "one_stage_per_beta": True,
                "manual_move_limit": None,
                "beta_factor": 2.0,
            },
        )
        trust_radius = STAGE_TRUST_RADIUS[beta]
        stage_lower = np.maximum(0.0, latent - trust_radius)
        stage_upper = np.minimum(1.0, latent + trust_radius)
        optimizer = make_optimizer(
            evaluator,
            0,
            xtol_rel=1.0e-8,
            ftol_rel=1.0e-5,
            maxeval=MAXIMUM_EVALUATIONS[beta],
            lower_bounds=stage_lower,
            upper_bounds=stage_upper,
        )
        emit(
            events,
            "bounded_beta_stage_start",
            beta=beta,
            maximum_evaluations=MAXIMUM_EVALUATIONS[beta],
            official_dfm_scaling=DFM_CONTRACT.penalty_scaling(beta),
            stage_trust_radius=trust_radius,
            stage_latent_lower_range=[
                float(np.min(stage_lower)), float(np.max(stage_lower))
            ],
            stage_latent_upper_range=[
                float(np.min(stage_upper)), float(np.max(stage_upper))
            ],
        )
        optimum = optimizer.optimize(latent.ravel()).reshape(MAPPING.shape)
        result_code = optimizer.last_optimize_result()
        if result_code < 0:
            raise RuntimeError(f"NLopt LD_MMA failed at beta={beta}: {result_code}")
        final_point = evaluator.point(optimum.ravel())
        latent = optimum
        fixed_source_power = evaluator.fixed_source_power_W
        evaluation_counter = evaluator.evaluation_counter
        global_evaluation = evaluator.global_evaluation
        rho = MAPPING.physical(latent, beta)
        discreteness = 1.0 - float(np.mean(4.0 * rho * (1.0 - rho)))
        checkpoint = raw_root / f"beta_{beta:g}_completed.npz"
        np.savez_compressed(checkpoint, latent=latent, rho=rho, beta=np.asarray(beta))
        manifest.setdefault("stage_checkpoints", {})[f"beta_{beta:g}"] = {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
            "nlopt_result_code": result_code,
            "discreteness": discreteness,
            "full_physics_evaluations": evaluator.stage_full_physics_evaluations,
        }
        write_json(published / "RAW_ARTIFACT_MANIFEST.json", manifest)
        emit(
            events,
            "bounded_beta_stage_complete",
            beta=beta,
            result_code=result_code,
            discreteness=discreteness,
            evaluations=evaluator.stage_full_physics_evaluations,
        )
    if final_point is None or fixed_source_power is None:
        raise RuntimeError("optimization did not produce a physical evaluation")

    final_beta = float(history[-1]["beta"])
    final_rho = MAPPING.physical(latent, final_beta)
    thresholded = final_rho >= 0.5
    current_gradient_latent = MAPPING.vjp(
        latent, final_point.gradient_physical_A, final_beta
    )
    repair = gradient_aware_exact_repair(
        thresholded,
        objective_gradient=current_gradient_latent,
        geometry_mode=CONTRACT.geometry_mode,
        contact_axis=CONTRACT.contact_axis,
        maximum_candidates=4,
    )
    candidate_arrays = repair.pop("candidate_arrays")
    write_json(published / "EXACT_REPAIR_DIAGNOSTIC.json", repair)
    if not repair["passed"] or not candidate_arrays:
        raise RuntimeError("bounded exact repair found no 500-nm-feasible candidate")

    candidate_results = []
    candidate_root = raw_root / f"exact_attempt_beta{final_beta:g}"
    for rank, candidate in enumerate(candidate_arrays):
        audit, _ = exact_binary_audit(
            candidate,
            geometry_mode=CONTRACT.geometry_mode,
            contact_axis=CONTRACT.contact_axis,
        )
        if not audit["passed"]:
            raise RuntimeError("repair returned a candidate that failed independent audit")
        candidate_results.append(
            evaluate_exact_candidate(
                rho=candidate,
                rank=rank,
                candidate_root=candidate_root,
                polarization=args.polarization,
                gpu=args.gpu,
                base_fsp=base_fsp,
                base_sha256=args.base_sha256,
                reference_objective_A=float(final_point.result["objective_A"]),
            )
        )
    best = max(candidate_results, key=lambda row: row["objective_A"])
    best_rho = candidate_arrays[int(best["rank"])]
    best_audit, _ = exact_binary_audit(
        best_rho,
        geometry_mode=CONTRACT.geometry_mode,
        contact_axis=CONTRACT.contact_axis,
    )
    continuous_reference_A = equivalent_current(
        float(final_point.result["objective_A"]), fixed_source_power
    )
    best_reference_A = equivalent_current(float(best["objective_A"]), fixed_source_power)
    objective_gate_passed = bool(best["objective_gate_passed"])
    final_passed = bool(best_audit["passed"] and objective_gate_passed)
    final = {
        "passed": final_passed,
        "status": (
            "VALIDATED_OFFICIAL_DFM_EXACT_BINARY_PTE_OPTIMIZATION"
            if final_passed
            else "FAILED_EXACT_BINARY_OBJECTIVE_PRESERVATION"
        ),
        "polarization": args.polarization,
        "axis_contract": "Lumerical x=b, y=a, z=c",
        "algorithm": "NLopt LD_MMA",
        "initial_density": "uniform latent rho=0.5",
        "beta_schedule_visited": sorted({float(row["beta"]) for row in history}),
        "final_beta": final_beta,
        "full_physics_evaluations_before_exact_candidates": global_evaluation,
        "exact_candidate_physics_evaluations": len(candidate_results),
        "continuous_reference_current_A": continuous_reference_A,
        "exact_reference_current_A": best_reference_A,
        "exact_cleanup_relative_current_change": (
            (best_reference_A - continuous_reference_A)
            / max(abs(continuous_reference_A), 1.0e-30)
        ),
        "exact_binary_audit": best_audit,
        "objective_preservation_gate_passed": objective_gate_passed,
        "chosen_candidate": best,
        "all_exact_candidates": candidate_results,
        "official_ansys_dfm": DFM_CONTRACT.audit(),
        "same_beta_restart_loop_used": False,
        "empirical_gradient_rescaling": False,
    }
    write_json(published / "FINAL_RESULT.json", final)
    print(json.dumps(final, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
