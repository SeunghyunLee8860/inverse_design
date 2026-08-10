#!/usr/bin/env python3
"""Contact-anchored PTE topology optimization using NLopt LD_MMA.

Unlike the discarded Run014 update and the diagnostic custom-MMA Run016,
this driver delegates every design update and moving-asymptote decision to
NLopt's LD_MMA implementation.  There is no user-defined move limit, Adam
state, normalized-gradient direction, or post-update clipping.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
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
from photothermal_pte.optimization_runs.tairte4_flake_topology.run_true_mma_optimization import (
    BETA_SCHEDULE,
    FULL_SOLID_TERMINAL_CONDUCTANCE_S,
    MORPHOLOGY_START_BETA,
    OBJECTIVE_SCALE_AT_REFERENCE_POWER_A,
    REFERENCE_INCIDENT_POWER_W,
    canonical_constraints,
    equivalent_current,
    evaluate,
    publish_plot,
    record_manifest_entry,
    sha256,
    stage_morphology_caps,
    verify_file,
    write_json,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
MAXIMUM_STAGE_EVALUATIONS = 40
NLOPT_FTOL_REL = 1.0e-3
# The first LD_MMA asymptote step from uniform rho can be O(1e-4).  A loose
# 1e-3 relative-x tolerance incorrectly promoted beta after that single trial.
# Keep x-tolerance far below that initialization scale; objective plateau and
# the strict constraints remain the practical stage stopping conditions.
NLOPT_XTOL_REL = 1.0e-7
NLOPT_CONSTRAINT_TOL = 1.0e-6


def emit(path: Path, event: str, **values: object) -> None:
    row = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **values,
    }
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


@dataclass
class Point:
    x: np.ndarray
    rho: np.ndarray
    result: dict[str, object]
    gradient_physical_A: np.ndarray
    gradient_conductance_S: np.ndarray
    objective_scaled_minimize: float
    objective_gradient_latent_minimize: np.ndarray
    constraint_names: list[str]
    constraint_values: np.ndarray
    constraint_gradients: np.ndarray
    summary: dict[str, object]


class StageEvaluator:
    """One cached full-physics callback shared by NLopt objective/constraints."""

    def __init__(
        self,
        *,
        beta: float,
        polarization: str,
        gpu: int,
        raw_root: Path,
        published: Path,
        events: Path,
        history: list[dict[str, object]],
        manifest: dict[str, object],
        base_fsp: Path,
        base_sha256: str,
        jacobian_dir: Path,
        minimum_conductance_S: float,
        morphology_caps: np.ndarray,
        fixed_source_power_W: float | None,
        evaluation_counter: int,
        global_evaluation: int,
        constraint_device: str,
        algorithm_label: str = "NLopt LD_MMA",
        output_slug: str = "nlopt_mma",
    ) -> None:
        self.beta = beta
        self.polarization = polarization
        self.gpu = gpu
        self.raw_root = raw_root
        self.published = published
        self.events = events
        self.history = history
        self.manifest = manifest
        self.base_fsp = base_fsp
        self.base_sha256 = base_sha256
        self.jacobian_dir = jacobian_dir
        self.minimum_conductance_S = minimum_conductance_S
        self.morphology_caps = morphology_caps
        self.fixed_source_power_W = fixed_source_power_W
        self.evaluation_counter = evaluation_counter
        self.global_evaluation = global_evaluation
        self.constraint_device = constraint_device
        self.algorithm_label = algorithm_label
        self.output_slug = output_slug
        self.last: Point | None = None
        self.stage_full_physics_evaluations = 0

    def point(self, vector: np.ndarray) -> Point:
        x = np.asarray(vector, dtype=np.float64).reshape(MAPPING.shape)
        if self.last is not None and np.array_equal(x, self.last.x):
            return self.last
        if np.any(x < 0.0) or np.any(x > 1.0) or not np.all(np.isfinite(x)):
            raise RuntimeError("NLopt supplied an invalid latent design")
        rho = MAPPING.physical(x, self.beta)
        self.evaluation_counter += 1
        self.global_evaluation += 1
        output = self.raw_root / (
            f"evaluation_{self.evaluation_counter:04d}_beta{self.beta:g}_{self.output_slug}"
        )
        result, gradient_physical, gradient_conductance = evaluate(
            rho,
            polarization=self.polarization,
            output=output,
            gpu=self.gpu,
            events=self.events,
            base_fsp=self.base_fsp,
            base_sha256=self.base_sha256,
            jacobian_dir=self.jacobian_dir,
        )
        self.stage_full_physics_evaluations += 1
        source_power = float(result["forward"]["source_power_W"])
        if self.fixed_source_power_W is None:
            self.fixed_source_power_W = source_power
        source_change = abs(source_power - self.fixed_source_power_W) / self.fixed_source_power_W
        if source_change >= 0.005:
            raise RuntimeError("fixed-source-power audit changed by >=0.5%")
        objective_A = float(result["objective_A"])
        objective_scaled = -equivalent_current(
            objective_A, self.fixed_source_power_W
        ) / OBJECTIVE_SCALE_AT_REFERENCE_POWER_A
        gradient_latent = MAPPING.vjp(x, gradient_physical, self.beta)
        objective_gradient = (
            -gradient_latent
            * REFERENCE_INCIDENT_POWER_W
            / self.fixed_source_power_W
            / OBJECTIVE_SCALE_AT_REFERENCE_POWER_A
        )
        names, fval, dfdx, summary, _ = canonical_constraints(
            latent=x,
            beta=self.beta,
            terminal_conductance_S=float(result["terminal_conductance_S"]),
            gradient_terminal_conductance_physical_S=gradient_conductance,
            minimum_terminal_conductance_S=self.minimum_conductance_S,
            morphology_caps=self.morphology_caps,
            device=self.constraint_device,
        )
        point = Point(
            x=x.copy(),
            rho=rho,
            result=result,
            gradient_physical_A=gradient_physical,
            gradient_conductance_S=gradient_conductance,
            objective_scaled_minimize=float(objective_scaled),
            objective_gradient_latent_minimize=objective_gradient,
            constraint_names=names,
            constraint_values=fval,
            constraint_gradients=dfdx,
            summary=summary,
        )
        previous_x = self.last.x if self.last is not None else x
        step = x - previous_x
        row = {
            "role": "nlopt_evaluation",
            "algorithm": self.algorithm_label,
            "nlopt_version": nlopt.__version__,
            "evaluation_id": self.evaluation_counter,
            "global_update": self.global_evaluation,
            "stage_update": self.stage_full_physics_evaluations,
            "beta": self.beta,
            "objective_A": objective_A,
            "objective_at_reference_power_A": equivalent_current(
                objective_A, self.fixed_source_power_W
            ),
            "objective_scaled_minimize": float(objective_scaled),
            "fixed_source_power_W": self.fixed_source_power_W,
            "source_power_relative_change": source_change,
            "terminal_conductance_S": float(result["terminal_conductance_S"]),
            "minimum_terminal_conductance_S": self.minimum_conductance_S,
            "constraint_names": names,
            "constraint_values": fval.tolist(),
            "maximum_constraint_value": float(np.max(fval)),
            "morphology_caps": self.morphology_caps.tolist(),
            "gray_fraction_0p01_0p99": summary["gray_fraction_0p01_0p99"],
            "binarization": summary["binarization_mean_4rho1mrho"],
            "exact_bad_cells": summary["exact"]["total_bad_cell_count"],
            "rho_mean": summary["rho_mean"],
            "latent_range": [float(np.min(x)), float(np.max(x))],
            "maximum_absolute_step_from_previous_evaluation": float(np.max(np.abs(step))),
            "rms_step_from_previous_evaluation": float(np.sqrt(np.mean(step * step))),
            "manual_move_limit": None,
            "used_adam": False,
            "gradient_direction_normalization": False,
            "post_update_hard_clipping": False,
            "symmetry_constraint": False,
            "volume_constraint": False,
        }
        self.history.append(row)
        self.manifest["evaluations"][f"{self.evaluation_counter:04d}"] = record_manifest_entry(result)
        write_json(self.raw_root / "history.json", self.history)
        write_json(self.published / "optimization_history.json", self.history)
        write_json(self.published / "RAW_ARTIFACT_MANIFEST.json", self.manifest)
        write_json(self.published / "latest_summary.json", summary)
        write_json(self.published / f"evaluation_{self.global_evaluation:04d}.json", row)
        publish_plot(
            self.published,
            self.history,
            rho,
            gradient_physical,
            summary,
            evaluation_id=self.evaluation_counter,
            label=f"{self.output_slug}_evaluation",
        )
        emit(
            self.events,
            "nlopt_full_physics_evaluation",
            beta=self.beta,
            evaluation_id=self.evaluation_counter,
            objective_at_reference_power_A=row["objective_at_reference_power_A"],
            maximum_constraint_value=row["maximum_constraint_value"],
            manual_move_limit=None,
        )
        self.last = point
        return point

    def objective(self, vector: np.ndarray, gradient: np.ndarray) -> float:
        point = self.point(vector)
        if gradient.size:
            gradient[:] = point.objective_gradient_latent_minimize.ravel()
        return point.objective_scaled_minimize

    def constraints(
        self, result: np.ndarray, vector: np.ndarray, gradient: np.ndarray
    ) -> None:
        point = self.point(vector)
        result[:] = point.constraint_values
        if gradient.size:
            gradient[:, :] = point.constraint_gradients.reshape(
                len(point.constraint_names), -1
            )


def make_optimizer(evaluator: StageEvaluator, constraint_count: int) -> nlopt.opt:
    variable_count = int(np.prod(MAPPING.shape))
    optimizer = nlopt.opt(nlopt.LD_MMA, variable_count)
    optimizer.set_lower_bounds(np.zeros(variable_count))
    optimizer.set_upper_bounds(np.ones(variable_count))
    optimizer.set_min_objective(evaluator.objective)
    optimizer.add_inequality_mconstraint(
        evaluator.constraints,
        np.full(constraint_count, NLOPT_CONSTRAINT_TOL),
    )
    optimizer.set_ftol_rel(NLOPT_FTOL_REL)
    optimizer.set_xtol_rel(NLOPT_XTOL_REL)
    optimizer.set_maxeval(MAXIMUM_STAGE_EVALUATIONS)
    return optimizer


def initial_manifest(base_fsp: Path, base_sha256: str, jacobian_dir: Path) -> dict[str, object]:
    certificate = jacobian_dir / "component_yee_jacobian_result.json"
    return {
        "schema": "nlopt-ld-mma-contact-anchored-raw-artifact-manifest-v1",
        "raw_artifacts_committed_to_git": False,
        "optimizer": {
            "library": "NLopt",
            "version": nlopt.__version__,
            "algorithm": "LD_MMA",
            "manual_move_limit": None,
            "custom_mma_update_used": False,
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
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--base-fsp", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--jacobian-dir", required=True, type=Path)
    parser.add_argument("--connectivity-fraction", type=float, default=0.10)
    parser.add_argument("--constraint-device", default="cuda:0")
    parser.add_argument("--maximum-beta-stages", type=int, default=0)
    args = parser.parse_args()
    CONTRACT.validate()
    if CONTRACT.geometry_mode != "contact_anchored":
        raise RuntimeError("NLopt production requires contact_anchored geometry")
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
    minimum_conductance = (
        args.connectivity_fraction * FULL_SOLID_TERMINAL_CONDUCTANCE_S
    )
    latent = np.full(MAPPING.shape, 0.5, dtype=np.float64)
    history: list[dict[str, object]] = []
    manifest = initial_manifest(base_fsp, args.base_sha256, jacobian_dir)
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
        )
        optimizer = make_optimizer(evaluator, constraint_count)
        emit(
            events,
            "nlopt_stage_start",
            beta=beta,
            algorithm="LD_MMA",
            nlopt_version=nlopt.__version__,
            manual_move_limit=None,
            ftol_rel=NLOPT_FTOL_REL,
            xtol_rel=NLOPT_XTOL_REL,
            maxeval=MAXIMUM_STAGE_EVALUATIONS,
        )
        optimum = optimizer.optimize(latent.ravel()).reshape(MAPPING.shape)
        result_code = optimizer.last_optimize_result()
        final_point = evaluator.point(optimum.ravel())
        if np.max(final_point.constraint_values) > NLOPT_CONSTRAINT_TOL:
            raise RuntimeError("NLopt stage returned an infeasible physical design")
        if result_code == nlopt.MAXEVAL_REACHED:
            raise RuntimeError("NLopt stage hit maxeval without convergence; refusing beta promotion")
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
        }
        write_json(published / "RAW_ARTIFACT_MANIFEST.json", manifest)
        emit(
            events,
            "nlopt_stage_complete",
            beta=beta,
            result_code=result_code,
            evaluations=evaluator.stage_full_physics_evaluations,
        )

    if args.maximum_beta_stages:
        return 0
    final_rho = MAPPING.physical(latent, BETA_SCHEDULE[-1])
    exact, _ = exact_binary_audit(final_rho)
    if exact["total_bad_cell_count"] != 0 or float(
        np.mean((final_rho > 0.01) & (final_rho < 0.99))
    ) >= 0.01:
        raise RuntimeError("final NLopt continuous design did not pass binary/500 nm gates")
    binary = (final_rho >= 0.5).astype(np.float64)
    binary_audit, _ = exact_binary_audit(binary)
    if binary_audit["total_bad_cell_count"] != 0:
        raise RuntimeError("thresholded NLopt binary failed exact 500 nm audit")
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
        raise RuntimeError("fresh NLopt exact-binary evaluation failed")
    final_result = json.loads(final_result_path.read_text())
    if not final_result.get("passed"):
        raise RuntimeError("fresh NLopt exact-binary result is not passed")
    write_json(published / "FINAL_RESULT.json", {
        "passed": True,
        "status": "VALIDATED_NLOPT_LD_MMA_EXACT_BINARY_CONTACT_ANCHORED_PTE_OPTIMIZATION",
        "polarization": args.polarization,
        "algorithm": "NLopt LD_MMA",
        "nlopt_version": nlopt.__version__,
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
