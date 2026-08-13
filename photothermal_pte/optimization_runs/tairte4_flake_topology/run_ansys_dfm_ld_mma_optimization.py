#!/usr/bin/env python3
"""Audited beta continuation for pure terminal-PTE-current LD_MMA.

The fabrication term follows the v261 LumOpt continuation pattern: a
grayscale phase at beta=1, multiplicative beta continuation, a bounded
continuation budget with an audited plateau exit, and a DFM penalty that grows
smoothly with projection strength.  This production contract uses a factor of
two so that a converged beta stage is not repeatedly re-solved.  Unlike the
superseded fixed-cap driver, this implementation has no late constraint-
restoration loop.  Raw current and the fabrication penalty remain separately
auditable at every full-physics evaluation.
"""

from __future__ import annotations

import argparse
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
from photothermal_pte.optimization_runs.tairte4_flake_topology.run_nlopt_mma_optimization import (
    StageEvaluator,
    emit,
    initial_manifest,
    make_optimizer,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.run_pure_current_ld_mma_optimization import (
    midpoint_projection_derivative,
    verify_optimizer_code_manifest,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.repair_exact_binary_candidates import (
    active_set_repair,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.run_true_mma_optimization import (
    REFERENCE_INCIDENT_POWER_W,
    sha256,
    verify_file,
    write_json,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
SCHEMA = "ansys-v261-style-dfm-pure-current-ld-mma-v5"
BETA_FACTOR = 2.0
# Do not saturate continuation at beta=128.  In production runs an exact
# 500-nm cleanup can still perturb the terminal-current objective even when the
# projected density is already visually binary.  Continue the standard
# factor-two schedule through 256, 512, and 1024, and rely on the existing
# fresh-GPU objective-preservation gate (<=1% loss) rather than beta alone to
# promote a final design.
MAXIMUM_BETA = 1024.0
GRAYSCALE_EVALUATIONS = 40
CONTINUATION_EVALUATIONS = 20
DFM_ACTIVATION_BETA = 4.0
DFM_PENALTY_AT_ACTIVATION = 10.0
DFM_PENALTY_MAXIMUM = 1.0e4
FINAL_DISCRETENESS_MINIMUM = 0.99
FINAL_GRAY_FRACTION_MAXIMUM = 0.01
TARGET_INITIAL_PHYSICAL_STEP = 0.025
BASE_RHO_INIT = 10.0
PLATEAU_MINIMUM_EVALUATIONS = 5
PLATEAU_WINDOW_EVALUATIONS = 4
PLATEAU_FOM_RELATIVE_RANGE_MAXIMUM = 1.0e-4
PLATEAU_RMS_STEP_MAXIMUM = 1.0e-4
PLATEAU_GRAY_ABSOLUTE_RANGE_MAXIMUM = 1.0e-3
EXACT_CLEANUP_MINIMUM_BETA = 16.0
EXACT_CLEANUP_MAXIMUM_GRAY_FRACTION = 0.02
CONSTRAINT_PLATEAU_RELATIVE_RANGE_MAXIMUM = 1.0e-2
CONSTRAINT_PLATEAU_EXACT_ABSOLUTE_RANGE_MINIMUM = 2

# Constraint-aware continuation is a separate production contract.  The
# original Ansys-style path remains available for provenance, while new runs
# may explicitly require fabrication progress before beta promotion.  Exact
# thresholded counts are deliberately diagnostic-only at beta 1 and 2 because
# a gray field has no unique binary topology.  From beta 4 onward they become
# an audited promotion gate, while differentiable KS-opening inequalities
# provide the actual MMA gradients.
CONSTRAINT_AWARE_HARD_START_BETA = 4.0
CONSTRAINT_AWARE_EXACT_REDUCTION = {
    4.0: 0.75,
    8.0: 0.50,
    16.0: 0.25,
}
CONSTRAINT_AWARE_EXACT_ZERO_BETA = 32.0
CONSTRAINT_AWARE_CAP_INITIAL_REDUCTION = 0.05
CONSTRAINT_AWARE_CAP_REPEAT_REDUCTION = 0.10
CONSTRAINT_AWARE_CAP_FLOOR = 1.0e-8
CONSTRAINT_AWARE_DFM_WEIGHT_AT_BETA1 = 1.0
CONSTRAINT_AWARE_DFM_WEIGHT_MAXIMUM = 1.0e4


class AdaptivePlateauStop(Exception):
    """Internal signal that advances continuation after an audited plateau."""


def adaptive_plateau_diagnostic(
    stage_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Return the deterministic multi-metric continuation plateau audit."""
    count = len(stage_rows)
    diagnostic: dict[str, object] = {
        "passed": False,
        "minimum_evaluations": PLATEAU_MINIMUM_EVALUATIONS,
        "window_evaluations": PLATEAU_WINDOW_EVALUATIONS,
        "fom_relative_range_maximum": PLATEAU_FOM_RELATIVE_RANGE_MAXIMUM,
        "rms_step_maximum": PLATEAU_RMS_STEP_MAXIMUM,
        "gray_absolute_range_maximum": PLATEAU_GRAY_ABSOLUTE_RANGE_MAXIMUM,
        "stage_evaluations_observed": count,
    }
    if count < PLATEAU_MINIMUM_EVALUATIONS:
        diagnostic["reason"] = "minimum_stage_evaluations_not_reached"
        return diagnostic
    window = stage_rows[-PLATEAU_WINDOW_EVALUATIONS:]
    fom = np.asarray(
        [float(row["objective_at_reference_power_A"]) for row in window],
        dtype=np.float64,
    )
    rms_step = np.asarray(
        [float(row["rms_step_from_previous_evaluation"]) for row in window],
        dtype=np.float64,
    )
    gray = np.asarray(
        [float(row["gray_fraction_0p01_0p99"]) for row in window],
        dtype=np.float64,
    )
    smooth = np.asarray(
        [
            float(row.get("smooth_solid_constraint", np.inf))
            + float(row.get("smooth_void_constraint", np.inf))
            for row in window
        ],
        dtype=np.float64,
    )
    exact_bad = np.asarray(
        [int(row.get("exact_bad_cells", -1)) for row in window], dtype=np.int64
    )
    fom_scale = max(float(np.max(np.abs(fom))), np.finfo(np.float64).tiny)
    fom_relative_range = float(np.ptp(fom) / fom_scale)
    rms_step_maximum = float(np.max(rms_step))
    gray_absolute_range = float(np.ptp(gray))
    smooth_scale = max(float(np.max(np.abs(smooth))), np.finfo(np.float64).tiny)
    smooth_relative_range = float(np.ptp(smooth) / smooth_scale)
    exact_range = int(np.ptp(exact_bad))
    exact_range_limit = max(
        CONSTRAINT_PLATEAU_EXACT_ABSOLUTE_RANGE_MINIMUM,
        int(np.ceil(0.01 * max(int(np.max(exact_bad)), 1))),
    )
    constraint_plateau = bool(
        np.all(np.isfinite(smooth))
        and smooth_relative_range < CONSTRAINT_PLATEAU_RELATIVE_RANGE_MAXIMUM
        and exact_range <= exact_range_limit
    )
    passed = bool(
        fom_relative_range < PLATEAU_FOM_RELATIVE_RANGE_MAXIMUM
        and rms_step_maximum < PLATEAU_RMS_STEP_MAXIMUM
        and gray_absolute_range < PLATEAU_GRAY_ABSOLUTE_RANGE_MAXIMUM
    )
    diagnostic.update(
        {
            "passed": passed,
            "reason": "adaptive_plateau" if passed else "stage_still_moving",
            "window_evaluation_ids": [int(row["evaluation_id"]) for row in window],
            "fom_relative_range": fom_relative_range,
            "rms_step_maximum_observed": rms_step_maximum,
            "gray_absolute_range": gray_absolute_range,
            "smooth_constraint_relative_range": smooth_relative_range,
            "exact_bad_cell_range": exact_range,
            "exact_bad_cell_range_limit": exact_range_limit,
            "constraint_plateau": constraint_plateau,
        }
    )
    return diagnostic


def constraint_aware_exact_target(beta: float, stage_entry_bad: int) -> int | None:
    """Return the exact-audit target required before beta promotion.

    The thresholded audit is not used at beta 1 or 2.  At beta 4--16 each
    continuation stage must remove a documented fraction of the violations
    present when that beta was entered.  Beta 32 and above require zero.
    """

    if stage_entry_bad < 0:
        raise ValueError("stage-entry exact violation count must be nonnegative")
    if beta < CONSTRAINT_AWARE_HARD_START_BETA:
        return None
    if beta >= CONSTRAINT_AWARE_EXACT_ZERO_BETA:
        return 0
    factor = CONSTRAINT_AWARE_EXACT_REDUCTION.get(float(beta))
    if factor is None:
        # Non-power-of-two warm starts use the next stricter tabulated gate.
        candidates = sorted(CONSTRAINT_AWARE_EXACT_REDUCTION)
        factor = next(
            (CONSTRAINT_AWARE_EXACT_REDUCTION[value] for value in candidates if beta <= value),
            0.0,
        )
    return int(np.floor(float(factor) * stage_entry_bad))


def constraint_aware_dfm_penalty_weight(beta: float) -> float:
    """Gradually strengthen differentiable local-feature pressure with beta."""

    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    return float(
        min(
            CONSTRAINT_AWARE_DFM_WEIGHT_AT_BETA1 * beta * beta,
            CONSTRAINT_AWARE_DFM_WEIGHT_MAXIMUM,
        )
    )


def constraint_aware_promotion_diagnostic(
    stage_rows: list[dict[str, object]],
    *,
    beta: float,
    exact_bad_target: int | None,
    require_hard_feasible: bool,
) -> dict[str, object]:
    """Require objective convergence *and* fabrication progress for promotion."""

    diagnostic = adaptive_plateau_diagnostic(stage_rows)
    diagnostic["policy"] = "constraint_aware_beta_promotion_v1"
    diagnostic["beta"] = float(beta)
    diagnostic["exact_bad_target"] = exact_bad_target
    if not stage_rows:
        diagnostic.update(
            {
                "hard_constraints_feasible": not require_hard_feasible,
                "exact_target_passed": exact_bad_target is None,
                "exact_nonincreasing": True,
                "promotion_passed": False,
            }
        )
        return diagnostic
    latest = stage_rows[-1]
    maximum_constraint = latest.get("maximum_constraint_value")
    hard_feasible = bool(
        not require_hard_feasible
        or (
            maximum_constraint is not None
            and np.isfinite(float(maximum_constraint))
            and float(maximum_constraint) <= 0.0
        )
    )
    latest_bad = int(latest.get("exact_bad_cells", -1))
    first_bad = int(stage_rows[0].get("exact_bad_cells", latest_bad))
    exact_passed = bool(
        exact_bad_target is None
        or (latest_bad >= 0 and latest_bad <= exact_bad_target)
    )
    exact_nonincreasing = bool(
        exact_bad_target is None
        or (latest_bad >= 0 and first_bad >= 0 and latest_bad <= first_bad)
    )
    promotion_passed = bool(
        diagnostic.get("passed")
        and hard_feasible
        and exact_passed
        and exact_nonincreasing
    )
    diagnostic.update(
        {
            "objective_design_gray_plateau": bool(diagnostic.get("passed")),
            "hard_constraints_feasible": hard_feasible,
            "latest_exact_bad_cells": latest_bad,
            "stage_first_exact_bad_cells": first_bad,
            "exact_target_passed": exact_passed,
            "exact_nonincreasing": exact_nonincreasing,
            "promotion_passed": promotion_passed,
            "reason": (
                "constraint_aware_promotion_passed"
                if promotion_passed
                else "constraint_aware_stage_not_ready"
            ),
        }
    )
    return diagnostic


class AdaptiveStageEvaluator(StageEvaluator):
    """Stage evaluator that exits LD_MMA only after a documented plateau."""

    plateau_diagnostic: dict[str, object] | None = None

    def __init__(
        self,
        *args: object,
        constraint_aware_continuation: bool = False,
        exact_bad_target: int | None = None,
        require_hard_feasible: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.constraint_aware_continuation = bool(constraint_aware_continuation)
        self.exact_bad_target = exact_bad_target
        self.require_hard_feasible = bool(require_hard_feasible)

    def objective(self, vector: np.ndarray, gradient: np.ndarray) -> float:
        value = super().objective(vector, gradient)
        stage_rows = self.history[-self.stage_full_physics_evaluations :]
        if self.constraint_aware_continuation:
            diagnostic = constraint_aware_promotion_diagnostic(
                stage_rows,
                beta=self.beta,
                exact_bad_target=self.exact_bad_target,
                require_hard_feasible=self.require_hard_feasible,
            )
            stop = bool(diagnostic["promotion_passed"])
        else:
            diagnostic = adaptive_plateau_diagnostic(stage_rows)
            stop = bool(diagnostic["passed"])
        self.plateau_diagnostic = diagnostic
        if stop:
            event_payload = dict(diagnostic)
            event_payload.setdefault("beta", self.beta)
            emit(
                self.events,
                (
                    "constraint_aware_continuation_promotion"
                    if self.constraint_aware_continuation
                    else "adaptive_continuation_plateau"
                ),
                **event_payload,
            )
            raise AdaptivePlateauStop("audited continuation plateau")
        return value


def beta_sequence(start_beta: float = 1.0) -> tuple[float, ...]:
    """Return a factor-two continuation beginning at any audited checkpoint."""
    if not np.isfinite(start_beta) or not 0.0 < start_beta <= MAXIMUM_BETA:
        raise ValueError("start beta must be finite and in (0, maximum beta]")
    values = [float(start_beta)]
    while values[-1] < MAXIMUM_BETA:
        values.append(float(f"{min(MAXIMUM_BETA, values[-1] * BETA_FACTOR):.12g}"))
    return tuple(values)


def dfm_penalty_weight(beta: float) -> float:
    """Increase minimum-feature pressure gradually with projection strength."""
    if beta < DFM_ACTIVATION_BETA:
        return 0.0
    return float(
        min(
            DFM_PENALTY_AT_ACTIVATION
            * (beta / DFM_ACTIVATION_BETA) ** 2,
            DFM_PENALTY_MAXIMUM,
        )
    )


def stage_controls(beta: float) -> dict[str, object]:
    reference = midpoint_projection_derivative(1.0)
    ratio = midpoint_projection_derivative(beta) / reference
    return {
        "initial_step": TARGET_INITIAL_PHYSICAL_STEP / ratio,
        "rho_init": BASE_RHO_INIT * ratio * ratio,
        "always_improve": 1,
        "inner_gradients": 1,
        "projection_midpoint_derivative": midpoint_projection_derivative(beta),
        "projection_derivative_ratio_to_beta1": ratio,
        "manual_move_limit": None,
    }


def final_geometry_gate(rho: np.ndarray) -> dict[str, object]:
    exact, _ = exact_binary_audit(rho)
    gray = float(np.mean((rho > 0.01) & (rho < 0.99)))
    binarization = float(np.mean(4.0 * rho * (1.0 - rho)))
    discreteness = 1.0 - binarization
    passed = bool(
        discreteness > FINAL_DISCRETENESS_MINIMUM
        and gray < FINAL_GRAY_FRACTION_MAXIMUM
        and exact["total_bad_cell_count"] == 0
    )
    return {
        "passed": passed,
        "discreteness": discreteness,
        "binarization_mean_4rho1mrho": binarization,
        "gray_fraction_0p01_0p99": gray,
        "exact_500nm_audit": exact,
    }


def evaluate_exact_cleanup_candidates(
    rho: np.ndarray,
    *,
    raw_root: Path,
    base_fsp: Path,
    base_sha256: str,
    polarization: str,
    gpu: int,
    reference_objective_A: float,
    attempt_label: str,
) -> dict[str, object]:
    """Force both 500-nm phase audits to zero, then select by fresh physics."""
    thresholded = np.asarray(rho >= 0.5, dtype=bool)
    cleanup_root = raw_root / f"forced_exact_500nm_cleanup_{attempt_label}"
    cleanup_root.mkdir(parents=True, exist_ok=True)
    if any(cleanup_root.iterdir()):
        raise RuntimeError("refusing to overwrite a non-empty exact-cleanup directory")
    candidates: dict[str, dict[str, object]] = {}
    for order in ("solid_first", "void_first"):
        candidate, repair_history, stop_reason = active_set_repair(
            thresholded, order, 100
        )
        audit, _ = exact_binary_audit(candidate.astype(np.float64))
        if not audit["passed"]:
            candidates[order] = {
                "audit": audit,
                "repair_stop_reason": stop_reason,
                "repair_history": repair_history,
                "changed_node_count": int(np.count_nonzero(candidate != thresholded)),
                "density": None,
                "result": None,
                "numerical_physics_gates_passed": False,
                "eligible_for_selection": False,
            }
            continue
        density_path = cleanup_root / f"{order}_density.npz"
        np.savez_compressed(density_path, rho=candidate.astype(np.float64))
        output = cleanup_root / f"{order}_objective"
        command = [
            sys.executable,
            "-m",
            "photothermal_pte.optimization_runs.tairte4_flake_topology.evaluate_binary_objective",
            "--base-fsp", str(base_fsp),
            "--base-sha256", base_sha256,
            "--rho-npz", str(density_path),
            "--output-dir", str(output),
            "--polarization", polarization,
            "--gpu-device", f"GPU {gpu}",
            "--cuda-device", "0",
            "--reference-objective-A", str(reference_objective_A),
        ]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        completed = subprocess.run(command, cwd=REPOSITORY, env=environment)
        result_path = output / "binary_objective_result.json"
        if not result_path.is_file():
            raise RuntimeError(f"{order} objective result is missing")
        result = json.loads(result_path.read_text())
        gates = result.get("gates", {})
        numerical_pass = bool(
            float(gates.get("optical_closure", np.inf)) < 0.005
            and float(gates.get("Q_mapping_error", np.inf)) < 0.005
            and float(gates.get("thermal_forward_residual", np.inf)) < 1.0e-8
            and float(gates.get("thermal_energy_balance", np.inf)) < 0.01
            and float(gates.get("electrical_weighting_residual", np.inf)) < 1.0e-8
            and np.isfinite(float(result.get("objective_A", np.nan)))
            and float(result.get("terminal_conductance_S", 0.0)) > 0.0
        )
        if completed.returncode not in (0, 1) or not numerical_pass:
            raise RuntimeError(f"{order} exact candidate failed numerical physics gates")
        candidates[order] = {
            "audit": audit,
            "repair_stop_reason": stop_reason,
            "repair_history": repair_history,
            "changed_node_count": int(np.count_nonzero(candidate != thresholded)),
            "density": {
                "path": str(density_path),
                "size_bytes": density_path.stat().st_size,
                "sha256": sha256(density_path),
            },
            "result": result,
            "numerical_physics_gates_passed": numerical_pass,
            "eligible_for_selection": True,
        }
    eligible = [
        name for name, row in candidates.items()
        if bool(row.get("eligible_for_selection"))
    ]
    if not eligible:
        raise RuntimeError("no exact-cleanup ordering reached zero 500 nm violations")
    selected = max(
        eligible,
        key=lambda name: float(candidates[name]["result"]["objective_A"]),
    )
    return {
        "trigger": "beta_raised_and_fom_plus_constraints_plateaued",
        "selection_rule": "highest fresh unrescaled exact-binary terminal current",
        "selected": selected,
        "candidates": candidates,
    }


def cleanup_objective_preserved(cleanup: dict[str, object]) -> bool:
    """Return whether the selected exact candidate passed its fixed 1% gate."""
    selected = str(cleanup["selected"])
    candidate = cleanup["candidates"][selected]
    return bool(candidate["result"].get("passed", False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--published-dir", required=True, type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--base-fsp", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--jacobian-dir", required=True, type=Path)
    parser.add_argument("--constraint-device", default="cuda:0")
    parser.add_argument(
        "--initial-latent-npz",
        type=Path,
        help=(
            "Warm-start density variable containing a 'latent' array. This starts "
            "a fresh NLopt LD_MMA stage because internal MMA asymptotes are not serializable."
        ),
    )
    parser.add_argument(
        "--recovery-append",
        action="store_true",
        help="Append a warm-restart stage to an interrupted Run044/Run045 history.",
    )
    parser.add_argument(
        "--start-beta",
        type=float,
        help="Continuation beta at which the warm restart begins.",
    )
    parser.add_argument(
        "--output-slug",
        default="ansys_dfm_ld_mma_recovery1",
        help="Unique raw evaluation suffix used by a recovery generation.",
    )
    parser.add_argument(
        "--maximum-stages",
        type=int,
        default=0,
        help="Offline/smoke limiter only; zero means continue to the final gate.",
    )
    parser.add_argument(
        "--hard-morphology-constraints",
        action="store_true",
        help=(
            "Use the solid/void opening residuals as two NLopt LD_MMA "
            "inequalities instead of adding them to the scalar objective."
        ),
    )
    parser.add_argument(
        "--hard-cap-relative-slack",
        type=float,
        default=0.01,
        help="Relative slack above each stage's feasible starting residual.",
    )
    parser.add_argument(
        "--hard-rho-init",
        type=float,
        default=0.01,
        help=(
            "NLopt CCSA curvature initialization used only with explicit hard "
            "morphology inequalities. A smaller positive value avoids relearning "
            "the first several near-zero steps after a warm restart."
        ),
    )
    parser.add_argument(
        "--constraint-aware-continuation",
        action="store_true",
        help=(
            "Use gradual KS-opening pressure from beta=1, activate explicit "
            "solid/void inequalities at beta=4, and prohibit beta promotion "
            "until the stage-specific exact 500-nm audit target is met."
        ),
    )
    args = parser.parse_args()

    CONTRACT.validate()
    if CONTRACT.geometry_mode not in {"contact_anchored", "left_right_contact_anchored"}:
        raise RuntimeError("production optimization requires a contact-anchored geometry")
    if not 0.0 <= args.hard_cap_relative_slack <= 0.10:
        raise ValueError("hard morphology cap slack must be in [0, 0.10]")
    if not np.isfinite(args.hard_rho_init) or args.hard_rho_init <= 0.0:
        raise ValueError("--hard-rho-init must be finite and positive")
    if args.constraint_aware_continuation and args.hard_morphology_constraints:
        raise ValueError(
            "constraint-aware continuation owns the dynamic hard constraints; "
            "do not combine it with the legacy hard-recovery flag"
        )
    code_provenance = verify_optimizer_code_manifest()
    base_fsp = verify_file(args.base_fsp, args.base_sha256)
    jacobian_dir = args.jacobian_dir.expanduser().resolve()
    certificate = jacobian_dir / "component_yee_jacobian_result.json"
    if not certificate.is_file() or not json.loads(certificate.read_text()).get("passed"):
        raise RuntimeError("component-Yee Jacobian certificate is missing or failed")

    raw_root = args.raw_root.expanduser().resolve()
    published = args.published_dir.expanduser().resolve()
    if args.recovery_append and args.initial_latent_npz is None:
        raise RuntimeError("--recovery-append requires --initial-latent-npz")
    if not args.recovery_append:
        if raw_root.exists() and any(raw_root.iterdir()):
            raise RuntimeError(f"refusing non-empty raw root: {raw_root}")
        if published.exists() and any(published.iterdir()):
            raise RuntimeError(f"refusing non-empty published directory: {published}")
    raw_root.mkdir(parents=True, exist_ok=True)
    published.mkdir(parents=True, exist_ok=True)
    events = raw_root / "events.jsonl"

    initialization: dict[str, object]
    if args.initial_latent_npz is None:
        latent = np.full(MAPPING.shape, 0.5, dtype=np.float64)
        initialization = {"kind": "uniform_latent", "value": 0.5}
    else:
        initial_path = args.initial_latent_npz.expanduser().resolve()
        if not initial_path.is_file():
            raise RuntimeError(f"initial latent NPZ is missing: {initial_path}")
        with np.load(initial_path) as loaded:
            if "latent" not in loaded:
                raise RuntimeError("initial latent NPZ does not contain 'latent'")
            latent = np.asarray(loaded["latent"], dtype=np.float64)
        if latent.shape != MAPPING.shape:
            raise RuntimeError(f"initial latent shape {latent.shape} != {MAPPING.shape}")
        if not np.all(np.isfinite(latent)) or np.any(latent < 0.0) or np.any(latent > 1.0):
            raise RuntimeError("initial latent design is non-finite or outside [0,1]")
        initialization = {
            "kind": "native_LD_MMA_warm_restart",
            "path": str(initial_path),
            "size_bytes": initial_path.stat().st_size,
            "sha256": sha256(initial_path),
            "note": "NLopt internal asymptotes reset at the last fully evaluated design.",
        }

    if args.recovery_append:
        history_path = published / "optimization_history.json"
        manifest_path = published / "RAW_ARTIFACT_MANIFEST.json"
        if not history_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("recovery append requires existing history and manifest")
        history = json.loads(history_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(history, list) or not history:
            raise RuntimeError("recovery history is missing or empty")
        if not isinstance(manifest, dict) or not isinstance(manifest.get("evaluations"), dict):
            raise RuntimeError("recovery manifest is invalid")
    else:
        history = []
        manifest = initial_manifest(base_fsp, args.base_sha256, jacobian_dir)
        manifest.update({
            "schema": SCHEMA,
            "initialization": initialization,
            "polarization": args.polarization,
            "beta_continuation": {
                "source": "Ansys v261 LumOpt topology.py/optimization.py",
                "factor": BETA_FACTOR,
                "grayscale_evaluations": GRAYSCALE_EVALUATIONS,
                "continuation_evaluations": CONTINUATION_EVALUATIONS,
                "maximum_beta": MAXIMUM_BETA,
            },
            "dfm_penalty": {
                "minimum_feature_nm": 500.0,
                "activation_beta": DFM_ACTIVATION_BETA,
                "formula": "min(10*(beta/4)^2,1e4)",
                "maximum": DFM_PENALTY_MAXIMUM,
                "hard_inequality_restoration_loop": False,
            },
            "termination_gate": {
                "discreteness_minimum": FINAL_DISCRETENESS_MINIMUM,
                "gray_fraction_maximum": FINAL_GRAY_FRACTION_MAXIMUM,
                "exact_500nm_bad_nodes": 0,
            },
            "code_provenance": code_provenance,
        })
    if args.recovery_append:
        manifest["code_provenance"] = code_provenance
        manifest.setdefault("recovery_chain", []).append(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "initialization": initialization,
                "start_beta": args.start_beta,
                "output_slug": args.output_slug,
                "reason": "transient HPC-license checkout failure or storage exhaustion",
                "optimizer_state": "fresh NLopt LD_MMA stage; prior asymptotes are not serializable",
            }
        )
    manifest["schema"] = SCHEMA
    manifest.setdefault("beta_continuation", {}).update({
        "factor": BETA_FACTOR,
        "maximum_beta": MAXIMUM_BETA,
    })
    manifest["beta_continuation"]["adaptive_plateau"] = {
        "enabled": True,
        "minimum_evaluations": PLATEAU_MINIMUM_EVALUATIONS,
        "window_evaluations": PLATEAU_WINDOW_EVALUATIONS,
        "fom_relative_range_maximum": PLATEAU_FOM_RELATIVE_RANGE_MAXIMUM,
        "rms_step_maximum": PLATEAU_RMS_STEP_MAXIMUM,
        "gray_absolute_range_maximum": PLATEAU_GRAY_ABSOLUTE_RANGE_MAXIMUM,
        "constraint_relative_range_maximum": CONSTRAINT_PLATEAU_RELATIVE_RANGE_MAXIMUM,
    }
    if args.constraint_aware_continuation:
        manifest["constraint_aware_continuation"] = {
            "enabled": True,
            "policy": "constraint_aware_beta_promotion_v1",
            "hard_constraint_start_beta": CONSTRAINT_AWARE_HARD_START_BETA,
            "exact_count_interpretation_beta_1_2": "diagnostic_only_for_gray_fields",
            "exact_reduction_fraction_by_beta": {
                str(key): value
                for key, value in CONSTRAINT_AWARE_EXACT_REDUCTION.items()
            },
            "exact_zero_required_from_beta": CONSTRAINT_AWARE_EXACT_ZERO_BETA,
            "cap_initial_reduction": CONSTRAINT_AWARE_CAP_INITIAL_REDUCTION,
            "cap_repeat_reduction": CONSTRAINT_AWARE_CAP_REPEAT_REDUCTION,
            "cap_floor": CONSTRAINT_AWARE_CAP_FLOOR,
            "promotion_requires": [
                "objective_design_gray_plateau",
                "active_KS_constraints_feasible",
                "exact_bad_count_at_or_below_stage_target",
                "exact_bad_count_nonincreasing_within_stage",
            ],
            "termination_requires": [
                "promotion_gate",
                "gray_fraction_below_final_limit",
                "discreteness_above_final_limit",
                "exact_bad_count_zero",
            ],
        }
    manifest["forced_exact_cleanup_policy"] = {
        "enabled": True,
        "minimum_beta": EXACT_CLEANUP_MINIMUM_BETA,
        "maximum_gray_fraction": EXACT_CLEANUP_MAXIMUM_GRAY_FRACTION,
        "requires_fom_plateau": True,
        "requires_smooth_and_exact_constraint_plateau": True,
        "failed_objective_preservation_action": (
            "record diagnostic and continue beta continuation"
        ),
        "candidates": ["solid_first", "void_first"],
        "selection": "fresh unrescaled GPU objective after both exact audits reach zero",
    }
    manifest["dfm_penalty"] = {
        "minimum_feature_nm": 500.0,
        "activation_beta": DFM_ACTIVATION_BETA,
        "weight_at_activation": DFM_PENALTY_AT_ACTIVATION,
        "formula": "min(10*(beta/4)^2,1e4)",
        "maximum": DFM_PENALTY_MAXIMUM,
        "hard_inequality_restoration_loop": False,
        "interpretation": "soft differentiable solid/void feature pressure strengthened with beta",
    }
    if args.hard_morphology_constraints:
        manifest["dfm_penalty"] = {
            "minimum_feature_nm": 500.0,
            "enabled": False,
            "reason": "solid and void residuals are separate LD_MMA inequalities",
        }
        manifest["hard_morphology_constraints"] = {
            "enabled": True,
            "names": ["500nm_solid_opening", "500nm_void_opening"],
            "cap_policy": "fixed within each beta stage at (1+slack)*stage-start residual",
            "relative_slack": args.hard_cap_relative_slack,
            "penalty_weight": 0.0,
            "aggregation": "ks_max",
        }
    elif args.constraint_aware_continuation:
        manifest["dfm_penalty"] = {
            "minimum_feature_nm": 500.0,
            "enabled": True,
            "aggregation": "ks_max",
            "formula": "min(1.0*beta^2,1e4)",
            "role": "gradual objective pressure; beta>=4 also uses separate LD_MMA inequalities",
        }
        manifest["hard_morphology_constraints"] = {
            "enabled_from_beta": CONSTRAINT_AWARE_HARD_START_BETA,
            "names": ["500nm_solid_opening", "500nm_void_opening"],
            "cap_policy": "stage-entry reduction followed by audited same-beta tightening",
            "penalty_weight": "separate gradual KS objective pressure remains active",
            "aggregation": "ks_max",
        }
    write_json(published / "RAW_ARTIFACT_MANIFEST.json", manifest)
    emit(
        events,
        "ansys_dfm_ld_mma_recovery" if args.recovery_append else "ansys_dfm_ld_mma_start",
        schema=SCHEMA,
        polarization=args.polarization,
        gpu=args.gpu,
        initialization=initialization,
    )

    if args.recovery_append:
        evaluation_counter = max(int(row["evaluation_id"]) for row in history)
        global_evaluation = max(int(row["global_full_physics_evaluation"]) for row in history)
        source_powers = np.asarray([float(row["fixed_source_power_W"]) for row in history])
        fixed_source_power = float(source_powers[0])
        if np.max(np.abs(source_powers - fixed_source_power) / fixed_source_power) >= 0.005:
            raise RuntimeError("recovery history violates fixed-source-power gate")
        completed_stages = len(manifest.get("stages", []))
    else:
        fixed_source_power = None
        evaluation_counter = 0
        global_evaluation = 0
        completed_stages = 0
    final_point = None
    final_beta = 1.0
    forced_cleanup: dict[str, object] | None = None
    if args.start_beta is None:
        start_beta = float(history[-1]["beta"]) if history else 1.0
    else:
        start_beta = float(args.start_beta)
    schedule = beta_sequence(start_beta)
    schedule_index = 0
    first_recovery_stage = bool(args.recovery_append)

    hard_fixed_caps: np.ndarray | None = None
    if args.hard_morphology_constraints:
        hard_contract = manifest.get("hard_morphology_constraints", {})
        recorded_caps = hard_contract.get("fixed_caps")
        recorded_aggregation = hard_contract.get("aggregation")
        if recorded_aggregation != "ks_max":
            recorded_caps = None
        if recorded_caps is None and history and recorded_aggregation == "ks_max":
            for row in history:
                if row.get("morphology_aggregation") != "ks_max":
                    continue
                candidate = np.asarray(row.get("morphology_caps", []), dtype=np.float64)
                if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
                    recorded_caps = candidate.tolist()
                    break
        if recorded_caps is None:
            initial_summary, _ = metrics(
                latent,
                start_beta,
                device=args.constraint_device,
                morphology_aggregation="ks_max",
            )
            initial_residuals = np.asarray(
                [
                    initial_summary["smooth_solid_constraint"],
                    initial_summary["smooth_void_constraint"],
                ],
                dtype=np.float64,
            )
            hard_fixed_caps = np.maximum(
                np.finfo(np.float64).eps,
                (1.0 + args.hard_cap_relative_slack) * initial_residuals,
            )
        else:
            hard_fixed_caps = np.asarray(recorded_caps, dtype=np.float64)
        if hard_fixed_caps.shape != (2,) or not np.all(np.isfinite(hard_fixed_caps)):
            raise RuntimeError("recorded hard morphology caps are invalid")
        if np.any(hard_fixed_caps <= 0.0):
            raise RuntimeError("recorded hard morphology caps must be positive")
        manifest["hard_morphology_constraints"].update(
            {
                "cap_policy": "fixed from the first exact-feasible recovery seed",
                "fixed_caps": hard_fixed_caps.tolist(),
                "rho_init": args.hard_rho_init,
                "aggregation": "ks_max",
            }
        )
        write_json(published / "RAW_ARTIFACT_MANIFEST.json", manifest)

    constraint_caps: np.ndarray | None = None
    constraint_beta: float | None = None
    constraint_exact_entry: int | None = None
    constraint_exact_target: int | None = None
    constraint_stage_attempt = 0

    while True:
        if args.maximum_stages and completed_stages >= args.maximum_stages:
            return 0
        beta = schedule[schedule_index]
        final_beta = beta
        if args.constraint_aware_continuation:
            penalty_weight = constraint_aware_dfm_penalty_weight(beta)
        else:
            penalty_weight = (
                0.0 if args.hard_morphology_constraints else dfm_penalty_weight(beta)
            )
        use_ks = bool(
            args.hard_morphology_constraints or args.constraint_aware_continuation
        )
        stage_initial_summary, _ = metrics(
            latent,
            beta,
            device=args.constraint_device,
            morphology_aggregation="ks_max" if use_ks else "mean",
        )
        stage_initial_residuals = np.asarray(
            [
                stage_initial_summary["smooth_solid_constraint"],
                stage_initial_summary["smooth_void_constraint"],
            ],
            dtype=np.float64,
        )
        controls = stage_controls(beta)
        constraint_hard_active = bool(
            args.constraint_aware_continuation
            and beta >= CONSTRAINT_AWARE_HARD_START_BETA
        )
        if args.constraint_aware_continuation:
            if constraint_beta is None or not np.isclose(
                constraint_beta, beta, rtol=0.0, atol=1.0e-12
            ):
                constraint_beta = float(beta)
                constraint_exact_entry = int(
                    stage_initial_summary["exact"]["total_bad_cell_count"]
                )
                constraint_exact_target = constraint_aware_exact_target(
                    beta, constraint_exact_entry
                )
                constraint_stage_attempt = 0
                constraint_caps = None
            constraint_stage_attempt += 1
            if constraint_hard_active and constraint_caps is None:
                constraint_caps = np.maximum(
                    CONSTRAINT_AWARE_CAP_FLOOR,
                    (1.0 - CONSTRAINT_AWARE_CAP_INITIAL_REDUCTION)
                    * stage_initial_residuals,
                )
            if constraint_hard_active:
                assert constraint_caps is not None
                morphology_caps = constraint_caps.copy()
                morphology_start_beta = 0.0
                hard_constraint_count = 2
                optimizer_rho_init = float(args.hard_rho_init)
                optimizer_always_improve = None
                optimizer_inner_gradients = None
            else:
                morphology_caps = np.asarray([np.inf, np.inf])
                morphology_start_beta = np.inf
                hard_constraint_count = 0
                optimizer_rho_init = float(controls["rho_init"])
                optimizer_always_improve = int(controls["always_improve"])
                optimizer_inner_gradients = int(controls["inner_gradients"])
        elif args.hard_morphology_constraints:
            assert hard_fixed_caps is not None
            morphology_caps = hard_fixed_caps.copy()
            morphology_start_beta = 0.0
            hard_constraint_count = 2
            optimizer_rho_init = float(args.hard_rho_init)
            optimizer_always_improve = None
            optimizer_inner_gradients = None
        else:
            morphology_caps = np.asarray([np.inf, np.inf])
            morphology_start_beta = np.inf
            hard_constraint_count = 0
            optimizer_rho_init = float(controls["rho_init"])
            optimizer_always_improve = int(controls["always_improve"])
            optimizer_inner_gradients = int(controls["inner_gradients"])
        nominal_stage_budget = (
            GRAYSCALE_EVALUATIONS
            if completed_stages == 0 and np.isclose(beta, 1.0)
            else CONTINUATION_EVALUATIONS
        )
        if first_recovery_stage:
            prior_at_beta = sum(
                int(np.isclose(float(row["beta"]), beta, rtol=0.0, atol=1.0e-10))
                for row in history
            )
            stage_budget = max(2, nominal_stage_budget - prior_at_beta)
        else:
            stage_budget = nominal_stage_budget
        evaluator = AdaptiveStageEvaluator(
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
            morphology_caps=morphology_caps,
            fixed_source_power_W=fixed_source_power,
            evaluation_counter=evaluation_counter,
            global_evaluation=global_evaluation,
            constraint_device=args.constraint_device,
            algorithm_label=(
                "NLopt LD_MMA + constraint-aware DFM continuation"
                if args.constraint_aware_continuation
                else (
                    "NLopt LD_MMA + explicit solid/void DFM inequalities"
                    if args.hard_morphology_constraints
                    else "NLopt LD_MMA + Ansys-style DFM penalty"
                )
            ),
            output_slug=args.output_slug if args.recovery_append else "ansys_dfm_ld_mma",
            include_terminal_conductance_constraint=False,
            morphology_start_beta=morphology_start_beta,
            optimizer_controls={
                **controls,
                "beta_factor": BETA_FACTOR,
                "stage_budget": stage_budget,
                "dfm_penalty_weight": penalty_weight,
                "rho_init": optimizer_rho_init,
                "always_improve": optimizer_always_improve,
                "inner_gradients": optimizer_inner_gradients,
                "hard_constraint_ccsa_parameters": (
                    {
                        "rho_init": optimizer_rho_init,
                        "always_improve": "NLopt default",
                        "inner_gradients": "NLopt default",
                    }
                    if args.hard_morphology_constraints or constraint_hard_active
                    else None
                ),
                "constraint_aware_continuation": args.constraint_aware_continuation,
                "constraint_stage_attempt": constraint_stage_attempt,
                "exact_bad_target": constraint_exact_target,
            },
            morphology_penalty_weight=penalty_weight,
            morphology_aggregation="ks_max" if use_ks else "mean",
            constraint_aware_continuation=args.constraint_aware_continuation,
            exact_bad_target=constraint_exact_target,
            require_hard_feasible=constraint_hard_active,
        )
        optimizer = make_optimizer(
            evaluator,
            hard_constraint_count,
            initial_step=float(controls["initial_step"]),
            rho_init=optimizer_rho_init,
            always_improve=optimizer_always_improve,
            inner_gradients=optimizer_inner_gradients,
            xtol_rel=0.0,
            ftol_rel=0.0,
            maxeval=stage_budget,
        )
        emit(
            events,
            "ansys_dfm_stage_start",
            beta=beta,
            stage_index=completed_stages,
            maximum_evaluations=stage_budget,
            dfm_penalty_weight=penalty_weight,
            active_hard_constraints=(
                ["500nm_solid_opening", "500nm_void_opening"]
                if args.hard_morphology_constraints or constraint_hard_active
                else []
            ),
            morphology_caps=morphology_caps.tolist(),
            stage_initial_morphology_residuals=stage_initial_residuals.tolist(),
            constraint_aware_continuation=args.constraint_aware_continuation,
            constraint_stage_attempt=constraint_stage_attempt,
            exact_bad_entry=constraint_exact_entry,
            exact_bad_target=constraint_exact_target,
            manual_move_limit=None,
        )
        stage_stop_reason = "nlopt"
        try:
            optimum = optimizer.optimize(latent.ravel()).reshape(MAPPING.shape)
        except AdaptivePlateauStop:
            if evaluator.last is None:
                raise RuntimeError("adaptive plateau fired before a completed evaluation")
            optimum = evaluator.last.x.copy()
            stage_stop_reason = "adaptive_plateau"
        result_code = optimizer.last_optimize_result()
        if result_code < 0 and stage_stop_reason != "adaptive_plateau":
            raise RuntimeError(f"NLopt LD_MMA failed with result code {result_code}")
        final_point = evaluator.point(optimum.ravel())
        latent = optimum
        fixed_source_power = evaluator.fixed_source_power_W
        evaluation_counter = evaluator.evaluation_counter
        global_evaluation = evaluator.global_evaluation
        completed_stages += 1
        first_recovery_stage = False

        rho = MAPPING.physical(latent, beta)
        gate = final_geometry_gate(rho)
        summary, _ = metrics(
            latent,
            beta,
            device=args.constraint_device,
            morphology_aggregation=(
                "ks_max" if use_ks else "mean"
            ),
        )
        checkpoint = raw_root / f"stage_{completed_stages:04d}_beta{beta:g}.npz"
        np.savez_compressed(checkpoint, latent=latent, rho=rho, beta=np.asarray(beta))
        stage_record = {
            "stage_index": completed_stages,
            "beta": beta,
            "dfm_penalty_weight": penalty_weight,
            "hard_morphology_constraints": args.hard_morphology_constraints,
            "constraint_aware_continuation": args.constraint_aware_continuation,
            "constraint_stage_attempt": constraint_stage_attempt,
            "exact_bad_entry": constraint_exact_entry,
            "exact_bad_target": constraint_exact_target,
            "morphology_caps": morphology_caps.tolist(),
            "stage_initial_morphology_residuals": stage_initial_residuals.tolist(),
            "nlopt_result_code": int(result_code),
            "stage_stop_reason": stage_stop_reason,
            "adaptive_plateau_diagnostic": evaluator.plateau_diagnostic,
            "full_physics_evaluations": evaluator.stage_full_physics_evaluations,
            "raw_current_at_reference_power_A": history[-1][
                "objective_at_reference_power_A"
            ],
            "smooth_solid_residual": summary["smooth_solid_constraint"],
            "smooth_void_residual": summary["smooth_void_constraint"],
            "final_geometry_gate": gate,
            "checkpoint": {
                "path": str(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
            },
        }
        manifest.setdefault("stages", []).append(stage_record)
        write_json(published / "RAW_ARTIFACT_MANIFEST.json", manifest)
        write_json(published / "latest_stage.json", stage_record)
        emit(events, "ansys_dfm_stage_complete", **stage_record)
        promotion = evaluator.plateau_diagnostic or constraint_aware_promotion_diagnostic(
            history[-evaluator.stage_full_physics_evaluations :],
            beta=beta,
            exact_bad_target=constraint_exact_target,
            require_hard_feasible=constraint_hard_active,
        )
        if gate["passed"] and (
            not args.constraint_aware_continuation
            or bool(promotion.get("promotion_passed"))
        ):
            break
        plateau = promotion if args.constraint_aware_continuation else (
            evaluator.plateau_diagnostic or {}
        )
        exact_bad = int(gate["exact_500nm_audit"]["total_bad_cell_count"])
        cleanup_ready = bool(
            not args.constraint_aware_continuation
            and
            beta >= EXACT_CLEANUP_MINIMUM_BETA
            and float(gate["gray_fraction_0p01_0p99"]) < EXACT_CLEANUP_MAXIMUM_GRAY_FRACTION
            and exact_bad > 0
            and bool(plateau.get("passed"))
            and bool(plateau.get("constraint_plateau"))
        )
        if cleanup_ready:
            emit(
                events,
                "forced_exact_cleanup_start",
                beta=beta,
                exact_bad_cells=exact_bad,
                plateau_diagnostic=plateau,
            )
            cleanup_attempt = evaluate_exact_cleanup_candidates(
                rho,
                raw_root=raw_root,
                base_fsp=base_fsp,
                base_sha256=args.base_sha256,
                polarization=args.polarization,
                gpu=args.gpu,
                reference_objective_A=float(final_point.result["objective_A"]),
                attempt_label=f"stage{completed_stages:04d}_beta{beta:g}",
            )
            manifest.setdefault("forced_exact_cleanup_attempts", []).append(
                cleanup_attempt
            )
            write_json(published / "RAW_ARTIFACT_MANIFEST.json", manifest)
            write_json(
                published
                / f"FORCED_EXACT_CLEANUP_STAGE_{completed_stages:04d}.json",
                cleanup_attempt,
            )
            if cleanup_objective_preserved(cleanup_attempt):
                forced_cleanup = cleanup_attempt
                manifest["forced_exact_cleanup"] = forced_cleanup
                write_json(published / "RAW_ARTIFACT_MANIFEST.json", manifest)
                write_json(published / "FORCED_EXACT_CLEANUP.json", forced_cleanup)
                emit(
                    events,
                    "forced_exact_cleanup_complete",
                    selected=forced_cleanup["selected"],
                    objective_preservation_passed=True,
                )
                break
            emit(
                events,
                "forced_exact_cleanup_rejected",
                selected=cleanup_attempt["selected"],
                objective_preservation_passed=False,
                action="continue_beta_continuation",
            )
        if args.constraint_aware_continuation:
            promotion_passed = bool(promotion.get("promotion_passed"))
            if promotion_passed and schedule_index < len(schedule) - 1:
                schedule_index += 1
                constraint_beta = None
                constraint_caps = None
                constraint_exact_entry = None
                constraint_exact_target = None
                constraint_stage_attempt = 0
            elif constraint_hard_active:
                latest_maximum_constraint = history[-1].get("maximum_constraint_value")
                latest_feasible = bool(
                    latest_maximum_constraint is not None
                    and float(latest_maximum_constraint) <= 0.0
                )
                exact_missed = bool(
                    constraint_exact_target is not None
                    and exact_bad > constraint_exact_target
                )
                if latest_feasible and exact_missed:
                    assert constraint_caps is not None
                    constraint_caps = np.maximum(
                        CONSTRAINT_AWARE_CAP_FLOOR,
                        (1.0 - CONSTRAINT_AWARE_CAP_REPEAT_REDUCTION)
                        * constraint_caps,
                    )
                    emit(
                        events,
                        "constraint_aware_caps_tightened",
                        beta=beta,
                        stage_attempt=constraint_stage_attempt,
                        exact_bad_cells=exact_bad,
                        exact_bad_target=constraint_exact_target,
                        new_caps=constraint_caps.tolist(),
                    )
        elif schedule_index < len(schedule) - 1:
            schedule_index += 1
        # At maximum beta, continue fixed-budget LD_MMA stages with the
        # saturated official DFM penalty until the explicit final gate passes.

    assert final_point is not None
    final_rho = MAPPING.physical(latent, final_beta)
    if forced_cleanup is not None:
        selected = str(forced_cleanup["selected"])
        selected_row = forced_cleanup["candidates"][selected]
        with np.load(selected_row["density"]["path"]) as loaded:
            binary = np.asarray(loaded["rho"], dtype=np.float64)
        binary_result = selected_row["result"]
    else:
        binary = (final_rho >= 0.5).astype(np.float64)
        binary_result = None
    exact, _ = exact_binary_audit(binary)
    if exact["total_bad_cell_count"] != 0:
        raise RuntimeError("internal error: final thresholded density failed exact 500nm gate")
    binary_path = raw_root / "final_exact_binary_density.npz"
    np.savez_compressed(binary_path, rho=binary)
    if binary_result is None:
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
        result_path = final_output / "binary_objective_result.json"
        if completed.returncode or not result_path.is_file():
            raise RuntimeError("fresh exact-binary evaluation failed")
        binary_result = json.loads(result_path.read_text())
        if not binary_result.get("passed"):
            raise RuntimeError("fresh exact-binary result is not passed")
    objective_preservation_passed = bool(binary_result.get("passed"))
    final = {
        "schema": SCHEMA,
        "passed": objective_preservation_passed,
        "status": (
            "VALIDATED_ANSYS_STYLE_DFM_LD_MMA_EXACT_BINARY_PTE_OPTIMIZATION"
            if objective_preservation_passed
            else "COMPLETED_EXACT_BINARY_WITH_OBJECTIVE_PRESERVATION_GATE_FAILED"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "polarization": args.polarization,
        "algorithm": "NLopt LD_MMA",
        "objective": "signed full-flake terminal PTE current",
        "reference_incident_power_W": REFERENCE_INCIDENT_POWER_W,
        "final_beta": final_beta,
        "full_physics_evaluations": global_evaluation,
        "completed_stages": completed_stages,
        "final_geometry_gate": final_geometry_gate(binary),
        "binary_result": binary_result,
        "manual_move_limit": None,
        "connectivity_constraint": False,
        "symmetry_constraint": False,
        "volume_constraint": False,
        "posthoc_morphology_repair": forced_cleanup is not None,
        "forced_exact_cleanup": forced_cleanup,
    }
    write_json(published / "FINAL_RESULT.json", final)
    emit(events, "ansys_dfm_optimization_complete", **final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
