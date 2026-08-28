"""Production beta continuation and Au-specific fabrication constraints.

The flake-topology reference run used three design inequalities at high beta:
terminal conductance plus solid/void minimum-feature constraints.  The Au in
this run is a floating optical/thermal structure, not a measurement terminal,
so a terminal-conductance floor would be unphysical.  Its three high-beta
design inequalities are instead minimum Au feature, minimum void/spacing, and
an explicit grayness cap.

The two epigraph inequalities that define the signed dual-polarization current
objective are always present and are not counted as fabrication constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import optimize as scipy_optimize

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    OPTIMIZER_250NM_MAPPING,
    calibrated_lumerical_250nm_dfm_caps,
    smooth_lumerical_250nm_constraints,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_optimizer import (
    CURRENT_SCALE_A,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_signed_objective import (
    signed_dual_objective_point,
)


BETA_SCHEDULE = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)

# One MMA object owns one fixed-beta subproblem for its complete lifetime.
# These are high emergency ceilings, not routine chunks: normal termination is
# requested from the physics callback only after the feasible signed objective
# reaches the audited plateau.  A new same-beta MMA is permitted only as crash
# recovery because NLopt does not expose serializable MMA asymptotes.
STAGE_MAXEVAL = {
    1.0: 64,
    2.0: 48,
    4.0: 48,
    8.0: 48,
    16.0: 64,
    32.0: 64,
    64.0: 80,
    128.0: 96,
}
MINIMUM_CONTINUATION_EVALUATIONS = sum(STAGE_MAXEVAL[beta] for beta in BETA_SCHEDULE)
# These values are MMA initial step sizes, not permanent bounds around the
# stage starting point.  The old fixed box made the beta-16 grayness cap
# mathematically unreachable even after every allowed retry.
MMA_INITIAL_STEP = {
    1.0: 0.025,
    2.0: 0.020,
    4.0: 0.018,
    8.0: 0.015,
    16.0: 0.012,
    32.0: 0.010,
    64.0: 0.008,
    128.0: 0.006,
}

# The caps are applied only once their corresponding constraint is active.
# Intermediate DFM caps are fixed from the reprojected stage baseline and may
# never relax a previously active cap.  Beta=128 uses the independently
# calibrated exact-pass binary-reference caps.
DFM_BASELINE_REDUCTION = {
    4.0: (0.92, np.inf),
    8.0: (0.85, 0.98),
    16.0: (0.72, 0.88),
    32.0: (0.52, 0.70),
    64.0: (0.30, 0.45),
    128.0: (0.0, 0.0),
}
GRAYNESS_TARGET_CAP = {
    16.0: 0.60,
    32.0: 0.20,
    64.0: 0.040,
    128.0: 0.005,
}
# An active cap is the tighter of the previous cap and a controlled reduction
# from the reprojected stage baseline, but never tighter than that beta's
# target on its first attempt.  This is constraint continuation: it prevents
# the optimizer from being handed a grossly infeasible subproblem while still
# ending at the unchanged beta-128 binary gate.
GRAYNESS_BASELINE_REDUCTION = {
    16.0: 0.90,
    32.0: 0.70,
    64.0: 0.45,
    128.0: 0.25,
}
FINAL_GRAYNESS_CAP = GRAYNESS_TARGET_CAP[BETA_SCHEDULE[-1]]

DESIGN_CONSTRAINT_TOLERANCE = 1.0e-3
EPIGRAPH_CONSTRAINT_TOLERANCE = 1.0e-6
STAGE_FTOL_REL = 0.0
STAGE_XTOL_REL = 0.0
INITIAL_MAXIMIN_WARM_MAXIMUM_CHANGE = 0.050
STAGE_PLATEAU_MINIMUM_FEASIBLE_POINTS = 6
STAGE_PLATEAU_WINDOW = 4
STAGE_PLATEAU_RELATIVE_TOLERANCE = 1.0e-2
STAGE_PLATEAU_ABSOLUTE_TOLERANCE_NA = 1.0e-3


def linearized_maximin_box_warm_start(
    *,
    latent: np.ndarray,
    current_a_A: float,
    current_b_A: float,
    gradient_a_latent_A: np.ndarray,
    gradient_b_latent_A: np.ndarray,
    maximum_change: float,
) -> dict[str, Any]:
    """Solve the exact first-order two-utility box trust-region problem.

    At exact uniform rho=0.5 both PTE currents should vanish by symmetry.  The
    residual current difference is then at the Maxwell/PDE numerical floor.
    A raw epigraph MMA step spends expensive evaluations balancing that tiny
    offset instead of breaking symmetry.  This helper maximizes the
    linearized ``min(I_Ea, -I_Eb)`` over a bounded latent-density step.

    The two-constraint box problem has a scalar convex dual.  For an Ea dual
    weight ``w``, the box support direction is selected from
    ``w*grad(I_Ea) + (1-w)*grad(-I_Eb)``.  Minimizing that dual on ``[0,1]``
    gives the exact max-min linearized step without inventing a polarization
    weighting or changing the production objective.
    """

    value = np.asarray(latent, dtype=np.float64)
    gradient_a = np.asarray(gradient_a_latent_A, dtype=np.float64)
    gradient_b_utility = -np.asarray(gradient_b_latent_A, dtype=np.float64)
    if (
        value.shape != CONTRACT.design_node_shape
        or gradient_a.shape != value.shape
        or gradient_b_utility.shape != value.shape
    ):
        raise ValueError("warm-start latent and gradients must match the design shape")
    if not (
        np.all(np.isfinite(value))
        and np.all(np.isfinite(gradient_a))
        and np.all(np.isfinite(gradient_b_utility))
    ):
        raise ValueError("warm-start latent and gradients must be finite")
    step = float(maximum_change)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("maximum_change must be finite and positive")
    if np.min(value) < 0.0 or np.max(value) > 1.0:
        raise ValueError("warm-start latent must lie inside [0,1]")
    utility_a = float(current_a_A)
    utility_b = -float(current_b_A)
    if not np.all(np.isfinite((utility_a, utility_b))):
        raise ValueError("warm-start currents must be finite")

    lower = np.maximum(-step, -value)
    upper = np.minimum(step, 1.0 - value)

    def dual(weight_a: float) -> float:
        weight = float(weight_a)
        combined = weight * gradient_a + (1.0 - weight) * gradient_b_utility
        support = np.sum(np.maximum(combined * lower, combined * upper))
        return float(weight * utility_a + (1.0 - weight) * utility_b + support)

    minimized = scipy_optimize.minimize_scalar(
        dual,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1.0e-12, "maxiter": 200},
    )
    candidates = (0.0, 1.0, float(minimized.x))
    weight_a = min(candidates, key=dual)
    combined = weight_a * gradient_a + (1.0 - weight_a) * gradient_b_utility
    delta = np.where(combined > 0.0, upper, np.where(combined < 0.0, lower, 0.0))
    warm_latent = value + delta
    predicted_a = utility_a + float(np.sum(gradient_a * delta))
    predicted_b = utility_b + float(np.sum(gradient_b_utility * delta))
    initial_balanced = min(utility_a, utility_b)
    predicted_balanced = min(predicted_a, predicted_b)
    if predicted_balanced <= initial_balanced:
        raise RuntimeError(
            "linearized max-min warm start did not improve the objective"
        )
    return {
        "latent": warm_latent,
        "delta": delta,
        "dual_weight_Ea": float(weight_a),
        "initial_utilities_A": {"Ea": utility_a, "Eb": utility_b},
        "predicted_utilities_A": {"Ea": predicted_a, "Eb": predicted_b},
        "initial_balanced_utility_A": initial_balanced,
        "predicted_balanced_utility_A": predicted_balanced,
        "predicted_improvement_A": predicted_balanced - initial_balanced,
        "maximum_allowed_change": step,
        "maximum_abs_change": float(np.max(np.abs(delta))),
        "delta_L2": float(np.linalg.norm(delta)),
        "lower_bound_active_count": int(np.count_nonzero(delta == lower)),
        "upper_bound_active_count": int(np.count_nonzero(delta == upper)),
        "method": "exact_linearized_two_utility_box_dual_v1",
    }


def active_design_constraint_names(beta: float) -> tuple[str, ...]:
    """Return the gradual 0 -> 1 -> 2 -> 3 fabrication schedule."""

    value = float(beta)
    if value < 4.0:
        return ()
    if value < 8.0:
        return ("minimum_250nm_Au_feature",)
    if value < 16.0:
        return (
            "minimum_250nm_Au_feature",
            "minimum_250nm_void_spacing",
        )
    return (
        "minimum_250nm_Au_feature",
        "minimum_250nm_void_spacing",
        "grayness_mean_4rho1mrho",
    )


def grayness_value_gradient(
    latent: np.ndarray, beta: float
) -> tuple[float, np.ndarray]:
    """Return mean 4*rho*(1-rho) and its exact latent pullback."""

    value = np.asarray(latent, dtype=np.float64)
    rho = OPTIMIZER_250NM_MAPPING.physical(value, float(beta))
    grayness = float(np.mean(4.0 * rho * (1.0 - rho)))
    gradient_projected = (4.0 - 8.0 * rho) / rho.size
    gradient_latent = OPTIMIZER_250NM_MAPPING.vjp(
        value, gradient_projected, float(beta)
    )
    return grayness, gradient_latent


def stage_design_caps(
    *,
    beta: float,
    baseline_dfm_values: np.ndarray,
    baseline_grayness: float,
    previous_dfm_caps: np.ndarray | None,
    previous_grayness_cap: float | None,
) -> dict[str, Any]:
    """Fix monotone, locally staged fabrication caps for one beta stage."""

    beta_value = float(beta)
    baseline = np.asarray(baseline_dfm_values, dtype=np.float64)
    if baseline.shape != (2,) or not np.all(np.isfinite(baseline)):
        raise ValueError("baseline DFM values must be a finite length-two vector")
    calibrated, calibration = calibrated_lumerical_250nm_dfm_caps()
    prior = (
        np.full(2, np.inf, dtype=np.float64)
        if previous_dfm_caps is None
        else np.asarray(previous_dfm_caps, dtype=np.float64)
    )
    if prior.shape != (2,) or np.any(np.isnan(prior)) or np.any(prior <= 0.0):
        raise ValueError("previous DFM caps must be positive length-two values")
    gray_baseline = float(baseline_grayness)
    if not np.isfinite(gray_baseline) or not 0.0 <= gray_baseline <= 1.0:
        raise ValueError("baseline grayness must be finite inside [0,1]")
    prior_gray = (
        np.inf if previous_grayness_cap is None else float(previous_grayness_cap)
    )
    if np.isnan(prior_gray) or prior_gray <= 0.0:
        raise ValueError("previous grayness cap must be positive")
    active = active_design_constraint_names(beta_value)
    caps = prior.copy()
    if active:
        factors = np.asarray(DFM_BASELINE_REDUCTION[beta_value], dtype=np.float64)
        proposed = np.maximum(calibrated, factors * baseline)
        caps[0] = min(caps[0], proposed[0])
        if len(active) >= 2:
            caps[1] = min(caps[1], proposed[1])
    if len(active) < 3:
        gray_cap = np.inf
    else:
        target = GRAYNESS_TARGET_CAP[beta_value]
        # Beta=128 is already a new fixed subproblem. Giving it the immutable
        # final cap at entry avoids changing constraints and resetting MMA
        # several times inside the same beta.
        proposed_gray = (
            target
            if beta_value == BETA_SCHEDULE[-1]
            else max(
                target,
                GRAYNESS_BASELINE_REDUCTION[beta_value] * gray_baseline,
            )
        )
        gray_cap = min(prior_gray, proposed_gray)
    return {
        "beta": beta_value,
        "active_names": list(active),
        "DFM_caps": caps,
        "grayness_cap": float(gray_cap),
        "baseline_grayness": gray_baseline,
        "grayness_target_cap": float(GRAYNESS_TARGET_CAP.get(beta_value, np.inf)),
        "grayness_baseline_reduction": float(
            GRAYNESS_BASELINE_REDUCTION.get(beta_value, np.nan)
        ),
        "calibrated_final_DFM_caps": calibrated,
        "DFM_calibration": calibration,
    }


def stage_objective_progress(
    callback_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Audit whether feasible balanced utility stopped improving recently."""

    feasible = [
        row
        for row in callback_history
        if float(row.get("maximum_design_constraint", -np.inf))
        <= DESIGN_CONSTRAINT_TOLERANCE
    ]
    values = np.asarray(
        [float(row["balanced_utility_nA"]) for row in feasible],
        dtype=np.float64,
    )
    enough = values.size >= STAGE_PLATEAU_MINIMUM_FEASIBLE_POINTS
    has_prior_window = values.size > STAGE_PLATEAU_WINDOW
    if has_prior_window:
        prior_best = float(np.max(values[:-STAGE_PLATEAU_WINDOW]))
        recent_best = float(np.max(values[-STAGE_PLATEAU_WINDOW:]))
        improvement = recent_best - prior_best
        tolerance = max(
            STAGE_PLATEAU_ABSOLUTE_TOLERANCE_NA,
            STAGE_PLATEAU_RELATIVE_TOLERANCE
            * max(abs(recent_best), STAGE_PLATEAU_ABSOLUTE_TOLERANCE_NA),
        )
    else:
        prior_best = np.nan
        recent_best = float(np.max(values)) if values.size else np.nan
        improvement = np.inf
        tolerance = np.nan
    return {
        "converged": bool(enough and has_prior_window and improvement <= tolerance),
        "feasible_physics_points": int(values.size),
        "minimum_feasible_points": STAGE_PLATEAU_MINIMUM_FEASIBLE_POINTS,
        "window": STAGE_PLATEAU_WINDOW,
        "prior_best_balanced_utility_nA": prior_best,
        "recent_best_balanced_utility_nA": recent_best,
        "recent_best_improvement_nA": float(improvement),
        "allowed_improvement_nA": float(tolerance),
    }


def design_constraint_point(
    latent: np.ndarray,
    *,
    beta: float,
    dfm_caps: np.ndarray,
    grayness_cap: float,
) -> dict[str, Any]:
    """Evaluate normalized g(x)<=0 fabrication inequalities."""

    active = active_design_constraint_names(beta)
    dfm_values, dfm_gradients, _ = smooth_lumerical_250nm_constraints(latent, beta)
    grayness, grayness_gradient = grayness_value_gradient(latent, beta)
    caps = np.asarray(dfm_caps, dtype=np.float64)
    values: list[float] = []
    gradients: list[np.ndarray] = []
    if len(active) >= 1:
        values.append(float(dfm_values[0] / caps[0] - 1.0))
        gradients.append(dfm_gradients[0] / caps[0])
    if len(active) >= 2:
        values.append(float(dfm_values[1] / caps[1] - 1.0))
        gradients.append(dfm_gradients[1] / caps[1])
    if len(active) >= 3:
        if not np.isfinite(grayness_cap) or grayness_cap <= 0.0:
            raise ValueError("active grayness cap must be finite and positive")
        values.append(float(grayness / grayness_cap - 1.0))
        gradients.append(grayness_gradient / grayness_cap)
    return {
        "names": list(active),
        "normalized_values": np.asarray(values, dtype=np.float64),
        "normalized_gradients": (
            np.stack(gradients)
            if gradients
            else np.empty((0, *CONTRACT.design_node_shape), dtype=np.float64)
        ),
        "raw_DFM_values": dfm_values,
        "DFM_caps": caps,
        "grayness": grayness,
        "grayness_cap": float(grayness_cap),
    }


@dataclass
class ContinuationEpigraphProblem:
    """NLopt callback surface for one fixed-beta production stage."""

    evaluate_physics: Callable[[np.ndarray], dict[str, Any]]
    beta: float
    dfm_caps: np.ndarray
    grayness_cap: float
    history_prefix: list[dict[str, Any]] | None = None
    progress_callback: Callable[["ContinuationEpigraphProblem"], None] | None = None

    def __post_init__(self) -> None:
        self.beta = float(self.beta)
        self.dfm_caps = np.asarray(self.dfm_caps, dtype=np.float64)
        self.callback_history: list[dict[str, Any]] = []
        self.history_prefix = [dict(row) for row in (self.history_prefix or [])]
        self._candidate_latents: list[np.ndarray] = []
        self._candidate_points: list[dict[str, Any]] = []
        self._last_latent: np.ndarray | None = None
        self._last_point: dict[str, Any] | None = None
        self._force_stop: Callable[[], None] | None = None
        self.plateau_stop_requested = False
        self.plateau_result: dict[str, Any] | None = None

    @property
    def complete_callback_history(self) -> list[dict[str, Any]]:
        return [*self.history_prefix, *self.callback_history]

    def bind_force_stop(self, callback: Callable[[], None]) -> None:
        """Bind the sole NLopt lifetime stop used by the physics plateau gate."""

        self._force_stop = callback

    @property
    def variable_count(self) -> int:
        return int(np.prod(CONTRACT.design_node_shape)) + 1

    @property
    def design_constraint_count(self) -> int:
        return len(active_design_constraint_names(self.beta))

    @property
    def total_constraint_count(self) -> int:
        return 2 + self.design_constraint_count

    def point(self, vector: np.ndarray) -> dict[str, Any]:
        value = np.asarray(vector, dtype=np.float64)
        if value.shape != (self.variable_count,):
            raise ValueError("optimizer vector has the wrong shape")
        latent = value[:-1].reshape(CONTRACT.design_node_shape)
        if self._last_latent is not None and np.array_equal(latent, self._last_latent):
            assert self._last_point is not None
            return {**self._last_point, "epigraph_nA": float(value[-1])}
        evaluated = self.evaluate_physics(latent)
        signed = signed_dual_objective_point(
            latent=latent,
            beta=self.beta,
            current_a_A=float(evaluated["currents_A"]["Ea"]),
            current_b_A=float(evaluated["currents_A"]["Eb"]),
            gradient_a_projected_A=evaluated["gradient_Ea_projected_A"],
            gradient_b_projected_A=evaluated["gradient_Eb_projected_A"],
            epigraph_A=float(value[-1]) * CURRENT_SCALE_A,
            mapping=OPTIMIZER_250NM_MAPPING,
        )
        design = design_constraint_point(
            latent,
            beta=self.beta,
            dfm_caps=self.dfm_caps,
            grayness_cap=self.grayness_cap,
        )
        point = {
            **evaluated,
            **signed,
            "latent": latent,
            "epigraph_nA": float(value[-1]),
            "design_constraints": design,
        }
        self._last_latent = latent.copy()
        self._last_point = point
        normalized_design = np.asarray(design["normalized_values"], dtype=np.float64)
        maximum_design = (
            float(np.max(normalized_design)) if normalized_design.size else -np.inf
        )
        self.callback_history.append(
            {
                "callback_index": len(self.history_prefix) + len(self.callback_history),
                "current_Ea_nA": 1.0e9 * float(point["current_a_A"]),
                "current_Eb_nA": 1.0e9 * float(point["current_b_A"]),
                "balanced_utility_nA": 1.0e9 * float(point["balanced_utility_A"]),
                "design_constraint_names": design["names"],
                "design_constraint_values": design["normalized_values"].tolist(),
                "raw_DFM_values": design["raw_DFM_values"].tolist(),
                "grayness": design["grayness"],
                "maximum_design_constraint": maximum_design,
                "design_feasible": bool(maximum_design <= DESIGN_CONSTRAINT_TOLERANCE),
            }
        )
        self._candidate_latents.append(latent.copy())
        self._candidate_points.append(point)
        if self.progress_callback is not None:
            self.progress_callback(self)
        progress = stage_objective_progress(self.complete_callback_history)
        signs_pass = bool(
            float(point["current_a_A"]) > 0.0 and float(point["current_b_A"]) < 0.0
        )
        if (
            progress["converged"]
            and maximum_design <= DESIGN_CONSTRAINT_TOLERANCE
            and signs_pass
            and not self.plateau_stop_requested
        ):
            self.plateau_result = progress
            self.plateau_stop_requested = True
            if self._force_stop is None:
                raise RuntimeError(
                    "physics plateau reached before NLopt stop was bound"
                )
            self._force_stop()
        return point

    def selected_candidate(self) -> dict[str, Any]:
        """Return the best feasible point, or least-violating point if needed."""

        if not self.callback_history:
            raise RuntimeError("cannot select a continuation candidate without points")
        feasible = [
            index
            for index, row in enumerate(self.callback_history)
            if bool(row["design_feasible"])
        ]
        if feasible:
            index = max(
                feasible,
                key=lambda candidate: float(
                    self.callback_history[candidate]["balanced_utility_nA"]
                ),
            )
            reason = "maximum_balanced_utility_among_design_feasible_points"
        else:
            index = min(
                range(len(self.callback_history)),
                key=lambda candidate: (
                    max(
                        0.0,
                        float(
                            self.callback_history[candidate][
                                "maximum_design_constraint"
                            ]
                        ),
                    ),
                    -float(self.callback_history[candidate]["balanced_utility_nA"]),
                ),
            )
            reason = "minimum_design_violation_then_maximum_balanced_utility"
        return {
            "callback_index": int(self.callback_history[index]["callback_index"]),
            "reason": reason,
            "latent": self._candidate_latents[index].copy(),
            "point": self._candidate_points[index],
            "audit": dict(self.callback_history[index]),
        }

    def objective(self, vector: np.ndarray, gradient: np.ndarray) -> float:
        if gradient.size:
            gradient[:] = 0.0
            gradient[-1] = 1.0
        return float(vector[-1])

    def constraints(
        self, result: np.ndarray, vector: np.ndarray, gradient: np.ndarray
    ) -> None:
        point = self.point(vector)
        epigraph = (
            np.asarray(point["epigraph_constraints_A"], dtype=np.float64)
            / CURRENT_SCALE_A
        )
        design = point["design_constraints"]
        result[:] = np.concatenate((epigraph, design["normalized_values"]))
        if gradient.size:
            gradient[:] = 0.0
            gradient[:2, :-1] = (
                np.asarray(point["constraint_gradients_latent_A"], dtype=np.float64)
                / CURRENT_SCALE_A
            ).reshape(2, -1)
            gradient[:2, -1] = 1.0
            if self.design_constraint_count:
                gradient[2:, :-1] = np.asarray(
                    design["normalized_gradients"], dtype=np.float64
                ).reshape(self.design_constraint_count, -1)


def continuation_contract() -> dict[str, Any]:
    """Return a serializable audit of the production continuation policy."""

    return {
        "beta_schedule": list(BETA_SCHEDULE),
        "stage_safety_maxeval": {
            str(key): value for key, value in STAGE_MAXEVAL.items()
        },
        "MMA_lifecycle": {
            "normal": "exactly one MMA object for each fixed beta",
            "normal_stop": "callback-requested stop after audited physics plateau",
            "same_beta_new_MMA": (
                "crash recovery only; NLopt MMA internal asymptotes are not serializable"
            ),
            "beta_change_new_MMA": True,
        },
        "continuation_evaluation_budget": {
            "all_stage_emergency_ceiling": MINIMUM_CONTINUATION_EVALUATIONS,
            "counting_rule": (
                "each beta lifetime includes its starting physics point; the initial "
                "maximin warm start remains inside the beta-1 emergency ceiling"
            ),
        },
        "latent_bounds": [0.0, 1.0],
        "stage_start_is_not_a_permanent_move_box": True,
        "MMA_initial_step": {
            str(key): value for key, value in MMA_INITIAL_STEP.items()
        },
        "design_constraint_activation": {
            str(beta): list(active_design_constraint_names(beta))
            for beta in BETA_SCHEDULE
        },
        "grayness_target_caps": {
            str(key): value for key, value in GRAYNESS_TARGET_CAP.items()
        },
        "grayness_baseline_reduction": {
            str(key): value for key, value in GRAYNESS_BASELINE_REDUCTION.items()
        },
        "final_grayness_gate": FINAL_GRAYNESS_CAP,
        "stage_ftol_rel": STAGE_FTOL_REL,
        "stage_xtol_rel": STAGE_XTOL_REL,
        "initial_maximin_warm_maximum_change": (INITIAL_MAXIMIN_WARM_MAXIMUM_CHANGE),
        "external_physics_objective_plateau_gate": {
            "minimum_feasible_points": STAGE_PLATEAU_MINIMUM_FEASIBLE_POINTS,
            "window": STAGE_PLATEAU_WINDOW,
            "relative_tolerance": STAGE_PLATEAU_RELATIVE_TOLERANCE,
            "absolute_tolerance_nA": STAGE_PLATEAU_ABSOLUTE_TOLERANCE_NA,
        },
        "initial_density": "exact uniform latent rho=0.5",
        "floating_Au_terminal_conductance_constraint": False,
        "objective": "maximize min(I_Ea, -I_Eb) by epigraph LD_MMA",
        "final_promotion_requires": [
            "I_Ea > 0 and I_Eb < 0",
            "continuous grayness <= final cap",
            "thresholded cell mask exact 250 nm solid/void audit",
            "fresh ordinary-dispersive exact-binary Au Maxwell forwards for Ea/Eb",
            "Ea and Eb 100-to-50-nm optical lateral comparisons below 0.5%",
            "100-nm and 50-nm optical raw Q through adaptive custom-CUDA PDE convergence",
            "same-PDE-grid 100-to-50-nm current and temperature changes below 0.5%",
            "fine-reference exact-binary I_Ea > 0 and I_Eb < 0",
        ],
        "posthoc_morphology_repair": False,
    }
