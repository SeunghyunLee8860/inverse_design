from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import OptimizeResult, minimize

from .robin import HardEvaluation
from .scaled import SignedBranchEvaluation, SignedBranchObjective


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    x_scaled: np.ndarray
    smooth_current_A: float
    minimization_objective: float
    minimum_constraint: float


@dataclass(frozen=True)
class SignedSLSQPResult:
    branch_sign: int
    start_scaled: np.ndarray
    scipy_result: OptimizeResult
    smooth: SignedBranchEvaluation
    hard: HardEvaluation
    constraints: np.ndarray
    history: tuple[IterationRecord, ...]
    unique_forward_adjoint_evaluations: int


class _CachedEvaluator:
    """Share one PDE/adjoint solve between SLSQP's fun, jac, and callback."""

    def __init__(self, objective: SignedBranchObjective, branch_sign: int):
        self.objective = objective
        self.branch_sign = branch_sign
        self._x: np.ndarray | None = None
        self._value: SignedBranchEvaluation | None = None
        self.evaluation_count = 0

    def __call__(self, x: np.ndarray) -> SignedBranchEvaluation:
        values = np.asarray(x, dtype=float)
        if self._x is None or not np.array_equal(values, self._x):
            self._value = self.objective.evaluate(values, branch_sign=self.branch_sign)
            self._x = values.copy()
            self.evaluation_count += 1
        assert self._value is not None
        return self._value


def run_signed_slsqp(
    objective: SignedBranchObjective,
    start_scaled: np.ndarray,
    *,
    branch_sign: int,
    minimum_length_m: float,
    maximum_length_m: float,
    minimum_gap_m: float,
    max_iterations: int = 250,
    function_tolerance: float = 1.0e-11,
) -> SignedSLSQPResult:
    """Optimize one signed branch from one feasible dimensionless start."""

    x0 = np.asarray(start_scaled, dtype=float)
    if x0.shape != (4,):
        raise ValueError("start_scaled must have shape (4,)")
    if branch_sign not in (-1, +1):
        raise ValueError("branch_sign must be +1 or -1")

    perimeter = objective.model.perimeter
    gap_fraction = minimum_gap_m / perimeter.perimeter_m
    initial_constraints, _ = perimeter.separation_constraints_scaled(
        x0, gap_fraction
    )
    if np.min(initial_constraints) < -1.0e-12:
        raise ValueError(
            f"SLSQP start is infeasible; constraints={initial_constraints.tolist()}"
        )

    cached = _CachedEvaluator(objective, branch_sign)
    history: list[IterationRecord] = []

    def fun(x: np.ndarray) -> float:
        return cached(x).minimization_objective

    def jac(x: np.ndarray) -> np.ndarray:
        return cached(x).minimization_gradient_scaled

    def constraint_fun(x: np.ndarray) -> np.ndarray:
        return perimeter.separation_constraints_scaled(x, gap_fraction)[0]

    def constraint_jac(x: np.ndarray) -> np.ndarray:
        return perimeter.separation_constraints_scaled(x, gap_fraction)[1]

    def callback(x: np.ndarray) -> None:
        evaluation = cached(x)
        constraints = constraint_fun(x)
        history.append(
            IterationRecord(
                iteration=len(history) + 1,
                x_scaled=np.asarray(x, dtype=float).copy(),
                smooth_current_A=evaluation.current_A,
                minimization_objective=evaluation.minimization_objective,
                minimum_constraint=float(np.min(constraints)),
            )
        )

    result = minimize(
        fun,
        x0,
        method="SLSQP",
        jac=jac,
        bounds=objective.length_bounds(minimum_length_m, maximum_length_m),
        constraints={
            "type": "ineq",
            "fun": constraint_fun,
            "jac": constraint_jac,
        },
        callback=callback,
        options={
            "maxiter": int(max_iterations),
            "ftol": float(function_tolerance),
            "disp": False,
        },
    )
    smooth = cached(np.asarray(result.x, dtype=float))
    constraints = constraint_fun(result.x)
    canonical_physical = smooth.canonical_design.to_physical(
        perimeter.perimeter_m
    )
    hard = objective.model.hard_evaluate(canonical_physical)
    return SignedSLSQPResult(
        branch_sign=branch_sign,
        start_scaled=x0.copy(),
        scipy_result=result,
        smooth=smooth,
        hard=hard,
        constraints=constraints,
        history=tuple(history),
        unique_forward_adjoint_evaluations=cached.evaluation_count,
    )
