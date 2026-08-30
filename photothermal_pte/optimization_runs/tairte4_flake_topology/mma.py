"""Persistent method-of-moving-asymptotes subproblems.

This is the optimizer used by the fresh contact-anchored runs.  It is kept
separate from the historical Run014 driver so that an Adam update cannot be
mistaken for MMA again.  The caller supplies a minimization gradient and
constraints in the canonical form ``g_j(x) <= 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import optimize


@dataclass(frozen=True)
class MMAState:
    iteration: int
    xold1: np.ndarray
    xold2: np.ndarray
    low: np.ndarray
    upp: np.ndarray


def initialize_mma_state(x: np.ndarray) -> MMAState:
    vector = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("initial MMA variables are non-finite")
    return MMAState(
        iteration=0,
        xold1=vector.copy(),
        xold2=vector.copy(),
        low=vector - 0.5,
        upp=vector + 0.5,
    )


def save_mma_state(path: Path, state: MMAState) -> None:
    np.savez_compressed(
        path,
        iteration=np.asarray(state.iteration),
        xold1=state.xold1,
        xold2=state.xold2,
        low=state.low,
        upp=state.upp,
    )


def load_mma_state(path: Path) -> MMAState:
    with np.load(path) as data:
        return MMAState(
            iteration=int(data["iteration"]),
            xold1=np.asarray(data["xold1"], dtype=np.float64),
            xold2=np.asarray(data["xold2"], dtype=np.float64),
            low=np.asarray(data["low"], dtype=np.float64),
            upp=np.asarray(data["upp"], dtype=np.float64),
        )


def mma_step(
    x: np.ndarray,
    objective_gradient_minimize: np.ndarray,
    constraint_values: np.ndarray,
    constraint_gradients: np.ndarray,
    state: MMAState,
    *,
    move_limit: float,
    xmin: float = 0.0,
    xmax: float = 1.0,
) -> tuple[np.ndarray, MMAState, dict[str, object]]:
    """Solve one persistent separable MMA approximation.

    ``move_limit`` is only a trust-region bound; it is not an update size.
    The actual step follows from the objective, active inequalities and the
    persistent moving asymptotes.
    """

    vector = np.asarray(x, dtype=np.float64).reshape(-1)
    df0 = np.asarray(objective_gradient_minimize, dtype=np.float64).reshape(-1)
    fval = np.asarray(constraint_values, dtype=np.float64).reshape(-1)
    dfdx = np.asarray(constraint_gradients, dtype=np.float64)
    if df0.shape != vector.shape:
        raise ValueError("objective gradient shape mismatch")
    if dfdx.shape != (fval.size, vector.size):
        raise ValueError("constraint Jacobian shape mismatch")
    if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(df0)):
        raise ValueError("MMA variables/objective gradient are non-finite")
    if not np.all(np.isfinite(fval)) or not np.all(np.isfinite(dfdx)):
        raise ValueError("MMA constraints are non-finite")
    if not 0.0 < move_limit <= xmax - xmin:
        raise ValueError("invalid MMA move limit")
    if state.xold1.shape != vector.shape:
        raise ValueError("MMA state shape mismatch")

    span = np.full_like(vector, xmax - xmin)
    iteration = state.iteration + 1
    if iteration <= 2:
        low = vector - 0.5 * span
        upp = vector + 0.5 * span
    else:
        trend = (vector - state.xold1) * (state.xold1 - state.xold2)
        factor = np.where(trend > 0.0, 1.2, np.where(trend < 0.0, 0.7, 1.0))
        low = vector - factor * (state.xold1 - state.low)
        upp = vector + factor * (state.upp - state.xold1)
        low = np.clip(low, vector - 10.0 * span, vector - 0.01 * span)
        upp = np.clip(upp, vector + 0.01 * span, vector + 10.0 * span)

    alfa = np.maximum.reduce((
        np.full_like(vector, xmin),
        low + 0.1 * (vector - low),
        vector - move_limit * span,
    ))
    beta_bound = np.minimum.reduce((
        np.full_like(vector, xmax),
        upp - 0.1 * (upp - vector),
        vector + move_limit * span,
    ))
    ux = np.maximum(upp - vector, 1.0e-12)
    xl = np.maximum(vector - low, 1.0e-12)
    raa = 1.0e-5

    p0_raw = np.maximum(df0, 0.0)
    q0_raw = np.maximum(-df0, 0.0)
    regularizer0 = 1.0e-3 * (p0_raw + q0_raw) + raa
    p0 = (p0_raw + regularizer0) * ux * ux
    q0 = (q0_raw + regularizer0) * xl * xl
    p_raw = np.maximum(dfdx, 0.0)
    q_raw = np.maximum(-dfdx, 0.0)
    regularizer = 1.0e-3 * (p_raw + q_raw) + raa / max(fval.size, 1)
    pmat = (p_raw + regularizer) * ux[None, :] ** 2
    qmat = (q_raw + regularizer) * xl[None, :] ** 2
    bvec = pmat @ (1.0 / ux) + qmat @ (1.0 / xl) - fval

    def primal(lam: np.ndarray) -> np.ndarray:
        p = p0 + pmat.T @ lam
        q = q0 + qmat.T @ lam
        sqrt_p = np.sqrt(np.maximum(p, 1.0e-300))
        sqrt_q = np.sqrt(np.maximum(q, 1.0e-300))
        free = (sqrt_q * upp + sqrt_p * low) / (sqrt_p + sqrt_q)
        return np.clip(free, alfa, beta_bound)

    def negative_dual(lam: np.ndarray) -> tuple[float, np.ndarray]:
        candidate = primal(lam)
        uc = np.maximum(upp - candidate, 1.0e-12)
        cl = np.maximum(candidate - low, 1.0e-12)
        p = p0 + pmat.T @ lam
        q = q0 + qmat.T @ lam
        dual = np.sum(p / uc + q / cl) - float(np.dot(lam, bvec))
        residual = pmat @ (1.0 / uc) + qmat @ (1.0 / cl) - bvec
        return -float(dual), -residual

    dual = optimize.minimize(
        negative_dual,
        np.zeros(fval.size, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * fval.size,
        options={"ftol": 1.0e-12, "gtol": 1.0e-9, "maxiter": 500},
    )
    if not dual.success:
        raise RuntimeError(f"MMA dual solve failed: {dual.message}")
    candidate = primal(np.asarray(dual.x, dtype=np.float64))
    if not np.all(np.isfinite(candidate)):
        raise RuntimeError("MMA produced a non-finite candidate")
    next_state = MMAState(
        iteration=iteration,
        xold1=vector.copy(),
        xold2=state.xold1.copy(),
        low=low,
        upp=upp,
    )
    approximation = pmat @ (1.0 / np.maximum(upp - candidate, 1.0e-12)) + qmat @ (
        1.0 / np.maximum(candidate - low, 1.0e-12)
    ) - bvec
    diagnostics = {
        "algorithm": "persistent_separable_method_of_moving_asymptotes",
        "dual_success": True,
        "dual_iterations": int(dual.nit),
        "dual_variables": np.asarray(dual.x, dtype=float).tolist(),
        "approximated_constraint_values": approximation.tolist(),
        "maximum_absolute_step": float(np.max(np.abs(candidate - vector))),
        "rms_step": float(np.sqrt(np.mean((candidate - vector) ** 2))),
        "move_limit_is_bound_not_learning_rate": float(move_limit),
        "used_adam": False,
        "gradient_direction_normalization": False,
        "hard_clip_after_update": False,
    }
    return candidate, next_state, diagnostics
