"""Stateful MMA and 500 nm manufacturability metrics for Run 002.

The production density map is finite and nonperiodic.  All differentiable
constraints are therefore differentiated on the filtered field and pulled
back with the exact finite-filter transpose from ``ProductionDensityMapping``.
The binary morphology audit is deliberately separate: it is a fail-closed
fabrication audit, not a post-processing repair operation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, optimize
import torch

from production_density_mapping import ProductionDensityMapping


MINIMUM_FEATURE_M = 500.0e-9
MORPHOLOGY_RADIUS_M = MINIMUM_FEATURE_M / 2.0
ETA_ERODED = 0.75
ETA_DILATED = 0.25
ZHOU_DECAY_M2 = 1.0e-10


def disk_footprint(radius_pixels: int) -> np.ndarray:
    axis = np.arange(-radius_pixels, radius_pixels + 1)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return xx * xx + yy * yy <= radius_pixels * radius_pixels


def exact_binary_audit(rho: np.ndarray, spacing_m: float) -> dict[str, object]:
    """Audit a 0.5-threshold design for 500 nm solid and void openings.

    A 500 nm minimum feature is represented by a radius-250 nm opening test.
    Outside the finite design window is physical void: solid features cannot
    rely on it, while void channels may connect to it.
    """

    binary = np.asarray(rho, float) >= 0.5
    radius_pixels = int(np.ceil(MORPHOLOGY_RADIUS_M / spacing_m - 1.0e-12))
    footprint = disk_footprint(radius_pixels)
    solid_open = ndimage.binary_opening(
        binary, structure=footprint, border_value=0
    )
    void_open = ndimage.binary_opening(
        ~binary, structure=footprint, border_value=1
    )
    bad_solid = binary & ~solid_open
    bad_void = (~binary) & ~void_open
    return {
        "minimum_feature_nm": MINIMUM_FEATURE_M * 1.0e9,
        "opening_radius_nm": MORPHOLOGY_RADIUS_M * 1.0e9,
        "opening_radius_pixels": radius_pixels,
        "footprint_pixel_count": int(np.count_nonzero(footprint)),
        "solid_fraction": float(np.mean(binary)),
        "solid_bad_cell_count": int(np.count_nonzero(bad_solid)),
        "void_bad_cell_count": int(np.count_nonzero(bad_void)),
        "solid_bad_fraction_all_cells": float(np.mean(bad_solid)),
        "void_bad_fraction_all_cells": float(np.mean(bad_void)),
        "solid_pass": bool(not np.any(bad_solid)),
        "void_pass": bool(not np.any(bad_void)),
        "binary": binary,
        "bad_solid": bad_solid,
        "bad_void": bad_void,
    }


def _torch_projection(filtered: torch.Tensor, beta: float, eta: float) -> torch.Tensor:
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return (
        np.tanh(beta * eta) + torch.tanh(beta * (filtered - eta))
    ) / denominator


def smooth_constraints(
    latent: np.ndarray,
    beta: float,
    mapping: ProductionDensityMapping,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Return Zhou solid/void indicators and grayness with latent gradients."""

    filtered_np = mapping.filtered(latent)
    filtered = torch.tensor(filtered_np, dtype=torch.float64, requires_grad=True)
    rho = _torch_projection(filtered, float(beta), mapping.eta)
    gx, gy = torch.gradient(filtered, spacing=(mapping.spacing_m, mapping.spacing_m))
    suppression = torch.exp(-ZHOU_DECAY_M2 * (gx * gx + gy * gy))
    solid = torch.mean(
        rho * suppression * torch.relu(ETA_ERODED - filtered) ** 2
    )
    void = torch.mean(
        (1.0 - rho) * suppression * torch.relu(filtered - ETA_DILATED) ** 2
    )
    gray = torch.mean(4.0 * rho * (1.0 - rho))
    values = {"solid": solid, "void": void, "gray": gray}
    result_values: dict[str, float] = {}
    gradients: dict[str, np.ndarray] = {}
    for index, (name, value) in enumerate(values.items()):
        gradient_filtered = torch.autograd.grad(
            value,
            filtered,
            retain_graph=index < len(values) - 1,
        )[0].detach().cpu().numpy()
        result_values[name] = float(value.detach().cpu())
        gradients[name] = mapping.filter_transpose(gradient_filtered)
    return result_values, gradients


def design_metrics(
    latent: np.ndarray,
    beta: float,
    mapping: ProductionDensityMapping | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    mapping = mapping or ProductionDensityMapping()
    latent = np.asarray(latent, float)
    filtered = mapping.filtered(latent)
    rho = mapping.physical(latent, beta)
    smooth, gradients = smooth_constraints(latent, beta, mapping)
    exact = exact_binary_audit(rho, mapping.spacing_m)
    metrics: dict[str, object] = {
        "beta": float(beta),
        "latent_min": float(np.min(latent)),
        "latent_max": float(np.max(latent)),
        "filtered_min": float(np.min(filtered)),
        "filtered_max": float(np.max(filtered)),
        "rho_min": float(np.min(rho)),
        "rho_max": float(np.max(rho)),
        "rho_mean": float(np.mean(rho)),
        "gray_fraction_0p01_0p99": float(np.mean((rho > 0.01) & (rho < 0.99))),
        "gray_fraction_0p05_0p95": float(np.mean((rho > 0.05) & (rho < 0.95))),
        "binarization_metric_mean_4rho1mrho": smooth["gray"],
        "smooth_solid_constraint": smooth["solid"],
        "smooth_void_constraint": smooth["void"],
        "exact_binary_audit": {k: v for k, v in exact.items() if not isinstance(v, np.ndarray)},
    }
    arrays = {
        "latent": latent,
        "filtered": filtered,
        "rho": rho,
        "binary": exact["binary"],
        "bad_solid": exact["bad_solid"],
        "bad_void": exact["bad_void"],
        **{f"gradient_{key}": value for key, value in gradients.items()},
    }
    return metrics, arrays


@dataclass
class MMAState:
    iteration: int
    xold1: np.ndarray
    xold2: np.ndarray
    low: np.ndarray
    upp: np.ndarray


def initialize_mma_state(x: np.ndarray) -> MMAState:
    x = np.asarray(x, float).ravel()
    return MMAState(
        iteration=0,
        xold1=x.copy(),
        xold2=x.copy(),
        low=x - 0.5,
        upp=x + 0.5,
    )


def mma_step(
    x: np.ndarray,
    objective_gradient_minimize: np.ndarray,
    constraint_values: np.ndarray,
    constraint_gradients: np.ndarray,
    state: MMAState,
    move: float = 0.03,
    xmin: float = 0.0,
    xmax: float = 1.0,
) -> tuple[np.ndarray, MMAState, dict[str, object]]:
    """Solve one separable convex MMA subproblem with persistent asymptotes."""

    x = np.asarray(x, float).ravel()
    df0 = np.asarray(objective_gradient_minimize, float).ravel()
    fval = np.asarray(constraint_values, float).ravel()
    dfdx = np.asarray(constraint_gradients, float)
    if dfdx.shape != (fval.size, x.size):
        raise ValueError("constraint Jacobian shape mismatch")
    xmami = np.full_like(x, xmax - xmin)
    iteration = state.iteration + 1
    if iteration <= 2:
        low = x - 0.5 * xmami
        upp = x + 0.5 * xmami
    else:
        trend = (x - state.xold1) * (state.xold1 - state.xold2)
        factor = np.where(trend > 0.0, 1.2, np.where(trend < 0.0, 0.7, 1.0))
        low = x - factor * (state.xold1 - state.low)
        upp = x + factor * (state.upp - state.xold1)
        low = np.maximum(low, x - 10.0 * xmami)
        low = np.minimum(low, x - 0.01 * xmami)
        upp = np.minimum(upp, x + 10.0 * xmami)
        upp = np.maximum(upp, x + 0.01 * xmami)
    albefa = 0.1
    alfa = np.maximum.reduce((
        np.full_like(x, xmin),
        low + albefa * (x - low),
        x - move * xmami,
    ))
    beta_bound = np.minimum.reduce((
        np.full_like(x, xmax),
        upp - albefa * (upp - x),
        x + move * xmami,
    ))
    ux = np.maximum(upp - x, 1.0e-12)
    xl = np.maximum(x - low, 1.0e-12)
    raa = 1.0e-5
    p0_raw = np.maximum(df0, 0.0)
    q0_raw = np.maximum(-df0, 0.0)
    regularizer0 = 1.0e-3 * (p0_raw + q0_raw) + raa
    p0 = (p0_raw + regularizer0) * ux * ux
    q0 = (q0_raw + regularizer0) * xl * xl
    p_raw = np.maximum(dfdx, 0.0)
    q_raw = np.maximum(-dfdx, 0.0)
    regularizer = 1.0e-3 * (p_raw + q_raw) + raa / max(1, fval.size)
    P = (p_raw + regularizer) * ux[np.newaxis, :] ** 2
    Q = (q_raw + regularizer) * xl[np.newaxis, :] ** 2
    b = P @ (1.0 / ux) + Q @ (1.0 / xl) - fval

    def primal(lam: np.ndarray) -> np.ndarray:
        p = p0 + P.T @ lam
        q = q0 + Q.T @ lam
        sqrt_p = np.sqrt(np.maximum(p, 1.0e-300))
        sqrt_q = np.sqrt(np.maximum(q, 1.0e-300))
        unconstrained = (sqrt_q * upp + sqrt_p * low) / (sqrt_p + sqrt_q)
        return np.minimum(np.maximum(unconstrained, alfa), beta_bound)

    def negative_dual(lam: np.ndarray) -> tuple[float, np.ndarray]:
        candidate = primal(lam)
        uc = np.maximum(upp - candidate, 1.0e-12)
        cl = np.maximum(candidate - low, 1.0e-12)
        p = p0 + P.T @ lam
        q = q0 + Q.T @ lam
        dual = np.sum(p / uc + q / cl) - float(np.dot(lam, b))
        residual = P @ (1.0 / uc) + Q @ (1.0 / cl) - b
        return -float(dual), -residual

    dual = optimize.minimize(
        negative_dual,
        np.zeros(fval.size, dtype=float),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * fval.size,
        options={"ftol": 1.0e-12, "gtol": 1.0e-9, "maxiter": 500},
    )
    candidate = primal(np.asarray(dual.x, float))
    next_state = MMAState(
        iteration=iteration,
        xold1=x.copy(),
        xold2=state.xold1.copy(),
        low=low,
        upp=upp,
    )
    approximation = P @ (1.0 / np.maximum(upp - candidate, 1.0e-12)) + Q @ (
        1.0 / np.maximum(candidate - low, 1.0e-12)
    ) - b
    diagnostics = {
        "dual_success": bool(dual.success),
        "dual_message": str(dual.message),
        "dual_iterations": int(dual.nit),
        "dual_variables": np.asarray(dual.x, float).tolist(),
        "approximated_constraint_values": approximation.tolist(),
        "max_abs_step": float(np.max(np.abs(candidate - x))),
        "move_limit": float(move),
    }
    return candidate, next_state, diagnostics


def save_mma_state(path, state: MMAState) -> None:
    np.savez_compressed(
        path,
        iteration=np.asarray(state.iteration),
        xold1=state.xold1,
        xold2=state.xold2,
        low=state.low,
        upp=state.upp,
    )


def load_mma_state(path) -> MMAState:
    data = np.load(path)
    return MMAState(
        iteration=int(data["iteration"]),
        xold1=np.asarray(data["xold1"], float),
        xold2=np.asarray(data["xold2"], float),
        low=np.asarray(data["low"], float),
        upp=np.asarray(data["upp"], float),
    )
