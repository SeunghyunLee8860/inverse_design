"""Fixed-inequality manufacturability support for Run 003."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
import torch


HERE = Path(__file__).resolve().parent
RUN002 = HERE.parent / "run_002_gaussian10_w8p5_current_max"
if str(RUN002) not in sys.path:
    sys.path.insert(0, str(RUN002))

from beta_continuation_support import (  # noqa: E402
    MMAState,
    initialize_mma_state,
    load_mma_state,
    mma_step,
    save_mma_state,
)
from production_density_mapping import ProductionDensityMapping  # noqa: E402


MINIMUM_FEATURE_M = 500.0e-9
MORPHOLOGY_RADIUS_M = MINIMUM_FEATURE_M / 2.0
ETA_ERODED = 0.75
ETA_DILATED = 0.25
ZHOU_DECAY_M2 = 1.0e-10
PNORM = 8.0
LEGACY_CONSTRAINT_CONTRACT = "legacy_zhou_p8_v1"
DISK_CONSTRAINT_CONTRACT = "soft_disk_opening_500nm_v3_exact_nonincrease"
DISK_CONSTRAINT_START_BETA = 32.0
DISK_RECOVERY_START_GLOBAL_ITERATION = 85
DISK_SHARPEN_GAMMA = 64.0
DISK_SOFT_EXTREMA_TAU = 1.0e-3
DEFAULT_MMA_MOVE = 0.02
MINIMUM_MMA_MOVE = 0.00125
MOVE_PLATEAU_WINDOW = 4


def disk_structuring_element(radius_pixels: int) -> np.ndarray:
    axis = np.arange(-radius_pixels, radius_pixels + 1)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return xx * xx + yy * yy <= radius_pixels * radius_pixels


def exact_binary_audit(rho: np.ndarray, spacing_m: float) -> dict[str, object]:
    binary = np.asarray(rho, float) >= 0.5
    radius_pixels = int(np.ceil(MORPHOLOGY_RADIUS_M / spacing_m - 1.0e-12))
    structure = disk_structuring_element(radius_pixels)
    solid_open = ndimage.binary_opening(binary, structure=structure, border_value=0)
    void_open = ndimage.binary_opening(~binary, structure=structure, border_value=1)
    bad_solid = binary & ~solid_open
    bad_void = (~binary) & ~void_open
    return {
        "minimum_feature_nm": MINIMUM_FEATURE_M * 1.0e9,
        "opening_radius_nm": MORPHOLOGY_RADIUS_M * 1.0e9,
        "opening_radius_pixels": radius_pixels,
        "structuring_element_pixel_count": int(np.count_nonzero(structure)),
        "design_pixel_count": int(binary.size),
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


def _projection(filtered: torch.Tensor, beta: float, eta: float) -> torch.Tensor:
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return (
        np.tanh(beta * eta) + torch.tanh(beta * (filtered - eta))
    ) / denominator


def legacy_zhou_constraint_values_and_gradients(
    latent: np.ndarray,
    beta: float,
    mapping: ProductionDensityMapping | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return p=8 Zhou solid/void values and exact latent gradients."""

    mapping = mapping or ProductionDensityMapping()
    filtered_np = mapping.filtered(latent)
    filtered = torch.tensor(filtered_np, dtype=torch.float64, requires_grad=True)
    rho = _projection(filtered, float(beta), mapping.eta)
    gx, gy = torch.gradient(
        filtered, spacing=(mapping.spacing_m, mapping.spacing_m)
    )
    suppression = torch.exp(-ZHOU_DECAY_M2 * (gx * gx + gy * gy))
    fields = (
        rho * suppression * torch.relu(ETA_ERODED - filtered) ** 2,
        (1.0 - rho) * suppression * torch.relu(filtered - ETA_DILATED) ** 2,
    )
    values: list[float] = []
    gradients: list[np.ndarray] = []
    local: dict[str, np.ndarray] = {}
    for index, (name, field) in enumerate(zip(("solid", "void"), fields)):
        aggregate = (torch.mean(torch.abs(field) ** PNORM) + 1.0e-300) ** (
            1.0 / PNORM
        )
        gradient_filtered = torch.autograd.grad(
            aggregate, filtered, retain_graph=index == 0
        )[0].detach().cpu().numpy()
        values.append(float(aggregate.detach().cpu()))
        gradients.append(mapping.filter_transpose(gradient_filtered))
        local[f"{name}_penalty_field"] = field.detach().cpu().numpy()
    return np.asarray(values), np.stack(gradients), local


def _disk_offsets(radius_pixels: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (di, dj)
        for di in range(-radius_pixels, radius_pixels + 1)
        for dj in range(-radius_pixels, radius_pixels + 1)
        if di * di + dj * dj <= radius_pixels * radius_pixels
    )


def _constraint_device() -> torch.device:
    requested = os.environ.get("RUN003_CONSTRAINT_TORCH_DEVICE", "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"requested morphology-constraint device {requested!r} is unavailable"
        )
    return device


def _shifted_disk_stack(
    value: torch.Tensor,
    offsets: tuple[tuple[int, int], ...],
    radius_pixels: int,
    border_value: float,
) -> torch.Tensor:
    padded = torch.nn.functional.pad(
        value,
        (radius_pixels, radius_pixels, radius_pixels, radius_pixels),
        value=float(border_value),
    )
    height, width = value.shape
    return torch.stack(
        tuple(
            padded[
                radius_pixels + di : radius_pixels + di + height,
                radius_pixels + dj : radius_pixels + dj + width,
            ]
            for di, dj in offsets
        ),
        dim=0,
    )


def _soft_disk_opening(
    value: torch.Tensor,
    offsets: tuple[tuple[int, int], ...],
    radius_pixels: int,
    border_value: float,
) -> torch.Tensor:
    """Differentiable opening with the exact audit's disk and border policy."""

    log_count = float(np.log(len(offsets)))
    erosion_stack = _shifted_disk_stack(
        value, offsets, radius_pixels, border_value
    )
    eroded = -DISK_SOFT_EXTREMA_TAU * (
        torch.logsumexp(-erosion_stack / DISK_SOFT_EXTREMA_TAU, dim=0)
        - log_count
    )
    dilation_stack = _shifted_disk_stack(
        eroded, offsets, radius_pixels, border_value
    )
    return DISK_SOFT_EXTREMA_TAU * (
        torch.logsumexp(dilation_stack / DISK_SOFT_EXTREMA_TAU, dim=0)
        - log_count
    )


def disk_constraint_values_and_gradients(
    latent: np.ndarray,
    beta: float,
    mapping: ProductionDensityMapping | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return solid/void penalties aligned with the exact 500 nm disk audit.

    The projected density is smoothly sharpened about the exact audit threshold
    before applying a log-sum-exp approximation of the same disk opening used
    by :func:`exact_binary_audit`.  Its positive opening residual tracks the
    exact bad-cell masks; the exact nondifferentiable audit remains the final
    zero-violation gate because the smooth extrema have a small finite floor.
    The physical-density cotangent is propagated through the certified finite
    conic filter/projection transpose; no post-hoc morphology edits the design.
    """

    mapping = mapping or ProductionDensityMapping()
    latent = np.asarray(latent, dtype=float)
    rho_np = mapping.physical(latent, float(beta))
    radius_pixels = int(
        np.ceil(MORPHOLOGY_RADIUS_M / mapping.spacing_m - 1.0e-12)
    )
    offsets = _disk_offsets(radius_pixels)
    rho = torch.tensor(
        rho_np,
        dtype=torch.float64,
        device=_constraint_device(),
        requires_grad=True,
    )
    sharpened = torch.sigmoid(DISK_SHARPEN_GAMMA * (rho - 0.5))
    values: list[float] = []
    gradients: list[np.ndarray] = []
    local: dict[str, np.ndarray] = {}
    for index, (name, phase, border_value) in enumerate(
        (
            ("solid", sharpened, 0.0),
            ("void", 1.0 - sharpened, 1.0),
        )
    ):
        opened = _soft_disk_opening(
            phase, offsets, radius_pixels, border_value
        )
        residual = torch.relu(phase - opened)
        aggregate = torch.mean(residual)
        gradient_rho = torch.autograd.grad(
            aggregate, rho, retain_graph=index == 0
        )[0].detach().cpu().numpy()
        values.append(float(aggregate.detach().cpu()))
        gradients.append(mapping.vjp(latent, gradient_rho, float(beta)))
        local[f"{name}_penalty_field"] = residual.detach().cpu().numpy()
    return np.asarray(values), np.stack(gradients), local


def constraint_contract(beta: float) -> str:
    return (
        DISK_CONSTRAINT_CONTRACT
        if float(beta) >= DISK_CONSTRAINT_START_BETA
        else LEGACY_CONSTRAINT_CONTRACT
    )


def constraint_values_and_gradients(
    latent: np.ndarray,
    beta: float,
    mapping: ProductionDensityMapping | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if constraint_contract(beta) == DISK_CONSTRAINT_CONTRACT:
        return disk_constraint_values_and_gradients(latent, beta, mapping)
    return legacy_zhou_constraint_values_and_gradients(latent, beta, mapping)


def stage_caps(beta: float) -> np.ndarray:
    if beta <= 2.0:
        value = 0.040
    elif beta <= 4.0:
        value = 0.030
    elif beta <= 8.0:
        value = 0.020
    elif beta <= 16.0:
        value = 0.008
    elif beta <= 32.0:
        value = 0.002
    elif beta <= 64.0:
        value = 0.001
    elif beta <= 128.0:
        value = 0.0005
    elif beta <= 256.0:
        value = 0.00025
    elif beta <= 512.0:
        value = 0.0001
    else:
        value = 0.00005
    return np.asarray([value, value], float)


def design_metrics(
    latent: np.ndarray,
    beta: float,
    mapping: ProductionDensityMapping | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    mapping = mapping or ProductionDensityMapping()
    latent = np.asarray(latent, float)
    filtered = mapping.filtered(latent)
    rho = mapping.physical(latent, beta)
    constraints, gradients, local = constraint_values_and_gradients(
        latent, beta, mapping
    )
    exact = exact_binary_audit(rho, mapping.spacing_m)
    caps = stage_caps(beta)
    normalized = constraints / caps
    metrics: dict[str, object] = {
        "beta": float(beta),
        "constraint_contract": constraint_contract(beta),
        "latent_min": float(np.min(latent)),
        "latent_max": float(np.max(latent)),
        "filtered_min": float(np.min(filtered)),
        "filtered_max": float(np.max(filtered)),
        "rho_min": float(np.min(rho)),
        "rho_max": float(np.max(rho)),
        "rho_mean": float(np.mean(rho)),
        "gray_fraction_0p01_0p99": float(np.mean((rho > 0.01) & (rho < 0.99))),
        "gray_fraction_0p05_0p95": float(np.mean((rho > 0.05) & (rho < 0.95))),
        "binarization_metric_mean_4rho1mrho": float(
            np.mean(4.0 * rho * (1.0 - rho))
        ),
        "solid_constraint": float(constraints[0]),
        "void_constraint": float(constraints[1]),
        "solid_constraint_cap": float(caps[0]),
        "void_constraint_cap": float(caps[1]),
        "maximum_normalized_constraint": float(np.max(normalized)),
        "constraints_feasible": bool(np.all(normalized <= 1.0 + 5.0e-3)),
        "exact_binary_audit": {
            key: value for key, value in exact.items() if not isinstance(value, np.ndarray)
        },
    }
    arrays = {
        "latent": latent,
        "filtered": filtered,
        "rho": rho,
        "binary": exact["binary"],
        "bad_solid": exact["bad_solid"],
        "bad_void": exact["bad_void"],
        "gradient_solid": gradients[0],
        "gradient_void": gradients[1],
        **local,
    }
    return metrics, arrays


def projected_binary_gate(metrics: dict[str, object]) -> bool:
    exact = metrics["exact_binary_audit"]
    return bool(
        metrics["gray_fraction_0p01_0p99"] < 0.001
        and metrics["binarization_metric_mean_4rho1mrho"] < 0.001
        and exact["solid_bad_cell_count"] == 0
        and exact["void_bad_cell_count"] == 0
    )


def transient_license_failure(payload: dict[str, object]) -> bool:
    """Recognize explicit external license checkout/startup failures only."""

    if payload.get("passed"):
        return False
    message = json.dumps(payload, default=str).lower()
    signatures = (
        "unable to checkout the requested hpc license",
        "requires 9 licenses for feature fdtd_solutions_engine",
        "failed to start messaging, check licenses",
        "failed to set up ansys license sharing",
        "ansysli exited or could not read server port",
        "could not bind socket on port",
    )
    return any(signature in message for signature in signatures)


def normalized_violation(values: np.ndarray, caps: np.ndarray) -> float:
    return float(np.linalg.norm(np.maximum(np.asarray(values) / caps - 1.0, 0.0)))


def candidate_acceptance(
    current_objective: float,
    candidate_objective: float,
    current_constraints: np.ndarray,
    candidate_constraints: np.ndarray,
    caps: np.ndarray,
    current_exact_bad_counts: np.ndarray | None = None,
    candidate_exact_bad_counts: np.ndarray | None = None,
) -> dict[str, object]:
    current_v = normalized_violation(current_constraints, caps)
    candidate_v = normalized_violation(candidate_constraints, caps)
    objective_ratio = float(candidate_objective / current_objective)
    current_feasible = current_v <= 5.0e-3
    candidate_feasible = candidate_v <= 5.0e-3
    if current_feasible:
        accepted = candidate_feasible and objective_ratio >= 0.998
        reason = "remain feasible and preserve actual FOM"
    else:
        violation_reduction = (current_v - candidate_v) / max(current_v, 1.0e-300)
        accepted = (
            (candidate_feasible or violation_reduction >= 0.005)
            and objective_ratio >= 0.95
        )
        reason = "reduce fixed-cap violation while limiting actual FOM loss"
    exact_gate_enabled = (
        current_exact_bad_counts is not None
        and candidate_exact_bad_counts is not None
    )
    current_exact = (
        np.asarray(current_exact_bad_counts, dtype=int)
        if current_exact_bad_counts is not None else np.asarray([], dtype=int)
    )
    candidate_exact = (
        np.asarray(candidate_exact_bad_counts, dtype=int)
        if candidate_exact_bad_counts is not None else np.asarray([], dtype=int)
    )
    if exact_gate_enabled:
        if current_exact.shape != (2,) or candidate_exact.shape != (2,):
            raise ValueError("exact bad-cell counts must be [solid, void]")
        exact_total_nonincreasing = bool(
            np.sum(candidate_exact) <= np.sum(current_exact)
        )
        accepted = bool(accepted and exact_total_nonincreasing)
    else:
        exact_total_nonincreasing = True
    return {
        "accepted": bool(accepted),
        "reason": reason,
        "current_feasible": bool(current_feasible),
        "candidate_feasible": bool(candidate_feasible),
        "current_normalized_violation": current_v,
        "candidate_normalized_violation": candidate_v,
        "relative_violation_reduction": float(
            (current_v - candidate_v) / max(current_v, 1.0e-300)
        ) if current_v > 0.0 else 0.0,
        "objective_ratio": objective_ratio,
        "exact_DRC_gate_enabled": bool(exact_gate_enabled),
        "current_exact_bad_counts": current_exact.tolist(),
        "candidate_exact_bad_counts": candidate_exact.tolist(),
        "current_exact_bad_total": int(np.sum(current_exact)),
        "candidate_exact_bad_total": int(np.sum(candidate_exact)),
        "exact_total_nonincreasing": bool(exact_total_nonincreasing),
    }


@dataclass(frozen=True)
class StageConvergence:
    converged: bool
    reason: str
    maximum_relative_fom_change: float
    maximum_rho_rms_change: float
    maximum_rho_max_change: float


def stage_convergence(history: list[dict[str, object]], beta: float) -> StageConvergence:
    accepted = [
        row for row in history
        if float(row["beta"]) == float(beta) and row["role"] == "accepted_mma"
        and not (
            float(beta) == DISK_CONSTRAINT_START_BETA
            and int(row.get("global_iteration", -1))
            <= DISK_RECOVERY_START_GLOBAL_ITERATION
        )
    ]
    minimum = 10 if beta <= 2.0 else 8
    window = 4
    if len(accepted) < minimum or len(accepted) < window:
        return StageConvergence(False, f"need at least {minimum} accepted updates", np.inf, np.inf, np.inf)
    recent = accepted[-window:]
    if not bool(recent[-1]["constraints_feasible"]):
        return StageConvergence(False, "fixed solid/void inequalities are not feasible", np.inf, np.inf, np.inf)
    rel = [abs(float(row["relative_fom_change"])) for row in recent]
    rms = [float(row["rho_rms_change"]) for row in recent]
    maximum = [float(row["rho_max_change"]) for row in recent]
    gates = (max(rel) < 0.005, max(rms) < 0.0025, max(maximum) < 0.015)
    return StageConvergence(
        bool(all(gates)),
        "all four-update FOM/density plateau gates pass" if all(gates) else "recent FOM/density changes have not plateaued",
        max(rel), max(rms), max(maximum),
    )


def adaptive_move_ceiling(
    history: list[dict[str, object]],
    beta: float,
    accepted_moves: list[float],
) -> float:
    """Return a monotone MMA move ceiling without weakening convergence gates.

    A move reduction is permitted only after four accepted updates at the
    current (smallest previously accepted) move have established an FOM
    plateau while the physical-density plateau still fails.  The ceiling can
    therefore only decrease, and each new ceiling receives four genuine
    solver evaluations before another reduction is considered.
    """

    accepted = [
        row for row in history
        if float(row["beta"]) == float(beta) and row["role"] == "accepted_mma"
        and not (
            float(beta) == DISK_CONSTRAINT_START_BETA
            and int(row.get("global_iteration", -1))
            <= DISK_RECOVERY_START_GLOBAL_ITERATION
        )
    ]
    if len(accepted_moves) != len(accepted):
        raise ValueError("accepted move provenance does not match accepted history")
    if not accepted:
        return DEFAULT_MMA_MOVE
    ceiling = min([DEFAULT_MMA_MOVE, *map(float, accepted_moves)])
    if ceiling <= MINIMUM_MMA_MOVE * (1.0 + 1.0e-12):
        return MINIMUM_MMA_MOVE
    tail_at_ceiling = 0
    for move in reversed(accepted_moves):
        if np.isclose(float(move), ceiling, rtol=0.0, atol=1.0e-15):
            tail_at_ceiling += 1
        else:
            break
    if tail_at_ceiling < MOVE_PLATEAU_WINDOW:
        return ceiling
    recent = accepted[-MOVE_PLATEAU_WINDOW:]
    if not bool(recent[-1]["constraints_feasible"]):
        return ceiling
    fom_plateau = max(
        abs(float(row["relative_fom_change"])) for row in recent
    ) < 0.005
    density_plateau = (
        max(float(row["rho_rms_change"]) for row in recent) < 0.0025
        and max(float(row["rho_max_change"]) for row in recent) < 0.015
    )
    if fom_plateau and not density_plateau:
        return max(MINIMUM_MMA_MOVE, ceiling / 2.0)
    return ceiling


__all__ = [
    "DISK_CONSTRAINT_START_BETA", "DISK_RECOVERY_START_GLOBAL_ITERATION",
    "MMAState", "ProductionDensityMapping", "adaptive_move_ceiling", "candidate_acceptance",
    "constraint_contract", "constraint_values_and_gradients",
    "disk_constraint_values_and_gradients", "design_metrics", "exact_binary_audit",
    "initialize_mma_state", "load_mma_state", "mma_step",
    "projected_binary_gate", "save_mma_state", "stage_caps",
    "stage_convergence",
    "transient_license_failure",
]
