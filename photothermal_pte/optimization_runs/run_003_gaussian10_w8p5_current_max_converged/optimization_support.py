"""Fixed-inequality manufacturability support for Run 003."""

from __future__ import annotations

from dataclasses import dataclass
import json
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


def constraint_values_and_gradients(
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
    else:
        value = 0.0001
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
    """Recognize only the explicit external HPC-license checkout failure."""

    if payload.get("passed"):
        return False
    message = json.dumps(payload, default=str).lower()
    signatures = (
        "unable to checkout the requested hpc license",
        "requires 9 licenses for feature fdtd_solutions_engine",
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


__all__ = [
    "MMAState", "ProductionDensityMapping", "candidate_acceptance",
    "constraint_values_and_gradients", "design_metrics", "exact_binary_audit",
    "initialize_mma_state", "load_mma_state", "mma_step",
    "projected_binary_gate", "save_mma_state", "stage_caps",
    "stage_convergence",
    "transient_license_failure",
]
