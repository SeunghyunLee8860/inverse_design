"""Low-beta topology-exploration support for the Run 005 pilot."""

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
DISK_CONSTRAINT_CONTRACT = "soft_disk_opening_500nm_from_iteration_zero_v5"
MMA_CAP_CONTRACT = "run005_full_continuation_fixed_per_beta_caps_v8"
DISK_CONSTRAINT_START_BETA = 2.0
MMA_TOPOLOGY_RESET_GLOBAL_ITERATION = 9
DISK_SHARPEN_GAMMA = 64.0
DISK_SOFT_EXTREMA_TAU = 1.0e-3
DEFAULT_MMA_MOVE = 0.01
MINIMUM_MMA_MOVE = 0.0025
MOVE_PLATEAU_WINDOW = 4
FEASIBLE_FOM_RETENTION = 0.998
LOW_BETA_CATASTROPHIC_EXACT_GROWTH_FACTOR = 1.50
LOW_BETA_CATASTROPHIC_EXACT_GROWTH_ABSOLUTE = 25
SMALL_MOVE_HALT_THRESHOLD = 0.0025
DEFAULT_STAGE_CAPS_PATH = HERE / "stage_caps.json"
EXACT_ACTIVE_SET_RESTORATION_CONTRACT = (
    "sequential_exact_drc_active_set_filter_vjp_v2"
)
EXACT_ACTIVE_SET_PHASE_WEIGHTS = (
    0.0, 0.05, 0.10, 0.20, 0.25, 0.33, 0.50, 0.75,
    1.0, 1.5, 2.0, 3.0, 4.0, 8.0,
)
EXACT_ACTIVE_SET_MOVES = (
    0.20, 0.15, 0.10, 0.075, 0.05, 0.04, 0.03,
    0.025, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.0025,
)


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
    requested = os.environ.get("RUN005_CONSTRAINT_TORCH_DEVICE", "cpu")
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
    del beta
    return DISK_CONSTRAINT_CONTRACT


def constraint_values_and_gradients(
    latent: np.ndarray,
    beta: float,
    mapping: ProductionDensityMapping | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if constraint_contract(beta) == DISK_CONSTRAINT_CONTRACT:
        return disk_constraint_values_and_gradients(latent, beta, mapping)
    return legacy_zhou_constraint_values_and_gradients(latent, beta, mapping)


def _stage_caps_path() -> Path:
    return Path(
        os.environ.get("RUN005_STAGE_CAPS_FILE", str(DEFAULT_STAGE_CAPS_PATH))
    ).expanduser().resolve()


def _beta_key(beta: float) -> str:
    value = float(beta)
    if not value.is_integer():
        raise ValueError("continuation beta must be an integer")
    return str(int(value))


def stage_caps(beta: float) -> np.ndarray:
    """Return the immutable solid/void cap for one beta stage.

    Beta=2 keeps the reviewed Run005 exploration envelope.  Every later stage
    is calibrated exactly once from its incoming checkpoint and persisted in
    ``stage_caps.json``.  Missing entries fail closed; a candidate can never
    silently move its own cap.
    """

    if np.isclose(float(beta), 2.0, rtol=0.0, atol=1.0e-12):
        # Final beta=2 topology-search epoch, reviewed after g009.  Two
        # consecutive 0.0025 moves still improved the solver-backed FOM by
        # 2.31% and 2.26%, proving that the stage had not plateaued; the old
        # void cap was dictating the move.  This deliberately loose pair is
        # immutable for the remaining bounded beta=2 budget and is not
        # recalibrated per proposal.  Later beta stages restore phase-wise
        # nonincrease and exact-DRC cleanup gates.
        return np.asarray((6.00e-4, 2.00e-4), float)
    path = _stage_caps_path()
    if not path.exists():
        raise RuntimeError(f"missing fixed continuation-cap registry: {path}")
    payload = json.loads(path.read_text())
    entry = payload.get("stages", {}).get(_beta_key(beta))
    if entry is None:
        raise RuntimeError(
            f"beta={float(beta):g} has no checkpoint-calibrated fixed cap"
        )
    caps = np.asarray(entry["caps"], dtype=float)
    if caps.shape != (2,) or not np.all(np.isfinite(caps)) or np.any(caps <= 0):
        raise RuntimeError(f"invalid fixed caps for beta={float(beta):g}")
    return caps


def calibrate_stage_caps(
    latent: np.ndarray,
    beta: float,
    target_occupancy: float,
    mapping: ProductionDensityMapping | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Persist one cap epoch from the incoming checkpoint, never per step."""

    if float(beta) <= 2.0:
        values, _, _ = constraint_values_and_gradients(latent, beta, mapping)
        caps = stage_caps(beta)
        return caps, {
            "beta": float(beta),
            "incoming_values": values.tolist(),
            "caps": caps.tolist(),
            "target_occupancy": None,
            "actual_occupancy": (values / caps).tolist(),
            "source": "reviewed_beta2_fixed_envelope",
        }
    occupancy = float(target_occupancy)
    if not 0.5 <= occupancy <= 1.5:
        raise ValueError("target cap occupancy must be in [0.5, 1.5]")
    path = _stage_caps_path()
    payload = json.loads(path.read_text()) if path.exists() else {
        "contract": MMA_CAP_CONTRACT,
        "stages": {},
    }
    if payload.get("contract") != MMA_CAP_CONTRACT:
        raise RuntimeError("stage-cap registry contract mismatch")
    key = _beta_key(beta)
    existing = payload.setdefault("stages", {}).get(key)
    if existing is not None:
        return stage_caps(beta), existing
    values, _, _ = constraint_values_and_gradients(latent, beta, mapping)
    caps = np.maximum(values / occupancy, 1.0e-12)
    entry = {
        "beta": float(beta),
        "incoming_values": values.tolist(),
        "caps": caps.tolist(),
        "target_occupancy": occupancy,
        "actual_occupancy": (values / caps).tolist(),
        "source": (
            "single_stage_start_reprojection_cleanup_tightening"
            if occupancy > 1.0
            else "single_stage_start_reprojection_exploration_envelope"
        ),
    }
    payload["stages"][key] = entry
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return caps, entry


def smooth_feasibility_move_retries(move_ceiling: float) -> tuple[float, ...]:
    """Bounded low-beta move trials used only before an expensive solve.

    A smaller move can be considered when the offline smooth constraints reject
    a proposal.  The sequence never goes below 0.0025 and must not be reused
    after a solver-backed FOM rejection.
    """

    ceiling = float(move_ceiling)
    if not np.isfinite(ceiling) or ceiling <= 0.0:
        raise ValueError("move ceiling must be finite and positive")
    retries: list[float] = []
    trial = ceiling
    while trial >= MINIMUM_MMA_MOVE * (1.0 - 1.0e-12):
        retries.append(max(MINIMUM_MMA_MOVE, trial))
        if np.isclose(retries[-1], MINIMUM_MMA_MOVE, rtol=0.0, atol=1.0e-15):
            break
        trial /= 2.0
    return tuple(retries)


def mma_effective_constraint_caps(
    values: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Use fixed stage caps and preserve each phase from beta=4 onward."""

    current = np.asarray(values, dtype=float)
    caps = stage_caps(beta)
    if current.shape != caps.shape or not np.all(np.isfinite(current)):
        raise ValueError("constraint values must be a finite solid/void pair")
    if np.any(current <= 0.0):
        raise ValueError("constraint values must be positive")
    return np.minimum(caps, current) if float(beta) >= 4.0 else caps


def catastrophic_exact_growth(
    current_total: int,
    candidate_total: int,
) -> tuple[bool, int]:
    """Return the low-beta diagnostic safety decision and integer limit.

    Exact thresholded morphology is discontinuous and is not a low-beta step
    objective.  The pilot halts only if the count grows by more than both 50%
    and 25 cells; it never line-searches into exact-cell micro-repair.
    """

    current = int(current_total)
    candidate = int(candidate_total)
    if current < 0 or candidate < 0:
        raise ValueError("exact bad-cell totals must be nonnegative")
    limit = max(
        int(np.ceil(current * LOW_BETA_CATASTROPHIC_EXACT_GROWTH_FACTOR)),
        current + LOW_BETA_CATASTROPHIC_EXACT_GROWTH_ABSOLUTE,
    )
    return bool(candidate > limit), limit


def two_consecutive_subthreshold_moves(accepted_moves: list[float]) -> bool:
    """Detect the approved halt after two consecutive micro-repair moves."""

    return bool(
        len(accepted_moves) >= 2
        and float(accepted_moves[-1]) <= SMALL_MOVE_HALT_THRESHOLD
        and float(accepted_moves[-2]) <= SMALL_MOVE_HALT_THRESHOLD
    )


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


def exact_active_set_restoration_candidate(
    latent: np.ndarray,
    beta: float,
    objective_gradient_A: np.ndarray,
    mapping: ProductionDensityMapping | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Choose a bounded differentiable exact-DRC restoration candidate.

    The exact opening audit supplies a *frozen active set* only.  Bad solid
    cells contribute ``+rho`` and bad void cells contribute ``-w*rho`` to a
    local sequential surrogate.  Its physical-density cotangent is propagated
    through the certified projection/filter VJP.  No binary opening, closing,
    threshold assignment, or per-pixel repair is applied to the returned
    latent design.

    Candidate selection is solver-free and fail closed: it requires a strict
    decrease in exact bad-cell total and at least a one-percent decrease in the
    fixed-cap smooth violation.  Among the smallest exact totals, the stored
    Maxwell/thermal adjoint selects the best predicted objective change.
    A fresh solver evaluation remains mandatory before acceptance.
    """

    mapping = mapping or ProductionDensityMapping()
    latent = np.asarray(latent, dtype=float)
    objective_gradient_A = np.asarray(objective_gradient_A, dtype=float)
    if latent.shape != mapping.shape or objective_gradient_A.shape != latent.shape:
        raise ValueError("latent/objective gradient shape does not match mapping")
    if not np.all(np.isfinite(objective_gradient_A)):
        raise ValueError("objective gradient contains NaN or Inf")

    current_metrics, current_arrays = design_metrics(latent, beta, mapping)
    current_exact = current_metrics["exact_binary_audit"]
    current_total = int(
        current_exact["solid_bad_cell_count"]
        + current_exact["void_bad_cell_count"]
    )
    if current_total == 0:
        raise RuntimeError("exact active-set restoration requested with zero violations")
    caps = stage_caps(beta)
    current_values = np.asarray(
        [current_metrics["solid_constraint"], current_metrics["void_constraint"]],
        dtype=float,
    )
    current_violation = normalized_violation(current_values, caps)
    bad_solid = np.asarray(current_arrays["bad_solid"], dtype=float)
    bad_void = np.asarray(current_arrays["bad_void"], dtype=float)

    # First screen every bounded candidate with only the exact audit.  The
    # relatively expensive differentiable disk constraint is evaluated only
    # for exact-count levels that may be admissible.
    by_total: dict[int, list[dict[str, object]]] = {}
    for void_weight in EXACT_ACTIVE_SET_PHASE_WEIGHTS:
        physical_cotangent = bad_solid - float(void_weight) * bad_void
        gradient = mapping.vjp(latent, physical_cotangent, beta)
        maximum = float(np.max(np.abs(gradient)))
        if not np.isfinite(maximum) or maximum <= 0.0:
            continue
        direction = gradient / maximum
        for move in EXACT_ACTIVE_SET_MOVES:
            candidate = np.clip(latent - float(move) * direction, 0.0, 1.0)
            candidate_rho = mapping.physical(candidate, beta)
            audit = exact_binary_audit(candidate_rho, mapping.spacing_m)
            total = int(
                audit["solid_bad_cell_count"] + audit["void_bad_cell_count"]
            )
            if total >= current_total:
                continue
            delta = candidate - latent
            by_total.setdefault(total, []).append({
                "candidate": candidate,
                "void_weight": float(void_weight),
                "move": float(move),
                "predicted_objective_delta_A": float(
                    np.sum(objective_gradient_A * delta)
                ),
                "exact_solid_bad_cells": int(audit["solid_bad_cell_count"]),
                "exact_void_bad_cells": int(audit["void_bad_cell_count"]),
                "latent_rms_change": float(np.sqrt(np.mean(delta * delta))),
                "latent_max_change": float(np.max(np.abs(delta))),
            })

    for exact_total in sorted(by_total):
        admissible: list[tuple[dict[str, object], dict[str, object]]] = []
        for candidate_record in by_total[exact_total]:
            metrics, _ = design_metrics(
                np.asarray(candidate_record["candidate"], dtype=float), beta, mapping
            )
            values = np.asarray(
                [metrics["solid_constraint"], metrics["void_constraint"]],
                dtype=float,
            )
            violation = normalized_violation(values, caps)
            if not (
                violation <= current_violation * (1.0 - 0.01)
                or bool(metrics["constraints_feasible"])
            ):
                continue
            candidate_record = dict(candidate_record)
            candidate_record.update({
                "candidate_smooth_values": values.tolist(),
                "candidate_normalized_violation": float(violation),
                "candidate_metrics": metrics,
            })
            admissible.append((candidate_record, metrics))
        if not admissible:
            continue
        admissible.sort(
            key=lambda item: (
                -float(item[0]["predicted_objective_delta_A"]),
                float(item[0]["candidate_normalized_violation"]),
                float(item[0]["move"]),
                float(item[0]["void_weight"]),
            )
        )
        selected, _ = admissible[0]
        candidate = np.asarray(selected.pop("candidate"), dtype=float)
        diagnostics = {
            "contract": EXACT_ACTIVE_SET_RESTORATION_CONTRACT,
            "posthoc_binary_repair": False,
            "current_exact_solid_bad_cells": int(
                current_exact["solid_bad_cell_count"]
            ),
            "current_exact_void_bad_cells": int(
                current_exact["void_bad_cell_count"]
            ),
            "current_exact_bad_total": current_total,
            "current_smooth_values": current_values.tolist(),
            "current_normalized_violation": float(current_violation),
            "screened_exact_reducing_candidate_count": int(
                sum(len(values) for values in by_total.values())
            ),
            "selected_exact_bad_total": int(exact_total),
            **selected,
        }
        return candidate, diagnostics

    raise RuntimeError(
        "no bounded exact-active-set candidate reduced both exact DRC and "
        "the smooth fixed-cap violation"
    )


def exact_sign_feasibility_restoration_candidate(
    latent: np.ndarray,
    beta: float,
    objective_gradient_A: np.ndarray,
    current_objective_A: float,
    mapping: ProductionDensityMapping | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve the exact audit's local sign constraints on the selected GPU.

    This fallback is used only when the one-step active-set directions cannot
    strictly decrease exact and smooth metrics together.  The idempotent exact
    opening supplies a target *sign* for the filtered field.  A continuous
    latent array is then optimized on the configured CUDA device so that
    ``H x`` lies on the requested side of 0.5 with a small margin.  The target
    binary is never returned or assigned to the design.
    """

    mapping = mapping or ProductionDensityMapping()
    latent = np.asarray(latent, dtype=float)
    objective_gradient_A = np.asarray(objective_gradient_A, dtype=float)
    if latent.shape != mapping.shape or objective_gradient_A.shape != latent.shape:
        raise ValueError("latent/objective gradient shape does not match mapping")
    objective = float(current_objective_A)
    if not np.isfinite(objective) or objective <= 0.0:
        raise ValueError("current objective must be finite and positive")

    current_metrics, current_arrays = design_metrics(latent, beta, mapping)
    binary = np.asarray(current_arrays["binary"], dtype=bool)
    target = (
        (binary & ~np.asarray(current_arrays["bad_solid"], dtype=bool))
        | np.asarray(current_arrays["bad_void"], dtype=bool)
    )
    target_audit = exact_binary_audit(target.astype(float), mapping.spacing_m)
    if (
        target_audit["solid_bad_cell_count"] != 0
        or target_audit["void_bad_cell_count"] != 0
    ):
        raise RuntimeError(
            "one exact opening/void-fill target is not idempotent; refusing "
            "an unbounded morphology-repair sequence"
        )

    device = _constraint_device()
    if device.type != "cuda":
        raise RuntimeError("exact sign-feasibility restoration requires CUDA")
    lower_np = np.maximum(0.0, latent - 0.50)
    upper_np = np.minimum(1.0, latent + 0.50)
    value = torch.tensor(latent, dtype=torch.float64, device=device)
    lower = torch.tensor(lower_np, dtype=torch.float64, device=device)
    upper = torch.tensor(upper_np, dtype=torch.float64, device=device)
    sign = torch.tensor(
        2.0 * target.astype(float) - 1.0,
        dtype=torch.float64,
        device=device,
    )
    kernel = torch.tensor(
        mapping.kernel,
        dtype=torch.float64,
        device=device,
    )[None, None, :, :]
    normalization = torch.tensor(
        mapping.normalization,
        dtype=torch.float64,
        device=device,
    )
    padding = (mapping.kernel.shape[0] // 2, mapping.kernel.shape[1] // 2)
    margin = 2.0e-3
    step = 1.0e-2
    maximum_iterations = 1000
    audit_interval = 10
    iterations = 0
    mismatch = int(binary.size)
    final_audit: dict[str, object] | None = None
    for iterations in range(1, maximum_iterations + 1):
        value.requires_grad_(True)
        filtered = torch.nn.functional.conv2d(
            value[None, None, :, :], kernel, padding=padding
        )[0, 0] / normalization
        violation = torch.relu(margin - sign * (filtered - 0.5))
        loss = 0.5 * torch.sum(violation * violation)
        gradient = torch.autograd.grad(loss, value)[0]
        maximum = torch.max(torch.abs(gradient))
        if not bool(torch.isfinite(maximum)) or float(maximum.detach().cpu()) <= 0.0:
            raise RuntimeError("exact sign-feasibility gradient vanished")
        with torch.no_grad():
            value = torch.clamp(value - step * gradient / maximum, lower, upper)
        if iterations % audit_interval:
            continue
        candidate = value.detach().cpu().numpy()
        filtered_np = mapping.filtered(candidate)
        mismatch = int(np.count_nonzero((filtered_np >= 0.5) != target))
        final_audit = exact_binary_audit(
            mapping.physical(candidate, beta), mapping.spacing_m
        )
        if (
            mismatch == 0
            and final_audit["solid_bad_cell_count"] == 0
            and final_audit["void_bad_cell_count"] == 0
        ):
            break
    else:
        raise RuntimeError(
            "CUDA exact sign-feasibility solve did not reach the target "
            f"within {maximum_iterations} iterations"
        )

    candidate = value.detach().cpu().numpy()
    candidate_metrics, _ = design_metrics(candidate, beta, mapping)
    current_values = np.asarray(
        [current_metrics["solid_constraint"], current_metrics["void_constraint"]]
    )
    candidate_values = np.asarray([
        candidate_metrics["solid_constraint"],
        candidate_metrics["void_constraint"],
    ])
    caps = stage_caps(beta)
    current_violation = normalized_violation(current_values, caps)
    candidate_violation = normalized_violation(candidate_values, caps)
    if not (
        candidate_violation <= current_violation * (1.0 - 0.01)
        or bool(candidate_metrics["constraints_feasible"])
    ):
        raise RuntimeError(
            "exact sign-feasibility candidate did not improve the smooth gate"
        )
    delta = candidate - latent
    predicted_delta = float(np.sum(objective_gradient_A * delta))
    predicted_ratio = predicted_delta / objective
    if predicted_ratio < -2.0e-3:
        raise RuntimeError(
            "exact sign-feasibility candidate exceeds the 0.2% linearized "
            "FOM-loss guard"
        )
    diagnostics = {
        "contract": EXACT_ACTIVE_SET_RESTORATION_CONTRACT,
        "submethod": "cuda_filtered_sign_feasibility_projection",
        "posthoc_binary_repair": False,
        "target_binary_was_assigned_to_design": False,
        "target_changed_cell_count": int(np.count_nonzero(target != binary)),
        "iterations": int(iterations),
        "filtered_sign_margin": margin,
        "normalized_gradient_step": step,
        "latent_trust_bound": 0.50,
        "final_target_sign_mismatch_count": mismatch,
        "current_exact_bad_total": int(
            current_metrics["exact_binary_audit"]["solid_bad_cell_count"]
            + current_metrics["exact_binary_audit"]["void_bad_cell_count"]
        ),
        "selected_exact_bad_total": 0,
        "exact_solid_bad_cells": 0,
        "exact_void_bad_cells": 0,
        "current_smooth_values": current_values.tolist(),
        "candidate_smooth_values": candidate_values.tolist(),
        "current_normalized_violation": float(current_violation),
        "candidate_normalized_violation": float(candidate_violation),
        "predicted_objective_delta_A": predicted_delta,
        "predicted_objective_relative_change": predicted_ratio,
        "latent_rms_change": float(np.sqrt(np.mean(delta * delta))),
        "latent_max_change": float(np.max(np.abs(delta))),
        "candidate_metrics": candidate_metrics,
    }
    return candidate, diagnostics


def projected_binary_gate(metrics: dict[str, object]) -> bool:
    exact = metrics["exact_binary_audit"]
    return bool(
        metrics["gray_fraction_0p01_0p99"] < 0.001
        and metrics["binarization_metric_mean_4rho1mrho"] < 0.001
        and exact["solid_bad_cell_count"] == 0
        and exact["void_bad_cell_count"] == 0
    )


def transient_license_failure(
    payload: dict[str, object], diagnostic_text: str = ""
) -> bool:
    """Recognize explicit external license checkout/startup failures only.

    Lumerical sometimes replaces the concrete FlexNet failure in its compact
    JSON-facing exception with a generic resource-name error.  The concrete
    checkout failure remains in the engine ``*_p0.log``.  Callers may supply
    that solver-log text here; generic resource errors alone remain
    non-retryable.
    """

    if payload.get("passed"):
        return False
    message = (json.dumps(payload, default=str) + "\n" + diagnostic_text).lower()
    signatures = (
        "unable to checkout the requested hpc license",
        "requires 9 licenses for feature fdtd_solutions_engine",
        "licensed number of users already reached",
        "flexnet licensing error:-4,132",
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
    smooth_trust_relative: float = 5.0e-3,
) -> dict[str, object]:
    current_v = normalized_violation(current_constraints, caps)
    candidate_v = normalized_violation(candidate_constraints, caps)
    objective_ratio = float(candidate_objective / current_objective)
    if smooth_trust_relative < 0.0:
        raise ValueError("smooth_trust_relative must be nonnegative")
    current_feasible = bool(np.all(
        np.asarray(current_constraints) <= np.asarray(caps) * (1.0 + smooth_trust_relative)
    ))
    candidate_feasible = bool(np.all(
        np.asarray(candidate_constraints) <= np.asarray(caps) * (1.0 + smooth_trust_relative)
    ))
    restoration = bool(
        current_v > 0.0
        and candidate_v <= current_v * (1.0 - 0.01)
    )
    accepted = bool(
        (candidate_feasible or restoration)
        and objective_ratio >= FEASIBLE_FOM_RETENTION
    )
    reason = (
        "satisfy the smooth caps, or reduce an incoming infeasible smooth "
        "violation by at least 1%, while limiting actual FOM loss to 0.2%"
    )
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
        "smooth_restoration_step": bool(restoration),
        "current_normalized_violation": current_v,
        "candidate_normalized_violation": candidate_v,
        "smooth_trust_relative": float(smooth_trust_relative),
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
        if float(row["beta"]) == float(beta)
        and row["role"] in ("accepted_mma", "accepted_drc_restoration")
    ]
    minimum = 8 if beta <= 2.0 else 6
    window = 4
    if len(accepted) < minimum or len(accepted) < window:
        return StageConvergence(False, f"need at least {minimum} accepted updates", np.inf, np.inf, np.inf)
    recent = accepted[-window:]
    if float(beta) >= 128.0 and (
        int(recent[-1].get("solid_bad_cells", 0))
        + int(recent[-1].get("void_bad_cells", 0))
    ) != 0:
        return StageConvergence(
            False,
            "exact 500 nm violations remain in the final restoration epoch",
            np.inf,
            np.inf,
            np.inf,
        )
    if not bool(recent[-1]["constraints_feasible"]):
        return StageConvergence(False, "fixed solid/void inequalities are not feasible", np.inf, np.inf, np.inf)
    rel = [abs(float(row["relative_fom_change"])) for row in recent]
    rms = [float(row["rho_rms_change"]) for row in recent]
    maximum = [float(row["rho_max_change"]) for row in recent]
    gates = (max(rel) < 0.005, max(rms) < 0.005, max(maximum) < 0.03)
    return StageConvergence(
        bool(all(gates)),
        "all four-update FOM/density plateau gates pass" if all(gates) else "recent FOM/density changes have not plateaued",
        max(rel), max(rms), max(maximum),
    )


def low_beta_morphology_limited_transition(
    history: list[dict[str, object]],
    beta: float,
    accepted_moves: list[float],
) -> bool:
    """End beta=2 when its minimum safe step worsens exact morphology."""

    if float(beta) != 2.0:
        return False
    accepted = [
        row for row in history
        if float(row["beta"]) == 2.0 and row["role"] == "accepted_mma"
        and int(row.get("global_iteration", -1))
        > MMA_TOPOLOGY_RESET_GLOBAL_ITERATION
    ]
    if len(accepted) < 4 or len(accepted_moves) != len(accepted):
        return False
    previous_move, latest_move = map(float, accepted_moves[-2:])
    previous_exact = (
        int(accepted[-2]["solid_bad_cells"])
        + int(accepted[-2]["void_bad_cells"])
    )
    latest_exact = (
        int(accepted[-1]["solid_bad_cells"])
        + int(accepted[-1]["void_bad_cells"])
    )
    return bool(
        previous_move <= 0.005 + 1.0e-15
        and latest_move <= MINIMUM_MMA_MOVE + 1.0e-15
        and latest_exact > previous_exact
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
            <= MMA_TOPOLOGY_RESET_GLOBAL_ITERATION
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
    "DISK_CONSTRAINT_START_BETA", "MMA_TOPOLOGY_RESET_GLOBAL_ITERATION",
    "MMAState", "ProductionDensityMapping", "adaptive_move_ceiling", "candidate_acceptance",
    "calibrate_stage_caps", "constraint_contract", "constraint_values_and_gradients",
    "disk_constraint_values_and_gradients", "design_metrics", "exact_binary_audit",
    "catastrophic_exact_growth",
    "initialize_mma_state", "load_mma_state", "mma_step",
    "mma_effective_constraint_caps",
    "projected_binary_gate", "save_mma_state", "stage_caps",
    "smooth_feasibility_move_retries",
    "stage_convergence", "low_beta_morphology_limited_transition",
    "transient_license_failure",
    "two_consecutive_subthreshold_moves",
]
