"""Finite-window minimum-feature Au solid/void constraints and exact audit.

The design outside the centered 8 um x 8 um window is void.  The production
optimizer uses a finite conic filter, tanh projection, and differentiable
solid/void opening residuals.  Promotion uses the separate thresholded audit;
the exact audit is never differentiated or silently relabelled as a smooth
constraint.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
import torch

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.legacy_v261_optical_support.production_density_mapping import (
    ProductionDensityMapping,
)


MAPPING = ProductionDensityMapping(
    shape=CONTRACT.design_shape,
    spacing_m=CONTRACT.design_pitch_m,
    radius_m=CONTRACT.filter_radius_m,
    eta=CONTRACT.projection_eta,
)
DEFAULT_OPENING_TAU = 0.02
DEFAULT_POSITIVE_PART_TAU = 0.01
DEFAULT_KS_ALPHA = 48.0


def physical_disk_footprint(radius_m: float, spacing_m: float) -> np.ndarray:
    if radius_m <= 0.0 or spacing_m <= 0.0:
        raise ValueError("radius and spacing must be positive")
    extent = int(np.ceil(radius_m / spacing_m))
    offset = np.arange(-extent, extent + 1, dtype=float) * spacing_m
    xx, yy = np.meshgrid(offset, offset, indexing="ij")
    result = np.hypot(xx, yy) <= radius_m + 1.0e-15 * spacing_m
    if not result[extent, extent]:
        raise RuntimeError("morphology footprint lost its origin")
    return result


def exact_500nm_audit(
    physical_density: np.ndarray,
    spacing_m: float = 100.0e-9,
    minimum_feature_m: float = CONTRACT.minimum_solid_feature_m,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Audit thresholded phases using half the requested feature as radius.

    Outside the finite design window is void.  The test is an exact discrete
    audit on the chosen 100 nm grid; it is not used as a differentiable
    optimizer constraint and no geometry is silently repaired here.
    """

    rho = np.asarray(physical_density, dtype=float)
    if rho.ndim != 2 or not np.all(np.isfinite(rho)):
        raise ValueError("physical density must be a finite 2-D array")
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or not 0.0 < threshold_value < 1.0:
        raise ValueError("binary threshold must lie strictly inside (0,1)")
    binary = rho >= threshold_value
    footprint = physical_disk_footprint(0.5 * minimum_feature_m, spacing_m)
    solid_open = ndimage.binary_opening(binary, structure=footprint, border_value=0)
    void_open = ndimage.binary_opening(~binary, structure=footprint, border_value=1)
    bad_solid = binary & ~solid_open
    bad_void = (~binary) & ~void_open
    return {
        "minimum_feature_nm": minimum_feature_m * 1.0e9,
        "opening_radius_nm": 0.5 * minimum_feature_m * 1.0e9,
        "spacing_nm": spacing_m * 1.0e9,
        "threshold": threshold_value,
        "footprint_pixel_count": int(np.count_nonzero(footprint)),
        "solid_bad_cell_count": int(np.count_nonzero(bad_solid)),
        "void_bad_cell_count": int(np.count_nonzero(bad_void)),
        "solid_pass": bool(not np.any(bad_solid)),
        "void_pass": bool(not np.any(bad_void)),
        "binary": binary,
        "bad_solid": bad_solid,
        "bad_void": bad_void,
    }


def _offsets(
    spacing_m: float = CONTRACT.design_pitch_m,
    minimum_feature_m: float = CONTRACT.minimum_solid_feature_m,
) -> tuple[tuple[int, int], ...]:
    footprint = physical_disk_footprint(0.5 * minimum_feature_m, spacing_m)
    center = tuple(value // 2 for value in footprint.shape)
    return tuple(
        (int(i - center[0]), int(j - center[1]))
        for i, j in np.argwhere(footprint)
    )


def _shift_stack(value: torch.Tensor, border: float) -> torch.Tensor:
    return _shift_stack_for_feature(
        value,
        border,
        spacing_m=CONTRACT.design_pitch_m,
        minimum_feature_m=CONTRACT.minimum_solid_feature_m,
    )


def _shift_stack_for_feature(
    value: torch.Tensor,
    border: float,
    *,
    spacing_m: float,
    minimum_feature_m: float,
) -> torch.Tensor:
    offsets = _offsets(
        spacing_m=spacing_m,
        minimum_feature_m=minimum_feature_m,
    )
    extent = max(max(abs(i), abs(j)) for i, j in offsets)
    padded = torch.nn.functional.pad(
        value, (extent, extent, extent, extent), value=float(border)
    )
    nx, ny = value.shape
    return torch.stack(
        [
            padded[
                extent + i : extent + i + nx,
                extent + j : extent + j + ny,
            ]
            for i, j in offsets
        ]
    )


def _soft_open(
    value: torch.Tensor,
    *,
    outside_phase: float,
    tau: float,
    spacing_m: float = CONTRACT.design_pitch_m,
    minimum_feature_m: float = CONTRACT.minimum_solid_feature_m,
) -> torch.Tensor:
    """Smooth disk opening with the physical phase outside the finite window."""

    offsets = _offsets(
        spacing_m=spacing_m,
        minimum_feature_m=minimum_feature_m,
    )
    count_log = float(np.log(len(offsets)))
    shifted = _shift_stack_for_feature(
        value,
        outside_phase,
        spacing_m=spacing_m,
        minimum_feature_m=minimum_feature_m,
    )
    eroded = -tau * (torch.logsumexp(-shifted / tau, dim=0) - count_log)
    shifted_eroded = _shift_stack_for_feature(
        eroded,
        outside_phase,
        spacing_m=spacing_m,
        minimum_feature_m=minimum_feature_m,
    )
    return tau * (torch.logsumexp(shifted_eroded / tau, dim=0) - count_log)


def smooth_500nm_physical_constraints(
    physical_density: np.ndarray,
    *,
    spacing_m: float = CONTRACT.design_pitch_m,
    minimum_solid_feature_m: float = CONTRACT.minimum_solid_feature_m,
    minimum_void_feature_m: float = CONTRACT.minimum_void_feature_m,
    tau: float = DEFAULT_OPENING_TAU,
    positive_tau: float = DEFAULT_POSITIVE_PART_TAU,
    ks_alpha: float = DEFAULT_KS_ALPHA,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return solid/void opening residuals and physical-cell gradients.

    Values are KS aggregates of ``relu(phase - opening(phase))``.  The solid
    phase sees void outside the design window; the void phase sees void
    outside. The positive part is a softplus approximation whose pointwise
    excess over ``max(0,x)`` is bounded by ``positive_tau*log(2)``. This
    matches the exact finite-window audit and is independent of the upstream
    optical density carrier.
    """

    rho_array = np.asarray(physical_density, dtype=np.float64)
    if rho_array.shape != CONTRACT.design_shape or not np.all(
        np.isfinite(rho_array)
    ):
        raise ValueError(
            f"physical cell density must be finite with shape {CONTRACT.design_shape}"
        )
    if (
        spacing_m <= 0.0
        or minimum_solid_feature_m <= 0.0
        or minimum_void_feature_m <= 0.0
        or tau <= 0.0
        or positive_tau <= 0.0
        or ks_alpha <= 0.0
    ):
        raise ValueError("DFM length scales and smoothing parameters must be positive")
    rho = torch.tensor(rho_array, dtype=torch.float64, requires_grad=True)
    values: list[float] = []
    gradients: list[np.ndarray] = []
    fields: dict[str, np.ndarray] = {}
    phases = (
        ("solid", rho, 0.0, minimum_solid_feature_m),
        ("void", 1.0 - rho, 1.0, minimum_void_feature_m),
    )
    for index, (name, phase, outside, minimum_feature_m) in enumerate(phases):
        opened = _soft_open(
            phase,
            outside_phase=outside,
            tau=float(tau),
            spacing_m=float(spacing_m),
            minimum_feature_m=float(minimum_feature_m),
        )
        raw_residual = phase - opened
        residual = float(positive_tau) * torch.nn.functional.softplus(
            raw_residual / float(positive_tau)
        )
        flattened = residual.reshape(-1)
        aggregate = (
            torch.logsumexp(float(ks_alpha) * flattened, dim=0)
            - np.log(flattened.numel())
        ) / float(ks_alpha)
        gradient_rho = torch.autograd.grad(
            aggregate, rho, retain_graph=index == 0
        )[0]
        values.append(float(aggregate.detach()))
        gradients.append(gradient_rho.detach().cpu().numpy())
        fields[f"{name}_opening"] = opened.detach().cpu().numpy()
        fields[f"{name}_raw_residual"] = raw_residual.detach().cpu().numpy()
        fields[f"{name}_residual"] = residual.detach().cpu().numpy()
    return np.asarray(values), np.stack(gradients), fields


def smooth_500nm_constraints(
    latent: np.ndarray,
    beta: float,
    *,
    tau: float = DEFAULT_OPENING_TAU,
    positive_tau: float = DEFAULT_POSITIVE_PART_TAU,
    ks_alpha: float = DEFAULT_KS_ALPHA,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return legacy 80x80-cell constraint values and latent gradients."""

    latent_array = np.asarray(latent, dtype=np.float64)
    if latent_array.shape != CONTRACT.design_shape:
        raise ValueError(f"latent density must have shape {CONTRACT.design_shape}")
    rho = MAPPING.physical(latent_array, float(beta))
    values, physical_gradients, fields = smooth_500nm_physical_constraints(
        rho,
        tau=tau,
        positive_tau=positive_tau,
        ks_alpha=ks_alpha,
    )
    latent_gradients = np.stack(
        [
            MAPPING.vjp(latent_array, gradient, float(beta))
            for gradient in physical_gradients
        ]
    )
    return values, latent_gradients, fields


def density_metrics(latent: np.ndarray, beta: float) -> dict[str, object]:
    latent_array = np.asarray(latent, dtype=np.float64)
    filtered = MAPPING.filtered(latent_array)
    rho = MAPPING.physical(latent_array, float(beta))
    audit = exact_500nm_audit(rho)
    return {
        "beta": float(beta),
        "latent_min": float(np.min(latent_array)),
        "latent_max": float(np.max(latent_array)),
        "filtered_min": float(np.min(filtered)),
        "filtered_max": float(np.max(filtered)),
        "rho_min": float(np.min(rho)),
        "rho_max": float(np.max(rho)),
        "rho_mean": float(np.mean(rho)),
        "gray_fraction_0p01_0p99": float(
            np.mean((rho > 0.01) & (rho < 0.99))
        ),
        "binarization_mean_4rho1mrho": float(
            np.mean(4.0 * rho * (1.0 - rho))
        ),
        "solid_bad_cell_count": int(audit["solid_bad_cell_count"]),
        "void_bad_cell_count": int(audit["void_bad_cell_count"]),
        "exact_bad_cell_count": int(
            audit["solid_bad_cell_count"] + audit["void_bad_cell_count"]
        ),
        "exact_minimum_feature_pass": bool(
            audit["solid_pass"] and audit["void_pass"]
        ),
        "exact_500nm_pass": bool(audit["solid_pass"] and audit["void_pass"]),
    }
