"""Filter, projection, and 500 nm morphology tools for flake topology."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
import torch

from photothermal_pte.optimization_runs.run_002_gaussian10_w8p5_current_max.production_density_mapping import (
    ProductionDensityMapping,
)


SHAPE = (161, 161)
SPACING_M = 100.0e-9
MINIMUM_FEATURE_M = 500.0e-9
OPENING_RADIUS_M = 0.5 * MINIMUM_FEATURE_M
MAPPING = ProductionDensityMapping(
    shape=SHAPE,
    spacing_m=SPACING_M,
    radius_m=MINIMUM_FEATURE_M,
    eta=0.5,
)


def disk() -> np.ndarray:
    radius = int(np.ceil(OPENING_RADIUS_M / SPACING_M - 1.0e-12))
    axis = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return xx * xx + yy * yy <= radius * radius


def exact_binary_audit(rho: np.ndarray) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Audit both phases with fixed-solid TaIrTe4 outside the design window."""

    binary = np.asarray(rho, dtype=float) >= 0.5
    structure = disk()
    solid_open = ndimage.binary_opening(binary, structure=structure, border_value=1)
    void_open = ndimage.binary_opening(~binary, structure=structure, border_value=0)
    bad_solid = binary & ~solid_open
    bad_void = (~binary) & ~void_open
    return {
        "minimum_feature_nm": 500.0,
        "opening_radius_nm": 250.0,
        "opening_radius_pixels": int((structure.shape[0] - 1) // 2),
        "outside_design_phase": "fixed_solid_TaIrTe4_frame",
        "solid_bad_cell_count": int(np.count_nonzero(bad_solid)),
        "void_bad_cell_count": int(np.count_nonzero(bad_void)),
        "total_bad_cell_count": int(np.count_nonzero(bad_solid) + np.count_nonzero(bad_void)),
        "solid_fraction": float(np.mean(binary)),
        "passed": bool(not np.any(bad_solid) and not np.any(bad_void)),
    }, {"binary": binary, "bad_solid": bad_solid, "bad_void": bad_void}


def _projection(filtered: torch.Tensor, beta: float) -> torch.Tensor:
    denominator = 2.0 * np.tanh(0.5 * beta)
    return (np.tanh(0.5 * beta) + torch.tanh(beta * (filtered - 0.5))) / denominator


def _offsets() -> tuple[tuple[int, int], ...]:
    structure = disk()
    radius = (structure.shape[0] - 1) // 2
    return tuple(
        (i - radius, j - radius)
        for i, j in np.argwhere(structure)
    )


def _shifted(value: torch.Tensor, border: float) -> torch.Tensor:
    offsets = _offsets()
    radius = max(max(abs(i), abs(j)) for i, j in offsets)
    padded = torch.nn.functional.pad(value, (radius, radius, radius, radius), value=border)
    nx, ny = value.shape
    return torch.stack(
        [padded[radius+i:radius+i+nx, radius+j:radius+j+ny] for i, j in offsets]
    )


def _soft_open(value: torch.Tensor, border: float, tau: float = 1.0e-3) -> torch.Tensor:
    count = float(np.log(len(_offsets())))
    eroded = -tau * (torch.logsumexp(-_shifted(value, border) / tau, dim=0) - count)
    return tau * (torch.logsumexp(_shifted(eroded, border) / tau, dim=0) - count)


def morphology_values_gradients(
    latent: np.ndarray,
    beta: float,
    *,
    device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Differentiable solid/void opening residuals with the exact border phase."""

    latent = np.asarray(latent, dtype=np.float64)
    rho_np = MAPPING.physical(latent, beta)
    rho = torch.tensor(rho_np, dtype=torch.float64, device=device, requires_grad=True)
    sharpened = torch.sigmoid(64.0 * (rho - 0.5))
    values = []
    gradients = []
    fields = {}
    for index, (name, phase, border) in enumerate(
        (("solid", sharpened, 1.0), ("void", 1.0 - sharpened, 0.0))
    ):
        residual = torch.relu(phase - _soft_open(phase, border))
        aggregate = torch.mean(residual)
        gradient_rho = torch.autograd.grad(aggregate, rho, retain_graph=index == 0)[0]
        values.append(float(aggregate.detach().cpu()))
        gradients.append(MAPPING.vjp(latent, gradient_rho.detach().cpu().numpy(), beta))
        fields[f"{name}_residual"] = residual.detach().cpu().numpy()
    return np.asarray(values), np.stack(gradients), fields


def metrics(latent: np.ndarray, beta: float, *, device: str = "cuda:0") -> tuple[dict, dict]:
    latent = np.asarray(latent, dtype=np.float64)
    filtered = MAPPING.filtered(latent)
    rho = MAPPING.physical(latent, beta)
    constraint_values, constraint_gradients, fields = morphology_values_gradients(
        latent, beta, device=device
    )
    exact, exact_arrays = exact_binary_audit(rho)
    summary = {
        "beta": float(beta),
        "latent_range": [float(np.min(latent)), float(np.max(latent))],
        "filtered_range": [float(np.min(filtered)), float(np.max(filtered))],
        "rho_range": [float(np.min(rho)), float(np.max(rho))],
        "rho_mean": float(np.mean(rho)),
        "gray_fraction_0p01_0p99": float(np.mean((rho > 0.01) & (rho < 0.99))),
        "binarization_mean_4rho1mrho": float(np.mean(4.0 * rho * (1.0 - rho))),
        "smooth_solid_constraint": float(constraint_values[0]),
        "smooth_void_constraint": float(constraint_values[1]),
        "exact": exact,
    }
    arrays = {
        "latent": latent,
        "filtered": filtered,
        "rho": rho,
        "constraint_gradients": constraint_gradients,
        **fields,
        **exact_arrays,
    }
    return summary, arrays
