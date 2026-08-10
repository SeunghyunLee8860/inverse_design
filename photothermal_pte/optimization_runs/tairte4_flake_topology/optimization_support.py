"""Filter, projection, and 500 nm morphology tools for flake topology."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
import torch

from photothermal_pte.optimization_runs.run_002_gaussian10_w8p5_current_max.production_density_mapping import (
    ProductionDensityMapping,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT


SHAPE = CONTRACT.design_node_shape
SPACING_M = CONTRACT.design_step_m
MINIMUM_FEATURE_M = CONTRACT.minimum_feature_m
OPENING_RADIUS_M = 0.5 * MINIMUM_FEATURE_M
MAPPING = ProductionDensityMapping(
    shape=SHAPE,
    spacing_m=SPACING_M,
    # The projection filter is a regularizer, not the minimum-feature
    # certificate.  A 500 nm filter radius over-smoothed the old Run012/013
    # topology before the morphology audit became active.  Use the physical
    # feature radius here and certify both phases separately below.
    radius_m=OPENING_RADIUS_M,
    eta=0.5,
)


def disk() -> np.ndarray:
    # A design node represents a 100 nm square support.  Two node-centre
    # offsets plus the two half-cell extents give a 500 nm physical diameter.
    # ceil(250/100)=3 used by Run012/013 instead produced a conservative
    # roughly 600--700 nm audit and was stricter than the requested contract.
    radius = int(np.floor(OPENING_RADIUS_M / SPACING_M + 1.0e-12))
    axis = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return xx * xx + yy * yy <= radius * radius


def exact_binary_audit(rho: np.ndarray) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Audit both phases with the actual material adjoining each design edge."""

    binary = np.asarray(rho, dtype=float) >= 0.5
    structure = disk()
    radius = (structure.shape[0] - 1) // 2
    if CONTRACT.geometry_mode == "contact_anchored":
        solid_padded = np.zeros((binary.shape[0] + 2 * radius, binary.shape[1] + 2 * radius), dtype=bool)
        solid_padded[radius:-radius, radius:-radius] = binary
        solid_padded[:, :radius] = True
        solid_padded[:, -radius:] = True
        solid_open = ndimage.binary_opening(solid_padded, structure=structure)[radius:-radius, radius:-radius]
        void_padded = ~solid_padded
        void_open = ndimage.binary_opening(void_padded, structure=structure)[radius:-radius, radius:-radius]
        outside_phase = "fixed_solid_at_top_bottom_and_void_at_left_right"
    else:
        solid_open = ndimage.binary_opening(binary, structure=structure, border_value=1)
        void_open = ndimage.binary_opening(~binary, structure=structure, border_value=0)
        outside_phase = "fixed_solid_TaIrTe4_frame"
    bad_solid = binary & ~solid_open
    bad_void = (~binary) & ~void_open
    return {
        "minimum_feature_nm": 500.0,
        "opening_radius_nm": 250.0,
        "opening_radius_pixels": int((structure.shape[0] - 1) // 2),
        "realized_discrete_opening_max_center_offset_nm": float(radius * SPACING_M * 1.0e9),
        "realized_discrete_opening_pixel_support_diameter_nm": float((2 * radius + 1) * SPACING_M * 1.0e9),
        "discretization_note": (
            "requested 500 nm pixel-support diameter is represented by five "
            "100 nm samples (two centre offsets plus two half-cell extents)"
        ),
        "outside_design_phase": outside_phase,
        "solid_bad_cell_count": int(np.count_nonzero(bad_solid)),
        "void_bad_cell_count": int(np.count_nonzero(bad_void)),
        "total_bad_cell_count": int(np.count_nonzero(bad_solid) + np.count_nonzero(bad_void)),
        "counted_entity": "design nodes (legacy field names retain *_cell_count)",
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
    if CONTRACT.geometry_mode == "contact_anchored":
        # Array axis 0 is Lumerical x and axis 1 is y.  The design meets fixed
        # TaIrTe4 contacts across the y-min/y-max edges and void across x edges.
        solid_phase = border == 1.0
        padded = torch.full(
            (value.shape[0] + 2 * radius, value.shape[1] + 2 * radius),
            0.0 if solid_phase else 1.0,
            dtype=value.dtype,
            device=value.device,
        )
        padded[radius:-radius, radius:-radius] = value
        padded[:, :radius] = 1.0 if solid_phase else 0.0
        padded[:, -radius:] = 1.0 if solid_phase else 0.0
    else:
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
