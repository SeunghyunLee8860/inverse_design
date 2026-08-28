"""Latent-to-projected design map for the Lumerical 4-um Au topology.

The optimization carrier is nodal: 81x81 bounded latent variables are
filtered and projected on the exact coordinates consumed by ``importnk2``.
The resulting projected occupancy is the one shared state. Custom PDE and DFM
cell fields are derived only through the committed four-node cell average.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.dfm import (
    DEFAULT_KS_ALPHA,
    DEFAULT_OPENING_TAU,
    DEFAULT_POSITIVE_PART_TAU,
    exact_500nm_audit,
    physical_disk_footprint,
    smooth_500nm_physical_constraints,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (
    canonical_density_nodes,
    density_state_audit,
    nodal_to_cell_average,
    nodal_to_cell_jvp,
    nodal_to_cell_vjp,
)


LUMERICAL_MINIMUM_SOLID_FEATURE_M = 250.0e-9
LUMERICAL_MINIMUM_VOID_FEATURE_M = 250.0e-9
LUMERICAL_FILTER_RADIUS_M = 250.0e-9
LUMERICAL_DFM_CALIBRATION_MARGIN = 1.0e-4
RELAXED_PROJECTED_DENSITY_FLOOR = 3.0e-5


def _projection(value: np.ndarray, *, beta: float, eta: float) -> np.ndarray:
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return (
        np.tanh(beta * eta) + np.tanh(beta * (value - eta))
    ) / denominator


def _projection_derivative(
    value: np.ndarray, *, beta: float, eta: float
) -> np.ndarray:
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (value - eta)) ** 2) / denominator


def _validated_beta(beta: float) -> float:
    value = float(beta)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("projection beta must be finite and positive")
    return value


@dataclass(frozen=True)
class LumericalNodalDesignMapping:
    """Finite conic filter and tanh projection with an exact transpose."""

    shape: tuple[int, int] = CONTRACT.design_node_shape
    spacing_m: float = CONTRACT.design_pitch_m
    radius_m: float = CONTRACT.filter_radius_m
    eta: float = CONTRACT.projection_eta
    minimum_solid_feature_m: float = CONTRACT.minimum_solid_feature_m
    minimum_void_feature_m: float = CONTRACT.minimum_void_feature_m
    projected_density_floor: float = 0.0

    def __post_init__(self) -> None:
        if self.shape != CONTRACT.design_node_shape:
            raise ValueError(
                "Lumerical optimizer mapping must use the canonical 81x81 nodes"
            )
        if self.spacing_m <= 0.0 or self.radius_m <= 0.0:
            raise ValueError("spacing and radius must be positive")
        if not 0.0 < self.eta < 1.0:
            raise ValueError("projection eta must lie strictly inside (0,1)")
        if not 0.0 <= self.projected_density_floor < 0.5:
            raise ValueError("projected density floor must lie in [0,0.5)")
        extent = int(np.ceil(self.radius_m / self.spacing_m))
        offsets = np.arange(-extent, extent + 1, dtype=np.float64) * self.spacing_m
        xx, yy = np.meshgrid(offsets, offsets, indexing="ij")
        distance = np.hypot(xx, yy)
        kernel = np.maximum(0.0, self.radius_m - distance)
        kernel[distance >= self.radius_m] = 0.0
        normalization = ndimage.convolve(
            np.ones(self.shape, dtype=np.float64),
            kernel,
            mode="constant",
            cval=0.0,
        )
        if not np.all(normalization > 0.0):
            raise RuntimeError("finite nodal filter has an empty row")
        object.__setattr__(self, "kernel", np.ascontiguousarray(kernel))
        object.__setattr__(
            self, "normalization", np.ascontiguousarray(normalization)
        )

    def _array(self, value: np.ndarray, *, label: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != self.shape or not np.all(np.isfinite(result)):
            raise ValueError(f"{label} must be finite with shape {self.shape}")
        return result

    def _convolve(self, value: np.ndarray) -> np.ndarray:
        return ndimage.convolve(value, self.kernel, mode="constant", cval=0.0)

    def filtered(self, latent: np.ndarray) -> np.ndarray:
        value = self._array(latent, label="latent density")
        return self._convolve(value) / self.normalization

    def filter_transpose(self, cotangent: np.ndarray) -> np.ndarray:
        value = self._array(cotangent, label="filter cotangent")
        return self._convolve(value / self.normalization)

    def physical(self, latent: np.ndarray, beta: float) -> np.ndarray:
        beta_value = _validated_beta(beta)
        projected = _projection(
            self.filtered(latent), beta=beta_value, eta=self.eta
        )
        projected = 0.5 + (1.0 - 2.0 * self.projected_density_floor) * (
            projected - 0.5
        )
        return canonical_density_nodes(projected)

    def jvp(
        self, latent: np.ndarray, direction: np.ndarray, beta: float
    ) -> np.ndarray:
        beta_value = _validated_beta(beta)
        filtered = self.filtered(latent)
        filtered_direction = self.filtered(
            self._array(direction, label="latent direction")
        )
        projection_tangent = _projection_derivative(
            filtered, beta=beta_value, eta=self.eta
        ) * filtered_direction
        return (1.0 - 2.0 * self.projected_density_floor) * projection_tangent

    def vjp(
        self, latent: np.ndarray, projected_cotangent: np.ndarray, beta: float
    ) -> np.ndarray:
        beta_value = _validated_beta(beta)
        filtered = self.filtered(latent)
        cotangent = self._array(
            projected_cotangent, label="projected-density cotangent"
        )
        return self.filter_transpose(
            cotangent
            * (1.0 - 2.0 * self.projected_density_floor)
            * _projection_derivative(filtered, beta=beta_value, eta=self.eta)
        )

    def audit(self) -> dict[str, Any]:
        constant_error = float(
            np.max(
                np.abs(
                    self.filtered(np.ones(self.shape, dtype=np.float64)) - 1.0
                )
            )
        )
        return {
            "schema": "lumerical-4um-nodal-filter-projection-v1",
            "latent_shape_xy": list(self.shape),
            "projected_shape_xy": list(self.shape),
            "downstream_cell_shape_xy": list(CONTRACT.design_shape),
            "spacing_nm": self.spacing_m * 1.0e9,
            "conic_filter_radius_nm": self.radius_m * 1.0e9,
            "minimum_solid_feature_nm": self.minimum_solid_feature_m * 1.0e9,
            "minimum_void_feature_nm": self.minimum_void_feature_m * 1.0e9,
            "projection_eta": self.eta,
            "relaxed_projected_density_bounds": [
                self.projected_density_floor,
                1.0 - self.projected_density_floor,
            ],
            "exact_zero_one_reserved_for_final_binary_reevaluation": bool(
                self.projected_density_floor > 0.0
            ),
            "kernel_nonzero_count": int(np.count_nonzero(self.kernel)),
            "constant_preservation_max_abs": constant_error,
            "finite_nonperiodic_boundary": True,
            "forward_filter": "D^-1 C with zero padding and truncated-row normalization",
            "transpose_filter": "C D^-1 with the same symmetric conic kernel",
            "shared_projected_density_is_nodal": True,
            "legacy_80x80_cell_mapping_is_optimizer_carrier": False,
            "optical_rho_power": None,
            "np_density_used": False,
        }


NOMINAL_MAPPING = LumericalNodalDesignMapping()
OPTIMIZER_250NM_MAPPING = LumericalNodalDesignMapping(
    radius_m=LUMERICAL_FILTER_RADIUS_M,
    minimum_solid_feature_m=LUMERICAL_MINIMUM_SOLID_FEATURE_M,
    minimum_void_feature_m=LUMERICAL_MINIMUM_VOID_FEATURE_M,
    projected_density_floor=RELAXED_PROJECTED_DENSITY_FLOOR,
)


def projected_cell_density(
    latent: np.ndarray,
    beta: float,
    *,
    mapping: LumericalNodalDesignMapping = OPTIMIZER_250NM_MAPPING,
) -> np.ndarray:
    """Return the only allowed custom-PDE/DFM cell density from latent rho."""

    return nodal_to_cell_average(mapping.physical(latent, beta))


def projected_cell_jvp(
    latent: np.ndarray,
    direction: np.ndarray,
    beta: float,
    *,
    mapping: LumericalNodalDesignMapping = OPTIMIZER_250NM_MAPPING,
) -> np.ndarray:
    return nodal_to_cell_jvp(mapping.jvp(latent, direction, beta))


def projected_cell_vjp(
    latent: np.ndarray,
    cell_cotangent: np.ndarray,
    beta: float,
    *,
    mapping: LumericalNodalDesignMapping = OPTIMIZER_250NM_MAPPING,
) -> np.ndarray:
    return mapping.vjp(latent, nodal_to_cell_vjp(cell_cotangent), beta)


def smooth_lumerical_250nm_constraints(
    latent: np.ndarray,
    beta: float,
    *,
    mapping: LumericalNodalDesignMapping = OPTIMIZER_250NM_MAPPING,
    tau: float = DEFAULT_OPENING_TAU,
    positive_tau: float = DEFAULT_POSITIVE_PART_TAU,
    ks_alpha: float = DEFAULT_KS_ALPHA,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Apply physical-cell DFM residuals and pull them back to nodal latent rho."""

    nodes = mapping.physical(latent, beta)
    cells = nodal_to_cell_average(nodes)
    values, cell_gradients, fields = smooth_500nm_physical_constraints(
        cells,
        spacing_m=CONTRACT.design_pitch_m,
        minimum_solid_feature_m=LUMERICAL_MINIMUM_SOLID_FEATURE_M,
        minimum_void_feature_m=LUMERICAL_MINIMUM_VOID_FEATURE_M,
        tau=tau,
        positive_tau=positive_tau,
        ks_alpha=ks_alpha,
    )
    latent_gradients = np.stack(
        [
            mapping.vjp(latent, nodal_to_cell_vjp(gradient), beta)
            for gradient in cell_gradients
        ]
    )
    fields = {
        **fields,
        "projected_nodal_density": nodes,
        "projected_cell_density": cells,
    }
    return values, latent_gradients, fields


def calibrated_lumerical_250nm_dfm_caps() -> tuple[np.ndarray, dict[str, Any]]:
    """Calibrate smooth caps on exact-pass 250-nm binary reference patterns."""

    nx, ny = CONTRACT.design_shape
    minimum_feature_cells = int(
        np.ceil(LUMERICAL_MINIMUM_SOLID_FEATURE_M / CONTRACT.design_pitch_m)
    )
    if minimum_feature_cells < 1 or minimum_feature_cells >= min(nx, ny):
        raise RuntimeError("invalid 250-nm DFM calibration feature width")
    footprint = physical_disk_footprint(
        0.5 * LUMERICAL_MINIMUM_SOLID_FEATURE_M,
        CONTRACT.design_pitch_m,
    )

    def stadium(length_cells: int, *, vertical: bool, iterations: int = 1) -> np.ndarray:
        seed = np.zeros(CONTRACT.design_shape, dtype=bool)
        start = (nx - length_cells) // 2
        stop = start + length_cells
        if vertical:
            seed[start:stop, ny // 2] = True
        else:
            seed[nx // 2, start:stop] = True
        return ndimage.binary_dilation(
            seed,
            structure=footprint,
            iterations=iterations,
        )

    horizontal = stadium(31, vertical=False)
    vertical = stadium(31, vertical=True)
    outer = stadium(31, vertical=False, iterations=4)
    inner_void = stadium(11, vertical=False)
    patterns: dict[str, np.ndarray] = {
        "minimum_disk": stadium(1, vertical=False).astype(np.float64),
        "horizontal_stadium": horizontal.astype(np.float64),
        "vertical_stadium": vertical.astype(np.float64),
        "rounded_ring_with_internal_void": (outer & ~inner_void).astype(np.float64),
    }

    rows: list[dict[str, Any]] = []
    values: list[np.ndarray] = []
    for name, density in patterns.items():
        exact = exact_500nm_audit(
            density,
            spacing_m=CONTRACT.design_pitch_m,
            minimum_feature_m=LUMERICAL_MINIMUM_SOLID_FEATURE_M,
        )
        if not bool(exact["solid_pass"] and exact["void_pass"]):
            raise RuntimeError(f"250-nm DFM calibration pattern failed exact audit: {name}")
        smooth, _, _ = smooth_500nm_physical_constraints(
            density,
            spacing_m=CONTRACT.design_pitch_m,
            minimum_solid_feature_m=LUMERICAL_MINIMUM_SOLID_FEATURE_M,
            minimum_void_feature_m=LUMERICAL_MINIMUM_VOID_FEATURE_M,
        )
        values.append(smooth)
        rows.append(
            {
                "name": name,
                "smooth_values": smooth.tolist(),
                "exact_solid_pass": True,
                "exact_void_pass": True,
            }
        )
    reference_max = np.max(np.stack(values), axis=0)
    caps = reference_max + LUMERICAL_DFM_CALIBRATION_MARGIN
    return caps, {
        "schema": "lumerical-250nm-dfm-calibration-v1",
        "minimum_solid_feature_nm": 250.0,
        "minimum_void_feature_nm": 250.0,
        "opening_radius_nm": 125.0,
        "design_pitch_nm": CONTRACT.design_pitch_m * 1.0e9,
        "minimum_feature_cells_ceil": minimum_feature_cells,
        "opening_footprint_pixel_count": int(np.count_nonzero(footprint)),
        "calibration_margin": LUMERICAL_DFM_CALIBRATION_MARGIN,
        "reference_max": reference_max.tolist(),
        "caps": caps.tolist(),
        "patterns": rows,
    }


def exact_binary_cell_candidate(
    projected_nodes: np.ndarray, *, threshold: float = 0.5
) -> tuple[np.ndarray, dict[str, object]]:
    """Create the separately reevaluated 80x80 ordinary-Au candidate mask."""

    threshold_value = float(threshold)
    if not 0.0 < threshold_value < 1.0:
        raise ValueError("binary threshold must lie strictly inside (0,1)")
    cells = nodal_to_cell_average(projected_nodes)
    mask = np.ascontiguousarray(cells >= threshold_value, dtype=np.uint8)
    audit = exact_500nm_audit(
        cells,
        spacing_m=CONTRACT.design_pitch_m,
        minimum_feature_m=LUMERICAL_MINIMUM_SOLID_FEATURE_M,
        threshold=threshold_value,
    )
    return mask, {
        **audit,
        "candidate_rule": "threshold four-node cell-average occupancy",
        "threshold": threshold_value,
        "minimum_solid_feature_nm": LUMERICAL_MINIMUM_SOLID_FEATURE_M * 1.0e9,
        "minimum_void_feature_nm": LUMERICAL_MINIMUM_VOID_FEATURE_M * 1.0e9,
        "requires_ordinary_dispersive_au_reevaluation": True,
    }


def design_state_audit(
    latent: np.ndarray,
    beta: float,
    *,
    mapping: LumericalNodalDesignMapping = OPTIMIZER_250NM_MAPPING,
) -> dict[str, Any]:
    projected = mapping.physical(latent, beta)
    return {
        "mapping": mapping.audit(),
        "beta": float(beta),
        "minimum_solid_feature_nm": LUMERICAL_MINIMUM_SOLID_FEATURE_M * 1.0e9,
        "minimum_void_feature_nm": LUMERICAL_MINIMUM_VOID_FEATURE_M * 1.0e9,
        "latent_range": [float(np.min(latent)), float(np.max(latent))],
        "shared_projected_density": density_state_audit(projected),
        "projected_cell_range": [
            float(np.min(nodal_to_cell_average(projected))),
            float(np.max(nodal_to_cell_average(projected))),
        ],
    }
