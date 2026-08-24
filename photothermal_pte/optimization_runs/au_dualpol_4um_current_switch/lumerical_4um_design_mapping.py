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
    smooth_500nm_physical_constraints,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (
    canonical_density_nodes,
    density_state_audit,
    nodal_to_cell_average,
    nodal_to_cell_jvp,
    nodal_to_cell_vjp,
)


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

    def __post_init__(self) -> None:
        if self.shape != CONTRACT.design_node_shape:
            raise ValueError(
                "Lumerical optimizer mapping must use the canonical 81x81 nodes"
            )
        if self.spacing_m <= 0.0 or self.radius_m <= 0.0:
            raise ValueError("spacing and radius must be positive")
        if not 0.0 < self.eta < 1.0:
            raise ValueError("projection eta must lie strictly inside (0,1)")
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
        return canonical_density_nodes(projected)

    def jvp(
        self, latent: np.ndarray, direction: np.ndarray, beta: float
    ) -> np.ndarray:
        beta_value = _validated_beta(beta)
        filtered = self.filtered(latent)
        filtered_direction = self.filtered(
            self._array(direction, label="latent direction")
        )
        return _projection_derivative(
            filtered, beta=beta_value, eta=self.eta
        ) * filtered_direction

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
            "projection_eta": self.eta,
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


def projected_cell_density(
    latent: np.ndarray, beta: float, *, mapping: LumericalNodalDesignMapping = NOMINAL_MAPPING
) -> np.ndarray:
    """Return the only allowed custom-PDE/DFM cell density from latent rho."""

    return nodal_to_cell_average(mapping.physical(latent, beta))


def projected_cell_jvp(
    latent: np.ndarray,
    direction: np.ndarray,
    beta: float,
    *,
    mapping: LumericalNodalDesignMapping = NOMINAL_MAPPING,
) -> np.ndarray:
    return nodal_to_cell_jvp(mapping.jvp(latent, direction, beta))


def projected_cell_vjp(
    latent: np.ndarray,
    cell_cotangent: np.ndarray,
    beta: float,
    *,
    mapping: LumericalNodalDesignMapping = NOMINAL_MAPPING,
) -> np.ndarray:
    return mapping.vjp(latent, nodal_to_cell_vjp(cell_cotangent), beta)


def smooth_lumerical_500nm_constraints(
    latent: np.ndarray,
    beta: float,
    *,
    mapping: LumericalNodalDesignMapping = NOMINAL_MAPPING,
    tau: float = DEFAULT_OPENING_TAU,
    positive_tau: float = DEFAULT_POSITIVE_PART_TAU,
    ks_alpha: float = DEFAULT_KS_ALPHA,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Apply physical-cell DFM residuals and pull them back to nodal latent rho."""

    nodes = mapping.physical(latent, beta)
    cells = nodal_to_cell_average(nodes)
    values, cell_gradients, fields = smooth_500nm_physical_constraints(
        cells,
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
        minimum_feature_m=CONTRACT.minimum_solid_feature_m,
        threshold=threshold_value,
    )
    return mask, {
        **audit,
        "candidate_rule": "threshold four-node cell-average occupancy",
        "threshold": threshold_value,
        "requires_ordinary_dispersive_au_reevaluation": True,
    }


def design_state_audit(
    latent: np.ndarray,
    beta: float,
    *,
    mapping: LumericalNodalDesignMapping = NOMINAL_MAPPING,
) -> dict[str, Any]:
    projected = mapping.physical(latent, beta)
    return {
        "mapping": mapping.audit(),
        "beta": float(beta),
        "latent_range": [float(np.min(latent)), float(np.max(latent))],
        "shared_projected_density": density_state_audit(projected),
        "projected_cell_range": [
            float(np.min(nodal_to_cell_average(projected))),
            float(np.max(nodal_to_cell_average(projected))),
        ],
    }
