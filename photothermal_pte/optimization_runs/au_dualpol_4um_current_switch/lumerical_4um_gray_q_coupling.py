"""Conservative gray-density Lumerical Yee-Q to custom-PDE coupling.

The imported Au topology is not an exact material mask during a relaxed
iteration.  Filtering native Yee absorption by equality to exact Au or air
would therefore discard the design-layer loss.  This module instead embeds
all three collocated native ``Q`` components into the thermal finite-volume
grid by literal Cartesian overlap.

The forward map returns thermal-cell *power* in watts.  Its pullback accepts
``d objective / d cell_power`` and returns one cotangent for each native
component ``Q`` density.  Both operations use the same separable overlap
operator; there is no clipping, material classification, smoothing, gain, or
global rescaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from photothermal_pte.finite_inverse_design.finite_q_mapping import (
    apply_material_intersection_density_separable,
    nodal_control_volume_edges,
    transpose_material_intersection_density_separable,
)


COMPONENTS = "xyz"


def adjoint_bilinear_dot_audit(
    *,
    source_components: Mapping[str, np.ndarray],
    source_pullback: Mapping[str, np.ndarray],
    mapped_output: np.ndarray,
    output_cotangent: np.ndarray,
    source_scale: float = 1.0,
) -> dict[str, float]:
    """Return a cancellation-safe audit of ``<M*x,y> = <x,M.T*y>``.

    A relative error normalized only by the two signed contractions is not a
    valid pass/fail metric when the physical result is close to zero.  That
    situation occurs at the symmetric, uniform starting design: large
    positive and negative current contributions cancel by roughly eight
    orders of magnitude.  The normwise denominator below is the standard
    Cauchy--Schwarz scale for a bilinear adjoint identity and remains well
    conditioned under such cancellation.  The signed relative error is
    retained as a diagnostic, but must not be used as the numerical gate.
    """

    scale = float(source_scale)
    if not np.isfinite(scale):
        raise ValueError("source_scale must be finite")
    if set(source_components) != set(source_pullback):
        raise ValueError("source and pullback component names differ")

    pulled_contraction = 0.0
    pulled_norm_bound = 0.0
    for component in sorted(source_components):
        source = np.asarray(source_components[component], dtype=np.float64)
        pulled = np.asarray(source_pullback[component], dtype=np.float64)
        if source.shape != pulled.shape:
            raise ValueError(f"{component} source and pullback shapes differ")
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(pulled)):
            raise ValueError(f"{component} source or pullback is non-finite")
        pulled_contraction += scale * float(np.sum(pulled * source))
        pulled_norm_bound += abs(scale) * float(
            np.linalg.norm(pulled.ravel()) * np.linalg.norm(source.ravel())
        )

    mapped = np.asarray(mapped_output, dtype=np.float64)
    cotangent = np.asarray(output_cotangent, dtype=np.float64)
    if mapped.shape != cotangent.shape:
        raise ValueError("mapped output and output cotangent shapes differ")
    if not np.all(np.isfinite(mapped)) or not np.all(np.isfinite(cotangent)):
        raise ValueError("mapped output or output cotangent is non-finite")
    mapped_contraction = float(np.sum(cotangent * mapped))
    mapped_norm_bound = float(
        np.linalg.norm(cotangent.ravel()) * np.linalg.norm(mapped.ravel())
    )

    absolute_error = abs(pulled_contraction - mapped_contraction)
    signed_scale = max(
        abs(pulled_contraction),
        abs(mapped_contraction),
        np.finfo(float).tiny,
    )
    normwise_scale = max(
        pulled_norm_bound,
        mapped_norm_bound,
        np.finfo(float).tiny,
    )
    return {
        "pullback_contraction": pulled_contraction,
        "mapped_contraction": mapped_contraction,
        "absolute_error": absolute_error,
        "signed_relative_error": absolute_error / signed_scale,
        "normwise_scale": normwise_scale,
        "normwise_relative_error": absolute_error / normwise_scale,
        "cancellation_ratio": signed_scale / normwise_scale,
    }


def _edges(values: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, ...]:
    result = tuple(np.asarray(axis, dtype=np.float64).reshape(-1) for axis in values)
    for axis, name in zip(result, COMPONENTS, strict=True):
        if axis.size < 2 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
            raise ValueError(f"invalid {name} coordinate/edge array")
    return result


def _volumes(edges: tuple[np.ndarray, ...]) -> np.ndarray:
    widths = tuple(np.diff(axis) for axis in edges)
    return (
        widths[0][:, None, None]
        * widths[1][None, :, None]
        * widths[2][None, None, :]
    )


def component_coordinates_from_raw(
    raw: Mapping[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read component-specific coordinates from a script-25 raw artifact."""

    return {
        component: tuple(
            np.asarray(raw[f"Q{component}_{axis}_m"], dtype=np.float64).reshape(-1)
            for axis in COMPONENTS
        )
        for component in COMPONENTS
    }


def component_q_from_raw(
    raw: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Read the three native volumetric absorption components."""

    return {
        component: np.asarray(raw[f"Q{component}_W_m3"], dtype=np.float64)
        for component in COMPONENTS
    }


@dataclass(frozen=True)
class GrayYeeQCoupling:
    """Memory-bounded exact-overlap operator for a frozen Yee/thermal grid."""

    target_edges_m: tuple[np.ndarray, np.ndarray, np.ndarray]
    source_edges_m: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]

    @classmethod
    def from_component_coordinates(
        cls,
        component_coordinates_m: Mapping[
            str, tuple[np.ndarray, np.ndarray, np.ndarray]
        ],
        target_edges_m: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> "GrayYeeQCoupling":
        target = _edges(target_edges_m)
        source: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for component in COMPONENTS:
            coordinates = _edges(component_coordinates_m[component])
            source[component] = tuple(
                nodal_control_volume_edges(axis) for axis in coordinates
            )
            for axis, source_axis, target_axis in zip(
                COMPONENTS, source[component], target, strict=True
            ):
                tolerance = 1.0e-15
                if (
                    source_axis[0] < target_axis[0] - tolerance
                    or source_axis[-1] > target_axis[-1] + tolerance
                ):
                    raise ValueError(
                        f"Q{component} {axis} support is not contained in the thermal grid"
                    )
        return cls(target_edges_m=target, source_edges_m=source)

    @property
    def target_shape(self) -> tuple[int, int, int]:
        return tuple(axis.size - 1 for axis in self.target_edges_m)

    @property
    def target_volume_m3(self) -> np.ndarray:
        return _volumes(self.target_edges_m)

    def source_shape(self, component: str) -> tuple[int, int, int]:
        return tuple(axis.size - 1 for axis in self.source_edges_m[component])

    def source_volume_m3(self, component: str) -> np.ndarray:
        return _volumes(self.source_edges_m[component])

    def map_density(
        self,
        q_components_W_m3: Mapping[str, np.ndarray],
        *,
        require_nonnegative: bool = True,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Map native Q density to total thermal-cell source density."""

        support = np.ones(self.target_shape, dtype=bool)
        total = np.zeros(self.target_shape, dtype=np.float64)
        records: dict[str, dict[str, float | list[int]]] = {}
        input_power = 0.0
        output_power = 0.0
        for component in COMPONENTS:
            q = np.asarray(q_components_W_m3[component], dtype=np.float64)
            if q.shape != self.source_shape(component):
                raise ValueError(
                    f"Q{component} shape {q.shape} != {self.source_shape(component)}"
                )
            if not np.all(np.isfinite(q)):
                raise ValueError(f"Q{component} contains NaN or Inf")
            if require_nonnegative and np.any(q < 0.0):
                raise ValueError(f"Q{component} contains negative absorption")
            mapped, overlap, audit = apply_material_intersection_density_separable(
                source_density=q,
                source_edges_m=self.source_edges_m[component],
                target_edges_m=self.target_edges_m,
                target_material_support_mask=support,
            )
            source_volume = self.source_volume_m3(component)
            overlap_error = float(np.max(np.abs(overlap - source_volume)))
            total += mapped
            input_power += float(audit["material_attributed_source_power_W"])
            output_power += float(audit["target_integrated_power_W"])
            records[component] = {
                "shape": list(q.shape),
                "input_power_W": float(audit["material_attributed_source_power_W"]),
                "mapped_power_W": float(audit["target_integrated_power_W"]),
                "relative_power_error": float(audit["relative_power_error"]),
                "source_overlap_volume_max_abs_error_m3": overlap_error,
            }
        relative_error = abs(input_power - output_power) / max(
            abs(input_power), abs(output_power), np.finfo(float).tiny
        )
        return total, {
            "method": "all_native_component_yee_Q_exact_overlap_v1",
            "input_power_W": input_power,
            "mapped_power_W": output_power,
            "relative_power_error": relative_error,
            "component": records,
            "material_equality_filter": False,
            "density_dependent_geometric_mask": False,
            "operations_absent": [
                "clipping",
                "smoothing",
                "gain",
                "global_rescaling",
                "nearest_cell_relocation",
            ],
        }

    def map_power(
        self,
        q_components_W_m3: Mapping[str, np.ndarray],
        *,
        require_nonnegative: bool = True,
    ) -> tuple[np.ndarray, dict[str, object]]:
        density, audit = self.map_density(
            q_components_W_m3, require_nonnegative=require_nonnegative
        )
        return density * self.target_volume_m3, audit

    def pullback_cell_power(
        self, cell_power_cotangent: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Apply the exact transpose from cell-power to native-Q density."""

        cotangent = np.asarray(cell_power_cotangent, dtype=np.float64)
        if cotangent.shape != self.target_shape or not np.all(np.isfinite(cotangent)):
            raise ValueError("cell-power cotangent has the wrong shape or is non-finite")
        # The separable transpose consumes a cotangent with respect to target
        # density.  Since p_cell = Q_cell * V_cell, multiply by volume first.
        density_cotangent = cotangent * self.target_volume_m3
        support = np.ones(self.target_shape, dtype=bool)
        return {
            component: transpose_material_intersection_density_separable(
                target_density_sensitivity=density_cotangent,
                source_edges_m=self.source_edges_m[component],
                target_edges_m=self.target_edges_m,
                target_material_support_mask=support,
            )
            for component in COMPONENTS
        }

    def transpose_dot_audit(
        self,
        *,
        seed: int = 4_002_608_24,
    ) -> dict[str, float]:
        """Check the implemented power-map transpose with signed probes."""

        rng = np.random.default_rng(seed)
        direction = {
            component: rng.normal(size=self.source_shape(component))
            for component in COMPONENTS
        }
        target = rng.normal(size=self.target_shape)
        mapped, _ = self.map_power(direction, require_nonnegative=False)
        pulled = self.pullback_cell_power(target)
        left = float(np.sum(target * mapped))
        right = float(
            sum(np.sum(pulled[component] * direction[component]) for component in COMPONENTS)
        )
        relative_error = abs(left - right) / max(
            abs(left), abs(right), np.finfo(float).tiny
        )
        return {
            "left": left,
            "right": right,
            "relative_error": relative_error,
        }
