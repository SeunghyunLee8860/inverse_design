"""Conservative optical-Q coupling for the paper-reduced thermal model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conservative_remap import (
    ConservativeDensityRemap,
    NodalDensityRemap1D,
    build_conservative_density_remap,
    build_nodal_density_remap_1d,
    nodal_control_volume_edges,
)
from paper_reduced_thermal import reduced_flake_grid
from yee_absorption_functional import trapezoid_weights


@dataclass(frozen=True)
class ReducedOpticalRemap:
    source_W_m3: np.ndarray
    native_power_W: float
    common_power_W: float
    thermal_power_W: float
    q_common_W_m3: np.ndarray
    component_remaps: dict[int, NodalDensityRemap1D]
    optical_to_thermal: ConservativeDensityRemap

    def native_weight_from_thermal_sensitivity(
        self,
        *,
        thermal_density_sensitivity: np.ndarray,
        native_component_volume_m3: dict[int, np.ndarray],
        native_shape: tuple[int, int, int, int],
    ) -> np.ndarray:
        sensitivity = np.asarray(thermal_density_sensitivity, float)
        if sensitivity.shape != self.source_W_m3.shape:
            raise ValueError(
                f"thermal sensitivity {sensitivity.shape} != "
                f"{self.source_W_m3.shape}"
            )
        common = self.optical_to_thermal.transpose(sensitivity)
        native_coefficient = np.zeros(native_shape, float)
        native_coefficient[..., 0] = self.component_remaps[
            0
        ].transpose_axis(common, axis=0)
        native_coefficient[..., 1] = self.component_remaps[
            1
        ].transpose_axis(common, axis=1)
        native_weight = np.zeros(native_shape, float)
        for component in range(3):
            volume = np.asarray(
                native_component_volume_m3[component], float
            )
            if np.any(volume <= 0.0):
                raise RuntimeError(
                    "native observation quadrature has nonpositive volume"
                )
            native_weight[..., component] = (
                native_coefficient[..., component] / volume
            )
        return native_weight


def remap_absorption_to_reduced_thermal(optical) -> ReducedOpticalRemap:
    """Map one ``AbsorptionForward`` to the 24x24x2 reduced flake grid."""
    grid = optical.grid
    q_native = np.asarray(
        optical.observation.density_component_W_m3, float
    )
    deltas = {
        axis: np.asarray(
            optical.field_result[f"fom_delta_{axis}"], float
        ).reshape(-1)
        for axis in "xyz"
    }
    component_remaps: dict[int, NodalDensityRemap1D] = {}
    q_common_components = np.zeros_like(q_native)
    for component, axis in enumerate("xy"):
        source_coordinates = np.asarray(grid[axis], float) + deltas[axis]
        remap = build_nodal_density_remap_1d(
            source_coordinates_m=source_coordinates,
            target_coordinates_m=np.asarray(grid[axis], float),
            periodic=True,
        )
        component_remaps[component] = remap
        q_common_components[..., component] = remap.apply_axis(
            q_native[..., component], axis=component
        )
    if optical.epsilon_imaginary[2] != 0.0:
        raise RuntimeError(
            "nonzero z loss needs a validated shifted-z remap"
        )
    q_common = np.sum(q_common_components, axis=-1)
    common_volume = (
        trapezoid_weights(grid["x"])[:, None, None]
        * trapezoid_weights(grid["y"])[None, :, None]
        * trapezoid_weights(grid["z"])[None, None, :]
    )
    native_power = float(optical.observation.power_total_W)
    common_power = float(np.sum(common_volume * q_common))

    target_edges = reduced_flake_grid()
    source_edges = tuple(
        nodal_control_volume_edges(np.asarray(grid[axis], float))
        for axis in "xyz"
    )
    optical_to_thermal = build_conservative_density_remap(
        source_edges_m=source_edges,
        target_edges_m=target_edges,
    )
    source = optical_to_thermal.apply(q_common)
    thermal_power = float(
        np.sum(optical_to_thermal.target_volume_m3 * source)
    )
    return ReducedOpticalRemap(
        source_W_m3=source,
        native_power_W=native_power,
        common_power_W=common_power,
        thermal_power_W=thermal_power,
        q_common_W_m3=q_common,
        component_remaps=component_remaps,
        optical_to_thermal=optical_to_thermal,
    )
