"""Fail-closed contract for the paired TaIrTe4-flake optimizations.

The fast optical dimensions in this module are candidates until the GPU
runsetup, source-only, domain, and mesh audits pass.  They must not be
reported as validated merely because the values are internally consistent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erf, exp, sqrt
import os

import numpy as np


@dataclass(frozen=True)
class TaIrTe4FlakeContract:
    geometry_mode: str = "fixed_frame"
    axis_contract: str = "lumerical_x_b_y_a"
    wavelength_m: float = 10.0e-6
    target_waist_m: float = 8.5e-6
    calibrated_source_object_waist_m: float = 8.36043075475035e-6
    flake_span_m: float = 24.0e-6
    design_span_x_m: float = 16.0e-6
    design_span_y_m: float = 16.0e-6
    fixed_contact_depth_m: float = 4.0e-6
    flake_thickness_m: float = 100.0e-9
    design_step_m: float = 100.0e-9
    flake_dz_m: float = 10.0e-9
    optical_lateral_span_m: float = 40.0e-6
    optical_z_min_m: float = -4.0e-6
    optical_z_max_m: float = 4.0e-6
    source_span_m: float = 34.0e-6
    source_z_m: float = 2.0e-6
    focus_z_m: float = 0.0
    pml_layers: int = 24
    mesh_accuracy: int = 3
    outer_xy_max_step_m: float = 500.0e-9
    interface_xy_step_m: float = 100.0e-9
    bottom_sio2_thickness_m: float = 285.0e-9
    minimum_feature_m: float = 500.0e-9
    thermal_lateral_span_m: float = 64.0e-6
    sigma_void_fraction: float = 1.0e-8
    sigma_penalty: float = 2.0
    alpha_penalty: float = 2.0

    @property
    def design_span_m(self) -> float:
        """Legacy square-span alias retained for immutable Run010 readers."""
        if not np_isclose(self.design_span_x_m, self.design_span_y_m):
            raise AttributeError("rectangular design has separate x/y spans")
        return self.design_span_x_m

    @property
    def fixed_frame_width_m(self) -> float:
        return 0.5 * (self.flake_span_m - self.design_span_x_m)

    @property
    def flake_bounding_half_span_m(self) -> float:
        if self.geometry_mode == "diagonal_45_contact_anchored":
            return 0.5 * self.flake_span_m * sqrt(2.0)
        return 0.5 * self.flake_span_m

    @staticmethod
    def rotated_uv(x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return +45-degree device coordinates while crystal axes stay fixed."""

        return (x_m + y_m) / sqrt(2.0), (-x_m + y_m) / sqrt(2.0)

    def flake_support_mask(self, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("flake support coordinates must have matching shapes")
        if self.geometry_mode != "diagonal_45_contact_anchored":
            half = 0.5 * self.flake_span_m
            return (np.abs(x) <= half + 1.0e-18) & (np.abs(y) <= half + 1.0e-18)
        u, v = self.rotated_uv(x, y)
        half = 0.5 * self.flake_span_m
        return (np.abs(u) <= half + 1.0e-18) & (np.abs(v) <= half + 1.0e-18)

    @property
    def contact_axis(self) -> str:
        """Electrical terminal axis for the selected finite-flake geometry."""
        if self.geometry_mode == "diagonal_45_contact_anchored":
            return "diagonal_45"
        if self.geometry_mode == "left_right_contact_anchored":
            return "x"
        return "y"

    @property
    def fixed_design_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """Return no fixed-solid regions for ideal electrical-only terminals."""

        shape = self.design_node_shape
        empty = np.zeros(shape, dtype=bool)
        return empty.copy(), empty.copy()

    @property
    def fixed_design_solid_mask(self) -> np.ndarray:
        low, high = self.fixed_design_contact_masks
        return low | high

    @property
    def fixed_design_void_mask(self) -> np.ndarray:
        return np.zeros(self.design_node_shape, dtype=bool)

    @property
    def designable_node_mask(self) -> np.ndarray:
        return ~(self.fixed_design_solid_mask | self.fixed_design_void_mask)

    def apply_fixed_contact_density(self, value: np.ndarray) -> np.ndarray:
        """Return a copy with terminal-overlap strips held at solid TaIrTe4."""

        result = np.asarray(value).copy()
        if result.shape != self.design_node_shape:
            raise ValueError(
                f"design value shape {result.shape} != {self.design_node_shape}"
            )
        result[self.fixed_design_void_mask] = 0.0
        result[self.fixed_design_solid_mask] = 1.0
        return result

    def zero_fixed_contact_gradient(self, value: np.ndarray) -> np.ndarray:
        """Return a copy with derivatives of locked terminal nodes set to zero."""

        result = np.asarray(value).copy()
        if result.shape != self.design_node_shape:
            raise ValueError(
                f"design gradient shape {result.shape} != {self.design_node_shape}"
            )
        result[~self.designable_node_mask] = 0.0
        return result

    def terminal_node_masks(self, nodes_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return diagonal low/high equipotential masks on a full-flake mesh."""

        nodes = np.asarray(nodes_m, dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("terminal coordinates must have shape (N, 2)")
        if self.contact_axis != "diagonal_45":
            raise ValueError("terminal-node masks are only defined for diagonal contacts")
        u, _ = self.rotated_uv(nodes[:, 0], nodes[:, 1])
        edge = 0.5 * self.flake_span_m
        tolerance = 1.0e-18
        low = u <= -edge + self.fixed_contact_depth_m + tolerance
        high = u >= edge - self.fixed_contact_depth_m - tolerance
        if not np.any(low) or not np.any(high) or np.any(low & high):
            raise RuntimeError("invalid or empty diagonal terminal masks")
        return low, high

    @property
    def design_intervals(self) -> tuple[int, int]:
        return (
            int(round(self.design_span_x_m / self.design_step_m)),
            int(round(self.design_span_y_m / self.design_step_m)),
        )

    @property
    def design_node_shape(self) -> tuple[int, int]:
        return tuple(value + 1 for value in self.design_intervals)

    @property
    def design_bounds_m(self) -> dict[str, tuple[float, float]]:
        return {
            "x": (-0.5 * self.design_span_x_m, 0.5 * self.design_span_x_m),
            "y": (-0.5 * self.design_span_y_m, 0.5 * self.design_span_y_m),
        }

    @property
    def flake_node_shape(self) -> tuple[int, int]:
        count = int(round(self.flake_span_m / self.design_step_m)) + 1
        return count, count

    @property
    def crystal_bounding_intervals(self) -> int:
        return int(np.ceil(self.flake_span_m * sqrt(2.0) / self.design_step_m))

    @property
    def crystal_bounding_node_shape(self) -> tuple[int, int]:
        count = self.crystal_bounding_intervals + 1
        return count, count

    @property
    def crystal_bounding_span_m(self) -> float:
        return self.crystal_bounding_intervals * self.design_step_m

    @property
    def design_node_slices(self) -> tuple[slice, slice]:
        if self.geometry_mode == "diagonal_45_contact_anchored":
            return slice(None), slice(None)
        half_flake = 0.5 * self.flake_span_m
        slices = []
        for axis in "xy":
            low, high = self.design_bounds_m[axis]
            start = int(round((low + half_flake) / self.design_step_m))
            stop = int(round((high + half_flake) / self.design_step_m)) + 1
            slices.append(slice(start, stop))
        return slices[0], slices[1]

    @property
    def feature_cells(self) -> float:
        return self.minimum_feature_m / self.design_step_m

    def square_gaussian_fraction(self, span_m: float) -> float:
        half = 0.5 * span_m
        return erf(sqrt(2.0) * half / self.target_waist_m) ** 2

    def boundary_intensity_fraction(self, half_span_m: float) -> float:
        return exp(-2.0 * (half_span_m / self.target_waist_m) ** 2)

    def validate(self) -> None:
        positive_names = (
            "wavelength_m",
            "target_waist_m",
            "calibrated_source_object_waist_m",
            "flake_span_m",
            "design_span_x_m",
            "design_span_y_m",
            "fixed_contact_depth_m",
            "flake_thickness_m",
            "design_step_m",
            "flake_dz_m",
            "optical_lateral_span_m",
            "source_span_m",
            "source_z_m",
            "outer_xy_max_step_m",
            "interface_xy_step_m",
            "bottom_sio2_thickness_m",
            "minimum_feature_m",
            "thermal_lateral_span_m",
        )
        positive = {name: getattr(self, name) for name in positive_names}
        if any(float(value) <= 0.0 for value in positive.values()):
            raise ValueError(f"contract lengths must be positive: {positive}")
        if self.axis_contract != "lumerical_x_b_y_a":
            raise ValueError("TaIrTe4 optimization requires Lumerical x=b, y=a")
        if self.geometry_mode not in {
            "fixed_frame",
            "contact_anchored",
            "left_right_contact_anchored",
            "diagonal_45_contact_anchored",
        }:
            raise ValueError(f"unsupported geometry mode: {self.geometry_mode}")
        if (
            self.design_span_x_m > self.flake_span_m
            or self.design_span_y_m > self.flake_span_m
        ):
            raise ValueError("design must fit inside the finite TaIrTe4 support")
        if self.geometry_mode == "fixed_frame" and (
            self.design_span_x_m >= self.flake_span_m
            or self.design_span_y_m >= self.flake_span_m
        ):
            raise ValueError("fixed-frame design must be enclosed on all four sides")
        if self.geometry_mode == "contact_anchored" and not np_isclose(
            self.design_span_x_m, self.flake_span_m
        ):
            raise ValueError("contact-anchored design must span the full flake width")
        if self.geometry_mode == "left_right_contact_anchored" and not np_isclose(
            self.design_span_y_m, self.flake_span_m
        ):
            raise ValueError("left/right-contact design must span the full flake height")
        if self.geometry_mode == "diagonal_45_contact_anchored":
            expected = int(round(self.flake_span_m / self.design_step_m))
            if self.design_intervals != (expected, expected):
                raise ValueError(
                    "45-degree local device grid must remain exactly 24 x 24 um"
                )
        if self.flake_span_m >= self.optical_lateral_span_m:
            raise ValueError("finite flake must not touch transverse PML")
        if not 0.0 < self.source_span_m < self.optical_lateral_span_m:
            raise ValueError("source must be finite and remain inside the FDTD span")
        if 0.5 * (self.optical_lateral_span_m - self.source_span_m) < 2.0e-6:
            raise ValueError("source-to-transverse-PML clearance must be at least 2 um")
        if not self.optical_z_min_m < self.focus_z_m < self.source_z_m < self.optical_z_max_m:
            raise ValueError("invalid focus/source/z-PML ordering")
        for intervals, span in zip(
            self.design_intervals, (self.design_span_x_m, self.design_span_y_m)
        ):
            if abs(intervals * self.design_step_m - span) > 1e-18:
                raise ValueError("design step must divide each design span")
        if abs(round(self.flake_thickness_m / self.flake_dz_m) * self.flake_dz_m - self.flake_thickness_m) > 1e-18:
            raise ValueError("flake dz must divide the flake thickness")
        if self.feature_cells < 5.0 - 1e-12:
            raise ValueError("500 nm feature must have at least five design cells")
        if self.geometry_mode == "fixed_frame" and self.fixed_frame_width_m < 2.0 * self.minimum_feature_m:
            raise ValueError("fixed electrical frame is too narrow")
        if self.geometry_mode in {
            "contact_anchored",
            "left_right_contact_anchored",
            "diagonal_45_contact_anchored",
        } and self.fixed_contact_depth_m < 2.0 * self.minimum_feature_m:
            raise ValueError("fixed contact strip is too shallow")
        if not 0.0 < self.sigma_void_fraction < 1.0e-4:
            raise ValueError("void conductivity is a numerical regularization only")

    def audit(self) -> dict[str, object]:
        self.validate()
        return {
            "status": "CANDIDATE_TAIRTE4_FLAKE_FAST_CONTRACT_PENDING_GPU_AUDIT",
            "contract": asdict(self),
            "derived": {
                "fixed_frame_width_m": self.fixed_frame_width_m,
                "design_node_shape": list(self.design_node_shape),
                "minimum_feature_cells": self.feature_cells,
                "Gaussian_power_fraction_in_design_bounding_square": self.square_gaussian_fraction(
                    min(self.design_span_x_m, self.design_span_y_m)
                ),
                "Gaussian_power_fraction_in_flake_square": self.square_gaussian_fraction(self.flake_span_m),
                "Gaussian_intensity_fraction_at_flake_edge": self.boundary_intensity_fraction(0.5 * self.flake_span_m),
                "Gaussian_intensity_fraction_at_transverse_PML": self.boundary_intensity_fraction(0.5 * self.optical_lateral_span_m),
            },
            "immutable_physics": {
                "periodic_or_Bloch": False,
                "six_boundaries": "PML",
                "source": "finite scalar Gaussian",
                "coordinate_mapping": (
                    "Run58-style Maxwell approximation uses a centered "
                    "unrotated sheet with x=b, y=a, z=c; thermal and "
                    "electrical geometry is rotated +45 degrees"
                    if self.geometry_mode == "diagonal_45_contact_anchored"
                    else "Lumerical x=b, y=a, z=c"
                ),
                "design_endpoints": {"rho=0": "air/void", "rho=1": "TaIrTe4"},
                "fixed_contact_regions": {
                    "x": "left_right",
                    "y": "top_bottom",
                    "diagonal_45": "opposite_full_edges_of_plus45_degree_flake",
                }[self.contact_axis],
                "symmetry_constraint": False,
                "Q_clipping_smoothing_gain_or_rescaling": False,
                "CPU_FDTD_fallback": False,
            },
            "required_before_optimization": [
                "GPU runsetup and realized mesh audit",
                "40 versus 48 um optical-domain comparison",
                "100 versus 50 nm representative frozen-density comparison",
                "rho-dependent electrical weighting AD-FD",
                "combined optical-thermal-electrical AD-FD",
            ],
        }


def np_isclose(a: float, b: float, tolerance: float = 1e-18) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def _selected_contract() -> TaIrTe4FlakeContract:
    mode = os.environ.get("TAIRTE4_TOPOLOGY_GEOMETRY", "fixed_frame")
    if mode == "fixed_frame":
        return TaIrTe4FlakeContract()
    if mode == "contact_anchored":
        return TaIrTe4FlakeContract(
            geometry_mode="contact_anchored",
            design_span_x_m=24.0e-6,
            design_span_y_m=20.0e-6,
            fixed_contact_depth_m=2.0e-6,
        )
    if mode == "left_right_contact_anchored":
        return TaIrTe4FlakeContract(
            geometry_mode="left_right_contact_anchored",
            design_span_x_m=20.0e-6,
            design_span_y_m=24.0e-6,
            fixed_contact_depth_m=2.0e-6,
        )
    if mode == "diagonal_45_contact_anchored":
        return TaIrTe4FlakeContract(
            geometry_mode="diagonal_45_contact_anchored",
            design_span_x_m=24.0e-6,
            design_span_y_m=24.0e-6,
            fixed_contact_depth_m=2.0e-6,
            optical_lateral_span_m=48.0e-6,
        )
    raise RuntimeError(f"unknown TAIRTE4_TOPOLOGY_GEOMETRY={mode!r}")


CONTRACT = _selected_contract()
