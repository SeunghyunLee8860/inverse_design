"""Fail-closed contract for the paired TaIrTe4-flake optimizations.

The fast optical dimensions in this module are candidates until the GPU
runsetup, source-only, domain, and mesh audits pass.  They must not be
reported as validated merely because the values are internally consistent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erf, exp, sqrt


@dataclass(frozen=True)
class TaIrTe4FlakeContract:
    axis_contract: str = "lumerical_x_b_y_a"
    wavelength_m: float = 10.0e-6
    target_waist_m: float = 8.5e-6
    calibrated_source_object_waist_m: float = 8.36043075475035e-6
    flake_span_m: float = 24.0e-6
    design_span_m: float = 16.0e-6
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
    def fixed_frame_width_m(self) -> float:
        return 0.5 * (self.flake_span_m - self.design_span_m)

    @property
    def design_intervals(self) -> int:
        return int(round(self.design_span_m / self.design_step_m))

    @property
    def design_node_shape(self) -> tuple[int, int]:
        count = self.design_intervals + 1
        return count, count

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
            "design_span_m",
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
        if self.design_span_m >= self.flake_span_m:
            raise ValueError("design must be enclosed by a fixed TaIrTe4 frame")
        if self.flake_span_m >= self.optical_lateral_span_m:
            raise ValueError("finite flake must not touch transverse PML")
        if not 0.0 < self.source_span_m < self.optical_lateral_span_m:
            raise ValueError("source must be finite and remain inside the FDTD span")
        if 0.5 * (self.optical_lateral_span_m - self.source_span_m) < 2.0e-6:
            raise ValueError("source-to-transverse-PML clearance must be at least 2 um")
        if not self.optical_z_min_m < self.focus_z_m < self.source_z_m < self.optical_z_max_m:
            raise ValueError("invalid focus/source/z-PML ordering")
        if abs(self.design_intervals * self.design_step_m - self.design_span_m) > 1e-18:
            raise ValueError("design step must divide the design span")
        if abs(round(self.flake_thickness_m / self.flake_dz_m) * self.flake_dz_m - self.flake_thickness_m) > 1e-18:
            raise ValueError("flake dz must divide the flake thickness")
        if self.feature_cells < 5.0 - 1e-12:
            raise ValueError("500 nm feature must have at least five design cells")
        if self.fixed_frame_width_m < 2.0 * self.minimum_feature_m:
            raise ValueError("fixed electrical frame is too narrow")
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
                "Gaussian_power_fraction_in_design_square": self.square_gaussian_fraction(self.design_span_m),
                "Gaussian_power_fraction_in_flake_square": self.square_gaussian_fraction(self.flake_span_m),
                "Gaussian_intensity_fraction_at_flake_edge": self.boundary_intensity_fraction(0.5 * self.flake_span_m),
                "Gaussian_intensity_fraction_at_transverse_PML": self.boundary_intensity_fraction(0.5 * self.optical_lateral_span_m),
            },
            "immutable_physics": {
                "periodic_or_Bloch": False,
                "six_boundaries": "PML",
                "source": "finite scalar Gaussian",
                "coordinate_mapping": "Lumerical x=b, y=a, z=c",
                "design_endpoints": {"rho=0": "air/void", "rho=1": "TaIrTe4"},
                "fixed_top_bottom_contact_regions": True,
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


CONTRACT = TaIrTe4FlakeContract()
