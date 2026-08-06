"""Offline contract audit for the 10 um, w0=8.5 um PTE optimization.

This module performs no Maxwell, thermal, adjoint, or optimizer solve.  It
turns the requested Gaussian source and the repository's Kitamura silica
closure into explicit geometry, truncation, thickness, and interface
scenarios that can be reviewed before a licensed run starts.
"""

from __future__ import annotations

from itertools import product
from math import erf, exp, pi, sqrt
from typing import Any

import numpy as np

from photothermal_pte.validation.paper_ir_sanity.run_lumerical_device_a_ir_q import (
    kitamura_2007_sio2_epsilon,
)


WAVELENGTH_UM = 10.0
WAIST_UM = 8.5
SOURCE_SPAN_UM = 40.0
FDTD_DOMAIN_UM = 48.0
SOURCE_TO_FOCUS_UM = 5.0
DESIGN_HEIGHT_CANDIDATES_UM = (0.6, 1.0, 1.5)


def gaussian_square_audit(
    *, waist_um: float, source_span_um: float, domain_um: float
) -> dict[str, float]:
    """Return analytic aperture diagnostics for I=I0 exp(-2 r^2/w0^2)."""
    half = 0.5 * float(source_span_um)
    boundary = exp(-2.0 * (half / float(waist_um)) ** 2)
    captured_square = erf(sqrt(2.0) * half / float(waist_um)) ** 2
    return {
        "source_boundary_intensity_over_peak": boundary,
        "fitted_infinite_gaussian_square_captured_fraction": captured_square,
        "comparison_inscribed_circle_captured_fraction": 1.0 - boundary,
        "source_to_PML_lateral_clearance_um_per_side": 0.5
        * (float(domain_um) - float(source_span_um)),
    }

def silica_10um() -> dict[str, float]:
    epsilon = complex(kitamura_2007_sio2_epsilon(WAVELENGTH_UM * 1.0e-6))
    refractive_index = complex(np.sqrt(epsilon))
    return {
        "epsilon_real": float(epsilon.real),
        "epsilon_imag": float(epsilon.imag),
        "n_real": float(refractive_index.real),
        "n_imag": float(refractive_index.imag),
    }


def thickness_audit(height_um: float, refractive_index: complex) -> dict[str, float]:
    """Report propagation-scale diagnostics, not a device FOM prediction."""
    phase = (
        2.0
        * pi
        * (float(refractive_index.real) - 1.0)
        * float(height_um)
        / WAVELENGTH_UM
    )
    intensity_factor = exp(
        -4.0
        * pi
        * float(refractive_index.imag)
        * float(height_um)
        / WAVELENGTH_UM
    )
    return {
        "height_um": float(height_um),
        "air_relative_phase_delay_rad": phase,
        "air_relative_phase_delay_over_pi": phase / pi,
        "bulk_propagation_intensity_factor": intensity_factor,
        "aspect_ratio_at_500nm_min_feature": float(height_um) / 0.5,
    }


def build_contract_audit() -> dict[str, Any]:
    aperture = gaussian_square_audit(
        waist_um=WAIST_UM,
        source_span_um=SOURCE_SPAN_UM,
        domain_um=FDTD_DOMAIN_UM,
    )
    silica = silica_10um()
    refractive_index = complex(silica["n_real"], silica["n_imag"])
    rayleigh_um = pi * WAIST_UM**2 / WAVELENGTH_UM
    scenarios = [
        {
            "bottom_interface": bottom,
            "design_interface": design,
            "G_bottom_W_m2K": 7.37e6 if bottom == "thermally_grown" else 7.37e4,
            "G_design_W_m2K": 7.37e6 if design == "thermally_grown" else 7.37e4,
        }
        for bottom, design in product(
            ("thermally_grown", "evaporated"), repeat=2
        )
    ]
    return {
        "status": "PLANNED_GAUSSIAN10_SOURCE_AND_DESIGN_WINDOW_AUDIT",
        "solver_runs": {
            "Maxwell": False,
            "thermal": False,
            "adjoint": False,
            "optimization": False,
        },
        "source": {
            "wavelength_um": WAVELENGTH_UM,
            "target_realized_waist_um": WAIST_UM,
            "source_span_um": SOURCE_SPAN_UM,
            "candidate_FDTD_domain_um": FDTD_DOMAIN_UM,
            "audit_FDTD_domains_um": [56.0, 64.0],
            "rayleigh_range_um": rayleigh_um,
            "ideal_beam_radius_at_source_plane_um": WAIST_UM
            * sqrt(1.0 + (SOURCE_TO_FOCUS_UM / rayleigh_um) ** 2),
            **aperture,
        },
        "optical_material": {
            "SiO2_model": "Kitamura_2007_fused_silica_explicit_10um_closure",
            "thermally_grown_vs_evaporated_optical_difference": (
                "not assumed without process-specific complex-n data"
            ),
            **silica,
        },
        "design": {
            "coarse_window_selection_canvas_um": {
                "x": [-10.0, 10.0],
                "y": [-10.0, 10.0],
                "sampling_nm": 100.0,
            },
            "candidate_axis_aligned_windows_um": [
                {"name": "a_positive_strip", "x": [-6.0, 6.0], "y": [1.0, 7.0]},
                {"name": "a_negative_strip", "x": [-6.0, 6.0], "y": [-7.0, -1.0]},
                {"name": "b_positive_strip", "x": [1.0, 7.0], "y": [-6.0, 6.0]},
                {"name": "b_negative_strip", "x": [-7.0, -1.0], "y": [-6.0, 6.0]},
                {"name": "centered_control", "x": [-5.0, 5.0], "y": [-5.0, 5.0]},
            ],
            "window_selection_rule": (
                "select the smallest reviewed window retaining at least 90% "
                "of the coarse-canvas absolute physical-density gradient L1"
            ),
            "production_node_spacing_nm": 50.0,
            "minimum_solid_nm": 500.0,
            "minimum_void_nm": 500.0,
            "differentiable_steering_target_nm": 525.0,
            "recommended_height_um": 1.0,
            "height_candidates": [
                thickness_audit(value, refractive_index)
                for value in DESIGN_HEIGHT_CANDIDATES_UM
            ],
        },
        "thermal_interface_scenarios": scenarios,
        "gates": {
            "analytic_source_boundary_intensity_over_peak_lt": 1.0e-4,
            "analytic_square_captured_fraction_gt": 0.999,
            "realized_waist_relative_error_lt": 0.005,
            "source_fit_NRMSE_lt": 0.005,
            "domain_objective_relative_change_lt": 0.005,
            "domain_gradient_angle_deg_lt": 1.0,
        },
        "analytic_aperture_gate_pass": bool(
            aperture["source_boundary_intensity_over_peak"] < 1.0e-4
            and aperture["fitted_infinite_gaussian_square_captured_fraction"]
            > 0.999
        ),
    }
