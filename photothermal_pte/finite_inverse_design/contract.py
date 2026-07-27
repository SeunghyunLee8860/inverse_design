"""Immutable constants for the approved finite 2 um certificate."""

from __future__ import annotations

import numpy as np


DESIGN_SPAN_M = 2.0e-6
FLAKE_THICKNESS_M = 100.0e-9
DESIGN_HEIGHT_M = 600.0e-9
BOTTOM_SIO2_THICKNESS_M = 285.0e-9
OPTICAL_SI_DEPTH_M = 2.0e-6
THERMAL_SI_DEPTH_M = 20.0e-6

FLAKE_BOUNDS_Z_M = (-100.0e-9, 0.0)
DESIGN_BOUNDS_M = {
    "x": (-1.0e-6, 1.0e-6),
    "y": (-1.0e-6, 1.0e-6),
    "z": (0.0, 600.0e-9),
}
BOTTOM_SIO2_BOUNDS_Z_M = (-385.0e-9, -100.0e-9)

DESIGN_DX_M = 25.0e-9
DESIGN_DY_M = 25.0e-9
DESIGN_DZ_M = 50.0e-9
DESIGN_NX = 81
DESIGN_NY = 81
DESIGN_NZ = 13
FILTER_RADIUS_M = 500.0e-9
PROJECTION_ETA = 0.5
CERTIFICATE_BETA = 8.0

SOURCE_WAVELENGTH_START_M = 3.0e-6
SOURCE_WAVELENGTH_STOP_M = 6.0e-6
ANALYSIS_WAVELENGTH_M = 4.0e-6
INCIDENT_INTENSITY_W_M2 = 1.0
PML_LAYERS = 24
FLAKE_DZ_M = 5.0e-9
OPTICAL_SOURCE_STATUS = (
    "VALIDATED_EMPTY_AIR_CPU_TFSF_SOURCE_GATE_DEVICE_PENDING"
)
OPTICAL_SOURCE = (
    "CPU TFSF normal-incidence candidate: protected empty-air ROI gate "
    "passed; finite-flake device geometry still pending"
)
OPTICAL_ROI_BOUNDS_M = {
    "x": (-1.0e-6, 1.0e-6),
    "y": (-1.0e-6, 1.0e-6),
}
OPTICAL_DOMAIN_SPAN_CANDIDATES_M = (4.0e-6, 6.0e-6, 8.0e-6)
PML_LAYER_CANDIDATES = (16, 24, 32)
LARGE_FLAKE_SPAN_CANDIDATES_M = (4.0e-6, 6.0e-6)


def flake_bounds_m(span_m: float) -> dict[str, tuple[float, float]]:
    """Bounds for a named large-flake numerical scenario."""
    if span_m not in LARGE_FLAKE_SPAN_CANDIDATES_M:
        raise ValueError(f"undeclared flake span: {span_m}")
    half = 0.5 * span_m
    return {
        "x": (-half, half),
        "y": (-half, half),
        "z": FLAKE_BOUNDS_Z_M,
    }

N_AIR = 1.0
N_SIO2 = 1.38

T_BATH_K = 300.0
KAPPA_TAIRTE4_W_MK = np.asarray([14.4, 3.8, 1.0])
KAPPA_SIO2_W_MK = 1.38
KAPPA_SI_W_MK = 145.0
KAPPA_AIR_W_MK = 0.026
G_TAIRTE4_AIR_W_M2K = 1.0
G_TAIRTE4_BOTTOM_SIO2_W_M2K = 7.37e6
G_TAIRTE4_DEPOSITED_SIO2_W_M2K = 7.37e4
G_SIO2_SI_W_M2K = 1.1e9
H_SIO2_AIR_W_M2K = 10.0

SIGMA_A_S_M = 4.91e5
SIGMA_B_S_M = 1.10e5
SEEBECK_A_V_K = -6.0e-6
SEEBECK_B_V_K = 27.0e-6

# Opposite diagonal equipotential lines span 2*sqrt(2) um. For a unit
# weighting-potential difference, grad(psi)=(xhat+yhat)/(4 um).
WEIGHTING_FIELD_X_M_INV = 1.0 / (4.0e-6)
WEIGHTING_FIELD_Y_M_INV = 1.0 / (4.0e-6)

PRIMARY_FD_STEP = 0.005
DIAGNOSTIC_FD_STEPS = (0.0025, 0.01)
