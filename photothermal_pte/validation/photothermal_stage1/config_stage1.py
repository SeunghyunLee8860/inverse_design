"""Configuration for the isolated photothermal stage-1 validation.

All values passed to Lumerical use SI units.  Thermal properties for the
physical stack intentionally default to ``None`` unless they were explicitly
provided by the project specification.  The analytic slab is independent of
those physical-stack properties.
"""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
OUTPUT_ROOT = HERE / "output"

LUMERICAL_ROOTS = {
    "v261": Path("/opt/lumerical/v261"),
    "v251": Path("/opt/lumerical/v251"),
}
LUMERICAL_ORDER = ("v261", "v251")

RANDOM_SEED = 20260723

# Optical source normalization target.
INCIDENT_INTENSITY_W_M2 = float(
    os.environ.get("STAGE1_INCIDENT_INTENSITY_W_M2", "1.0e4")
)
WAVELENGTH_M = 4.0e-6
POLARIZATION = "x"
FDTD_SIMULATION_TIME_S = float(
    os.environ.get("STAGE1_FDTD_SIMULATION_TIME_S", "4.0e-12")
)

# Analytic HEAT-only slab.
ANALYTIC_X_SPAN_M = 1.0e-6
ANALYTIC_Y_SPAN_M = 1.0e-6
ANALYTIC_LENGTH_M = 1.0e-6
ANALYTIC_K_W_MK = 10.0
ANALYTIC_Q_W_M3 = 1.0e14
BATH_TEMPERATURE_K = 300.0
ANALYTIC_MESH_TARGETS_M = (1.0e-7, 5.0e-8, 2.5e-8)

# Physical stack.  Geometry values are checked against the imported repository
# builder before use and the actual values are written to every run summary.
DESIGN_THERMAL_MODE = os.environ.get(
    "STAGE1_DESIGN_THERMAL_MODE", "required"
).strip().lower()
if DESIGN_THERMAL_MODE not in {"required", "pipeline_placeholder"}:
    raise ValueError(
        "STAGE1_DESIGN_THERMAL_MODE must be required or pipeline_placeholder"
    )


def _optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    return None if value in (None, "") else float(value)


# The design material has no physical thermal model in the optical repository.
DESIGN_K_W_MK = _optional_float("STAGE1_DESIGN_K_W_MK")
DESIGN_RHO_KG_M3 = _optional_float("STAGE1_DESIGN_RHO_KG_M3")
DESIGN_CP_J_KG_K = _optional_float("STAGE1_DESIGN_CP_J_KG_K")

# These must be explicitly supplied for physical-stack HEAT runs.  No silent
# database/default value is accepted by the stage-1 scripts.
SIO2_K_W_MK = _optional_float("STAGE1_SIO2_K_W_MK")
SIO2_RHO_KG_M3 = _optional_float("STAGE1_SIO2_RHO_KG_M3")
SIO2_CP_J_KG_K = _optional_float("STAGE1_SIO2_CP_J_KG_K")
SI_K_W_MK = _optional_float("STAGE1_SI_K_W_MK")
SI_RHO_KG_M3 = _optional_float("STAGE1_SI_RHO_KG_M3")
SI_CP_J_KG_K = _optional_float("STAGE1_SI_CP_J_KG_K")

# Values explicitly supplied in the project request.
TAIRTE4_K_DIAGONAL_W_MK = (14.4, 3.8, 1.0)
TAIRTE4_ISOTROPIC_OVERRIDE_W_MK = _optional_float("STAGE1_TAIRTE4_ISOTROPIC_K_W_MK")
TAIRTE4_RHO_KG_M3 = _optional_float("STAGE1_TAIRTE4_RHO_KG_M3")
TAIRTE4_CP_J_KG_K = _optional_float("STAGE1_TAIRTE4_CP_J_KG_K")

# Placeholder mode is numerical plumbing only.  Every downstream artifact is
# required to carry PLACEHOLDER_LABEL when this value is used.
PLACEHOLDER_DESIGN_K_W_MK = float(
    os.environ.get("STAGE1_PLACEHOLDER_DESIGN_K_W_MK", "1.0")
)
PLACEHOLDER_LABEL = "NOT_PHYSICAL_PLACEHOLDER_DESIGN_THERMAL_PROPERTY"

# Deterministic fixed binary optical design.  It is created directly as a
# density array; no optimizer, projection, beta continuation, or mapping is
# called.
FIXED_DESIGN_KIND = "centered_binary_disk"
FIXED_DESIGN_DISK_RADIUS_M = 1.5e-6

# Initial validation thresholds.
ANALYTIC_RELATIVE_DT_MAX_LIMIT = 0.01
ANALYTIC_NRMSE_LIMIT = 0.01
ANALYTIC_ENERGY_ERROR_LIMIT = 0.01
OPTICAL_ENERGY_ERROR_LIMIT = 0.05
HEAT_IMPORT_POWER_ERROR_LIMIT = 0.005
HEAT_SCALING_R2_LIMIT = 0.999
HEAT_SCALING_SLOPE_ERROR_LIMIT = 0.02
TRANSIENT_STEADY_ERROR_LIMIT = 0.01
TRANSIENT_TIMESTEP_ERROR_LIMIT = 0.02
