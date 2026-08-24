"""Fail-closed mesh/time convergence contract for exact-Au Lumerical runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class LumericalMeshSpec:
    label: str
    flake_dxy_m: float
    stack_dz_m: float
    outer_dxy_m: float
    mesh_accuracy: int
    pml_layers: int
    lateral_span_m: float
    z_min_m: float
    z_max_m: float
    simulation_time_s: float
    auto_shutoff_min: float
    conformal_mesh: str = "conformal variant 1"

    def validate(self) -> "LumericalMeshSpec":
        if not self.label or any(character.isspace() for character in self.label):
            raise ValueError("mesh label must be nonempty and contain no whitespace")
        for name in ("flake_dxy_m", "stack_dz_m", "outer_dxy_m"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 500.0e-9:
                raise ValueError(f"invalid {name}: {value}")
        if self.outer_dxy_m < self.flake_dxy_m:
            raise ValueError("outer dxy cannot be finer than the flake override")
        if not 1 <= int(self.mesh_accuracy) == self.mesh_accuracy <= 8:
            raise ValueError("mesh_accuracy must be an integer in [1,8]")
        if not 4 <= int(self.pml_layers) == self.pml_layers <= 64:
            raise ValueError("pml_layers must be an integer in [4,64]")
        if self.lateral_span_m < 20.0e-6:
            raise ValueError("lateral span cannot truncate the provisional 20-um domain")
        if not self.z_min_m <= -3.0e-6 < 0.0 < 3.0e-6 <= self.z_max_m:
            raise ValueError("z domain must contain the complete provisional +/-3 um domain")
        if not self.simulation_time_s > 0.0:
            raise ValueError("simulation time must be positive")
        if not 0.0 < self.auto_shutoff_min <= 1.0e-5:
            raise ValueError("auto shutoff must lie in (0,1e-5]")
        if self.conformal_mesh != "conformal variant 1":
            raise ValueError("only conformal variant 1 is authorized before controls")
        return self

    def audit(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


BASELINE = LumericalMeshSpec(
    label="baseline_xy100_z20_pml8_span20_z6_t1ps",
    flake_dxy_m=100.0e-9,
    stack_dz_m=20.0e-9,
    outer_dxy_m=200.0e-9,
    mesh_accuracy=3,
    pml_layers=8,
    lateral_span_m=20.0e-6,
    z_min_m=-3.0e-6,
    z_max_m=3.0e-6,
    simulation_time_s=1.0e-12,
    auto_shutoff_min=1.0e-7,
).validate()

TIME_CANDIDATES_S = (1.0e-12, 2.0e-12, 4.0e-12)
Z_CANDIDATES_M = (20.0e-9, 10.0e-9, 5.0e-9, 2.5e-9)
XY_CANDIDATES_M = (100.0e-9, 50.0e-9, 25.0e-9)
PML_LAYER_CANDIDATES = (8, 12, 16)
LATERAL_SPAN_CANDIDATES_M = (20.0e-6, 24.0e-6, 28.0e-6)
Z_DOMAIN_CANDIDATES_M = ((-3.0e-6, 3.0e-6), (-4.0e-6, 4.0e-6))
POLARIZATIONS = ("Ea", "Eb")
GEOMETRY_CONTROLS = ("empty", "full", "simple_L")

RELATIVE_GATE = 5.0e-3
Q_FLUX_GATE = 2.0e-2
SOURCE_PROFILE_GATE = 5.0e-3


def replace_labeled(spec: LumericalMeshSpec, label: str, **updates: Any) -> LumericalMeshSpec:
    return replace(spec, label=label, **updates).validate()


def candidate_axes() -> dict[str, list[Any]]:
    """Return ordered one-axis-at-a-time candidates, not a Cartesian sweep."""

    return {
        "time_simulation_s": list(TIME_CANDIDATES_S),
        "optical_z_stack_dz_m": list(Z_CANDIDATES_M),
        "optical_xy_flake_dxy_m": list(XY_CANDIDATES_M),
        "pml_layers": list(PML_LAYER_CANDIDATES),
        "lateral_span_m": list(LATERAL_SPAN_CANDIDATES_M),
        "z_domain_bounds_m": [list(item) for item in Z_DOMAIN_CANDIDATES_M],
    }


def convergence_contract_audit() -> dict[str, Any]:
    return {
        "status": "AUDITED_EXACT_AU_LUMERICAL_MESH_MATRIX_NOT_RUN",
        "policy": (
            "sequential one-axis-at-a-time convergence; each next axis uses the "
            "selected prior-axis values; a failed finest pair extends that axis"
        ),
        "baseline": BASELINE.audit(),
        "axis_order": [
            "source_profile_and_incident_power",
            "time_and_auto_shutoff",
            "optical_z_stack",
            "optical_xy_flake_and_Au_edges",
            "PML_layers",
            "lateral_domain_clearance",
            "z_domain_clearance",
        ],
        "candidate_axes": candidate_axes(),
        "required_polarizations": list(POLARIZATIONS),
        "required_exact_geometry_controls": list(GEOMETRY_CONTROLS),
        "per_mesh_source_calibration": {
            "required_for_each_polarization": True,
            "source_only_all_air": True,
            "target_plane_incident_power": True,
            "target_waist_and_Gaussian_profile_fit": True,
            "field_or_Q_rescaling_allowed": False,
            "reporting_power_normalization_only": True,
        },
        "per_material_case_gates": [
            "engine log proves requested B200 and no CPU fallback",
            "actual mesh coordinate readback is available",
            "auto-shutoff and duration-pair stationarity pass",
            "native-Yee Q is finite and unclipped",
            "six-face inward flux agrees with volume Q",
            "Lumerical fitted epsilon readback matches sampled targets",
            "canonical exact-Au geometry SHA matches downstream solvers",
        ],
        "pairwise_metrics": [
            "incident power and realized source profile",
            "total and component/material absorbed power",
            "native-Yee Q volume-L2 NRMSE after conservative remap",
            "TaIrTe4 temperature-rise field NRMSE and Tmax",
            "PTE current magnitude and sign",
        ],
        "gates": {
            "relative_scalar_and_field": RELATIVE_GATE,
            "Q_vs_six_face_flux": Q_FLUX_GATE,
            "source_profile": SOURCE_PROFILE_GATE,
        },
        "promotion": {
            "is_mesh_certificate": False,
            "requires_confirmed_physical_device_contract": True,
            "requires_actual_B200_results": True,
            "requires_all_three_controls_and_both_polarizations": True,
        },
    }
