"""Fail-closed mesh/time convergence contract for exact-Au Lumerical runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


MESH_REFINEMENT_CANDIDATES = (
    "conformal variant 0",
    "conformal variant 1",
    "staircase",
)


@dataclass(frozen=True)
class LumericalMeshSpec:
    label: str
    flake_dxy_m: float
    stack_dz_m: float
    bulk_dz_m: float
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
        # CLI inputs are commonly expressed in nm/um and multiplied back to
        # metres.  Keep a sub-attometre allowance for that unit round trip;
        # it is many orders below every candidate mesh interval.
        unit_roundoff_m = 1.0e-18
        if not self.label or any(character.isspace() for character in self.label):
            raise ValueError("mesh label must be nonempty and contain no whitespace")
        for name in (
            "flake_dxy_m",
            "stack_dz_m",
            "bulk_dz_m",
            "outer_dxy_m",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 500.0e-9:
                raise ValueError(f"invalid {name}: {value}")
        if self.outer_dxy_m < self.flake_dxy_m:
            raise ValueError("outer dxy cannot be finer than the flake override")
        if self.bulk_dz_m < self.stack_dz_m:
            raise ValueError("bulk dz cannot be finer than the stack override")
        if not 1 <= int(self.mesh_accuracy) == self.mesh_accuracy <= 8:
            raise ValueError("mesh_accuracy must be an integer in [1,8]")
        if not 4 <= int(self.pml_layers) == self.pml_layers <= 64:
            raise ValueError("pml_layers must be an integer in [4,64]")
        if self.lateral_span_m < 20.0e-6 - unit_roundoff_m:
            raise ValueError("lateral span cannot truncate the provisional 20-um domain")
        if not (
            self.z_min_m <= -3.0e-6 + unit_roundoff_m
            and 3.0e-6 - unit_roundoff_m <= self.z_max_m
        ):
            raise ValueError("z domain must contain the complete provisional +/-3 um domain")
        if not self.simulation_time_s > 0.0:
            raise ValueError("simulation time must be positive")
        if not 0.0 < self.auto_shutoff_min <= 1.0e-5:
            raise ValueError("auto shutoff must lie in (0,1e-5]")
        if self.conformal_mesh not in MESH_REFINEMENT_CANDIDATES:
            raise ValueError(
                "mesh refinement must be one of "
                f"{MESH_REFINEMENT_CANDIDATES}, got {self.conformal_mesh!r}"
            )
        return self

    def audit(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


BASELINE = LumericalMeshSpec(
    label="baseline_xy100_z20_pml8_span20_z6_t1ps",
    flake_dxy_m=100.0e-9,
    stack_dz_m=20.0e-9,
    bulk_dz_m=200.0e-9,
    outer_dxy_m=200.0e-9,
    mesh_accuracy=3,
    pml_layers=8,
    lateral_span_m=20.0e-6,
    z_min_m=-3.0e-6,
    z_max_m=3.0e-6,
    simulation_time_s=1.0e-12,
    auto_shutoff_min=1.0e-7,
).validate()

# One-step calibration from the all-air v261 baseline runs on both Ea and Eb.
# This is the source-object setting; the required realized flake-plane waist
# remains exactly 4 um and is remeasured for every numerical mesh.
BASELINE_SOURCE_OBJECT_W0_UM = 3.956143303046143

TIME_CANDIDATES_S = (1.0e-12, 2.0e-12, 4.0e-12)
FULL_Z_CANDIDATES_M = (
    (20.0e-9, 200.0e-9),
    (10.0e-9, 100.0e-9),
    (5.0e-9, 50.0e-9),
    (2.5e-9, 25.0e-9),
)
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

    baseline_pml_clearance = (
        BASELINE.pml_layers + 1
    ) * BASELINE.outer_dxy_m
    fixed_flux_half_span = 0.5 * BASELINE.lateral_span_m - baseline_pml_clearance
    return {
        "time_simulation_s": list(TIME_CANDIDATES_S),
        "optical_full_domain_z_m": [
            {"stack_dz_m": stack, "bulk_air_pml_dz_m": bulk}
            for stack, bulk in FULL_Z_CANDIDATES_M
        ],
        "optical_xy_flake_dxy_m": list(XY_CANDIDATES_M),
        # Au has |epsilon| much larger than air/TaIrTe4 at 4 um. Ansys warns
        # that CV1 can create metal-interface artifacts in this regime and
        # requires convergence comparison with the default CV0/staircase
        # treatment. Hold the already selected spatial grid fixed here.
        "metal_interface_mesh_refinement": list(MESH_REFINEMENT_CANDIDATES),
        # Hold the non-PML physical interior and flux surface fixed while the
        # absorbing-layer count changes.  Increasing PML layers inside a fixed
        # 20-um domain would otherwise consume the 16-um flake/control box.
        "pml_layers": [
            {
                "pml_layers": layers,
                "lateral_span_m": 2.0
                * (
                    fixed_flux_half_span
                    + (layers + 1) * BASELINE.outer_dxy_m
                ),
            }
            for layers in PML_LAYER_CANDIDATES
        ],
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
            "optical_z_full_domain_stack_bulk_air_and_PML",
            "metal_interface_mesh_refinement_CV0_CV1_staircase",
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
            "CV0/CV1/staircase metal-interface method is explicitly selected and converged",
            "stack plus Si-bulk/air/PML z-step readback meets the requested full-domain limits",
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
