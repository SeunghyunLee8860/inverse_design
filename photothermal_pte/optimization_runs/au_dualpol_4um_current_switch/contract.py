"""Immutable physical and numerical contract for the 4 um Au design.

Coordinates follow the established repository convention: Lumerical x is
TaIrTe4 b and Lumerical y is TaIrTe4 a.  The measurement terminals are fixed
top-Au strips included in Maxwell, thermal, and explicit 3-D electrical models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_material_fraction import (
    AU_MATERIAL_FRACTION_EXPONENT,
    AU_MATERIAL_FRACTION_LAW,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.au_density_relaxation import (
    CONTRACT as AU_DENSITY_RELAXATION,
)


@dataclass(frozen=True)
class Contract:
    wavelength_m: float = 4.0e-6
    gaussian_waist_m: float = 4.0e-6
    optical_lateral_span_m: float = 20.0e-6
    source_aperture_span_m: float = 16.0e-6
    reporting_incident_power_W: float = 285.0e-6
    flake_span_x_m: float = 16.0e-6
    flake_span_y_m: float = 16.0e-6
    flake_thickness_m: float = 100.0e-9
    design_span_x_m: float = 8.0e-6
    design_span_y_m: float = 8.0e-6
    design_thickness_m: float = 50.0e-9
    design_pitch_m: float = 100.0e-9
    measurement_electrode_material: str = "Au"
    measurement_electrode_overlap_x_m: float = 1.0e-6
    measurement_electrode_span_y_m: float = 16.0e-6
    measurement_electrode_thickness_m: float = 50.0e-9
    measurement_electrode_contact_S_m2: float = 1.0e10
    electrical_model: str = "explicit_3d_top_contact_volumetric_current_v1"
    tairte4_sigma_z_S_m: float = 1.10e4
    tairte4_sigma_z_scenario: str = (
        "unmeasured_named_baseline_0p1_times_sigma_b; "
        "must_report_1p10e3_1p10e4_1p10e5_S_m_sensitivity"
    )
    tairte4_seebeck_z_V_K: float = 0.0
    tairte4_seebeck_z_scenario: str = (
        "unmeasured_out_of_plane_thermopower_omitted_not_fitted"
    )
    minimum_solid_feature_m: float = 250.0e-9
    minimum_void_feature_m: float = 250.0e-9
    filter_radius_m: float = 250.0e-9
    projection_eta: float = 0.5
    sio2_thickness_m: float = 285.0e-9
    pml_cells_each_face: int = 8
    axis_x: str = "b"
    axis_y: str = "a"
    low_terminal: str = "x_min"
    high_terminal: str = "x_max"
    positive_current_direction: str = "plus_x_internal_current_from_x_min_to_x_max"
    target_e_parallel_a: str = "plus_x_from_x_min_to_x_max"
    target_e_parallel_b: str = "minus_x_from_x_max_to_x_min"
    ta_sio2_thermal_scenario: str = "thermally_grown"
    g_ta_sio2_W_m2K: float = 7.37e6
    g_ta_air_W_m2K: float = 1.0
    au_ta_thermal_contact_scenario: str = "numerical_scenario_Rpp_5p8e-8_m2K_W"
    g_au_ta_W_m2K: float = 1.0 / 5.8e-8
    au_ta_electrical_contact_scenario: str = "numerical_scenario_rhoc_1e-10_ohm_m2"
    electrical_contact_S_m2: float = 1.0e10
    au_bulk_electrical_conductivity_S_m: float = 1.0 / 2.43e-8
    au_bulk_thermal_conductivity_W_mK: float = 317.0
    au_bulk_seebeck_V_K: float = 1.94e-6
    au_bulk_seebeck_reference: str = (
        "Cusack_and_Kendall_1958_Proc_Phys_Soc_72_898_"
        "doi_10.1088/0370-1328/72/5/429_absolute_Au_at_300K"
    )
    au_tairte4_interfacial_seebeck_V_K: float = 0.0
    au_transport_parameter_scope: str = (
        "bulk_room_temperature_reference_not_certified_for_50nm_film"
    )
    au_thermopower_discretization: str = (
        "bulk_isotropic_3d_Au_edges_floor_subtracted"
    )
    au_material_fraction_law: str = AU_MATERIAL_FRACTION_LAW
    au_material_fraction_exponent: float = AU_MATERIAL_FRACTION_EXPONENT
    lumerical_optical_density_law: str = AU_DENSITY_RELAXATION.law
    lumerical_optical_rho_power: float | None = AU_DENSITY_RELAXATION.optical_rho_power
    production_maxwell_route: str = (
        "Lumerical_nk_density_relaxation_then_exact_dispersive_Au_final"
    )
    production_continuous_topology_relaxation_allowed: bool = True
    production_relaxation_is_physical_gray_au: bool = False
    final_geometry_identity: str = (
        "exact 0/1 mask plus physical x/y edges, z bounds, and x=b/y=a mapping"
    )

    @property
    def design_shape(self) -> tuple[int, int]:
        return (
            int(round(self.design_span_x_m / self.design_pitch_m)),
            int(round(self.design_span_y_m / self.design_pitch_m)),
        )

    @property
    def design_node_shape(self) -> tuple[int, int]:
        """Nodal density shape spanning all 80x80 physical cell edges."""

        nx, ny = self.design_shape
        return nx + 1, ny + 1

    @property
    def aperture_boundary_intensity_fraction(self) -> float:
        half = 0.5 * self.source_aperture_span_m
        return math.exp(-2.0 * (half / self.gaussian_waist_m) ** 2)

    @property
    def flake_boundary_intensity_fraction(self) -> float:
        half = 0.5 * min(self.flake_span_x_m, self.flake_span_y_m)
        return math.exp(-2.0 * (half / self.gaussian_waist_m) ** 2)

    def audit(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            design_shape=list(self.design_shape),
            design_node_shape=list(self.design_node_shape),
            source_aperture_boundary_intensity_over_peak=self.aperture_boundary_intensity_fraction,
            flake_boundary_intensity_over_peak=self.flake_boundary_intensity_fraction,
            periodic_boundary=False,
            optical_boundaries="six PML",
            optical_electrodes_included=True,
            electrical_terminal_model=(
                "explicit top Au strips spanning full y: psi=0 on the left "
                "strip and psi=1 on the right strip; finite Au/TaIrTe4 contact "
                "and explicit TaIrTe4 z conduction"
            ),
            au_role=(
                "floating patterned nanostructure; optical absorber/scatterer and "
                "thermal/electrical shunt, not a measurement electrode"
            ),
            au_thermopower_role=(
                "bulk-reference Au Seebeck source is active on 3-D Au volume "
                "edges; the numerical void conductivity floor produces no "
                "thermopower; unknown vertical Au/TaIrTe4 interface thermopower "
                "is not invented and remains zero"
            ),
            gray_density_role=(
                "filtered/projected topology occupancy, not carrier density; "
                "all physics share the same occupancy but use documented "
                "constitutive maps; Lumerical optical uses n-k interpolation "
                "then square with no rho**3; final promotion requires an "
                "exact-binary ordinary dispersive-Au reevaluation"
            ),
            final_geometry_role=(
                "the promoted 0/1 candidate uses one hash-identical physical "
                "Au geometry across Maxwell, thermal, and electrical solvers"
            ),
            objective=(
                "maximize t subject to +I(E||a)>=t and -I(E||b)>=t; "
                "+I is conventional current along solver +x, from x_min to x_max"
            ),
        )
        return payload


CONTRACT = Contract()
