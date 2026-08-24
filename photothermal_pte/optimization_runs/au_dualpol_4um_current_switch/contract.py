"""Immutable physical and numerical contract for the 4 um Au design.

Coordinates follow the established repository convention: Lumerical x is
TaIrTe4 b and Lumerical y is TaIrTe4 a.  The electrical measurement terminals
are boundary conditions on the fixed flake, not optically modelled Au pads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    AU_MATERIAL_FRACTION_EXPONENT,
    AU_MATERIAL_FRACTION_LAW,
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
    minimum_solid_feature_m: float = 500.0e-9
    minimum_void_feature_m: float = 500.0e-9
    filter_radius_m: float = 500.0e-9
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
    au_material_fraction_law: str = AU_MATERIAL_FRACTION_LAW
    au_material_fraction_exponent: float = AU_MATERIAL_FRACTION_EXPONENT
    production_maxwell_route: str = "Lumerical_exact_dispersive_Au_geometry"
    production_gray_au_allowed: bool = False
    production_geometry_identity: str = (
        "exact 0/1 mask plus physical x/y edges, z bounds, and x=b/y=a mapping"
    )

    @property
    def design_shape(self) -> tuple[int, int]:
        return (
            int(round(self.design_span_x_m / self.design_pitch_m)),
            int(round(self.design_span_y_m / self.design_pitch_m)),
        )

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
            source_aperture_boundary_intensity_over_peak=self.aperture_boundary_intensity_fraction,
            flake_boundary_intensity_over_peak=self.flake_boundary_intensity_fraction,
            periodic_boundary=False,
            optical_boundaries="six PML",
            optical_electrodes_included=False,
            electrical_terminal_model=(
                "Dirichlet weighting potential psi=0 at x_min and psi=1 at x_max "
                "on the fixed TaIrTe4 flake"
            ),
            au_role=(
                "floating patterned nanostructure; optical absorber/scatterer and "
                "thermal/electrical shunt, not a measurement electrode"
            ),
            gray_density_role=(
                "historical FDTDX consistency diagnostic only; the legacy "
                "optical, thermal, and electrical operators share one linear "
                "fraction, but no gray field is authorized for production"
            ),
            production_geometry_role=(
                "every Maxwell/thermal/electrical evaluation consumes one "
                "hash-identical exact Au geometry; solver cut cells are only "
                "converged discretizations of that fixed physical boundary"
            ),
            objective=(
                "maximize t subject to +I(E||a)>=t and -I(E||b)>=t; "
                "+I is conventional current along solver +x, from x_min to x_max"
            ),
        )
        return payload


CONTRACT = Contract()
