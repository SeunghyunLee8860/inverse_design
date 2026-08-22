"""Traceable geometry contracts for the 2024 T and 2022 Z architectures.

The papers used graphene/BP/PdSe2.  These contracts change only the active
2-D thermoelectric material to the project's fixed 100-nm TaIrTe4 flake.
Unknown contact/topography details and 10-um geometry extrapolations remain
explicit scenarios; they are never silently promoted to paper values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, pi
from pathlib import Path
from typing import Any


WAVELENGTH_M = 10.0e-6
TAIRTE4_THICKNESS_NM = 100.0
AU_ORDAL_K_10UM = 69.2
K_SIO2_W_MK = 1.38
G_SIO2_SI_SCENARIO_W_M2K = 1.1e9

PAPER_2024_MAIN = Path("/home/seunghyun/tairte4/papers/s41467-024-51599-w.pdf")
PAPER_2024_SUPPLEMENT = Path(
    "/home/seunghyun/tairte4/papers/41467_2024_51599_MOESM1_ESM.pdf"
)
PAPER_2022_MAIN = Path("/home/seunghyun/tairte4/papers/s41467-022-32309-w.pdf")
PAPER_2022_SUPPLEMENT = Path(
    "/home/seunghyun/tairte4/papers/41467_2022_32309_MOESM1_ESM.pdf"
)

EXPECTED_PDF_SHA256 = {
    str(PAPER_2024_SUPPLEMENT): "72c2c1264c8d53e4fc22b356fbfd1cf99c229a5b7cabd3f8069f516d156cc2fb",
    str(PAPER_2022_SUPPLEMENT): "927d41f1d6f62916ba15cdf0eb0ec9a37edf457e0cb7b8133365d1d0f13b342b",
}


@dataclass(frozen=True)
class Layer:
    order_top_to_bottom: int
    name: str
    material: str
    thickness_nm: float | None
    role: str
    provenance_kind: str
    provenance: str
    optical_reference: bool = True
    thermal_reference: bool = True


@dataclass(frozen=True)
class Architecture:
    key: str
    title: str
    paper_like: bool
    original_active_materials: tuple[str, ...]
    substituted_active_material: str
    layers: tuple[Layer, ...]
    optical_substrate_reduction_allowed: bool
    optical_substrate_reduction_reason: str
    paper_illumination: str
    project_illumination: str
    unresolved: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


Z_PUBLISHED_DIMENSIONS_NM = (
    {
        "metamaterial": "M1",
        "wavelength_nm": 4500.0,
        "P1_nm": 4200.0,
        "P2_nm": 2500.0,
        "L1_nm": 1950.0,
        "L2_nm": 1400.0,
        "W1_nm": 1150.0,
        "W2_nm": 900.0,
        "Al2O3_D_nm": 200.0,
    },
    {
        "metamaterial": "M2",
        "wavelength_nm": 5300.0,
        "P1_nm": 5100.0,
        "P2_nm": 2600.0,
        "L1_nm": 2300.0,
        "L2_nm": 1700.0,
        "W1_nm": 1360.0,
        "W2_nm": 1100.0,
        "Al2O3_D_nm": 200.0,
    },
    {
        "metamaterial": "M3",
        "wavelength_nm": 6000.0,
        "P1_nm": 5600.0,
        "P2_nm": 2700.0,
        "L1_nm": 2700.0,
        "L2_nm": 2000.0,
        "W1_nm": 1570.0,
        "W2_nm": 1300.0,
        "Al2O3_D_nm": 200.0,
    },
    {
        "metamaterial": "M4",
        "wavelength_nm": 7000.0,
        "P1_nm": 6600.0,
        "P2_nm": 3100.0,
        "L1_nm": 3300.0,
        "L2_nm": 2500.0,
        "W1_nm": 2000.0,
        "W2_nm": 1600.0,
        "Al2O3_D_nm": 230.0,
    },
    {
        "metamaterial": "M5",
        "wavelength_nm": 8000.0,
        "P1_nm": 7600.0,
        "P2_nm": 4100.0,
        "L1_nm": 4000.0,
        "L2_nm": 3000.0,
        "W1_nm": 2600.0,
        "W2_nm": 2100.0,
        "Al2O3_D_nm": 270.0,
    },
)


def architectures() -> dict[str, Architecture]:
    """Return the three non-interchangeable physical contracts."""

    basic = Architecture(
        key="A_DIRECT_AU_TAIRTE4",
        title="Direct floating Au nanoantenna on fixed TaIrTe4 control",
        paper_like=False,
        original_active_materials=("TaIrTe4",),
        substituted_active_material="TaIrTe4",
        layers=(
            Layer(0, "superstrate", "air", None, "illumination half-space", "project", "current Au validation"),
            Layer(1, "design", "Au", 50.0, "floating nanoantenna design", "project", "current Au validation"),
            Layer(2, "active_2d", "TaIrTe4", TAIRTE4_THICKNESS_NM, "PTE flake", "project", "fixed-flake contract"),
            Layer(3, "oxide", "SiO2", 285.0, "explicit optical/thermal substrate", "project_scenario", "current substrate contract"),
            Layer(4, "substrate", "Si", None, "semi-infinite substrate", "project_scenario", "current substrate contract"),
        ),
        optical_substrate_reduction_allowed=False,
        optical_substrate_reduction_reason=(
            "No opaque backplane separates the device from SiO2/Si; replacing the stack "
            "by one substrate changes reflection, phase, and TaIrTe4/Au absorption unless "
            "an endpoint equivalence test is first passed."
        ),
        paper_illumination="not applicable",
        project_illumination="finite scalar Gaussian, lambda=10 um, w0=8.5 um",
        unresolved=("thin-film Au/TaIrTe4 electrical and thermal contact values",),
    )

    t_2024 = Architecture(
        key="B_T_2024_TAIRTE4_SUBSTITUTION",
        title="2024 inverse-T metamaterial-perfect-absorber stack with TaIrTe4",
        paper_like=True,
        original_active_materials=("CVD monolayer graphene",),
        substituted_active_material="TaIrTe4",
        layers=(
            Layer(0, "superstrate", "air", None, "illumination half-space", "paper", "2024 Fig. 1b"),
            Layer(1, "passivation", "Al2O3", 50.0, "top encapsulation; NIR device reference", "paper_device_reference", "2024 Supplementary Fig. 17; MIR value not fixed"),
            Layer(2, "inverse_T_resonator", "Ti/Au", None, "floating asymmetric resonator", "paper_unresolved_thickness", "2024 Fig. 1 and Methods"),
            Layer(3, "active_2d", "TaIrTe4", TAIRTE4_THICKNESS_NM, "graphene replaced only here", "tairte4_substitution", "project fixed-flake contract"),
            Layer(4, "cavity_spacer", "Al2O3", 35.0, "MPA spacer/gate dielectric", "paper", "2024 Supplementary Note 4"),
            Layer(5, "back_reflector", "Au", 200.0, "opaque mirror/gate; thickness is numerical closure", "numerical_closure", "paper specifies Au mirror but not this thickness"),
            Layer(6, "thermal_oxide", "SiO2", 1500.0, "thermally grown oxide around/below buried mirror", "paper", "2024 Methods"),
            Layer(7, "substrate", "intrinsic Si", None, "physical wafer and heat sink", "paper", "2024 Methods"),
        ),
        optical_substrate_reduction_allowed=True,
        optical_substrate_reduction_reason=(
            "The Au mirror is optically opaque. SiO2/Si below it may be omitted from the "
            "Maxwell domain after an Au-thickness/PML convergence gate; they remain in the "
            "explicit thermal reference or are replaced by a validated reduced impedance."
        ),
        paper_illumination="normal-incidence plane wave, periodic unit cell",
        project_illumination="paper unit-cell control first; then finite 10-um Gaussian",
        unresolved=(
            "exact 10-um inverse-T arm dimensions (not published)",
            "2024 MIR T-resonator thickness and passivation at 10 um",
            "Au/Ti-induced TaIrTe4 Seebeck and conductivity change",
        ),
    )

    z_2022 = Architecture(
        key="B_Z_2022_TAIRTE4_SUBSTITUTION",
        title="2022 chiral Z metamaterial stack with dry-transferred TaIrTe4",
        paper_like=True,
        original_active_materials=("graphene", "black phosphorus", "PdSe2"),
        substituted_active_material="TaIrTe4",
        layers=(
            Layer(0, "superstrate", "air", None, "illumination half-space", "paper", "2022 Fig. 1"),
            Layer(1, "active_2d", "TaIrTe4", TAIRTE4_THICKNESS_NM, "dry-transferred 2D material replacement", "tairte4_substitution", "2022 fabrication order + project fixed-flake contract"),
            Layer(2, "chiral_Z_resonator", "Cr/Au", 55.0, "5-nm Cr + 50-nm Au antenna/electrode topography", "paper", "2022 Methods"),
            Layer(3, "cavity_spacer", "Al2O3", None, "M1-M5 use D=200--270 nm", "paper_table", "2022 Supplementary Table 1"),
            Layer(4, "back_reflector", "Au", 200.0, "opaque backplate", "paper", "2022 Methods"),
            Layer(5, "thermal_oxide", "SiO2", 285.0, "thermally grown oxide", "paper", "2022 Methods"),
            Layer(6, "substrate", "heavily p-doped Si", None, "physical wafer and heat sink", "paper", "2022 Methods"),
        ),
        optical_substrate_reduction_allowed=True,
        optical_substrate_reduction_reason=(
            "The published 200-nm Au backplate makes transmission negligible. SiO2/Si "
            "below it may be omitted optically after convergence, but not silently removed "
            "from the heat-removal model."
        ),
        paper_illumination="normal-incidence plane wave, x/y periodic unit cell",
        project_illumination="published unit-cell controls; then finite 10-um Gaussian",
        unresolved=(
            "10-um Z dimensions (published table stops at 8 um)",
            "100-nm TaIrTe4 conformal contact versus bridging over 50-nm Au steps",
            "TaIrTe4/Cr/Au electrical and thermal interface properties",
        ),
    )
    return {item.key: item for item in (basic, t_2024, z_2022)}


def optical_backplane_attenuation(
    thickness_nm: float, wavelength_m: float = WAVELENGTH_M, k: float = AU_ORDAL_K_10UM
) -> dict[str, float]:
    """Bulk-Au propagation diagnostic, not a multilayer transmission solve."""

    thickness_m = thickness_nm * 1.0e-9
    intensity_skin_depth_m = wavelength_m / (4.0 * pi * k)
    return {
        "thickness_nm": float(thickness_nm),
        "wavelength_um": wavelength_m * 1.0e6,
        "Au_k": float(k),
        "intensity_skin_depth_nm": intensity_skin_depth_m * 1.0e9,
        "thickness_in_intensity_skin_depths": thickness_m / intensity_skin_depth_m,
        "bulk_intensity_propagation_factor": exp(-thickness_m / intensity_skin_depth_m),
    }


def reduced_substrate_impedance(
    oxide_thickness_nm: float,
    k_oxide_W_mK: float = K_SIO2_W_MK,
    g_oxide_si_W_m2K: float = G_SIO2_SI_SCENARIO_W_M2K,
) -> dict[str, float | str]:
    """One-dimensional candidate Robin impedance below a backplane.

    This deliberately excludes semi-infinite spreading resistance, so it is a
    screening candidate that must match the explicit 3-D substrate before use.
    """

    oxide_resistance = oxide_thickness_nm * 1.0e-9 / k_oxide_W_mK
    interface_resistance = 1.0 / g_oxide_si_W_m2K
    total = oxide_resistance + interface_resistance
    return {
        "oxide_thickness_nm": float(oxide_thickness_nm),
        "k_oxide_W_mK": float(k_oxide_W_mK),
        "G_oxide_Si_W_m2K": float(g_oxide_si_W_m2K),
        "oxide_resistance_m2K_W": oxide_resistance,
        "interface_resistance_m2K_W": interface_resistance,
        "one_dimensional_total_resistance_m2K_W": total,
        "candidate_Robin_G_W_m2K": 1.0 / total,
        "status": "UNVALIDATED_REDUCED_THERMAL_SUBSTRATE_CANDIDATE",
        "limitation": "does not include 3-D lateral spreading in the Si substrate",
    }


def proposed_10um_seed_sweeps() -> dict[str, Any]:
    """Return named numerical seeds without relabelling them as paper values."""

    return {
        "T_2024": {
            "paper_reference": {
                "resonance_wavelength_um": 4.75,
                "unit_cell_nm": [1500.0, 1000.0],
                "source": "2024 Supplementary Fig. 14",
            },
            "ten_um_initialization_only": {
                "unit_cell_scale_factors_relative_to_4p75um_reference": [1.75, 2.0, 2.25],
                "Al2O3_spacer_nm": [35.0, 75.0, 150.0, 300.0],
                "passivation_nm": [0.0, 50.0],
                "resonator_Au_nm": [33.0, 50.0, 75.0],
                "fixed_Ti_nm": [0.0, 5.0],
                "label": "numerical sweep, not published 10-um dimensions",
            },
        },
        "Z_2022": {
            "paper_reference": Z_PUBLISHED_DIMENSIONS_NM[-1],
            "ten_um_initialization_only": {
                "in_plane_scale_factors_relative_to_M5": [1.0, 1.125, 1.25],
                "Al2O3_spacer_nm": [200.0, 270.0, 350.0, 500.0],
                "TaIrTe4_topography": ["conformal_endpoint", "bridged_endpoint"],
                "label": "numerical sweep, not published 10-um dimensions",
            },
        },
    }

