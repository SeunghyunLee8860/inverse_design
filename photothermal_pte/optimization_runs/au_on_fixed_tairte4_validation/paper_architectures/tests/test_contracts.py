from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from contracts import (  # noqa: E402
    Z_PUBLISHED_DIMENSIONS_NM,
    architectures,
    optical_backplane_attenuation,
    reduced_substrate_impedance,
)


def _names(key: str) -> list[str]:
    return [layer.name for layer in architectures()[key].layers]


def test_only_active_2d_material_is_tairte4() -> None:
    for item in architectures().values():
        active = [layer for layer in item.layers if layer.name == "active_2d"]
        assert len(active) == 1
        assert active[0].material == "TaIrTe4"
        assert active[0].thickness_nm == 100.0


def test_2024_T_layer_order() -> None:
    names = _names("B_T_2024_TAIRTE4_SUBSTITUTION")
    assert names.index("inverse_T_resonator") < names.index("active_2d")
    assert names.index("active_2d") < names.index("cavity_spacer")
    assert names.index("cavity_spacer") < names.index("back_reflector")


def test_2022_Z_layer_and_fabrication_order() -> None:
    names = _names("B_Z_2022_TAIRTE4_SUBSTITUTION")
    assert names.index("active_2d") < names.index("chiral_Z_resonator")
    assert names.index("chiral_Z_resonator") < names.index("cavity_spacer")
    assert names.index("cavity_spacer") < names.index("back_reflector")


def test_only_backplane_architectures_allow_optical_substrate_reduction() -> None:
    items = architectures()
    assert not items["A_DIRECT_AU_TAIRTE4"].optical_substrate_reduction_allowed
    assert items["B_T_2024_TAIRTE4_SUBSTITUTION"].optical_substrate_reduction_allowed
    assert items["B_Z_2022_TAIRTE4_SUBSTITUTION"].optical_substrate_reduction_allowed


def test_published_Z_table_exact_endpoints() -> None:
    assert len(Z_PUBLISHED_DIMENSIONS_NM) == 5
    assert Z_PUBLISHED_DIMENSIONS_NM[0] == {
        "metamaterial": "M1",
        "wavelength_nm": 4500.0,
        "P1_nm": 4200.0,
        "P2_nm": 2500.0,
        "L1_nm": 1950.0,
        "L2_nm": 1400.0,
        "W1_nm": 1150.0,
        "W2_nm": 900.0,
        "Al2O3_D_nm": 200.0,
    }
    assert Z_PUBLISHED_DIMENSIONS_NM[-1]["wavelength_nm"] == 8000.0
    assert Z_PUBLISHED_DIMENSIONS_NM[-1]["Al2O3_D_nm"] == 270.0


def test_200nm_Au_is_bulk_optically_opaque_diagnostic() -> None:
    result = optical_backplane_attenuation(200.0)
    assert 11.0 < result["intensity_skin_depth_nm"] < 12.0
    assert result["bulk_intensity_propagation_factor"] < 1.0e-7


def test_reduced_thermal_boundary_is_not_promoted() -> None:
    thin = reduced_substrate_impedance(285.0)
    thick = reduced_substrate_impedance(1500.0)
    assert thin["status"].startswith("UNVALIDATED")
    assert thick["status"].startswith("UNVALIDATED")
    assert thick["candidate_Robin_G_W_m2K"] < thin["candidate_Robin_G_W_m2K"]

