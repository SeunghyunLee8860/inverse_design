from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_exact_au import (
    AU_MATERIAL,
    MATERIAL_FIT_WAVELENGTH_BAND_M,
    SIO2_MATERIAL,
    SOURCE_WAVELENGTH_BAND_M,
    TAIRTE4_MATERIAL,
    add_dispersive_materials,
    add_exact_stack_geometry,
    control_geometry_audits,
    design_edges,
    exact_control_masks,
    mask_rectangles,
    material_contract_audit,
    sampled_material_data,
)


class _FakeFdtd:
    def __init__(self) -> None:
        self.material_types: list[str] = []
        self.material_properties: dict[tuple[object, str], object] = {}
        self.rectangles: list[dict[str, object]] = []

    def addmaterial(self, material_type: str) -> str:
        self.material_types.append(material_type)
        return f"material_{len(self.material_types)}"

    def setmaterial(self, material: object, property_name: str, value: object) -> None:
        self.material_properties[(material, property_name)] = value

    def addrect(self) -> dict[str, object]:
        rectangle: dict[str, object] = {}
        self.rectangles.append(rectangle)
        return rectangle


def _reconstruct(rectangles: list[dict[str, float]]) -> np.ndarray:
    x, y = design_edges()
    result = np.zeros(CONTRACT.design_shape, dtype=np.uint8)
    for rectangle in rectangles:
        ix = np.flatnonzero(
            (x[:-1] >= rectangle["x_min_m"] - 1e-18)
            & (x[1:] <= rectangle["x_max_m"] + 1e-18)
        )
        iy = np.flatnonzero(
            (y[:-1] >= rectangle["y_min_m"] - 1e-18)
            & (y[1:] <= rectangle["y_max_m"] + 1e-18)
        )
        assert not np.any(result[np.ix_(ix, iy)])
        result[np.ix_(ix, iy)] = 1
    return result


def test_exact_control_masks_round_trip_through_nonoverlapping_rectangles() -> None:
    x, y = design_edges()
    z = np.asarray([0.0, CONTRACT.design_thickness_m])
    masks = exact_control_masks()
    assert set(masks) == {"empty", "full", "simple_L"}
    for name, mask in masks.items():
        rectangles = mask_rectangles(
            mask, x_edges_m=x, y_edges_m=y, z_bounds_m=z
        )
        assert np.array_equal(_reconstruct(rectangles), mask), name
    assert len(
        mask_rectangles(masks["empty"], x_edges_m=x, y_edges_m=y, z_bounds_m=z)
    ) == 0
    assert len(
        mask_rectangles(masks["full"], x_edges_m=x, y_edges_m=y, z_bounds_m=z)
    ) == 1


def test_control_geometry_hashes_are_unique_and_bind_100nm_grid() -> None:
    audits = control_geometry_audits()
    assert len({item["geometry_sha256"] for item in audits.values()}) == 3
    for item in audits.values():
        assert item["mask_shape_xy"] == list(CONTRACT.design_shape)
        assert item["axis_mapping"] == {"x": "b", "y": "a"}
        assert item["z_bounds_m"] == [0.0, CONTRACT.design_thickness_m]


def test_sampled_material_contract_is_dispersive_passive_and_guarded() -> None:
    data = sampled_material_data()
    assert data["frequency_hz"].size == 161
    assert np.all(np.diff(data["frequency_hz"]) > 0.0)
    assert SOURCE_WAVELENGTH_BAND_M[0] > MATERIAL_FIT_WAVELENGTH_BAND_M[0]
    assert SOURCE_WAVELENGTH_BAND_M[1] < MATERIAL_FIT_WAVELENGTH_BAND_M[1]
    assert np.ptp(data["epsilon_au"].real) > 1.0
    assert np.ptp(data["epsilon_au"].imag) > 1.0
    for key, values in data.items():
        assert np.all(np.isfinite(values)), key
        if key.startswith("epsilon_"):
            assert np.all(values.imag >= 0.0), key
    assert np.array_equal(data["epsilon_ta_x_b"], data["epsilon_ta_z_c"])
    audit = material_contract_audit()
    assert audit["status"].endswith("NOT_FIT_READBACK")
    assert audit["gates"]["single_frequency_constant_nk_Au_prohibited"] is True
    assert audit["Au_fit"]["requires_post_run_Lumerical_fit_readback"] is True
    assert audit["default_non_Au_fit"][
        "requires_post_run_Lumerical_fit_readback"
    ] is True


def test_layout_builder_uses_sampled_materials_and_exact_au_prisms_only() -> None:
    fdtd = _FakeFdtd()
    material_audit = add_dispersive_materials(fdtd)
    assert fdtd.material_types == ["Sampled data", "Sampled data", "Sampled 3D data"]
    assert "(n,k) Material" not in fdtd.material_types
    assert material_audit["status"].endswith("NOT_FIT_READBACK")
    geometry = add_exact_stack_geometry(fdtd, exact_control_masks()["simple_L"])
    au_rectangles = [item for item in fdtd.rectangles if item["material"] == AU_MATERIAL]
    assert len(au_rectangles) == geometry["Au_rectangle_count"]
    assert len(au_rectangles) > 0
    assert geometry["status"] == "PROVISIONAL_UNCONFIRMED_DEVICE_GEOMETRY"
    assert geometry["exact_au_geometry"]["occupied_cell_count"] == 414


def test_au_fit_sweep_does_not_change_other_sampled_material_settings() -> None:
    fdtd = _FakeFdtd()
    audit = add_dispersive_materials(
        fdtd,
        au_max_coefficients=6,
        au_fit_tolerance=0.125,
    )
    assert fdtd.material_properties[(AU_MATERIAL, "max coefficients")] == 6
    assert fdtd.material_properties[(AU_MATERIAL, "tolerance")] == 0.125
    assert fdtd.material_properties[(SIO2_MATERIAL, "max coefficients")] == 20
    assert fdtd.material_properties[(SIO2_MATERIAL, "tolerance")] == 0.0
    assert fdtd.material_properties[(TAIRTE4_MATERIAL, "max coefficients")] == 20
    assert fdtd.material_properties[(TAIRTE4_MATERIAL, "tolerance")] == 0.0
    assert audit["Au_fit"]["max_coefficients"] == 6
    assert audit["Au_fit"]["tolerance"] == 0.125


def test_invalid_au_fit_parameters_fail_closed() -> None:
    for max_coefficients, tolerance in ((0, 0.0), (21, 0.0), (6, -1.0)):
        with np.testing.assert_raises(ValueError):
            material_contract_audit(
                au_max_coefficients=max_coefficients,
                au_fit_tolerance=tolerance,
            )
