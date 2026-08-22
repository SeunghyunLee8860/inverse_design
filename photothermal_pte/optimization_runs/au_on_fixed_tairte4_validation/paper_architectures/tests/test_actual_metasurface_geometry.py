from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parents[1]


def load_geometry_module():
    path = HERE / "05_actual_metasurface_geometry.py"
    spec = importlib.util.spec_from_file_location("test_actual_geometry_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_inverse_t_polygon_is_inside_disclosed_unit_cell() -> None:
    module = load_geometry_module()
    geometry = module.inverse_t_mir_4750nm()
    vertices = np.asarray(geometry.polygons[0].vertices_nm, float)
    assert np.min(vertices[:, 0]) >= -0.5 * geometry.period_x_nm
    assert np.max(vertices[:, 0]) <= 0.5 * geometry.period_x_nm
    assert np.min(vertices[:, 1]) >= -0.5 * geometry.period_y_nm
    assert np.max(vertices[:, 1]) <= 0.5 * geometry.period_y_nm
    assert abs(module.signed_polygon_area_nm2(geometry.polygons[0].vertices_nm)) == 240_000.0


def test_tairte4_axis_mapping_and_thickness_are_fixed() -> None:
    module = load_geometry_module()
    for geometry in (
        module.inverse_t_mir_4750nm(),
        module.z_m5_8um_geometry_topology_audit(),
    ):
        assert geometry.active_material == "TaIrTe4"
        assert geometry.active_thickness_nm == 100.0
        assert geometry.axis_mapping == {"x": "b", "y": "a", "z": "c=b closure"}


def test_z_dimension_envelopes_cannot_be_promoted_to_cad() -> None:
    module = load_geometry_module()
    geometry = module.z_m5_8um_geometry_topology_audit()
    assert "TOPOLOGY_BLOCKED" in geometry.key
    assert all(item.provenance_kind == "dimension_envelope_only" for item in geometry.polygons)
    assert any("forbidden" in item for item in geometry.unresolved)
