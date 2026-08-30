from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
MAPPING_PATH = HERE / "results_finite_T_Z_array_material_Q_mapping" / "FINITE_T_Z_ARRAY_MATERIAL_Q_MAPPING_SUMMARY.json"


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAP = load("test_array_map", "86_map_finite_t_z_array_material_q.py")
SOLVE = load("test_array_solve", "87_solve_finite_t_z_array_thermal_electrical.py")


def mapping() -> dict:
    return json.loads(MAPPING_PATH.read_text())


def test_t11x15_rectilinear_decomposition_preserves_exact_area():
    case = mapping()["cases"]["T11x15_Ea_Au_on"]
    primitive = case["array_contract"]["primitive"]["polygons"][0]
    primitive_rectangles = MAP.polygon_rectangles_m(primitive)
    rectangles = case["top_Au_rectangles_m"]
    assert case["array_contract"]["nx_along_b"] == 11
    assert case["array_contract"]["ny_along_a"] == 15
    assert len(primitive_rectangles) == 4
    assert len(rectangles) == 660
    total_area = sum((item[1] - item[0]) * (item[3] - item[2]) for item in rectangles)
    assert np.isclose(total_area, 165 * 0.24e-12, rtol=0.0, atol=1e-24)


def test_z1x3_is_one_column_three_rows_and_six_disjoint_rectangles():
    case = mapping()["cases"]["Z1x3_Ea_Au_on"]
    contract = case["array_contract"]
    rectangles = case["top_Au_rectangles_m"]
    assert contract["nx_along_b"] == 1
    assert contract["ny_along_a"] == 3
    assert len(rectangles) == 6
    assert min(item[2] for item in rectangles) == -3.9e-6
    assert max(item[3] for item in rectangles) == 3.9e-6


def test_array_thermal_area_fraction_is_conservative_and_bounded():
    published = mapping()
    for case in ("T11x15_Ea_Au_on", "Z1x3_Ea_Au_on"):
        rectangles = published["cases"][case]["top_Au_rectangles_m"]
        edges = MAP.BASE.thermal_edges(published["cases"][case]["architecture"])
        fraction = SOLVE.overlap_fraction(edges[0], edges[1], rectangles)
        area = np.diff(edges[0])[:, None] * np.diff(edges[1])[None, :]
        exact = sum((item[1] - item[0]) * (item[3] - item[2]) for item in rectangles)
        assert np.min(fraction) >= 0.0
        assert np.max(fraction) <= 1.0
        assert np.isclose(np.sum(fraction * area), exact, rtol=0.0, atol=1e-24)


def test_published_array_summary_is_fail_closed_and_validated():
    path = HERE / "results_finite_T_Z_array_multiphysics_summary" / "FINITE_T_Z_ARRAY_MULTIPHYSICS_SUMMARY.json"
    summary = json.loads(path.read_text())
    assert summary["status"] == "VALIDATED_FINITE_T11X15_Z1X3_MAXWELL_THERMAL_ELECTRICAL_AU_EFFECT_FORWARD"
    assert all(summary["gates"].values())
    assert len(summary["primary_results"]) == 4
