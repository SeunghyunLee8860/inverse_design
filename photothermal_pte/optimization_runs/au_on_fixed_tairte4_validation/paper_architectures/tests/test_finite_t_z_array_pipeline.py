from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
RAW = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_array_Q")


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAP = load("test_array_map", "86_map_finite_t_z_array_material_q.py")
SOLVE = load("test_array_solve", "87_solve_finite_t_z_array_thermal_electrical.py")


def result(case: str) -> dict:
    path = next((RAW / case).glob("FINITE_*_Q.json"))
    return json.loads(path.read_text())


def test_t11x15_rectilinear_decomposition_preserves_exact_area():
    payload = result("T11x15_Ea_Au_on")
    boxes, rectangles = MAP.top_au_boxes(payload)
    assert len(payload["geometry"]["top_Au_polygons"]) == 165
    assert len(boxes) == len(rectangles) == 660
    total_area = sum((box[0][1] - box[0][0]) * (box[1][1] - box[1][0]) for box in boxes)
    assert np.isclose(total_area, 165 * 0.24e-12, rtol=0.0, atol=1e-24)


def test_z1x3_is_one_column_three_rows_and_six_disjoint_rectangles():
    payload = result("Z1x3_Ea_Au_on")
    contract = payload["geometry"]["array_contract"]
    boxes, rectangles = MAP.top_au_boxes(payload)
    assert contract["nx_along_b"] == 1
    assert contract["ny_along_a"] == 3
    assert len(payload["geometry"]["top_Au_polygons"]) == 6
    assert len(boxes) == len(rectangles) == 6
    assert min(item[2] for item in rectangles) == -3.9e-6
    assert max(item[3] for item in rectangles) == 3.9e-6


def test_array_thermal_area_fraction_is_conservative_and_bounded():
    mapping = json.loads((HERE / "results_finite_T_Z_array_material_Q_mapping" / "FINITE_T_Z_ARRAY_MATERIAL_Q_MAPPING_SUMMARY.json").read_text())
    for case in ("T11x15_Ea_Au_on", "Z1x3_Ea_Au_on"):
        rectangles = mapping["cases"][case]["top_Au_rectangles_m"]
        edges = MAP.BASE.thermal_edges(mapping["cases"][case]["architecture"])
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

