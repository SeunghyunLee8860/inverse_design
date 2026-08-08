from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "run_002_gaussian10_w8p5_current_max" / "audit_production_material_q_attribution.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run002_material_q_attribution", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_nodal_dual_cells_use_trapezoid_end_weights() -> None:
    module = load_module()
    nodes = np.asarray([0.0, 1.0, 3.0])
    assert np.array_equal(module.nodal_edges(nodes), [0.0, 0.5, 2.0, 3.0])


def test_literal_box_partition_has_no_gain() -> None:
    module = load_module()
    nodes = np.asarray([0.0, 1.0, 2.0])
    edges = tuple(module.nodal_edges(nodes) for _ in range(3))
    density = np.ones((3, 3, 3))
    domain = {axis: (0.0, 2.0) for axis in "xyz"}
    left = {"x": (0.0, 1.0), "y": (0.0, 2.0), "z": (0.0, 2.0)}
    right = {"x": (1.0, 2.0), "y": (0.0, 2.0), "z": (0.0, 2.0)}
    total = module.box_power(density, edges, domain)
    assert total == 8.0
    assert module.box_power(density, edges, left) + module.box_power(density, edges, right) == total


def test_default_design_half_span_remains_backward_compatible() -> None:
    module = load_module()
    assert np.isclose(module.DEFAULT_DESIGN_HALF_SPAN_M, 10.0e-6)
