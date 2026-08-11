from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "legacy_v261_optical_support"
    / "audit_production_candidate_geometry.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("run002_production_geometry", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coarse_design_canvas_nodes_and_bounds() -> None:
    module = load_module()
    x, y, z = module.design_nodes()
    assert (x.size, y.size, z.size) == (201, 201, 21)
    assert np.allclose(x[[0, -1]], [-10e-6, 10e-6], rtol=0.0, atol=1e-18)
    assert np.allclose(y[[0, -1]], [-10e-6, 10e-6], rtol=0.0, atol=1e-18)
    assert np.allclose(z[[0, -1]], [0.0, 1e-6], rtol=0.0, atol=1e-18)
    assert np.isclose(x[1] - x[0], 100e-9, rtol=0.0, atol=1e-18)
    assert np.isclose(z[1] - z[0], 50e-9, rtol=0.0, atol=1e-18)


def test_selected_production_nodes_match_frozen_window() -> None:
    module = load_module()
    x, y, z = module.design_nodes(
        module.SELECTED_DESIGN_BOUNDS,
        module.SELECTED_DESIGN_SHAPE,
    )
    assert (x.size, y.size, z.size) == (373, 373, 21)
    assert np.allclose(x[[0, -1]], [-9.3e-6, 9.3e-6], rtol=0.0, atol=1e-18)
    assert np.allclose(y[[0, -1]], [-9.3e-6, 9.3e-6], rtol=0.0, atol=1e-18)
    assert np.isclose(x[1] - x[0], 50e-9, rtol=0.0, atol=1e-18)
    assert np.isclose(y[1] - y[0], 50e-9, rtol=0.0, atol=1e-18)
    assert np.isclose(z[1] - z[0], 50e-9, rtol=0.0, atol=1e-18)


def test_tairte4_axis_mapping_is_complex_and_passive_at_10um() -> None:
    module = load_module()
    epsilon = module.material_epsilon()
    assert epsilon["x"] == epsilon["z"]
    assert epsilon["x"].imag > 0.0
    assert epsilon["y"].imag > 0.0


def test_q_control_volume_is_symmetric_for_pabs_flux_matching() -> None:
    module = load_module()
    for bounds in module.Q_BOUNDS.values():
        assert np.isclose(bounds[0], -bounds[1], rtol=0.0, atol=1e-18)
