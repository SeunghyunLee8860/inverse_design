from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CARRIER = load(
    "au_temperature_density_endpoint",
    "31_run_au_temperature_density_endpoint_control.py",
)


def test_box_tetrahedra_cover_box_exactly() -> None:
    bounds = {
        "x": (-2.0, 3.0),
        "y": (-5.0, 7.0),
        "z": (11.0, 13.0),
    }
    vertices, connectivity = CARRIER.box_tetrahedra(bounds)
    assert vertices.shape == (8, 3)
    assert connectivity.shape == (6, 4)
    assert np.min(connectivity) == 1
    assert np.max(connectivity) == 8

    volume = 0.0
    for tetrahedron in connectivity.astype(int) - 1:
        p0, p1, p2, p3 = vertices[tetrahedron]
        volume += abs(np.linalg.det(np.stack((p1 - p0, p2 - p0, p3 - p0)))) / 6.0
    expected = np.prod([hi - lo for lo, hi in bounds.values()])
    np.testing.assert_allclose(volume, expected, rtol=1.0e-15)


def test_json_safe_preserves_complex_parts_and_array_shape() -> None:
    assert CARRIER.json_safe(2.0 + 3.0j) == [2.0, 3.0]
    assert CARRIER.json_safe(np.float64(4.0)) == 4.0
    assert CARRIER.json_safe(np.asarray([[1.0, 2.0]])) == [[1.0, 2.0]]


def test_exact_au_endpoint_and_span_are_explicit() -> None:
    epsilon = complex(CARRIER.N_AU, CARRIER.K_AU) ** 2
    np.testing.assert_allclose(epsilon, complex(-4642.23, 1674.64))
    assert CARRIER.CARRIER_SPAN_K > 0.0
    assert CARRIER.T_REF_K == 300.0
