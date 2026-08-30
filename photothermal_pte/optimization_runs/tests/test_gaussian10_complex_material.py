from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPOSITORY
    / "photothermal_pte"
    / "optimization_runs"
    / "legacy_v261_optical_support"
    / "run_complex_material_control.py"
)


def load_control():
    spec = importlib.util.spec_from_file_location("run002_complex_control", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complex_material_endpoints_and_gray_are_passive():
    control = load_control()
    eps0, n0 = control.complex_index(0.0)
    eps05, n05 = control.complex_index(0.5)
    eps1, n1 = control.complex_index(1.0)
    assert eps0 == 1.0 + 0.0j
    assert n0 == 1.0 + 0.0j
    assert np.isclose(eps05, 0.5 * (eps0 + eps1))
    assert n05.imag > 0.0 and n1.imag > 0.0
    assert np.isclose(n05**2, eps05)
    assert np.isclose(n1**2, eps1)


def test_imported_complex_material_grid_matches_block_bounds():
    control = load_control()
    x, y, z = control.imported_nodes()
    assert (x.size, y.size, z.size) == (101, 101, 21)
    for axis, values in zip("xyz", (x, y, z)):
        assert np.isclose(values[0], control.BLOCK_BOUNDS[axis][0])
        assert np.isclose(values[-1], control.BLOCK_BOUNDS[axis][1])
        assert np.all(np.diff(values) > 0.0)
