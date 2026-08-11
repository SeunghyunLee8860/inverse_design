from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "legacy_v261_optical_support"
    / "build_nonuniform_complex_yee_jacobian.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("run002_nonuniform_jacobian", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nonuniform_baseline_has_fd_margin_and_is_reproducible() -> None:
    module = load_module()
    first = module.baseline_density()
    second = module.baseline_density()
    assert first.shape == (101, 101)
    assert np.array_equal(first, second)
    assert np.min(first) > 0.2
    assert np.max(first) < 0.8


def test_complex_density_law_uses_passive_square_root() -> None:
    module = load_module()
    for rho in (0.0, 0.5, 1.0):
        density = np.full((101, 101), rho)
        index = module.imported_index(density)
        expected = 1.0 + rho * (module.epsilon_sio2() - 1.0)
        assert index.shape == (101, 101, 21)
        assert np.all(index.imag >= 0.0)
        assert np.allclose(index**2, expected, rtol=1e-14, atol=1e-14)
