from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "boundary_corner_localization",
    ROOT / "14_analyze_au_boundary_corner_localization.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_decode_complex_field_transposes_engine_zyx_to_xyz(tmp_path: Path) -> None:
    path = tmp_path / "field.h5"
    raw = np.zeros((2, 3, 4, 2), dtype=float)
    raw[..., 0] = np.arange(24).reshape(2, 3, 4)
    raw[..., 1] = -raw[..., 0]
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("E", data=raw)
        decoded = MODULE.decode_complex_field(dataset)
    assert decoded.shape == (4, 3, 2)
    assert decoded[3, 2, 1] == 23.0 - 23.0j


def test_profile_metrics_constant_kernel_has_exact_trapezoid_integral() -> None:
    y = np.linspace(-MODULE.HALF_Y_M, MODULE.HALF_Y_M, 201)
    kernel = np.ones_like(y)
    metrics = MODULE.profile_metrics(y, kernel)
    np.testing.assert_allclose(metrics["full_integral_raw"], 20.0e-6)
    np.testing.assert_allclose(
        metrics["endpoint_fraction_of_full"], 1.0 / (y.size - 1)
    )


def test_relative_is_symmetric() -> None:
    assert MODULE.relative(2.0, 1.0) == MODULE.relative(1.0, 2.0) == 0.5
