from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_z16_interface_diagnostic import (
    weighted_error_diagnostic,
)


def test_weighted_error_diagnostic_identical_and_scaled_fields():
    field = np.ones((2, 2, 8), dtype=np.complex128) * (1.0 + 2.0j)
    weights = np.ones_like(field, dtype=np.float64)
    identical = weighted_error_diagnostic(field, field, weights)
    assert identical["complex_E_NRMSE"] == 0.0
    assert identical["scale_aligned_complex_E_NRMSE"] < 1.0e-14

    scaled = weighted_error_diagnostic(field, 2.0 * field, weights)
    assert scaled["complex_E_NRMSE"] == pytest.approx(0.5)
    assert scaled["best_scale_amplitude"] == pytest.approx(0.5)
    assert scaled["scale_aligned_complex_E_NRMSE"] < 1.0e-14


def test_boundary_concentration_detects_boundary_only_error():
    coarse = np.ones((2, 2, 8), dtype=np.complex128)
    fine = coarse.copy()
    fine[..., 0] += 1.0
    fine[..., -1] += 1.0
    result = weighted_error_diagnostic(coarse, fine, np.ones(coarse.shape))
    assert result["boundary_concentration"]["1"]["error_fraction"] == 1.0
    assert result["trimmed_boundary_complex_E_NRMSE"]["1"] == 0.0


def test_weighted_error_diagnostic_rejects_bad_shape_or_weight():
    field = np.ones((2, 2, 4), dtype=np.complex128)
    with pytest.raises(ValueError):
        weighted_error_diagnostic(field, field[..., :3], np.ones(field.shape))
    bad = np.ones(field.shape)
    bad[0, 0, 0] = -1.0
    with pytest.raises(ValueError):
        weighted_error_diagnostic(field, field, bad)


def test_single_plane_probe_boundary_diagnostic_is_valid():
    coarse = np.ones((2, 2, 1), dtype=np.complex128)
    fine = coarse * 1.1
    result = weighted_error_diagnostic(coarse, fine, np.ones(coarse.shape))
    assert result["complex_E_NRMSE"] > 0.0
    assert result["boundary_concentration"]["1"]["error_fraction"] == 1.0
    assert result["boundary_concentration"]["2"]["error_fraction"] == 1.0
    assert result["trimmed_boundary_complex_E_NRMSE"] == {}
