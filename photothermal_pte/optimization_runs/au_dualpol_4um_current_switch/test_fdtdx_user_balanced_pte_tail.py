from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_user_balanced_pte_tail_case as case,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_user_balanced_pte_tail_certificate as certificate,
)


def _report(current_A: float = 2.0, scale: float = 1.0) -> dict:
    return {
        "normalization": {"mapped_scaled_absorbed_power_W": scale},
        "thermal_solution": {
            "ta_max_temperature_rise_K": 2.0 * scale,
            "ta_mean_temperature_rise_K": scale,
            "ta_gradient_combined_l2_K_m": 3.0 * scale,
        },
        "pte_solution": {"signed_current_A": current_A * scale},
    }


def _fields(scale: float = 1.0) -> dict[str, np.ndarray]:
    coordinate = np.arange(160, dtype=np.float64)
    return {
        "source_power_xy_W": scale * np.ones((266, 266)),
        "ta_temperature_rise_K": scale * np.ones((160, 160)),
        "ta_gradient_x_K_m": scale * np.ones((160, 160)),
        "ta_gradient_y_K_m": 2.0 * scale * np.ones((160, 160)),
        "pte_current_density_A_m2": scale * np.ones((160, 160)),
        "ta_electrical_weighting_V": np.ones((160, 160)),
        "ta_x_centers_m": coordinate,
        "ta_y_centers_m": coordinate.copy(),
    }


def test_case_locks_one_common_downstream_mesh_and_never_promotes():
    assert case.THERMAL_XY_REFINEMENT_FACTOR == 2
    assert case.THERMAL_Z_REFINEMENT_FACTOR == 2
    assert case.THERMAL_DOMAIN == {
        "lateral_half_span_um": 48,
        "substrate_depth_um": 30,
        "top_air_height_um": 3.0,
    }
    assert case.EXPECTED_THERMAL_SHAPE == (548, 548, 72)
    assert "DIAGNOSTIC" in case.STATUS_READY


def test_block_restriction_mean_and_sum_are_explicit():
    value = np.arange(16, dtype=float).reshape(4, 4)
    mean = case._restrict_blocks(value, 2, reduction="mean")
    total = case._restrict_blocks(value, 2, reduction="sum")
    np.testing.assert_allclose(total, 4.0 * mean)
    with pytest.raises(ValueError, match="mean or sum"):
        case._restrict_blocks(value, 2, reduction="maximum")


def test_identical_pte_tail_pair_passes_every_metric_and_sign_gate():
    result = certificate.compare_pair(
        _report(), _report(), _fields(), _fields()
    )
    assert result["pass"] is True
    assert all(value == 0.0 for value in result["metrics"].values())
    assert all(result["threshold_checks"].values())
    assert all(result["invariant_checks"].values())


def test_current_sign_flip_fails_even_if_absolute_fields_match():
    result = certificate.compare_pair(
        _report(current_A=2.0),
        _report(current_A=-2.0),
        _fields(),
        _fields(),
    )
    assert result["pass"] is False
    assert result["invariant_checks"]["pte_current_nonzero_sign_stable"] is False


def test_six_percent_current_change_fails_five_percent_tail_limit():
    result = certificate.compare_pair(
        _report(current_A=1.0),
        _report(current_A=1.06),
        _fields(),
        _fields(),
    )
    assert result["pass"] is False
    assert result["threshold_checks"]["pte_current_relative_change"] is False


def test_weighting_change_fails_common_electrical_operator_invariant():
    fine = _fields()
    fine["ta_electrical_weighting_V"] = 1.001 * fine[
        "ta_electrical_weighting_V"
    ]
    result = certificate.compare_pair(_report(), _report(), _fields(), fine)
    assert result["pass"] is False
    assert (
        result["invariant_checks"][
            "electrical_weighting_identical_to_roundoff"
        ]
        is False
    )
