from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_frozen_q_thermal_z_certificate as certificate,
)


def _report(maximum: float = 2.0, mean: float = 1.0) -> dict:
    return {
        "thermal_solution": {
            "ta_max_temperature_rise_K": maximum,
            "ta_mean_temperature_rise_K": mean,
        }
    }


def _snapshot(scale: float = 1.0) -> dict[str, np.ndarray]:
    coordinate = np.linspace(-1.0, 1.0, 160)
    temperature = scale * np.ones((160, 160))
    gradient_x = scale * np.ones((160, 160))
    gradient_y = 2.0 * scale * np.ones((160, 160))
    return {
        "ta_temperature_rise_K": temperature,
        "ta_gradient_x_K_m": gradient_x,
        "ta_gradient_y_K_m": gradient_y,
        "ta_x_centers_m": coordinate,
        "ta_y_centers_m": coordinate.copy(),
        "thermal_x_centers_m": coordinate,
        "thermal_y_centers_m": coordinate.copy(),
        "source_power_xy_W": np.ones((266, 266)),
    }


def test_identical_tail_pair_passes_all_predeclared_gates():
    result = certificate.compare_pair(
        1,
        2,
        "Ea",
        {1: _report(), 2: _report()},
        {1: _snapshot(), 2: _snapshot()},
    )
    assert result["pass"] is True
    assert all(value == 0.0 for value in result["metrics"].values())


@pytest.mark.parametrize(
    ("coarse_scale", "expected_failed_gate"),
    [
        (0.95, "ta_temperature_map_nrmse_within_2pct"),
        (0.50, "ta_combined_gradient_nrmse_within_5pct"),
    ],
)
def test_field_error_fails_closed(coarse_scale, expected_failed_gate):
    result = certificate.compare_pair(
        1,
        2,
        "Eb",
        {1: _report(), 2: _report()},
        {1: _snapshot(coarse_scale), 2: _snapshot()},
    )
    assert result["pass"] is False
    assert expected_failed_gate in result["failed_gates"]


@pytest.mark.parametrize(
    ("coarse_report", "expected_failed_gate"),
    [
        (_report(maximum=1.0), "ta_max_temperature_relative_within_2pct"),
        (_report(mean=0.5), "ta_mean_temperature_relative_within_2pct"),
    ],
)
def test_scalar_temperature_error_fails_closed(
    coarse_report, expected_failed_gate
):
    result = certificate.compare_pair(
        2,
        4,
        "Ea",
        {2: coarse_report, 4: _report()},
        {2: _snapshot(), 4: _snapshot()},
    )
    assert result["pass"] is False
    assert expected_failed_gate in result["failed_gates"]


def test_source_distribution_change_fails_closed():
    coarse = _snapshot()
    fine = _snapshot()
    coarse["source_power_xy_W"] = 0.99 * coarse["source_power_xy_W"]
    result = certificate.compare_pair(
        1,
        2,
        "Ea",
        {1: _report(), 2: _report()},
        {1: coarse, 2: fine},
    )
    assert result["pass"] is False
    assert "source_xy_power_distribution_exact_to_roundoff" in result["failed_gates"]


@pytest.mark.parametrize("passed", [False, True])
def test_selection_never_promotes_production_or_optimizer(passed):
    result = certificate.selection(passed)
    assert result["selected_diagnostic_frozen_q_thermal_z_factor"] == (
        2 if passed else None
    )
    assert result["thermal_xy_converged"] is False
    assert result["optical_mesh_converged"] is False
    assert result["electrical_mesh_converged"] is False
    assert result["production_multiphysics_mesh_selected"] is False
    assert result["optimizer_start_allowed"] is False


def test_thresholds_are_predeclared_and_not_relaxed():
    assert certificate.SUCCESSIVE_PAIRS == ((1, 2), (2, 4))
    assert certificate.TEMPERATURE_MAP_NRMSE_LIMIT == 0.02
    assert certificate.TA_MAX_RELATIVE_LIMIT == 0.02
    assert certificate.TA_MEAN_RELATIVE_LIMIT == 0.02
    assert certificate.GRADIENT_COMBINED_NRMSE_LIMIT == 0.05
