from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_frozen_q_thermal_sensitivity_certificate as certificate,
)


def _report(maximum: float = 2.0, mean: float = 1.0) -> dict:
    return {
        "thermal_solution": {
            "ta_base_max_temperature_rise_K": maximum,
            "ta_base_mean_temperature_rise_K": mean,
            "solver": {"boundary_power_W": {"z_min": 1.0}},
        }
    }


def _snapshot(scale: float = 1.0) -> dict[str, np.ndarray]:
    coordinate = np.linspace(-1.0, 1.0, 160)
    return {
        "ta_temperature_rise_K": scale * np.ones((160, 160)),
        "ta_gradient_x_K_m": scale * np.ones((160, 160)),
        "ta_gradient_y_K_m": 2.0 * scale * np.ones((160, 160)),
        "ta_x_centers_m": coordinate,
        "ta_y_centers_m": coordinate.copy(),
        "source_power_xy_W": np.ones((266, 266)),
    }


def test_identical_scenario_has_zero_sensitivity_and_passes_integrity():
    result = certificate.compare_to_baseline(
        "top_h0",
        "Ea",
        {"baseline": _report(), "top_h0": _report()},
        {"baseline": _snapshot(), "top_h0": _snapshot()},
    )
    assert result["integrity_pass"] is True
    assert all(value == 0.0 for value in result["metrics"].values())
    assert result["physical_acceptance_threshold_applied"] is False


def test_source_change_fails_integrity_without_inventing_physical_threshold():
    scenario = _snapshot()
    scenario["source_power_xy_W"] *= 0.99
    result = certificate.compare_to_baseline(
        "au_ta_1e6",
        "Eb",
        {"baseline": _report(), "au_ta_1e6": _report()},
        {"baseline": _snapshot(), "au_ta_1e6": scenario},
    )
    assert result["integrity_pass"] is False
    assert "source_distribution_unchanged" in result["failed_integrity_checks"]
    assert result["physical_acceptance_threshold_applied"] is False


@pytest.mark.parametrize(
    ("scenario_id", "key", "expected"),
    [
        ("baseline", "top_air_convection_W_m2K", 10.0),
        ("far_xy_adiabatic", "far_xy_boundary", "adiabatic"),
        ("au_ta_perfect", "g_au_ta_W_m2K", None),
        ("ta_kz_0p5", "ta_kappa_xyz_W_mK", [3.8, 14.4, 0.5]),
    ],
)
def test_expected_parameters_apply_one_named_scenario(scenario_id, key, expected):
    result = certificate.expected_model_parameters(scenario_id)
    assert result[key] == expected


@pytest.mark.parametrize(
    ("characterized", "baseline", "selected"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_selection_characterizes_but_never_converges_physical_uncertainty(
    characterized, baseline, selected
):
    result = certificate.selection(characterized, baseline)
    assert result["thermal_scenario_sensitivity_characterized"] is selected
    assert result["device_specific_physical_bounds_supplied"] is False
    assert result["thermal_boundary_condition_uncertainty_converged"] is False
    assert result["thermal_interface_parameter_uncertainty_converged"] is False
    assert result["thermal_domain_and_boundary_converged"] is False
    assert result["optimizer_start_allowed"] is False
