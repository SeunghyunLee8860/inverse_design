from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_frozen_q_thermal_xy_certificate as certificate,
)


def _report(maximum: float = 2.0, mean: float = 1.0) -> dict:
    return {
        "thermal_solution": {
            "ta_base_max_temperature_rise_K": maximum,
            "ta_base_mean_temperature_rise_K": mean,
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


def test_identical_xy_pair_passes_all_gates():
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
    ("coarse_scale", "failed_gate"),
    [
        (0.95, "ta_temperature_map_nrmse_within_2pct"),
        (0.5, "ta_combined_gradient_nrmse_within_5pct"),
    ],
)
def test_xy_field_error_fails_closed(coarse_scale, failed_gate):
    result = certificate.compare_pair(
        2,
        4,
        "Eb",
        {2: _report(), 4: _report()},
        {2: _snapshot(coarse_scale), 4: _snapshot()},
    )
    assert result["pass"] is False
    assert failed_gate in result["failed_gates"]


def test_xy_source_change_fails_closed():
    coarse = _snapshot()
    fine = _snapshot()
    coarse["source_power_xy_W"] *= 0.99
    result = certificate.compare_pair(
        1,
        2,
        "Ea",
        {1: _report(), 2: _report()},
        {1: coarse, 2: fine},
    )
    assert result["pass"] is False
    assert "source_xy_distribution_exact_to_roundoff" in result["failed_gates"]


@pytest.mark.parametrize(
    ("pairs", "baseline", "selected"),
    [(True, True, 2), (True, False, None), (False, True, None)],
)
def test_selection_requires_pairs_and_prior_baseline(pairs, baseline, selected):
    result = certificate.selection(pairs, baseline)
    assert result["selected_diagnostic_frozen_q_thermal_xy_factor"] == selected
    assert result["selected_diagnostic_frozen_q_thermal_z_factor"] == selected
    assert result["thermal_domain_and_boundary_converged"] is False
    assert result["optical_mesh_converged"] is False
    assert result["electrical_mesh_converged"] is False
    assert result["production_multiphysics_mesh_selected"] is False
    assert result["optimizer_start_allowed"] is False


def test_xy_certificate_reuses_strict_predeclared_limits():
    assert certificate.SUCCESSIVE_PAIRS == ((1, 2), (2, 4))
    assert certificate.TEMPERATURE_MAP_NRMSE_LIMIT == 0.02
    assert certificate.TA_MAX_RELATIVE_LIMIT == 0.02
    assert certificate.TA_MEAN_RELATIVE_LIMIT == 0.02
    assert certificate.GRADIENT_COMBINED_NRMSE_LIMIT == 0.05
    assert certificate.BASE_COORDINATE_ATOL_M == 2e-18
