from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_frozen_q_thermal_domain_certificate as certificate,
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


def test_identical_domain_pair_passes_all_gates():
    result = certificate.compare_pair(
        "baseline",
        "lateral_48",
        "Ea",
        {"baseline": _report(), "lateral_48": _report()},
        {"baseline": _snapshot(), "lateral_48": _snapshot()},
    )
    assert result["pass"] is True
    assert all(value == 0.0 for value in result["metrics"].values())


@pytest.mark.parametrize(
    ("coarse_scale", "failed_gate"),
    [
        (0.98, "ta_temperature_map_nrmse_within_1pct"),
        (0.95, "ta_combined_gradient_nrmse_within_2pct"),
    ],
)
def test_domain_field_error_fails_closed(coarse_scale, failed_gate):
    result = certificate.compare_pair(
        "combined_mid",
        "combined_large",
        "Eb",
        {"combined_mid": _report(), "combined_large": _report()},
        {"combined_mid": _snapshot(coarse_scale), "combined_large": _snapshot()},
    )
    assert result["pass"] is False
    assert failed_gate in result["failed_gates"]


def test_domain_source_change_fails_closed():
    coarse = _snapshot()
    fine = _snapshot()
    coarse["source_power_xy_W"] *= 0.99
    result = certificate.compare_pair(
        "baseline",
        "substrate_30",
        "Ea",
        {"baseline": _report(), "substrate_30": _report()},
        {"baseline": coarse, "substrate_30": fine},
    )
    assert result["pass"] is False
    assert "source_xy_distribution_exact_to_roundoff" in result["failed_gates"]


@pytest.mark.parametrize(
    ("pairs", "baseline", "selected"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_selection_requires_all_pairs_and_exact_prior_baseline(
    pairs, baseline, selected
):
    result = certificate.selection(pairs, baseline)
    assert result["thermal_domain_size_converged"] is selected
    assert result["thermal_boundary_condition_uncertainty_converged"] is False
    assert result["thermal_domain_and_boundary_converged"] is False
    assert result["production_multiphysics_mesh_selected"] is False
    assert result["optimizer_start_allowed"] is False


def test_certificate_predeclares_all_axis_and_combined_ladders():
    assert set(certificate.DOMAIN_LADDERS) == {
        "lateral",
        "substrate",
        "top_air",
        "combined",
    }
    assert certificate.TEMPERATURE_MAP_NRMSE_LIMIT == 0.01
    assert certificate.GRADIENT_COMBINED_NRMSE_LIMIT == 0.02
    assert len(certificate.CASE_CONFIGS) == 9
