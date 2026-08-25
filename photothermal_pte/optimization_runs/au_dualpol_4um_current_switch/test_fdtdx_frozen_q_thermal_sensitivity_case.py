from __future__ import annotations

import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_frozen_q_thermal_sensitivity_case as sensitivity,
)


@pytest.mark.parametrize("scenario_id", tuple(sensitivity.SCENARIO_PARAMETERS))
def test_every_predeclared_sensitivity_scenario_is_selectable(scenario_id):
    result = sensitivity.scenario_parameters(scenario_id)
    assert result == sensitivity.SCENARIO_PARAMETERS[scenario_id]
    assert result is not sensitivity.SCENARIO_PARAMETERS[scenario_id]


def test_mutating_returned_kappa_list_does_not_change_contract():
    result = sensitivity.scenario_parameters("ta_kz_0p5")
    result["ta_kappa_xyz_W_mK"][2] = 99.0
    assert sensitivity.SCENARIO_PARAMETERS["ta_kz_0p5"][
        "ta_kappa_xyz_W_mK"
    ] == [3.8, 14.4, 0.5]


def test_undeclared_sensitivity_scenario_fails_closed():
    with pytest.raises(ValueError, match="thermal sensitivity scenario"):
        sensitivity.scenario_parameters("invented")


def test_sensitivity_runner_locks_selected_mesh_domain_and_blocks_promotion():
    assert sensitivity.THERMAL_XY_REFINEMENT_FACTOR == 2
    assert sensitivity.THERMAL_Z_REFINEMENT_FACTOR == 2
    assert sensitivity.SELECTED_DOMAIN == {
        "lateral_half_span_um": 48,
        "substrate_depth_um": 30,
        "top_air_height_um": 3.0,
    }
    assert "DIAGNOSTIC" in sensitivity.STATUS_READY
