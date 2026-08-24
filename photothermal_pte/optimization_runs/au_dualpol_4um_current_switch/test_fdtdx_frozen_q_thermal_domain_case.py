from __future__ import annotations

import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_frozen_q_thermal_domain_case as domain_case,
)


@pytest.mark.parametrize(
    ("axis", "level", "expected"),
    [
        ("lateral", 64.0, (64, 20, 2.0)),
        ("substrate", 40.0, (32, 40, 2.0)),
        ("top_air", 4.0, (32, 20, 4.0)),
        ("combined", 1.0, (48, 30, 3.0)),
        ("combined", 2.0, (64, 40, 4.0)),
    ],
)
def test_domain_configuration_changes_exactly_one_boundary(axis, level, expected):
    result = domain_case.domain_configuration(axis, level)
    assert (
        result["lateral_half_span_um"],
        result["substrate_depth_um"],
        result["top_air_height_um"],
    ) == expected


@pytest.mark.parametrize(
    ("axis", "level"),
    [("x", 32.0), ("lateral", 40.0), ("substrate", 50.0), ("top_air", 2.5)],
)
def test_undeclared_domain_case_fails_closed(axis, level):
    with pytest.raises(ValueError, match="thermal domain"):
        domain_case.domain_configuration(axis, level)


def test_domain_runner_locks_selected_diagnostic_mesh_and_blocks_promotion():
    assert domain_case.THERMAL_XY_REFINEMENT_FACTOR == 2
    assert domain_case.THERMAL_Z_REFINEMENT_FACTOR == 2
    assert domain_case.DOMAIN_LEVELS == {
        "lateral": (32.0, 48.0, 64.0),
        "substrate": (20.0, 30.0, 40.0),
        "top_air": (2.0, 3.0, 4.0),
        "combined": (1.0, 2.0),
    }
    assert "DIAGNOSTIC" in domain_case.STATUS_READY
