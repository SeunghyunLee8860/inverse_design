from __future__ import annotations

import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_increment_state_full_z_certificate as certificate,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_full_z_certificate import (
    LEVELS,
    Z_FACTOR,
)


def test_expected_cases_use_24_period_increment_state_ladder_contract():
    cases = {level: certificate.expected_full_z_case(level) for level in LEVELS}
    baseline = dict(cases[LEVELS[0]].mesh.__dict__)
    baseline.pop("z_factor")
    for level, case in cases.items():
        mesh = dict(case.mesh.__dict__)
        assert mesh.pop("z_factor") == Z_FACTOR[level]
        assert mesh == baseline
        assert case.time.total_periods == 24
        assert case.time.window_periods == 4
        assert case.time.courant_factor == 0.5


def test_hash_parsers_are_exact_and_reject_duplicates():
    hashes = [f"{level}={'a' * 64}" for level in LEVELS]
    assert set(certificate._parse_level_hashes(hashes, "test")) == set(LEVELS)
    with pytest.raises(ValueError):
        certificate._parse_level_hashes(hashes[:-1], "test")
    with pytest.raises(ValueError):
        certificate._parse_level_hashes([*hashes, hashes[0]], "test")

    reports = [
        f"{level}:{polarization}={'b' * 64}"
        for level in LEVELS
        for polarization in ("Ea", "Eb")
    ]
    parsed = certificate._parse_report_hashes(reports)
    assert all(set(parsed[level]) == {"Ea", "Eb"} for level in LEVELS)
    with pytest.raises(ValueError):
        certificate._parse_report_hashes(reports[:-1])


def test_normalization_recomputes_scaled_raw_q():
    source_pair = {
        "common_normalization": {
            "common_power_scale": 10.0,
            "common_field_amplitude_scale": 10.0**0.5,
        },
        "comparison": {"mean_unscaled_incident_power_W": 2.0},
    }
    snapshot = {
        "power_late": {
            "total_W": 3.0,
            "by_material": {
                "au": {"total_W": 1.0},
                "tairte4": {"total_W": 2.0},
            },
        }
    }
    payload = {
        "normalization_policy": {
            "raw_fields_and_Q_are_unscaled": True,
            "per_polarization_power_matching_forbidden": True,
            "common_power_scale": 10.0,
            "common_field_amplitude_scale": 10.0**0.5,
        },
        "evaluation": {
            "flux": {"source_reference_all_air_unscaled_W": 2.0},
            "common_285uW_reporting": {
                "late_total_Q_W": 30.0,
                "late_Au_Q_W": 10.0,
                "late_TaIrTe4_Q_W": 20.0,
            },
        },
    }
    checks = certificate._normalization_checks(payload, source_pair, snapshot)
    assert all(checks.values())
    payload["evaluation"]["common_285uW_reporting"]["late_Au_Q_W"] = 11.0
    changed = certificate._normalization_checks(payload, source_pair, snapshot)
    assert changed["scaled_Q_recomputes_from_raw"] is False


def test_missing_runner_and_normalization_inputs_fail_closed():
    runner = certificate._runner_audit({})
    assert runner["ready"] is False
    assert not any(runner["checks"].values())
    normalization = certificate._normalization_checks({}, {}, {"power_late": {}})
    assert normalization == {"normalization_inputs_available": False}
