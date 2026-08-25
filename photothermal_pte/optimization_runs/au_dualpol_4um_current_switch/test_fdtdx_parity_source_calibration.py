from __future__ import annotations

import math

import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_source_calibration import (
    ETA0_OHM,
    SCHEMA_CASE,
    TARGET_POWER_W,
    _report_hash,
    aggregate_cases,
    case_gate,
    normalized_flux_to_si_W,
    polarization_components,
)


def _case(polarization: str, power_W: float) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": SCHEMA_CASE,
        "status": "PASS_SOURCE_CASE",
        "polarization": polarization,
        "metrics": {"incident_power_late_W": power_W},
        "git_commit": "a" * 40,
        "git_status_porcelain": "",
        "script_sha256": "b" * 64,
        "cublas_runtime_version": 130601,
    }
    payload["report_sha256"] = _report_hash(payload)
    return payload


def test_si_flux_conversion_uses_pinned_fdtdx_field_convention() -> None:
    assert normalized_flux_to_si_W(2.0) == pytest.approx(2.0 * ETA0_OHM)


def test_polarization_component_mapping_matches_minus_z_propagation() -> None:
    assert polarization_components("Ea") == (1, 0, 0)
    assert polarization_components("Eb") == (0, 1, 1)
    with pytest.raises(ValueError, match="unknown polarization"):
        polarization_components("Ex")


def test_case_gate_requires_all_independent_controls() -> None:
    metrics: dict[str, float | bool] = {
        "finite": True,
        "incident_power_late_W": 1.0,
        "td_phasor_mismatch_relative": 0.001,
        "previous_late_mismatch_relative": 0.001,
        "vacuum_impedance_error_relative": 0.001,
        "source_injection_cross_polarization_ratio": 1.0e-8,
        "source_injection_longitudinal_ratio": 1.0e-8,
    }
    assert case_gate(metrics) == "PASS_SOURCE_CASE"
    for key in (
        "td_phasor_mismatch_relative",
        "previous_late_mismatch_relative",
        "vacuum_impedance_error_relative",
        "source_injection_cross_polarization_ratio",
        "source_injection_longitudinal_ratio",
    ):
        blocked = dict(metrics)
        blocked[key] = math.inf
        assert case_gate(blocked) == "BLOCKED"


def test_aggregate_keeps_separate_Ea_Eb_normalization() -> None:
    ea = _case("Ea", 1.0e-6)
    eb = _case("Eb", 1.001e-6)
    report = aggregate_cases(ea, eb)
    assert report["status"] == "PASS_SOURCE_CALIBRATION"
    assert report["power_or_Q_scale_to_target"] == pytest.approx(
        {"Ea": TARGET_POWER_W / 1.0e-6, "Eb": TARGET_POWER_W / 1.001e-6}
    )
    assert report["Ea_Eb_incident_power_mismatch_relative"] < 0.005
    assert "independently" in report["normalization_contract"]


def test_aggregate_rejects_hash_tampering_and_power_mismatch() -> None:
    ea = _case("Ea", 1.0e-6)
    eb = _case("Eb", 1.0e-6)
    eb["metrics"]["incident_power_late_W"] = 2.0e-6
    with pytest.raises(RuntimeError, match="hash mismatch"):
        aggregate_cases(ea, eb)

    eb = _case("Eb", 2.0e-6)
    assert aggregate_cases(ea, eb)["status"] == "BLOCKED"
