from __future__ import annotations

import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_forensic_audit import (
    audit,
)


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return audit()


def test_historical_endpoint_does_not_meet_the_requested_sign_gate(report) -> None:
    endpoint = report["endpoint"]
    rows = endpoint["nominal_by_z_factor"]
    assert [row["factor"] for row in rows] == [1, 2, 4]
    assert not any(row["opposite_sign"] for row in rows)
    assert not endpoint["all_nominal_pairs_opposite_sign"]

    binary = endpoint["exact_binary"]
    assert binary["candidate_count"] == 6
    assert binary["exact_500nm_pass_count"] == 6
    assert binary["opposite_sign_count"] == 0
    assert binary["campaign_opposite_sign_gate"] is False


def test_historical_mesh_campaign_is_not_a_mesh_certificate(report) -> None:
    mesh = report["mesh"]
    assert mesh["factors"] == [1, 2, 4]
    assert mesh["central_dx_m"] == pytest.approx([1.0e-7] * 3)
    assert mesh["central_dy_m"] == pytest.approx([1.0e-7] * 3)
    assert mesh["xy_refined"] is False
    assert mesh["passing_comparison_count"] == 0
    assert mesh["passing_final_pair_count"] == 0
    assert mesh["selected_optical_z_contract"] is None
    assert mesh["production_mesh_certificate_exists"] is False
    assert mesh["production_combined_adfd_certificate_exists"] is False
    worst = mesh["factor_2_to_4_worst_relative_changes"]
    assert worst["remapped_Q_volume_L2_NRMSE"] > 0.34
    assert worst["Tmax_relative_change"] > 0.30
    assert worst["current_relative_change"] > 0.37


def test_device_material_and_interface_assumptions_remain_open(report) -> None:
    device = report["device"]
    assert device["confirmed_count"] == 0
    assert len(device["unresolved_confirmations"]) == 10

    materials = report["materials"]
    assert materials["TaIrTe4_c_equals_b"] is True
    assert materials["single_frequency_readback_only"] is True

    implementation = report["implementation"]
    assert implementation["pml_profile_explicit"] is False
    assert implementation["uses_uniform_material_object_for_tairte4"] is True
    assert implementation["uses_uniform_material_object_for_au"] is True
    assert implementation["requests_subpixel_smoothing"] is False
    assert implementation["raw_artifacts_absolute"] is True


def test_audit_is_fail_closed(report) -> None:
    assert report["status"] == "BLOCKED_FDTDX_FORENSIC_AUDIT"
    blockers = set(report["blockers"])
    assert {
        "NO_OPPOSITE_SIGN_NOMINAL_ENDPOINT",
        "EXACT_BINARY_CANDIDATES_FAIL_SIGN",
        "OPTICAL_Z_CONVERGENCE_FAILED",
        "OPTICAL_XY_NOT_REFINED",
        "THERMAL_MESH_NOT_CERTIFIED",
        "ELECTRICAL_MESH_NOT_CERTIFIED",
        "PML_PROFILE_NOT_EXPLICIT_OR_CONVERGED",
    } <= blockers
    assert report["decision"]["resume_historical_optimizer"] is False
