from __future__ import annotations

from dataclasses import replace

import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_mesh_contract import (
    BASELINE,
    GEOMETRY_CONTROLS,
    MESH_REFINEMENT_CANDIDATES,
    POLARIZATIONS,
    SELECTED_DEVELOPMENT_MESH,
    SELECTED_DEVELOPMENT_MESH_STATUS,
    convergence_contract_audit,
)


def test_mesh_contract_is_sequential_and_covers_all_required_axes() -> None:
    audit = convergence_contract_audit()
    assert audit["status"].endswith("NOT_RUN")
    assert audit["required_polarizations"] == list(POLARIZATIONS)
    assert audit["required_exact_geometry_controls"] == list(GEOMETRY_CONTROLS)
    assert audit["axis_order"] == [
        "source_profile_and_incident_power",
        "time_and_auto_shutoff",
        "optical_z_full_domain_stack_bulk_air_and_PML",
        "metal_interface_mesh_refinement_CV0_CV1_staircase",
        "optical_xy_flake_and_Au_edges",
        "PML_layers",
        "lateral_domain_clearance",
        "z_domain_clearance",
    ]
    candidates = audit["candidate_axes"]
    assert candidates["optical_full_domain_z_m"][-1] == {
        "stack_dz_m": 0.625e-9,
        "bulk_air_pml_dz_m": 6.25e-9,
    }
    assert candidates["optical_full_domain_z_m"][-2] == {
        "stack_dz_m": 1.25e-9,
        "bulk_air_pml_dz_m": 12.5e-9,
    }
    assert candidates["optical_xy_flake_dxy_m"][-1] == 25.0e-9
    assert candidates["metal_interface_mesh_refinement"] == list(
        MESH_REFINEMENT_CANDIDATES
    )
    assert [item["pml_layers"] for item in candidates["pml_layers"]] == [
        8,
        12,
        16,
    ]
    assert [
        round(item["lateral_span_m"] * 1e6, 12)
        for item in candidates["pml_layers"]
    ] == [20.0, 21.6, 23.2]
    assert audit["promotion"]["is_mesh_certificate"] is False


def test_selected_development_mesh_is_explicitly_nonpromotable() -> None:
    selected = SELECTED_DEVELOPMENT_MESH
    assert selected.label == "fine_z2p5_bulk50_xy100_cv0_pml8_span20_z6_t1ps"
    assert selected.stack_dz_m == 2.5e-9
    assert selected.bulk_dz_m == 50.0e-9
    assert selected.conformal_mesh == "conformal variant 0"
    audit = convergence_contract_audit()["selected_development_mesh"]
    assert audit["status"] == SELECTED_DEVELOPMENT_MESH_STATUS
    assert audit["formal_convergence_certificate"] is False
    assert audit["same_mesh_adfd_scope"] == (
        "one common Ea/Eb beta-4 latent direction passed on RTX"
    )
    assert audit["requires_same_mesh_adfd_before_optimizer"] is False
    assert audit["requires_finer_final_binary_reevaluation"] is True


def test_cli_unit_round_trip_does_not_reject_exact_baseline_bounds() -> None:
    round_trip = replace(
        BASELINE,
        lateral_span_m=(BASELINE.lateral_span_m * 1e6) * 1e-6,
        z_min_m=(BASELINE.z_min_m * 1e6) * 1e-6,
        z_max_m=(BASELINE.z_max_m * 1e6) * 1e-6,
    )
    assert round_trip.validate() is round_trip


@pytest.mark.parametrize(
    "bad",
    [
        replace(BASELINE, label="bad label"),
        replace(BASELINE, stack_dz_m=0.0),
        replace(BASELINE, bulk_dz_m=10e-9),
        replace(BASELINE, outer_dxy_m=50e-9),
        replace(BASELINE, pml_layers=2),
        replace(BASELINE, lateral_span_m=19e-6),
        replace(BASELINE, z_min_m=-2e-6),
        replace(BASELINE, auto_shutoff_min=1e-3),
        replace(BASELINE, conformal_mesh="unvalidated metal average"),
    ],
)
def test_invalid_mesh_contracts_fail_closed(bad) -> None:
    with pytest.raises(ValueError):
        bad.validate()
