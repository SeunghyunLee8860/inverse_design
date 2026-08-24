from __future__ import annotations

import copy

import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_frozen_q_thermal_z_case as thermal_case,
)


def _blocked_certificate() -> dict:
    return {
        "version": thermal_case.Z32_CERTIFICATE_VERSION,
        "status": thermal_case.Z32_STATUS_BLOCKED,
        "ready": False,
        "global_checks": {"artifacts": True, "comparison": True},
        "failed_global_checks": [],
        "promotion": {
            "full_domain_z_converged": False,
            "selected_mesh_level": None,
            "z_only_ladder_terminated": True,
            "z64_run_allowed": False,
            "optimizer_start_allowed": False,
        },
        "optimizer_start_allowed": False,
    }


def test_blocked_optical_certificate_is_required_for_diagnostic():
    checks = thermal_case.certificate_control_checks(_blocked_certificate())
    assert all(checks.values())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "VALIDATED"),
        ("ready", True),
        ("optimizer_start_allowed", True),
    ],
)
def test_certificate_control_rejects_false_promotion(field, value):
    payload = _blocked_certificate()
    payload[field] = value
    assert not all(thermal_case.certificate_control_checks(payload).values())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("full_domain_z_converged", True),
        ("selected_mesh_level", "z32"),
        ("z_only_ladder_terminated", False),
        ("z64_run_allowed", True),
        ("optimizer_start_allowed", True),
    ],
)
def test_certificate_control_rejects_mesh_or_optimizer_promotion(field, value):
    payload = copy.deepcopy(_blocked_certificate())
    payload["promotion"][field] = value
    assert not all(thermal_case.certificate_control_checks(payload).values())


def test_material_slices_reconstruct_certified_z32_placement():
    slices = thermal_case.material_slices(
        {
            "au_design": [[58, 138], [58, 138], [672, 736]],
            "fixed_tairte4": [[18, 178], [18, 178], [512, 672]],
        }
    )
    assert slices["au"] == (slice(58, 138), slice(58, 138), slice(672, 736))
    assert slices["tairte4"] == (
        slice(18, 178),
        slice(18, 178),
        slice(512, 672),
    )


@pytest.mark.parametrize(
    "placement",
    [
        {},
        {"au_design": [[58, 138], [58, 138], [672, 736]]},
        {
            "au_design": [[58, 58], [58, 138], [672, 736]],
            "fixed_tairte4": [[18, 178], [18, 178], [512, 672]],
        },
        {
            "au_design": [[58.0, 138], [58, 138], [672, 736]],
            "fixed_tairte4": [[18, 178], [18, 178], [512, 672]],
        },
    ],
)
def test_invalid_material_placement_fails_closed(placement):
    with pytest.raises(ValueError, match="invalid"):
        thermal_case.material_slices(placement)


def test_refinement_and_solver_limits_are_predeclared():
    assert thermal_case.ALLOWED_Z_REFINEMENT_FACTORS == (1, 2, 4)
    assert thermal_case.MAPPING_RTOL <= 5e-12
    assert thermal_case.POWER_RTOL <= 5e-12
    assert thermal_case.THERMAL_RESIDUAL_LIMIT <= 2e-8
    assert thermal_case.ENERGY_BALANCE_LIMIT <= 2e-8
