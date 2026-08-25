from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_fixed_materials import (
    TA_A,
    TA_B,
    coefficient_hash,
    fdtdx_api_audit,
    fixed_material_audit,
    load_material_targets,
    lorentz_parameters,
    realized_epsilon,
    recurrence_roots,
)


def test_frozen_targets_match_authoritative_material_json() -> None:
    materials = load_material_targets()["materials"]
    assert TA_A.target_epsilon == complex(
        materials["TaIrTe4"]["a"]["epsilon"]["real"],
        materials["TaIrTe4"]["a"]["epsilon"]["imag"],
    )
    assert TA_B.target_epsilon == complex(
        materials["TaIrTe4"]["b"]["epsilon"]["real"],
        materials["TaIrTe4"]["b"]["epsilon"]["imag"],
    )
    assert materials["TaIrTe4"]["b"] == materials["TaIrTe4"]["c"]


def test_tairte4_float32_responses_meet_error_gate() -> None:
    error_a = abs(realized_epsilon(TA_A) - TA_A.target_epsilon) / abs(TA_A.target_epsilon)
    error_b = abs(realized_epsilon(TA_B) - TA_B.target_epsilon) / abs(TA_B.target_epsilon)
    assert error_a < 4.4e-6
    assert error_b < 1.2e-7
    assert realized_epsilon(TA_A).imag > 0.0
    assert realized_epsilon(TA_B).imag > 0.0


def test_tairte4_recurrences_are_strictly_stable_positive_lorentz_models() -> None:
    for carrier in (TA_A, TA_B):
        assert np.max(np.abs(recurrence_roots(carrier))) < 1.0
        assert 1.0 - np.float32(carrier.c2) - abs(np.float32(carrier.c1)) > 0.0
        assert all(value > 0.0 for value in lorentz_parameters(carrier).values())


def test_pinned_fdtdx_reproduces_fixed_coefficients_exactly() -> None:
    audit = fdtdx_api_audit()
    assert audit["status"] == "PASS"
    assert all(item["exact"] for item in audit["materials"].values())


def test_complete_fixed_material_certificate_and_hash() -> None:
    audit = fixed_material_audit()
    assert audit["status"] == "PASS"
    assert audit["axis_order_solver_xyz"] == ["b", "a", "c_equals_b"]
    assert all(audit["substrates"]["checks"].values())
    assert coefficient_hash() == "7aa3f50f5ca3cf1f0d9222d7d5e16a7e82ca9d0a5c55d75a0617a09191148057"
