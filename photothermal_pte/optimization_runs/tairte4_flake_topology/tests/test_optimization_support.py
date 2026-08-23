import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import (
    MAPPING,
    exact_binary_audit,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.repair_exact_binary_candidates import (
    active_set_repair,
    gradient_aware_exact_repair,
)


def test_mapping_preserves_uniform_density_and_transpose():
    uniform = np.full(MAPPING.shape, 0.5)
    assert np.array_equal(MAPPING.filtered(uniform), uniform)
    rng = np.random.default_rng(12)
    direction = rng.normal(size=MAPPING.shape)
    cotangent = rng.normal(size=MAPPING.shape)
    left = np.sum(MAPPING.jvp(uniform, direction, 2.0) * cotangent)
    right = np.sum(direction * MAPPING.vjp(uniform, cotangent, 2.0))
    assert abs(left - right) / max(abs(left), abs(right)) < 1e-12


def test_exact_audit_uses_geometry_specific_outside_phase():
    solid = CONTRACT.apply_fixed_contact_density(np.ones(MAPPING.shape))
    audit, _ = exact_binary_audit(solid)
    assert audit["passed"]
    expected = {
        "contact_anchored": "fixed_solid_at_top_bottom_and_void_at_left_right",
        "left_right_contact_anchored": "fixed_solid_at_left_right_and_void_at_top_bottom",
        "diagonal_45_contact_anchored": (
            "fixed_solid_at_rotated_low_high_edges_and_void_at_rotated_sides"
        ),
    }.get(CONTRACT.geometry_mode, "fixed_solid_TaIrTe4_frame")
    assert audit["outside_design_phase"] == expected
    assert audit["opening_radius_nm"] == 250.0
    assert audit["opening_radius_pixels"] == 2
    assert audit["realized_discrete_opening_max_center_offset_nm"] == 200.0
    assert audit["realized_discrete_opening_pixel_support_diameter_nm"] == 500.0
    assert audit["counted_entity"].startswith("design nodes")


def test_diagonal_exact_audit_rejects_removed_locked_contact() -> None:
    if CONTRACT.geometry_mode != "diagonal_45_contact_anchored":
        return
    solid = CONTRACT.apply_fixed_contact_density(np.ones(MAPPING.shape))
    solid[CONTRACT.fixed_design_solid_mask] = 0.0
    audit, arrays = exact_binary_audit(solid)
    assert not audit["passed"]
    assert audit["fixed_contact_violation_count"] == np.count_nonzero(
        CONTRACT.fixed_design_solid_mask
    )
    assert np.array_equal(
        arrays["fixed_contact_violation"], CONTRACT.fixed_design_solid_mask
    )


def test_diagonal_exact_audit_rejects_solid_outside_flake() -> None:
    if CONTRACT.geometry_mode != "diagonal_45_contact_anchored":
        return
    assert not np.any(CONTRACT.fixed_design_void_mask)


def test_exact_audit_explicit_geometry_overrides_process_default():
    solid = np.ones((21, 21))
    audit, _ = exact_binary_audit(
        solid,
        geometry_mode="contact_anchored",
        contact_axis="y",
    )
    assert audit["geometry_mode"] == "contact_anchored"
    assert audit["contact_axis"] == "y"
    assert audit["outside_design_phase"] == (
        "fixed_solid_at_top_bottom_and_void_at_left_right"
    )


def test_gradient_aware_repair_resolves_alternating_phase_cycle():
    rows = (
        "010010111111111",
        "101001011011011",
        "110011101000111",
        "001111111111110",
        "111111111001001",
        "110111001111111",
        "010101011011100",
        "111111111001111",
        "010101100100111",
        "101110100111001",
        "111101101110111",
        "111101101111110",
        "010111010000101",
        "011111111110011",
        "110111111100001",
    )
    source = np.asarray([[value == "1" for value in row] for row in rows])
    _, _, stop_reason = active_set_repair(
        source,
        "solid_first",
        20,
        geometry_mode="contact_anchored",
        contact_axis="y",
    )
    assert stop_reason == "cycle_detected"
    repaired = gradient_aware_exact_repair(
        source,
        geometry_mode="contact_anchored",
        contact_axis="y",
        maximum_candidates=2,
    )
    assert repaired["passed"]
    assert repaired["candidates"]
    assert all(
        row["exact_audit"]["total_bad_cell_count"] == 0
        for row in repaired["candidates"]
    )
