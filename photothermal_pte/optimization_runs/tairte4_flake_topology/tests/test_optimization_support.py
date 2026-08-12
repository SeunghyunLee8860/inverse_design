import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import (
    MAPPING,
    exact_binary_audit,
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
    solid = np.ones(MAPPING.shape)
    audit, _ = exact_binary_audit(solid)
    assert audit["passed"]
    expected = {
        "contact_anchored": "fixed_solid_at_top_bottom_and_void_at_left_right",
        "left_right_contact_anchored": "fixed_solid_at_left_right_and_void_at_top_bottom",
    }.get(CONTRACT.geometry_mode, "fixed_solid_TaIrTe4_frame")
    assert audit["outside_design_phase"] == expected
    assert audit["opening_radius_nm"] == 250.0
    assert audit["opening_radius_pixels"] == 2
    assert audit["realized_discrete_opening_max_center_offset_nm"] == 200.0
    assert audit["realized_discrete_opening_pixel_support_diameter_nm"] == 500.0
    assert audit["counted_entity"].startswith("design nodes")
