import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT


def test_contract_is_enclosed_and_resolves_feature() -> None:
    CONTRACT.validate()
    if CONTRACT.geometry_mode == "contact_anchored":
        assert np.isclose(CONTRACT.fixed_frame_width_m, 0.0)
        assert CONTRACT.design_node_shape == (241, 201)
        assert CONTRACT.contact_axis == "y"
    elif CONTRACT.geometry_mode == "left_right_contact_anchored":
        assert CONTRACT.design_node_shape == (201, 241)
        assert CONTRACT.contact_axis == "x"
    elif CONTRACT.geometry_mode == "diagonal_45_contact_anchored":
        assert CONTRACT.design_node_shape == (241, 241)
        assert CONTRACT.flake_node_shape == (241, 241)
        assert CONTRACT.crystal_bounding_node_shape == (341, 341)
        assert CONTRACT.contact_axis == "diagonal_45"
        low, high = CONTRACT.fixed_design_contact_masks
        outside = CONTRACT.fixed_design_void_mask
        assert not np.any(low) and not np.any(high)
        assert not np.any(outside)
    else:
        assert np.isclose(CONTRACT.fixed_frame_width_m, 4.0e-6)
        assert CONTRACT.design_node_shape == (161, 161)
    assert CONTRACT.feature_cells == 5.0
    assert CONTRACT.square_gaussian_fraction(CONTRACT.flake_span_m) > 0.99
    assert CONTRACT.boundary_intensity_fraction(0.5 * CONTRACT.optical_lateral_span_m) < 2.0e-4


def test_diagonal_contact_density_lock_is_exact() -> None:
    if CONTRACT.geometry_mode != "diagonal_45_contact_anchored":
        return
    value = np.zeros(CONTRACT.design_node_shape)
    locked = CONTRACT.apply_fixed_contact_density(value)
    assert np.all(locked == 0.0)
    gradient = CONTRACT.zero_fixed_contact_gradient(np.ones_like(value))
    assert np.all(gradient == 1.0)


def test_diagonal_density_maps_to_global_crystal_grid_without_expanding_flake() -> None:
    if CONTRACT.geometry_mode != "diagonal_45_contact_anchored":
        return
    from photothermal_pte.optimization_runs.tairte4_flake_topology.rotated_device import (
        crystal_nodes_m,
        device_to_crystal_field,
    )

    mapped = device_to_crystal_field(np.ones(CONTRACT.design_node_shape))
    x, y = crystal_nodes_m()
    xx, yy = np.meshgrid(x, y, indexing="ij")
    support = CONTRACT.flake_support_mask(xx, yy)
    assert mapped.shape == CONTRACT.crystal_bounding_node_shape
    assert np.all(mapped[~support] == 0.0)
    assert np.allclose(mapped[support], 1.0)
    assert np.isclose(np.max(np.abs(x)), 12.0e-6 * np.sqrt(2.0), atol=0.1e-6)


def test_diagonal_flake_keeps_crystal_axes_and_original_side_length() -> None:
    if CONTRACT.geometry_mode != "diagonal_45_contact_anchored":
        return
    assert CONTRACT.axis_contract == "lumerical_x_b_y_a"
    assert np.isclose(CONTRACT.flake_span_m, 24.0e-6)
    assert np.isclose(CONTRACT.flake_span_m**2, 576.0e-12)
    vertices_uv = np.asarray(
        [(-12.0, -12.0), (-12.0, 12.0), (12.0, 12.0), (12.0, -12.0)]
    ) * 1.0e-6
    u, v = vertices_uv.T
    x = (u - v) / np.sqrt(2.0)
    y = (u + v) / np.sqrt(2.0)
    assert np.isclose(np.max(np.abs(x)), 12.0e-6 * np.sqrt(2.0))
    assert np.isclose(np.max(np.abs(y)), 12.0e-6 * np.sqrt(2.0))
