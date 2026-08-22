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
        assert CONTRACT.design_node_shape == (201, 241)
        assert CONTRACT.flake_node_shape == (241, 241)
        assert CONTRACT.contact_axis == "diagonal_45"
        low, high = CONTRACT.fixed_design_contact_masks
        assert np.any(low) and np.any(high)
        assert not np.any(low & high)
        assert low[0, 0] and high[-1, -1]
        assert not low[-1, -1] and not high[0, 0]
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
    assert np.all(locked[CONTRACT.fixed_design_solid_mask] == 1.0)
    assert np.all(locked[~CONTRACT.fixed_design_solid_mask] == 0.0)
    gradient = CONTRACT.zero_fixed_contact_gradient(np.ones_like(value))
    assert np.all(gradient[CONTRACT.fixed_design_solid_mask] == 0.0)
    assert np.all(gradient[~CONTRACT.fixed_design_solid_mask] == 1.0)
