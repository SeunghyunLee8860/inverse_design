import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.thermal import (
    G_TAIRTE4_SIO2_W_M2K,
    K_AU_W_MK,
    K_AIR_W_MK,
    K_TAIRTE4_XYZ_W_MK,
    TAIRTE4_SIO2_INTERFACE_CONDUCTANCE_W_M2K,
    TAIRTE4_SIO2_INTERFACE_SCENARIO,
    build_state,
    cell_to_node,
    cell_to_node_transpose,
    nodal_to_cell,
    nodal_to_cell_transpose,
)


def test_default_interface_scenario_is_thermally_grown():
    assert TAIRTE4_SIO2_INTERFACE_SCENARIO == "thermally_grown"
    assert G_TAIRTE4_SIO2_W_M2K == 7.37e6
    assert TAIRTE4_SIO2_INTERFACE_CONDUCTANCE_W_M2K == {
        "thermally_grown": 7.37e6,
        "evaporated": 7.37e4,
    }


def test_nodal_cell_transpose():
    rng = np.random.default_rng(4)
    node = rng.normal(size=CONTRACT.design_node_shape)
    cell_shape = (
        (CONTRACT.crystal_bounding_intervals,) * 2
        if CONTRACT.geometry_mode == "diagonal_45_contact_anchored"
        else CONTRACT.design_intervals
    )
    cell = rng.normal(size=cell_shape)
    assert np.isclose(np.sum(nodal_to_cell(node) * cell), np.sum(node * nodal_to_cell_transpose(cell)), rtol=1e-13)


def test_temperature_cell_node_transpose():
    rng = np.random.default_rng(5)
    node_shape = (
        CONTRACT.crystal_bounding_node_shape
        if CONTRACT.geometry_mode == "diagonal_45_contact_anchored"
        else CONTRACT.flake_node_shape
    )
    cell_shape = tuple(value - 1 for value in node_shape)
    cell = rng.normal(size=cell_shape)
    node = rng.normal(size=node_shape)
    assert np.isclose(np.sum(cell_to_node(cell) * node), np.sum(cell * cell_to_node_transpose(node)), rtol=1e-13)


def test_bottom_contact_endpoints_and_axis_mapping():
    zero = build_state(np.zeros(CONTRACT.design_node_shape))
    one = build_state(np.ones(CONTRACT.design_node_shape))
    assert np.allclose(K_TAIRTE4_XYZ_W_MK, (3.8, 14.4, 1.0))
    zero_design_kappa = zero.kappa_W_mK[zero.masks["design_effective"]]
    assert np.any(np.isclose(zero_design_kappa[:, 0], K_AIR_W_MK))
    if CONTRACT.geometry_mode == "diagonal_45_contact_anchored":
        assert np.any(np.isclose(zero_design_kappa[:, 0], K_TAIRTE4_XYZ_W_MK[0]))
    one_design_kappa = one.kappa_W_mK[one.masks["design_effective"]]
    assert np.any(
        np.all(np.isclose(one_design_kappa, K_TAIRTE4_XYZ_W_MK), axis=1)
    )
    assert np.isclose(
        one.bottom_tairte4_path_resistance_m2K_W
        - 0.5 * one.widths_m[2][one.bottom_face] / 1.38
        - 0.5 * one.widths_m[2][one.bottom_face + 1] / 1.0,
        1.0 / G_TAIRTE4_SIO2_W_M2K,
    )


def test_linear_gray_law_has_unit_derivative_at_void_endpoint():
    state = build_state(np.zeros(CONTRACT.design_node_shape), gray_exponent=1.0)
    assert np.array_equal(
        state.dphi_drho_cell,
        np.ones(
            (CONTRACT.crystal_bounding_intervals,) * 2
            if CONTRACT.geometry_mode == "diagonal_45_contact_anchored"
            else CONTRACT.design_intervals
        ),
    )


def test_explicit_au_electrodes_follow_terminal_axis_without_expanding_flake():
    au_interface_conductance = 19.89e6
    state_x = build_state(
        np.ones(CONTRACT.design_node_shape),
        au_contact_axis="x",
        au_tairte4_interface_conductance_W_m2K=au_interface_conductance,
    )
    state_y = build_state(
        np.ones(CONTRACT.design_node_shape),
        au_contact_axis="y",
        au_tairte4_interface_conductance_W_m2K=au_interface_conductance,
    )
    x = 0.5 * (state_x.edges_m[0][:-1] + state_x.edges_m[0][1:])
    y = 0.5 * (state_x.edges_m[1][:-1] + state_x.edges_m[1][1:])
    z = 0.5 * (state_x.edges_m[2][:-1] + state_x.edges_m[2][1:])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    au_x = state_x.masks["Au_electrodes"]
    au_y = state_y.masks["Au_electrodes"]
    assert np.any(au_x) and np.any(au_y)
    assert np.all(np.abs(xx[au_x]) < 12.0e-6)
    assert np.all(np.abs(yy[au_x]) < 12.0e-6)
    assert np.all((zz[au_x] >= 0.0) & (zz[au_x] < 50.0e-9))
    assert np.all(np.abs(xx[au_x]) >= 10.0e-6)
    assert np.all(np.abs(yy[au_y]) >= 10.0e-6)
    assert np.all(state_x.kappa_W_mK[au_x] == K_AU_W_MK)
    assert np.all(state_y.kappa_W_mK[au_y] == K_AU_W_MK)

    top_flake_face = int(
        np.flatnonzero(
            np.isclose(state_x.edges_m[2], 0.0, rtol=0.0, atol=2.0e-18)
        )[0] - 1
    )
    x_inside_flake = np.abs(x) < 12.0e-6
    y_inside_flake = np.abs(y) < 12.0e-6
    x_contact = (np.abs(x) >= 10.0e-6) & (np.abs(x) < 12.0e-6)
    y_contact = (np.abs(y) >= 10.0e-6) & (np.abs(y) < 12.0e-6)
    expected_resistance = 1.0 / au_interface_conductance
    assert np.allclose(
        state_x.interface_resistance_m2K_W["z"][
            np.ix_(x_contact, y_inside_flake, [top_flake_face])
        ],
        expected_resistance,
    )
    assert np.allclose(
        state_y.interface_resistance_m2K_W["z"][
            np.ix_(x_inside_flake, y_contact, [top_flake_face])
        ],
        expected_resistance,
    )


def test_diagonal_au_stays_on_rotated_flake_terminal_strips():
    if CONTRACT.geometry_mode != "diagonal_45_contact_anchored":
        return
    state = build_state(
        np.ones(CONTRACT.design_node_shape), au_contact_axis="diagonal_45"
    )
    x = 0.5 * (state.edges_m[0][:-1] + state.edges_m[0][1:])
    y = 0.5 * (state.edges_m[1][:-1] + state.edges_m[1][1:])
    z = 0.5 * (state.edges_m[2][:-1] + state.edges_m[2][1:])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    u, v = CONTRACT.rotated_uv(xx, yy)
    au = state.masks["Au_electrodes"]
    assert np.any(au)
    assert np.all(np.abs(u[au]) < 12.0e-6)
    assert np.all(np.abs(v[au]) < 12.0e-6)
    assert np.all(np.abs(u[au]) >= 10.0e-6)
    assert np.all((zz[au] >= 0.0) & (zz[au] < 50.0e-9))
