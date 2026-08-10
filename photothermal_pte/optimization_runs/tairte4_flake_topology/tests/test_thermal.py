import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.thermal import (
    G_TAIRTE4_SIO2_W_M2K,
    K_AIR_W_MK,
    K_TAIRTE4_XYZ_W_MK,
    build_state,
    cell_to_node,
    cell_to_node_transpose,
    nodal_to_cell,
    nodal_to_cell_transpose,
)


def test_nodal_cell_transpose():
    rng = np.random.default_rng(4)
    node = rng.normal(size=CONTRACT.design_node_shape)
    cell = rng.normal(size=CONTRACT.design_intervals)
    assert np.isclose(np.sum(nodal_to_cell(node) * cell), np.sum(node * nodal_to_cell_transpose(cell)), rtol=1e-13)


def test_temperature_cell_node_transpose():
    rng = np.random.default_rng(5)
    cell = rng.normal(size=(240, 240))
    node = rng.normal(size=(241, 241))
    assert np.isclose(np.sum(cell_to_node(cell) * node), np.sum(cell * cell_to_node_transpose(node)), rtol=1e-13)


def test_bottom_contact_endpoints_and_axis_mapping():
    zero = build_state(np.zeros(CONTRACT.design_node_shape))
    one = build_state(np.ones(CONTRACT.design_node_shape))
    assert np.allclose(K_TAIRTE4_XYZ_W_MK, (3.8, 14.4, 1.0))
    assert np.isclose(zero.kappa_W_mK[zero.masks["design_effective"]][0, 0], K_AIR_W_MK)
    assert np.allclose(one.kappa_W_mK[one.masks["design_effective"]][0], K_TAIRTE4_XYZ_W_MK)
    assert np.isclose(
        one.bottom_tairte4_path_resistance_m2K_W
        - 0.5 * one.widths_m[2][one.bottom_face] / 1.38
        - 0.5 * one.widths_m[2][one.bottom_face + 1] / 1.0,
        1.0 / G_TAIRTE4_SIO2_W_M2K,
    )
