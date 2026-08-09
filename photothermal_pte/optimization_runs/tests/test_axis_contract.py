import numpy as np
import pytest

from photothermal_pte.optimization_runs.axis_contract import X_B_Y_A
from photothermal_pte.thermal_adjoint.local_pte_functional import (
    build_local_pte_functional,
)


def test_paper_coordinate_contract_values():
    assert X_B_Y_A.crystal_axis_by_solver_axis == ("b", "a", "c")
    assert X_B_Y_A.epsilon_axis_by_solver_axis == ("b", "a", "b_closure_for_c")
    assert X_B_Y_A.kappa_xyz_W_mK == (3.8, 14.4, 1.0)
    assert X_B_Y_A.sigma_xy_S_m == (1.10e5, 4.91e5)
    assert X_B_Y_A.seebeck_xy_V_K == (27.0e-6, -6.0e-6)


def test_paper_coordinate_contract_polarizations():
    X_B_Y_A.validate_polarization("E_parallel_b", 0.0)
    X_B_Y_A.validate_polarization("E_parallel_a", 90.0)
    with pytest.raises(RuntimeError):
        X_B_Y_A.validate_polarization("E_parallel_b", 90.0)
    assert np.isclose(X_B_Y_A.polarization_angle_deg["E_parallel_a"], 90.0)


def test_x_b_y_a_coefficients_pair_with_literal_x_y_derivatives():
    edges = np.arange(4, dtype=float)
    active = np.ones((3, 3, 3), dtype=bool)
    ids = np.arange(active.size).reshape(active.shape)
    functional = build_local_pte_functional(
        x_edges_m=edges,
        y_edges_m=edges,
        z_edges_m=edges,
        active_mask=active,
        active_ids=ids,
        fom_mask=active,
        sigma_a_S_m=X_B_Y_A.sigma_xy_S_m[0],
        sigma_b_S_m=X_B_Y_A.sigma_xy_S_m[1],
        seebeck_a_V_K=X_B_Y_A.seebeck_xy_V_K[0],
        seebeck_b_V_K=X_B_Y_A.seebeck_xy_V_K[1],
    )
    assert functional.sigma_a_S_m == X_B_Y_A.sigma_xy_S_m[0]
    assert functional.seebeck_a_V_K == X_B_Y_A.seebeck_xy_V_K[0]
    assert functional.sigma_b_S_m == X_B_Y_A.sigma_xy_S_m[1]
    assert functional.seebeck_b_V_K == X_B_Y_A.seebeck_xy_V_K[1]
