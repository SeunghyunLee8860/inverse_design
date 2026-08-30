from __future__ import annotations

import numpy as np

from tairte4_boundary_adjoint.baseline import ElectricalModel, load_config
from tairte4_boundary_adjoint.robin import DifferentiableContactModel
from tairte4_boundary_adjoint.scaled import SignedBranchObjective


def _objective() -> SignedBranchObjective:
    electrical = ElectricalModel(load_config())
    x = electrical.mesh.nodes_m[:, 0]
    y = electrical.mesh.nodes_m[:, 1]
    temperature = (
        300.0
        + 10.0
        * np.exp(-2.0 * ((x - 2.0e-6) ** 2 + (y + 3.0e-6) ** 2) / (8.5e-6) ** 2)
    ).reshape(electrical.mesh.shape)
    model = DifferentiableContactModel(
        electrical,
        temperature,
        contact_conductance_S_m2=1.0e12,
        transition_m=0.75e-6,
    )
    return SignedBranchObjective(model)


def test_signed_branches_are_exact_opposites() -> None:
    objective = _objective()
    perimeter = objective.perimeter_m
    x = np.asarray([-0.01, 8e-6 / perimeter, 0.51, 11e-6 / perimeter])
    plus = objective.evaluate(x, branch_sign=+1)
    minus = objective.evaluate(x, branch_sign=-1)
    assert plus.minimization_objective == -minus.minimization_objective
    np.testing.assert_allclose(
        plus.minimization_gradient_scaled,
        -minus.minimization_gradient_scaled,
        rtol=2e-14,
        atol=2e-14,
    )


def test_lifted_center_crosses_seam_without_a_box_boundary() -> None:
    objective = _objective()
    perimeter = objective.perimeter_m
    x = np.asarray([-0.01, 8e-6 / perimeter, 0.51, 11e-6 / perimeter])
    shifted = x.copy()
    shifted[0] += 1.0
    first = objective.evaluate(x, branch_sign=+1)
    second = objective.evaluate(shifted, branch_sign=+1)
    np.testing.assert_allclose(first.current_A, second.current_A, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(
        first.minimization_gradient_scaled,
        second.minimization_gradient_scaled,
        rtol=1e-10,
        atol=1e-12,
    )
    assert first.canonical_design.center_0_lifted == 0.99
    bounds = objective.length_bounds(1e-6, 20.7e-6)
    assert bounds[0] == (None, None)
    assert bounds[2] == (None, None)


def test_scaled_constraint_is_periodic_and_order_one() -> None:
    objective = _objective()
    perimeter = objective.perimeter_m
    x = np.asarray([-0.01, 8e-6 / perimeter, 0.51, 11e-6 / perimeter])
    shifted = x.copy()
    shifted[[0, 2]] += np.asarray([2.0, -3.0])
    gap = 0.5e-6 / perimeter
    values, jacobian = objective.model.perimeter.separation_constraints_scaled(x, gap)
    shifted_values, shifted_jacobian = (
        objective.model.perimeter.separation_constraints_scaled(shifted, gap)
    )
    np.testing.assert_allclose(values, shifted_values, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(jacobian, shifted_jacobian, rtol=0.0, atol=2e-13)
    assert np.max(np.abs(jacobian)) < 2.0 * np.pi + 1e-12


def test_scaled_objective_gradient_uses_declared_chain_rule() -> None:
    objective = _objective()
    perimeter = objective.perimeter_m
    x = np.asarray([0.10, 8e-6 / perimeter, 0.60, 11e-6 / perimeter])
    result = objective.evaluate(x, branch_sign=-1)
    expected = (
        perimeter
        * result.forward.current_gradient_A_per_m
        / result.current_scale_A
    )
    np.testing.assert_allclose(result.minimization_gradient_scaled, expected)
