from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    NOMINAL_MAPPING,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_signed_objective import (
    signed_dual_objective_point,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.objective import (
    epigraph_constraints,
)


def test_signed_dual_point_matches_latent_directional_fd() -> None:
    rng = np.random.default_rng(7042)
    latent = np.full(NOMINAL_MAPPING.shape, 0.5)
    direction = rng.normal(size=NOMINAL_MAPPING.shape)
    direction /= np.max(np.abs(direction))
    gradient_a_projected = rng.normal(size=NOMINAL_MAPPING.shape) * 1.0e-12
    gradient_b_projected = rng.normal(size=NOMINAL_MAPPING.shape) * 1.0e-12
    beta = 4.0

    def currents(value: np.ndarray) -> tuple[float, float]:
        projected = NOMINAL_MAPPING.physical(value, beta)
        return (
            -8.0e-9 + float(np.vdot(gradient_a_projected, projected - 0.5)),
            -15.0e-9 + float(np.vdot(gradient_b_projected, projected - 0.5)),
        )

    current_a, current_b = currents(latent)
    point = signed_dual_objective_point(
        latent=latent,
        beta=beta,
        current_a_A=current_a,
        current_b_A=current_b,
        gradient_a_projected_A=gradient_a_projected,
        gradient_b_projected_A=gradient_b_projected,
        epigraph_A=current_a,
    )
    assert point["active_polarization"] == "Ea"
    assert point["balanced_utility_A"] == current_a
    assert point["epigraph_constraints_A"][0] == pytest.approx(0.0)
    assert point["epigraph_constraints_A"][1] < 0.0

    step = 1.0e-5
    plus = currents(latent + step * direction)
    minus = currents(latent - step * direction)
    fd_currents = (np.asarray(plus) - np.asarray(minus)) / (2.0 * step)
    ad_currents = np.asarray(
        (
            np.vdot(point["gradient_a_latent_A"], direction),
            np.vdot(point["gradient_b_latent_A"], direction),
        )
    )
    np.testing.assert_allclose(ad_currents, fd_currents, rtol=2.0e-8, atol=1.0e-17)

    fixed_t = current_a
    fd_constraints = (
        epigraph_constraints(*plus, fixed_t)
        - epigraph_constraints(*minus, fixed_t)
    ) / (2.0 * step)
    ad_constraints = (
        point["constraint_gradients_latent_A"].reshape(2, -1)
        @ direction.ravel()
    )
    np.testing.assert_allclose(
        ad_constraints, fd_constraints, rtol=2.0e-8, atol=1.0e-17
    )
    fd_balanced = (
        min(plus[0], -plus[1]) - min(minus[0], -minus[1])
    ) / (2.0 * step)
    ad_balanced = float(np.vdot(point["balanced_gradient_latent_A"], direction))
    assert ad_balanced == pytest.approx(fd_balanced, rel=2.0e-8, abs=1.0e-17)


def test_signed_dual_point_keeps_epigraph_valid_at_utility_tie() -> None:
    latent = np.full(NOMINAL_MAPPING.shape, 0.5)
    zero = np.zeros(NOMINAL_MAPPING.shape)
    point = signed_dual_objective_point(
        latent=latent,
        beta=2.0,
        current_a_A=3.0e-9,
        current_b_A=-3.0e-9,
        gradient_a_projected_A=zero,
        gradient_b_projected_A=zero,
        epigraph_A=3.0e-9,
    )
    assert point["active_polarization"] == "tie"
    assert point["balanced_minimum_nondifferentiable_tie"] is True
    assert point["balanced_gradient_latent_A"] is None
    np.testing.assert_array_equal(point["epigraph_constraints_A"], 0.0)
    np.testing.assert_array_equal(point["constraint_gradients_epigraph"], 1.0)


@pytest.mark.parametrize("field", ("latent", "gradient_a_projected_A"))
def test_signed_dual_point_rejects_invalid_design_arrays(field: str) -> None:
    values: dict[str, object] = {
        "latent": np.full(NOMINAL_MAPPING.shape, 0.5),
        "beta": 2.0,
        "current_a_A": 1.0,
        "current_b_A": -1.0,
        "gradient_a_projected_A": np.zeros(NOMINAL_MAPPING.shape),
        "gradient_b_projected_A": np.zeros(NOMINAL_MAPPING.shape),
        "epigraph_A": 0.0,
    }
    values[field] = np.zeros((3, 4))
    with pytest.raises(ValueError):
        signed_dual_objective_point(**values)  # type: ignore[arg-type]
