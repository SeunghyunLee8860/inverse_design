import numpy as np

from photothermal_pte.finite_inverse_design.native_yee_q import EPS0
from photothermal_pte.finite_inverse_design.run_corrected_combined_physical_rho_pte_adfd import (
    component_gradient,
    fd_step_convergence,
)
from photothermal_pte.finite_inverse_design.run_v261_large_background_mixed_optical_adfd import (
    component_volumes,
)


class CaptureVjp:
    def __init__(self) -> None:
        self.calls = []

    def vjp(self, cotangent):
        self.calls.append(cotangent)
        return np.zeros((81, 81), float)


def test_component_gradient_uses_complete_yee_dual_cell_volume() -> None:
    axis = np.asarray([-1.0, 0.0, 1.0])
    grid = {
        "x": axis,
        "y": axis,
        "z": axis,
        "f": np.asarray([1.0]),
        "delta_x": np.zeros(3),
        "delta_y": np.zeros(3),
        "delta_z": np.zeros(3),
    }
    electric = np.ones((3, 3, 3, 1, 3), np.complex128)
    base = {"grid": grid, "electric": electric}
    operator = CaptureVjp()
    component_gradient(
        operator=operator,
        base=base,
        adjoint_electric=electric,
        coefficient=np.zeros((3, 3, 3, 3), float),
        profile_scale=1.0,
        base_amplitude=1.0,
    )
    expected = component_volumes(grid)
    assert len(operator.calls) == 2
    indirect = operator.calls[0]
    for index, component in enumerate("xyz"):
        np.testing.assert_array_equal(
            indirect[component],
            2.0 * EPS0 * expected[index],
        )


def test_fd_step_convergence_accepts_stable_nonmonotone_noise_plateau():
    rows = [
        {"finite_difference_directional_A": 1.0e-15},
        {"finite_difference_directional_A": 1.0e-15 + 3.0e-21},
        {"finite_difference_directional_A": 1.0e-15 - 2.0e-21},
    ]

    result = fd_step_convergence(rows)

    assert not result["strict_monotone_difference_reduction"]
    assert result["step_plateau_relative"] < 1.0e-3
    assert result["step_convergence_passed"]
