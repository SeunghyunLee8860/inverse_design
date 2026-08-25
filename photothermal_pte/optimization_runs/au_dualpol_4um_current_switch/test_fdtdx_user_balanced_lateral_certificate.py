from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_lateral_certificate import (
    STATUS_BLOCKED,
    STATUS_PAIR_PASS,
    _interpolate_axis,
    _restrict_component_q_to_coarse_xy,
)


def test_complex_lateral_interpolation_is_exact_for_affine_field() -> None:
    source = np.linspace(-1.0, 1.0, 9)
    target = np.linspace(-0.75, 0.75, 4)
    other = np.arange(3.0)
    value = (2.0 + 3.0j) * source[:, None] + other[None, :]
    actual = _interpolate_axis(value, source, target, axis=0)
    expected = (2.0 + 3.0j) * target[:, None] + other[None, :]
    assert actual == pytest.approx(expected)


def _control_widths(
    edges: np.ndarray, bounds: tuple[int, int], component: int, axis: int
) -> np.ndarray:
    lower, upper = bounds
    indices = np.arange(lower, upper)
    if component == axis:
        return edges[indices + 1] - edges[indices]
    return 0.5 * (edges[indices + 1] - edges[indices - 1])


@pytest.mark.parametrize("component", (0, 1, 2))
def test_xy_q_restriction_preserves_constant_density_and_power(component: int) -> None:
    coarse_axis = np.arange(-1.0, 43.0)
    fine_axis = np.arange(-1.0, 42.5, 0.5)
    z_edges = np.array([0.0, 1.0, 2.0])
    coarse_edges = (coarse_axis, coarse_axis, z_edges)
    fine_edges = (fine_axis, fine_axis, z_edges)
    coarse_bounds = ((1, 41), (1, 41), (0, 2))
    fine_bounds = ((2, 82), (2, 82), (0, 2))
    coarse_shape = (40, 40, 2)
    fine_shape = (80, 80, 2)
    coarse_q = np.full(coarse_shape, 7.0)
    fine_q = np.full(fine_shape, 7.0)
    coarse_wx = _control_widths(coarse_axis, coarse_bounds[0], component, 0)
    coarse_wy = _control_widths(coarse_axis, coarse_bounds[1], component, 1)
    fine_wx = _control_widths(fine_axis, fine_bounds[0], component, 0)
    fine_wy = _control_widths(fine_axis, fine_bounds[1], component, 1)
    z_measure = np.array([0.8, 1.2])
    coarse_volume = (
        coarse_wx[:, None, None]
        * coarse_wy[None, :, None]
        * z_measure[None, None, :]
    )
    fine_volume = (
        fine_wx[:, None, None]
        * fine_wy[None, :, None]
        * z_measure[None, None, :]
    )
    coarse, restricted, _, audit = _restrict_component_q_to_coarse_xy(
        coarse_q,
        fine_q,
        coarse_volume,
        fine_volume,
        coarse_edges,
        fine_edges,
        coarse_bounds,
        fine_bounds,
        component,
    )
    assert restricted == pytest.approx(coarse)
    assert audit["ready"] is True
    assert audit["fine_restriction_relative_power_error"] <= 5.0e-13


def test_pair_pass_status_does_not_claim_production_convergence() -> None:
    assert STATUS_PAIR_PASS != STATUS_BLOCKED
    assert "PAIR_PASS_DIAGNOSTIC" in STATUS_PAIR_PASS
