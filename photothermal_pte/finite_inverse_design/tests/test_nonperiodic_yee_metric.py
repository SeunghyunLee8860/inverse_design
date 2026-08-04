import numpy as np
import pytest

from photothermal_pte.finite_inverse_design.nonperiodic_yee_metric import (
    clipped_component_yee_volumes,
    clipped_voronoi_weights,
)


def test_clipped_weights_integrate_exact_support_length() -> None:
    coordinate = np.array([-0.9, -0.4, 0.1, 0.6, 1.1])
    weights = clipped_voronoi_weights(coordinate, -1.0, 1.0)
    assert np.sum(weights) == pytest.approx(2.0)
    assert weights[-1] > 0.0


def test_all_component_volumes_equal_exact_box_volume() -> None:
    grid = {
        "x": np.linspace(-1.0, 1.0, 5),
        "y": np.linspace(-1.0, 1.0, 5),
        "z": np.array([0.0, 0.2, 0.4, 0.7]),
        "delta_x": np.full(5, 0.25),
        "delta_y": np.full(5, 0.25),
        "delta_z": np.array([0.1, 0.1, 0.15, 0.15]),
    }
    bounds = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 0.6)}
    volumes = clipped_component_yee_volumes(grid, bounds)
    for component in range(3):
        assert np.sum(volumes[component]) == pytest.approx(2.4)
