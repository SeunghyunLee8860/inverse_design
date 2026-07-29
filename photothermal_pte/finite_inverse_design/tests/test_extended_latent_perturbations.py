from __future__ import annotations

import numpy as np

from photothermal_pte.finite_inverse_design.run_extended_latent_perturbation_adfd import (
    extra_directions,
)


def test_extra_latent_directions_are_named_finite_and_normalized():
    directions = extra_directions()
    assert list(directions) == [
        "uniform",
        "x_antisymmetric",
        "y_antisymmetric",
        "diagonal_quadrupole",
        "radial_ring",
    ]
    for direction in directions.values():
        assert direction.shape == (81, 81)
        assert np.all(np.isfinite(direction))
        assert np.isclose(np.max(np.abs(direction)), 1.0)


def test_extra_latent_directions_are_not_collinear():
    values = np.stack(
        [direction.reshape(-1) for direction in extra_directions().values()]
    )
    normalized = values / np.linalg.norm(values, axis=1, keepdims=True)
    gram = normalized @ normalized.T
    assert np.max(np.abs(gram - np.eye(5))) < 0.75
