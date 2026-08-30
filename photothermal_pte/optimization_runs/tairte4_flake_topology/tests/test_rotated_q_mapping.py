from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.rotated_q_mapping import (
    build_rotated_scalar_map,
    make_control_volume_conservative,
)


def test_rotated_scalar_map_preserves_constant_and_transpose_identity() -> None:
    source = tuple(np.linspace(-2.0, 2.0, 9) for _ in range(3))
    target = (
        np.linspace(-1.0, 1.0, 7),
        np.linspace(-1.0, 1.0, 7),
        np.linspace(-1.0, 1.0, 5),
    )
    support = np.ones(tuple(value.size for value in target), dtype=bool)
    mapping = build_rotated_scalar_map(source, target, support)
    assert np.allclose(mapping.apply(np.ones(mapping.source_shape)), 1.0)
    rng = np.random.default_rng(640)
    left = rng.normal(size=mapping.source_shape)
    right = rng.normal(size=mapping.target_shape)
    assert np.isclose(
        np.vdot(mapping.apply(left), right),
        np.vdot(left, mapping.transpose(right)),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_conservative_rotation_preserves_represented_control_volume_power() -> None:
    source = tuple(np.linspace(-2.0, 2.0, 9) for _ in range(3))
    target = (
        np.linspace(-1.0, 1.0, 7),
        np.linspace(-1.0, 1.0, 7),
        np.linspace(-1.0, 1.0, 5),
    )
    support = np.ones(tuple(value.size for value in target), dtype=bool)
    raw = build_rotated_scalar_map(source, target, support)
    source_volume = np.full(raw.source_shape, 0.125)
    target_volume = np.full(raw.target_shape, 0.05)
    mapping, represented = make_control_volume_conservative(
        raw, source_volume, target_volume
    )
    rng = np.random.default_rng(641)
    density = rng.random(raw.source_shape)
    expected = float(np.sum(density[represented] * source_volume[represented]))
    actual = float(np.sum(mapping.apply(density) * target_volume))
    assert np.isclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)
