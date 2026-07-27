import numpy as np
import pytest

from photothermal_pte.finite_inverse_design.finite_q_mapping import (
    build_conservative_embedding_remap,
    exact_nonzero_box,
    nodal_control_volume_edges,
)


def test_embedding_conserves_power_and_exact_transpose() -> None:
    source_edges = (
        np.array([-0.7, -0.2, 0.4]),
        np.array([-0.5, 0.1, 0.6]),
        np.array([-0.2, 0.0, 0.3]),
    )
    target_edges = (
        np.linspace(-1.0, 1.0, 9),
        np.linspace(-1.0, 1.0, 11),
        np.linspace(-0.5, 0.5, 7),
    )
    remap = build_conservative_embedding_remap(
        source_edges_m=source_edges,
        target_edges_m=target_edges,
    )
    rng = np.random.default_rng(2026072702)
    source = rng.random(remap.source_shape)
    target_weight = rng.normal(size=remap.target_shape)
    target = remap.apply(source)
    assert remap.power_target(target) == pytest.approx(
        remap.power_source(source), rel=2e-13
    )
    assert np.sum(target_weight * target) == pytest.approx(
        np.sum(remap.transpose(target_weight) * source), rel=2e-13
    )


def test_embedding_fails_closed_if_target_crops_source() -> None:
    with pytest.raises(ValueError, match="fully contained"):
        build_conservative_embedding_remap(
            source_edges_m=(
                np.array([-2.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
            ),
            target_edges_m=(
                np.array([-1.0, 1.0]),
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
            ),
        )


def test_nodal_edges_match_trapezoid_length_and_nonzero_box() -> None:
    coordinate = np.array([-1.0, -0.3, 0.2, 1.0])
    edges = nodal_control_volume_edges(coordinate)
    assert np.sum(np.diff(edges)) == pytest.approx(2.0)
    density = np.zeros((5, 6, 7))
    density[1:4, 2:5, 3:6] = 2.0
    box, outside_nonzero = exact_nonzero_box(density)
    assert outside_nonzero == 0
    assert density[box].shape == (3, 3, 3)
