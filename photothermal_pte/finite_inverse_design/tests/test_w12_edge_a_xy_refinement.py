from __future__ import annotations

import numpy as np

from photothermal_pte.validation.paper_ir_sanity.summarize_w12_edge_a_xy_refinement import (
    bounded_dual_cells,
    overlap_fraction,
    remap_energy,
)


def test_bounded_dual_cells_drops_zero_overlap_endpoint() -> None:
    coordinate = np.asarray([-1.0, 0.0, 1.0, 2.0])
    indices, edges = bounded_dual_cells(coordinate, -1.5, 1.5)
    assert np.array_equal(indices, np.asarray([0, 1, 2]))
    assert np.array_equal(edges, np.asarray([-1.5, -0.5, 0.5, 1.5]))


def test_separable_conservative_remap_preserves_energy() -> None:
    source_edges = (
        np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
        np.asarray([0.0, 0.4, 1.0]),
        np.asarray([0.0, 0.2, 0.6, 1.0]),
    )
    target_edges = (
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([0.0, 0.2, 0.7, 1.0]),
        np.asarray([0.0, 0.5, 1.0]),
    )
    operators = tuple(
        overlap_fraction(target, source)
        for target, source in zip(target_edges, source_edges)
    )
    rng = np.random.default_rng(4)
    source_energy = rng.random((4, 2, 3))
    target_energy = remap_energy(source_energy, operators)
    assert target_energy.shape == (2, 3, 2)
    assert np.isclose(
        np.sum(target_energy),
        np.sum(source_energy),
        rtol=2.0e-15,
        atol=0.0,
    )
