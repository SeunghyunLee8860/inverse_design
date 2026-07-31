from __future__ import annotations

import numpy as np

from photothermal_pte.validation.paper_ir_sanity.summarize_w12_edge_a_xy_refinement import (
    bounded_dual_cells,
    overlap_fraction,
    remap_energy,
)
from photothermal_pte.validation.paper_ir_sanity.analyze_w12_interface_downstream import (
    project_energy_to_support,
)
from photothermal_pte.validation.paper_ir_sanity.extract_w12_interface_index_readonly import (
    local_dual_widths,
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


def test_nearest_support_projection_updates_noncontiguous_input() -> None:
    base = np.zeros((3, 3, 2), float)
    base[0, 0, 0] = 2.0
    base[2, 2, 1] = 3.0
    source = np.moveaxis(base, 0, 1)
    assert not source.flags.c_contiguous
    support = np.zeros(source.shape, bool)
    support[0, 1, 0] = True
    support[2, 1, 1] = True
    edges = tuple(
        np.arange(size + 1, dtype=float) for size in source.shape
    )
    projected, audit = project_energy_to_support(
        source, edges, support
    )
    assert np.isclose(np.sum(projected), 5.0)
    assert np.count_nonzero(projected[~support]) == 0
    assert audit["outside_support_nonzero_cells_after_projection"] == 0
    assert audit["relative_power_error"] < 1.0e-15


def test_component_local_dual_widths() -> None:
    coordinate = np.asarray([-1.0, -0.25, 0.25, 1.0])
    widths = local_dual_widths(coordinate)
    assert np.allclose(widths, [0.75, 0.625, 0.625, 0.75])
