from __future__ import annotations

import numpy as np

from photothermal_pte.validation.paper_ir_sanity.analyze_w2_masked_planar_offline import (
    bounded_dual_cells,
    conservative_remap,
    half_plane_cut_fraction,
    ideal_symmetric_gaussian_control,
    signed_power_decomposition,
)


def test_exact_half_plane_cut_cells_have_half_area() -> None:
    coordinate = np.linspace(-1.0, 1.0, 18)
    fraction = half_plane_cut_fraction(
        coordinate, coordinate, [-1.0, 1.0], [-1.0, 1.0]
    )
    width = bounded_dual_cells(coordinate, -1.0, 1.0)[2]
    area = width[:, None] * width[None, :]
    assert np.isclose(np.sum(fraction * area) / np.sum(area), 0.5)
    assert np.count_nonzero((fraction > 0.0) & (fraction < 1.0)) > 0


def test_ideal_equal_power_half_plane_nrmse_is_one() -> None:
    result = ideal_symmetric_gaussian_control()
    assert np.isclose(result["P_half_over_P_full"], 0.5, atol=1e-12)
    assert np.isclose(
        result["equal_power_full_half_NRMSE"], 1.0, atol=1e-2
    )


def test_signed_power_decomposition_closes_without_absolute_value() -> None:
    volume = np.ones((2, 2, 1))
    full = np.full((2, 2, 1), 2.0)
    masked = np.full((2, 2, 1), 1.2)
    edge = np.full((2, 2, 1), 1.5)
    result = signed_power_decomposition(full, masked, edge, volume)
    assert result["D_EM_W_signed"] < 0.0
    assert result["decomposition_closure_W"] == 0.0


def test_conservative_dual_cell_remap_preserves_power() -> None:
    source_coordinates = {
        "x": np.linspace(-1.0, 1.0, 11),
        "y": np.linspace(-1.0, 1.0, 9),
        "z": np.linspace(-0.5, 0.5, 7),
    }
    target_coordinates = {
        "x": np.linspace(-1.0, 1.0, 14),
        "y": np.linspace(-1.0, 1.0, 12),
        "z": np.linspace(-0.5, 0.5, 8),
    }
    bounds = {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [-0.5, 0.5]}
    xx, yy, zz = np.meshgrid(
        source_coordinates["x"],
        source_coordinates["y"],
        source_coordinates["z"],
        indexing="ij",
    )
    values = 2.0 + 0.2 * xx - 0.1 * yy + 0.05 * zz
    _, audit = conservative_remap(
        values, source_coordinates, target_coordinates, bounds
    )
    assert audit["relative_power_error"] < 1e-14
