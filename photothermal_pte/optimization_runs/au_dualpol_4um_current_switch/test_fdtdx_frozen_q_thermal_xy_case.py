from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_frozen_q_thermal_xy_case as xy_case,
)


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_block_mean_restricts_to_original_cells(factor):
    base = np.arange(12.0).reshape(3, 4)
    refined = np.repeat(np.repeat(base, factor, axis=0), factor, axis=1)
    np.testing.assert_array_equal(
        xy_case._restrict_blocks(refined, factor, reduction="mean"), base
    )


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_block_sum_exactly_conserves_power(factor):
    base = np.arange(1.0, 13.0).reshape(3, 4)
    refined = np.repeat(
        np.repeat(base / factor**2, factor, axis=0), factor, axis=1
    )
    restricted = xy_case._restrict_blocks(refined, factor, reduction="sum")
    np.testing.assert_allclose(restricted, base, rtol=0.0, atol=1e-15)
    assert np.sum(restricted) == np.sum(refined)


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_invalid_block_factor_fails_closed(factor):
    with pytest.raises(ValueError, match="positive integer"):
        xy_case._restrict_blocks(np.ones((4, 4)), factor, reduction="mean")


def test_incompatible_block_shape_and_reduction_fail_closed():
    with pytest.raises(ValueError, match="incompatible"):
        xy_case._restrict_blocks(np.ones((5, 4)), 2, reduction="mean")
    with pytest.raises(ValueError, match="reduction"):
        xy_case._restrict_blocks(np.ones((4, 4)), 2, reduction="median")


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_base_centers_reconstruct_original_uniform_subdivision(factor):
    coarse_edges = np.asarray([0.0, 1.0, 3.0])
    refined_edges = np.concatenate(
        [
            np.linspace(coarse_edges[index], coarse_edges[index + 1], factor, endpoint=False)
            for index in range(2)
        ]
        + [coarse_edges[-1:]]
    )
    refined_centers = 0.5 * (refined_edges[:-1] + refined_edges[1:])
    np.testing.assert_allclose(
        xy_case._base_centers(refined_centers, factor),
        0.5 * (coarse_edges[:-1] + coarse_edges[1:]),
        rtol=0.0,
        atol=1e-15,
    )


def test_xy_ladder_fixes_selected_diagnostic_z_and_blocks_promotion():
    assert xy_case.THERMAL_Z_REFINEMENT_FACTOR == 2
    assert xy_case.ALLOWED_XY_REFINEMENT_FACTORS == (1, 2, 4)
    assert "DIAGNOSTIC" in xy_case.STATUS_READY
