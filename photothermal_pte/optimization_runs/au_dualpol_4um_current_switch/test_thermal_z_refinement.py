from __future__ import annotations

import inspect

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    _refine_edges,
    build_thermal_state,
    thermal_edges,
)


def _count_layer(edges: np.ndarray, lower: float, upper: float) -> int:
    centers = 0.5 * (edges[:-1] + edges[1:])
    return int(np.count_nonzero((centers >= lower) & (centers < upper)))


def test_default_thermal_edges_are_exact_factor_one_alias():
    default = thermal_edges()
    explicit = thermal_edges(1)
    assert all(
        np.array_equal(left, right)
        for left, right in zip(default, explicit, strict=True)
    )
    assert np.array_equal(default[0], default[1])


@pytest.mark.parametrize("factor", [2, 4])
def test_z_refinement_preserves_interfaces_and_multiplies_every_interval(factor):
    coarse = thermal_edges(1)
    refined = thermal_edges(factor)
    assert np.array_equal(refined[0], coarse[0])
    assert np.array_equal(refined[1], coarse[1])
    assert refined[2].size - 1 == factor * (coarse[2].size - 1)
    assert np.array_equal(refined[2][::factor], coarse[2])
    assert np.all(np.diff(refined[2]) > 0.0)
    assert _count_layer(refined[2], -0.385e-6, -0.1e-6) == 3 * factor
    assert _count_layer(refined[2], -0.1e-6, 0.0) == 10 * factor
    assert _count_layer(refined[2], 0.0, 0.05e-6) == 3 * factor


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_invalid_refinement_factor_fails_closed(factor):
    with pytest.raises(ValueError, match="positive integer"):
        thermal_edges(factor)


@pytest.mark.parametrize(
    "edges",
    [
        np.asarray([0.0]),
        np.asarray([0.0, np.nan]),
        np.asarray([0.0, np.inf]),
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([0.0, -1.0]),
        np.zeros((2, 2)),
    ],
)
def test_invalid_edges_fail_closed(edges):
    with pytest.raises(ValueError, match="finite and strictly increasing"):
        _refine_edges(edges, 2)


def test_build_thermal_state_keeps_default_and_exposes_keyword_only_z_factor():
    signature = inspect.signature(build_thermal_state)
    assert signature.parameters["z_refinement_factor"].default == 1
    assert (
        signature.parameters["z_refinement_factor"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert signature.parameters["xy_refinement_factor"].default == 1
    assert (
        signature.parameters["xy_refinement_factor"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


@pytest.mark.parametrize("factor", [2, 4])
def test_xy_refinement_preserves_all_original_faces_and_leaves_z_fixed(factor):
    coarse = thermal_edges(2, xy_refinement_factor=1)
    refined = thermal_edges(2, xy_refinement_factor=factor)
    assert refined[0].size - 1 == factor * (coarse[0].size - 1)
    assert refined[1].size - 1 == factor * (coarse[1].size - 1)
    assert np.array_equal(refined[0][::factor], coarse[0])
    assert np.array_equal(refined[1][::factor], coarse[1])
    assert np.array_equal(refined[2], coarse[2])
    assert _count_layer(refined[0], -8e-6, 8e-6) == 160 * factor
    assert _count_layer(refined[0], -4e-6, 4e-6) == 80 * factor


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_invalid_xy_refinement_factor_fails_closed(factor):
    with pytest.raises(ValueError, match="positive integer"):
        thermal_edges(2, xy_refinement_factor=factor)


@pytest.mark.parametrize(
    ("keyword", "levels", "expected_bounds"),
    [
        ("lateral_half_span_um", (32, 48, 64), (-64e-6, 64e-6)),
        ("substrate_depth_um", (20, 30, 40), (-40e-6, 2e-6)),
        ("top_air_height_um", (2.0, 3.0, 4.0), (-20e-6, 4e-6)),
    ],
)
def test_domain_ladders_preserve_the_baseline_and_only_extend_outer_faces(
    keyword, levels, expected_bounds
):
    baseline = thermal_edges(2, xy_refinement_factor=2)
    finest = thermal_edges(2, xy_refinement_factor=2, **{keyword: levels[-1]})
    if keyword == "lateral_half_span_um":
        assert np.array_equal(finest[0][16:-16], baseline[0])
        assert np.array_equal(finest[1][16:-16], baseline[1])
        assert np.array_equal(finest[2], baseline[2])
        assert (finest[0][0], finest[0][-1]) == expected_bounds
    else:
        assert np.array_equal(finest[0], baseline[0])
        assert np.array_equal(finest[1], baseline[1])
        if keyword == "substrate_depth_um":
            assert np.array_equal(finest[2][4:], baseline[2])
        else:
            assert np.array_equal(finest[2][:-8], baseline[2])
        np.testing.assert_allclose(
            (finest[2][0], finest[2][-1]), expected_bounds, rtol=0.0, atol=1e-20
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("lateral_half_span_um", 36, "lateral half-span"),
        ("substrate_depth_um", 25, "substrate depth"),
        ("top_air_height_um", 2.5, "top-air height"),
    ],
)
def test_undeclared_domain_levels_fail_closed(keyword, value, message):
    with pytest.raises(ValueError, match=message):
        thermal_edges(**{keyword: value})
