from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


THERMAL = Path(__file__).resolve().parents[1]
if str(THERMAL) not in sys.path:
    sys.path.insert(0, str(THERMAL))

from conservative_remap import (
    build_conservative_density_remap,
    build_nodal_density_remap_1d,
    nodal_control_volume_edges,
)


def _remap():
    source = (
        np.asarray([-3.0, -1.8, -0.2, 1.1, 3.0]) * 1e-6,
        np.asarray([-3.0, -0.7, 0.6, 3.0]) * 1e-6,
        np.asarray([-100.0, -73.0, -22.0, 0.0]) * 1e-9,
    )
    target = (
        np.asarray([-3.0, -2.2, 0.4, 3.0]) * 1e-6,
        np.asarray([-3.0, -1.3, 1.7, 3.0]) * 1e-6,
        np.asarray([-100.0, -50.0, 0.0]) * 1e-9,
    )
    return build_conservative_density_remap(
        source_edges_m=source, target_edges_m=target
    )


def test_constant_density_and_total_power_are_exact():
    remap = _remap()
    source = np.full(remap.source_shape, 7.25)
    target = remap.apply(source)
    assert np.allclose(target, 7.25, rtol=1e-14, atol=1e-14)
    assert np.isclose(
        remap.power_target(target),
        remap.power_source(source),
        rtol=2e-15,
    )


def test_density_operator_weighted_dot_transpose():
    remap = _remap()
    rng = np.random.default_rng(2026072606)
    source = rng.normal(size=remap.source_shape)
    target_weight = rng.normal(size=remap.target_shape)
    left = float(np.sum(target_weight * remap.apply(source)))
    right = float(np.sum(remap.transpose(target_weight) * source))
    assert np.isclose(left, right, rtol=2e-15, atol=1e-30)


def test_power_weight_pullback_includes_target_volume_once():
    remap = _remap()
    rng = np.random.default_rng(2026072607)
    source = rng.normal(size=remap.source_shape)
    thermal_adjoint = rng.normal(size=remap.target_shape)
    left = float(
        np.sum(
            thermal_adjoint
            * remap.target_volume_m3
            * remap.apply(source)
        )
    )
    optical_weight = remap.transpose(
        thermal_adjoint * remap.target_volume_m3
    )
    right = float(np.sum(optical_weight * source))
    assert np.isclose(left, right, rtol=2e-15, atol=1e-30)


def test_different_bounds_fail_closed():
    with pytest.raises(ValueError, match="fail closed"):
        build_conservative_density_remap(
            source_edges_m=(
                np.asarray([0.0, 1.0]),
                np.asarray([0.0, 1.0]),
                np.asarray([0.0, 1.0]),
            ),
            target_edges_m=(
                np.asarray([0.0, 0.9]),
                np.asarray([0.0, 1.0]),
                np.asarray([0.0, 1.0]),
            ),
        )


def test_nodal_shifted_periodic_remap_conserves_mass_and_transpose():
    target = np.linspace(-3e-6, 3e-6, 13)
    delta = 0.5 * (target[1] - target[0])
    source = target + delta
    remap = build_nodal_density_remap_1d(
        source_coordinates_m=source,
        target_coordinates_m=target,
        periodic=True,
    )
    rng = np.random.default_rng(2026072608)
    density = rng.normal(size=(source.size, 4, 3))
    mapped = remap.apply_axis(density, axis=0)
    source_mass = np.einsum(
        "i,ijk->", remap.source_weight_m, density
    )
    target_mass = np.einsum(
        "i,ijk->", remap.target_weight_m, mapped
    )
    assert np.isclose(source_mass, target_mass, rtol=2e-14, atol=1e-30)
    sensitivity = rng.normal(size=mapped.shape)
    left = float(np.sum(sensitivity * mapped))
    right = float(
        np.sum(remap.transpose_axis(sensitivity, axis=0) * density)
    )
    assert np.isclose(left, right, rtol=2e-14, atol=1e-30)


def test_nodal_control_edges_reproduce_trapezoid_weights():
    coordinates = np.asarray([-3.0, -2.2, -0.1, 0.7, 3.0]) * 1e-6
    edges = nodal_control_volume_edges(coordinates)
    expected = np.asarray(
        [
            0.5 * (coordinates[1] - coordinates[0]),
            0.5 * (coordinates[2] - coordinates[0]),
            0.5 * (coordinates[3] - coordinates[1]),
            0.5 * (coordinates[4] - coordinates[2]),
            0.5 * (coordinates[4] - coordinates[3]),
        ]
    )
    assert np.allclose(np.diff(edges), expected, rtol=2e-15, atol=1e-30)
