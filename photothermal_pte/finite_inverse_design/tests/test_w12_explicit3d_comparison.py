from __future__ import annotations

import numpy as np

from photothermal_pte.validation.paper_ir_sanity import (
    compare_w12_50nm_maxwell_analytic_explicit3d as comparison,
)
from photothermal_pte.validation.paper_ir_sanity import (
    run_device_a_explicit_thermal_pte as thermal,
)


def small_flake_geometry() -> thermal.Geometry:
    x_edges = np.asarray([-2.0e-6, -0.5e-6, 0.5e-6, 2.0e-6])
    y_edges = np.asarray([-2.0e-6, -0.5e-6, 0.5e-6, 2.0e-6])
    z_edges = np.asarray([-130.0e-9, -65.0e-9, 0.0])
    shape = (3, 3, 2)
    flake = np.ones(shape, bool)
    material = np.full(shape, 3, np.uint8)
    kappa = np.broadcast_to(
        thermal.KAPPA_TAIRTE4_LAB_W_MK, (*shape, 3)
    ).copy()
    return thermal.Geometry(
        x_edges,
        y_edges,
        z_edges,
        material,
        flake,
        kappa,
        {
            "x": np.zeros((2, 3, 2)),
            "y": np.zeros((3, 2, 2)),
            "z": np.zeros((3, 3, 1)),
        },
    )


def test_analytic_q_is_full_volumetric_cell_integral() -> None:
    geometry = small_flake_geometry()
    beam = {
        "fit": {
            "waist_x_m": 12.0e-6,
            "waist_y_m": 11.8e-6,
            "center_x_m": 0.15e-6,
            "center_y_m": -0.1e-6,
        }
    }
    q, contract = comparison.analytic_volumetric_q(geometry, "a", beam)
    volume = comparison.cell_volume(geometry)
    sigma_x = beam["fit"]["waist_x_m"] / 2.0
    sigma_y = beam["fit"]["waist_y_m"] / 2.0
    lateral_fraction = np.sum(
        comparison.exact_gaussian_cell_fractions(
            geometry.x_edges_m, beam["fit"]["center_x_m"], sigma_x
        )
    ) * np.sum(
        comparison.exact_gaussian_cell_fractions(
            geometry.y_edges_m, beam["fit"]["center_y_m"], sigma_y
        )
    )
    expected = (
        comparison.INCIDENT_POWER_W
        * comparison.TMM_ABSORPTION["a"]
        * lateral_fraction
    )
    assert q.shape == geometry.flake_mask.shape
    assert np.all(np.isfinite(q))
    assert np.min(q) >= 0.0
    assert np.isclose(np.sum(q * volume), expected, rtol=1.0e-13)
    assert contract["discretization"].startswith("analytic Gaussian")
    assert "sheet" not in contract["equation"].lower()


def test_gradient_axis_contract_is_lab_x_b_y_a() -> None:
    assert np.array_equal(
        thermal.KAPPA_TAIRTE4_LAB_W_MK,
        np.asarray([3.8, 14.4, 1.0]),
    )
