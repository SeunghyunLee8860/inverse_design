import numpy as np
import pytest

from photothermal_pte.validation.paper_ir_sanity.run_device_a_explicit_thermal_pte import (
    Geometry,
    measure_weighted_mean,
    solve_weighting_potential,
    straight_edge_temperature_metrics,
)


def test_weighted_mean_uses_literal_cell_measure() -> None:
    values = np.array([1.0, 3.0])
    mask = np.array([True, True])
    measure = np.array([1.0, 3.0])
    assert measure_weighted_mean(values, mask, measure) == pytest.approx(2.5)
    assert np.mean(values) == pytest.approx(2.0)


def test_weighting_contact_uses_local_boundary_cell_half_width() -> None:
    x_edges = np.array([-5.0, 5.0]) * 1.0e-6
    y_edges = (
        np.array([-12.0, -10.0, -9.0, -8.0, 0.0, 8.0, 9.0, 10.0, 12.0])
        * 1.0e-6
    )
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    flake = np.zeros((1, y.size), bool)
    flake[:, 1:7] = True
    psi, _, _, diagnostics = solve_weighting_potential(
        x_edges,
        y_edges,
        flake,
    )
    expected = (y[1:7] + 10.0e-6) / (20.0e-6)
    assert psi[0, 1:7] == pytest.approx(expected, rel=2e-13, abs=2e-14)
    assert diagnostics["top_contact_half_width_m"]["minimum"] == pytest.approx(
        0.5e-6
    )
    assert diagnostics["bottom_contact_half_width_m"][
        "minimum"
    ] == pytest.approx(0.5e-6)


def test_straight_edge_retains_all_five_gradient_observables() -> None:
    x_edges = np.linspace(-2.5, 2.5, 6)
    y_edges = np.linspace(-2.5, 2.5, 6)
    z_edges = np.array([-1.0, 0.0])
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    flake_xy = yy <= xx
    flake = flake_xy[:, :, None]
    temperature = (2.0 * xx + 3.0 * yy)[:, :, None]
    shape = temperature.shape
    geometry = Geometry(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        material_id=np.zeros(shape, np.uint8),
        flake_mask=flake,
        kappa_W_mK=np.ones((*shape, 3)),
        interface_resistance_m2K_W={
            "x": np.zeros((shape[0] - 1, shape[1], shape[2])),
            "y": np.zeros((shape[0], shape[1] - 1, shape[2])),
            "z": np.zeros((shape[0], shape[1], shape[2] - 1)),
        },
    )
    metrics, fields = straight_edge_temperature_metrics(
        temperature,
        geometry,
        edge_window_um=1.0e5,
    )
    assert metrics["max_abs_grad_T_x_K_m"] == pytest.approx(2.0)
    assert metrics["max_abs_grad_T_y_K_m"] == pytest.approx(3.0)
    assert metrics["max_inplane_gradient_K_m"] == pytest.approx(np.sqrt(13.0))
    assert metrics["max_abs_edge_normal_gradient_K_m"] == pytest.approx(
        1.0 / np.sqrt(2.0)
    )
    assert metrics["max_abs_edge_tangent_gradient_K_m"] == pytest.approx(
        5.0 / np.sqrt(2.0)
    )
    assert "grad_T_tangent_K_m" in fields
