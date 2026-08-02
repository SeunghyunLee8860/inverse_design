import json
from pathlib import Path

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photothermal_pte.validation.paper_ir_sanity.coordinate_plot import (
    cell_field,
    dual_edges_from_centers,
    strict_centered_xy_mask,
)

from photothermal_pte.validation.paper_ir_sanity.run_analytic_q_remap_control import (
    analytic_q_on_edges,
)
from photothermal_pte.validation.paper_ir_sanity.run_device_a_explicit_thermal_pte import (
    Geometry,
    measure_weighted_mean,
    pte_current,
    pte_current_internal_face_bilinear,
    pte_current_strict_centered,
    solve_weighting_potential,
    straight_edge_temperature_metrics,
    strict_centered_cell_gradient,
)


def test_coordinate_plot_preserves_nonuniform_cell_edges() -> None:
    x_edges = np.asarray([-3.0, -2.0, 0.0, 4.0])
    y_edges = np.asarray([-5.0, -1.0, 2.0])
    values = np.arange(6.0).reshape(3, 2)
    figure, axis = plt.subplots()
    image = cell_field(axis, x_edges, y_edges, values)
    coordinates = image.get_coordinates()
    plt.close(figure)
    assert np.array_equal(coordinates[0, :, 0], x_edges)
    assert np.array_equal(coordinates[:, 0, 1], y_edges)


def test_dual_edges_from_centers_has_half_cell_outer_boundaries() -> None:
    centers = np.asarray([-2.0, -1.0, 2.0, 6.0])
    assert np.array_equal(
        dual_edges_from_centers(centers),
        np.asarray([-2.5, -1.5, 0.5, 4.0, 8.0]),
    )


def test_strict_centered_gradient_masks_any_missing_xy_neighbour() -> None:
    x = np.arange(5.0)
    y = np.arange(5.0)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    values = 2.0 * xx + 3.0 * yy
    mask = np.ones((5, 5), bool)
    mask[2, 3] = False
    gx, gy, valid = strict_centered_cell_gradient(values, mask, x, y)
    assert np.array_equal(valid, strict_centered_xy_mask(mask))
    assert not valid[2, 2]
    assert not valid[2, 4]
    assert not valid[1, 3]
    assert not valid[3, 3]
    assert np.all(np.isnan(gx[~valid]))
    assert np.all(np.isnan(gy[~valid]))
    assert np.allclose(gx[valid], 2.0)
    assert np.allclose(gy[valid], 3.0)


def test_physical_field_plotters_do_not_use_imshow() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "validation"
        / "paper_ir_sanity"
    )
    offenders = []
    for path in directory.glob("*.py"):
        if path.name == "digitize_device_a_geometry.py":
            continue
        if ".imshow(" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == []


def test_pte_volume_and_thickness_integrated_area_forms_are_equivalent() -> None:
    x_edges = np.linspace(-2.0e-6, 2.0e-6, 5)
    y_edges = np.linspace(-2.0e-6, 2.0e-6, 5)
    z_edges = np.array([-130.0e-9, -65.0e-9, 0.0])
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    temperature = 2.0e5 * xx - 3.0e5 * yy + 1.0e5 * zz
    shape = temperature.shape
    flake = np.ones(shape, bool)
    geometry = Geometry(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        material_id=np.full(shape, 3, np.uint8),
        flake_mask=flake,
        kappa_W_mK=np.ones((*shape, 3)),
        interface_resistance_m2K_W={
            "x": np.zeros((shape[0] - 1, shape[1], shape[2])),
            "y": np.zeros((shape[0], shape[1] - 1, shape[2])),
            "z": np.zeros((shape[0], shape[1], shape[2] - 1)),
        },
    )
    current, fields = pte_current(
        temperature,
        geometry,
        np.full(shape[:2], 2.0e4),
        np.full(shape[:2], -1.0e4),
    )
    assert current == pytest.approx(
        float(fields["PTE_current_thickness_integrated_area_A"][0]),
        rel=2e-14,
    )
    assert fields["PTE_volume_area_equivalence_relative_error"][0] < 2e-14


def test_internal_face_pte_pairs_temperature_and_weighting_on_same_faces() -> None:
    x_edges = np.asarray([-2.0, -1.0, 0.5, 3.0]) * 1.0e-6
    y_edges = np.asarray([-3.0, -0.5, 1.0, 4.0]) * 1.0e-6
    z_edges = np.asarray([-130.0, -60.0, 0.0]) * 1.0e-9
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    ax, ay = 2.0e5, -3.0e5
    bx, by = -4.0e4, 5.0e4
    temperature = ax * xx + ay * yy + 7.0e4 * zz
    psi = bx * xx[:, :, 0] + by * yy[:, :, 0]
    shape = temperature.shape
    geometry = Geometry(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        material_id=np.full(shape, 3, np.uint8),
        flake_mask=np.ones(shape, bool),
        kappa_W_mK=np.ones((*shape, 3)),
        interface_resistance_m2K_W={
            "x": np.zeros((shape[0] - 1, shape[1], shape[2])),
            "y": np.zeros((shape[0], shape[1] - 1, shape[2])),
            "z": np.zeros((shape[0], shape[1], shape[2] - 1)),
        },
    )
    current, fields = pte_current_internal_face_bilinear(
        temperature, geometry, psi
    )
    thickness = z_edges[-1] - z_edges[0]
    x_center_span = x[-1] - x[0]
    y_center_span = y[-1] - y[0]
    full_x_width = x_edges[-1] - x_edges[0]
    full_y_width = y_edges[-1] - y_edges[0]
    expected_x = (
        -1.10e5 * 27.0e-6 * ax * bx
        * x_center_span * full_y_width * thickness
    )
    expected_y = (
        -4.91e5 * -6.0e-6 * ay * by
        * full_x_width * y_center_span * thickness
    )
    assert fields["current_x_faces_A"] == pytest.approx(expected_x, rel=2e-14)
    assert fields["current_y_faces_A"] == pytest.approx(expected_y, rel=2e-14)
    assert current == pytest.approx(expected_x + expected_y, rel=2e-14)


def test_strict_pte_masks_boundary_cells_and_any_missing_neighbour() -> None:
    edges = np.linspace(-2.5e-6, 2.5e-6, 6)
    z_edges = np.asarray([-130.0e-9, 0.0])
    x = 0.5 * (edges[:-1] + edges[1:])
    xx, yy = np.meshgrid(x, x, indexing="ij")
    temperature = (2.0e5 * xx + 3.0e5 * yy)[:, :, None]
    psi = -4.0e4 * xx + 5.0e4 * yy
    shape = temperature.shape
    flake = np.ones(shape, bool)
    flake[2, 3, 0] = False
    geometry = Geometry(
        x_edges_m=edges,
        y_edges_m=edges,
        z_edges_m=z_edges,
        material_id=np.where(flake, 3, 0).astype(np.uint8),
        flake_mask=flake,
        kappa_W_mK=np.ones((*shape, 3)),
        interface_resistance_m2K_W={
            "x": np.zeros((shape[0] - 1, shape[1], shape[2])),
            "y": np.zeros((shape[0], shape[1] - 1, shape[2])),
            "z": np.zeros((shape[0], shape[1], 0)),
        },
    )
    _, fields = pte_current_strict_centered(temperature, geometry, psi)
    expected_valid = strict_centered_xy_mask(flake[:, :, 0])
    assert np.array_equal(fields["valid_xy_mask"], expected_valid)
    contribution = fields["cell_contribution_A"][:, :, 0]
    assert np.count_nonzero(contribution[~expected_valid]) == 0


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


def test_analytic_q_uses_exact_half_measure_on_diagonal_cells() -> None:
    edges_xy = np.array([-1.0e-6, 0.0, 1.0e-6])
    edges_z = np.array([-130.0e-9, 0.0])
    q = analytic_q_on_edges((edges_xy, edges_xy, edges_z), "a")
    assert q[0, 1, 0] == pytest.approx(0.0)
    assert q[1, 0, 0] > 0.0
    assert q[0, 0, 0] == pytest.approx(0.5 * q[1, 0, 0], rel=1e-13)
    assert q[1, 1, 0] == pytest.approx(0.5 * q[1, 0, 0], rel=1e-13)


def test_offline_paper_ir_summary_is_fail_closed_after_planar_audit() -> None:
    repository = Path(__file__).resolve().parents[3]
    summary_path = (
        repository
        / "photothermal_pte"
        / "reports"
        / "paper_ir_offline_q_thermal_controls"
        / "paper_ir_offline_controls_summary.json"
    )
    summary = json.loads(summary_path.read_text())
    assert (
        summary["validated_subgates"]["diagnostic_Q_observable_convergence"]
        == "VALIDATED_DIAGNOSTIC_Q_OBSERVABLE_CONVERGENCE"
    )
    assert not summary["unresolved_or_blocked"]["auto_shutoff"]["passed"]
    assert (
        summary["unresolved_or_blocked"]["three_source_decomposition"]
        == "BLOCKED_PLANAR_STACK_Q_ARTIFACT_UNAVAILABLE"
    )
    assert summary["execution_scope"]["new_FDTD_run"] is False
    assert summary["execution_scope"]["PTE_run"] is False
