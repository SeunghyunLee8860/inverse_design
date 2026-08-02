import json
from pathlib import Path
from types import SimpleNamespace

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
from photothermal_pte.validation.paper_ir_sanity.analyze_device_a_current_cause_controls import (
    INCIDENT_POWER_W,
    WAIST_M,
    gaussian_cell_power,
    planar_tmm_coefficients,
    tmm_q_per_incident_intensity,
)
from photothermal_pte.validation.paper_ir_sanity.analyze_device_a_spatial_current_decomposition import (
    distance_to_polygon_boundary,
    distance_to_segment,
    partition_sheet_current,
)
from photothermal_pte.validation.paper_ir_sanity.analyze_device_a_q_current_colocalization import (
    normalized_correlation,
    partition,
)

from photothermal_pte.validation.paper_ir_sanity.run_analytic_q_remap_control import (
    analytic_q_on_edges,
)
from photothermal_pte.validation.paper_ir_sanity.run_device_a_explicit_thermal_pte import (
    Geometry,
    load_optical_coordinate_frame,
    measure_weighted_mean,
    pte_current,
    pte_current_internal_face_bilinear,
    pte_current_strict_centered,
    solve_weighting_potential,
    straight_edge_temperature_metrics,
    strict_centered_cell_gradient,
)
from photothermal_pte.validation.paper_ir_sanity.register_device_a_fig3h_approx import (
    affine_pixel_to_device_um,
    source_device_envelope,
)
from photothermal_pte.validation.paper_ir_sanity.run_lumerical_device_a_ir_q import (
    geometry_scenario_label,
    maximum_absolute_lateral_flux_fraction,
    validate_device_a_incident_reference_contract,
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


def test_fig3h_affine_registration_preserves_axis_and_scale_contract() -> None:
    mapped = affine_pixel_to_device_um(
        np.asarray([[402.5, 485.5], [413.7, 474.3]]),
        panel_center_px=np.asarray([402.5, 485.5]),
        flake_center_um=np.asarray([0.0, 0.0]),
        pixels_per_um=11.2,
    )
    assert mapped[0] == pytest.approx([0.0, 0.0])
    assert mapped[1] == pytest.approx([1.0, 1.0])


def test_registered_source_requires_larger_domain_if_span_is_preserved() -> None:
    beam = np.asarray([-16.5625, 3.0])
    top = np.asarray([[-17.8453, 11.768], [20.221, 17.956]])
    bottom = np.asarray([[-17.8453, -15.249], [20.221, -11.657]])
    old = source_device_envelope(
        beam_um=beam,
        source_span_um=50.0,
        domain_um=60.0,
        top_metal_um=top,
        bottom_metal_um=bottom,
    )
    expanded = source_device_envelope(
        beam_um=beam,
        source_span_um=50.0,
        domain_um=64.0,
        top_metal_um=top,
        bottom_metal_um=bottom,
    )
    assert not old["passes_existing_loader_clearance_gate"]
    assert expanded["passes_existing_loader_clearance_gate"]
    assert old["minimum_PML_clearance_um"]["x"] < 0.0
    assert expanded["minimum_PML_clearance_um"]["x"] > 1.0


def test_thermal_runner_reads_actual_64um_optical_frame(tmp_path: Path) -> None:
    case_dir = tmp_path / "optical_case"
    case_dir.mkdir()
    payload = {"domain_um": 64.0, "source_span_um": 50.0}
    (case_dir / "case_result.json").write_text(json.dumps(payload))
    path, result, domain_um, source_span_um = load_optical_coordinate_frame(case_dir)
    assert path == case_dir / "case_result.json"
    assert result == payload
    assert domain_um == 64.0
    assert source_span_um == 50.0


def test_thermal_runner_fails_closed_without_explicit_optical_frame(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "optical_case"
    case_dir.mkdir()
    (case_dir / "case_result.json").write_text(json.dumps({"domain_um": 64.0}))
    with pytest.raises(ValueError, match="source_span_um"):
        load_optical_coordinate_frame(case_dir)


def test_device_a_metadata_does_not_claim_edge_free_planar_geometry() -> None:
    assert "physical edges" in geometry_scenario_label("device-a-polygon")
    assert "edge-free" not in geometry_scenario_label("device-a-polygon")
    assert "edge-free planar" in geometry_scenario_label("planar-stack")


def test_outer_lateral_flux_gate_uses_all_four_signed_faces() -> None:
    box = {
        "faces": {
            "x_min": {"normalized_signed_axis_flux": -2.0e-7},
            "x_max": {"normalized_signed_axis_flux": 8.0e-7},
            "y_min": {"normalized_signed_axis_flux": -5.0e-7},
            "y_max": {"normalized_signed_axis_flux": 3.0e-7},
        }
    }
    assert maximum_absolute_lateral_flux_fraction(box) == 8.0e-7


def device_a_reference_contract_fixture() -> tuple[dict, SimpleNamespace]:
    args = SimpleNamespace(
        domain_um=64.0,
        pml_layers=24,
        flake_dz_nm=10.0,
        source_span_um=50.0,
        waist_um=8.75,
        source_object_waist_um=8.610602974768,
        beam_x_um=-5.89174723756906,
        beam_y_um=2.0,
        source_start_m=9.428571428571428e-6,
        source_stop_m=13.2e-6,
        substrate_optical_model="paper-kitamura-palik-nk-11um",
    )
    source = {
        "beam_center_m": [-5.89174723756906e-6, 2.0e-6],
        "source_span_m": 50.0e-6,
        "physical_target_waist_radius_m": 8.75e-6,
        "Lumerical_source_object_waist_radius_m": 8.610602974768e-6,
        "numerical_pulse_band_m": [9.428571428571428e-6, 13.2e-6],
    }
    payload = {
        "domain_um": 64.0,
        "pml_layers": 24,
        "flake_dz_nm": 10.0,
        "pre_run_contract": {
            "geometry": {
                "source": source,
                "substrate_optical_contract": {
                    "model": "paper-kitamura-palik-nk-11um"
                },
            }
        },
    }
    return payload, args


def test_device_a_incident_reference_requires_same_translated_beam() -> None:
    payload, args = device_a_reference_contract_fixture()
    audit = validate_device_a_incident_reference_contract(payload, args)
    assert audit["passed"]
    payload["pre_run_contract"]["geometry"]["source"]["beam_center_m"][1] = 0.0
    with pytest.raises(RuntimeError, match="active scan position"):
        validate_device_a_incident_reference_contract(payload, args)


def test_device_a_incident_reference_requires_same_domain_pml_and_dz() -> None:
    payload, args = device_a_reference_contract_fixture()
    for key, wrong in (("domain_um", 60.0), ("pml_layers", 32), ("flake_dz_nm", 5.0)):
        altered = json.loads(json.dumps(payload))
        altered[key] = wrong
        with pytest.raises(RuntimeError, match="numerical contract"):
            validate_device_a_incident_reference_contract(altered, args)


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


def test_planar_sio2_tmm_volume_loss_matches_flux_absorptance() -> None:
    thickness_m = 285.0e-9
    nodes, weights = np.polynomial.legendre.leggauss(64)
    depth_m = 0.5 * thickness_m * (nodes + 1.0)
    integrated_absorptance = 0.5 * thickness_m * np.sum(
        weights * tmm_q_per_incident_intensity(depth_m)
    )
    expected = planar_tmm_coefficients()["SiO2_absorptance_from_flux"]
    assert integrated_absorptance == pytest.approx(expected, rel=2.0e-12)


def test_planar_sio2_gaussian_cell_integration_preserves_wide_plane_power() -> None:
    edges_m = np.linspace(-30.0e-6, 30.0e-6, 601)
    x_fraction = gaussian_cell_power(edges_m, center_m=-5.891747237569059e-6)
    y_fraction = gaussian_cell_power(edges_m, center_m=0.0)
    captured_power = INCIDENT_POWER_W * np.sum(x_fraction) * np.sum(y_fraction)
    assert captured_power / INCIDENT_POWER_W == pytest.approx(
        0.9999999820978916, rel=2.0e-12
    )
    assert WAIST_M == pytest.approx(8.75e-6)


def test_device_a_current_cause_summary_keeps_planar_oxide_diagnostic_only() -> None:
    repository = Path(__file__).resolve().parents[3]
    summary_path = (
        repository
        / "photothermal_pte"
        / "reports"
        / "paper_ir_device_a_current_cause_controls"
        / "device_a_current_cause_controls_summary.json"
    )
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "COMPLETED_DEVICE_A_CURRENT_CAUSE_CONTROLS"
    assert summary["interpretation_limits"]["planar_SiO2_is_not_production"]
    ratios = summary["sampled_maximum_ratios"]
    assert ratios["Ta-only uniform 45deg weighting"]["abs_b_over_abs_a"] < ratios[
        "Ta-only actual weighting"
    ]["abs_b_over_abs_a"]
    assert ratios["Ta+planar-SiO2 actual weighting"]["abs_b_over_abs_a"] < 1.0


def test_spatial_decomposition_distances_are_geometric_not_grid_ordered() -> None:
    xx, yy = np.meshgrid(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]), indexing="ij")
    diagonal = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    distance = distance_to_segment(xx, yy, diagonal)
    assert distance[0, 0] == pytest.approx(0.0)
    assert distance[1, 1] == pytest.approx(0.0)
    assert distance[0, 1] == pytest.approx(1.0 / np.sqrt(2.0))
    square = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    center_distance = distance_to_polygon_boundary(
        np.asarray([[0.5]]), np.asarray([[0.5]]), square
    )
    assert center_distance[0, 0] == pytest.approx(0.5)


def test_spatial_current_partition_uses_literal_cell_area() -> None:
    sheet = np.asarray([[1.0, -2.0], [3.0, 4.0]])
    area = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    masks = {
        "left": np.asarray([[True, True], [False, False]]),
        "right": np.asarray([[False, False], [True, True]]),
    }
    partitioned = partition_sheet_current(sheet, area, masks)
    assert partitioned["left"] == pytest.approx(-3.0)
    assert partitioned["right"] == pytest.approx(25.0)
    assert sum(partitioned.values()) == pytest.approx(np.sum(sheet * area))


def test_device_a_spatial_current_decomposition_closes_and_localizes_edge() -> None:
    repository = Path(__file__).resolve().parents[3]
    summary_path = (
        repository
        / "photothermal_pte"
        / "reports"
        / "paper_ir_device_a_spatial_current_decomposition"
        / "device_a_spatial_current_decomposition_summary.json"
    )
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "COMPLETED_DEVICE_A_SPATIAL_CURRENT_DECOMPOSITION"
    assert summary["maximum_decomposition_closure_relative_error"] < 1.0e-12
    assert all(summary["numerical_gates"].values())
    for row in summary["same_position_a_minus_b"]:
        assert row["a_minus_b_total_current_A"] > 0.0
        assert row["dominant_absolute_current_difference_region"] == "free_edge_within_1um"
        assert row["device_region_a_minus_b_A"]["free_edge_within_1um"] > 0.0
        assert row["device_region_a_minus_b_A"]["flake_interior"] < 0.0


def test_q_current_colocalization_helpers_preserve_signed_values() -> None:
    values = np.asarray([[1.0, -2.0], [3.0, 4.0]])
    masks = {
        "diagonal": np.asarray([[True, False], [False, True]]),
        "off_diagonal": np.asarray([[False, True], [True, False]]),
    }
    result = partition(values, masks)
    assert result["diagonal"] == pytest.approx(5.0)
    assert result["off_diagonal"] == pytest.approx(1.0)
    assert normalized_correlation(values, 2.0 * values + 3.0, np.ones_like(values, bool)) == pytest.approx(1.0)


def test_device_a_q_current_colocalization_identifies_equal_power_edge_enrichment() -> None:
    repository = Path(__file__).resolve().parents[3]
    summary_path = (
        repository
        / "photothermal_pte"
        / "reports"
        / "paper_ir_device_a_q_current_colocalization"
        / "device_a_q_current_colocalization_summary.json"
    )
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "COMPLETED_DEVICE_A_Q_CURRENT_COLOCALIZATION"
    assert all(summary["numerical_gates"].values())
    for row in summary["same_position_b_over_a"]:
        assert row["total_power_b_over_a"] > 1.0
        assert row["total_current_b_over_a"] < 1.0
        assert row["current_efficiency_b_over_a"] < 1.0
        assert row["free_edge_power_fraction_a_over_b"] > 1.0
        assert row["nearest_0p25um_power_fraction_a_over_b"] > 2.5


def test_device_a_edge_source_thermal_split_is_causal_and_closes() -> None:
    repository = Path(__file__).resolve().parents[3]
    summary_path = (
        repository
        / "photothermal_pte"
        / "reports"
        / "paper_ir_device_a_edge_source_thermal_superposition"
        / "device_a_edge_source_thermal_superposition_summary.json"
    )
    summary = json.loads(summary_path.read_text())
    assert (
        summary["status"]
        == "VALIDATED_DEVICE_A_FREE_EDGE_Q_CAUSAL_CURRENT_SPLIT"
    )
    assert all(summary["numerical_gates"].values())
    for row in summary["same_position_a_minus_b"]:
        assert row["full_a_minus_b_current_A"] > 0.0
        assert row["free_edge_source_a_minus_b_current_A"] > 0.0
        assert row["free_edge_fraction_of_full_a_minus_b"] > 0.9
    near_edge = summary["same_position_a_minus_b"][:2]
    assert all(
        row["free_edge_fraction_of_full_a_minus_b"] > 1.0
        and row["remainder_source_a_minus_b_current_A"] < 0.0
        for row in near_edge
    )
