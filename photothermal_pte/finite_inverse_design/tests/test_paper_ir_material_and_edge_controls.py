from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from photothermal_pte.validation.paper_ir_sanity import (
    audit_paper_ir_checkpoint_failure as checkpoint_audit,
)
from photothermal_pte.validation.paper_ir_sanity import (
    audit_straight_edge_robust_gradient as robust,
)
from photothermal_pte.validation.paper_ir_sanity import (
    run_straight_edge_analytic_source_controls as controls,
)
from photothermal_pte.validation.paper_ir_sanity import (
    run_lumerical_device_a_ir_q as device_a_optical,
)
from photothermal_pte.validation.paper_ir_sanity import (
    run_device_a_explicit_thermal_pte as device_a_thermal,
)
from photothermal_pte.validation.paper_ir_sanity import (
    plot_device_a_final_beam_center_plan as nine_position_plan,
)


def test_paper_ir_c_table_is_exact_b_closure() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "bundle"
        / "perm_data.txt"
    )
    data = np.loadtxt(path)
    assert data.shape[1] == 7
    assert np.array_equal(data[:, 3:5], data[:, 5:7])


def test_kitamura_2007_sio2_exact_11um_value_and_passivity() -> None:
    epsilon = complex(
        device_a_optical.kitamura_2007_sio2_epsilon(11.0e-6)
    )
    assert epsilon.real == pytest.approx(4.051707451517633, rel=1e-14)
    assert epsilon.imag == pytest.approx(0.6568047491827695, rel=1e-14)
    refractive_index = np.sqrt(epsilon)
    assert refractive_index.real == pytest.approx(
        2.0194436826147366, rel=1e-14
    )
    assert refractive_index.imag == pytest.approx(
        0.16262021932999673, rel=1e-14
    )
    band = device_a_optical.kitamura_2007_sio2_epsilon(
        np.linspace(7.0e-6, 13.0e-6, 1201)
    )
    assert np.all(np.isfinite(band))
    assert np.all(np.imag(band) >= 0.0)


def test_finite_numerical_pulse_is_frequency_centered_at_11um() -> None:
    center_frequency = 0.5 * device_a_optical.C0 * (
        1.0 / device_a_optical.SOURCE_CENTERED_START_M
        + 1.0 / device_a_optical.SOURCE_CENTERED_STOP_M
    )
    center_wavelength = device_a_optical.C0 / center_frequency
    assert center_wavelength == pytest.approx(
        device_a_optical.WAVELENGTH_M,
        rel=0.0,
        abs=1.0e-18,
    )
    assert device_a_optical.SOURCE_CENTERED_STOP_M <= 13.2e-6


def test_device_a_beam_override_uses_fixed_lumerical_device_frame(
    tmp_path: Path,
) -> None:
    payload = {
        "flake_vertices_code_um": [[-2, -2], [2, -2], [2, 2], [-2, 2]],
        "top_metal_polygon_code_um": [[-3, 3], [3, 3], [3, 4], [-3, 4]],
        "bottom_metal_polygon_code_um": [
            [-3, -4],
            [3, -4],
            [3, -3],
            [-3, -3],
        ],
        "pre_registered_beam_center_code_um": [-8, 0],
    }
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(payload))
    contract = device_a_optical.load_digitized_device_a_contract(
        path,
        domain_um=64.0,
        source_span_um=50.0,
        beam_center_code_um_override=(0.0, 3.0),
    )
    assert contract["beam_center_digitized_override_applied"]
    assert np.array_equal(
        contract["beam_center_digitized_original_um"], [-8.0, 0.0]
    )
    assert np.array_equal(contract["beam_center_digitized_um"], [0.0, 3.0])
    assert np.allclose(contract["beam_center_simulation_um"], [0.0, 0.0])
    assert np.allclose(
        np.asarray(contract["flake_vertices_simulation_um"]),
        np.asarray(payload["flake_vertices_code_um"]) + [0.0, -3.0],
    )
    assert np.allclose(contract["FDTD_lateral_center_simulation_um"], [0.0, 0.0])
    assert contract["minimum_lateral_PML_clearance_um"] == {
        "x": 7.0,
        "y": 7.0,
    }
    canonical = contract["canonical_lumerical_frame"]
    assert canonical["name"] == "DEVICE_A_FIXED_LUMERICAL_X_B_Y_A"
    assert canonical["origin_in_digitized_code_um"] == [0.0, 3.0]
    assert canonical["code_to_canonical_lumerical_shift_um"] == [0.0, -3.0]
    assert np.allclose(canonical["beam_center_um"], [0.0, 0.0])
    assert np.allclose(
        canonical["flake_vertices_um"],
        np.asarray(payload["flake_vertices_code_um"]) + [0.0, -3.0],
    )
    assert np.allclose(
        np.asarray(contract["flake_vertices_simulation_um"])
        + canonical["solver_to_canonical_lumerical_translation_um"],
        canonical["flake_vertices_um"],
    )
    edge_contract = device_a_optical.load_digitized_device_a_contract(
        path,
        domain_um=64.0,
        source_span_um=50.0,
    )
    assert np.array_equal(
        edge_contract["flake_vertices_simulation_um"],
        contract["flake_vertices_simulation_um"],
    )
    assert np.allclose(edge_contract["beam_center_simulation_um"], [-8.0, -3.0])
    assert np.allclose(
        edge_contract["FDTD_lateral_center_simulation_um"], [-8.0, -3.0]
    )
    assert edge_contract["canonical_lumerical_frame"][
        "solver_to_canonical_lumerical_translation_um"
    ] == [0.0, 0.0]


def test_two_terminal_resistance_audit_recovers_rectangle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    width = 2.0e-6
    length = 4.0e-6
    thickness = 130.0e-9
    x_edges = np.linspace(-0.5 * width, 0.5 * width, 41)
    y_edges = np.linspace(-0.5 * length, 0.5 * length, 81)
    flake = np.ones((x_edges.size - 1, y_edges.size - 1), bool)
    full_contact = np.asarray([[-2.0, 0.0], [2.0, 0.0]])
    monkeypatch.setattr(device_a_thermal, "TOP_CONTACT_SEGMENT_UM", full_contact)
    monkeypatch.setattr(device_a_thermal, "BOTTOM_CONTACT_SEGMENT_UM", full_contact)
    result = device_a_thermal.audit_two_terminal_resistance(
        x_edges,
        y_edges,
        flake,
        thickness_m=thickness,
        measured_resistance_ohm=1.0,
    )
    expected = length / (
        device_a_thermal.SIGMA_LAB_S_M[1] * width * thickness
    )
    assert result["predicted_resistance_ohm"] == pytest.approx(
        expected, rel=1e-12
    )
    assert result["terminal_current_balance_relative_error"] < 1e-11
    assert result["linear_residual_relative"] < 1e-12


def test_named_tairte4_sio2_interface_scenarios_change_only_target_z_faces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        device_a_thermal,
        "FLAKE_VERTICES_UM",
        np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]),
    )
    common = dict(
        domain_m=6.0e-6,
        si_depth_m=1.0e-6,
        core_step_m=0.5e-6,
        flake_dz_m=10.0e-9,
    )
    grown = device_a_thermal.build_geometry(
        **common,
        tairte4_sio2_G_W_m2K=(
            device_a_thermal.G_TAIRTE4_THERMALLY_GROWN_SIO2_W_M2K
        ),
    )
    evaporated = device_a_thermal.build_geometry(
        **common,
        tairte4_sio2_G_W_m2K=(
            device_a_thermal.G_TAIRTE4_EVAPORATED_SIO2_W_M2K
        ),
    )
    lower = grown.material_id[:, :, :-1]
    upper = grown.material_id[:, :, 1:]
    target = ((lower == 2) & (upper == 3)) | ((lower == 3) & (upper == 2))
    assert np.any(target)
    assert np.array_equal(
        grown.interface_resistance_m2K_W["x"],
        evaporated.interface_resistance_m2K_W["x"],
    )
    assert np.array_equal(
        grown.interface_resistance_m2K_W["y"],
        evaporated.interface_resistance_m2K_W["y"],
    )
    difference = (
        grown.interface_resistance_m2K_W["z"]
        != evaporated.interface_resistance_m2K_W["z"]
    )
    assert np.array_equal(difference, target)
    assert np.all(
        grown.interface_resistance_m2K_W["z"][target]
        == 1.0 / device_a_thermal.G_TAIRTE4_THERMALLY_GROWN_SIO2_W_M2K
    )
    assert np.all(
        evaporated.interface_resistance_m2K_W["z"][target]
        == 1.0 / device_a_thermal.G_TAIRTE4_EVAPORATED_SIO2_W_M2K
    )


def test_device_a_nine_position_contract_is_fixed_lumerical_x_b_y_a() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "reports"
        / "paper_ir_device_a_fig3h_registered_scan"
        / "device_a_fig3h_registered_geometry.json"
    )
    contract = nine_position_plan.build_position_contract(
        json.loads(path.read_text())
    )
    assert contract["coordinate_frame"]["axis_mapping"] == "x=b, y=a"
    assert contract["coordinate_frame"]["only_source_is_translated"]
    assert len(contract["cases"]) == 9
    by_label = {case["label"]: case for case in contract["cases"]}
    assert by_label["outside_middle"]["beam_center_lumerical_um"] == [
        -16.5625,
        0.0,
    ]
    assert by_label["inside_middle"]["beam_center_lumerical_um"] == [
        -6.0,
        0.0,
    ]
    for label in ("edge_top", "edge_middle", "edge_bottom"):
        assert by_label[label]["edge_segment_index"] is not None


def test_equal_absorbed_power_control_is_exact_and_analytic_only() -> None:
    edges = np.asarray([-1.0e-6, 0.0, 1.0e-6])
    z_edges = np.linspace(-130e-9, 0.0, 6)
    x = 0.5 * (edges[:-1] + edges[1:])
    y = 0.5 * (edges[:-1] + edges[1:])
    flake_xy = y[None, :] <= x[:, None]
    geometry = SimpleNamespace(
        x_edges_m=edges,
        y_edges_m=edges,
        z_edges_m=z_edges,
        flake_mask=flake_xy[:, :, None]
        & np.ones((1, 1, z_edges.size - 1), bool),
    )
    target = 2.5e-6
    q_a, contract_a = controls.source_for_control(
        geometry,
        control="equal_absorbed_power_shape_control",
        polarization="a",
        equal_power_W=target,
    )
    q_b, contract_b = controls.source_for_control(
        geometry,
        control="equal_absorbed_power_shape_control",
        polarization="b",
        equal_power_W=target,
    )
    assert controls.integrate_volume(
        q_a,
        edges,
        edges,
        z_edges,
    ) == pytest.approx(target, rel=1e-13)
    assert controls.integrate_volume(
        q_b,
        edges,
        edges,
        z_edges,
    ) == pytest.approx(target, rel=1e-13)
    assert not contract_a["raw_Lumerical_Q_modified"]
    assert not contract_b["raw_Lumerical_Q_modified"]


def test_physical_line_quadratic_fit_recovers_linear_gradient() -> None:
    coordinate = np.linspace(-12e-6, 12e-6, 241)
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="ij")
    temperature = 2.0 * xx + 3.0 * yy
    temperature[yy > xx] = np.nan
    fitted = robust.quadratic_edge_fit(
        coordinate,
        coordinate,
        temperature,
        robust.N_BANDS_UM["primary"],
    )
    expected_dn = 1.0 / np.sqrt(2.0)
    assert np.allclose(fitted["dT_dx_K_m"], 2.0, atol=1e-9)
    assert np.allclose(fitted["dT_dn_K_m"], expected_dn, atol=1e-9)
    assert np.max(fitted["fit_relative_residual"]) < 1e-10


def test_checkpoint_audit_relative_change_matches_existing_symmetric_metric() -> None:
    assert checkpoint_audit.rel_change(80.0, 100.0) == pytest.approx(0.2)
    assert checkpoint_audit.rel_change(100.0, 80.0) == pytest.approx(0.2)


def test_checkpoint_audit_coordinate_summary_uses_literal_steps() -> None:
    summary = checkpoint_audit.coordinate_summary(
        np.asarray([-2.0e-6, -1.0e-6, 1.0e-6, 4.0e-6])
    )
    assert summary["count"] == 4
    assert summary["minimum_step_m"] == pytest.approx(1.0e-6)
    assert summary["median_step_m"] == pytest.approx(2.0e-6)
    assert summary["maximum_step_m"] == pytest.approx(3.0e-6)
