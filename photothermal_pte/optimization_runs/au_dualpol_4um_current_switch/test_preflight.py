from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import CONTRACT
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.dfm import (
    MAPPING,
    exact_500nm_audit,
    smooth_500nm_constraints,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.objective import (
    epigraph_constraints,
    smooth_minimum,
    useful_currents,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    AU_MATERIAL_FRACTION_EXPONENT,
    AU_MATERIAL_FRACTION_LAW,
    au_material_fraction,
    d_au_material_fraction_drho,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.robust_contract import (
    ROBUST_ETAS,
    constraint_labels,
    current_constraint_keys,
    gray_constraint_keys,
    grayness,
    grayness_cotangent,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.production_readiness import (
    DEVICE_STATUS,
    GRADIENT_STATUS,
    MESH_STATUS,
    REQUIRED_DEVICE_CONFIRMATIONS,
    REQUIRED_MESH_COVERAGE,
    readiness_audit,
    sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_4um_model import (
    ABSORPTION_LOSS_BASIS,
    CLOSED_SURFACE_PHASOR_APODIZATION,
    CLOSED_SURFACE_PHASOR_WINDOW,
    LAYOUT,
    MAX_IGNORED_SUBSTRATE_EPSILON_IMAG,
    _lossless_uniform_permittivity,
    grid_edges_sha256,
    realized_discrete_susceptibility,
    source_calibration_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    load_current_source_calibration,
    require_material_fraction,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.mesh_variants import (
    FULL_DOMAIN_Z,
    PARTIAL_MATERIAL_Z,
    variant_edges,
    variant_layout,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    N_DESIGN,
    N_TA,
    SEEBECK_TA_XY_V_K,
    SIGMA_TA_XY_S_M,
    STEP_M,
    TA_THICKNESS_M,
    current_integrand,
    electrical_load,
    ta_id,
    temperature_pullback,
)


def test_contract_geometry_and_source_boundary() -> None:
    assert CONTRACT.design_shape == (80, 80)
    assert CONTRACT.axis_x == "b" and CONTRACT.axis_y == "a"
    assert CONTRACT.flake_boundary_intensity_fraction < 5.0e-4


def test_all_physics_share_linear_au_material_fraction() -> None:
    rho = np.asarray((0.0, 0.25, 0.5, 1.0))
    assert AU_MATERIAL_FRACTION_LAW == "shared_linear_projected_density"
    assert AU_MATERIAL_FRACTION_EXPONENT == 1.0
    assert CONTRACT.au_material_fraction_law == AU_MATERIAL_FRACTION_LAW
    assert CONTRACT.au_material_fraction_exponent == AU_MATERIAL_FRACTION_EXPONENT
    assert np.array_equal(au_material_fraction(rho), rho)
    assert np.array_equal(d_au_material_fraction_drho(rho), np.ones_like(rho))


def test_source_calibration_is_bound_to_exact_grid_and_time_contract() -> None:
    calibration = source_calibration_contract()
    assert calibration["grid_edges_sha256"] == grid_edges_sha256()
    assert calibration["total_periods"] == 16
    assert calibration["phasor_window_periods"] == 4
    assert calibration["polarization_vectors"]["Ea"] == [0.0, 1.0, 0.0]
    assert calibration["polarization_vectors"]["Eb"] == [1.0, 0.0, 0.0]
    assert LAYOUT.pml_cells_xy == LAYOUT.pml_cells_z == 8


def test_closed_surface_phasor_uses_safe_rectangular_window() -> None:
    assert CLOSED_SURFACE_PHASOR_WINDOW == "rectangular_switch_only"
    assert CLOSED_SURFACE_PHASOR_APODIZATION is None


def test_heat_uses_realized_float32_ade_loss() -> None:
    assert ABSORPTION_LOSS_BASIS == (
        "realized_float32_discrete_ADE_susceptibility"
    )
    value = realized_discrete_susceptibility(
        (1.9996999118283947, -0.9996999118283949, 0.0032755750868792505),
        2.0 * np.pi * 299_792_458.0 / 4.0e-6,
        4.166939126386172e-18,
    )
    assert np.isclose(value.real, -826.3737182617188, rtol=1e-7)
    assert np.isclose(value.imag, 125.59490203857422, rtol=1e-7)


def test_full_z_mesh_factor_one_is_exact_baseline_and_z_pml_is_independent() -> None:
    baseline = grid_edges_sha256()
    factor_one = variant_edges(1, FULL_DOMAIN_Z)
    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.mesh_variants import (
        edges_sha256,
    )

    assert edges_sha256(factor_one) == baseline
    full_two = variant_layout(2, FULL_DOMAIN_Z)
    partial_two = variant_layout(2, PARTIAL_MATERIAL_Z)
    assert full_two.pml_cells_xy == partial_two.pml_cells_xy == 8
    assert full_two.pml_cells_z == 16
    assert partial_two.pml_cells_z == 8
    assert full_two.source_z_start == 2 * LAYOUT.source_z_start


def test_lossy_substrate_cannot_be_silently_replaced_by_real_epsilon() -> None:
    assert _lossless_uniform_permittivity(2.0 + 0.0j, "test") == 2.0
    with np.testing.assert_raises(RuntimeError):
        _lossless_uniform_permittivity(
            2.0 + 2.0 * MAX_IGNORED_SUBSTRATE_EPSILON_IMAG * 1j, "test"
        )


def test_historical_validation_without_material_law_fails_closed() -> None:
    with np.testing.assert_raises(RuntimeError):
        require_material_fraction({"status": "historical"}, "test artifact")


def test_source_calibration_loader_rejects_stale_grid_contract(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "status": "VALIDATED_FDTDX_4UM_SOURCE_POWER_CALIBRATION",
                "source_calibration_contract": {"grid_edges_sha256": "stale"},
                "cases": [],
            }
        ),
        encoding="utf-8",
    )
    with np.testing.assert_raises(RuntimeError):
        load_current_source_calibration(path)


def test_robust_contract_includes_nominal_and_all_grayness_constraints() -> None:
    assert ROBUST_ETAS == (0.35, 0.50, 0.65)
    assert len(current_constraint_keys()) == 6
    assert len(gray_constraint_keys()) == 3
    assert len(constraint_labels()) == 9
    assert "eta_0.50_Ea" in current_constraint_keys()
    assert "eta_0.50_Eb" in current_constraint_keys()
    assert "eta_0.50_grayness" in gray_constraint_keys()
    rng = np.random.default_rng(20260824)
    rho = 0.1 + 0.8 * rng.random((7, 9))
    direction = rng.standard_normal(rho.shape)
    step = 1.0e-6
    finite_difference = (
        grayness(rho + step * direction)
        - grayness(rho - step * direction)
    ) / (2.0 * step)
    analytic = float(np.vdot(grayness_cotangent(rho), direction))
    assert abs(finite_difference - analytic) < 1.0e-9


def test_production_readiness_requires_hash_linked_complete_certificates(
    tmp_path,
) -> None:
    device_path = tmp_path / "device.json"
    calibration_path = tmp_path / "calibration.json"
    mesh_path = tmp_path / "mesh.json"
    gradient_path = tmp_path / "gradient.json"
    material = {
        "law": AU_MATERIAL_FRACTION_LAW,
        "exponent": AU_MATERIAL_FRACTION_EXPONENT,
        "optical_fraction": "au_material_fraction(rho)",
        "thermal_fraction": "au_material_fraction(rho)",
        "electrical_fraction": "au_material_fraction(rho)",
        "gray_density_is_physical_geometry": False,
        "promotion_requires_exact_binary_density": True,
    }
    device = {
        "status": DEVICE_STATUS,
        "confirmations": {
            name: True for name in REQUIRED_DEVICE_CONFIRMATIONS
        },
    }
    device_path.write_text(json.dumps(device), encoding="utf-8")
    calibration = {
        "status": "VALIDATED_FDTDX_4UM_SOURCE_POWER_CALIBRATION",
        "source_calibration_contract": source_calibration_contract(),
        "cases": [
            {
                "polarization": polarization,
                "incident_power_W": 1.0,
                "finite": True,
            }
            for polarization in ("Ea", "Eb")
        ],
        "common_reference_incident_power_W": 1.0,
    }
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    mesh = {
        "status": MESH_STATUS,
        "au_material_fraction": material,
        "coverage": {name: True for name in REQUIRED_MESH_COVERAGE},
        "device_certificate_sha256": sha256(device_path),
        "source_calibration_sha256": sha256(calibration_path),
    }
    mesh_path.write_text(json.dumps(mesh), encoding="utf-8")
    gradient = {
        "status": GRADIENT_STATUS,
        "au_material_fraction": material,
        "mesh_certificate_sha256": sha256(mesh_path),
        "direction_count": 4,
        "maximum_normalized_error": 0.009,
    }
    gradient_path.write_text(json.dumps(gradient), encoding="utf-8")
    assert readiness_audit(
        mesh_path, gradient_path, device_path, calibration_path
    )["ready"]
    gradient["mesh_certificate_sha256"] = "wrong"
    gradient_path.write_text(json.dumps(gradient), encoding="utf-8")
    result = readiness_audit(
        mesh_path, gradient_path, device_path, calibration_path
    )
    assert not result["ready"]
    assert "gradient_uses_mesh_certificate" in result["failed_checks"]


def test_default_physical_device_contract_is_deliberately_blocked() -> None:
    path = Path(__file__).resolve().parent / "physical_device_contract.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED_DEVICE_GEOMETRY_CONFIRMATION_REQUIRED"
    assert not all(payload["confirmations"].values())


def test_historical_partial_z_results_are_marked_stale() -> None:
    path = (
        Path(__file__).resolve().parent
        / "results_4um_dualpol_au_z_mesh_convergence"
        / "Z_MESH_CONVERGENCE_SUMMARY.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "STALE_HISTORICAL_PARTIAL_Z_NOT_CURRENT_CONTRACT"
    assert payload["invalidation"]["do_not_use_as_current_numerical_evidence"]


def test_signed_opposite_current_objective() -> None:
    assert useful_currents(3.0e-9, -4.0e-9) == (3.0e-9, 4.0e-9)
    assert np.all(epigraph_constraints(3e-9, -4e-9, 2e-9) <= 0.0)
    value, derivative = smooth_minimum(3e-9, -4e-9, scale_A=1e-9)
    assert value < 3e-9
    assert derivative[0] > 0.0 and derivative[1] < 0.0


def test_weighting_integral_sign_means_right_to_left_internal_current() -> None:
    # Along x=b, S_b>0 and T increasing toward the unit-potential terminal
    # produces Jloc=-sigma_b*S_b*grad(T), hence a negative collected current.
    temperature = np.broadcast_to(
        np.arange(N_TA, dtype=np.float64)[:, None], (N_TA, N_TA)
    ).copy()
    psi = np.zeros(N_TA * N_TA + N_DESIGN * N_DESIGN, dtype=np.float64)
    for i in range(N_TA):
        psi[[ta_id(i, j) for j in range(N_TA)]] = i / (N_TA - 1)
    expected = (
        -SIGMA_TA_XY_S_M[0]
        * TA_THICKNESS_M
        * SEEBECK_TA_XY_V_K[0]
        * N_TA
    )
    objective = float(electrical_load(temperature) @ psi)
    integrated_map = float(np.sum(current_integrand(temperature, psi)) * STEP_M**2)
    pullback = float(np.vdot(temperature_pullback(psi), temperature))
    assert expected < 0.0
    assert np.isclose(objective, expected, rtol=1e-13, atol=0.0)
    assert np.isclose(integrated_map, expected, rtol=1e-13, atol=0.0)
    assert np.isclose(pullback, expected, rtol=1e-13, atol=0.0)


def test_exact_solid_void_audit_detects_subminimum_features() -> None:
    rho = np.zeros((80, 80))
    rho[39:41, 39:41] = 1.0
    audit = exact_500nm_audit(rho)
    assert audit["solid_bad_cell_count"] == 4
    assert audit["void_bad_cell_count"] == 0
    assert not audit["solid_pass"] and audit["void_pass"]


def test_finite_density_mapping_has_exact_discrete_transpose() -> None:
    rng = np.random.default_rng(20260823)
    latent = 0.2 + 0.6 * rng.random(CONTRACT.design_shape)
    direction = rng.standard_normal(CONTRACT.design_shape)
    cotangent = rng.standard_normal(CONTRACT.design_shape)
    jvp = MAPPING.jvp(latent, direction, beta=3.0)
    vjp = MAPPING.vjp(latent, cotangent, beta=3.0)
    left = float(np.vdot(cotangent, jvp))
    right = float(np.vdot(vjp, direction))
    assert abs(left - right) / max(abs(left), 1e-30) < 1e-12


def test_smooth_solid_void_constraint_directional_derivatives() -> None:
    rng = np.random.default_rng(14)
    latent = 0.2 + 0.6 * rng.random(CONTRACT.design_shape)
    direction = rng.standard_normal(CONTRACT.design_shape)
    direction /= np.max(np.abs(direction))
    values, gradients, _ = smooth_500nm_constraints(latent, beta=4.0)
    step = 2.5e-4
    plus = smooth_500nm_constraints(latent + step * direction, beta=4.0)[0]
    minus = smooth_500nm_constraints(latent - step * direction, beta=4.0)[0]
    finite_difference = (plus - minus) / (2.0 * step)
    adjoint = gradients.reshape(2, -1) @ direction.ravel()
    error = np.abs(adjoint - finite_difference) / np.maximum(
        np.abs(finite_difference), 1e-14
    )
    assert np.all(np.isfinite(values))
    assert np.max(error) < 5e-3
