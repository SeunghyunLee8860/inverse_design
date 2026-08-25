from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.dfm import (
    exact_500nm_audit,
    smooth_500nm_physical_constraints,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    LUMERICAL_MINIMUM_SOLID_FEATURE_M,
    LUMERICAL_MINIMUM_VOID_FEATURE_M,
    OPTIMIZER_250NM_MAPPING as NOMINAL_MAPPING,
    calibrated_lumerical_250nm_dfm_caps,
    design_state_audit,
    exact_binary_cell_candidate,
    projected_cell_density,
    projected_cell_jvp,
    projected_cell_vjp,
    smooth_lumerical_250nm_constraints,
)


def _latent() -> np.ndarray:
    x = np.linspace(-1.0, 1.0, CONTRACT.design_node_shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, CONTRACT.design_node_shape[1])[None, :]
    return 0.5 + 0.16 * np.sin(0.8 * np.pi * x) * np.cos(0.6 * np.pi * y)


def test_lumerical_mapping_uses_nodal_carrier_and_preserves_constants() -> None:
    audit = NOMINAL_MAPPING.audit()
    assert audit["latent_shape_xy"] == [81, 81]
    assert audit["projected_shape_xy"] == [81, 81]
    assert audit["downstream_cell_shape_xy"] == [80, 80]
    assert audit["legacy_80x80_cell_mapping_is_optimizer_carrier"] is False
    assert audit["optical_rho_power"] is None
    assert audit["np_density_used"] is False
    assert audit["conic_filter_radius_nm"] == 250.0
    assert audit["minimum_solid_feature_nm"] == 250.0
    assert audit["minimum_void_feature_nm"] == 250.0
    assert audit["constant_preservation_max_abs"] < 1.0e-14
    projected = NOMINAL_MAPPING.physical(
        np.full(CONTRACT.design_node_shape, 0.5), beta=4.0
    )
    assert np.allclose(projected, 0.5, rtol=0.0, atol=2.0e-15)


def test_nodal_filter_projection_jvp_and_vjp() -> None:
    rng = np.random.default_rng(20260824)
    latent = _latent()
    direction = rng.standard_normal(CONTRACT.design_node_shape)
    direction /= np.max(np.abs(direction))
    cotangent = rng.standard_normal(CONTRACT.design_node_shape)
    beta = 4.0
    jvp = NOMINAL_MAPPING.jvp(latent, direction, beta)
    vjp = NOMINAL_MAPPING.vjp(latent, cotangent, beta)
    left = float(np.vdot(cotangent, jvp))
    right = float(np.vdot(vjp, direction))
    assert abs(left - right) / max(abs(left), abs(right), 1.0e-30) < 1.0e-12
    step = 1.0e-6
    finite_difference = (
        NOMINAL_MAPPING.physical(latent + step * direction, beta)
        - NOMINAL_MAPPING.physical(latent - step * direction, beta)
    ) / (2.0 * step)
    assert np.linalg.norm(jvp - finite_difference) / np.linalg.norm(
        finite_difference
    ) < 1.0e-8


def test_projected_cell_chain_has_exact_transpose_and_fd() -> None:
    rng = np.random.default_rng(31)
    latent = _latent()
    direction = rng.standard_normal(CONTRACT.design_node_shape)
    direction /= np.max(np.abs(direction))
    cotangent = rng.standard_normal(CONTRACT.design_shape)
    beta = 5.0
    tangent = projected_cell_jvp(latent, direction, beta)
    pullback = projected_cell_vjp(latent, cotangent, beta)
    left = float(np.vdot(cotangent, tangent))
    right = float(np.vdot(pullback, direction))
    assert abs(left - right) / max(abs(left), abs(right), 1.0e-30) < 1.0e-12
    step = 1.0e-6
    finite_difference = (
        projected_cell_density(latent + step * direction, beta)
        - projected_cell_density(latent - step * direction, beta)
    ) / (2.0 * step)
    assert np.linalg.norm(tangent - finite_difference) / np.linalg.norm(
        finite_difference
    ) < 1.0e-8


def test_lumerical_cell_dfm_constraint_gradient_reaches_nodal_latent() -> None:
    rng = np.random.default_rng(44)
    latent = _latent()
    direction = rng.standard_normal(CONTRACT.design_node_shape)
    direction /= np.max(np.abs(direction))
    beta = 4.0
    values, gradients, fields = smooth_lumerical_250nm_constraints(latent, beta)
    step = 2.5e-4
    plus = smooth_lumerical_250nm_constraints(
        latent + step * direction, beta
    )[0]
    minus = smooth_lumerical_250nm_constraints(
        latent - step * direction, beta
    )[0]
    finite_difference = (plus - minus) / (2.0 * step)
    adjoint = gradients.reshape(2, -1) @ direction.ravel()
    error = np.abs(adjoint - finite_difference) / np.maximum(
        np.maximum(np.abs(adjoint), np.abs(finite_difference)), 1.0e-14
    )
    assert values.shape == (2,)
    assert gradients.shape == (2, *CONTRACT.design_node_shape)
    assert fields["projected_nodal_density"].shape == CONTRACT.design_node_shape
    assert fields["projected_cell_density"].shape == CONTRACT.design_shape
    assert np.max(error) < 5.0e-3


def test_binary_candidate_is_cell_based_and_requires_exact_au_reevaluation() -> None:
    projected = np.zeros(CONTRACT.design_node_shape, dtype=np.float64)
    projected[40:42, 40:42] = 1.0
    mask, audit = exact_binary_cell_candidate(projected)
    assert mask.shape == CONTRACT.design_shape
    assert mask.dtype == np.uint8
    assert audit["candidate_rule"] == "threshold four-node cell-average occupancy"
    assert audit["requires_ordinary_dispersive_au_reevaluation"] is True
    assert audit["solid_pass"] is True
    assert audit["void_pass"] is True
    assert audit["minimum_solid_feature_nm"] == 250.0
    state = design_state_audit(_latent(), beta=4.0)
    assert state["shared_projected_density"]["nodal_shape_xy"] == [81, 81]
    assert state["shared_projected_density"]["pde_cell_shape_xy"] == [80, 80]


def test_250nm_dfm_caps_are_calibrated_on_exact_pass_binary_patterns() -> None:
    caps, calibration = calibrated_lumerical_250nm_dfm_caps()
    assert caps.shape == (2,)
    assert np.all(np.isfinite(caps))
    assert np.all(caps > 0.0)
    assert calibration["minimum_solid_feature_nm"] == 250.0
    assert calibration["minimum_void_feature_nm"] == 250.0
    assert calibration["opening_radius_nm"] == 125.0
    assert calibration["minimum_feature_cells_ceil"] == 3
    assert calibration["opening_footprint_pixel_count"] == 5
    assert len(calibration["patterns"]) == 4
    for row in calibration["patterns"]:
        assert row["exact_solid_pass"] is True
        assert row["exact_void_pass"] is True
        assert np.all(np.asarray(row["smooth_values"]) <= caps)


def test_subminimum_solid_line_violates_exact_and_smooth_250nm_gates() -> None:
    density = np.zeros(CONTRACT.design_shape, dtype=np.float64)
    density[40, 20:60] = 1.0
    exact = exact_500nm_audit(
        density,
        spacing_m=CONTRACT.design_pitch_m,
        minimum_feature_m=LUMERICAL_MINIMUM_SOLID_FEATURE_M,
    )
    smooth, _, _ = smooth_500nm_physical_constraints(
        density,
        spacing_m=CONTRACT.design_pitch_m,
        minimum_solid_feature_m=LUMERICAL_MINIMUM_SOLID_FEATURE_M,
        minimum_void_feature_m=LUMERICAL_MINIMUM_VOID_FEATURE_M,
    )
    caps, _ = calibrated_lumerical_250nm_dfm_caps()
    assert exact["solid_pass"] is False
    assert exact["void_pass"] is True
    assert smooth[0] > caps[0]
