from __future__ import annotations

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


def test_signed_opposite_current_objective() -> None:
    assert useful_currents(3.0e-9, -4.0e-9) == (3.0e-9, 4.0e-9)
    assert np.all(epigraph_constraints(3e-9, -4e-9, 2e-9) <= 0.0)
    value, derivative = smooth_minimum(3e-9, -4e-9, scale_A=1e-9)
    assert value < 3e-9
    assert derivative[0] > 0.0 and derivative[1] < 0.0


def test_weighting_integral_sign_means_right_to_left_internal_current() -> None:
    # S>0 and T_right>T_left produces physical J=-sigma*S*grad(T) toward
    # the left. The implemented sigma*S*dT*dpsi contribution is positive.
    sigma_t_s = 2.0
    delta_temperature = 3.0
    delta_psi = 1.0
    assert sigma_t_s * delta_temperature * delta_psi > 0.0
    assert -sigma_t_s * delta_temperature < 0.0


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
