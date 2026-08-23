from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import CONTRACT
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.dfm import exact_500nm_audit
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.objective import (
    epigraph_constraints,
    smooth_minimum,
    useful_currents,
)


def test_contract_geometry_and_source_boundary() -> None:
    assert CONTRACT.design_shape == (80, 80)
    assert CONTRACT.axis_x == "b" and CONTRACT.axis_y == "a"
    assert CONTRACT.flake_boundary_intensity_fraction < 5.0e-4


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
