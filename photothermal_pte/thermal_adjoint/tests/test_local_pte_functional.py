from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PHOTOTHERMAL = Path(__file__).resolve().parents[2]
THERMAL = PHOTOTHERMAL / "thermal_adjoint"
FVM = PHOTOTHERMAL / "validation" / "photothermal_stage1"
for path in (THERMAL, FVM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anisotropic_heat_fvm import assemble_steady_diagonal_kappa
from local_pte_functional import build_local_pte_functional


def _system(nx=7, ny=6, nz=3, periodic=()):
    x = np.linspace(-3.0e-6, 3.0e-6, nx + 1)
    y = np.linspace(-2.0e-6, 2.0e-6, ny + 1)
    z = np.linspace(-100.0e-9, 0.0, nz + 1)
    shape = (nx, ny, nz)
    kappa = np.broadcast_to([14.4, 3.8, 1.0], (*shape, 3)).copy()
    system = assemble_steady_diagonal_kappa(
        x_edges_m=x,
        y_edges_m=y,
        z_edges_m=z,
        kappa_W_mK=kappa,
        dirichlet_temperature_K={"z_min": 0.0},
        periodic_axes=periodic,
    )
    return x, y, z, system


def test_constant_temperature_has_zero_local_pte():
    x, y, z, system = _system()
    functional = build_local_pte_functional(
        x_edges_m=x,
        y_edges_m=y,
        z_edges_m=z,
        active_mask=system.active_mask,
        active_ids=system.active_ids,
        fom_mask=system.active_mask,
    )
    temperature = np.full(system.matrix_W_K.shape[0], 17.0)
    assert abs(functional.evaluate_active(temperature)) < 1.0e-22
    assert abs(np.dot(functional.temperature_source_A_m_K, temperature)) < 1.0e-22


def test_affine_temperature_matches_analytic_functional_and_transpose():
    x, y, z, system = _system()
    functional = build_local_pte_functional(
        x_edges_m=x,
        y_edges_m=y,
        z_edges_m=z,
        active_mask=system.active_mask,
        active_ids=system.active_ids,
        fom_mask=system.active_mask,
    )
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    a, b = 2.3e6, -1.7e6
    full = (
        4.0
        + a * xc[:, None, None]
        + b * yc[None, :, None]
        + np.zeros((1, 1, z.size - 1))
    )
    active = full[system.active_mask]
    expected = -(
        functional.sigma_a_S_m * functional.seebeck_a_V_K * a
        + functional.sigma_b_S_m * functional.seebeck_b_V_K * b
    ) * np.sum(functional.integration_volume_m3) / np.sqrt(2.0)
    value = functional.evaluate_active(active)
    transpose_value = float(
        np.dot(functional.temperature_source_A_m_K, active)
    )
    assert np.isclose(value, expected, rtol=2.0e-13, atol=1.0e-25)
    assert np.isclose(transpose_value, value, rtol=2.0e-13, atol=1.0e-25)


def test_periodic_full_mask_telescopes_and_seam_is_used():
    x, y, z, system = _system(nx=12, ny=10, periodic=("x", "y"))
    functional = build_local_pte_functional(
        x_edges_m=x,
        y_edges_m=y,
        z_edges_m=z,
        active_mask=system.active_mask,
        active_ids=system.active_ids,
        fom_mask=system.active_mask,
        periodic_axes=("x", "y"),
    )
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    field = (
        np.sin(2.0 * np.pi * (xc - x[0]) / (x[-1] - x[0]))[:, None, None]
        + 0.4
        * np.cos(2.0 * np.pi * (yc - y[0]) / (y[-1] - y[0]))[None, :, None]
        + np.zeros((1, 1, z.size - 1))
    )
    value = functional.evaluate_active(field[system.active_mask])
    # The continuous periodic integral is exactly zero.  Its discrete value is
    # cancellation-limited, so use an absolute floor instead of dividing by a
    # quantity that is itself near machine zero.
    assert abs(value) <= 1.0e-25
    first_id = int(system.active_ids[0, 0, 0])
    last_id = int(system.active_ids[-1, 0, 0])
    row = int(np.flatnonzero(
        np.all(functional.selected_cells == [0, 0, 0], axis=1)
    )[0])
    assert functional.derivative_x_m_inv[row, last_id] != 0.0
    row_scale = np.max(np.abs(functional.derivative_x_m_inv[row]))
    assert abs(functional.derivative_x_m_inv[row, first_id]) <= 1.0e-13 * row_scale


def test_finite_mask_stencils_do_not_sample_outside_mask():
    x, y, z, system = _system(nx=9, ny=8, nz=2)
    mask = np.zeros(system.shape, bool)
    mask[2:7, 2:6, :] = True
    functional = build_local_pte_functional(
        x_edges_m=x,
        y_edges_m=y,
        z_edges_m=z,
        active_mask=system.active_mask,
        active_ids=system.active_ids,
        fom_mask=mask,
    )
    outside = np.zeros(system.shape)
    outside[~mask] = 1.0e9
    assert functional.evaluate_active(outside[system.active_mask]) == 0.0
