from __future__ import annotations

import numpy as np

from photothermal_pte.finite_inverse_design.contract import (
    SIGMA_A_S_M,
    SIGMA_B_S_M,
    SEEBECK_A_V_K,
    SEEBECK_B_V_K,
    WEIGHTING_FIELD_X_M_INV,
    WEIGHTING_FIELD_Y_M_INV,
)
from photothermal_pte.finite_inverse_design.uniform_weighting_pte import (
    build_uniform_45deg_weighting_pte,
)
from photothermal_pte.validation.photothermal_stage1.anisotropic_heat_fvm import (
    assemble_steady_diagonal_kappa,
)


def _functional():
    x = np.linspace(-2.0e-6, 2.0e-6, 18)
    y = np.linspace(-2.0e-6, 2.0e-6, 16)
    z = np.linspace(-100.0e-9, 0.0, 5)
    shape = (x.size - 1, y.size - 1, z.size - 1)
    kappa = np.broadcast_to([14.4, 3.8, 1.0], (*shape, 3)).copy()
    system = assemble_steady_diagonal_kappa(
        x_edges_m=x,
        y_edges_m=y,
        z_edges_m=z,
        kappa_W_mK=kappa,
        dirichlet_temperature_K={"z_min": 0.0},
    )
    functional = build_uniform_45deg_weighting_pte(
        x_edges_m=x,
        y_edges_m=y,
        z_edges_m=z,
        active_mask=system.active_mask,
        active_ids=system.active_ids,
        flake_mask=system.active_mask,
    )
    return x, y, z, system, functional


def test_uniform_weighting_pte_affine_field_matches_integral():
    x, y, z, system, functional = _functional()
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    a, b = 2.3e6, -1.7e6
    temperature = (
        4.0
        + a * xc[:, None, None]
        + b * yc[None, :, None]
        + np.zeros((1, 1, z.size - 1))
    )[system.active_mask]
    volume = (
        (x[-1] - x[0])
        * (y[-1] - y[0])
        * (z[-1] - z[0])
    )
    expected = -volume * (
        SIGMA_A_S_M
        * SEEBECK_A_V_K
        * a
        * WEIGHTING_FIELD_X_M_INV
        + SIGMA_B_S_M
        * SEEBECK_B_V_K
        * b
        * WEIGHTING_FIELD_Y_M_INV
    )
    assert np.isclose(
        functional.evaluate_active(temperature),
        expected,
        rtol=3e-13,
        atol=1e-25,
    )


def test_uniform_weighting_forward_and_temperature_source_are_identical():
    _, _, _, system, functional = _functional()
    rng = np.random.default_rng(2026072705)
    temperature = rng.normal(size=system.matrix_W_K.shape[0])
    forward = functional.evaluate_active(temperature)
    transpose = float(
        np.dot(functional.temperature_source_A_K, temperature)
    )
    assert np.isclose(forward, transpose, rtol=3e-13, atol=1e-25)
    assert functional.base.periodic_axes == ()
