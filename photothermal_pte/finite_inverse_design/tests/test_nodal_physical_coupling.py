from __future__ import annotations

import numpy as np

from photothermal_pte.finite_inverse_design.nodal_physical_coupling import (
    NodalPhysicalCoupling,
)


def coupling(cell_nm: float) -> NodalPhysicalCoupling:
    nodes = np.linspace(-1e-6, 1e-6, 81)
    cells = int(round(2000.0 / cell_nm))
    edges = np.linspace(-1e-6, 1e-6, cells + 1)
    z = np.linspace(0.0, 600e-9, 13)
    return NodalPhysicalCoupling(
        x_nodes_m=nodes,
        y_nodes_m=nodes,
        optical_z_nodes_m=z,
        thermal_x_edges_m=edges,
        thermal_y_edges_m=edges,
    )


def test_endpoints_affine_and_no_periodic_corner_wrap():
    model = coupling(100.0)
    zeros = np.zeros(model.physical_shape)
    ones = np.ones(model.physical_shape)
    assert np.array_equal(model.optical(zeros), np.zeros(model.optical_shape))
    assert np.array_equal(model.optical(ones), np.ones(model.optical_shape))
    assert np.allclose(model.thermal(zeros), 0.0, atol=1e-15)
    assert np.allclose(model.thermal(ones), 1.0, atol=1e-13)

    x = model.x_nodes_m[:, None]
    y = model.y_nodes_m[None, :]
    affine = 0.5 + 0.1 * x / 1e-6 - 0.07 * y / 1e-6
    xc = 0.5 * (
        model.thermal_x_edges_m[:-1] + model.thermal_x_edges_m[1:]
    )
    yc = 0.5 * (
        model.thermal_y_edges_m[:-1] + model.thermal_y_edges_m[1:]
    )
    expected = (
        0.5 + 0.1 * xc[:, None] / 1e-6 - 0.07 * yc[None, :] / 1e-6
    )
    assert np.allclose(model.thermal(affine), expected, atol=5e-14)

    corner = np.zeros(model.physical_shape)
    corner[0, 0] = 1.0
    mapped = model.thermal(corner)
    assert mapped[0, 0] > 0.0
    assert mapped[-1, -1] == 0.0
    assert mapped[-1, 0] == 0.0
    assert mapped[0, -1] == 0.0


def test_optical_and_thermal_jvp_vjp_dot_products():
    rng = np.random.default_rng(2026072708)
    rho = rng.uniform(0.2, 0.8, size=(81, 81))
    direction = rng.normal(size=rho.shape)
    for cell_nm in (100.0, 50.0):
        model = coupling(cell_nm)
        optical_weight = rng.normal(size=model.optical_shape)
        thermal_weight = rng.normal(size=model.thermal_shape)
        assert np.isclose(
            np.sum(model.optical_jvp(direction) * optical_weight),
            np.sum(direction * model.optical_vjp(optical_weight)),
            rtol=2e-13,
            atol=1e-12,
        )
        assert np.isclose(
            np.sum(model.thermal_jvp(direction) * thermal_weight),
            np.sum(direction * model.thermal_vjp(thermal_weight)),
            rtol=2e-13,
            atol=1e-12,
        )
        step = 1e-5
        optical_fd = (
            model.optical(rho + step * direction)
            - model.optical(rho - step * direction)
        ) / (2.0 * step)
        thermal_fd = (
            model.thermal(rho + step * direction)
            - model.thermal(rho - step * direction)
        ) / (2.0 * step)
        assert np.allclose(
            optical_fd, model.optical_jvp(direction), rtol=2e-11
        )
        assert np.allclose(
            thermal_fd, model.thermal_jvp(direction), rtol=2e-11
        )
