from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from photothermal_pte.validation.paper_ir_sanity import (
    audit_straight_edge_robust_gradient as robust,
)
from photothermal_pte.validation.paper_ir_sanity import (
    run_straight_edge_analytic_source_controls as controls,
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
