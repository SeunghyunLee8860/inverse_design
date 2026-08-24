from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_gray_q_coupling import (
    GrayYeeQCoupling,
)


def _coupling() -> GrayYeeQCoupling:
    coordinates = {
        "x": (
            np.asarray([-0.8, -0.1, 0.7]),
            np.asarray([-0.9, -0.2, 0.5, 0.9]),
            np.asarray([-0.6, 0.0, 0.8]),
        ),
        "y": (
            np.asarray([-0.7, 0.0, 0.8]),
            np.asarray([-0.8, -0.1, 0.6, 0.9]),
            np.asarray([-0.5, 0.1, 0.7]),
        ),
        "z": (
            np.asarray([-0.9, -0.3, 0.4]),
            np.asarray([-0.7, 0.0, 0.8]),
            np.asarray([-0.8, -0.2, 0.3, 0.9]),
        ),
    }
    target = (
        np.asarray([-1.0, -0.45, 0.2, 0.55, 1.0]),
        np.asarray([-1.0, -0.35, 0.15, 0.65, 1.0]),
        np.asarray([-1.0, -0.4, 0.25, 1.0]),
    )
    return GrayYeeQCoupling.from_component_coordinates(coordinates, target)


def test_gray_q_coupling_conserves_all_component_power() -> None:
    coupling = _coupling()
    q = {
        component: np.arange(np.prod(coupling.source_shape(component)), dtype=float).reshape(
            coupling.source_shape(component)
        )
        + 1.0
        for component in "xyz"
    }
    mapped, audit = coupling.map_power(q)
    expected = sum(
        float(np.sum(q[component] * coupling.source_volume_m3(component)))
        for component in "xyz"
    )
    assert np.isclose(float(np.sum(mapped)), expected, rtol=2.0e-13, atol=1.0e-30)
    assert audit["relative_power_error"] < 2.0e-13
    assert audit["material_equality_filter"] is False
    assert audit["density_dependent_geometric_mask"] is False


def test_gray_q_coupling_transpose_is_exact() -> None:
    audit = _coupling().transpose_dot_audit()
    assert audit["relative_error"] < 1.0e-12


def test_gray_q_coupling_rejects_negative_physical_absorption() -> None:
    coupling = _coupling()
    q = {
        component: np.ones(coupling.source_shape(component), dtype=float)
        for component in "xyz"
    }
    q["y"][0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="negative absorption"):
        coupling.map_power(q)
