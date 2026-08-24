from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.au_density_relaxation import (
    ordal_au_index,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (
    DENSITY_IMPORT_OBJECT,
    add_density_stack_geometry,
    canonical_density_nodes,
    density_nodes,
    density_state_audit,
    density_state_sha256,
    nodal_to_cell_average,
    nodal_to_cell_jvp,
    nodal_to_cell_vjp,
)


class _FakeFdtd:
    def __init__(self) -> None:
        self.rectangles: list[dict[str, object]] = []
        self.import_properties: list[dict[str, object]] = []
        self.import_calls: list[dict[str, np.ndarray]] = []

    def addrect(self) -> dict[str, object]:
        rectangle: dict[str, object] = {}
        self.rectangles.append(rectangle)
        return rectangle

    def addimport(self, properties: dict[str, object]) -> dict[str, object]:
        self.import_properties.append(dict(properties))
        return self.import_properties[-1]

    def importnk2(
        self,
        index: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> int:
        self.import_calls.append(
            {
                "index": np.asarray(index).copy(),
                "x": np.asarray(x).copy(),
                "y": np.asarray(y).copy(),
                "z": np.asarray(z).copy(),
            }
        )
        return 1


def test_density_nodes_cover_exact_8um_window_at_100nm_pitch() -> None:
    x, y, z = density_nodes()
    assert CONTRACT.design_shape == (80, 80)
    assert CONTRACT.design_node_shape == (81, 81)
    assert x.size == y.size == 81 and z.size == 2
    assert x[0] == -4.0e-6 and x[-1] == 4.0e-6
    assert y[0] == -4.0e-6 and y[-1] == 4.0e-6
    assert np.allclose(np.diff(x), 100.0e-9, rtol=0.0, atol=1.0e-18)
    assert np.allclose(np.diff(y), 100.0e-9, rtol=0.0, atol=1.0e-18)
    assert np.array_equal(z, np.asarray([0.0, 50.0e-9]))


def test_nodal_to_cell_map_preserves_constants_and_range() -> None:
    for value in (0.0, 0.25, 1.0):
        nodes = np.full(CONTRACT.design_node_shape, value)
        cells = nodal_to_cell_average(nodes)
        assert cells.shape == CONTRACT.design_shape
        assert np.array_equal(cells, np.full(CONTRACT.design_shape, value))
    rng = np.random.default_rng(20260824)
    nodes = rng.random(CONTRACT.design_node_shape)
    cells = nodal_to_cell_average(nodes)
    assert np.min(cells) >= np.min(nodes)
    assert np.max(cells) <= np.max(nodes)


def test_nodal_to_cell_map_has_exact_discrete_transpose() -> None:
    rng = np.random.default_rng(7)
    direction = rng.standard_normal(CONTRACT.design_node_shape)
    cotangent = rng.standard_normal(CONTRACT.design_shape)
    left = float(np.vdot(nodal_to_cell_jvp(direction), cotangent))
    right = float(np.vdot(direction, nodal_to_cell_vjp(cotangent)))
    assert abs(left - right) / max(abs(left), 1.0e-300) < 2.0e-15


def test_nodal_to_cell_pullback_matches_independent_fd() -> None:
    rng = np.random.default_rng(13)
    nodes = 0.1 + 0.8 * rng.random(CONTRACT.design_node_shape)
    direction = rng.standard_normal(CONTRACT.design_node_shape)
    direction /= np.max(np.abs(direction))
    cells = nodal_to_cell_average(nodes)
    cell_cotangent = 2.0 * cells
    gradient = nodal_to_cell_vjp(cell_cotangent)
    analytic = float(np.vdot(gradient, direction))
    step = 1.0e-6
    plus = float(np.sum(nodal_to_cell_average(nodes + step * direction) ** 2))
    minus = float(np.sum(nodal_to_cell_average(nodes - step * direction) ** 2))
    finite_difference = (plus - minus) / (2.0 * step)
    assert abs(analytic - finite_difference) / max(abs(analytic), 1.0e-300) < 2.0e-8


def test_density_hash_and_audit_bind_shared_nodal_state() -> None:
    nodes = np.full(CONTRACT.design_node_shape, 0.5)
    baseline = density_state_sha256(nodes)
    changed = nodes.copy()
    changed[40, 40] += 1.0e-6
    assert baseline != density_state_sha256(changed)
    audit = density_state_audit(nodes)
    assert audit["density_state_sha256"] == baseline
    assert audit["nodal_shape_xy"] == [81, 81]
    assert audit["pde_cell_shape_xy"] == [80, 80]
    assert audit["optical_rho_power"] is None
    assert audit["gray_state_claimed_as_fabricated_material"] is False


def test_layout_uses_one_import_object_and_no_exact_au_prisms() -> None:
    nodes = np.linspace(0.0, 1.0, 81)[:, None] * np.ones((1, 81))
    fdtd = _FakeFdtd()
    audit = add_density_stack_geometry(fdtd, nodes)
    assert [item["name"] for item in fdtd.import_properties] == [DENSITY_IMPORT_OBJECT]
    assert len(fdtd.import_calls) == 1
    imported = fdtd.import_calls[0]
    assert imported["index"].shape == (81, 81, 2)
    assert imported["index"][0, 0, 0] == 1.0 + 0.0j
    assert imported["index"][-1, 0, 0] == ordal_au_index()
    assert len(fdtd.rectangles) == 3
    assert not any(str(item["name"]).startswith("exact_Au_prism") for item in fdtd.rectangles)
    assert audit["Maxwell_solve_run"] is False
    assert audit["density_state"]["density_state_sha256"] == density_state_sha256(nodes)


@pytest.mark.parametrize(
    "bad",
    (
        np.zeros((80, 80)),
        np.zeros((82, 81)),
        np.full((81, 81), np.nan),
        np.full((81, 81), 1.01),
    ),
)
def test_density_nodes_fail_closed_on_wrong_state(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        canonical_density_nodes(bad)
