from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_4um_model as optical_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_mesh import (
    UserBalancedMeshSpec,
    grid_edges,
    lateral_segments,
    layout_values,
    mesh_audit,
    mesh_context,
    pml_face_parameters,
    vertical_segments,
)


def test_balanced_xy_mesh_matches_user_request() -> None:
    lateral = {segment.name: segment for segment in lateral_segments()}
    assert lateral["au_design_window"].step_m == pytest.approx(100e-9)
    assert lateral["left_flake_wing"].step_m == pytest.approx(100e-9)
    assert lateral["right_flake_wing"].step_m == pytest.approx(100e-9)
    assert lateral["left_air_margin"].step_m == pytest.approx(200e-9)
    assert lateral["right_air_margin"].step_m == pytest.approx(200e-9)
    assert lateral["left_pml"].cells == 8
    assert lateral["right_pml"].cells == 8


def test_balanced_z_mesh_matches_user_request_and_declares_si_exception() -> None:
    vertical = {segment.name: segment for segment in vertical_segments()}
    for name, cells in (("sio2", 57), ("tairte4", 20), ("au", 10)):
        assert vertical[name].cells == cells
        assert vertical[name].step_m == pytest.approx(5e-9)
    for name in ("near_air", "middle_air", "source_air"):
        assert vertical[name].step_m == pytest.approx(50e-9)
    assert vertical["resolved_si"].cells == 20
    assert vertical["resolved_si"].step_m == pytest.approx(50.75e-9)
    audit = mesh_audit()
    assert audit["known_pitch_exception"]["relative_difference"] == pytest.approx(0.015)
    assert audit["known_pitch_exception"]["physical_boundary_was_not_moved"]
    assert vertical["bottom_pml_si"].cells == 8
    assert vertical["top_pml_air"].cells == 8


def test_balanced_grid_shape_bounds_and_required_edges() -> None:
    x, y, z = grid_edges()
    assert [x.size - 1, y.size - 1, z.size - 1] == [186, 186, 150]
    assert mesh_audit()["yee_cell_count"] == 186 * 186 * 150
    assert (x[0], x[-1], y[0], y[-1], z[0], z[-1]) == pytest.approx(
        (-10e-6, 10e-6, -10e-6, 10e-6, -3e-6, 3e-6)
    )
    for value in (-0.588e-6, -0.385e-6, -0.100e-6, 0.0, 0.050e-6, 0.250e-6, 0.500e-6, 0.750e-6):
        assert np.count_nonzero(np.isclose(z, value, rtol=0.0, atol=2e-18)) == 1


def test_balanced_layout_and_pml_are_explicit() -> None:
    layout = layout_values()
    assert layout["pml_cells_xy"] == 8
    assert layout["pml_cells_z"] == 8
    assert layout["flake_xy_cells"] == 160
    assert layout["au_xy_cells"] == 80
    assert layout["sio2_cells"] == 57
    assert layout["tairte4_cells"] == 20
    assert layout["au_cells"] == 10
    profiles = pml_face_parameters()
    assert set(profiles) == {"minx", "maxx", "miny", "maxy", "minz", "maxz"}


def test_balanced_context_restores_historical_builder() -> None:
    old_layout = optical_model.LAYOUT
    old_edges = optical_model.grid_edges
    with mesh_context() as audit:
        assert optical_model.LAYOUT.flake_xy_cells == 160
        assert optical_model.LAYOUT.sio2_cells == 57
        assert np.array_equal(optical_model.grid_edges()[0], grid_edges()[0])
        assert audit == mesh_audit()
    assert optical_model.LAYOUT is old_layout
    assert optical_model.grid_edges is old_edges


@pytest.mark.parametrize(
    "kwargs",
    (
        {"profile": "wrong"},
        {"design_xy_factor": 2},
        {"lateral_pml_thickness_m": 2e-6},
        {"z_pml_thickness_m": 2e-6},
    ),
)
def test_balanced_spec_rejects_silent_changes(kwargs) -> None:
    with pytest.raises(ValueError):
        UserBalancedMeshSpec(**kwargs)
