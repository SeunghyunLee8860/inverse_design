from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_4um_model as optical_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    TimeSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_lateral_refinement import (
    UserBalancedLateralRefinementSpec,
    case_contract,
    grid_edges,
    lateral_segments,
    layout_values,
    material_spec,
    mesh_audit,
    mesh_context,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_refinement import (
    grid_edges as z_grid_edges,
)


def test_only_design_and_complete_flake_are_refined_to_50nm() -> None:
    segments = {segment.name: segment for segment in lateral_segments()}
    for name in ("left_flake_wing", "au_design_window", "right_flake_wing"):
        assert segments[name].step_m == pytest.approx(50.0e-9)
    for name in ("left_air_margin", "right_air_margin"):
        assert segments[name].step_m == pytest.approx(200.0e-9)
    assert segments["left_pml"].cells == segments["right_pml"].cells == 8


def test_lateral_refined_grid_is_346x346_at_fixed_z2() -> None:
    x, y, z = grid_edges()
    _, _, expected_z = z_grid_edges(2)
    assert [x.size - 1, y.size - 1, z.size - 1] == [346, 346, 300]
    assert np.array_equal(z, expected_z)
    assert mesh_audit()["yee_cell_count"] == 35_914_800
    assert mesh_audit()["grid_contract_sha256"] == (
        "341c9d45790cd090ce3b59928c59351f15f6146c22fc716cea1c467752a4f81c"
    )


def test_layout_and_exact_mask_replication_are_explicit() -> None:
    layout = layout_values()
    assert layout["flake_xy_cells"] == 320
    assert layout["au_xy_cells"] == 160
    assert layout["source_xy_cells"] == 320
    assert layout["pml_cells_xy"] == 8
    assert layout["pml_cells_z"] == 16
    assert material_spec().design_xy_factor == 2


def test_case_contract_is_canonical_24_4_and_fail_closed() -> None:
    case = case_contract(
        TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5)
    )
    assert case["mesh"] == mesh_audit()
    assert case["time"]["total_periods"] == 24
    assert case["time"]["window_periods"] == 4
    assert case["rules"]["optimizer_start_allowed"] is False


def test_mesh_context_restores_global_builder() -> None:
    old_layout = optical_model.LAYOUT
    old_edges = optical_model.grid_edges
    with mesh_context() as audit:
        assert optical_model.LAYOUT.flake_xy_cells == 320
        assert optical_model.LAYOUT.au_xy_cells == 160
        assert np.array_equal(optical_model.grid_edges()[0], grid_edges()[0])
        assert audit == mesh_audit()
    assert optical_model.LAYOUT is old_layout
    assert optical_model.grid_edges is old_edges


@pytest.mark.parametrize(
    "kwargs",
    (
        {"design_xy_factor": 1},
        {"full_domain_z_factor": 1},
        {"lateral_pml_thickness_m": 2.0e-6},
        {"z_pml_thickness_m": 2.0e-6},
    ),
)
def test_lateral_spec_rejects_silent_changes(kwargs) -> None:
    with pytest.raises(ValueError):
        UserBalancedLateralRefinementSpec(**kwargs)
