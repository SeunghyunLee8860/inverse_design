from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    TimeSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_refinement import (
    case_contract,
    grid_edges,
    mesh_audit,
)


def test_z4_is_nested_and_has_four_times_baseline_z_cells() -> None:
    _, _, z2 = grid_edges(2)
    _, _, z4 = grid_edges(4)
    audit = mesh_audit(4)

    assert audit["grid_shape_xyz"] == [186, 186, 600]
    assert audit["yee_cell_count"] == 20_757_600
    assert audit["pml_cells_each_face_xyz"] == [8, 8, 32]
    assert all(np.any(np.isclose(z4, edge, rtol=0.0, atol=2e-18)) for edge in z2)
    assert all(audit["invariants"].values())


def test_z4_case_is_distinct_from_z2() -> None:
    time = TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5)

    assert (
        case_contract(time, 4)["case_contract_sha256"]
        != case_contract(time, 2)["case_contract_sha256"]
    )
