from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    TimeSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_mesh import (
    grid_edges as baseline_grid_edges,
    mesh_audit as baseline_mesh_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_refinement import (
    case_contract,
    grid_edges,
    mesh_audit,
)


def test_full_domain_z2_retains_every_baseline_edge_and_doubles_cells() -> None:
    base_x, base_y, base_z = baseline_grid_edges()
    x, y, z = grid_edges(2)

    assert np.array_equal(x, base_x)
    assert np.array_equal(y, base_y)
    assert z.size - 1 == 2 * (base_z.size - 1)
    assert all(np.any(np.isclose(z, edge, rtol=0.0, atol=2e-18)) for edge in base_z)


def test_full_domain_z2_audit_is_explicit_and_fail_closed() -> None:
    audit = mesh_audit(2)

    assert audit["grid_shape_xyz"] == [186, 186, 300]
    assert audit["yee_cell_count"] == 2 * baseline_mesh_audit()["yee_cell_count"]
    assert audit["pml_cells_each_face_xyz"] == [8, 8, 16]
    assert all(audit["invariants"].values())
    assert audit["rules"]["optimizer_start_allowed"] is False
    assert audit["rules"]["one_refinement_pair_is_not_final_mesh_convergence"]


def test_full_domain_z2_case_hash_is_deterministic() -> None:
    time = TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5)

    assert case_contract(time, 2) == case_contract(time, 2)
    assert len(case_contract(time, 2)["case_contract_sha256"]) == 64
