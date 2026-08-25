from __future__ import annotations

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    TimeSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_source_only import (
    CASE_VERSION,
    balanced_case_contract,
)


def test_balanced_source_case_contract_is_deterministic_and_fail_closed() -> None:
    time = TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5)
    first = balanced_case_contract(time)
    second = balanced_case_contract(time)
    assert first == second
    assert first["version"] == CASE_VERSION
    assert len(first["case_contract_sha256"]) == 64
    assert first["mesh"]["grid_shape_xyz"] == [186, 186, 150]
    assert first["rules"]["optimizer_start_allowed"] is False
    assert first["rules"]["source_pair_required_before_material_case"] is True


def test_balanced_source_case_hash_changes_with_time() -> None:
    short = balanced_case_contract(
        TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5)
    )
    long = balanced_case_contract(
        TimeSpec(total_periods=32, window_periods=4, courant_factor=0.5)
    )
    assert short["case_contract_sha256"] != long["case_contract_sha256"]
