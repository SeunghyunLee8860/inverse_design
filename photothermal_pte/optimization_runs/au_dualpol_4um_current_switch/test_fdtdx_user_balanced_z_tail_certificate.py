from __future__ import annotations

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_certificate import (
    EXPECTED_FACTOR,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_tail_certificate import (
    FACTORS,
    LEVELS,
)


def test_tail_levels_and_shared_material_auditor_are_exact() -> None:
    assert LEVELS == ("z2", "z4")
    assert FACTORS == {"z2": 2, "z4": 4}
    assert all(EXPECTED_FACTOR[level] == factor for level, factor in FACTORS.items())
