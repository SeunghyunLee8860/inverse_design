from __future__ import annotations

import json
from pathlib import Path

from photothermal_pte.finite_inverse_design.run_combined_physical_rho_pte_adfd import (
    array_sha256,
)
from photothermal_pte.finite_inverse_design.run_scale_adaptive_near_null_combined_adfd import (
    near_null_directions,
    old_step,
)


RAW = Path(
    "/home/seunghyun/tairte4_artifacts/"
    "corrected_combined_physical_rho_pte_adfd_20260727_2/"
    "full_five_direction_failed_noise_plateau_result.json"
)


def test_near_null_direction_contract_matches_immutable_sweep():
    if not RAW.is_file():
        return
    raw = json.loads(RAW.read_text())
    for name, direction in near_null_directions().items():
        assert array_sha256(direction) == raw["directions"][name]["sha256"]


def test_old_step_selects_exact_immutable_rows():
    if not RAW.is_file():
        return
    raw = json.loads(RAW.read_text())
    direction = raw["scenarios"]["4um"]["directions"]["central_localized"]
    assert old_step(direction, 0.01)["step"] == 0.01
    assert old_step(direction, 0.005)["step"] == 0.005
