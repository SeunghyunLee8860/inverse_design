from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_only_contract import (
    CONTRACT,
    binary_mask_sha256,
    canonical_binary_mask,
)


def test_contract_forbids_old_maxwell_and_gray_paths() -> None:
    payload = asdict(CONTRACT)
    assert payload["maxwell_solver"].startswith("Ansys Lumerical FDTD")
    assert payload["maxwell_accelerator_required"] == "NVIDIA B200"
    assert payload["gray_au_allowed_in_any_physics"] is False
    assert payload["importnk_au_allowed"] is False
    assert payload["lumopt_topology_gradient_allowed"] is False
    assert payload["fdtdx_allowed"] is False
    assert payload["jax_maxwell_allowed"] is False


def test_binary_mask_hash_is_shape_and_layout_sensitive() -> None:
    mask = np.asarray([[0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    assert binary_mask_sha256(mask) == binary_mask_sha256(mask.astype(float))
    assert binary_mask_sha256(mask) != binary_mask_sha256(mask.T)
    changed = mask.copy()
    changed[0, 0] = 1
    assert binary_mask_sha256(mask) != binary_mask_sha256(changed)


@pytest.mark.parametrize(
    "bad",
    [
        np.asarray([0, 1]),
        np.asarray([[0.0, 0.5]]),
        np.asarray([[0.0, np.nan]]),
        np.empty((0, 2)),
    ],
)
def test_binary_mask_rejects_nonphysical_inputs(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        canonical_binary_mask(bad)

