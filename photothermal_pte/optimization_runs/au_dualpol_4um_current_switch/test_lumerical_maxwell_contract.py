from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_maxwell_contract import (
    CONTRACT,
    binary_mask_sha256,
    canonical_binary_mask,
    canonical_design_fraction,
    design_fraction_sha256,
)


def test_contract_preserves_lumerical_plus_custom_gpu_pde_architecture() -> None:
    payload = asdict(CONTRACT)
    assert payload["maxwell_solver"].startswith("Ansys Lumerical FDTD")
    assert payload["maxwell_accelerator_required"] == "NVIDIA B200"
    assert "custom CUDA" in payload["thermal_solver"]
    assert "custom CUDA" in payload["electrical_solver"]
    assert payload["continuous_relaxation_allowed_during_optimization"] is True
    assert payload["different_optical_thermal_electrical_design_fields_allowed"] is False
    assert payload["exact_binary_required_for_final_promotion"] is True
    assert payload["bundled_lumopt_topology_gradient_allowed_without_au_adfd"] is False
    assert payload["fdtdx_allowed"] is False
    assert payload["jax_maxwell_allowed"] is False


def test_continuous_design_fraction_is_allowed_and_hash_is_exact() -> None:
    fraction = np.asarray([[0.0, 0.25], [0.5, 1.0]])
    checked = canonical_design_fraction(fraction)
    assert checked.dtype == np.float64
    assert np.array_equal(checked, fraction)
    assert design_fraction_sha256(fraction) == design_fraction_sha256(fraction.copy())
    changed = fraction.copy()
    changed[0, 1] = np.nextafter(changed[0, 1], 1.0)
    assert design_fraction_sha256(fraction) != design_fraction_sha256(changed)


@pytest.mark.parametrize(
    "bad",
    [
        np.asarray([0.0, 1.0]),
        np.asarray([[-1.0e-4, 0.5]]),
        np.asarray([[0.0, 1.0001]]),
        np.asarray([[0.0, np.nan]]),
        np.empty((0, 2)),
    ],
)
def test_design_fraction_rejects_invalid_inputs(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        canonical_design_fraction(bad)


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
