from __future__ import annotations

import inspect

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_parity_ad_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_ad_contract import (
    Q_PREFACTOR,
    SOURCE_REFERENCE_POWER_W,
    ad_contract_audit,
    adfd_direction_audit,
    checkpoint_memory_lower_bounds,
    gradient_source_audit,
    latent_directions,
    normalized_target_absorption,
    target_au_imag_epsilon,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_fixed_materials import (
    TA_A,
    TA_B,
)


def test_pinned_gradient_sources_require_checkpointed_for_dispersion() -> None:
    audit = gradient_source_audit()
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert audit["method_required"] == "checkpointed"
    assert audit["reversible_allowed"] is False
    assert audit["source_sha256"] == {
        "fdtdx_config": "c33fdc10114cefee977a929f03e908b510520b9e5056ad5ff44ae79292c80f66",
        "fdtdx_dispersion": "298be6194ae1809744bfcf1fb0d102349ad05907efdcdd7c89ea1a27b67a32ff",
        "fdtdx_fdtd": "7c654097d43d5062afbef0cf8c479ba2a7db523b64683693fa4e24bc5070e4e0",
        "fdtdx_initialization": "a88a25f1907b40fc6e0e0ef736c91969f570ac4631014ec137c26186f5e3dd88",
        "fdtdx_update": "70a5f3fc95b9d78c255d4a82c2e084f7b538bf9e79f89ce95b25f13da2b216c5",
        "equinox_checkpointed": "586ab819bc827745e9ac3cd42da57ab57103c44638d4a5eb3f2e13dfb74950dc",
    }


def test_checkpoint_table_is_explicitly_only_a_lower_bound() -> None:
    audit = checkpoint_memory_lower_bounds()
    assert audit["status"] == "LOWER_BOUND_ONLY_NOT_A_FEASIBILITY_CLAIM"
    assert set(audit["candidates"]) == {"16", "32", "64", "96", "128", "192", "256"}
    totals = [
        audit["candidates"][key]["checkpoint_plus_persistent_lower_bound_bytes"]
        for key in ("16", "32", "64", "96", "128", "192", "256")
    ]
    assert totals == sorted(totals)
    assert "peak-memory" in audit["required_next_gate"]


def test_target_absorption_has_direct_nonlinear_density_dependence() -> None:
    rho = np.asarray([[0.0, 0.25], [0.5, 1.0]])
    assert np.array_equal(target_au_imag_epsilon(rho), 57.8 * rho + 69.36 * rho**2)
    e_au = np.ones((3, 2, 2, 1), dtype=np.complex128)
    e_ta = np.ones((3, 1, 1, 1), dtype=np.complex128)
    volume_au = np.ones_like(e_au.real)
    volume_ta = np.ones_like(e_ta.real)
    observed = normalized_target_absorption(
        rho_cell=rho,
        e_au=e_au,
        e_tairte4=e_ta,
        volume_au=volume_au,
        volume_tairte4=volume_ta,
    )
    expected_power = Q_PREFACTOR * (
        3.0 * np.sum(57.8 * rho + 69.36 * rho**2)
        + TA_B.target_epsilon_imag
        + TA_A.target_epsilon_imag
        + TA_B.target_epsilon_imag
    )
    assert observed == pytest.approx(expected_power / SOURCE_REFERENCE_POWER_W)


def test_numpy_and_jax_objectives_match_and_differentiate_direct_term() -> None:
    import jax
    import jax.numpy as jnp

    rho = np.full((2, 2), 0.4, dtype=np.float32)
    e_au = np.ones((3, 2, 2, 1), dtype=np.complex64)
    e_ta = np.ones((3, 1, 1, 1), dtype=np.complex64)
    volume_au = np.ones_like(e_au.real)
    volume_ta = np.ones_like(e_ta.real)
    numpy_value = normalized_target_absorption(
        rho_cell=rho,
        e_au=e_au,
        e_tairte4=e_ta,
        volume_au=volume_au,
        volume_tairte4=volume_ta,
    )

    def objective(value):
        return normalized_target_absorption(
            rho_cell=value,
            e_au=jnp.asarray(e_au),
            e_tairte4=jnp.asarray(e_ta),
            volume_au=jnp.asarray(volume_au),
            volume_tairte4=jnp.asarray(volume_ta),
            xp=jnp,
        )

    jax_value, gradient = jax.value_and_grad(objective)(jnp.asarray(rho))
    assert float(jax_value) == pytest.approx(float(numpy_value), rel=2.0e-6)
    assert np.all(np.asarray(gradient) > 0.0)


def test_four_directions_are_hash_bound_feasible_and_nontrivial() -> None:
    directions = latent_directions()
    assert list(directions) == [
        "uniform",
        "x_antisymmetric",
        "y_antisymmetric",
        "offcenter_localized_zero_mean",
    ]
    assert all(value.shape == (81, 81) for value in directions.values())
    assert all(np.max(np.abs(value)) == 1.0 for value in directions.values())
    audit = adfd_direction_audit()
    assert audit["status"] == "PASS"
    assert audit["required_centered_forwards"] == 16
    assert all(row["feasible"] for row in audit["directions"].values())
    assert all(row["nonzero_cell_tangent"] for row in audit["directions"].values())


def test_contract_is_audit_only_and_keeps_optimizer_disabled() -> None:
    audit = ad_contract_audit()
    assert audit["status"] == "PASS_AUDIT_ONLY_NOT_GRADIENT"
    assert audit["production_gradient_validated"] is False
    assert audit["optimizer_enabled"] is False
    source = inspect.getsource(fdtdx_parity_ad_contract)
    assert "Q_clipping_allowed" in source
    assert "rho**3" not in source and "rho ** 3" not in source
