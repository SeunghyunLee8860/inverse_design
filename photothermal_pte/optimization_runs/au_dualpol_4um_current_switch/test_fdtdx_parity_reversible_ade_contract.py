from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_ade import (
    coefficients_numpy,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_reversible_ade_contract import (
    forward_diagonal_c4_free_ade,
    pinned_reversible_source_audit,
    reverse_diagonal_c4_free_ade,
    target_reverse_gain_audit,
)


def _target_like_state(dtype=np.float32):
    rng = np.random.default_rng(20260825)
    E_prev = rng.normal(scale=1.0e-2, size=(3, 2, 2, 2)).astype(dtype)
    P_curr = rng.normal(scale=1.0e-3, size=(3, 3, 2, 2, 2)).astype(dtype)
    P_prev = rng.normal(scale=1.0e-3, size=(3, 3, 2, 2, 2)).astype(dtype)
    inv_eps = rng.uniform(0.1, 1.0, size=(3, 2, 2, 2)).astype(dtype)
    drive = rng.normal(scale=1.0e-3, size=(3, 2, 2, 2)).astype(dtype)
    c1_scalar, c2_scalar, c3_scalar, _ = coefficients_numpy(np.float32(0.6))
    c1 = np.broadcast_to(c1_scalar[:, None, None, None, None], P_curr.shape).astype(dtype)
    c2 = np.broadcast_to(c2_scalar[:, None, None, None, None], P_curr.shape).astype(dtype)
    c3 = np.broadcast_to(c3_scalar[:, None, None, None, None], P_curr.shape).astype(dtype)
    return E_prev, P_curr, P_prev, inv_eps, drive, c1, c2, c3


def test_pinned_reversible_path_cannot_be_enabled_by_guard_removal() -> None:
    audit = pinned_reversible_source_audit()
    assert audit["status"] == "PASS"
    assert audit["guard_removal_is_valid_implementation"] is False
    assert set(audit["missing_wrapper_state"]) == {
        "dispersive_P_curr",
        "dispersive_P_prev",
        "dispersive_c1",
        "dispersive_c2",
        "dispersive_c3",
    }


def test_target_diagonal_ade_single_step_round_trip() -> None:
    E_prev, P_curr, P_prev, inv_eps, drive, c1, c2, c3 = _target_like_state()
    E_next, P_next, carried_P = forward_diagonal_c4_free_ade(
        E_prev=E_prev,
        P_curr=P_curr,
        P_prev=P_prev,
        inv_eps=inv_eps,
        courant_curl_H=drive,
        c1=c1,
        c2=c2,
        c3=c3,
    )
    recovered_E, recovered_P_curr, recovered_P_prev = reverse_diagonal_c4_free_ade(
        E_next=E_next,
        P_next=P_next,
        P_curr=carried_P,
        inv_eps=inv_eps,
        courant_curl_H=drive,
        c1=c1,
        c2=c2,
        c3=c3,
    )
    np.testing.assert_allclose(recovered_E, E_prev, rtol=2.0e-6, atol=2.0e-9)
    np.testing.assert_allclose(recovered_P_curr, P_curr, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        recovered_P_prev, P_prev, rtol=2.0e-5, atol=2.0e-9
    )


def test_zero_c2_padding_is_safe_but_nonzero_residual_fails_closed() -> None:
    E_prev, P_curr, P_prev, inv_eps, drive, c1, c2, c3 = _target_like_state()
    c1 = c1.copy()
    c2 = c2.copy()
    c3 = c3.copy()
    P_curr = P_curr.copy()
    P_prev = P_prev.copy()
    c1[2, :, 0] = 0
    c2[2, :, 0] = 0
    c3[2, :, 0] = 0
    P_curr[2, :, 0] = 0
    P_prev[2, :, 0] = 0
    E_next, P_next, carried_P = forward_diagonal_c4_free_ade(
        E_prev=E_prev,
        P_curr=P_curr,
        P_prev=P_prev,
        inv_eps=inv_eps,
        courant_curl_H=drive,
        c1=c1,
        c2=c2,
        c3=c3,
    )
    _, _, recovered = reverse_diagonal_c4_free_ade(
        E_next=E_next,
        P_next=P_next,
        P_curr=carried_P,
        inv_eps=inv_eps,
        courant_curl_H=drive,
        c1=c1,
        c2=c2,
        c3=c3,
    )
    assert np.count_nonzero(recovered[2, :, 0]) == 0
    corrupted = P_next.copy()
    corrupted[2, 0, 0, 0, 0] = np.float32(1.0e-6)
    with pytest.raises(ValueError, match="zero-c2"):
        reverse_diagonal_c4_free_ade(
            E_next=E_next,
            P_next=corrupted,
            P_curr=carried_P,
            inv_eps=inv_eps,
            courant_curl_H=drive,
            c1=c1,
            c2=c2,
            c3=c3,
        )


def test_target_inverse_damping_requires_sliced_exact_resets() -> None:
    audit = target_reverse_gain_audit()
    assert audit["status"] == "PASS_REQUIRES_SLICED_RESETS"
    assert audit["maximum_single_step_inverse_c2_magnitude"] > 1.0
    assert audit["unsliced_256163_step_reverse_allowed"] is False
    assert audit["sliced_exact_state_resets_required"] is True
    assert max(
        row["conservative_4096_step_power"]
        for row in audit["coefficients"].values()
    ) > 1000.0
