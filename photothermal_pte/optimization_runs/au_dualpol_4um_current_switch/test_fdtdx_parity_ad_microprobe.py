from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_ad_microprobe import (
    MAX_DIRECTIONAL_RELATIVE_ERROR,
    connectivity_gate,
    symmetric_relative_error,
)


def test_symmetric_relative_error_is_scale_symmetric() -> None:
    assert symmetric_relative_error(3.0, 2.0) == 0.2
    assert symmetric_relative_error(2.0, 3.0) == 0.2
    assert symmetric_relative_error(-3.0, -2.0) == 0.2


def test_connectivity_gate_passes_consistent_nonzero_gradient() -> None:
    status, gates = connectivity_gate(
        value=1.0e-4,
        gradient=np.asarray([[1.0e-3, -2.0e-3]]),
        ad_directional=3.0,
        fd_directional=2.9,
        value_and_grad_seconds=10.0,
    )
    assert status == "PASS_BOUNDED_AD_CONNECTIVITY_ONLY"
    assert gates["directional_relative_error"] < MAX_DIRECTIONAL_RELATIVE_ERROR


def test_connectivity_gate_blocks_direct_only_or_zero_maxwell_gradient() -> None:
    status, gates = connectivity_gate(
        value=1.0e-4,
        gradient=np.zeros((2, 2)),
        ad_directional=0.0,
        fd_directional=0.0,
        value_and_grad_seconds=10.0,
    )
    assert status == "BLOCKED"
    assert gates["nonzero_gradient"] is False
    assert gates["nonzero_directionals"] is False


def test_connectivity_gate_blocks_wrong_sign_or_slow_gradient() -> None:
    status, gates = connectivity_gate(
        value=1.0e-4,
        gradient=np.ones((2, 2)),
        ad_directional=1.0,
        fd_directional=-1.0,
        value_and_grad_seconds=30.0 * 60.0,
    )
    assert status == "BLOCKED"
    assert gates["same_directional_sign"] is False
    assert gates["bounded_value_and_grad_runtime"] is False


def test_connectivity_gate_blocks_nonfinite_outputs() -> None:
    status, gates = connectivity_gate(
        value=float("nan"),
        gradient=np.asarray([float("inf")]),
        ad_directional=float("nan"),
        fd_directional=1.0,
        value_and_grad_seconds=1.0,
    )
    assert status == "BLOCKED"
    assert gates["finite_value"] is False
    assert gates["finite_gradient"] is False
    assert gates["finite_directionals"] is False
