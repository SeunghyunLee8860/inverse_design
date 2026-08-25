from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_blockwise_exact_microprobe import (
    MAX_DIRECTIONAL_RELATIVE_ERROR,
    blockwise_exact_connectivity_gate,
    blockwise_exact_source_audit,
)


def test_blockwise_exact_microprobe_source_audit_fails_closed() -> None:
    audit = blockwise_exact_source_audit()
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert len(audit["source_sha256"]) == 2


def test_blockwise_exact_connectivity_gate_passes_accurate_gradient() -> None:
    status, gates = blockwise_exact_connectivity_gate(
        value=1.0e-4,
        gradient=np.asarray([[1.0e-3, -2.0e-3]]),
        ad_directional=2.999,
        fd_directional=3.0,
        value_and_grad_seconds=10.0,
    )
    assert status == "PASS_BOUNDED_BLOCKWISE_EXACT_AD_CONNECTIVITY_ONLY"
    assert gates["directional_relative_error"] < MAX_DIRECTIONAL_RELATIVE_ERROR


def test_blockwise_exact_connectivity_gate_blocks_bad_gradient() -> None:
    status, gates = blockwise_exact_connectivity_gate(
        value=1.0e-4,
        gradient=np.zeros((2, 2)),
        ad_directional=1.0,
        fd_directional=-1.0,
        value_and_grad_seconds=30.0 * 60.0,
    )
    assert status == "BLOCKED"
    assert gates["nonzero_gradient"] is False
    assert gates["same_directional_sign"] is False
    assert gates["bounded_measured_value_and_grad_runtime"] is False
