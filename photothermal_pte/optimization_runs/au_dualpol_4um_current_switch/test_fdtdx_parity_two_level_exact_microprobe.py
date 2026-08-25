from __future__ import annotations

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_two_level_exact_microprobe import (
    two_level_exact_source_audit,
)


def test_two_level_exact_microprobe_source_audit_fails_closed() -> None:
    audit = two_level_exact_source_audit()
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert len(audit["source_sha256"]) == 2
