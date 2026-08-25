from __future__ import annotations

from pathlib import Path

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_z_certificate import (
    STATUS_BLOCKED,
    STATUS_CONVERGED,
    _file_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_source_pair import (
    sha256,
)


def test_file_audit_binds_absolute_path_and_bytes(tmp_path: Path) -> None:
    artifact = (tmp_path / "artifact.bin").resolve()
    artifact.write_bytes(b"exact bytes")

    audit = _file_audit(artifact, sha256(artifact))

    assert audit["ready"] is True
    assert all(audit["checks"].values())


def test_file_audit_detects_tampering(tmp_path: Path) -> None:
    artifact = (tmp_path / "artifact.bin").resolve()
    artifact.write_bytes(b"first")
    expected = sha256(artifact)
    artifact.write_bytes(b"second")

    audit = _file_audit(artifact, expected)

    assert audit["ready"] is False
    assert audit["checks"]["sha256_matches"] is False


def test_valid_but_failed_comparison_has_distinct_status() -> None:
    assert STATUS_BLOCKED != STATUS_CONVERGED
    assert STATUS_BLOCKED.startswith("VALIDATED_BLOCKED")
