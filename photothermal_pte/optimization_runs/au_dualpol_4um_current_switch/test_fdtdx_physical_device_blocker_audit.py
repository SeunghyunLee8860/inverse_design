from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_physical_device_blocker_audit as device_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.production_readiness import (
    REQUIRED_DEVICE_CONFIRMATIONS,
)


@pytest.fixture(scope="module")
def rectangular_report() -> dict[str, object]:
    return device_audit.rectangular_sign_audit()


def test_current_physical_device_contract_is_explicitly_blocked() -> None:
    path = Path(device_audit.__file__).resolve().parent / "physical_device_contract.json"
    report = device_audit.device_contract_audit(path)
    assert report["ready"] is True
    assert report["status"] == device_audit.DEVICE_STATUS
    assert report["confirmed_count"] == 0
    assert set(report["unconfirmed"]) == set(REQUIRED_DEVICE_CONFIRMATIONS)
    assert all(report["checks"].values())


def test_rectangular_sign_algebra_passes_only_its_declared_scope(
    rectangular_report,
) -> None:
    assert rectangular_report["ready"] is True
    assert rectangular_report["scope"] == (
        "implemented rectangular Ta-only thin-sheet diagnostic"
    )
    assert rectangular_report["weighting_potential_max_abs_error"] < 2.0e-10
    assert rectangular_report["current_x_A"] < 0.0
    assert rectangular_report["terminal_swapped_current_x_A"] > 0.0
    assert all(rectangular_report["checks"].values())
    json.dumps(rectangular_report, allow_nan=False)


def test_paper_audit_does_not_promote_historical_device_A(tmp_path: Path) -> None:
    papers = tmp_path / "papers"
    papers.mkdir()
    main = papers / device_audit.LOCAL_MAIN_PAPER
    main.write_bytes(b"current-main-paper")
    repository = tmp_path / "repository"
    historical = repository / device_audit.HISTORICAL_PAPER_CONTRACT
    historical.parent.mkdir(parents=True)
    historical.write_text(
        json.dumps(
            {
                "sources": {
                    "main_paper": {
                        "path": "/missing/old-main.pdf",
                        "sha256": hashlib.sha256(b"old-main").hexdigest(),
                    },
                    "supporting_information": {
                        "path": "/missing/old-si.pdf",
                        "sha256": hashlib.sha256(b"old-si").hexdigest(),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    report = device_audit.paper_evidence_audit(papers, repository)
    assert report["local_main"]["exists"] is True
    assert report["local_supplement"]["exists"] is False
    assert report["paper_equation_basis_complete_in_current_papers_root"] is False
    assert report["historical_device_A_contract"][
        "embedded_paths_are_currently_available"
    ] is False
    assert report["historical_device_A_contract"][
        "local_main_matches_embedded_bytes"
    ] is False
    assert report["historical_device_A_is_not_target_device_authority"] is True
