from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    TimeSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_exact_binary_control import (
    EXPECTED_FDTDX_COMMIT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_user_balanced_source_only as source_only,
    fdtdx_user_balanced_source_pair as source_pair,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(tmp_path: Path, polarization: str, power: float) -> tuple[Path, str]:
    raw = tmp_path / f"{polarization}.npz"
    np.savez_compressed(raw, target=np.ones((3, 2, 2, 1), dtype=np.complex64))
    time = TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5)
    numerical_case = source_only.balanced_case_contract(time)
    source_contract = {
        "wavelength_m": 4.0e-6,
        "polarization": polarization,
        "fixed_E_polarization_vector": (
            [0.0, 1.0, 0.0] if polarization == "Ea" else [1.0, 0.0, 0.0]
        ),
    }
    payload = {
        "version": source_only.VERSION,
        "status": source_only.STATUS_READY,
        "ready": True,
        "failed_checks": [],
        "scope": source_pair.CASE_SCOPE,
        "polarization": polarization,
        "numerical_case_contract": numerical_case,
        "mesh": numerical_case["mesh"],
        "time_contract": {
            **numerical_case["time"],
            "time_step_s": 1.0,
            "time_steps_total": 100,
        },
        "pml_face_parameters": {"same": True},
        "placement": {"same": True},
        "source_contract": source_contract,
        "all_air_material_readback": {"ready": True},
        "evaluation": {
            "ready": True,
            "gates": {"synthetic_gate": True},
            "flux": {"incident_plane_signed_W": power},
        },
        "reporting_incident_power_W": 285.0e-6,
        "raw": {
            "path": str(raw.resolve()),
            "sha256": _sha256(raw),
            "arrays": {"target": [3, 2, 2, 1]},
        },
        "runtime_lock": {"same": True},
        "provenance": {
            "repository_commit": "synthetic-clean-commit",
            "repository_dirty_porcelain_before": "",
            "repository_dirty_porcelain_after": "",
            "fdtdx_source": {
                "commit": EXPECTED_FDTDX_COMMIT,
                "dirty_porcelain": "",
            },
            "runner_sha256": "same-runner-sha",
        },
        "checks": {
            "increment_state_selected": True,
            "per_case_scaling_not_applied": True,
            "synthetic_provenance": True,
        },
    }
    report = tmp_path / f"{polarization}.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    return report.resolve(), _sha256(report)


def test_pair_uses_one_common_scale_and_validates_all_bytes(tmp_path: Path) -> None:
    ea, ea_hash = _report(tmp_path, "Ea", 1.000e-12)
    eb, eb_hash = _report(tmp_path, "Eb", 1.002e-12)

    result = source_pair.build_pair(ea, ea_hash, eb, eb_hash)

    assert result["ready"] is True
    assert result["failed_gates"] == []
    expected_scale = 285.0e-6 / 1.001e-12
    assert np.isclose(
        result["common_normalization"]["common_power_scale"], expected_scale
    )
    assert result["normalization_policy"]["per_polarization_power_matching_forbidden"]
    assert all(case["raw"]["ready"] for case in result["cases"].values())

    certificate = tmp_path / "pair.json"
    certificate.write_text(json.dumps(result), encoding="utf-8")
    _, audit = source_pair.validate_source_pair(
        certificate.resolve(),
        _sha256(certificate),
        TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5),
    )
    assert audit["ready"] is True


def test_pair_blocks_per_polarization_power_mismatch(tmp_path: Path) -> None:
    ea, ea_hash = _report(tmp_path, "Ea", 1.00e-12)
    eb, eb_hash = _report(tmp_path, "Eb", 1.02e-12)

    result = source_pair.build_pair(ea, ea_hash, eb, eb_hash)

    assert result["ready"] is False
    assert "source_power_relative_mismatch" in result["failed_gates"]


def test_pair_revalidation_detects_raw_tampering(tmp_path: Path) -> None:
    ea, ea_hash = _report(tmp_path, "Ea", 1.000e-12)
    eb, eb_hash = _report(tmp_path, "Eb", 1.002e-12)
    result = source_pair.build_pair(ea, ea_hash, eb, eb_hash)
    certificate = tmp_path / "pair.json"
    certificate.write_text(json.dumps(result), encoding="utf-8")
    with Path(result["cases"]["Ea"]["raw"]["path"]).open("ab") as stream:
        stream.write(b"tamper")

    _, audit = source_pair.validate_source_pair(
        certificate.resolve(),
        _sha256(certificate),
        TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5),
    )

    assert audit["ready"] is False
    assert "Ea_raw_bytes_match" in audit["failed_checks"]
