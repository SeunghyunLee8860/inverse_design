from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_anchor_placement import (
    expected_placement,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_increment_state_source_only as source_only,
    fdtdx_increment_state_source_pair as source_pair,
    fdtdx_increment_state_source_pair_validation as pair_validation,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    ANCHOR_CASE,
    TimeSpec,
    case_for_axis,
    case_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_exact_binary_control import (
    EXPECTED_FDTDX_COMMIT,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(tmp_path: Path, polarization: str, power: float) -> tuple[Path, str]:
    raw = tmp_path / f"{polarization}.npz"
    np.savez_compressed(raw, target=np.ones((3, 2, 2, 1), dtype=np.complex64))
    spec = case_for_axis(
        "full_domain_z",
        0,
        time=TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5),
        pml_alpha_scale=ANCHOR_CASE.pml_alpha_scale,
        pml_target_reflection=ANCHOR_CASE.pml_target_reflection,
    )
    contract = case_contract(spec)
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
        "scope": source_only.SCOPE,
        "polarization": polarization,
        "numerical_case_contract": contract,
        "mesh": contract["resolved_mesh"],
        "time_contract": {
            **contract["time_spec"],
            "time_step_s": 1.0,
            "time_steps_total": 1,
        },
        "pml_face_parameters": contract["resolved_pml_face_parameters"],
        "placement": expected_placement(spec.mesh),
        "source_contract": source_contract,
        "all_air_material_readback": {"ready": True},
        "dispersive_state_representation": "increment",
        "evaluation": {
            "ready": True,
            "gates": {"all_test_gates": True},
            "flux": {"incident_plane_signed_W": power},
        },
        "reporting_incident_power_W": 285.0e-6,
        "per_case_scale_not_authorized_until_pair_comparison": True,
        "raw": {
            "path": str(raw),
            "sha256": _sha256(raw),
            "arrays": {"target": [3, 2, 2, 1]},
        },
        "runtime_lock": {"same": True},
        "provenance": {
            "repository_commit": "synthetic-clean-commit",
            "repository_dirty_porcelain": "",
            "fdtdx_source": {
                "commit": EXPECTED_FDTDX_COMMIT,
                "dirty_porcelain": "",
            },
            "runner_sha256": "same-runner-sha",
        },
        "provenance_checks": {"all_test_provenance": True},
    }
    report = tmp_path / f"{polarization}.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    return report, _sha256(report)


def test_pair_uses_one_common_scale_and_validates_raw_hashes(tmp_path):
    ea, ea_hash = _report(tmp_path, "Ea", 1.000e-12)
    eb, eb_hash = _report(tmp_path, "Eb", 1.002e-12)

    result = source_pair.build_pair(ea, ea_hash, eb, eb_hash)

    assert result["ready"] is True
    assert result["failed_gates"] == []
    assert (
        result["normalization_policy"]["per_polarization_power_matching_forbidden"]
        is True
    )
    expected_scale = 285.0e-6 / 1.001e-12
    assert np.isclose(
        result["common_normalization"]["common_power_scale"], expected_scale
    )
    assert all(case["raw"]["ready"] for case in result["cases"].values())


def test_pair_blocks_power_mismatch(tmp_path):
    ea, ea_hash = _report(tmp_path, "Ea", 1.00e-12)
    eb, eb_hash = _report(tmp_path, "Eb", 1.02e-12)

    result = source_pair.build_pair(ea, ea_hash, eb, eb_hash)

    assert result["ready"] is False
    assert "source_power_relative_mismatch" in result["failed_gates"]


def test_source_wrapper_rejects_busy_gpu_before_cuda_export():
    wrapper = (
        Path(source_only.__file__)
        .with_name("run_fdtdx_increment_state_source_gpu.sh")
        .read_text(encoding="utf-8")
    )
    assert wrapper.index("refusing busy GPU") < wrapper.index(
        "export CUDA_VISIBLE_DEVICES"
    )
    assert "--query-compute-apps" in wrapper
    assert "export JAX_PLATFORMS=cuda" in wrapper
    assert '"$@"' in wrapper
    assert "Lumerical" not in wrapper


def test_certificate_revalidates_report_and_raw_bytes(tmp_path):
    ea, ea_hash = _report(tmp_path, "Ea", 1.000e-12)
    eb, eb_hash = _report(tmp_path, "Eb", 1.002e-12)
    pair = source_pair.build_pair(ea, ea_hash, eb, eb_hash)
    certificate = tmp_path / "pair.json"
    certificate.write_text(json.dumps(pair), encoding="utf-8")
    spec = case_for_axis(
        "full_domain_z",
        0,
        time=TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5),
        pml_alpha_scale=ANCHOR_CASE.pml_alpha_scale,
        pml_target_reflection=ANCHOR_CASE.pml_target_reflection,
    )

    _, audit = pair_validation.validate_source_pair(
        certificate, _sha256(certificate), spec
    )
    assert audit["ready"] is True

    ea_raw = Path(pair["cases"]["Ea"]["raw"]["path"])
    with ea_raw.open("ab") as stream:
        stream.write(b"tamper")
    _, tampered = pair_validation.validate_source_pair(
        certificate, _sha256(certificate), spec
    )
    assert tampered["ready"] is False
    assert "Ea_raw_bytes_match" in tampered["failed_checks"]


def test_missing_certificate_fails_closed(tmp_path):
    spec = case_for_axis(
        "full_domain_z",
        0,
        time=TimeSpec(total_periods=24, window_periods=4, courant_factor=0.5),
        pml_alpha_scale=ANCHOR_CASE.pml_alpha_scale,
        pml_target_reflection=ANCHOR_CASE.pml_target_reflection,
    )
    payload, audit = pair_validation.validate_source_pair(
        (tmp_path / "missing.json").resolve(), "0" * 64, spec
    )
    assert payload == {}
    assert audit["ready"] is False
    assert "certificate_exists" in audit["failed_checks"]
    assert "Ea_report_bytes_match" in audit["failed_checks"]
