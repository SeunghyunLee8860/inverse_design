from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_control_comparison import (
    compare_control_pair,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result(
    tmp_path: Path,
    label: str,
    dz: float,
    scale: float,
    *,
    dxy: float = 100e-9,
) -> Path:
    raw = tmp_path / f"{label}_raw.npz"
    x = np.asarray((-1.0, 1.0))
    y = np.asarray((-2.0, 2.0))
    ex = np.full((2, 2), scale + 1j * 0.5 * scale)
    ey = np.full((2, 2), 2.0 * scale - 1j * scale)
    ez = np.full((2, 2), 0.25 * scale + 1j * 0.1 * scale)
    np.savez_compressed(
        raw,
        endpoint_field_x_m=x,
        endpoint_field_y_m=y,
        endpoint_field_Ex_V_m=ex,
        endpoint_field_Ey_V_m=ey,
        endpoint_field_Ez_V_m=ez,
        endpoint_field_E2_V2_m2=np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2,
    )
    mesh = {
        "label": label,
        "flake_dxy_m": dxy,
        "stack_dz_m": dz,
        "bulk_dz_m": 10.0 * dz,
        "outer_dxy_m": 200e-9,
        "mesh_accuracy": 3,
        "pml_layers": 8,
        "lateral_span_m": 20e-6,
        "z_min_m": -3e-6,
        "z_max_m": 3e-6,
        "simulation_time_s": 1e-12,
        "auto_shutoff_min": 1e-7,
        "conformal_mesh": "conformal variant 0",
    }
    incident = scale**2
    payload = {
        "status": "PASSED_PROVISIONAL_LUMERICAL_CONTROL",
        "all_gates_passed": True,
        "case": "full",
        "case_label": "full",
        "polarization": "Ea",
        "accelerator_policy": "development",
        "B200_promotion_certified": False,
        "solver_version": "test",
        "GPU_log_evidence": {"requested_gpu_uuid": "GPU-test"},
        "layout": {
            "geometry": {
                "exact_au_geometry": {"geometry_sha256": "geometry-test"}
            }
        },
        "mesh_spec": mesh,
        "source_calibration_validation": {"passed": True},
        "reporting_normalization": {
            "source_only_incident_power_W_raw": incident
        },
        "P_Q_native_W_raw": 0.2 * incident,
        "P_six_face_W_raw": 0.2 * incident,
        "raw_artifacts": [
            {"path": str(raw), "sha256": _sha256(raw)}
        ],
    }
    result = tmp_path / f"{label}.json"
    result.write_text(json.dumps(payload), encoding="utf-8")
    return result


def test_source_normalization_removes_common_amplitude_change(tmp_path: Path) -> None:
    coarse = _result(tmp_path, "coarse", 2.0e-9, 2.0)
    fine = _result(tmp_path, "fine", 1.0e-9, 3.0)
    result = compare_control_pair(coarse, fine)
    assert result["all_gates_passed"] is True
    assert all(value == pytest.approx(0.0, abs=2e-16) for value in result["metrics"].values())
    assert result["normalization"]["complex_phase_alignment"] is False


def test_comparison_rejects_a_non_z_axis_change(tmp_path: Path) -> None:
    coarse = _result(tmp_path, "coarse", 2.0e-9, 2.0)
    fine = _result(tmp_path, "fine", 1.0e-9, 2.0)
    payload = json.loads(fine.read_text(encoding="utf-8"))
    payload["mesh_spec"]["pml_layers"] = 12
    fine.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fixed z mesh field pml_layers"):
        compare_control_pair(coarse, fine)


def test_comparison_rejects_raw_artifact_tampering(tmp_path: Path) -> None:
    coarse = _result(tmp_path, "coarse", 2.0e-9, 2.0)
    fine = _result(tmp_path, "fine", 1.0e-9, 2.0)
    payload = json.loads(fine.read_text(encoding="utf-8"))
    payload["raw_artifacts"][0]["sha256"] = "0" * 64
    fine.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="raw NPZ SHA mismatch"):
        compare_control_pair(coarse, fine)


def test_xy_comparison_allows_only_strict_flake_mesh_refinement(
    tmp_path: Path,
) -> None:
    coarse = _result(tmp_path, "coarse", 2.5e-9, 2.0, dxy=100e-9)
    fine = _result(tmp_path, "fine", 2.5e-9, 3.0, dxy=50e-9)
    result = compare_control_pair(coarse, fine, refinement_axis="xy")
    assert result["all_gates_passed"] is True
    assert result["contract"]["refinement_axis"] == "xy"


def test_xy_comparison_rejects_a_z_change(tmp_path: Path) -> None:
    coarse = _result(tmp_path, "coarse", 2.5e-9, 2.0, dxy=100e-9)
    fine = _result(tmp_path, "fine", 1.25e-9, 2.0, dxy=50e-9)
    with pytest.raises(RuntimeError, match="fixed xy mesh field stack_dz_m"):
        compare_control_pair(coarse, fine, refinement_axis="xy")
