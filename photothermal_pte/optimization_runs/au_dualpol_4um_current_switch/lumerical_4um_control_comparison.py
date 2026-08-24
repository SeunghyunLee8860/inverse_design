"""Fail-closed comparison of two Lumerical exact-control result bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_mesh_contract import (
    RELATIVE_GATE,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    sha256,
)


ENDPOINT_KEYS = tuple(
    f"endpoint_field_E{axis}_V_m" for axis in ("x", "y", "z")
)
FIXED_MESH_KEYS = (
    "flake_dxy_m",
    "outer_dxy_m",
    "mesh_accuracy",
    "pml_layers",
    "lateral_span_m",
    "z_min_m",
    "z_max_m",
    "simulation_time_s",
    "auto_shutoff_min",
    "conformal_mesh",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} does not contain a JSON object")
    return payload


def _raw_npz(payload: dict[str, Any], result_path: Path) -> tuple[Path, str]:
    records = [
        item
        for item in payload.get("raw_artifacts", [])
        if isinstance(item, dict) and str(item.get("path", "")).endswith("_raw.npz")
    ]
    if len(records) != 1:
        raise RuntimeError(f"{result_path} must name exactly one raw NPZ artifact")
    raw_path = Path(str(records[0]["path"])).resolve()
    expected_sha = str(records[0].get("sha256", ""))
    if not raw_path.is_file():
        raise RuntimeError(f"raw NPZ is absent: {raw_path}")
    actual_sha = sha256(raw_path)
    if not expected_sha or actual_sha != expected_sha:
        raise RuntimeError(
            f"raw NPZ SHA mismatch for {raw_path}: {actual_sha} != {expected_sha}"
        )
    return raw_path, actual_sha


def _require_matching_contract(
    coarse: dict[str, Any], fine: dict[str, Any]
) -> dict[str, Any]:
    for label, payload in (("coarse", coarse), ("fine", fine)):
        if not bool(payload.get("all_gates_passed")):
            raise RuntimeError(f"{label} result did not pass all solver gates")
        if not str(payload.get("status", "")).startswith("PASSED_"):
            raise RuntimeError(f"{label} result status is not passed")
        validation = payload.get("source_calibration_validation")
        if not isinstance(validation, dict) or not bool(validation.get("passed")):
            raise RuntimeError(f"{label} result lacks a passed source calibration")

    matching_paths = (
        ("case",),
        ("case_label",),
        ("polarization",),
        ("accelerator_policy",),
        ("B200_promotion_certified",),
        ("solver_version",),
        ("GPU_log_evidence", "requested_gpu_uuid"),
        ("layout", "geometry", "exact_au_geometry", "geometry_sha256"),
    )

    def nested(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
        value: Any = payload
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                raise RuntimeError(f"missing required result field: {'.'.join(keys)}")
            value = value[key]
        return value

    for keys in matching_paths:
        first = nested(coarse, keys)
        second = nested(fine, keys)
        if first != second:
            raise RuntimeError(
                f"coarse/fine mismatch at {'.'.join(keys)}: {first!r} != {second!r}"
            )

    coarse_mesh = coarse.get("mesh_spec")
    fine_mesh = fine.get("mesh_spec")
    if not isinstance(coarse_mesh, dict) or not isinstance(fine_mesh, dict):
        raise RuntimeError("coarse/fine result lacks mesh_spec")
    for key in FIXED_MESH_KEYS:
        if coarse_mesh.get(key) != fine_mesh.get(key):
            raise RuntimeError(
                f"comparison changes non-z mesh axis {key}: "
                f"{coarse_mesh.get(key)!r} != {fine_mesh.get(key)!r}"
            )
    for key in ("stack_dz_m", "bulk_dz_m"):
        if not float(fine_mesh[key]) < float(coarse_mesh[key]):
            raise RuntimeError(f"fine {key} must be strictly smaller than coarse")
    return {
        "case": coarse["case"],
        "polarization": coarse["polarization"],
        "accelerator_policy": coarse["accelerator_policy"],
        "B200_promotion_certified": coarse["B200_promotion_certified"],
        "solver_version": coarse["solver_version"],
        "requested_gpu_uuid": coarse["GPU_log_evidence"]["requested_gpu_uuid"],
        "geometry_sha256": coarse["layout"]["geometry"]["exact_au_geometry"][
            "geometry_sha256"
        ],
        "coarse_mesh_spec": coarse_mesh,
        "fine_mesh_spec": fine_mesh,
    }


def _fine_relative(coarse: float, fine: float) -> float:
    return abs(coarse - fine) / max(abs(fine), np.finfo(float).tiny)


def compare_control_pair(coarse_json: Path, fine_json: Path) -> dict[str, Any]:
    """Compare source-normalized scalar and endpoint-field results.

    The finer result is the denominator for every relative metric.  Complex
    fields are compared directly, without arbitrary phase alignment.
    """

    coarse_path = Path(coarse_json).resolve()
    fine_path = Path(fine_json).resolve()
    coarse = _load_json(coarse_path)
    fine = _load_json(fine_path)
    contract = _require_matching_contract(coarse, fine)
    coarse_npz_path, coarse_npz_sha = _raw_npz(coarse, coarse_path)
    fine_npz_path, fine_npz_sha = _raw_npz(fine, fine_path)

    coarse_incident = float(
        coarse["reporting_normalization"]["source_only_incident_power_W_raw"]
    )
    fine_incident = float(
        fine["reporting_normalization"]["source_only_incident_power_W_raw"]
    )
    if not (
        np.isfinite(coarse_incident)
        and np.isfinite(fine_incident)
        and coarse_incident > 0.0
        and fine_incident > 0.0
    ):
        raise RuntimeError("source-only incident powers must be positive and finite")

    with np.load(coarse_npz_path, allow_pickle=False) as coarse_raw, np.load(
        fine_npz_path, allow_pickle=False
    ) as fine_raw:
        for key in ("endpoint_field_x_m", "endpoint_field_y_m"):
            first = np.asarray(coarse_raw[key], dtype=np.float64)
            second = np.asarray(fine_raw[key], dtype=np.float64)
            if first.shape != second.shape or not np.allclose(
                first, second, rtol=0.0, atol=2.0e-18
            ):
                raise RuntimeError(f"endpoint coordinate mismatch at {key}")
        coarse_field = np.concatenate(
            [
                (np.asarray(coarse_raw[key]) / np.sqrt(coarse_incident)).ravel()
                for key in ENDPOINT_KEYS
            ]
        )
        fine_field = np.concatenate(
            [
                (np.asarray(fine_raw[key]) / np.sqrt(fine_incident)).ravel()
                for key in ENDPOINT_KEYS
            ]
        )
        coarse_e2 = (
            np.asarray(coarse_raw["endpoint_field_E2_V2_m2"], dtype=np.float64)
            / coarse_incident
        )
        fine_e2 = (
            np.asarray(fine_raw["endpoint_field_E2_V2_m2"], dtype=np.float64)
            / fine_incident
        )
    if not all(
        np.all(np.isfinite(value))
        for value in (coarse_field, fine_field, coarse_e2, fine_e2)
    ):
        raise RuntimeError("endpoint field bundle contains NaN or Inf")

    coarse_q = float(coarse["P_Q_native_W_raw"]) / coarse_incident
    fine_q = float(fine["P_Q_native_W_raw"]) / fine_incident
    coarse_flux = float(coarse["P_six_face_W_raw"]) / coarse_incident
    fine_flux = float(fine["P_six_face_W_raw"]) / fine_incident
    metrics = {
        "source_normalized_Q_change_relative": _fine_relative(coarse_q, fine_q),
        "source_normalized_flux_change_relative": _fine_relative(
            coarse_flux, fine_flux
        ),
        "source_normalized_complex_E_NRMSE": float(
            np.linalg.norm(coarse_field - fine_field)
            / max(np.linalg.norm(fine_field), np.finfo(float).tiny)
        ),
        "source_normalized_E2_NRMSE": float(
            np.linalg.norm(coarse_e2 - fine_e2)
            / max(np.linalg.norm(fine_e2), np.finfo(float).tiny)
        ),
    }
    gates = {key: value < RELATIVE_GATE for key, value in metrics.items()}
    passed = all(gates.values())
    return {
        "schema": "lumerical-4um-control-pair-comparison-v1",
        "status": (
            "PASSED_LUMERICAL_4UM_CONTROL_PAIR_MAXWELL_SUBGATE"
            if passed
            else "BLOCKED_LUMERICAL_4UM_CONTROL_PAIR_MAXWELL_SUBGATE"
        ),
        "contract": contract,
        "normalization": {
            "coarse_source_only_incident_power_W_raw": coarse_incident,
            "fine_source_only_incident_power_W_raw": fine_incident,
            "field_amplitude_divisor": "sqrt(source-only incident power)",
            "scalar_and_E2_divisor": "source-only incident power",
            "relative_denominator": "finer result",
            "complex_phase_alignment": False,
        },
        "metrics": metrics,
        "gate_limit_relative": RELATIVE_GATE,
        "gates": gates,
        "all_gates_passed": passed,
        "artifacts": {
            "coarse_json": str(coarse_path),
            "fine_json": str(fine_path),
            "coarse_raw_npz": str(coarse_npz_path),
            "fine_raw_npz": str(fine_npz_path),
            "coarse_raw_npz_sha256": coarse_npz_sha,
            "fine_raw_npz_sha256": fine_npz_sha,
        },
        "scope": (
            "Maxwell scalar and common endpoint-plane sub-gate only; not a "
            "volumetric-Q remap, thermal/current, both-polarization, all-control, "
            "final-topology, or B200 production certificate"
        ),
    }
