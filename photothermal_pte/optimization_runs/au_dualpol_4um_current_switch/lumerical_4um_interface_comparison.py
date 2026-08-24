"""Pure comparisons for the Lumerical CV0/CV1/staircase interface triage."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_control_comparison import (
    ENDPOINT_KEYS,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_mesh_contract import (
    RELATIVE_GATE,
)


INTERFACE_METHODS = {
    "cv0": "conformal variant 0",
    "cv1": "conformal variant 1",
    "staircase": "staircase",
}


def normalized_maxwell_bundle(
    result: Mapping[str, Any],
    raw: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    incident = float(
        result["reporting_normalization"]["source_only_incident_power_W_raw"]
    )
    if not np.isfinite(incident) or incident <= 0.0:
        raise RuntimeError("source-only incident power must be positive and finite")
    field = np.concatenate(
        [
            (np.asarray(raw[key]) / np.sqrt(incident)).ravel()
            for key in ENDPOINT_KEYS
        ]
    )
    e2 = (
        np.asarray(raw["endpoint_field_E2_V2_m2"], dtype=np.float64) / incident
    )
    if not np.all(np.isfinite(field)) or not np.all(np.isfinite(e2)):
        raise RuntimeError("normalized endpoint bundle contains NaN or Inf")
    return {
        "source_incident_power_W_raw": incident,
        "Q_over_incident": float(result["P_Q_native_W_raw"]) / incident,
        "flux_over_incident": float(result["P_six_face_W_raw"]) / incident,
        "field": field,
        "E2": e2,
        "x_m": np.asarray(raw["endpoint_field_x_m"], dtype=np.float64),
        "y_m": np.asarray(raw["endpoint_field_y_m"], dtype=np.float64),
    }


def compare_normalized_maxwell(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, bool]]:
    for axis in ("x_m", "y_m"):
        first = np.asarray(candidate[axis])
        second = np.asarray(reference[axis])
        if first.shape != second.shape or not np.allclose(
            first, second, rtol=0.0, atol=2.0e-18
        ):
            raise RuntimeError(f"endpoint coordinate mismatch at {axis}")
    field = np.asarray(candidate["field"])
    reference_field = np.asarray(reference["field"])
    e2 = np.asarray(candidate["E2"])
    reference_e2 = np.asarray(reference["E2"])
    if field.shape != reference_field.shape or e2.shape != reference_e2.shape:
        raise RuntimeError("endpoint field shapes differ")

    def relative(value: float, target: float) -> float:
        return abs(value - target) / max(abs(target), np.finfo(float).tiny)

    metrics = {
        "source_normalized_Q_change_relative": relative(
            float(candidate["Q_over_incident"]),
            float(reference["Q_over_incident"]),
        ),
        "source_normalized_flux_change_relative": relative(
            float(candidate["flux_over_incident"]),
            float(reference["flux_over_incident"]),
        ),
        "source_normalized_complex_E_NRMSE": float(
            np.linalg.norm(field - reference_field)
            / max(np.linalg.norm(reference_field), np.finfo(float).tiny)
        ),
        "source_normalized_E2_NRMSE": float(
            np.linalg.norm(e2 - reference_e2)
            / max(np.linalg.norm(reference_e2), np.finfo(float).tiny)
        ),
    }
    return metrics, {key: value < RELATIVE_GATE for key, value in metrics.items()}
