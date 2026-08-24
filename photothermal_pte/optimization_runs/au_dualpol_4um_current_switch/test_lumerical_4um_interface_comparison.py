from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_interface_comparison import (
    compare_normalized_maxwell,
    normalized_maxwell_bundle,
)


def _result(incident: float) -> dict[str, object]:
    return {
        "reporting_normalization": {
            "source_only_incident_power_W_raw": incident,
        },
        "P_Q_native_W_raw": 2.0 * incident,
        "P_six_face_W_raw": 3.0 * incident,
    }


def _raw(field_scale: float) -> dict[str, np.ndarray]:
    return {
        "endpoint_field_x_m": np.asarray((0.0, 1.0)),
        "endpoint_field_y_m": np.asarray((0.0, 1.0)),
        "endpoint_field_Ex_V_m": np.full((2, 2), field_scale, complex),
        "endpoint_field_Ey_V_m": np.full((2, 2), 2.0 * field_scale, complex),
        "endpoint_field_Ez_V_m": np.full((2, 2), 3.0 * field_scale, complex),
        "endpoint_field_E2_V2_m2": np.full((2, 2), 14.0 * field_scale**2),
    }


def test_normalization_removes_source_amplitude() -> None:
    first = normalized_maxwell_bundle(_result(1.0), _raw(1.0))
    second = normalized_maxwell_bundle(_result(4.0), _raw(2.0))
    metrics, gates = compare_normalized_maxwell(first, second)
    assert all(value == pytest.approx(0.0) for value in metrics.values())
    assert all(gates.values())


def test_complex_field_comparison_does_not_align_global_phase() -> None:
    reference = normalized_maxwell_bundle(_result(1.0), _raw(1.0))
    candidate = normalized_maxwell_bundle(_result(1.0), _raw(1.0))
    candidate["field"] = 1j * candidate["field"]
    metrics, gates = compare_normalized_maxwell(candidate, reference)
    assert metrics["source_normalized_complex_E_NRMSE"] == pytest.approx(np.sqrt(2.0))
    assert gates["source_normalized_complex_E_NRMSE"] is False
