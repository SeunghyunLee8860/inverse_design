from __future__ import annotations

import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_microbenchmark import (
    estimate_full_forward_seconds,
    fit_phase,
    require_idle_gpu,
)


def test_idle_gpu_guard_requires_exact_empty_uuid() -> None:
    gpu_rows = [
        ["6", "GPU-idle", "NVIDIA B200", "0", "183359", "0"],
        ["7", "GPU-busy", "NVIDIA B200", "100", "183359", "90"],
    ]
    snapshot = require_idle_gpu(
        "GPU-idle", gpu_rows=gpu_rows, process_rows=[]
    )
    assert snapshot["index"] == 6
    assert snapshot["compute_processes"] == []
    with pytest.raises(RuntimeError, match="not idle"):
        require_idle_gpu("GPU-busy", gpu_rows=gpu_rows, process_rows=[])
    with pytest.raises(RuntimeError, match="compute processes"):
        require_idle_gpu(
            "GPU-idle",
            gpu_rows=gpu_rows,
            process_rows=[["GPU-idle", "123", "/other/python", "10"]],
        )


def test_phase_fit_recovers_slope_from_repeated_timings() -> None:
    measurements = []
    for steps in (4, 16, 64):
        for repetition, noise in enumerate((-0.001, 0.001)):
            measurements.append(
                {
                    "phase": "late_window",
                    "steps": steps,
                    "repetition": repetition,
                    "seconds": 0.01 + 0.002 * steps + noise,
                }
            )
    fit = fit_phase(measurements, "late_window")
    assert fit["seconds_per_step"] == pytest.approx(0.002)
    assert fit["intercept_seconds"] == pytest.approx(0.01)


def test_full_forward_estimate_weights_both_detector_windows() -> None:
    fits = {
        "inactive": {"seconds_per_step": 1.0},
        "previous_window": {"seconds_per_step": 2.0},
        "late_window": {"seconds_per_step": 3.0},
    }
    assert estimate_full_forward_seconds(
        fits, total_steps=100, window_steps=10
    ) == 130.0
    with pytest.raises(ValueError, match="shorter"):
        estimate_full_forward_seconds(fits, total_steps=20, window_steps=10)
