from __future__ import annotations

import unittest

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_increment_state_jax_preflight import (
    late_window_bounds,
    summarize_precision,
)


class IncrementStateJaxPreflightTests(unittest.TestCase):
    def test_late_window_bounds_cover_last_sixteen_periods(self) -> None:
        period = 2.0
        time_s = np.arange(0.0, 64.0, 0.1)
        bounds = late_window_bounds(time_s, period)
        times = time_s[bounds[:, 0]]
        self.assertTrue(np.allclose(times, [32.0, 40.0, 48.0, 56.0]))
        self.assertEqual(bounds.shape, (4, 2))

    def test_precision_summary_passes_identical_settled_windows(self) -> None:
        values = np.asarray([2.0 + 1.0j] * 4)
        result = summarize_precision(values, values)
        self.assertTrue(result["ready"])
        self.assertTrue(all(result["gates"].values()))

    def test_precision_summary_fails_float32_drift(self) -> None:
        float32 = np.asarray([1.0, 1.0, 1.0, 1.1], dtype=np.complex128)
        float64 = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.complex128)
        result = summarize_precision(float32, float64)
        self.assertFalse(result["ready"])
        self.assertFalse(result["gates"]["float32_last_window_settled"])

    def test_invalid_window_vector_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            late_window_bounds(np.zeros((2, 2)), 1.0)
        with self.assertRaises(ValueError):
            summarize_precision(np.ones(3), np.ones(3))


if __name__ == "__main__":
    unittest.main()
