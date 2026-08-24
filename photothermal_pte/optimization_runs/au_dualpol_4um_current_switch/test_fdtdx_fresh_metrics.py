from __future__ import annotations

import unittest

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_metrics import (
    electric_yee_dual_volumes,
    weighted_complex_nrmse,
)


class _Grid:
    def __init__(self) -> None:
        self._widths = (
            np.asarray([1.0, 2.0]),
            np.asarray([3.0, 5.0]),
            np.asarray([7.0, 11.0]),
        )

    def cell_widths(self, axis: int) -> np.ndarray:
        return self._widths[axis]


class FdtdxFreshMetricsTest(unittest.TestCase):
    def test_electric_dual_volumes_are_component_specific_on_nonuniform_grid(
        self,
    ) -> None:
        volumes = electric_yee_dual_volumes(
            _Grid(), (slice(0, 2), slice(0, 2), slice(0, 2))
        )
        x = np.asarray([1.0, 2.0])
        y = np.asarray([3.0, 5.0])
        z = np.asarray([7.0, 11.0])
        x_dual = np.asarray([1.0, 1.5])
        y_dual = np.asarray([3.0, 4.0])
        z_dual = np.asarray([7.0, 9.0])
        expected = np.stack(
            (
                x[:, None, None] * y_dual[None, :, None] * z_dual[None, None, :],
                x_dual[:, None, None] * y[None, :, None] * z_dual[None, None, :],
                x_dual[:, None, None] * y_dual[None, :, None] * z[None, None, :],
            )
        )
        np.testing.assert_array_equal(volumes, expected)

    def test_nrmse_accepts_component_specific_weights(self) -> None:
        late = np.ones((3, 2, 2, 2), dtype=np.complex128)
        previous = late.copy()
        previous[2, 1, 1, 1] = 0.0
        weights = electric_yee_dual_volumes(
            _Grid(), (slice(0, 2), slice(0, 2), slice(0, 2))
        )
        result = weighted_complex_nrmse(late, previous, weights)
        expected = np.sqrt(weights[2, 1, 1, 1] / np.sum(weights))
        self.assertAlmostEqual(result, expected)

    def test_nrmse_retains_spatial_weight_backwards_compatibility(self) -> None:
        late = np.ones((3, 2, 2, 1), dtype=np.complex64)
        weights = np.ones((2, 2, 1), dtype=np.float64)
        self.assertEqual(weighted_complex_nrmse(late, late.copy(), weights), 0.0)


if __name__ == "__main__":
    unittest.main()
