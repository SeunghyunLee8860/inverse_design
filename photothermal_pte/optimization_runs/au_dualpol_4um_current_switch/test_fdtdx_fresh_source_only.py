from __future__ import annotations

import unittest

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_only import (
    beam_moments,
    polarization_audit,
    weighted_complex_nrmse,
)


class FdtdxFreshSourceOnlyTest(unittest.TestCase):
    def test_weighted_stationarity_is_zero_only_for_identical_complex_fields(self) -> None:
        late = np.ones((3, 2, 2, 1), dtype=np.complex64) * (1.0 + 2.0j)
        weights = np.ones((2, 2, 1), dtype=np.float64)
        self.assertEqual(weighted_complex_nrmse(late, late.copy(), weights), 0.0)
        changed = late.copy()
        changed[1] *= 1.01
        self.assertGreater(weighted_complex_nrmse(changed, late, weights), 0.0)

    def test_polarization_purity_uses_Ey_for_Ea_and_Ex_for_Eb(self) -> None:
        field = np.zeros((3, 4, 5, 1), dtype=np.complex64)
        field[1] = 2.0 + 1.0j
        weights = np.ones((4, 5, 1))
        ea = polarization_audit(field, "Ea", weights)
        eb = polarization_audit(field, "Eb", weights)
        self.assertEqual(ea["desired_component"], "Ey")
        self.assertEqual(ea["purity"], 1.0)
        self.assertEqual(eb["purity"], 0.0)

    def test_beam_moments_recover_centered_gaussian_waist(self) -> None:
        x = np.linspace(-8.0e-6, 8.0e-6, 801)
        y = x.copy()
        xx, yy = np.meshgrid(x, y, indexing="ij")
        waist = 4.0e-6
        amplitude = np.exp(-(xx**2 + yy**2) / waist**2)
        field = np.zeros((3, x.size, y.size, 1), dtype=np.complex128)
        field[1, :, :, 0] = amplitude
        moments = beam_moments(field, x, y, np.ones_like(amplitude))
        self.assertAlmostEqual(moments["center_x_m"], 0.0, places=15)
        self.assertAlmostEqual(moments["center_y_m"], 0.0, places=15)
        self.assertLess(
            abs(moments["second_moment_waist_x_m"] / waist - 1.0), 1.0e-3
        )
        self.assertLess(
            abs(moments["second_moment_waist_y_m"] / waist - 1.0), 1.0e-3
        )


if __name__ == "__main__":
    unittest.main()
