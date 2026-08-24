from __future__ import annotations

import unittest

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_coefficient_preflight import (
    coefficient_readback,
    contract_matrix,
)


class FdtdxFreshTwoPoleCoefficientPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.axis = {
            "candidate": {
                "poles": [
                    {"c1": 1.5, "c2": -0.5, "c3": 0.25},
                    {"c1": 1.25, "c2": -0.25, "c3": 0.125},
                ]
            }
        }

    def test_contract_matrix_adds_exact_zero_c4(self) -> None:
        expected = contract_matrix(self.axis)
        self.assertEqual(expected.shape, (2, 4))
        self.assertTrue(np.all(expected[:, 3] == 0.0))
        self.assertTrue(coefficient_readback(expected, expected)["exact"])

    def test_readback_rejects_one_ulp_or_shape_change(self) -> None:
        expected = contract_matrix(self.axis)
        changed = expected.copy()
        changed[0, 2] = np.nextafter(changed[0, 2], np.float32(np.inf))
        audit = coefficient_readback(expected, changed)
        self.assertFalse(audit["exact"])
        self.assertGreater(audit["maximum_absolute_error"], 0.0)
        self.assertFalse(coefficient_readback(expected, changed[:, :3])["exact"])

    def test_contract_matrix_requires_two_finite_poles(self) -> None:
        one = {"candidate": {"poles": self.axis["candidate"]["poles"][:1]}}
        with self.assertRaises(RuntimeError):
            contract_matrix(one)
        nonfinite = {
            "candidate": {
                "poles": [
                    self.axis["candidate"]["poles"][0],
                    {"c1": float("nan"), "c2": -0.25, "c3": 0.125},
                ]
            }
        }
        with self.assertRaises(RuntimeError):
            contract_matrix(nonfinite)


if __name__ == "__main__":
    unittest.main()
