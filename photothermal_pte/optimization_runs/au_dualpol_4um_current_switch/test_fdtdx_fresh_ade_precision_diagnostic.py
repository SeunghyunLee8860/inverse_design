from __future__ import annotations

import math
import unittest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_ade_precision_diagnostic import (
    FIT_RELATIVE_TOLERANCE,
    analyze_z_factor,
    load_au_epsilon,
    realized_float32_cfl,
)


class FdtdxFreshAdePrecisionDiagnosticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.epsilon = load_au_epsilon()
        cls.z8 = analyze_z_factor(8, cls.epsilon)
        cls.z16 = analyze_z_factor(16, cls.epsilon)

    def test_realized_float32_grid_reproduces_completed_z8_dt(self) -> None:
        self.assertTrue(
            math.isclose(
                realized_float32_cfl(8)["time_step_s"],
                2.083469563193086e-18,
                rel_tol=0.0,
                abs_tol=1.0e-32,
            )
        )

    def test_current_single_drude_refit_crosses_gate_at_z16(self) -> None:
        self.assertLess(
            self.z8["current_single_drude_refit"]["fit_relative_error"],
            FIT_RELATIVE_TOLERANCE,
        )
        self.assertGreater(
            self.z16["current_single_drude_refit"]["fit_relative_error"],
            FIT_RELATIVE_TOLERANCE,
        )
        self.assertTrue(
            math.isclose(
                self.z16["current_single_drude_refit"]["fit_relative_error"],
                1.1757851520369762e-4,
                rel_tol=2.0e-6,
            )
        )

    def test_wide_single_drude_scan_does_not_rescue_z16(self) -> None:
        wide = self.z16["wide_single_drude_scan"]
        self.assertGreater(wide["fit_relative_error"], FIT_RELATIVE_TOLERANCE)
        self.assertLess(
            wide["fit_relative_error"],
            self.z16["current_single_drude_refit"]["fit_relative_error"],
        )

    def test_stable_positive_two_drude_candidate_is_not_promoted(self) -> None:
        candidate = self.z16["stable_two_drude_candidate"]
        self.assertTrue(candidate["found"])
        self.assertTrue(candidate["fit_gate_passed"])
        self.assertTrue(candidate["candidate_only"])
        self.assertTrue(
            candidate["promotion_forbidden_until_same_law_time_and_z_validation"]
        )
        for pole in candidate["poles"]:
            self.assertTrue(pole["positive_strength"])
            self.assertTrue(pole["dc_root_not_above_one"])
            self.assertEqual(pole["c3"], pole["reconstructed_float32_c3"])


if __name__ == "__main__":
    unittest.main()
