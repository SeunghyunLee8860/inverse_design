from __future__ import annotations

import unittest

from photothermal_pte.optimization_runs.gaussian10_contract import (
    build_contract_audit,
)


class Gaussian10ContractTest(unittest.TestCase):
    def test_aperture_and_material_are_explicit(self) -> None:
        audit = build_contract_audit()
        self.assertTrue(audit["analytic_aperture_gate_pass"])
        self.assertLess(
            audit["source"]["source_boundary_intensity_over_peak"], 1.0e-4
        )
        self.assertGreater(
            audit["source"]["fitted_infinite_gaussian_square_captured_fraction"],
            0.999,
        )
        self.assertGreater(audit["optical_material"]["n_imag"], 0.0)
        self.assertEqual(len(audit["thermal_interface_scenarios"]), 4)

    def test_one_micron_is_the_frozen_middle_candidate(self) -> None:
        design = build_contract_audit()["design"]
        self.assertEqual(design["recommended_height_um"], 1.0)
        heights = [value["height_um"] for value in design["height_candidates"]]
        self.assertEqual(heights, [0.6, 1.0, 1.5])


if __name__ == "__main__":
    unittest.main()
