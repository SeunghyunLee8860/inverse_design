from __future__ import annotations

import unittest

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_only import (
    polarization_audit,
)


class FdtdxSourceOnlyPolarizationDefinitionTest(unittest.TestCase):
    def test_longitudinal_field_is_not_mislabeled_as_cross_polarization(self) -> None:
        field = np.zeros((3, 2, 2, 1), dtype=np.complex64)
        field[1] = 2.0
        field[2] = 1.0
        audit = polarization_audit(field, "Ea", np.ones((2, 2, 1)))
        self.assertEqual(audit["transverse_purity"], 1.0)
        self.assertAlmostEqual(audit["longitudinal_fraction_of_total"], 0.2)
        self.assertEqual(
            audit["purity_definition"],
            "desired transverse E divided by Ex plus Ey",
        )


if __name__ == "__main__":
    unittest.main()
