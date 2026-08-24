from __future__ import annotations

import unittest

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
    reference_mask,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_material import (
    MATERIAL_LAW,
    mask_material_audit,
    normalize_exact_mask,
    solver_mask,
)


class FdtdxExactBinaryMaterialTest(unittest.TestCase):
    def test_gray_and_even_float_endpoint_masks_are_rejected(self) -> None:
        for value in (0.0, 0.5, 1.0):
            mask = np.full((80, 80), value, dtype=np.float64)
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_exact_mask(mask)

    def test_invalid_shape_and_integer_value_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            normalize_exact_mask(np.zeros((79, 80), dtype=np.uint8))
        invalid = np.zeros((80, 80), dtype=np.int8)
        invalid[3, 4] = 2
        with self.assertRaises(ValueError):
            normalize_exact_mask(invalid)

    def test_local_mesh_mapping_is_piecewise_constant_and_area_exact(self) -> None:
        design = np.asarray(reference_mask("centered_square_2um"), dtype=np.uint8)
        spec = MeshSpec(design_xy_factor=4)
        expanded = solver_mask(design, spec)
        self.assertEqual(expanded.shape, (320, 320))
        self.assertEqual(int(expanded.sum()), int(design.sum()) * 16)
        self.assertTrue(np.array_equal(expanded[120:124, 120:124], np.ones((4, 4))))
        self.assertTrue(np.array_equal(expanded[116:120, 120:124], np.zeros((4, 4))))

    def test_material_audit_has_no_density_exponent(self) -> None:
        spec = MeshSpec(design_xy_factor=2)
        audit = mask_material_audit(reference_mask("x_bar_4um_by_1um"), spec)
        self.assertEqual(audit["material_law"], MATERIAL_LAW)
        self.assertFalse(audit["gray_density_allowed"])
        self.assertIsNone(audit["rho_power"])
        self.assertEqual(audit["design_solid_cells"], 400)
        self.assertEqual(audit["solver_solid_cells"], 1600)


if __name__ == "__main__":
    unittest.main()
