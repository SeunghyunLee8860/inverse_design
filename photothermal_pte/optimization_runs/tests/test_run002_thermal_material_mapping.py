from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


class Run002ThermalMaterialMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        run_dir = (
            Path(__file__).resolve().parents[1]
            / "run_002_gaussian10_w8p5_current_max"
        )
        sys.path.insert(0, str(run_dir))
        module_path = run_dir / "validate_production_thermal_material_adfd.py"
        spec = importlib.util.spec_from_file_location(
            "run002_thermal_material_adfd", module_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.module = module
        selected_path = run_dir / "selected_thermal_density_mapping.py"
        selected_spec = importlib.util.spec_from_file_location(
            "run002_selected_thermal_density_mapping", selected_path
        )
        assert selected_spec is not None and selected_spec.loader is not None
        selected = importlib.util.module_from_spec(selected_spec)
        selected_spec.loader.exec_module(selected)
        cls.selected = selected

    def test_nodal_to_cell_exact_transpose(self) -> None:
        rng = np.random.default_rng(2026080604)
        nodal = rng.normal(size=(17, 13))
        cell_dual = rng.normal(size=(16, 12))
        left = np.sum(self.module.nodal_to_cell(nodal) * cell_dual)
        right = np.sum(
            nodal * self.module.nodal_to_cell_transpose(cell_dual)
        )
        self.assertLess(
            abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny),
            1.0e-14,
        )

    def test_base_density_keeps_no_clipping_margin(self) -> None:
        base, directions = self.module.base_density_and_directions((201, 201))
        self.assertGreater(float(np.min(base)), 0.4)
        self.assertLess(float(np.max(base)), 0.6)
        for direction in directions.values():
            self.assertLess(float(np.max(base + 0.01 * direction)), 1.0)
            self.assertGreater(float(np.min(base - 0.01 * direction)), 0.0)

    def test_selected_373_to_186_mapping_and_exact_transpose(self) -> None:
        rng = np.random.default_rng(2026080607)
        nodal = rng.normal(size=(373, 373))
        dual = rng.normal(size=(186, 186))
        mapped = self.selected.selected_nodal_to_thermal_cell(nodal)
        self.assertEqual(mapped.shape, (186, 186))
        left = float(np.sum(mapped * dual))
        right = float(
            np.sum(
                nodal
                * self.selected.selected_nodal_to_thermal_cell_transpose(dual)
            )
        )
        self.assertLess(
            abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny),
            1.0e-13,
        )

    def test_selected_mapping_preserves_bilinear_integral(self) -> None:
        rng = np.random.default_rng(2026080608)
        nodal = rng.random((373, 373))
        mapped = self.selected.selected_nodal_to_thermal_cell(nodal)
        nodal_integral = self.selected.bilinear_integral(nodal)
        cell_integral = float(
            np.sum(mapped) * self.selected.THERMAL_CELL_STEP_M**2
        )
        self.assertLess(
            abs(nodal_integral - cell_integral)
            / max(abs(nodal_integral), abs(cell_integral)),
            1.0e-14,
        )


if __name__ == "__main__":
    unittest.main()
