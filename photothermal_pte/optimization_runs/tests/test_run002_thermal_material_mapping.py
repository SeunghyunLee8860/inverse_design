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


if __name__ == "__main__":
    unittest.main()
