from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
    material_stack_audit,
)


class FdtdxFreshMaterialStackTest(unittest.TestCase):
    def test_complete_stack_readback_preserves_tairte4_axis_mapping(self) -> None:
        shape = (80, 80, 4)
        inverse = np.ones((3, *shape), dtype=np.float32)
        inverse[:, :, :, 0] = np.float32(1.0 / 12.0)
        inverse[:, :, :, 1] = np.float32(1.0 / 2.0)
        coefficients = {
            "au": (0.11, -0.22, 0.33),
            "a": (0.41, -0.42, 0.43),
            "b": (0.51, -0.52, 0.53),
            "c": (0.61, -0.62, 0.63),
        }
        arrays = {
            name: np.zeros((1, 3, *shape), dtype=np.float32)
            for name in ("dispersive_c1", "dispersive_c2", "dispersive_c3")
        }
        for coefficient_index, name in enumerate(arrays):
            for component, axis in enumerate(("b", "a", "c")):
                arrays[name][0, component, :, :, 2] = np.float32(
                    coefficients[axis][coefficient_index]
                )
            arrays[name][0, :, :, :, 3] = np.float32(
                coefficients["au"][coefficient_index]
            )
        state = SimpleNamespace(
            inv_permittivities=inverse,
            dispersive_c1=arrays["dispersive_c1"],
            dispersive_c2=arrays["dispersive_c2"],
            dispersive_c3=arrays["dispersive_c3"],
        )
        epsilon_ta = {
            "a": 1.0 + 1.1j,
            "b": 1.0 + 1.2j,
            "c": 1.0 + 1.3j,
        }
        model = {
            "slices": {
                "fixed_silicon_substrate": (
                    slice(0, 80),
                    slice(0, 80),
                    slice(0, 1),
                ),
                "fixed_285nm_sio2": (
                    slice(0, 80),
                    slice(0, 80),
                    slice(1, 2),
                ),
                "fixed_tairte4": (
                    slice(0, 80),
                    slice(0, 80),
                    slice(2, 3),
                ),
                "au_design": (
                    slice(0, 80),
                    slice(0, 80),
                    slice(3, 4),
                ),
            },
            "coefficients": coefficients,
            "epsilon": {
                "silicon": 12.0 + 0.0j,
                "sio2": 2.0 + 0.0j,
                "tairte4": epsilon_ta,
                "au": 1.0 + 1.4j,
            },
            "discrete_susceptibility": {
                "a": epsilon_ta["a"] - 1.0,
                "b": epsilon_ta["b"] - 1.0,
                "c": epsilon_ta["c"] - 1.0,
                "au": 1.4j,
            },
            "absorption_loss_basis": "synthetic-discrete-ADE",
            "fits": {name: {"fit_relative_error": 0.0} for name in coefficients},
        }
        mask = np.ones((80, 80), dtype=np.uint8)
        audit = material_stack_audit(model, state, mask)
        self.assertTrue(audit["ready"], audit["failed_checks"])
        tairte4 = audit["tairte4_coefficient_readback"]["dispersive_c1"]
        self.assertEqual(
            [(item["component"], item["crystal_axis"]) for item in tairte4],
            [("Ex", "b"), ("Ey", "a"), ("Ez", "c")],
        )


if __name__ == "__main__":
    unittest.main()
