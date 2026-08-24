from __future__ import annotations

import unittest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_ade_precision_diagnostic import (
    C0_M_PER_S,
    WAVELENGTH_M,
    load_material_epsilon,
    realized_float32_cfl,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_increment_state_precision import (
    CARRIER_RELATIVE_ERROR_LIMIT,
    MATERIAL_AXES,
    audit_au_ordal_band,
    discrete_susceptibility,
    increment_state_coefficients,
    physical_pole_from_target,
    second_order_equivalent_susceptibility,
    simulate_axis,
)


class IncrementStatePrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.epsilon = load_material_epsilon()
        cls.poles = {
            name: physical_pole_from_target(name, cls.epsilon[name])
            for name in MATERIAL_AXES
        }

    def test_mesh_independent_physical_poles_match_carrier(self) -> None:
        self.assertEqual(tuple(self.epsilon), MATERIAL_AXES)
        self.assertEqual(self.poles["au"]["kind"], "Drude")
        self.assertEqual(self.poles["a"]["kind"], "Drude")
        self.assertEqual(self.poles["b"]["kind"], "Lorentz")
        self.assertEqual(self.poles["c"], self.poles["b"] | {"material_axis": "c"})
        for pole in self.poles.values():
            self.assertTrue(pole["passive_at_carrier"])
            self.assertLessEqual(
                pole["continuum_target_relative_error"], 1.0e-12
            )

    def test_float32_coefficients_are_stable_and_preserve_equations(self) -> None:
        omega = 2.0 * 3.141592653589793 * C0_M_PER_S / WAVELENGTH_M
        for z_factor in (8, 16, 32):
            dt_s = realized_float32_cfl(z_factor)["time_step_s"]
            for pole in self.poles.values():
                coefficients = increment_state_coefficients(pole, dt_s)
                self.assertTrue(coefficients["dynamic_state_stable"])
                direct = discrete_susceptibility(coefficients, omega, dt_s)
                eliminated = second_order_equivalent_susceptibility(
                    coefficients, omega, dt_s
                )
                self.assertLessEqual(
                    abs(direct - eliminated) / abs(direct), 1.0e-8
                )
                target = complex(*pole["target_susceptibility"])
                self.assertLessEqual(
                    abs(direct - target) / abs(target),
                    CARRIER_RELATIVE_ERROR_LIMIT,
                )

    def test_short_cpu_transient_gate_passes_at_z16(self) -> None:
        dt_s = realized_float32_cfl(16)["time_step_s"]
        for pole in self.poles.values():
            coefficients = increment_state_coefficients(pole, dt_s)
            result = simulate_axis(
                pole,
                coefficients,
                dt_s=dt_s,
                total_periods=16,
                startup_periods=2,
                window_periods=2,
            )
            self.assertTrue(result["ready"], result["gates"])
            self.assertTrue(all(result["gates"].values()))

    def test_au_ordal_band_sanity_is_bounded_but_not_overclaimed(self) -> None:
        result = audit_au_ordal_band(self.poles["au"])
        self.assertTrue(result["ready"])
        self.assertTrue(all(result["gates"].values()))
        self.assertLess(
            result["three_to_six_um"]["maximum_relative_error"], 0.02
        )
        self.assertIn("not a replacement", result["scope_note"])

    def test_invalid_material_axis_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            physical_pole_from_target("invalid", self.epsilon["au"])


if __name__ == "__main__":
    unittest.main()
