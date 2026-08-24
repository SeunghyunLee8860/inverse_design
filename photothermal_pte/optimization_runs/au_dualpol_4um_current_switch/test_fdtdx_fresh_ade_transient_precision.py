from __future__ import annotations

import unittest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_ade_transient_precision import (
    FLOAT32_VS_FLOAT64_LIMIT,
    FLOAT32_WINDOW_CHANGE_LIMIT,
    analyze_material_law,
    simulate_axis,
)


DT_Z16_S = 1.0422198660912219e-18
WAVELENGTH_M = 4.0e-6
AU_POLES = (
    (1.9999257326126099, -0.9999257326126099, 0.00010286720498697832),
    (1.9999256134033203, -0.9999256134033203, 0.00009998395398724824),
)
B_POLES = (
    (1.9990147352218628, -0.999015748500824, 0.000010596620995784178),
    (1.999090552330017, -0.9990915060043335, 0.000005018332558393013),
)


def candidate(poles: tuple[tuple[float, float, float], ...]) -> dict:
    return {
        "candidate": {
            "poles": [
                {"c1": c1, "c2": c2, "c3": c3}
                for c1, c2, c3 in poles
            ]
        }
    }


class FdtdxFreshAdeTransientPrecisionTest(unittest.TestCase):
    def test_z16_au_float32_recurrence_fails_long_time_gate(self) -> None:
        result = simulate_axis(
            AU_POLES,
            dt_s=DT_Z16_S,
            wavelength_m=WAVELENGTH_M,
            total_periods=32,
            startup_periods=4,
            window_periods=4,
        )
        self.assertTrue(
            result["gates"]["float64_reference_last_window_settled"]
        )
        self.assertFalse(result["gates"]["float32_last_window_settled"])
        self.assertFalse(
            result["gates"]["float32_matches_float64_late_response"]
        )
        self.assertGreater(
            result["precision"]["float32"]["last_relative_window_change"],
            0.017,
        )
        self.assertGreater(
            result["float32_vs_float64_late_relative_difference"], 0.03
        )
        self.assertGreater(
            result["carrier_conditioning"][0][
                "cancellation_condition_estimate"
            ],
            1.6e7,
        )

    def test_z16_b_scalar_recurrence_stays_below_operational_limits(self) -> None:
        result = simulate_axis(
            B_POLES,
            dt_s=DT_Z16_S,
            wavelength_m=WAVELENGTH_M,
            total_periods=32,
            startup_periods=4,
            window_periods=4,
        )
        self.assertLess(
            result["precision"]["float32"]["last_relative_window_change"],
            FLOAT32_WINDOW_CHANGE_LIMIT,
        )
        self.assertLess(
            result["float32_vs_float64_late_relative_difference"],
            FLOAT32_VS_FLOAT64_LIMIT,
        )

    def test_material_law_is_blocked_if_any_axis_transient_fails(self) -> None:
        law = {
            "version": "fdtdx-fresh-stable-two-pole-material-v1",
            "algorithm": {"target_wavelength_m": WAVELENGTH_M},
            "case_binding": {
                "realized_float32_cfl": {"time_step_s": DT_Z16_S},
                "time_spec": {
                    "source_startup_periods": 4,
                    "total_periods": 32,
                    "window_periods": 4,
                },
            },
            "material_axes": {
                "au": candidate(AU_POLES),
                "a": candidate(AU_POLES),
                "b": candidate(B_POLES),
                "c": candidate(B_POLES),
            },
            "promotion": {
                "candidate_only": True,
                "optimizer_start_allowed": False,
            },
        }
        result = analyze_material_law(law)
        self.assertFalse(result["ready"])
        self.assertEqual(
            result["status"], "BLOCKED_FDTDX_FRESH_ADE_TRANSIENT_PRECISION"
        )
        self.assertIn("float32_recurrence_all_settled", result["failed_gates"])
        self.assertTrue(result["gates"]["float64_reference_all_settled"])
        self.assertFalse(result["optimizer_start_allowed"])

    def test_invalid_material_axis_set_fails_closed(self) -> None:
        law = {
            "version": "fdtdx-fresh-stable-two-pole-material-v1",
            "algorithm": {"target_wavelength_m": WAVELENGTH_M},
            "case_binding": {
                "realized_float32_cfl": {"time_step_s": DT_Z16_S},
                "time_spec": {
                    "source_startup_periods": 4,
                    "total_periods": 32,
                    "window_periods": 4,
                },
            },
            "material_axes": {"au": candidate(AU_POLES)},
            "promotion": {
                "candidate_only": True,
                "optimizer_start_allowed": False,
            },
        }
        with self.assertRaisesRegex(RuntimeError, "structure gates failed"):
            analyze_material_law(law)


if __name__ == "__main__":
    unittest.main()
