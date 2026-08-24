from __future__ import annotations

import unittest

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_courant_certificate import (
    COURANT,
    LEVELS,
    _level_sha,
    compare_courant_pair,
    courant_raw_schema_checks,
    courant_selection_gates,
    expected_courant_case,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
    component_power,
)


class CourantCertificateTest(unittest.TestCase):
    @staticmethod
    def _comparison_inputs() -> tuple[dict, dict, dict]:
        snapshots: dict[str, dict] = {}
        payloads: dict[str, dict] = {}
        source_pairs: dict[str, dict] = {}
        for level in LEVELS:
            snapshots[level] = {}
            payloads[level] = {}
            source_pairs[level] = {
                "comparison": {
                    "unscaled_incident_power_W": {"Ea": 2.0, "Eb": 2.0},
                    "mean_unscaled_incident_power_W": 2.0,
                }
            }
            for polarization in ("Ea", "Eb"):
                volumes = {
                    "au": np.ones((3, 2, 2, 1)),
                    "tairte4": np.ones((3, 2, 2, 1)),
                }
                q = {
                    "au": np.ones((3, 2, 2, 1)),
                    "tairte4": np.ones((3, 2, 2, 1)) * 2.0,
                }
                by_material = {
                    material: component_power(q[material], volumes[material])
                    for material in ("au", "tairte4")
                }
                snapshots[level][polarization] = {
                    "probe": np.ones((3, 2, 2, 1), dtype=np.complex64),
                    "probe_weights": np.ones((3, 2, 2, 1)),
                    "fields_late": {
                        "au": np.ones((3, 2, 2, 1), dtype=np.complex64),
                        "tairte4": np.ones((3, 2, 2, 1), dtype=np.complex64),
                    },
                    "q_late": q,
                    "volumes": volumes,
                    "power_late": {
                        "by_material": by_material,
                        "total_W": sum(item["total_W"] for item in by_material.values()),
                    },
                }
                payloads[level][polarization] = {
                    "evaluation": {
                        "flux": {
                            "Q_vs_closed_phasor_symmetric_relative": 1.0e-3,
                            "Q_vs_closed_td_symmetric_relative": 1.0e-3,
                        },
                        "field_stationarity": {
                            "maximum_complex_E_NRMSE": 1.0e-3
                        },
                    }
                }
        return snapshots, payloads, source_pairs

    def test_expected_cases_change_only_courant(self) -> None:
        cases = {level: expected_courant_case(level) for level in LEVELS}
        first = cases[LEVELS[0]]
        for level in LEVELS:
            self.assertEqual(cases[level].mesh, first.mesh)
            self.assertEqual(cases[level].time.total_periods, 24)
            self.assertEqual(cases[level].time.window_periods, 4)
            self.assertEqual(cases[level].time.courant_factor, COURANT[level])

    def test_identical_successive_pair_passes(self) -> None:
        snapshots, payloads, sources = self._comparison_inputs()
        result = compare_courant_pair(
            "c0p5", "c0p375", snapshots, payloads, sources
        )
        self.assertTrue(result["pass"])
        self.assertTrue(all(result["checks"].values()))

    def test_probe_or_source_change_above_limit_blocks_pair(self) -> None:
        snapshots, payloads, sources = self._comparison_inputs()
        snapshots["c0p375"]["Eb"]["probe"] *= 1.10
        result = compare_courant_pair(
            "c0p5", "c0p375", snapshots, payloads, sources
        )
        self.assertFalse(result["checks"]["complex_E_fixed_probe_NRMSE"])
        snapshots, payloads, sources = self._comparison_inputs()
        sources["c0p375"]["comparison"]["unscaled_incident_power_W"]["Ea"] = 3.0
        result = compare_courant_pair(
            "c0p5", "c0p375", snapshots, payloads, sources
        )
        self.assertFalse(result["checks"]["source_power_relative_change"])

    def test_selection_requires_all_cases_and_both_pairs(self) -> None:
        ready = {
            level: {"Ea": True, "Eb": True}
            for level in LEVELS
        }
        pairs = {
            ("c0p5", "c0p375"): True,
            ("c0p375", "c0p25"): True,
        }
        self.assertTrue(all(courant_selection_gates(ready, pairs).values()))
        ready["c0p25"]["Eb"] = False
        self.assertFalse(all(courant_selection_gates(ready, pairs).values()))
        ready["c0p25"]["Eb"] = True
        pairs[("c0p375", "c0p25")] = False
        self.assertFalse(all(courant_selection_gates(ready, pairs).values()))

    def test_closed_td_schema_allows_only_inverse_courant_sample_axis(self) -> None:
        samples = {"c0p5": 6416, "c0p375": 8554, "c0p25": 12832}
        cases = {
            level: {
                polarization: {
                    "raw": {
                        "declared_arrays": {
                            "target": [3, 160, 160, 1],
                            "closed_td": [samples[level], 1],
                        }
                    }
                }
                for polarization in ("Ea", "Eb")
            }
            for level in LEVELS
        }
        self.assertTrue(all(courant_raw_schema_checks(cases).values()))
        cases["c0p25"]["Eb"]["raw"]["declared_arrays"]["target"] = [3, 80, 80, 1]
        self.assertFalse(
            courant_raw_schema_checks(cases)["non_time_raw_array_schema_identical"]
        )

    def test_hash_parser_is_exact_and_fail_closed(self) -> None:
        values = [f"{level}={'a' * 64}" for level in LEVELS]
        self.assertEqual(set(_level_sha(values, "contract")), set(LEVELS))
        with self.assertRaises(ValueError):
            _level_sha(values[:-1], "contract")
        with self.assertRaises(ValueError):
            _level_sha([*values[:-1], "c0p25=not-a-sha"], "contract")


if __name__ == "__main__":
    unittest.main()
