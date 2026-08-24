from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    grid_edges,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_anchor_placement import (
    expected_placement,
)

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
    component_power,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_full_z_certificate import (
    COURANT_FACTOR,
    LEVELS,
    Z_FACTOR,
    _interpolate_fine_to_coarse_z,
    _level_sha,
    _restrict_component_q_to_coarse_z,
    _z_control_intervals,
    compare_full_z_pair,
    expected_full_z_case,
    full_z_selection_gates,
    source_raw_grid_audit,
)


class FullZCertificateTest(unittest.TestCase):
    @staticmethod
    def _volume(
        edges: np.ndarray, bounds: tuple[int, int], nx: int = 2, ny: int = 2
    ) -> np.ndarray:
        values = []
        for component in range(3):
            left, right = _z_control_intervals(edges, bounds, component)
            values.append(
                np.broadcast_to(
                    (right - left)[None, None, :], (nx, ny, right.size)
                ).copy()
            )
        return np.stack(values)

    @classmethod
    def _comparison_inputs(cls) -> tuple[dict, dict, dict]:
        z_edges = {
            "z2": np.arange(-1.0, 3.5, 0.5),
            "z4": np.arange(-1.0, 3.25, 0.25),
        }
        z_bounds = {"z2": (2, 6), "z4": (4, 12)}
        snapshots: dict[str, dict] = {}
        payloads: dict[str, dict] = {}
        source_pairs: dict[str, dict] = {}
        xy_edges = np.asarray([0.0, 1.0, 2.0])
        for level in ("z2", "z4"):
            snapshots[level] = {}
            payloads[level] = {}
            source_pairs[level] = {
                "comparison": {
                    "unscaled_incident_power_W": {"Ea": 2.0, "Eb": 2.0},
                    "mean_unscaled_incident_power_W": 2.0,
                }
            }
            volume = cls._volume(z_edges[level], z_bounds[level])
            volumes = {"au": volume, "tairte4": volume.copy()}
            q = {
                "au": np.ones_like(volume) * 2.0,
                "tairte4": np.ones_like(volume) * 3.0,
            }
            by_material = {
                material: component_power(q[material], volumes[material])
                for material in ("au", "tairte4")
            }
            power = {
                "by_material": by_material,
                "total_W": sum(item["total_W"] for item in by_material.values()),
            }
            for polarization in ("Ea", "Eb"):
                snapshots[level][polarization] = {
                    "grid_edges": (xy_edges, xy_edges, z_edges[level]),
                    "probe": np.ones((3, 2, 2, 1), dtype=np.complex64),
                    "probe_weights": np.ones((3, 2, 2, 1)),
                    "fields_late": {
                        "au": np.ones_like(volume, dtype=np.complex64) * (1.0 + 0.5j),
                        "tairte4": np.ones_like(volume, dtype=np.complex64) * (2.0 - 0.25j),
                    },
                    "q_late": {name: value.copy() for name, value in q.items()},
                    "volumes": {
                        name: value.copy() for name, value in volumes.items()
                    },
                    "power_late": power,
                    "solver_mask": np.ones((2, 2), dtype=np.uint8),
                }
                placement = {
                    "au_design": [[0, 2], [0, 2], list(z_bounds[level])],
                    "fixed_tairte4": [[0, 2], [0, 2], list(z_bounds[level])],
                }
                payloads[level][polarization] = {
                    "placement": placement,
                    "evaluation": {
                        "flux": {
                            "Q_vs_closed_phasor_symmetric_relative": 1.0e-3,
                            "Q_vs_closed_td_symmetric_relative": 1.0e-3,
                        },
                        "field_stationarity": {
                            "maximum_complex_E_NRMSE": 1.0e-3
                        },
                    },
                }
        return snapshots, payloads, source_pairs

    def test_expected_cases_change_only_full_z_factor(self) -> None:
        cases = {level: expected_full_z_case(level) for level in LEVELS}
        first = cases[LEVELS[0]]
        first_mesh = dict(first.mesh.__dict__)
        first_mesh.pop("z_factor")
        for level in LEVELS:
            mesh = dict(cases[level].mesh.__dict__)
            self.assertEqual(mesh.pop("z_factor"), Z_FACTOR[level])
            self.assertEqual(mesh, first_mesh)
            self.assertEqual(cases[level].time.total_periods, 24)
            self.assertEqual(cases[level].time.window_periods, 4)
            self.assertEqual(cases[level].time.courant_factor, COURANT_FACTOR)

    def test_complex_linear_interpolation_uses_physical_z_without_extrapolation(self) -> None:
        fine_z = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0])
        coarse_z = np.asarray([0.0, 1.0, 2.0])
        fine = (2.0 + 3.0j) * fine_z[None, None, :] + (1.0 - 0.5j)
        result = _interpolate_fine_to_coarse_z(fine, fine_z, coarse_z)
        expected = (2.0 + 3.0j) * coarse_z[None, None, :] + (1.0 - 0.5j)
        self.assertTrue(np.allclose(result, expected))
        with self.assertRaises(ValueError):
            _interpolate_fine_to_coarse_z(
                fine, fine_z, np.asarray([-0.1, 1.0])
            )

    def test_conservative_component_restriction_preserves_constant_q(self) -> None:
        coarse_edges = np.arange(-1.0, 3.5, 0.5)
        fine_edges = np.arange(-1.0, 3.25, 0.25)
        coarse_bounds = (2, 6)
        fine_bounds = (4, 12)
        for component in range(3):
            coarse_volume = self._volume(coarse_edges, coarse_bounds)[component]
            fine_volume = self._volume(fine_edges, fine_bounds)[component]
            coarse_q = np.full(coarse_volume.shape, 7.0)
            fine_q = np.full(fine_volume.shape, 7.0)
            _, mapped, weights, audit = _restrict_component_q_to_coarse_z(
                coarse_q,
                fine_q,
                coarse_volume,
                fine_volume,
                coarse_edges,
                fine_edges,
                coarse_bounds,
                fine_bounds,
                component,
            )
            self.assertTrue(audit["ready"])
            self.assertGreaterEqual(audit["common_z_support_fraction"], 0.90)
            self.assertLessEqual(
                audit["fine_restriction_relative_power_error"], 5.0e-13
            )
            self.assertTrue(np.array_equal(mapped, np.full(mapped.shape, 7.0)))
            self.assertTrue(np.all(weights > 0.0))

    def test_float32_canonical_tairte4_support_is_not_a_false_failure(self) -> None:
        coarse_spec = expected_full_z_case("z2")
        fine_spec = expected_full_z_case("z4")
        coarse_edges = np.asarray(grid_edges(coarse_spec.mesh)[2], dtype=np.float32)
        fine_edges = np.asarray(grid_edges(fine_spec.mesh)[2], dtype=np.float32)
        coarse_bounds = tuple(
            expected_placement(coarse_spec.mesh)["fixed_tairte4"][2]
        )
        fine_bounds = tuple(
            expected_placement(fine_spec.mesh)["fixed_tairte4"][2]
        )
        for component in range(3):
            coarse_volume = self._volume(coarse_edges, coarse_bounds)[component]
            fine_volume = self._volume(fine_edges, fine_bounds)[component]
            _, _, _, audit = _restrict_component_q_to_coarse_z(
                np.ones_like(coarse_volume),
                np.ones_like(fine_volume),
                coarse_volume,
                fine_volume,
                coarse_edges,
                fine_edges,
                coarse_bounds,
                fine_bounds,
                component,
            )
            self.assertTrue(audit["ready"])
            self.assertGreater(
                audit["common_z_support_fraction"], 0.8999999
            )

    def test_identical_physical_fields_and_q_pass_full_z_pair(self) -> None:
        snapshots, payloads, sources = self._comparison_inputs()
        result = compare_full_z_pair("z2", "z4", snapshots, payloads, sources)
        self.assertTrue(result["pass"])
        self.assertTrue(all(result["checks"].values()))
        self.assertAlmostEqual(
            result["metrics"]["conservative_Q_volume_L2_NRMSE"], 0.0
        )

    def test_au_field_comparison_excludes_design_window_air(self) -> None:
        snapshots, payloads, sources = self._comparison_inputs()
        mask = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
        for level in ("z2", "z4"):
            for polarization in ("Ea", "Eb"):
                snapshots[level][polarization]["solver_mask"] = mask.copy()
        snapshots["z4"]["Ea"]["fields_late"]["au"][:, 1:, :, :] *= 100.0
        snapshots["z4"]["Ea"]["fields_late"]["au"][:, :, 1:, :] *= 100.0
        result = compare_full_z_pair("z2", "z4", snapshots, payloads, sources)
        self.assertEqual(
            result["per_polarization"]["Ea"][
                "material_region_complex_E_NRMSE_after_fine_to_coarse_z_interpolation"
            ]["au"],
            0.0,
        )

    def test_material_region_field_change_above_limit_blocks_pair(self) -> None:
        snapshots, payloads, sources = self._comparison_inputs()
        snapshots["z4"]["Ea"]["fields_late"]["au"] *= 1.10
        result = compare_full_z_pair("z2", "z4", snapshots, payloads, sources)
        self.assertFalse(result["pass"])
        self.assertFalse(
            result["checks"]["material_region_complex_E_max_NRMSE"]
        )

    def test_tangential_probe_change_above_limit_blocks_pair(self) -> None:
        snapshots, payloads, sources = self._comparison_inputs()
        snapshots["z4"]["Eb"]["probe"][:2] *= 1.10
        result = compare_full_z_pair("z2", "z4", snapshots, payloads, sources)
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["complex_E_fixed_probe_NRMSE"])

    def test_selection_requires_all_cases_and_both_successive_pairs(self) -> None:
        ready = {level: {"Ea": True, "Eb": True} for level in LEVELS}
        pairs = {("z2", "z4"): True, ("z4", "z8"): True}
        self.assertTrue(all(full_z_selection_gates(ready, pairs).values()))
        ready["z8"]["Eb"] = False
        self.assertFalse(all(full_z_selection_gates(ready, pairs).values()))
        ready["z8"]["Eb"] = True
        pairs[("z4", "z8")] = False
        self.assertFalse(all(full_z_selection_gates(ready, pairs).values()))

    def test_missing_source_pair_payload_blocks_comparison(self) -> None:
        snapshots, payloads, sources = self._comparison_inputs()
        del sources["z4"]["comparison"]
        result = compare_full_z_pair("z2", "z4", snapshots, payloads, sources)
        self.assertFalse(result["pass"])
        self.assertIn("source-pair payloads failed", result["error"])

    def test_source_raw_audit_rejects_existing_relative_runner_path(self) -> None:
        spec = expected_full_z_case("z2")
        expected_edges = grid_edges(spec.mesh)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runner = root / "runner.py"
            runner.write_text("# test runner\n", encoding="utf-8")
            runner_sha = hashlib.sha256(runner.read_bytes()).hexdigest()
            cases = {}
            for polarization in ("Ea", "Eb"):
                raw_path = root / f"{polarization}.npz"
                np.savez(
                    raw_path,
                    grid_x_edges_m=expected_edges[0],
                    grid_y_edges_m=expected_edges[1],
                    grid_z_edges_m=expected_edges[2],
                )
                report_path = root / f"{polarization}.json"
                report_path.write_text(
                    json.dumps(
                        {
                            "provenance": {
                                "runner_path": "runner.py",
                                "runner_sha256": runner_sha,
                            },
                            "raw": {
                                "path": str(raw_path),
                                "arrays": {
                                    f"grid_{axis}_edges_m": list(
                                        np.asarray(expected_edges[index]).shape
                                    )
                                    for index, axis in enumerate("xyz")
                                },
                            },
                            "placement": expected_placement(spec.mesh),
                        }
                    ),
                    encoding="utf-8",
                )
                cases[polarization] = {"report_path": str(report_path)}
            source_pair = {
                "cases": cases,
                "source_case_contracts": {
                    "placement": expected_placement(spec.mesh)
                },
            }
            previous = Path.cwd()
            os.chdir(root)
            try:
                audit = source_raw_grid_audit(source_pair, spec, root)
            finally:
                os.chdir(previous)
        self.assertFalse(audit["ready"])
        for polarization in ("Ea", "Eb"):
            self.assertFalse(
                audit["checks"][
                    f"{polarization}_source_runner_exists_and_matches"
                ]
            )

    def test_hash_parser_is_exact_and_fail_closed(self) -> None:
        values = [f"{level}={'a' * 64}" for level in LEVELS]
        self.assertEqual(set(_level_sha(values, "contract")), set(LEVELS))
        with self.assertRaises(ValueError):
            _level_sha(values[:-1], "contract")
        with self.assertRaises(ValueError):
            _level_sha([*values[:-1], "z8=not-a-sha"], "contract")


if __name__ == "__main__":
    unittest.main()
