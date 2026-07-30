from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from photothermal_pte.validation.paper_ir_sanity import (
    run_lumerical_device_a_ir_q as runner,
)
from photothermal_pte.validation.paper_ir_sanity import (
    summarize_paper_ir_diagnostic_smoke as summary_module,
)


class PaperIrDiagnosticSmokeTests(unittest.TestCase):
    def test_production_scalar_source_contract_defaults_are_fixed(self) -> None:
        argv = [
            "run_lumerical_device_a_ir_q.py",
            "--output-dir",
            "/tmp/not-created-by-parse",
            "--case",
            "empty-stack",
            "--polarization",
            "a",
            "--geometry",
            "planar-stack",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()
        self.assertEqual(args.execution_contract, "production")
        self.assertEqual(args.domain_um, 60.0)
        self.assertEqual(args.source_span_um, 50.0)
        self.assertEqual(args.waist_um, 12.0)
        self.assertTrue(
            np.allclose(args.absorption_bounds_m["x"], (-27e-6, 27e-6))
        )
        self.assertLess(args.inner_box["x"][1], 0.5 * args.domain_um * 1e-6)

    def test_reduced_contract_has_nested_nominal_bounds(self) -> None:
        argv = [
            "run_lumerical_device_a_ir_q.py",
            "--output-dir",
            "/tmp/not-created-by-parse",
            "--case",
            "finite-flake",
            "--polarization",
            "a",
            "--geometry",
            "straight-45-edge",
            "--domain-um",
            "12",
            "--source-span-um",
            "6",
            "--waist-um",
            "2",
            "--flake-dz-nm",
            "10",
            "--simulation-time-ps",
            "4",
            "--execution-contract",
            "diagnostic-smoke",
        ]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()
        self.assertEqual(args.execution_contract, "diagnostic-smoke")
        self.assertEqual(args.absorption_bounds_m["x"], (-4.5e-6, 4.5e-6))
        self.assertTrue(
            np.allclose(args.inner_box["x"], (-4.55e-6, 4.55e-6))
        )
        self.assertTrue(
            np.allclose(args.inner_box["z"], (-180.0e-9, 50.0e-9))
        )
        self.assertLess(args.inner_box["x"][1], 0.5 * args.domain_um * 1e-6)

    def test_diagnostic_contract_rejects_empty_stack(self) -> None:
        argv = [
            "run_lumerical_device_a_ir_q.py",
            "--output-dir",
            "/tmp/not-created-by-parse",
            "--case",
            "empty-stack",
            "--polarization",
            "a",
            "--domain-um",
            "12",
            "--source-span-um",
            "6",
            "--simulation-time-ps",
            "4",
            "--execution-contract",
            "diagnostic-smoke",
        ]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                runner.parse_args()

    def test_edge_isolation_contract_accepts_planar_time_checkpoints(self) -> None:
        for simulation_time_ps in ("1.2", "4"):
            argv = [
                "run_lumerical_device_a_ir_q.py",
                "--output-dir",
                "/tmp/not-created-by-parse",
                "--case",
                "finite-flake",
                "--polarization",
                "a",
                "--geometry",
                "planar-stack",
                "--domain-um",
                "12",
                "--source-span-um",
                "6",
                "--waist-um",
                "2",
                "--flake-dz-nm",
                "10",
                "--simulation-time-ps",
                simulation_time_ps,
                "--execution-contract",
                "edge-isolation-smoke",
            ]
            with patch.object(sys, "argv", argv):
                args = runner.parse_args()
            self.assertEqual(args.geometry, "planar-stack")
            self.assertEqual(args.execution_contract, "edge-isolation-smoke")
            self.assertEqual(args.absorption_bounds_m["x"], (-4.5e-6, 4.5e-6))

    def test_elliptical_gaussian_fit_recovers_center_and_waists(self) -> None:
        x = np.linspace(-4.5e-6, 4.5e-6, 121)
        y = np.linspace(-4.5e-6, 4.5e-6, 119)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        expected = {
            "center_x_m": 0.21e-6,
            "center_y_m": -0.17e-6,
            "waist_x_m": 2.1e-6,
            "waist_y_m": 1.8e-6,
        }
        intensity = 3.2 * np.exp(
            -2.0
            * (
                (xx - expected["center_x_m"]) ** 2
                / expected["waist_x_m"] ** 2
                + (yy - expected["center_y_m"]) ** 2
                / expected["waist_y_m"] ** 2
            )
        )
        fitted = runner.fit_elliptical_gaussian(x, y, intensity)
        self.assertTrue(fitted["fit_success"])
        for key, value in expected.items():
            self.assertAlmostEqual(fitted[key], value, delta=2e-10)
        self.assertLess(fitted["fit_relative_RMS_over_peak"], 1e-8)

    def test_published_failure_is_fail_closed(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        path = (
            repository
            / "photothermal_pte"
            / "reports"
            / "paper_ir_edge_material_gradient_controls"
            / "paper_ir_diagnostic_gpu_smoke_summary.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["official_status"],
            summary_module.OFFICIAL_STATUS,
        )
        self.assertEqual(
            payload["diagnostic_status"],
            summary_module.DIAGNOSTIC_STATUS,
        )
        self.assertFalse(payload["validated"])
        self.assertFalse(payload["production_paper_like_result"])
        self.assertGreater(
            payload["absorption"]["common_grid_six_face_relative_closure"],
            0.005,
        )
        self.assertLess(
            payload["absorption"]["native_common_relative_difference"],
            0.005,
        )
        self.assertFalse(
            payload["execution"]["auto_shutoff_gate_reached"]
        )
        self.assertFalse(payload["execution"]["CPU_FDTD_fallback"])
        self.assertFalse(payload["thermal_run"])
        self.assertFalse(payload["PTE_run"])
        self.assertFalse(payload["adjoint_run"])
        self.assertFalse(payload["optimization_run"])

    def test_bounded_dual_cell_weights_close_on_realized_faces(self) -> None:
        coordinate = np.asarray([0.25, 0.75])
        weights = runner.bounded_dual_cell_weights(
            coordinate,
            0.0,
            1.0,
        )
        self.assertTrue(
            np.allclose(weights, np.asarray([0.5, 0.5]))
        )
        volume = runner.integrate_xyz_bounded(
            np.ones((2, 2, 2)),
            {
                "x": coordinate,
                "y": coordinate,
                "z": coordinate,
            },
            {
                "x": (0.0, 1.0),
                "y": (0.0, 1.0),
                "z": (0.0, 1.0),
            },
        )
        self.assertAlmostEqual(volume, 1.0)

    def test_bounded_dual_cell_weights_clip_staggered_outer_sample(self) -> None:
        coordinate = np.asarray([0.25, 0.75, 1.25])
        weights = runner.bounded_dual_cell_weights(
            coordinate,
            0.0,
            1.0,
        )
        self.assertTrue(
            np.allclose(weights, np.asarray([0.5, 0.5, 0.0]))
        )
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)

    def test_matched_smoke_passes_closure_but_fails_shutoff(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        path = (
            repository
            / "photothermal_pte"
            / "reports"
            / "paper_ir_edge_material_gradient_controls"
            / "paper_ir_matched_control_volume_smoke_summary.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["smoke_status"],
            "FAILED_MATCHED_CONTROL_VOLUME_SMOKE_AUTO_SHUTOFF_UNRESOLVED",
        )
        self.assertFalse(payload["validated"])
        self.assertTrue(
            payload["acceptance"][
                "native_Yee_six_face_closure_lt_0p5_percent"
            ]
        )
        self.assertTrue(
            payload["acceptance"][
                "common_grid_six_face_closure_lt_0p5_percent"
            ]
        )
        self.assertFalse(
            payload["acceptance"]["auto_shutoff_lt_1e_minus_5"]
        )
        self.assertFalse(
            payload["interpretation"][
                "old_9p18_percent_is_FDTD_energy_error"
            ]
        )
        self.assertFalse(payload["thermal_run"])
        self.assertFalse(payload["PTE_run"])
        self.assertFalse(payload["adjoint_run"])
        self.assertFalse(payload["optimization_run"])


if __name__ == "__main__":
    unittest.main()
