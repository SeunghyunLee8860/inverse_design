from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_matrix import (
    BLOCKED_STATUS,
    MATRIX_STATUS,
    build_matrix_certificate,
    sha256,
)


def _component_power(q: np.ndarray, volume: np.ndarray) -> dict:
    component = np.sum(q * volume, axis=(1, 2, 3), dtype=np.float64)
    return {
        "component_W": {
            axis: float(component[index])
            for index, axis in enumerate(("x", "y", "z"))
        },
        "total_W": float(np.sum(component, dtype=np.float64)),
    }


class FdtdxFreshExactBinaryMatrixTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runner = self.root / "runner.py"
        self.material_contract = self.root / "material.json"
        self.source_pair = self.root / "source_pair.json"
        self.runner.write_text("# locked runner\n", encoding="utf-8")
        self.material_contract.write_text("{}\n", encoding="utf-8")
        self.source_pair.write_text("{}\n", encoding="utf-8")
        self.reports: dict[tuple[str, str], Path] = {}
        self.raws: dict[tuple[str, str], Path] = {}
        for reference in ("empty", "full_design_window"):
            for polarization in ("Ea", "Eb"):
                self._write_case(reference, polarization)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _paths(self) -> tuple[Path, Path, Path, Path]:
        return (
            self.reports[("empty", "Ea")],
            self.reports[("empty", "Eb")],
            self.reports[("full_design_window", "Ea")],
            self.reports[("full_design_window", "Eb")],
        )

    def _write_case(self, reference: str, polarization: str) -> None:
        name = f"{reference}_{polarization}"
        raw_path = self.root / f"{name}.npz"
        report_path = self.root / f"{name}.json"
        solid = reference == "full_design_window"
        mask = np.ones((2, 2), dtype=np.uint8) if solid else np.zeros(
            (2, 2), dtype=np.uint8
        )
        volume_au = np.ones((3, 2, 2, 1), dtype=np.float64)
        volume_ta = np.ones((3, 2, 2, 1), dtype=np.float64) * 2.0
        q_au = np.ones((3, 2, 2, 1), dtype=np.float64) * (0.25 if solid else 0.0)
        q_ta = np.ones((3, 2, 2, 1), dtype=np.float64) * (
            1.0 if polarization == "Ea" else 1.5
        )
        arrays = {
            "design_mask": mask,
            "solver_mask": mask.copy(),
            "q_au_previous_W_m3": q_au.copy(),
            "q_au_late_W_m3": q_au.copy(),
            "q_tairte4_previous_W_m3": q_ta.copy(),
            "q_tairte4_late_W_m3": q_ta.copy(),
            "electric_dual_volume_au_m3": volume_au,
            "electric_dual_volume_tairte4_m3": volume_ta,
        }
        np.savez_compressed(raw_path, **arrays)
        au_power = _component_power(q_au, volume_au)
        ta_power = _component_power(q_ta, volume_ta)
        window_power = {
            "by_material": {"au": au_power, "tairte4": ta_power},
            "total_W": au_power["total_W"] + ta_power["total_W"],
        }
        source_vector = (
            [0.0, 1.0, 0.0] if polarization == "Ea" else [1.0, 0.0, 0.0]
        )
        mask_sha256 = hashlib.sha256(
            np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
        ).hexdigest()
        payload = {
            "status": "VALIDATED_FDTDX_FRESH_EXACT_BINARY_PILOT_CASE",
            "ready": True,
            "scope": (
                "one fixed exact-binary optical material pilot; "
                "no thermal/electrical/adjoint/optimizer"
            ),
            "reference": reference,
            "polarization": polarization,
            "mesh": {"grid_contract_sha256": "mesh"},
            "time_contract": {"total_periods": 16, "time_steps_total": 100},
            "pml_face_parameters": {"minx": {"alpha_start": 1.0}},
            "placement": {"au_design": [[0, 2], [0, 2], [0, 1]]},
            "source_contract": {
                "polarization": polarization,
                "fixed_E_polarization_vector": source_vector,
                "direction": "-",
                "num_startup_periods": 4,
            },
            "source_pair": {
                "path": str(self.source_pair.resolve()),
                "expected_sha256": sha256(self.source_pair),
                "ready": True,
                "checks": {"certificate": True},
                "failed_checks": [],
            },
            "source_pair_contract_checks": {
                "mesh_matches_source_pair": True,
                "source_matches_polarization_case": True,
            },
            "material": {
                "ready": True,
                "checks": {"complete_stack": True},
                "failed_checks": [],
                "realized_material_response": {
                    "au": {"realized_epsilon": [-830.0, 127.0]},
                    "a": {"realized_epsilon": [-30.0, 50.0]},
                    "b": {"realized_epsilon": [15.0, 9.0]},
                    "c": {"realized_epsilon": [15.0, 9.0]},
                },
                "exact_binary_au": {
                    "ready": True,
                    "checks": {"no_gray_material_law": True},
                    "gray_density_allowed": False,
                    "rho_power": None,
                    "design_solid_cells": int(np.count_nonzero(mask)),
                    "solver_solid_cells": int(np.count_nonzero(mask)),
                    "design_mask_sha256": mask_sha256,
                    "solver_mask_sha256": mask_sha256,
                },
            },
            "evaluation": {
                "ready": True,
                "gates": {"Q_closed_phasor_closure": True},
                "failed_gates": [],
                "Q": {
                    "previous": copy.deepcopy(window_power),
                    "late": copy.deepcopy(window_power),
                    "previous_late_spatial_NRMSE": 0.0,
                    "previous_late_total_relative_change": 0.0,
                },
                "flux": {
                    "Q_vs_closed_phasor_symmetric_relative": 0.001,
                    "Q_vs_closed_td_symmetric_relative": 0.001,
                },
                "field_stationarity": {"maximum_complex_E_NRMSE": 0.001},
            },
            "raw": {
                "path": str(raw_path.resolve()),
                "sha256": sha256(raw_path),
                "arrays": {
                    array_name: list(array.shape)
                    for array_name, array in arrays.items()
                },
            },
            "normalization_policy": {
                "raw_fields_and_Q_are_unscaled": True,
                "per_polarization_matching_forbidden": True,
                "common_power_scale": 10.0,
                "common_field_amplitude_scale": 10.0**0.5,
            },
            "provenance": {
                "repository_commit": f"commit-{polarization}",
                "repository_dirty_porcelain": "",
                "fdtdx_source": {
                    "commit": "fdtdx-commit",
                    "dirty_porcelain": "",
                },
                "runtime_lock": {"jax": "locked"},
                "runner_path": str(self.runner.resolve()),
                "runner_sha256": sha256(self.runner),
                "material_contract_path": str(self.material_contract.resolve()),
                "material_contract_sha256": sha256(self.material_contract),
            },
            "optimizer_start_allowed": False,
        }
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        self.reports[(reference, polarization)] = report_path
        self.raws[(reference, polarization)] = raw_path

    def _mutate_report(self, key: tuple[str, str], mutate) -> None:
        path = self.reports[key]
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutate(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _build(self) -> dict:
        pair_audit = {
            "ready": True,
            "checks": {"all_bound_files_match": True},
            "failed_checks": [],
        }
        with patch(
            "photothermal_pte.optimization_runs.au_dualpol_4um_current_switch."
            "fdtdx_fresh_exact_binary_matrix.validate_source_pair",
            return_value=({}, pair_audit),
        ):
            return build_matrix_certificate(*self._paths())

    def test_valid_matrix_recomputes_all_raw_power_and_passes(self) -> None:
        result = self._build()
        self.assertTrue(result["ready"])
        self.assertEqual(result["status"], MATRIX_STATUS)
        self.assertEqual(result["failed_gates"], [])
        self.assertFalse(result["optimizer_start_allowed"])
        self.assertEqual(
            result["cases"]["empty"]["Ea"]["raw"]["derived"][
                "design_solid_cells"
            ],
            0,
        )
        self.assertEqual(
            result["cases"]["full_design_window"]["Eb"]["raw"]["derived"][
                "design_solid_cells"
            ],
            4,
        )

    def test_tampered_raw_file_blocks_matrix(self) -> None:
        self.raws[("empty", "Ea")].write_bytes(b"tampered")
        result = self._build()
        self.assertFalse(result["ready"])
        self.assertEqual(result["status"], BLOCKED_STATUS)
        self.assertIn("raw_files_and_recomputed_physics_ready", result["failed_gates"])

    def test_reported_power_that_disagrees_with_raw_blocks_matrix(self) -> None:
        self._mutate_report(
            ("full_design_window", "Eb"),
            lambda payload: payload["evaluation"]["Q"]["late"]["by_material"][
                "au"
            ].__setitem__("total_W", 123.0),
        )
        result = self._build()
        self.assertFalse(result["ready"])
        self.assertIn("raw_files_and_recomputed_physics_ready", result["failed_gates"])

    def test_gray_or_rho_power_material_law_blocks_matrix(self) -> None:
        def mutate(payload: dict) -> None:
            payload["material"]["exact_binary_au"]["gray_density_allowed"] = True
            payload["material"]["exact_binary_au"]["rho_power"] = 3

        self._mutate_report(("full_design_window", "Ea"), mutate)
        result = self._build()
        self.assertFalse(result["ready"])
        self.assertIn("case_exact_binary_gates", result["failed_gates"])

    def test_mesh_mismatch_blocks_matrix(self) -> None:
        self._mutate_report(
            ("empty", "Eb"),
            lambda payload: payload.__setitem__(
                "mesh", {"grid_contract_sha256": "different"}
            ),
        )
        result = self._build()
        self.assertFalse(result["ready"])
        self.assertIn("mesh_identical", result["failed_gates"])

    def test_source_pair_binding_mismatch_blocks_without_revalidation(self) -> None:
        self._mutate_report(
            ("full_design_window", "Eb"),
            lambda payload: payload["source_pair"].__setitem__(
                "expected_sha256", "0" * 64
            ),
        )
        result = self._build()
        self.assertFalse(result["ready"])
        self.assertIn("source_pair_binding_identical", result["failed_gates"])
        self.assertIn("source_pair_revalidation_ready", result["failed_gates"])


if __name__ == "__main__":
    unittest.main()
