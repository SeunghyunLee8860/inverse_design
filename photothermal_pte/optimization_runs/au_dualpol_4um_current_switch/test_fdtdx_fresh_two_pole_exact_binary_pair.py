from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    reference_mask,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    ANCHOR_CASE,
    case_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_exact_binary import (
    STATUS_READY as CASE_STATUS,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_exact_binary_pair import (
    BLOCKED_STATUS,
    CERTIFICATE_NAME,
    EXPECTED_SCOPE,
    HERE,
    PAIR_STATUS,
    build_pair_certificate,
    validate_pair_certificate,
    write_pair_certificate,
)


REFERENCE = "l_shape_4um_with_500nm_arms"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _component_power(q: np.ndarray, volume: np.ndarray) -> dict:
    component = np.sum(q * volume, axis=(1, 2, 3), dtype=np.float64)
    return {
        "component_W": {
            axis: float(component[index])
            for index, axis in enumerate(("x", "y", "z"))
        },
        "total_W": float(np.sum(component, dtype=np.float64)),
    }


def _law() -> dict:
    return {
        "material_law_contract_sha256": "1" * 64,
        "case_binding": {"case_file_sha256": "2" * 64},
        "material_axes": {"au": {}, "a": {}, "b": {}, "c": {}},
        "promotion": {"candidate_only": True, "optimizer_start_allowed": False},
    }


class FdtdxFreshTwoPoleExactBinaryPairTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runner = self.root / "runner.py"
        self.material_contract = self.root / "material.json"
        self.source_pair = self.root / "source_pair.json"
        self.runner.write_text("# runner\n", encoding="utf-8")
        self.material_contract.write_text("{}\n", encoding="utf-8")
        self.source_pair.write_text("{}\n", encoding="utf-8")
        self.law = _law()
        self.reports: dict[str, dict] = {}
        self.paths: dict[str, Path] = {}
        self.raws: dict[str, Path] = {}
        for polarization in ("Ea", "Eb"):
            self._make_report(polarization)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_report(self, polarization: str) -> None:
        mask = np.asarray(reference_mask(REFERENCE), dtype=np.uint8)
        volume_au = np.ones((3, 80, 80, 1), dtype=np.float64)
        volume_ta = np.full((3, 80, 80, 1), 2.0, dtype=np.float64)
        q_au = np.full(
            (3, 80, 80, 1), 0.25 if polarization == "Ea" else 0.5
        )
        q_ta = np.full(
            (3, 80, 80, 1), 1.0 if polarization == "Ea" else 1.5
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
        raw = self.root / f"{polarization}.npz"
        np.savez_compressed(raw, **arrays)
        au_power = _component_power(q_au, volume_au)
        ta_power = _component_power(q_ta, volume_ta)
        total = au_power["total_W"] + ta_power["total_W"]
        window = {
            "by_material": {"au": au_power, "tairte4": ta_power},
            "total_W": total,
        }
        scale = 10.0
        numerical = case_contract(ANCHOR_CASE)
        mask_hash = hashlib.sha256(
            np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
        ).hexdigest()
        law_audit = {
            "ready": True,
            "path": "/absolute/law.json",
            "expected_sha256": "3" * 64,
            "actual_sha256": "3" * 64,
            "material_law_contract_sha256": self.law[
                "material_law_contract_sha256"
            ],
        }
        source_pair_audit = {
            "path": str(self.source_pair.resolve()),
            "expected_sha256": _sha256(self.source_pair),
            "actual_sha256": _sha256(self.source_pair),
            "ready": True,
            "checks": {"all": True},
            "failed_checks": [],
        }
        implementation = HERE / "fdtdx_4um_model.py"
        payload = {
            "status": CASE_STATUS,
            "ready": True,
            "scope": EXPECTED_SCOPE,
            "reference": REFERENCE,
            "polarization": polarization,
            "numerical_case_contract": numerical,
            "numerical_case_file_audit": {
                "ready": True,
                "actual_sha256": "2" * 64,
                "expected_sha256": "2" * 64,
                "case_contract_sha256": numerical["case_contract_sha256"],
            },
            "candidate_material_law_contract": self.law,
            "candidate_material_law_file_audit": law_audit,
            "mesh": numerical["resolved_mesh"],
            "time_contract": {
                **numerical["time_spec"],
                "time_step_s": 1.0e-18,
                "time_steps_total": 100,
            },
            "pml_face_parameters": numerical["resolved_pml_face_parameters"],
            "placement": {"au_design": [[0, 80], [0, 80], [0, 1]]},
            "source_contract": {
                "polarization": polarization,
                "fixed_E_polarization_vector": (
                    [0.0, 1.0, 0.0]
                    if polarization == "Ea"
                    else [1.0, 0.0, 0.0]
                ),
                "direction": "-",
                "wavelength_m": 4.0e-6,
            },
            "source_pair": source_pair_audit,
            "source_pair_contract_checks": {"all": True},
            "material": {
                "ready": True,
                "checks": {"complete": True},
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
                    "design_mask_sha256": mask_hash,
                    "solver_mask_sha256": mask_hash,
                },
            },
            "evaluation": {
                "ready": True,
                "gates": {"all": True},
                "failed_gates": [],
                "Q": {
                    "previous": copy.deepcopy(window),
                    "late": copy.deepcopy(window),
                    "previous_late_spatial_NRMSE": 0.0,
                    "previous_late_total_relative_change": 0.0,
                },
                "field_stationarity": {"maximum_complex_E_NRMSE": 0.001},
                "flux": {
                    "absorbed_fraction_of_all_air_source": 0.25,
                    "Q_vs_closed_phasor_symmetric_relative": 0.001,
                    "Q_vs_closed_td_symmetric_relative": 0.001,
                },
                "common_285uW_reporting": {
                    "late_Au_Q_W": au_power["total_W"] * scale,
                    "late_TaIrTe4_Q_W": ta_power["total_W"] * scale,
                    "late_total_Q_W": total * scale,
                },
            },
            "raw": {
                "path": str(raw.resolve()),
                "sha256": _sha256(raw),
                "arrays": {
                    name: list(array.shape) for name, array in arrays.items()
                },
            },
            "normalization_policy": {
                "raw_fields_and_Q_are_unscaled": True,
                "per_polarization_matching_forbidden": True,
                "common_power_scale": scale,
                "common_field_amplitude_scale": scale**0.5,
            },
            "provenance": {
                "repository_commit": "material-runner-commit",
                "repository_dirty_porcelain": "",
                "fdtdx_source": {
                    "path": "/absolute/fdtdx",
                    "commit": "fdtdx-commit",
                    "dirty_porcelain": "",
                },
                "runtime_lock": {"jax": "locked"},
                "runner_path": str(self.runner.resolve()),
                "runner_sha256": _sha256(self.runner),
                "implementation_sha256": {
                    implementation.name: _sha256(implementation)
                },
                "material_contract_path": str(self.material_contract.resolve()),
                "material_contract_sha256": _sha256(self.material_contract),
            },
            "optimizer_start_allowed": False,
        }
        report = self.root / f"{polarization}.json"
        report.write_text(json.dumps(payload), encoding="utf-8")
        self.reports[polarization] = payload
        self.paths[polarization] = report
        self.raws[polarization] = raw

    def _write_reports(self) -> None:
        for polarization in ("Ea", "Eb"):
            self.paths[polarization].write_text(
                json.dumps(self.reports[polarization]), encoding="utf-8"
            )

    def _build(self) -> dict:
        self._write_reports()
        pair_audit = {"ready": True, "checks": {"all": True}, "failed_checks": []}
        with patch(
            "photothermal_pte.optimization_runs.au_dualpol_4um_current_switch."
            "fdtdx_fresh_two_pole_exact_binary_pair.material_law_from_contract",
            return_value=self.law,
        ), patch(
            "photothermal_pte.optimization_runs.au_dualpol_4um_current_switch."
            "fdtdx_fresh_two_pole_exact_binary_pair.validate_candidate_source_pair",
            return_value=({}, pair_audit),
        ):
            return build_pair_certificate(self.paths["Ea"], self.paths["Eb"])

    def test_valid_pair_recomputes_raw_physics_and_passes(self) -> None:
        result = self._build()
        self.assertTrue(result["ready"], result["failed_gates"])
        self.assertEqual(result["status"], PAIR_STATUS)
        self.assertEqual(result["failed_gates"], [])
        self.assertFalse(result["optimizer_start_allowed"])
        self.assertFalse(result["pte_current_claim_allowed"])
        self.assertIn("next separately hashed mesh level after z4", result["next_allowed_step"])
        self.assertEqual(
            result["cases"]["Ea"]["raw"]["derived"]["design_solid_cells"],
            375,
        )

    def test_tampered_raw_blocks_pair(self) -> None:
        self.raws["Ea"].write_bytes(b"tampered")
        result = self._build()
        self.assertFalse(result["ready"])
        self.assertEqual(result["status"], BLOCKED_STATUS)
        self.assertIn(
            "raw_files_and_recomputed_physics_ready", result["failed_gates"]
        )

    def test_reported_Q_disagreement_blocks_pair(self) -> None:
        self.reports["Eb"]["evaluation"]["Q"]["late"]["total_W"] = 123.0
        result = self._build()
        self.assertFalse(result["ready"])
        self.assertIn(
            "raw_files_and_recomputed_physics_ready", result["failed_gates"]
        )

    def test_gray_or_rho_power_blocks_pair(self) -> None:
        exact = self.reports["Ea"]["material"]["exact_binary_au"]
        exact["gray_density_allowed"] = True
        exact["rho_power"] = 3
        result = self._build()
        self.assertFalse(result["ready"])
        self.assertIn("case_exact_binary_gates", result["failed_gates"])
        self.assertIn("resolve the failed z4 within-case gates", result["next_allowed_step"])

    def test_different_material_law_blocks_pair(self) -> None:
        changed = copy.deepcopy(self.law)
        changed["material_law_contract_sha256"] = "9" * 64
        self.reports["Eb"]["candidate_material_law_contract"] = changed
        result = self._build()
        self.assertFalse(result["ready"])
        self.assertIn("canonical_case_and_material_law", result["failed_gates"])

    def test_written_certificate_revalidates_artifact_hashes(self) -> None:
        self._write_reports()
        output = self.root / "certificate"
        output.mkdir()
        pair_audit = {"ready": True, "checks": {"all": True}, "failed_checks": []}

        def clean_git(repository: Path, *arguments: str) -> str:
            del repository
            return "certificate-commit" if arguments == ("rev-parse", "HEAD") else ""

        with patch(
            "photothermal_pte.optimization_runs.au_dualpol_4um_current_switch."
            "fdtdx_fresh_two_pole_exact_binary_pair.material_law_from_contract",
            return_value=self.law,
        ), patch(
            "photothermal_pte.optimization_runs.au_dualpol_4um_current_switch."
            "fdtdx_fresh_two_pole_exact_binary_pair.validate_candidate_source_pair",
            return_value=({}, pair_audit),
        ), patch(
            "photothermal_pte.optimization_runs.au_dualpol_4um_current_switch."
            "fdtdx_fresh_two_pole_exact_binary_pair._git",
            side_effect=clean_git,
        ):
            written = write_pair_certificate(
                self.paths["Ea"], self.paths["Eb"], output
            )
        self.assertTrue(written["ready"], written["failed_gates"])
        certificate = output / CERTIFICATE_NAME
        _, audit = validate_pair_certificate(
            certificate,
            _sha256(certificate),
            ANCHOR_CASE,
            REFERENCE,
            self.law,
            self.reports["Ea"]["candidate_material_law_file_audit"],
        )
        self.assertTrue(audit["ready"], audit["failed_checks"])
        self.raws["Eb"].write_bytes(b"tampered after certification")
        _, changed = validate_pair_certificate(
            certificate,
            _sha256(certificate),
            ANCHOR_CASE,
            REFERENCE,
            self.law,
            self.reports["Ea"]["candidate_material_law_file_audit"],
        )
        self.assertFalse(changed["ready"])
        self.assertIn("Eb_raw_sha256_matches", changed["failed_checks"])


if __name__ == "__main__":
    unittest.main()
