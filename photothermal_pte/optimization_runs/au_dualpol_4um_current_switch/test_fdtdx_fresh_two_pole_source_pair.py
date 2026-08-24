from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_anchor_placement import (
    expected_placement,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    ANCHOR_CASE,
    case_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_source_only import (
    CASE_STATUS,
    SCOPE,
    candidate_source_model_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_source_pair import (
    BLOCKED_STATUS,
    PAIR_STATUS,
    build_candidate_pair_certificate,
    validate_candidate_source_pair,
    write_candidate_pair_certificate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _law() -> dict:
    axes = {
        name: {
            "candidate": {
                "poles": [
                    {"c1": 1.1, "c2": -0.1, "c3": 0.01},
                    {"c1": 1.2, "c2": -0.2, "c3": 0.02},
                ]
            }
        }
        for name in ("au", "a", "b", "c")
    }
    return {
        "material_law_contract_sha256": "1" * 64,
        "case_binding": {"case_file_sha256": "3" * 64},
        "material_axes": axes,
        "promotion": {"candidate_only": True, "optimizer_start_allowed": False},
    }


def _report(
    polarization: str,
    raw_path: Path,
    law: dict,
) -> dict:
    numerical = case_contract(ANCHOR_CASE)
    time_contract = dict(numerical["time_spec"])
    time_contract.update(time_step_s=1.0e-18, time_steps_total=100)
    law_file_audit = {
        "ready": True,
        "path": "/absolute/law.json",
        "expected_sha256": "2" * 64,
        "actual_sha256": "2" * 64,
        "material_law_contract_sha256": law[
            "material_law_contract_sha256"
        ],
    }
    return {
        "status": CASE_STATUS,
        "ready": True,
        "polarization": polarization,
        "scope": SCOPE,
        "numerical_case_contract": numerical,
        "numerical_case_file_audit": {
            "ready": True,
            "actual_sha256": "3" * 64,
            "case_contract_sha256": numerical["case_contract_sha256"],
        },
        "candidate_material_law_contract": law,
        "candidate_material_law_file_audit": law_file_audit,
        "candidate_source_model_audit": {
            "ready": True,
            "checks": {"all": True},
        },
        "pre_solve_checks": {"all": True},
        "mesh": numerical["resolved_mesh"],
        "time_contract": time_contract,
        "pml_face_parameters": numerical["resolved_pml_face_parameters"],
        "placement": expected_placement(ANCHOR_CASE.mesh),
        "source_contract": {
            "wavelength_m": 4.0e-6,
            "polarization": polarization,
            "fixed_E_polarization_vector": (
                [0.0, 1.0, 0.0]
                if polarization == "Ea"
                else [1.0, 0.0, 0.0]
            ),
            "direction": "-",
            "num_startup_periods": 4,
        },
        "all_air_material_readback": {"ready": True},
        "evaluation": {
            "ready": True,
            "gates": {"all": True},
            "flux": {"incident_plane_signed_W": 2.0e-12},
        },
        "reporting_incident_power_W": 285.0e-6,
        "per_case_scale_not_authorized_until_pair_comparison": True,
        "raw": {
            "path": str(raw_path.resolve()),
            "sha256": _sha256(raw_path),
            "arrays": {"target": [3, 2, 2, 1]},
        },
        "provenance": {
            "repository_commit": "candidate-source-commit",
            "repository_dirty_porcelain": "",
            "runner_sha256": "4" * 64,
            "implementation_sha256": {"model": "5" * 64},
            "fdtdx_source": {
                "path": "/absolute/fdtdx",
                "commit": "fdtdx-commit",
                "dirty_porcelain": "",
            },
            "runtime_lock": {"locked": True},
        },
        "optimizer_start_allowed": False,
    }


class FdtdxFreshTwoPoleSourcePairTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = {}
        self.reports = {}
        law = _law()
        for polarization in ("Ea", "Eb"):
            raw = self.root / f"{polarization}.npz"
            np.savez_compressed(
                raw, target=np.ones((3, 2, 2, 1), dtype=np.complex64)
            )
            self.paths[polarization] = self.root / f"{polarization}.json"
            self.reports[polarization] = _report(polarization, raw, law)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self) -> None:
        for polarization in ("Ea", "Eb"):
            self.paths[polarization].write_text(
                json.dumps(self.reports[polarization]), encoding="utf-8"
            )

    def test_matching_canonical_candidate_law_passes_pair(self) -> None:
        self._write()
        law = self.reports["Ea"]["candidate_material_law_contract"]
        with patch(
            "photothermal_pte.optimization_runs.au_dualpol_4um_current_switch."
            "fdtdx_fresh_two_pole_source_pair.material_law_from_contract",
            return_value=law,
        ):
            result = build_candidate_pair_certificate(
                self.paths["Ea"], self.paths["Eb"]
            )
        self.assertTrue(result["ready"], result["failed_gates"])
        self.assertEqual(result["status"], PAIR_STATUS)
        self.assertIn(
            "candidate_material_law_contract", result["source_case_contracts"]
        )

    def test_different_candidate_law_blocks_pair(self) -> None:
        changed = copy.deepcopy(
            self.reports["Eb"]["candidate_material_law_contract"]
        )
        changed["material_law_contract_sha256"] = "9" * 64
        self.reports["Eb"]["candidate_material_law_contract"] = changed
        self._write()
        result = build_candidate_pair_certificate(
            self.paths["Ea"], self.paths["Eb"]
        )
        self.assertFalse(result["ready"])
        self.assertEqual(result["status"], BLOCKED_STATUS)
        self.assertIn("candidate_material_law_audit_ready", result["failed_gates"])

    def test_written_pair_revalidates_exact_candidate_status_and_law(self) -> None:
        self._write()
        law = self.reports["Ea"]["candidate_material_law_contract"]
        output = self.root / "pair"
        output.mkdir()

        def clean_git(repository, *arguments):
            del repository
            return "candidate-certificate-commit" if arguments == ("rev-parse", "HEAD") else ""

        with patch(
            "photothermal_pte.optimization_runs.au_dualpol_4um_current_switch."
            "fdtdx_fresh_two_pole_source_pair.material_law_from_contract",
            return_value=law,
        ), patch(
            "photothermal_pte.optimization_runs.au_dualpol_4um_current_switch."
            "fdtdx_fresh_two_pole_source_pair._git",
            side_effect=clean_git,
        ):
            written = write_candidate_pair_certificate(
                self.paths["Ea"], self.paths["Eb"], output
            )
        self.assertTrue(written["ready"], written["failed_gates"])
        pair_path = output / "FDTDX_FRESH_TWO_POLE_SOURCE_ONLY_PAIR.json"
        payload, audit = validate_candidate_source_pair(
            pair_path,
            _sha256(pair_path),
            ANCHOR_CASE,
            law,
            self.reports["Ea"]["candidate_material_law_file_audit"],
        )
        self.assertEqual(payload["status"], PAIR_STATUS)
        self.assertTrue(audit["ready"], audit["failed_checks"])
        different = copy.deepcopy(law)
        different["material_law_contract_sha256"] = "8" * 64
        _, changed_audit = validate_candidate_source_pair(
            pair_path,
            _sha256(pair_path),
            ANCHOR_CASE,
            different,
            self.reports["Ea"]["candidate_material_law_file_audit"],
        )
        self.assertFalse(changed_audit["ready"])
        self.assertIn("candidate_material_law_exact", changed_audit["failed_checks"])

    def test_candidate_source_model_audit_checks_zero_air_arrays(self) -> None:
        law = _law()
        endpoints = {
            name: tuple(
                tuple(pole[key] for key in ("c1", "c2", "c3"))
                for pole in axis["candidate"]["poles"]
            )
            for name, axis in law["material_axes"].items()
        }
        zeros = np.zeros((2, 3, 2, 2, 2), dtype=np.float32)
        model = {
            "material_law_mode": "candidate-two-pole-contract",
            "material_law_contract_sha256": law["material_law_contract_sha256"],
            "num_dispersive_poles": 2,
            "coefficient_endpoints": endpoints,
            "fixed_c1": zeros,
            "fixed_c2": zeros,
            "fixed_c3": zeros,
            "jnp": np,
            "base": SimpleNamespace(dispersive_c4=None),
            "air_only_source_calibration": True,
            "slices": {},
            "jax": SimpleNamespace(devices=lambda: ("synthetic-device",)),
        }
        audit = candidate_source_model_audit(model, law)
        self.assertTrue(audit["ready"], audit["failed_checks"])
        changed = dict(model)
        changed["fixed_c1"] = np.ones_like(zeros)
        self.assertFalse(candidate_source_model_audit(changed, law)["ready"])


if __name__ == "__main__":
    unittest.main()
