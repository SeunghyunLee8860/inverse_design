from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_pair import (
    BLOCKED_STATUS,
    PAIR_STATUS,
    POWER_MISMATCH_RELATIVE_LIMIT,
    build_pair_certificate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(polarization: str, raw_path: Path, power: float) -> dict:
    return {
        "status": "VALIDATED_FDTDX_FRESH_SOURCE_ONLY_CASE",
        "ready": True,
        "polarization": polarization,
        "scope": "all-air source-only on validated fresh anchor",
        "mesh": {"grid_contract_sha256": "mesh"},
        "time_contract": {"total_periods": 16, "time_steps_total": 100},
        "pml_face_parameters": {"minx": {"alpha_start": 1.0}},
        "placement": {"gaussian_source": [[1, 2], [1, 2], [3, 4]]},
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
        "all_air_material_readback": {"ready": True, "c1_unique": [0.0]},
        "evaluation": {
            "ready": True,
            "gates": {"stationarity": True, "polarization": True},
            "flux": {"incident_plane_signed_W": power},
        },
        "reporting_incident_power_W": 285.0e-6,
        "per_case_scale_not_authorized_until_pair_comparison": True,
        "raw": {
            "path": str(raw_path.resolve()),
            "sha256": _sha256(raw_path),
            "arrays": {"target": [3, 2, 2, 1]},
        },
        "provenance": {
            "repository_commit": "source-commit",
            "repository_dirty_porcelain": "",
            "fdtdx_source": {
                "commit": "fdtdx-commit",
                "dirty_porcelain": "",
            },
            "runtime_lock": {"jax": "locked"},
        },
    }


class FdtdxFreshSourcePairTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ea_raw = self.root / "ea.npz"
        self.eb_raw = self.root / "eb.npz"
        self.ea_raw.write_bytes(b"ea raw")
        self.eb_raw.write_bytes(b"eb raw")
        self.ea_report = self.root / "ea.json"
        self.eb_report = self.root / "eb.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(
        self,
        ea: dict,
        eb: dict,
    ) -> None:
        self.ea_report.write_text(json.dumps(ea), encoding="utf-8")
        self.eb_report.write_text(json.dumps(eb), encoding="utf-8")

    def test_valid_pair_uses_one_common_power_and_field_scale(self) -> None:
        ea = _payload("Ea", self.ea_raw, 2.000e-12)
        eb = _payload("Eb", self.eb_raw, 2.004e-12)
        self._write(ea, eb)
        result = build_pair_certificate(self.ea_report, self.eb_report)
        normalization = result["common_normalization"]
        expected_scale = 285.0e-6 / 2.002e-12
        self.assertTrue(result["ready"])
        self.assertEqual(result["status"], PAIR_STATUS)
        self.assertAlmostEqual(normalization["common_power_scale"], expected_scale)
        self.assertAlmostEqual(
            normalization["common_field_amplitude_scale"], expected_scale**0.5
        )
        self.assertLess(
            result["comparison"]["relative_power_mismatch"],
            POWER_MISMATCH_RELATIVE_LIMIT,
        )

    def test_power_mismatch_above_limit_blocks_pair(self) -> None:
        ea = _payload("Ea", self.ea_raw, 2.0e-12)
        eb = _payload("Eb", self.eb_raw, 2.1e-12)
        self._write(ea, eb)
        result = build_pair_certificate(self.ea_report, self.eb_report)
        self.assertFalse(result["ready"])
        self.assertEqual(result["status"], BLOCKED_STATUS)
        self.assertIn("source_power_relative_mismatch", result["failed_gates"])

    def test_wrong_polarization_labels_block_pair(self) -> None:
        ea = _payload("Eb", self.ea_raw, 2.0e-12)
        eb = _payload("Ea", self.eb_raw, 2.0e-12)
        self._write(ea, eb)
        result = build_pair_certificate(self.ea_report, self.eb_report)
        self.assertFalse(result["ready"])
        self.assertIn("expected_polarizations", result["failed_gates"])

    def test_raw_hash_mismatch_blocks_pair(self) -> None:
        ea = _payload("Ea", self.ea_raw, 2.0e-12)
        eb = _payload("Eb", self.eb_raw, 2.0e-12)
        tampered = copy.deepcopy(eb)
        tampered["raw"]["sha256"] = "0" * 64
        self._write(ea, tampered)
        result = build_pair_certificate(self.ea_report, self.eb_report)
        self.assertFalse(result["ready"])
        self.assertIn("raw_sha256_matches", result["failed_gates"])

    def test_pml_mismatch_blocks_pair(self) -> None:
        ea = _payload("Ea", self.ea_raw, 2.0e-12)
        eb = _payload("Eb", self.eb_raw, 2.0e-12)
        eb["pml_face_parameters"]["minx"]["alpha_start"] = 2.0
        self._write(ea, eb)
        result = build_pair_certificate(self.ea_report, self.eb_report)
        self.assertFalse(result["ready"])
        self.assertIn("pml_face_parameters_identical", result["failed_gates"])

    def test_common_source_mismatch_blocks_pair(self) -> None:
        ea = _payload("Ea", self.ea_raw, 2.0e-12)
        eb = _payload("Eb", self.eb_raw, 2.0e-12)
        eb["source_contract"]["direction"] = "+"
        self._write(ea, eb)
        result = build_pair_certificate(self.ea_report, self.eb_report)
        self.assertFalse(result["ready"])
        self.assertIn("common_source_contract_identical", result["failed_gates"])

    def test_mesh_mismatch_blocks_pair(self) -> None:
        ea = _payload("Ea", self.ea_raw, 2.0e-12)
        eb = _payload("Eb", self.eb_raw, 2.0e-12)
        eb["mesh"] = {"grid_contract_sha256": "different"}
        self._write(ea, eb)
        result = build_pair_certificate(self.ea_report, self.eb_report)
        self.assertFalse(result["ready"])
        self.assertIn("mesh_contract_identical", result["failed_gates"])


if __name__ == "__main__":
    unittest.main()
