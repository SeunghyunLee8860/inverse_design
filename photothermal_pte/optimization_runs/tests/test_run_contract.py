from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from photothermal_pte.optimization_runs.run_contract import (
    ValidationError,
    validate_run_directory,
)


class OptimizationRunContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository_root = Path(__file__).resolve().parents[3]
        cls.baseline = (
            cls.repository_root
            / "photothermal_pte"
            / "optimization_runs"
            / "run_001_baseline_p1"
        )
        cls.gaussian10 = (
            cls.repository_root
            / "photothermal_pte"
            / "optimization_runs"
            / "run_002_gaussian10_w8p5_current_max"
        )

    def test_baseline_repository_contract(self) -> None:
        result = validate_run_directory(
            self.baseline,
            repository_root=self.repository_root,
            require_external=False,
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.repository_artifacts_checked, 6)
        self.assertEqual(result.status, "PLANNED")

    def test_periodic_boundary_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "run_999_bad_periodic"
            shutil.copytree(self.baseline, copied)
            config_path = copied / "run_config.json"
            config = json.loads(config_path.read_text())
            config["run_id"] = copied.name
            config["optical"]["periodic"] = True
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            status_path = copied / "STATUS.json"
            status = json.loads(status_path.read_text())
            status["run_id"] = copied.name
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            with self.assertRaisesRegex(ValidationError, "six-PML"):
                validate_run_directory(
                    copied,
                    repository_root=self.repository_root,
                )

    def test_gaussian10_repository_contract(self) -> None:
        result = validate_run_directory(
            self.gaussian10,
            repository_root=self.repository_root,
            require_external=False,
        )
        self.assertTrue(result.valid)
        self.assertEqual(
            result.status,
            "PRODUCTION_CANDIDATE_FORWARD_VALIDATED",
        )

    def test_gaussian10_lossless_sio2_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "run_997_bad_sio2"
            shutil.copytree(self.gaussian10, copied)
            config_path = copied / "run_config.json"
            config = json.loads(config_path.read_text())
            config["run_id"] = copied.name
            config["optical"]["sio2_optical_model"] = "lossless_n1.38"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            status_path = copied / "STATUS.json"
            status = json.loads(status_path.read_text())
            status["run_id"] = copied.name
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            with self.assertRaisesRegex(ValidationError, "lossless"):
                validate_run_directory(copied, repository_root=self.repository_root)

    def test_repository_sha_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "run_998_bad_sha"
            shutil.copytree(self.baseline, copied)
            config_path = copied / "run_config.json"
            config = json.loads(config_path.read_text())
            config["run_id"] = copied.name
            config["repository_inputs"] = deepcopy(
                config["repository_inputs"]
            )
            config["repository_inputs"][0]["sha256"] = "0" * 64
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            status_path = copied / "STATUS.json"
            status = json.loads(status_path.read_text())
            status["run_id"] = copied.name
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            with self.assertRaisesRegex(ValidationError, "SHA mismatch"):
                validate_run_directory(
                    copied,
                    repository_root=self.repository_root,
                )


if __name__ == "__main__":
    unittest.main()
