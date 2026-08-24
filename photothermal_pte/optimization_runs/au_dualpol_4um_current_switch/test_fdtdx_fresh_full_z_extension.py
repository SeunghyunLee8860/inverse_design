from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    file_sha256,
    load_case_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_full_z_extension_case import (
    EXTENSION_FACTORS,
    expected_extension_case,
    write_extension_case,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_full_z_extension_certificate import (
    ALLOWED_CROSS_COMMIT_PATHS,
    EXPECTED_PRIOR_FAILED_GATE,
    EXPECTED_PRIOR_STATUS,
    cross_commit_audit,
    prior_certificate_audit,
)


MODULE = (
    "photothermal_pte.optimization_runs.au_dualpol_4um_current_switch."
    "fdtdx_fresh_full_z_extension_certificate"
)


class FullZExtensionTest(unittest.TestCase):
    def test_extension_cases_change_only_z_factor_and_keep_time(self) -> None:
        cases = {
            level: expected_extension_case(level)
            for level in EXTENSION_FACTORS
        }
        first = cases["z16"]
        first_mesh = dict(first.mesh.__dict__)
        first_mesh.pop("z_factor")
        for level, factor in EXTENSION_FACTORS.items():
            mesh = dict(cases[level].mesh.__dict__)
            self.assertEqual(mesh.pop("z_factor"), factor)
            self.assertEqual(mesh, first_mesh)
            self.assertEqual(cases[level].time.total_periods, 24)
            self.assertEqual(cases[level].time.window_periods, 4)
            self.assertEqual(cases[level].time.courant_factor, 0.25)
    def test_z16_32_period_case_changes_only_total_duration(self) -> None:
        base = expected_extension_case("z16")
        longer = expected_extension_case("z16", 32)
        self.assertEqual(base.mesh, longer.mesh)
        self.assertEqual(base.pml_alpha_scale, longer.pml_alpha_scale)
        self.assertEqual(
            base.pml_target_reflection, longer.pml_target_reflection
        )
        self.assertEqual(base.time.window_periods, longer.time.window_periods)
        self.assertEqual(base.time.courant_factor, longer.time.courant_factor)
        self.assertEqual(
            base.time.source_startup_periods,
            longer.time.source_startup_periods,
        )
        self.assertEqual(base.time.total_periods, 24)
        self.assertEqual(longer.time.total_periods, 32)
        with self.assertRaises(ValueError):
            expected_extension_case("z16", 40)

    def test_extension_case_writer_is_canonical_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "z16.json"
            result = write_extension_case("z16", output)
            spec, payload, audit = load_case_contract(
                output, result["file_sha256"]
            )
            self.assertEqual(spec, expected_extension_case("z16"))
            self.assertTrue(audit["ready"])
            self.assertEqual(
                payload["resolved_mesh"]["grid_shape_xyz"],
                [196, 196, 640],
            )
            self.assertEqual(result["file_sha256"], file_sha256(output))
            with self.assertRaises(RuntimeError):
                write_extension_case("z16", output)

    def test_32_period_writer_is_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "z16_t32.json"
            result = write_extension_case("z16", output, 32)
            spec, payload, audit = load_case_contract(
                output, result["file_sha256"]
            )
            self.assertEqual(spec, expected_extension_case("z16", 32))
            self.assertEqual(result["total_periods"], 32)
            self.assertEqual(payload["time_spec"]["total_periods"], 32)
            self.assertTrue(audit["ready"])


    @staticmethod
    def _prior_payload() -> dict:
        return {
            "status": EXPECTED_PRIOR_STATUS,
            "ready": False,
            "gates": {
                "artifact_gate": True,
                EXPECTED_PRIOR_FAILED_GATE: False,
            },
            "failed_gates": [EXPECTED_PRIOR_FAILED_GATE],
            "successive_comparisons": {
                "z2_to_z4": {"pass": False},
                "z4_to_z8": {"pass": False},
            },
            "optimizer_start_allowed": False,
            "is_mesh_certificate": False,
            "is_full_domain_z_resolution_certificate": False,
            "provenance": {
                "certificate_repository_dirty_porcelain": "",
            },
        }

    def test_prior_blocked_certificate_is_revalidated_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "prior.json"
            path.write_text(
                json.dumps(self._prior_payload()),
                encoding="utf-8",
            )
            _, audit = prior_certificate_audit(
                path, file_sha256(path), root
            )
            self.assertTrue(audit["ready"])
            tampered = self._prior_payload()
            tampered["successive_comparisons"]["z4_to_z8"]["pass"] = True
            path.write_text(json.dumps(tampered), encoding="utf-8")
            _, audit = prior_certificate_audit(
                path, file_sha256(path), root
            )
            self.assertFalse(audit["ready"])
            self.assertFalse(
                audit["checks"]["both_original_pairs_were_retained_and_failed"]
            )

    def test_cross_commit_audit_allows_only_non_runner_paths(self) -> None:
        allowed = next(iter(ALLOWED_CROSS_COMMIT_PATHS))
        with patch(f"{MODULE}._git", return_value=allowed):
            audit = cross_commit_audit(
                Path("/repository"), {"a" * 40, "b" * 40}
            )
        self.assertTrue(audit["ready"])
        with patch(
            f"{MODULE}._git",
            return_value=(
                "photothermal_pte/optimization_runs/"
                "au_dualpol_4um_current_switch/fdtdx_fresh_source_only.py"
            ),
        ):
            audit = cross_commit_audit(
                Path("/repository"), {"a" * 40, "b" * 40}
            )
        self.assertFalse(audit["ready"])

    def test_cross_commit_audit_requires_two_distinct_commits(self) -> None:
        audit = cross_commit_audit(Path("/repository"), {"a" * 40})
        self.assertFalse(audit["ready"])


if __name__ == "__main__":
    unittest.main()
