from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
    mesh_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
    absorption_power_density,
    combined_weighted_nrmse,
    component_power,
    relative_difference,
    sha256,
    validate_source_pair,
)


class FdtdxFreshExactBinaryPilotTest(unittest.TestCase):
    def test_absorption_density_masks_every_Au_component_exactly(self) -> None:
        field = np.ones((3, 2, 2, 1), dtype=np.complex64) * (1.0 + 1.0j)
        mask = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
        q = absorption_power_density(field, 2.0, 3.0, mask)
        expected = 12.0 * mask[None, :, :, None]
        np.testing.assert_array_equal(q[:, mask == 0, :], 0.0)
        np.testing.assert_allclose(q, np.broadcast_to(expected, q.shape), rtol=1.0e-7)

    def test_absorption_rejects_float_occupancy_even_at_endpoints(self) -> None:
        field = np.ones((3, 2, 2, 1), dtype=np.complex64)
        with self.assertRaises(ValueError):
            absorption_power_density(
                field, 1.0, 1.0, np.asarray([[0.0, 1.0], [1.0, 0.0]])
            )

    def test_component_power_uses_component_specific_volumes(self) -> None:
        q = np.ones((3, 1, 1, 1), dtype=np.float64)
        volume = np.asarray([[[[1.0]]], [[[2.0]]], [[[3.0]]]])
        power = component_power(q, volume)
        self.assertEqual(power["component_W"], {"x": 1.0, "y": 2.0, "z": 3.0})
        self.assertEqual(power["total_W"], 6.0)

    def test_combined_q_nrmse_uses_physical_volumes(self) -> None:
        late = {"au": np.asarray([[[[2.0]]], [[[2.0]]], [[[2.0]]]])}
        previous = {"au": np.asarray([[[[1.0]]], [[[2.0]]], [[[2.0]]]])}
        volumes = {"au": np.asarray([[[[4.0]]], [[[1.0]]], [[[1.0]]]])}
        self.assertAlmostEqual(
            combined_weighted_nrmse(late, previous, volumes),
            np.sqrt(4.0 / 24.0),
        )

    def test_relative_difference_is_symmetric(self) -> None:
        self.assertEqual(relative_difference(2.0, 4.0), 0.5)
        self.assertEqual(relative_difference(4.0, 2.0), 0.5)

    def test_source_pair_revalidates_certificate_and_bound_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = {}
            raws = {}
            cases = {}
            for polarization in ("Ea", "Eb"):
                report = root / f"{polarization}.json"
                raw = root / f"{polarization}.npz"
                report.write_text("{}\n", encoding="utf-8")
                raw.write_bytes(polarization.encode("ascii"))
                reports[polarization] = report
                raws[polarization] = raw
                cases[polarization] = {
                    "report_path": str(report.resolve()),
                    "report_sha256": sha256(report),
                    "raw": {
                        "path": str(raw.resolve()),
                        "actual_sha256": sha256(raw),
                    },
                }
            generator = root / "generator.py"
            generator.write_text("# generator\n", encoding="utf-8")
            certificate = {
                "status": "VALIDATED_FDTDX_FRESH_SOURCE_ONLY_PAIR",
                "ready": True,
                "failed_gates": [],
                "gates": {"all": True},
                "normalization_policy": {
                    "per_polarization_power_matching_forbidden": True
                },
                "common_normalization": {
                    "reporting_target_incident_power_W": 285.0e-6,
                    "common_power_scale": 100.0,
                    "common_field_amplitude_scale": 10.0,
                },
                "comparison": {"mean_unscaled_incident_power_W": 2.85e-6},
                "cases": cases,
                "source_case_contracts": {"mesh": mesh_audit(MeshSpec())},
                "provenance": {
                    "certificate_generator_path": str(generator.resolve()),
                    "certificate_generator_sha256": sha256(generator),
                },
            }
            path = root / "pair.json"
            path.write_text(json.dumps(certificate), encoding="utf-8")
            payload, audit = validate_source_pair(path, sha256(path))
            self.assertTrue(audit["ready"])
            self.assertEqual(payload["status"], certificate["status"])

            _, wrong_hash = validate_source_pair(path, "0" * 64)
            self.assertFalse(wrong_hash["ready"])
            self.assertFalse(wrong_hash["checks"]["certificate_sha256_matches"])

            raw = raws["Eb"]
            raw.write_bytes(b"tampered")
            _, tampered = validate_source_pair(path, sha256(path))
            self.assertFalse(tampered["ready"])
            self.assertFalse(tampered["checks"]["case_raw_files_exist_and_match"])


if __name__ == "__main__":
    unittest.main()
