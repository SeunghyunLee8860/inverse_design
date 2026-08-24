from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    ANCHOR_CASE,
    FreshCaseSpec,
    TimeSpec,
    case_contract,
    file_sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
    validate_source_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_only import (
    resolve_case_input,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_pair import (
    build_pair_certificate,
)


def _source_report(
    polarization: str,
    raw_path: Path,
    power: float,
    numerical_case: dict,
) -> dict:
    time_contract = dict(numerical_case["time_spec"])
    time_contract.update(time_step_s=1e-18, time_steps_total=100)
    return {
        "status": "VALIDATED_FDTDX_FRESH_SOURCE_ONLY_CASE",
        "ready": True,
        "polarization": polarization,
        "scope": "all-air source-only for one hashed fresh numerical contract",
        "numerical_case_contract": numerical_case,
        "numerical_case_file_audit": {
            "ready": True,
            "case_contract_sha256": numerical_case["case_contract_sha256"],
        },
        "mesh": numerical_case["resolved_mesh"],
        "time_contract": time_contract,
        "pml_face_parameters": numerical_case["resolved_pml_face_parameters"],
        "placement": {"gaussian_source": [[1, 2], [1, 2], [3, 4]]},
        "source_contract": {
            "wavelength_m": 4e-6,
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
            "flux": {"incident_plane_signed_W": power},
        },
        "reporting_incident_power_W": 285e-6,
        "per_case_scale_not_authorized_until_pair_comparison": True,
        "raw": {
            "path": str(raw_path.resolve()),
            "sha256": file_sha256(raw_path),
            "arrays": {"target": [3, 2, 2, 1]},
        },
        "provenance": {
            "repository_commit": "clean-source-commit",
            "repository_dirty_porcelain": "",
            "fdtdx_source": {
                "commit": "pinned-fdtdx-commit",
                "dirty_porcelain": "",
            },
            "runtime_lock": {"locked": True},
        },
    }


class FreshRunnerCaseBindingTest(unittest.TestCase):
    def test_case_input_requires_path_and_hash_together(self) -> None:
        with self.assertRaises(ValueError):
            resolve_case_input(None, None)
        with self.assertRaises(ValueError):
            resolve_case_input(Path("/tmp/case.json"), None)
        with self.assertRaises(ValueError):
            resolve_case_input(None, "0" * 64)

    def test_external_case_input_is_bound_to_the_file_byte_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "case.json"
            requested = FreshCaseSpec(time=TimeSpec(total_periods=24))
            path.write_text(json.dumps(case_contract(requested)), encoding="utf-8")
            loaded, payload, audit = resolve_case_input(path, file_sha256(path))
            self.assertEqual(loaded, requested)
            self.assertEqual(audit["actual_sha256"], file_sha256(path))
            self.assertEqual(
                audit["case_contract_sha256"], payload["case_contract_sha256"]
            )

    def test_source_pair_rejects_different_or_jointly_noncanonical_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ea_raw = root / "ea.npz"
            eb_raw = root / "eb.npz"
            ea_raw.write_bytes(b"ea")
            eb_raw.write_bytes(b"eb")
            anchor = case_contract(ANCHOR_CASE)
            ea = _source_report("Ea", ea_raw, 2e-12, anchor)
            eb = _source_report("Eb", eb_raw, 2e-12, anchor)
            ea_path = root / "ea.json"
            eb_path = root / "eb.json"
            ea_path.write_text(json.dumps(ea), encoding="utf-8")
            eb_path.write_text(json.dumps(eb), encoding="utf-8")
            self.assertTrue(build_pair_certificate(ea_path, eb_path)["ready"])

            eb_different = copy.deepcopy(eb)
            eb_different["numerical_case_contract"] = case_contract(
                FreshCaseSpec(time=TimeSpec(total_periods=24))
            )
            eb_path.write_text(json.dumps(eb_different), encoding="utf-8")
            result = build_pair_certificate(ea_path, eb_path)
            self.assertFalse(result["ready"])
            self.assertFalse(result["gates"]["numerical_case_contract_identical"])

            jointly_tampered = copy.deepcopy(anchor)
            jointly_tampered["time_spec"]["total_periods"] = 32
            for path, report in ((ea_path, ea), (eb_path, eb)):
                report = copy.deepcopy(report)
                report["numerical_case_contract"] = jointly_tampered
                report["numerical_case_file_audit"]["case_contract_sha256"] = (
                    jointly_tampered["case_contract_sha256"]
                )
                path.write_text(json.dumps(report), encoding="utf-8")
            result = build_pair_certificate(ea_path, eb_path)
            self.assertFalse(result["ready"])
            self.assertFalse(result["gates"]["numerical_case_contract_canonical"])

    def test_pilot_rejects_a_valid_pair_for_a_different_time_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raws = {polarization: root / f"{polarization}.npz" for polarization in ("Ea", "Eb")}
            reports = {polarization: root / f"{polarization}.json" for polarization in ("Ea", "Eb")}
            anchor = case_contract(ANCHOR_CASE)
            for polarization in ("Ea", "Eb"):
                raws[polarization].write_bytes(polarization.encode("ascii"))
                report = _source_report(polarization, raws[polarization], 2e-12, anchor)
                reports[polarization].write_text(json.dumps(report), encoding="utf-8")
            pair = build_pair_certificate(reports["Ea"], reports["Eb"])
            self.assertTrue(pair["ready"])
            generator = root / "generator.py"
            generator.write_text("# generator\n", encoding="utf-8")
            pair["provenance"] = {
                "certificate_generator_path": str(generator.resolve()),
                "certificate_generator_sha256": file_sha256(generator),
            }
            pair_path = root / "pair.json"
            pair_path.write_text(json.dumps(pair), encoding="utf-8")

            different_time = FreshCaseSpec(time=TimeSpec(total_periods=24))
            _, audit = validate_source_pair(
                pair_path,
                file_sha256(pair_path),
                different_time,
            )
            self.assertFalse(audit["ready"])
            self.assertFalse(audit["checks"]["numerical_case_contract_exact"])
            self.assertFalse(audit["checks"]["time_request_exact"])


if __name__ == "__main__":
    unittest.main()
