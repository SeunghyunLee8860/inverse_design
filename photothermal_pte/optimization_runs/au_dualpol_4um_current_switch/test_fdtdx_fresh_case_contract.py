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
    case_for_axis,
    case_from_contract,
    file_sha256,
    load_case_contract,
)


class FreshCaseContractTest(unittest.TestCase):
    def test_anchor_contract_is_canonical_self_hashed_and_round_trips(self) -> None:
        payload = case_contract(ANCHOR_CASE)
        self.assertEqual(case_from_contract(payload), ANCHOR_CASE)
        self.assertEqual(payload["resolved_mesh"]["grid_shape_xyz"], [196, 196, 160])
        self.assertEqual(payload["resolved_mesh"]["yee_cell_count"], 6_146_560)
        self.assertEqual(len(payload["case_contract_sha256"]), 64)
        self.assertFalse(payload["rules"]["optimizer_start_allowed"])

    def test_time_contract_rejects_overlap_unsettled_startup_and_bad_courant(self) -> None:
        with self.assertRaises(ValueError):
            TimeSpec(total_periods=8, window_periods=4)
        with self.assertRaises(ValueError):
            TimeSpec(total_periods=10, window_periods=4)
        with self.assertRaises(ValueError):
            TimeSpec(source_startup_periods=3)
        with self.assertRaises(ValueError):
            TimeSpec(courant_factor=0.0)

    def test_axis_case_changes_requested_level_and_preserves_time_and_pml(self) -> None:
        time = TimeSpec(total_periods=24, window_periods=4, courant_factor=0.375)
        case = case_for_axis(
            "z_pml_thickness",
            1,
            time=time,
            pml_alpha_scale=2.0,
            pml_target_reflection=1e-7,
        )
        self.assertEqual(case.time, time)
        self.assertEqual(case.mesh.z_pml_thickness_m, 2.4e-6)
        self.assertEqual(case.pml_alpha_scale, 2.0)
        payload = case_contract(case)
        self.assertEqual(payload["resolved_mesh"]["grid_shape_xyz"], [196, 196, 192])
        self.assertNotEqual(
            payload["resolved_pml_face_parameters"]["minz"]["sigma_end"],
            case_contract(ANCHOR_CASE)["resolved_pml_face_parameters"]["minz"][
                "sigma_end"
            ],
        )

    def test_noncanonical_extra_or_tampered_resolved_data_fails_closed(self) -> None:
        payload = case_contract(ANCHOR_CASE)
        extra = copy.deepcopy(payload)
        extra["unexpected"] = True
        with self.assertRaises(ValueError):
            case_from_contract(extra)
        tampered = copy.deepcopy(payload)
        tampered["resolved_mesh"]["yee_cell_count"] += 1
        with self.assertRaises(ValueError):
            case_from_contract(tampered)
        gray = copy.deepcopy(payload)
        gray["mesh_spec"]["design_xy_factor"] = 1.5
        with self.assertRaises(ValueError):
            case_from_contract(gray)

    def test_file_loader_requires_absolute_path_byte_hash_and_canonical_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "case.json"
            payload = case_contract(
                FreshCaseSpec(time=TimeSpec(total_periods=24))
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            expected_sha = file_sha256(path)
            spec, loaded, audit = load_case_contract(path, expected_sha)
            self.assertEqual(spec.time.total_periods, 24)
            self.assertEqual(loaded, payload)
            self.assertTrue(audit["ready"])
            with self.assertRaises(RuntimeError):
                load_case_contract(path, "0" * 64)
            relative = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else None
            if relative is not None:
                with self.assertRaises(RuntimeError):
                    load_case_contract(relative, expected_sha)
            payload["time_spec"]["total_periods"] = 32
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_case_contract(path, file_sha256(path))

    def test_anchor_has_only_level_zero(self) -> None:
        with self.assertRaises(ValueError):
            case_for_axis(
                "anchor",
                1,
                time=TimeSpec(),
                pml_alpha_scale=1.0,
                pml_target_reflection=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
