from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    FreshCaseSpec,
    TimeSpec,
    case_contract,
    file_sha256,
    load_case_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_material_contract import (
    AXIS_TO_SOLVER_COMPONENT,
    material_law_contract,
    material_law_from_contract,
    load_material_law_contract,
)


def spec(z_factor: int) -> FreshCaseSpec:
    return FreshCaseSpec(
        mesh=MeshSpec(z_factor=z_factor),
        time=TimeSpec(
            total_periods=24,
            window_periods=4,
            courant_factor=0.25,
        ),
    )


class FdtdxFreshTwoPoleMaterialContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.fdtdx_source = Path(cls.temporary.name).resolve() / "fdtdx"
        (cls.fdtdx_source / "src/fdtdx/fdtd").mkdir(parents=True)
        (cls.fdtdx_source / "src/fdtdx/fdtd/update.py").write_text(
            "# pinned synthetic update\n", encoding="utf-8"
        )
        (cls.fdtdx_source / "src/fdtdx/dispersion.py").write_text(
            "# pinned synthetic dispersion\n", encoding="utf-8"
        )
        cls.spec = spec(16)
        cls.case = case_contract(cls.spec)
        cls.case_file_sha = "1" * 64
        cls.law = material_law_contract(
            cls.spec, cls.case, cls.case_file_sha, cls.fdtdx_source
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_z16_contract_is_candidate_only_and_covers_all_axes(self) -> None:
        self.assertEqual(
            self.law["material_binding"]["tairte4_crystal_to_solver_axis"],
            AXIS_TO_SOLVER_COMPONENT,
        )
        self.assertEqual(set(self.law["material_axes"]), {"au", "a", "b", "c"})
        self.assertTrue(all(self.law["checks"].values()))
        self.assertTrue(self.law["promotion"]["candidate_only"])
        self.assertFalse(self.law["promotion"]["is_material_certificate"])
        self.assertFalse(self.law["promotion"]["optimizer_start_allowed"])
        for item in self.law["material_axes"].values():
            candidate = item["candidate"]
            self.assertTrue(candidate["fit_gate_passed"])
            self.assertEqual(len(candidate["poles"]), 2)
            for pole in candidate["poles"]:
                self.assertTrue(pole["positive_strength"])
                self.assertTrue(pole["recurrence_roots_not_above_one"])

    def test_exact_reconstruction_rejects_tamper_or_different_case_hash(self) -> None:
        self.assertEqual(
            material_law_from_contract(
                self.law,
                self.spec,
                self.case,
                self.case_file_sha,
                self.fdtdx_source,
            ),
            self.law,
        )
        tampered = copy.deepcopy(self.law)
        tampered["material_axes"]["au"]["candidate"]["poles"][0]["c3"] *= 2
        with self.assertRaises(ValueError):
            material_law_from_contract(
                tampered,
                self.spec,
                self.case,
                self.case_file_sha,
                self.fdtdx_source,
            )
        with self.assertRaises(ValueError):
            material_law_from_contract(
                self.law,
                self.spec,
                self.case,
                "2" * 64,
                self.fdtdx_source,
            )

    def test_file_loader_requires_absolute_path_and_byte_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            case_path = root / "case.json"
            case_path.write_text(
                json.dumps(self.case, sort_keys=True) + "\n", encoding="utf-8"
            )
            loaded_spec, loaded_case, case_audit = load_case_contract(
                case_path, file_sha256(case_path)
            )
            law = material_law_contract(
                loaded_spec,
                loaded_case,
                case_audit["actual_sha256"],
                self.fdtdx_source,
            )
            law_path = root / "law.json"
            law_path.write_text(
                json.dumps(law, sort_keys=True) + "\n", encoding="utf-8"
            )
            loaded, audit = load_material_law_contract(
                law_path,
                file_sha256(law_path),
                loaded_spec,
                loaded_case,
                case_audit["actual_sha256"],
                self.fdtdx_source,
            )
            self.assertEqual(loaded, law)
            self.assertTrue(audit["ready"])
            with self.assertRaises(RuntimeError):
                load_material_law_contract(
                    law_path,
                    "0" * 64,
                    loaded_spec,
                    loaded_case,
                    case_audit["actual_sha256"],
                    self.fdtdx_source,
                )

    def test_unsupported_mesh_or_time_fails_closed(self) -> None:
        unsupported_mesh = FreshCaseSpec(
            mesh=MeshSpec(z_factor=16, design_xy_factor=2),
            time=self.spec.time,
        )
        with self.assertRaises(RuntimeError):
            material_law_contract(
                unsupported_mesh,
                case_contract(unsupported_mesh),
                self.case_file_sha,
                self.fdtdx_source,
            )
        unsupported_time = FreshCaseSpec(
            mesh=MeshSpec(z_factor=16),
            time=TimeSpec(total_periods=32, window_periods=4, courant_factor=0.25),
        )
        with self.assertRaises(RuntimeError):
            material_law_contract(
                unsupported_time,
                case_contract(unsupported_time),
                self.case_file_sha,
                self.fdtdx_source,
            )


if __name__ == "__main__":
    unittest.main()
