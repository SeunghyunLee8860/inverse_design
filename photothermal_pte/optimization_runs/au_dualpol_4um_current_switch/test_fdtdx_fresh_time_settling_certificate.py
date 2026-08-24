from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
    grid_edges,
    reference_mask,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_material import (
    mask_material_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    FreshCaseSpec,
    TimeSpec,
    case_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
    STATUS_READY as CASE_STATUS_READY,
    combined_weighted_nrmse,
    component_power,
    relative_difference,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_metrics import (
    weighted_complex_nrmse,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_time_settling_certificate import (
    EXACT_BINARY_CHECKS,
    EXPECTED_SCOPE,
    MATERIAL_CHECKS,
    PILOT_EVALUATION_GATES,
    REFERENCE_NAME,
    SOURCE_PAIR_CONTRACT_CHECKS,
    _material_case_audit,
    audit_raw_case,
    compare_time_pair,
    electric_yee_volumes_from_edges,
    expected_case,
    fixed_probe_and_weights,
    settling_selection_gates,
    sha256,
)


class TimeSettlingRawAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.spec = expected_case(24)
        self.contract_path = self.root / "contract.json"
        self.contract_path.write_text(
            json.dumps(case_contract(self.spec), sort_keys=True), encoding="utf-8"
        )
        self.contract_sha = sha256(self.contract_path)
        self.source_pair_path = self.root / "source_pair.json"
        self.source_pair_path.write_text("{}\n", encoding="utf-8")
        self.source_pair_sha = "a" * 64
        self.runner = self.root / "runner.py"
        self.runner.write_text("# fixed runner\n", encoding="utf-8")
        self.material_contract = self.root / "material.json"
        self.material_contract.write_text("{}\n", encoding="utf-8")
        self.raw_path = self.root / "fields.npz"
        self.report_path = self.root / "report.json"
        self.payload = self._write_valid_case()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _placement() -> dict[str, list[list[int]]]:
        return {
            "au_design": [[58, 138], [58, 138], [84, 85]],
            "fixed_tairte4": [[0, 2], [0, 2], [64, 65]],
            "target_field": [[58, 138], [58, 138], [108, 109]],
        }

    def _write_valid_case(self) -> dict:
        edges = tuple(
            np.asarray(value, dtype=np.float32) for value in grid_edges(MeshSpec())
        )
        placement = self._placement()
        au_bounds = tuple(tuple(value) for value in placement["au_design"])
        ta_bounds = tuple(tuple(value) for value in placement["fixed_tairte4"])
        au_volume = electric_yee_volumes_from_edges(edges, au_bounds)
        ta_volume = electric_yee_volumes_from_edges(edges, ta_bounds)
        mask = np.asarray(reference_mask(REFERENCE_NAME), dtype=np.uint8)
        au_field = np.ones(au_volume.shape, dtype=np.complex64)
        ta_field = np.full(ta_volume.shape, 2.0 + 0.5j, dtype=np.complex64)
        q_au = (
            np.broadcast_to(mask[None, :, :, None], au_volume.shape).astype(np.float64)
            * 0.25
        )
        q_ta = np.full(ta_volume.shape, 0.5, dtype=np.float64)
        target = np.ones((3, 80, 80, 1), dtype=np.complex64)
        arrays = {
            "design_mask": mask,
            "solver_mask": mask.copy(),
            "au_previous": au_field.copy(),
            "au_late": au_field.copy(),
            "tairte4_previous": ta_field.copy(),
            "tairte4_late": ta_field.copy(),
            "target": target,
            "q_au_previous_W_m3": q_au.copy(),
            "q_au_late_W_m3": q_au.copy(),
            "q_tairte4_previous_W_m3": q_ta.copy(),
            "q_tairte4_late_W_m3": q_ta.copy(),
            "electric_dual_volume_au_m3": au_volume,
            "electric_dual_volume_tairte4_m3": ta_volume,
            "grid_x_edges_m": edges[0],
            "grid_y_edges_m": edges[1],
            "grid_z_edges_m": edges[2],
        }
        np.savez_compressed(self.raw_path, **arrays)
        au_power = component_power(q_au, au_volume)
        ta_power = component_power(q_ta, ta_volume)
        power = {
            "by_material": {"au": au_power, "tairte4": ta_power},
            "total_W": au_power["total_W"] + ta_power["total_W"],
        }
        exact_binary = mask_material_audit(mask, self.spec.mesh)
        exact_binary.update(
            {
                "ready": True,
                "checks": {name: True for name in EXACT_BINARY_CHECKS},
            }
        )
        numerical_case = case_contract(self.spec)
        gates = {name: True for name in PILOT_EVALUATION_GATES}
        payload = {
            "status": CASE_STATUS_READY,
            "ready": True,
            "scope": EXPECTED_SCOPE,
            "reference": REFERENCE_NAME,
            "polarization": "Ea",
            "numerical_case_contract": numerical_case,
            "numerical_case_file_audit": {
                "path": str(self.contract_path),
                "expected_sha256": self.contract_sha,
                "actual_sha256": self.contract_sha,
                "case_contract_sha256": numerical_case["case_contract_sha256"],
                "ready": True,
                "checks": {"canonical": True},
            },
            "mesh": numerical_case["resolved_mesh"],
            "time_contract": {
                "total_periods": 24,
                "window_periods": 4,
                "source_startup_periods": 4,
                "courant_factor": 0.5,
                "time_step_s": 1.0e-18,
                "time_steps_total": 2400,
            },
            "pml_face_parameters": numerical_case["resolved_pml_face_parameters"],
            "placement": placement,
            "source_contract": {
                "polarization": "Ea",
                "fixed_E_polarization_vector": [0.0, 1.0, 0.0],
            },
            "source_pair": {
                "path": str(self.source_pair_path),
                "expected_sha256": self.source_pair_sha,
                "actual_sha256": self.source_pair_sha,
                "ready": True,
                "checks": {"bound": True},
                "failed_checks": [],
            },
            "source_pair_contract_checks": {
                name: True for name in SOURCE_PAIR_CONTRACT_CHECKS
            },
            "material": {
                "ready": True,
                "checks": {name: True for name in MATERIAL_CHECKS},
                "failed_checks": [],
                "exact_binary_au": exact_binary,
                "realized_material_response": {"au": "fixed", "a": "fixed"},
            },
            "evaluation": {
                "ready": True,
                "gates": gates,
                "failed_gates": [],
                "field_stationarity": {
                    "au_complex_E_NRMSE": weighted_complex_nrmse(
                        au_field, au_field, au_volume
                    ),
                    "tairte4_complex_E_NRMSE": weighted_complex_nrmse(
                        ta_field, ta_field, ta_volume
                    ),
                    "maximum_complex_E_NRMSE": 0.0,
                },
                "Q": {
                    "previous": copy.deepcopy(power),
                    "late": copy.deepcopy(power),
                    "previous_late_spatial_NRMSE": combined_weighted_nrmse(
                        {"au": q_au, "tairte4": q_ta},
                        {"au": q_au, "tairte4": q_ta},
                        {"au": au_volume, "tairte4": ta_volume},
                    ),
                    "previous_late_total_relative_change": relative_difference(
                        power["total_W"], power["total_W"]
                    ),
                },
                "flux": {
                    "Q_vs_closed_phasor_symmetric_relative": 1.0e-3,
                    "Q_vs_closed_td_symmetric_relative": 1.0e-3,
                },
            },
            "raw": {
                "path": str(self.raw_path),
                "sha256": sha256(self.raw_path),
                "arrays": {name: list(value.shape) for name, value in arrays.items()},
            },
            "normalization_policy": {
                "raw_fields_and_Q_are_unscaled": True,
                "per_polarization_matching_forbidden": True,
            },
            "provenance": {
                "repository_commit": "material-commit",
                "repository_dirty_porcelain": "",
                "fdtdx_source": {"commit": "fdtdx-commit", "dirty_porcelain": ""},
                "runtime_lock": {"runtime": "fixed"},
                "runner_path": str(self.runner),
                "runner_sha256": sha256(self.runner),
                "material_contract_path": str(self.material_contract),
                "material_contract_sha256": sha256(self.material_contract),
            },
            "optimizer_start_allowed": False,
        }
        self.report_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _audit_material(self) -> dict:
        _, audit, _ = _material_case_audit(
            self.report_path,
            self.root,
            24,
            "Ea",
            self.spec,
            self.contract_path,
            self.contract_sha,
            self.source_pair_path,
            self.source_pair_sha,
        )
        return audit

    def _mutate_report(self, mutate) -> None:
        payload = json.loads(self.report_path.read_text(encoding="utf-8"))
        mutate(payload)
        self.report_path.write_text(json.dumps(payload), encoding="utf-8")

    def test_valid_case_recomputes_raw_physics_and_passes(self) -> None:
        raw, snapshot = audit_raw_case(self.payload, self.spec)
        self.assertTrue(raw["ready"])
        self.assertIsNotNone(snapshot)
        self.assertTrue(
            raw["checks"][
                "stored_Yee_dual_volumes_match_grid_edges_and_placement"
            ]
        )
        self.assertTrue(raw["checks"]["Au_Q_is_exactly_zero_outside_binary_mask"])
        self.assertTrue(self._audit_material()["artifact_ready"])

    def test_tampered_raw_bytes_fail_sha_before_loading(self) -> None:
        self.raw_path.write_bytes(b"tampered")
        raw, snapshot = audit_raw_case(self.payload, self.spec)
        self.assertFalse(raw["ready"])
        self.assertFalse(raw["checks"]["sha256_matches"])
        self.assertIsNone(snapshot)

    def test_reported_Q_disagreement_blocks_raw_audit(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["evaluation"]["Q"]["late"]["total_W"] *= 2.0
        raw, _ = audit_raw_case(payload, self.spec)
        self.assertFalse(raw["ready"])
        self.assertFalse(raw["checks"]["raw_Q_integrals_match_report"])

    def test_wrong_reference_or_polarization_blocks_case_identity(self) -> None:
        self._mutate_report(
            lambda payload: payload.update(
                {"reference": "full_design_window", "polarization": "Eb"}
            )
        )
        audit = self._audit_material()
        self.assertFalse(audit["artifact_ready"])
        self.assertFalse(audit["checks"]["case_labels_exact"])

    def test_jointly_changed_case_contract_in_report_is_rejected(self) -> None:
        def mutate(payload: dict) -> None:
            payload["numerical_case_contract"]["time_spec"]["total_periods"] = 32

        self._mutate_report(mutate)
        audit = self._audit_material()
        self.assertFalse(audit["artifact_ready"])
        self.assertFalse(audit["checks"]["numerical_case_contract_exact"])

    def test_material_case_audit_accepts_exact_nondefault_courant_spec(self) -> None:
        spec = FreshCaseSpec(
            mesh=self.spec.mesh,
            time=TimeSpec(total_periods=24, window_periods=4, courant_factor=0.375),
        )
        contract_path = self.root / "contract_c0p375.json"
        numerical_case = case_contract(spec)
        contract_path.write_text(json.dumps(numerical_case), encoding="utf-8")
        contract_sha = sha256(contract_path)
        payload = json.loads(self.report_path.read_text(encoding="utf-8"))
        payload["numerical_case_contract"] = numerical_case
        payload["numerical_case_file_audit"].update(
            path=str(contract_path),
            expected_sha256=contract_sha,
            actual_sha256=contract_sha,
            case_contract_sha256=numerical_case["case_contract_sha256"],
        )
        payload["mesh"] = numerical_case["resolved_mesh"]
        payload["pml_face_parameters"] = numerical_case["resolved_pml_face_parameters"]
        payload["time_contract"].update(
            courant_factor=0.375, time_step_s=0.75e-18, time_steps_total=3200
        )
        self.report_path.write_text(json.dumps(payload), encoding="utf-8")
        _, audit, _ = _material_case_audit(
            self.report_path, self.root, 24, "Ea", spec, contract_path,
            contract_sha, self.source_pair_path, self.source_pair_sha,
        )
        self.assertTrue(audit["checks"]["time_request_and_courant_exact"])
        self.assertTrue(audit["artifact_ready"])

    def test_missing_pilot_gate_is_rejected(self) -> None:
        def mutate(payload: dict) -> None:
            payload["evaluation"]["gates"].pop("Q_closed_td_closure")

        self._mutate_report(mutate)
        audit = self._audit_material()
        self.assertFalse(audit["artifact_ready"])
        self.assertFalse(audit["checks"]["status_and_evaluation_consistent"])


class TimeSettlingComparisonTest(unittest.TestCase):
    @staticmethod
    def _inputs() -> tuple[dict, dict, dict]:
        snapshots: dict[int, dict[str, dict]] = {}
        payloads: dict[int, dict[str, dict]] = {}
        source_pairs: dict[int, dict] = {}
        for period in (16, 24, 32):
            snapshots[period] = {}
            payloads[period] = {}
            source_pairs[period] = {
                "comparison": {
                    "unscaled_incident_power_W": {"Ea": 2.0, "Eb": 2.0},
                    "mean_unscaled_incident_power_W": 2.0,
                }
            }
            for polarization in ("Ea", "Eb"):
                volume = {
                    "au": np.ones((3, 2, 2, 1)),
                    "tairte4": np.ones((3, 2, 2, 1)),
                }
                q = {
                    "au": np.ones((3, 2, 2, 1)),
                    "tairte4": np.ones((3, 2, 2, 1)) * 2.0,
                }
                power = {
                    "by_material": {
                        material: component_power(q[material], volume[material])
                        for material in ("au", "tairte4")
                    }
                }
                power["total_W"] = sum(
                    item["total_W"] for item in power["by_material"].values()
                )
                snapshots[period][polarization] = {
                    "probe": np.ones((3, 2, 2, 1), dtype=np.complex64),
                    "probe_weights": np.ones((3, 2, 2, 1)),
                    "fields_late": {
                        "au": np.ones((3, 2, 2, 1), dtype=np.complex64),
                        "tairte4": np.ones((3, 2, 2, 1), dtype=np.complex64),
                    },
                    "q_late": q,
                    "volumes": volume,
                    "power_late": power,
                }
                payloads[period][polarization] = {
                    "evaluation": {
                        "flux": {
                            "Q_vs_closed_phasor_symmetric_relative": 1.0e-3,
                            "Q_vs_closed_td_symmetric_relative": 1.0e-3,
                        },
                        "field_stationarity": {
                            "maximum_complex_E_NRMSE": 1.0e-3
                        },
                    }
                }
        return snapshots, payloads, source_pairs

    def test_identical_same_grid_pair_passes_all_optical_gates(self) -> None:
        snapshots, payloads, source_pairs = self._inputs()
        result = compare_time_pair(16, 24, snapshots, payloads, source_pairs)
        self.assertTrue(result["pass"])
        self.assertTrue(all(result["checks"].values()))

    def test_fixed_probe_change_above_limit_blocks_pair(self) -> None:
        snapshots, payloads, source_pairs = self._inputs()
        snapshots[24]["Eb"]["probe"] *= 1.10
        result = compare_time_pair(16, 24, snapshots, payloads, source_pairs)
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["complex_E_fixed_probe_NRMSE"])

    def test_fixed_probe_uses_float32_physical_edges_and_Yee_areas(self) -> None:
        micrometre = 1.0e-6
        edges = (
            np.arange(-5, 6, dtype=np.float32) * micrometre,
            np.arange(-5, 6, dtype=np.float32) * micrometre,
            np.asarray([0.25, 0.30], dtype=np.float32) * micrometre,
        )
        target = np.ones((3, 10, 10, 1), dtype=np.complex64)
        probe, weights, audit = fixed_probe_and_weights(
            target,
            edges,
            {"target_field": [[0, 10], [0, 10], [0, 1]]},
        )
        self.assertEqual(probe.shape, (3, 8, 8, 1))
        self.assertEqual(weights.shape, probe.shape)
        self.assertTrue(np.all(weights > 0.0))
        self.assertEqual(audit["global_cell_bounds"][:2], [[1, 9], [1, 9]])


class TimeSettlingSelectionTest(unittest.TestCase):
    @staticmethod
    def _valid_inputs() -> tuple[dict, dict, dict]:
        ready = {
            16: {"Ea": False, "Eb": False},
            24: {"Ea": True, "Eb": True},
            32: {"Ea": True, "Eb": True},
        }
        failed = {
            16: {
                "Ea": ["complex_field_stationarity"],
                "Eb": [
                    "complex_field_stationarity",
                    "Q_previous_late_spatial_change",
                ],
            },
            24: {"Ea": [], "Eb": []},
            32: {"Ea": [], "Eb": []},
        }
        pairs = {(16, 24): True, (24, 32): True}
        return ready, failed, pairs

    def test_rejected_16_period_coarse_is_allowed_and_24_is_selected(self) -> None:
        result = settling_selection_gates(*self._valid_inputs())
        self.assertTrue(all(result.values()))

    def test_24_or_32_internal_failure_blocks_selection(self) -> None:
        ready, failed, pairs = self._valid_inputs()
        ready[24]["Ea"] = False
        self.assertFalse(all(settling_selection_gates(ready, failed, pairs).values()))
        ready, failed, pairs = self._valid_inputs()
        ready[32]["Eb"] = False
        self.assertFalse(all(settling_selection_gates(ready, failed, pairs).values()))

    def test_cross_pair_failure_blocks_selection(self) -> None:
        ready, failed, pairs = self._valid_inputs()
        pairs[(24, 32)] = False
        result = settling_selection_gates(ready, failed, pairs)
        self.assertFalse(result["24_to_32_cross_comparison_passes"])


if __name__ == "__main__":
    unittest.main()
