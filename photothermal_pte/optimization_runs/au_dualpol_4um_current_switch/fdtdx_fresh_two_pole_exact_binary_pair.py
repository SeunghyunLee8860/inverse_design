#!/usr/bin/env python3
"""Fail-closed Ea/Eb pair certificate for one two-pole exact-binary case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any, Mapping

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    REFERENCE_NAMES,
    reference_mask,
    upsample_mask,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    FreshCaseSpec,
    case_contract,
    case_from_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_pair import (
    _atomic_json,
    sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_exact_binary import (
    STATUS_READY as CASE_STATUS,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_material_contract import (
    material_law_from_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_source_pair import (
    validate_candidate_source_pair,
)


PAIR_STATUS = "VALIDATED_FDTDX_FRESH_TWO_POLE_EXACT_BINARY_PAIR"
BLOCKED_STATUS = "BLOCKED_FDTDX_FRESH_TWO_POLE_EXACT_BINARY_PAIR"
EXCEPTION_STATUS = "BLOCKED_FDTDX_FRESH_TWO_POLE_EXACT_BINARY_PAIR_EXCEPTION"
CERTIFICATE_NAME = "FDTDX_FRESH_TWO_POLE_EXACT_BINARY_PAIR.json"
EXPECTED_SCOPE = (
    "one forward-only exact-binary optical material case under one "
    "candidate two-pole law; no thermal/electrical/adjoint/optimizer"
)
HERE = Path(__file__).resolve().parent
RAW_POWER_RTOL = 5.0e-13
RAW_POWER_ATOL_W = 1.0e-30


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"material report must contain one JSON object: {resolved}")
    return payload


def _all_true(values: Mapping[str, Any]) -> bool:
    return bool(values) and all(value is True for value in values.values())


def _file_audit(path_value: str, expected_sha256: str) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    absolute = path.is_absolute()
    resolved = path.resolve()
    exists = resolved.is_file()
    actual = sha256(resolved) if exists else None
    checks = {
        "path_is_absolute": absolute,
        "file_exists": exists,
        "sha256_matches": exists and actual == expected_sha256,
    }
    return {
        "path": str(resolved),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "checks": checks,
        "ready": all(checks.values()),
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.uint8)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _power_from_raw(q: np.ndarray, volume: np.ndarray) -> dict[str, Any]:
    q_array = np.asarray(q, dtype=np.float64)
    volume_array = np.asarray(volume, dtype=np.float64)
    if q_array.shape != volume_array.shape or q_array.ndim != 4:
        raise ValueError(
            f"Q/volume shape mismatch: {q_array.shape} versus {volume_array.shape}"
        )
    component = np.sum(q_array * volume_array, axis=(1, 2, 3), dtype=np.float64)
    return {
        "component_W": {
            axis: float(component[index])
            for index, axis in enumerate(("x", "y", "z"))
        },
        "total_W": float(np.sum(component, dtype=np.float64)),
    }


def _power_matches(recorded: Mapping[str, Any], computed: Mapping[str, Any]) -> bool:
    recorded_values = [
        float(recorded["component_W"][axis]) for axis in ("x", "y", "z")
    ] + [float(recorded["total_W"])]
    computed_values = [
        float(computed["component_W"][axis]) for axis in ("x", "y", "z")
    ] + [float(computed["total_W"])]
    return bool(
        np.allclose(
            recorded_values,
            computed_values,
            rtol=RAW_POWER_RTOL,
            atol=RAW_POWER_ATOL_W,
        )
    )


def _raw_case_audit(payload: dict[str, Any], reference: str) -> dict[str, Any]:
    raw = _file_audit(payload["raw"]["path"], payload["raw"]["sha256"])
    result: dict[str, Any] = {
        **raw,
        "declared_arrays": payload["raw"]["arrays"],
        "checks": {},
        "derived": {},
    }
    if not raw["ready"]:
        result["checks"] = raw["checks"]
        result["ready"] = False
        return result

    required = {
        "design_mask",
        "solver_mask",
        "q_au_previous_W_m3",
        "q_au_late_W_m3",
        "q_tairte4_previous_W_m3",
        "q_tairte4_late_W_m3",
        "electric_dual_volume_au_m3",
        "electric_dual_volume_tairte4_m3",
    }
    load_error = None
    checks: dict[str, bool] = {}
    derived: dict[str, Any] = {}
    try:
        with np.load(raw["path"], allow_pickle=False) as archive:
            files = set(archive.files)
            declared = payload["raw"]["arrays"]
            arrays_declared_exactly = files == set(declared)
            required_present = required.issubset(files)
            declared_shapes_match = arrays_declared_exactly and all(
                list(np.asarray(archive[name]).shape) == shape
                for name, shape in declared.items()
            )
            all_values_finite = arrays_declared_exactly and all(
                bool(np.all(np.isfinite(np.asarray(archive[name]))))
                for name in archive.files
            )
            if not required_present:
                raise ValueError("required exact-binary raw arrays are absent")

            design_mask = np.asarray(archive["design_mask"])
            solver_mask = np.asarray(archive["solver_mask"])
            factor = int(payload["mesh"]["spec"]["design_xy_factor"])
            expected_design = np.asarray(reference_mask(reference), dtype=np.uint8)
            expected_solver = np.asarray(
                upsample_mask(reference_mask(reference), factor), dtype=np.uint8
            )
            exact_binary = payload["material"]["exact_binary_au"]
            design_hash = _array_sha256(design_mask)
            solver_hash = _array_sha256(solver_mask)
            masks_integer = np.issubdtype(
                design_mask.dtype, np.integer
            ) and np.issubdtype(solver_mask.dtype, np.integer)
            masks_binary = set(np.unique(design_mask).tolist()).issubset(
                {0, 1}
            ) and set(np.unique(solver_mask).tolist()).issubset({0, 1})

            finite_nonnegative_q = True
            finite_positive_volume = True
            computed_power: dict[str, dict[str, Any]] = {
                "previous": {},
                "late": {},
            }
            reported_power_matches = True
            for window in ("previous", "late"):
                for material in ("au", "tairte4"):
                    q = np.asarray(archive[f"q_{material}_{window}_W_m3"])
                    volume = np.asarray(
                        archive[f"electric_dual_volume_{material}_m3"]
                    )
                    finite_nonnegative_q = finite_nonnegative_q and bool(
                        np.all(np.isfinite(q)) and np.all(q >= 0.0)
                    )
                    finite_positive_volume = finite_positive_volume and bool(
                        np.all(np.isfinite(volume)) and np.all(volume > 0.0)
                    )
                    computed = _power_from_raw(q, volume)
                    computed_power[window][material] = computed
                    recorded = payload["evaluation"]["Q"][window]["by_material"][
                        material
                    ]
                    reported_power_matches = (
                        reported_power_matches and _power_matches(recorded, computed)
                    )
                raw_total = sum(
                    computed_power[window][material]["total_W"]
                    for material in ("au", "tairte4")
                )
                recorded_total = float(
                    payload["evaluation"]["Q"][window]["total_W"]
                )
                reported_power_matches = reported_power_matches and math.isclose(
                    raw_total,
                    recorded_total,
                    rel_tol=RAW_POWER_RTOL,
                    abs_tol=RAW_POWER_ATOL_W,
                )

            scale = float(payload["normalization_policy"]["common_power_scale"])
            reporting = payload["evaluation"]["common_285uW_reporting"]
            normalized_matches = all(
                math.isclose(
                    computed_power["late"][material]["total_W"] * scale,
                    float(reporting[report_name]),
                    rel_tol=RAW_POWER_RTOL,
                    abs_tol=RAW_POWER_ATOL_W,
                )
                for material, report_name in (
                    ("au", "late_Au_Q_W"),
                    ("tairte4", "late_TaIrTe4_Q_W"),
                )
            )
            normalized_total = sum(
                computed_power["late"][material]["total_W"]
                for material in ("au", "tairte4")
            ) * scale
            normalized_matches = normalized_matches and math.isclose(
                normalized_total,
                float(reporting["late_total_Q_W"]),
                rel_tol=RAW_POWER_RTOL,
                abs_tol=RAW_POWER_ATOL_W,
            )

            checks = {
                **raw["checks"],
                "arrays_declared_exactly": arrays_declared_exactly,
                "required_arrays_present": required_present,
                "declared_shapes_match": declared_shapes_match,
                "all_array_values_finite": all_values_finite,
                "masks_have_integer_dtype": masks_integer,
                "masks_are_binary": masks_binary,
                "design_mask_matches_canonical_reference": np.array_equal(
                    design_mask, expected_design
                ),
                "solver_mask_matches_canonical_upsampling": np.array_equal(
                    solver_mask, expected_solver
                ),
                "mask_occupancy_matches_material_report": int(
                    np.count_nonzero(design_mask)
                )
                == int(exact_binary["design_solid_cells"])
                and int(np.count_nonzero(solver_mask))
                == int(exact_binary["solver_solid_cells"]),
                "mask_sha256_matches_material_report": design_hash
                == exact_binary["design_mask_sha256"]
                and solver_hash == exact_binary["solver_mask_sha256"],
                "Q_is_finite_nonnegative": finite_nonnegative_q,
                "electric_dual_volumes_are_finite_positive": finite_positive_volume,
                "raw_Q_integrals_match_report": reported_power_matches,
                "common_normalization_recomputes_reported_Q": normalized_matches,
            }
            derived = {
                "design_mask_sha256": design_hash,
                "solver_mask_sha256": solver_hash,
                "design_solid_cells": int(np.count_nonzero(design_mask)),
                "solver_solid_cells": int(np.count_nonzero(solver_mask)),
                "power_W": computed_power,
                "common_normalized_late_total_Q_W": normalized_total,
            }
    except Exception as error:
        load_error = repr(error)
    result["checks"] = checks
    result["failed_checks"] = [
        name for name, passed in checks.items() if not passed
    ]
    result["derived"] = derived
    result["load_error"] = load_error
    result["ready"] = load_error is None and bool(checks) and all(checks.values())
    return result


def _implementation_audit(expected: Mapping[str, str]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name, digest in expected.items():
        files[name] = _file_audit(str(HERE / name), digest)
    return {
        "files": files,
        "ready": bool(files) and all(item["ready"] for item in files.values()),
    }


def _case_audit(
    report_path: Path,
    expected_polarization: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = report_path.expanduser()
    resolved = path.resolve()
    payload = _load_json(resolved)
    raw = _raw_case_audit(payload, payload.get("reference"))
    provenance = payload["provenance"]
    audit = {
        "report_path": str(resolved),
        "report_path_is_absolute": path.is_absolute(),
        "report_sha256": sha256(resolved),
        "expected_polarization": expected_polarization,
        "recorded_polarization": payload.get("polarization"),
        "reference": payload.get("reference"),
        "status": payload.get("status"),
        "scope": payload.get("scope"),
        "ready": payload.get("ready"),
        "evaluation_ready": payload["evaluation"].get("ready"),
        "evaluation_gates_all_true": _all_true(
            payload["evaluation"].get("gates", {})
        ),
        "evaluation_failed_gates": payload["evaluation"].get("failed_gates"),
        "material_ready": payload["material"].get("ready"),
        "material_checks_all_true": _all_true(payload["material"].get("checks", {})),
        "material_failed_checks": payload["material"].get("failed_checks"),
        "exact_binary_ready": payload["material"]["exact_binary_au"].get("ready"),
        "exact_binary_checks_all_true": _all_true(
            payload["material"]["exact_binary_au"].get("checks", {})
        ),
        "gray_density_allowed": payload["material"]["exact_binary_au"].get(
            "gray_density_allowed"
        ),
        "rho_power": payload["material"]["exact_binary_au"].get("rho_power"),
        "case_file_audit": payload.get("numerical_case_file_audit"),
        "material_law_file_audit": payload.get(
            "candidate_material_law_file_audit"
        ),
        "source_pair": payload.get("source_pair"),
        "source_pair_contract_checks_all_true": _all_true(
            payload.get("source_pair_contract_checks", {})
        ),
        "normalization_policy": payload.get("normalization_policy"),
        "optimizer_start_allowed": payload.get("optimizer_start_allowed"),
        "repository_commit": provenance["repository_commit"],
        "repository_dirty_porcelain": provenance["repository_dirty_porcelain"],
        "fdtdx_source": provenance["fdtdx_source"],
        "runtime_lock": provenance["runtime_lock"],
        "runner": _file_audit(
            provenance["runner_path"], provenance["runner_sha256"]
        ),
        "material_contract": _file_audit(
            provenance["material_contract_path"],
            provenance["material_contract_sha256"],
        ),
        "implementation": _implementation_audit(
            provenance["implementation_sha256"]
        ),
        "raw": raw,
    }
    return payload, audit


def _common_source_contract(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload["source_contract"])
    result.pop("polarization", None)
    result.pop("fixed_E_polarization_vector", None)
    return result


def _source_pair_binding(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    audit = payload.get("source_pair", {})
    return audit.get("path"), audit.get("expected_sha256")


def _canonical_law_audit(
    ea: dict[str, Any], eb: dict[str, Any]
) -> tuple[FreshCaseSpec | None, dict[str, Any] | None, dict[str, Any]]:
    numerical = ea.get("numerical_case_contract")
    law = ea.get("candidate_material_law_contract")
    identical = numerical == eb.get("numerical_case_contract")
    law_identical = law == eb.get("candidate_material_law_contract")
    error = None
    case_spec = None
    reconstructed = None
    if identical and law_identical and isinstance(numerical, dict) and isinstance(law, dict):
        try:
            case_spec = case_from_contract(numerical)
            case_sha = ea["numerical_case_file_audit"]["actual_sha256"]
            source = Path(ea["provenance"]["fdtdx_source"]["path"])
            reconstructed = material_law_from_contract(
                law, case_spec, numerical, case_sha, source
            )
        except Exception as caught:
            error = repr(caught)
    checks = {
        "numerical_case_contracts_identical": identical,
        "numerical_case_contract_canonical": case_spec is not None,
        "candidate_material_law_contracts_identical": law_identical,
        "candidate_material_law_canonical": reconstructed == law
        and reconstructed is not None,
    }
    return case_spec, reconstructed, {
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
        "reconstruction_error": error,
    }


def build_pair_certificate(ea_report: Path, eb_report: Path) -> dict[str, Any]:
    ea, ea_audit = _case_audit(ea_report, "Ea")
    eb, eb_audit = _case_audit(eb_report, "Eb")
    payloads = (ea, eb)
    audits = (ea_audit, eb_audit)
    cases = {"Ea": ea_audit, "Eb": eb_audit}
    case_spec, reconstructed_law, law_audit = _canonical_law_audit(ea, eb)

    bindings = {_source_pair_binding(payload) for payload in payloads}
    if (
        len(bindings) == 1
        and case_spec is not None
        and reconstructed_law is not None
    ):
        pair_path, pair_sha = next(iter(bindings))
        try:
            _, source_pair_revalidation = validate_candidate_source_pair(
                Path(str(pair_path)),
                str(pair_sha),
                case_spec,
                reconstructed_law,
                ea["candidate_material_law_file_audit"],
            )
        except Exception as error:
            source_pair_revalidation = {
                "ready": False,
                "checks": {},
                "failed_checks": ["source_pair_revalidation_exception"],
                "error": repr(error),
            }
    else:
        source_pair_revalidation = {
            "ready": False,
            "checks": {},
            "failed_checks": ["source_pair_binding_or_contract_invalid"],
        }

    references = [audit["reference"] for audit in audits]
    source_vectors = [payload["source_contract"]["fixed_E_polarization_vector"] for payload in payloads]
    raw_mask_hashes = [
        audit["raw"].get("derived", {}).get("design_mask_sha256")
        for audit in audits
    ]
    gates = {
        "two_distinct_absolute_reports": ea_audit["report_path"]
        != eb_audit["report_path"]
        and all(audit["report_path_is_absolute"] for audit in audits),
        "expected_case_labels": ea_audit["recorded_polarization"] == "Ea"
        and eb_audit["recorded_polarization"] == "Eb"
        and references[0] in REFERENCE_NAMES
        and references[0] == references[1],
        "case_status_scope_and_ready": all(
            audit["status"] == CASE_STATUS
            and audit["scope"] == EXPECTED_SCOPE
            and audit["ready"] is True
            for audit in audits
        ),
        "case_evaluation_gates": all(
            audit["evaluation_ready"] is True
            and audit["evaluation_gates_all_true"]
            and audit["evaluation_failed_gates"] == []
            for audit in audits
        ),
        "case_material_gates": all(
            audit["material_ready"] is True
            and audit["material_checks_all_true"]
            and audit["material_failed_checks"] == []
            for audit in audits
        ),
        "case_exact_binary_gates": all(
            audit["exact_binary_ready"] is True
            and audit["exact_binary_checks_all_true"]
            and audit["gray_density_allowed"] is False
            and audit["rho_power"] is None
            for audit in audits
        ),
        "canonical_case_and_material_law": law_audit["ready"],
        "case_file_audits_identical_and_ready": ea_audit["case_file_audit"]
        == eb_audit["case_file_audit"]
        and ea_audit["case_file_audit"].get("ready") is True,
        "material_law_file_audits_identical_and_ready": ea_audit[
            "material_law_file_audit"
        ]
        == eb_audit["material_law_file_audit"]
        and ea_audit["material_law_file_audit"].get("ready") is True,
        "source_pair_binding_identical": len(bindings) == 1,
        "source_pair_report_audits_identical_and_ready": ea_audit["source_pair"]
        == eb_audit["source_pair"]
        and ea_audit["source_pair"].get("ready") is True
        and ea_audit["source_pair"].get("failed_checks") == [],
        "source_pair_contract_checks_all_true": all(
            audit["source_pair_contract_checks_all_true"] for audit in audits
        ),
        "source_pair_revalidation_ready": source_pair_revalidation.get("ready")
        is True,
        "raw_files_and_recomputed_physics_ready": all(
            audit["raw"].get("ready") is True for audit in audits
        ),
        "raw_paths_distinct": ea_audit["raw"]["path"]
        != eb_audit["raw"]["path"],
        "raw_schema_identical": ea_audit["raw"]["declared_arrays"]
        == eb_audit["raw"]["declared_arrays"],
        "canonical_mask_polarization_invariant": raw_mask_hashes[0] is not None
        and raw_mask_hashes[0] == raw_mask_hashes[1],
        "mesh_identical": ea["mesh"] == eb["mesh"],
        "time_contract_identical": ea["time_contract"] == eb["time_contract"],
        "pml_face_parameters_identical": ea["pml_face_parameters"]
        == eb["pml_face_parameters"],
        "placement_identical": ea["placement"] == eb["placement"],
        "material_readback_identical": ea["material"] == eb["material"],
        "common_source_contract_identical": _common_source_contract(ea)
        == _common_source_contract(eb),
        "source_polarization_vectors_exact": source_vectors[0]
        == [0.0, 1.0, 0.0]
        and source_vectors[1] == [1.0, 0.0, 0.0],
        "normalization_policy_identical": ea_audit["normalization_policy"]
        == eb_audit["normalization_policy"],
        "raw_unscaled_and_per_polarization_matching_forbidden": all(
            audit["normalization_policy"].get("raw_fields_and_Q_are_unscaled")
            is True
            and audit["normalization_policy"].get(
                "per_polarization_matching_forbidden"
            )
            is True
            for audit in audits
        ),
        "runner_identical_and_current": ea_audit["runner"]["expected_sha256"]
        == eb_audit["runner"]["expected_sha256"]
        and all(audit["runner"]["ready"] for audit in audits),
        "implementation_identical_and_current": ea["provenance"][
            "implementation_sha256"
        ]
        == eb["provenance"]["implementation_sha256"]
        and all(audit["implementation"]["ready"] for audit in audits),
        "material_contract_identical_and_current": ea_audit[
            "material_contract"
        ]["expected_sha256"]
        == eb_audit["material_contract"]["expected_sha256"]
        and all(audit["material_contract"]["ready"] for audit in audits),
        "fdtdx_source_and_runtime_identical": ea_audit["fdtdx_source"]
        == eb_audit["fdtdx_source"]
        and ea_audit["runtime_lock"] == eb_audit["runtime_lock"],
        "case_repository_commit_identical": ea_audit["repository_commit"]
        == eb_audit["repository_commit"],
        "case_repositories_and_fdtdx_clean": all(
            audit["repository_dirty_porcelain"] == ""
            and audit["fdtdx_source"].get("dirty_porcelain") == ""
            for audit in audits
        ),
        "optimizer_remains_forbidden": all(
            audit["optimizer_start_allowed"] is False for audit in audits
        ),
    }

    comparison: dict[str, Any] = {}
    for polarization, payload in zip(("Ea", "Eb"), payloads):
        evaluation = payload["evaluation"]
        late = evaluation["Q"]["late"]
        total = float(late["total_W"])
        au = float(late["by_material"]["au"]["total_W"])
        ta = float(late["by_material"]["tairte4"]["total_W"])
        comparison[polarization] = {
            "unscaled_total_Q_W": total,
            "unscaled_Au_Q_W": au,
            "unscaled_TaIrTe4_Q_W": ta,
            "Au_fraction_of_total_Q": au / total,
            "TaIrTe4_fraction_of_total_Q": ta / total,
            "common_285uW_reporting": evaluation["common_285uW_reporting"],
            "absorbed_fraction_of_all_air_source": evaluation["flux"][
                "absorbed_fraction_of_all_air_source"
            ],
            "Q_vs_closed_phasor_symmetric_relative": evaluation["flux"][
                "Q_vs_closed_phasor_symmetric_relative"
            ],
            "Q_vs_closed_td_symmetric_relative": evaluation["flux"][
                "Q_vs_closed_td_symmetric_relative"
            ],
            "maximum_complex_E_NRMSE": evaluation["field_stationarity"][
                "maximum_complex_E_NRMSE"
            ],
            "previous_late_spatial_Q_NRMSE": evaluation["Q"][
                "previous_late_spatial_NRMSE"
            ],
            "previous_late_total_Q_relative_change": evaluation["Q"][
                "previous_late_total_relative_change"
            ],
        }
    comparison["Eb_over_Ea"] = {
        "total_Q_ratio": comparison["Eb"]["unscaled_total_Q_W"]
        / comparison["Ea"]["unscaled_total_Q_W"],
        "Au_Q_ratio": comparison["Eb"]["unscaled_Au_Q_W"]
        / comparison["Ea"]["unscaled_Au_Q_W"],
        "TaIrTe4_Q_ratio": comparison["Eb"]["unscaled_TaIrTe4_Q_W"]
        / comparison["Ea"]["unscaled_TaIrTe4_Q_W"],
    }
    ready = all(gates.values())
    z_factor = int(ea["mesh"]["spec"]["z_factor"])
    next_allowed_step = (
        f"build the next separately hashed mesh level after z{z_factor} with "
        "its own all-air source pair; do not start adjoint, "
        "thermal/electrical, or optimization"
        if ready
        else f"resolve the failed z{z_factor} within-case gates under a "
        "separately hashed numerical case and source pair before any finer "
        "mesh, adjoint, thermal/electrical, or optimization step"
    )
    return {
        "status": PAIR_STATUS if ready else BLOCKED_STATUS,
        "ready": ready,
        "scope": (
            "one dual-polarization exact-binary optical material pair under "
            "one candidate two-pole law; no convergence/current/optimizer claim"
        ),
        "reference": references[0] if references[0] == references[1] else None,
        "cases": cases,
        "canonical_contract_audit": law_audit,
        "source_pair_revalidation": source_pair_revalidation,
        "shared_contract": {
            "numerical_case_contract": ea["numerical_case_contract"],
            "numerical_case_file_audit": ea["numerical_case_file_audit"],
            "candidate_material_law_contract": ea[
                "candidate_material_law_contract"
            ],
            "candidate_material_law_file_audit": ea[
                "candidate_material_law_file_audit"
            ],
            "mesh": ea["mesh"],
            "time_contract": ea["time_contract"],
            "pml_face_parameters": ea["pml_face_parameters"],
            "placement": ea["placement"],
            "material": ea["material"],
            "common_source_contract": _common_source_contract(ea),
            "normalization_policy": ea["normalization_policy"],
            "source_pair_binding": {
                "path": _source_pair_binding(ea)[0],
                "expected_sha256": _source_pair_binding(ea)[1],
            },
            "fdtdx_source": ea["provenance"]["fdtdx_source"],
            "runtime_lock": ea["provenance"]["runtime_lock"],
            "runner_sha256": ea["provenance"]["runner_sha256"],
            "implementation_sha256": ea["provenance"]["implementation_sha256"],
            "material_contract_sha256": ea["provenance"][
                "material_contract_sha256"
            ],
        },
        "comparison": comparison,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "optimizer_start_allowed": False,
        "pte_current_claim_allowed": False,
        "next_allowed_step": next_allowed_step,
    }


def write_pair_certificate(
    ea_report: Path, eb_report: Path, output_directory: Path
) -> dict[str, Any]:
    output = output_directory.expanduser().resolve()
    if not output.is_absolute() or not output.is_dir():
        raise RuntimeError("output directory must be an existing absolute directory")
    if any(output.iterdir()):
        raise RuntimeError("output directory must be empty before pair certification")
    result = build_pair_certificate(ea_report, eb_report)
    repository = Path(__file__).resolve().parents[3]
    dirty = _git(repository, "status", "--porcelain", "--untracked-files=all")
    result["provenance"] = {
        "certificate_repository_commit": _git(repository, "rev-parse", "HEAD"),
        "certificate_repository_dirty_porcelain": dirty,
        "certificate_generator_path": str(Path(__file__).resolve()),
        "certificate_generator_sha256": sha256(Path(__file__).resolve()),
    }
    result["gates"]["certificate_repository_clean"] = dirty == ""
    result["ready"] = all(result["gates"].values())
    result["status"] = PAIR_STATUS if result["ready"] else BLOCKED_STATUS
    result["failed_gates"] = [
        name for name, passed in result["gates"].items() if not passed
    ]
    _atomic_json(output / CERTIFICATE_NAME, result)
    return result


def validate_pair_certificate(
    path: Path,
    expected_sha256: str,
    expected_case: FreshCaseSpec,
    expected_reference: str,
    expected_material_law: dict[str, Any],
    expected_material_law_file_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    exists = resolved.is_file()
    actual = sha256(resolved) if exists else None
    payload = _load_json(resolved) if exists else {}
    cases = payload.get("cases", {})
    artifact_checks: dict[str, bool] = {}
    for polarization in ("Ea", "Eb"):
        case = cases.get(polarization, {})
        report_path = Path(case.get("report_path", "/nonexistent")).expanduser()
        raw_path = Path(case.get("raw", {}).get("path", "/nonexistent")).expanduser()
        artifact_checks[f"{polarization}_report_sha256_matches"] = (
            report_path.is_absolute()
            and report_path.is_file()
            and sha256(report_path.resolve()) == case.get("report_sha256")
        )
        artifact_checks[f"{polarization}_raw_sha256_matches"] = (
            raw_path.is_absolute()
            and raw_path.is_file()
            and sha256(raw_path.resolve())
            == case.get("raw", {}).get("actual_sha256")
        )
    provenance = payload.get("provenance", {})
    generator = Path(
        provenance.get("certificate_generator_path", "/nonexistent")
    ).expanduser()
    checks = {
        "certificate_exists": exists,
        "expected_sha256_is_lowercase_hex": len(expected_sha256) == 64
        and expected_sha256 == expected_sha256.lower()
        and all(character in "0123456789abcdef" for character in expected_sha256),
        "certificate_sha256_matches": actual == expected_sha256,
        "certificate_status_ready": payload.get("status") == PAIR_STATUS
        and payload.get("ready") is True,
        "certificate_gates_all_true": _all_true(payload.get("gates", {}))
        and payload.get("failed_gates") == [],
        "reference_exact": payload.get("reference") == expected_reference,
        "numerical_case_exact": payload.get("shared_contract", {}).get(
            "numerical_case_contract"
        )
        == case_contract(expected_case),
        "candidate_material_law_exact": payload.get("shared_contract", {}).get(
            "candidate_material_law_contract"
        )
        == expected_material_law,
        "candidate_material_law_file_exact": payload.get(
            "shared_contract", {}
        ).get("candidate_material_law_file_audit")
        == expected_material_law_file_audit,
        "source_pair_revalidation_ready": payload.get(
            "source_pair_revalidation", {}
        ).get("ready")
        is True,
        "optimizer_forbidden": payload.get("optimizer_start_allowed") is False,
        "pte_current_claim_forbidden": payload.get("pte_current_claim_allowed")
        is False,
        "certificate_generator_current": generator.is_absolute()
        and generator.is_file()
        and sha256(generator.resolve())
        == provenance.get("certificate_generator_sha256"),
        **artifact_checks,
    }
    return payload, {
        "path": str(resolved),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "ready": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ea", type=Path, required=True)
    parser.add_argument("--eb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = write_pair_certificate(args.ea, args.eb, args.output_dir)
    except Exception as error:
        failure = {
            "status": EXCEPTION_STATUS,
            "ready": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "optimizer_start_allowed": False,
            "pte_current_claim_allowed": False,
        }
        output = args.output_dir.expanduser().resolve()
        if output.is_dir() and not any(output.iterdir()):
            _atomic_json(output / CERTIFICATE_NAME, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "ready": result["ready"],
                "failed_gates": result["failed_gates"],
                "reference": result["reference"],
                "comparison": result["comparison"],
                "report": str(args.output_dir.resolve() / CERTIFICATE_NAME),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
