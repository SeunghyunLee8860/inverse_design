"""Fail-closed certificate for the four exact-binary FDTDX anchor controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
    STATUS_READY as CASE_STATUS,
    validate_source_pair,
)


CERTIFICATE_NAME = "FDTDX_FRESH_EXACT_BINARY_MATRIX.json"
MATRIX_STATUS = "VALIDATED_FDTDX_FRESH_EXACT_BINARY_CONTROL_MATRIX"
BLOCKED_STATUS = "BLOCKED_FDTDX_FRESH_EXACT_BINARY_CONTROL_MATRIX"
EXPECTED_SCOPE = (
    "one fixed exact-binary optical material pilot; "
    "no thermal/electrical/adjoint/optimizer"
)
REFERENCES = ("empty", "full_design_window")
POLARIZATIONS = ("Ea", "Eb")
RAW_POWER_RTOL = 5.0e-13
RAW_POWER_ATOL_W = 1.0e-30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.uint8)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"pilot report must contain one JSON object: {resolved}")
    return payload


def _file_audit(path_value: str, expected_sha256: str) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    absolute = path.is_absolute()
    resolved = path.resolve()
    exists = resolved.is_file()
    actual = sha256(resolved) if exists else None
    return {
        "path": str(resolved),
        "path_is_absolute": absolute,
        "exists": exists,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "sha256_matches": exists and actual == expected_sha256,
    }


def _all_true(values: dict[str, Any]) -> bool:
    return bool(values) and all(value is True for value in values.values())


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


def _power_matches(recorded: dict[str, Any], computed: dict[str, Any]) -> bool:
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


def _raw_case_audit(
    payload: dict[str, Any],
    reference: str,
) -> dict[str, Any]:
    raw = _file_audit(payload["raw"]["path"], payload["raw"]["sha256"])
    result: dict[str, Any] = {
        **raw,
        "declared_arrays": payload["raw"]["arrays"],
        "checks": {},
        "derived": {},
    }
    if not raw["exists"] or not raw["sha256_matches"]:
        result["checks"] = {
            "path_is_absolute": raw["path_is_absolute"],
            "file_exists": raw["exists"],
            "sha256_matches": raw["sha256_matches"],
        }
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
    with np.load(raw["path"], allow_pickle=False) as archive:
        files = set(archive.files)
        arrays_declared_exactly = files == set(payload["raw"]["arrays"])
        required_present = required.issubset(files)
        declared_shapes_match = all(
            name in files
            and list(np.asarray(archive[name]).shape) == declared_shape
            for name, declared_shape in payload["raw"]["arrays"].items()
        )
        if not required_present:
            result["checks"] = {
                "path_is_absolute": raw["path_is_absolute"],
                "file_exists": True,
                "sha256_matches": True,
                "arrays_declared_exactly": arrays_declared_exactly,
                "required_arrays_present": False,
                "declared_shapes_match": declared_shapes_match,
            }
            result["ready"] = False
            return result

        design_mask = np.asarray(archive["design_mask"])
        solver_mask = np.asarray(archive["solver_mask"])
        masks_integer = np.issubdtype(
            design_mask.dtype, np.integer
        ) and np.issubdtype(solver_mask.dtype, np.integer)
        masks_binary = set(np.unique(design_mask).tolist()).issubset({0, 1}) and set(
            np.unique(solver_mask).tolist()
        ).issubset({0, 1})
        design_solid = int(np.count_nonzero(design_mask))
        solver_solid = int(np.count_nonzero(solver_mask))
        expected_solid = 0 if reference == "empty" else int(design_mask.size)
        expected_solver_solid = 0 if reference == "empty" else int(solver_mask.size)

        finite_nonnegative_q = True
        finite_positive_volume = True
        computed_power: dict[str, dict[str, Any]] = {
            "previous": {},
            "late": {},
        }
        recorded_power_matches = True
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
                recorded_power_matches = recorded_power_matches and _power_matches(
                    recorded, computed
                )
            raw_total = sum(
                computed_power[window][material]["total_W"]
                for material in ("au", "tairte4")
            )
            recorded_total = float(payload["evaluation"]["Q"][window]["total_W"])
            recorded_power_matches = recorded_power_matches and math.isclose(
                raw_total,
                recorded_total,
                rel_tol=RAW_POWER_RTOL,
                abs_tol=RAW_POWER_ATOL_W,
            )

    exact_binary = payload["material"]["exact_binary_au"]
    late_au = computed_power["late"]["au"]["total_W"]
    late_total = sum(
        computed_power["late"][material]["total_W"]
        for material in ("au", "tairte4")
    )
    design_mask_sha256 = _array_sha256(design_mask)
    solver_mask_sha256 = _array_sha256(solver_mask)
    checks = {
        "path_is_absolute": raw["path_is_absolute"],
        "file_exists": True,
        "sha256_matches": True,
        "arrays_declared_exactly": arrays_declared_exactly,
        "required_arrays_present": required_present,
        "declared_shapes_match": declared_shapes_match,
        "masks_have_integer_dtype": masks_integer,
        "masks_are_binary": masks_binary,
        "mask_occupancy_matches_reference": design_solid == expected_solid
        and solver_solid == expected_solver_solid,
        "mask_occupancy_matches_material_report": design_solid
        == int(exact_binary["design_solid_cells"])
        and solver_solid == int(exact_binary["solver_solid_cells"]),
        "mask_sha256_matches_material_report": design_mask_sha256
        == exact_binary["design_mask_sha256"]
        and solver_mask_sha256 == exact_binary["solver_mask_sha256"],
        "Q_is_finite_nonnegative": finite_nonnegative_q,
        "electric_dual_volumes_are_finite_positive": finite_positive_volume,
        "raw_Q_integrals_match_report": recorded_power_matches,
        "empty_Au_Q_is_zero_or_full_Au_Q_is_positive": (
            late_au == 0.0 if reference == "empty" else late_au > 0.0
        ),
        "total_Q_is_finite_positive": math.isfinite(late_total)
        and late_total > 0.0,
    }
    result["checks"] = checks
    result["derived"] = {
        "design_mask_sha256": design_mask_sha256,
        "solver_mask_sha256": solver_mask_sha256,
        "design_solid_cells": design_solid,
        "solver_solid_cells": solver_solid,
        "power_W": computed_power,
    }
    result["ready"] = _all_true(checks)
    return result


def _case_audit(
    report_path: Path,
    expected_reference: str,
    expected_polarization: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = report_path.expanduser().resolve()
    payload = _load_json(resolved)
    evaluation = payload["evaluation"]
    material = payload["material"]
    exact_binary = material["exact_binary_au"]
    source_pair = payload["source_pair"]
    runner = _file_audit(
        payload["provenance"]["runner_path"],
        payload["provenance"]["runner_sha256"],
    )
    material_contract = _file_audit(
        payload["provenance"]["material_contract_path"],
        payload["provenance"]["material_contract_sha256"],
    )
    raw = _raw_case_audit(payload, expected_reference)
    audit = {
        "report_path": str(resolved),
        "report_path_is_absolute": report_path.expanduser().is_absolute(),
        "report_sha256": sha256(resolved),
        "expected_reference": expected_reference,
        "recorded_reference": payload.get("reference"),
        "expected_polarization": expected_polarization,
        "recorded_polarization": payload.get("polarization"),
        "status": payload.get("status"),
        "ready": payload.get("ready"),
        "scope": payload.get("scope"),
        "evaluation_ready": evaluation.get("ready"),
        "evaluation_gates_all_true": _all_true(evaluation.get("gates", {})),
        "evaluation_failed_gates": evaluation.get("failed_gates"),
        "material_ready": material.get("ready"),
        "material_checks_all_true": _all_true(material.get("checks", {})),
        "material_failed_checks": material.get("failed_checks"),
        "exact_binary_ready": exact_binary.get("ready"),
        "exact_binary_checks_all_true": _all_true(exact_binary.get("checks", {})),
        "gray_density_allowed": exact_binary.get("gray_density_allowed"),
        "rho_power": exact_binary.get("rho_power"),
        "source_pair_ready": source_pair.get("ready"),
        "source_pair_failed_checks": source_pair.get("failed_checks"),
        "source_pair_checks_all_true": _all_true(source_pair.get("checks", {})),
        "source_pair_path": source_pair.get("path"),
        "source_pair_expected_sha256": source_pair.get("expected_sha256"),
        "source_pair_contract_checks_all_true": _all_true(
            payload.get("source_pair_contract_checks", {})
        ),
        "normalization_policy": payload.get("normalization_policy"),
        "optimizer_start_allowed": payload.get("optimizer_start_allowed"),
        "repository_commit": payload["provenance"]["repository_commit"],
        "repository_dirty_porcelain": payload["provenance"][
            "repository_dirty_porcelain"
        ],
        "runner": runner,
        "material_contract": material_contract,
        "fdtdx_source": payload["provenance"]["fdtdx_source"],
        "runtime_lock": payload["provenance"]["runtime_lock"],
        "raw": raw,
    }
    return payload, audit


def _common_source_contract(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload["source_contract"])
    result.pop("polarization", None)
    result.pop("fixed_E_polarization_vector", None)
    return result


def _all_equal(values: list[Any]) -> bool:
    return bool(values) and all(value == values[0] for value in values[1:])


def build_matrix_certificate(
    empty_ea: Path,
    empty_eb: Path,
    full_ea: Path,
    full_eb: Path,
) -> dict[str, Any]:
    paths = {
        "empty": {"Ea": empty_ea, "Eb": empty_eb},
        "full_design_window": {"Ea": full_ea, "Eb": full_eb},
    }
    payloads: dict[str, dict[str, dict[str, Any]]] = {}
    cases: dict[str, dict[str, dict[str, Any]]] = {}
    flat_payloads: list[dict[str, Any]] = []
    flat_cases: list[dict[str, Any]] = []
    for reference in REFERENCES:
        payloads[reference] = {}
        cases[reference] = {}
        for polarization in POLARIZATIONS:
            payload, audit = _case_audit(
                paths[reference][polarization], reference, polarization
            )
            payloads[reference][polarization] = payload
            cases[reference][polarization] = audit
            flat_payloads.append(payload)
            flat_cases.append(audit)

    source_pair_bindings = {
        (case["source_pair_path"], case["source_pair_expected_sha256"])
        for case in flat_cases
    }
    pair_revalidation: dict[str, Any]
    if len(source_pair_bindings) == 1:
        pair_path, pair_sha = next(iter(source_pair_bindings))
        try:
            _, pair_revalidation = validate_source_pair(Path(pair_path), pair_sha)
        except Exception as error:  # fail closed but still write an audit record
            pair_revalidation = {
                "ready": False,
                "error": repr(error),
                "checks": {},
                "failed_checks": ["source_pair_revalidation_exception"],
            }
    else:
        pair_revalidation = {
            "ready": False,
            "checks": {},
            "failed_checks": ["source_pair_binding_not_identical"],
        }

    empty_masks = [
        cases["empty"][polarization]["raw"].get("derived", {}).get(
            "design_mask_sha256"
        )
        for polarization in POLARIZATIONS
    ]
    full_masks = [
        cases["full_design_window"][polarization]["raw"]
        .get("derived", {})
        .get("design_mask_sha256")
        for polarization in POLARIZATIONS
    ]
    report_paths = [case["report_path"] for case in flat_cases]
    raw_paths = [case["raw"]["path"] for case in flat_cases]
    source_contracts = [payload["source_contract"] for payload in flat_payloads]
    material_responses = [
        payload["material"]["realized_material_response"]
        for payload in flat_payloads
    ]
    gates = {
        "four_distinct_absolute_reports": len(set(report_paths)) == 4
        and all(case["report_path_is_absolute"] for case in flat_cases),
        "four_distinct_absolute_raw_files": len(set(raw_paths)) == 4
        and all(case["raw"]["path_is_absolute"] for case in flat_cases),
        "expected_case_labels": all(
            case["recorded_reference"] == case["expected_reference"]
            and case["recorded_polarization"] == case["expected_polarization"]
            for case in flat_cases
        ),
        "case_status_scope_and_ready": all(
            case["status"] == CASE_STATUS
            and case["ready"] is True
            and case["scope"] == EXPECTED_SCOPE
            for case in flat_cases
        ),
        "case_evaluation_gates": all(
            case["evaluation_ready"] is True
            and case["evaluation_gates_all_true"]
            and case["evaluation_failed_gates"] == []
            for case in flat_cases
        ),
        "case_material_gates": all(
            case["material_ready"] is True
            and case["material_checks_all_true"]
            and case["material_failed_checks"] == []
            for case in flat_cases
        ),
        "case_exact_binary_gates": all(
            case["exact_binary_ready"] is True
            and case["exact_binary_checks_all_true"]
            and case["gray_density_allowed"] is False
            and case["rho_power"] is None
            for case in flat_cases
        ),
        "case_source_pair_gates": all(
            case["source_pair_ready"] is True
            and case["source_pair_checks_all_true"]
            and case["source_pair_failed_checks"] == []
            and case["source_pair_contract_checks_all_true"]
            for case in flat_cases
        ),
        "source_pair_binding_identical": len(source_pair_bindings) == 1,
        "source_pair_revalidation_ready": pair_revalidation.get("ready") is True,
        "raw_files_and_recomputed_physics_ready": all(
            case["raw"].get("ready") is True for case in flat_cases
        ),
        "raw_array_schema_identical": _all_equal(
            [case["raw"]["declared_arrays"] for case in flat_cases]
        ),
        "empty_mask_polarization_invariant": _all_equal(empty_masks)
        and empty_masks[0] is not None,
        "full_mask_polarization_invariant": _all_equal(full_masks)
        and full_masks[0] is not None,
        "empty_and_full_masks_are_distinct": empty_masks[0] is not None
        and full_masks[0] is not None
        and empty_masks[0] != full_masks[0],
        "mesh_identical": _all_equal([payload["mesh"] for payload in flat_payloads]),
        "time_contract_identical": _all_equal(
            [payload["time_contract"] for payload in flat_payloads]
        ),
        "pml_face_parameters_identical": _all_equal(
            [payload["pml_face_parameters"] for payload in flat_payloads]
        ),
        "placement_identical": _all_equal(
            [payload["placement"] for payload in flat_payloads]
        ),
        "common_source_contract_identical": _all_equal(
            [_common_source_contract(payload) for payload in flat_payloads]
        ),
        "same_polarization_source_contract_identical": all(
            payloads["empty"][polarization]["source_contract"]
            == payloads["full_design_window"][polarization]["source_contract"]
            for polarization in POLARIZATIONS
        ),
        "source_polarization_vectors_exact": source_contracts[0][
            "fixed_E_polarization_vector"
        ]
        == [0.0, 1.0, 0.0]
        and source_contracts[1]["fixed_E_polarization_vector"]
        == [1.0, 0.0, 0.0]
        and source_contracts[2]["fixed_E_polarization_vector"]
        == [0.0, 1.0, 0.0]
        and source_contracts[3]["fixed_E_polarization_vector"]
        == [1.0, 0.0, 0.0],
        "normalization_policy_identical": _all_equal(
            [case["normalization_policy"] for case in flat_cases]
        ),
        "raw_unscaled_and_per_polarization_matching_forbidden": all(
            case["normalization_policy"].get("raw_fields_and_Q_are_unscaled")
            is True
            and case["normalization_policy"].get(
                "per_polarization_matching_forbidden"
            )
            is True
            for case in flat_cases
        ),
        "realized_material_response_identical": _all_equal(material_responses),
        "fdtdx_source_provenance_identical": _all_equal(
            [case["fdtdx_source"] for case in flat_cases]
        ),
        "runtime_lock_identical": _all_equal(
            [case["runtime_lock"] for case in flat_cases]
        ),
        "runner_sha256_identical_and_current": _all_equal(
            [case["runner"]["expected_sha256"] for case in flat_cases]
        )
        and all(case["runner"]["sha256_matches"] for case in flat_cases),
        "material_contract_identical_and_current": _all_equal(
            [case["material_contract"]["expected_sha256"] for case in flat_cases]
        )
        and all(
            case["material_contract"]["sha256_matches"] for case in flat_cases
        ),
        "case_repositories_and_fdtdx_clean": all(
            case["repository_dirty_porcelain"] == ""
            and case["fdtdx_source"].get("dirty_porcelain") == ""
            for case in flat_cases
        ),
        "optimizer_remains_forbidden": all(
            case["optimizer_start_allowed"] is False for case in flat_cases
        ),
    }
    ready = all(gates.values())

    comparison: dict[str, dict[str, Any]] = {}
    for reference in REFERENCES:
        comparison[reference] = {}
        for polarization in POLARIZATIONS:
            evaluation = payloads[reference][polarization]["evaluation"]
            comparison[reference][polarization] = {
                "unscaled_Q_W": evaluation["Q"]["late"]["total_W"],
                "unscaled_Au_Q_W": evaluation["Q"]["late"]["by_material"][
                    "au"
                ]["total_W"],
                "unscaled_TaIrTe4_Q_W": evaluation["Q"]["late"]["by_material"][
                    "tairte4"
                ]["total_W"],
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

    return {
        "status": MATRIX_STATUS if ready else BLOCKED_STATUS,
        "ready": ready,
        "scope": "four exact-binary optical anchor controls; no convergence or optimizer",
        "cases": cases,
        "source_pair_revalidation": pair_revalidation,
        "shared_contract": {
            "mesh": flat_payloads[0]["mesh"],
            "time_contract": flat_payloads[0]["time_contract"],
            "pml_face_parameters": flat_payloads[0]["pml_face_parameters"],
            "placement": flat_payloads[0]["placement"],
            "common_source_contract": _common_source_contract(flat_payloads[0]),
            "normalization_policy": flat_payloads[0]["normalization_policy"],
            "fdtdx_source": flat_cases[0]["fdtdx_source"],
            "runtime_lock": flat_cases[0]["runtime_lock"],
            "runner_sha256": flat_cases[0]["runner"]["expected_sha256"],
            "material_contract_sha256": flat_cases[0]["material_contract"][
                "expected_sha256"
            ],
        },
        "comparison": comparison,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "optimizer_start_allowed": False,
        "next_allowed_step": (
            "choose and hash a nontrivial exact-binary reference mask, then build "
            "time/z/x-y/PML/domain convergence controls; do not optimize"
        ),
    }


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_matrix_certificate(
    empty_ea: Path,
    empty_eb: Path,
    full_ea: Path,
    full_eb: Path,
    output_directory: Path,
) -> dict[str, Any]:
    output = output_directory.expanduser().resolve()
    if not output.is_absolute() or not output.is_dir():
        raise RuntimeError("output directory must be an existing absolute directory")
    if any(output.iterdir()):
        raise RuntimeError("output directory must be empty before matrix certification")

    result = build_matrix_certificate(empty_ea, empty_eb, full_ea, full_eb)
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
    result["status"] = MATRIX_STATUS if result["ready"] else BLOCKED_STATUS
    result["failed_gates"] = [
        name for name, passed in result["gates"].items() if not passed
    ]
    _atomic_json(output / CERTIFICATE_NAME, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--empty-ea", type=Path, required=True)
    parser.add_argument("--empty-eb", type=Path, required=True)
    parser.add_argument("--full-ea", type=Path, required=True)
    parser.add_argument("--full-eb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = write_matrix_certificate(
            args.empty_ea,
            args.empty_eb,
            args.full_ea,
            args.full_eb,
            args.output_dir,
        )
    except Exception as error:
        failure = {
            "status": "BLOCKED_FDTDX_FRESH_EXACT_BINARY_CONTROL_MATRIX_EXCEPTION",
            "ready": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        output = args.output_dir.expanduser().resolve()
        if output.is_dir() and not any(output.iterdir()):
            _atomic_json(output / CERTIFICATE_NAME, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    summary = {
        "status": result["status"],
        "ready": result["ready"],
        "failed_gates": result["failed_gates"],
        "comparison": result["comparison"],
        "report": str(
            args.output_dir.expanduser().resolve() / CERTIFICATE_NAME
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
