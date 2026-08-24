#!/usr/bin/env python3
"""Forward-only exact-binary material solve for one candidate two-pole law."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_4um_model as optical_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_dependency import (
    configured_source,
    require_source,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    REFERENCE_NAMES,
    reference_mask,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_material import (
    arrays_for_exact_binary,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    FreshCaseSpec,
    case_contract,
    file_sha256,
    load_case_contract,
    realized_time_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
    _atomic_json,
    _atomic_npz,
    _git,
    _output_directory,
    _power_evaluation,
    material_stack_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_mesh import (
    build_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_material_contract import (
    load_material_law_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_source_pair import (
    validate_candidate_source_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_runtime_preflight import (
    load_runtime_lock,
)


STATUS_READY = "VALIDATED_FDTDX_FRESH_TWO_POLE_EXACT_BINARY_CASE"
STATUS_BLOCKED = "BLOCKED_FDTDX_FRESH_TWO_POLE_EXACT_BINARY_CASE"
STATUS_EXCEPTION = "BLOCKED_FDTDX_FRESH_TWO_POLE_EXACT_BINARY_EXCEPTION"
JSON_NAME = "FDTDX_FRESH_TWO_POLE_EXACT_BINARY.json"
RAW_NAME = "FDTDX_FRESH_TWO_POLE_EXACT_BINARY_FIELDS.npz"
HERE = Path(__file__).resolve().parent
SHARED_SOURCE_IMPLEMENTATION_FILES = {
    "fdtdx_4um_model.py": HERE / "fdtdx_4um_model.py",
    "fdtdx_fresh_candidate_model_material.py": (
        HERE / "fdtdx_fresh_candidate_model_material.py"
    ),
    "fdtdx_fresh_mesh.py": HERE / "fdtdx_fresh_mesh.py",
    "fdtdx_fresh_source_only.py": HERE / "fdtdx_fresh_source_only.py",
    "fdtdx_fresh_two_pole_source_only.py": (
        HERE / "fdtdx_fresh_two_pole_source_only.py"
    ),
}
IMPLEMENTATION_FILES = {
    **SHARED_SOURCE_IMPLEMENTATION_FILES,
    "fdtdx_exact_binary_material.py": HERE / "fdtdx_exact_binary_material.py",
    "fdtdx_fresh_exact_binary_pilot.py": HERE / "fdtdx_fresh_exact_binary_pilot.py",
    "fdtdx_fresh_two_pole_source_pair.py": (
        HERE / "fdtdx_fresh_two_pole_source_pair.py"
    ),
    "fdtdx_fresh_two_pole_exact_binary.py": Path(__file__).resolve(),
}


def _implementation_sha256(files: dict[str, Path]) -> dict[str, str]:
    return {name: file_sha256(path) for name, path in files.items()}


def run(
    output_directory: Path,
    source: Path,
    source_pair_path: Path,
    source_pair_sha256: str,
    reference: str,
    polarization: str,
    case_spec: FreshCaseSpec,
    case_payload: dict[str, Any],
    case_file_audit: dict[str, Any],
    material_law: dict[str, Any],
    material_law_file_audit: dict[str, Any],
) -> dict[str, Any]:
    output = _output_directory(output_directory)
    if case_contract(case_spec) != case_payload:
        raise RuntimeError("case payload is not the exact reconstructed request")
    source_audit = require_source(source)
    source_pair, pair_audit = validate_candidate_source_pair(
        source_pair_path,
        source_pair_sha256,
        case_spec,
        material_law,
        material_law_file_audit,
    )
    pair_checks = pair_audit["checks"]
    pair_checks["fdtdx_source_matches_current"] = (
        source_pair["source_case_contracts"]["fdtdx_source"]
        == source_audit["actual"]
    )
    pair_checks["runtime_lock_matches_current"] = (
        source_pair["source_case_contracts"]["runtime_lock"]
        == load_runtime_lock()
    )
    current_shared_hashes = _implementation_sha256(
        SHARED_SOURCE_IMPLEMENTATION_FILES
    )
    pair_checks["shared_source_implementation_matches_current"] = (
        source_pair["source_case_contracts"].get(
            "candidate_source_implementation_sha256"
        )
        == current_shared_hashes
    )
    pair_audit["failed_checks"] = [
        name for name, passed in pair_checks.items() if not passed
    ]
    pair_audit["ready"] = all(pair_checks.values())
    if not pair_audit["ready"]:
        raise RuntimeError(f"candidate source pair revalidation failed: {pair_audit}")

    repository = Path(__file__).resolve().parents[3]
    repository_dirty = _git(
        repository, "status", "--porcelain", "--untracked-files=all"
    )
    if repository_dirty:
        raise RuntimeError("candidate material solve requires a clean repository")
    model = build_model(
        case_spec.mesh,
        polarization,
        total_periods=case_spec.time.total_periods,
        window_periods=case_spec.time.window_periods,
        courant_factor=case_spec.time.courant_factor,
        alpha_scale=case_spec.pml_alpha_scale,
        target_reflection=case_spec.pml_target_reflection,
        include_adjoint_source=False,
        air_only_source_calibration=False,
        material_law_contract=material_law,
    )
    current_time = realized_time_contract(case_spec, model)
    source_contracts = source_pair["source_case_contracts"]
    contract_checks = {
        "numerical_case_matches_source_pair": case_payload
        == source_contracts["numerical_case_contract"],
        "mesh_matches_source_pair": model["fresh_mesh_audit"]
        == source_contracts["mesh"],
        "time_matches_source_pair": current_time == source_contracts["time_contract"],
        "pml_matches_source_pair": model["pml_face_parameters"]
        == source_contracts["pml_face_parameters"],
        "placement_matches_source_pair": model["placement"]
        == source_contracts["placement"],
        "source_matches_polarization_case": model["source_contract"]
        == source_contracts["source_contracts"][polarization],
        "candidate_material_law_matches_source_pair": material_law
        == source_contracts["candidate_material_law_contract"],
        "candidate_model_mode_exact": (
            model.get("material_law_mode") == "candidate-two-pole-contract"
        ),
        "candidate_model_law_sha256_exact": (
            model.get("material_law_contract_sha256")
            == material_law["material_law_contract_sha256"]
        ),
        "candidate_model_two_poles": model.get("num_dispersive_poles") == 2,
        "adjoint_source_absent": "distributed_adjoint_source" not in model["slices"],
        "optimizer_forbidden_by_case": (
            case_payload["rules"]["optimizer_start_allowed"] is False
        ),
        "optimizer_forbidden_by_law": (
            material_law["promotion"]["optimizer_start_allowed"] is False
        ),
    }
    if not all(contract_checks.values()):
        raise RuntimeError(f"candidate material/source mismatch: {contract_checks}")

    mask = np.asarray(reference_mask(reference), dtype=np.uint8)
    arrays = arrays_for_exact_binary(model, mask, case_spec.mesh)
    material = material_stack_audit(model, arrays, mask, case_spec.mesh)
    if not material["ready"]:
        raise RuntimeError(f"candidate material readback failed: {material}")

    started = time.perf_counter()
    _, fdtd_output = model["fdtdx"].run_fdtd(
        arrays,
        model["placed"],
        model["config"],
        model["key"],
        show_progress=False,
    )
    marker = fdtd_output.detector_states["target_field"]["phasor"]
    model["jax"].block_until_ready(marker)
    solve_runtime = time.perf_counter() - started
    evaluation, raw_arrays = _power_evaluation(
        model, fdtd_output, mask, source_pair, case_spec.mesh
    )
    raw_path = output / RAW_NAME
    _atomic_npz(raw_path, raw_arrays)
    provenance = {
        "repository_commit": _git(repository, "rev-parse", "HEAD"),
        "repository_dirty_porcelain": repository_dirty,
        "fdtdx_source": source_audit["actual"],
        "runtime_lock": load_runtime_lock(),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "implementation_sha256": _implementation_sha256(IMPLEMENTATION_FILES),
        "material_contract_path": str(optical_model.MATERIAL_JSON.resolve()),
        "material_contract_sha256": file_sha256(optical_model.MATERIAL_JSON),
    }
    ready = (
        pair_audit["ready"]
        and all(contract_checks.values())
        and material["ready"]
        and evaluation["ready"]
        and repository_dirty == ""
    )
    payload = {
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "scope": (
            "one forward-only exact-binary optical material case under one "
            "candidate two-pole law; no thermal/electrical/adjoint/optimizer"
        ),
        "reference": reference,
        "polarization": polarization,
        "numerical_case_contract": case_payload,
        "numerical_case_file_audit": case_file_audit,
        "candidate_material_law_contract": material_law,
        "candidate_material_law_file_audit": material_law_file_audit,
        "mesh": model["fresh_mesh_audit"],
        "time_contract": current_time,
        "source_contract": model["source_contract"],
        "pml_face_parameters": model["pml_face_parameters"],
        "placement": model["placement"],
        "source_pair": pair_audit,
        "source_pair_contract_checks": contract_checks,
        "material": material,
        "evaluation": evaluation,
        "solve_runtime_s": solve_runtime,
        "raw": {
            "path": str(raw_path),
            "sha256": file_sha256(raw_path),
            "arrays": {
                name: list(np.asarray(value).shape)
                for name, value in raw_arrays.items()
            },
        },
        "normalization_policy": {
            "raw_fields_and_Q_are_unscaled": True,
            "per_polarization_matching_forbidden": True,
            "common_power_scale": source_pair["common_normalization"][
                "common_power_scale"
            ],
            "common_field_amplitude_scale": source_pair[
                "common_normalization"
            ]["common_field_amplitude_scale"],
        },
        "provenance": provenance,
        "optimizer_start_allowed": False,
    }
    _atomic_json(output / JSON_NAME, payload)
    return payload


def main() -> int:
    configured_output = os.environ.get("FDTDX_FRESH_OUTPUT_DIR", "").strip()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(configured_output) if configured_output else None,
    )
    parser.add_argument("--source", type=Path, default=configured_source())
    parser.add_argument("--source-pair", type=Path, required=True)
    parser.add_argument("--source-pair-sha256", required=True)
    parser.add_argument("--reference", choices=REFERENCE_NAMES, required=True)
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--case-contract", type=Path, required=True)
    parser.add_argument("--case-contract-sha256", required=True)
    parser.add_argument("--material-law", type=Path, required=True)
    parser.add_argument("--material-law-sha256", required=True)
    args = parser.parse_args()
    if args.output_dir is None:
        parser.error("--output-dir or FDTDX_FRESH_OUTPUT_DIR is required")
    try:
        case_spec, case_payload, case_audit = load_case_contract(
            args.case_contract, args.case_contract_sha256
        )
        material_law, material_law_audit = load_material_law_contract(
            args.material_law,
            args.material_law_sha256,
            case_spec,
            case_payload,
            case_audit["actual_sha256"],
            args.source,
        )
        result = run(
            args.output_dir,
            args.source,
            args.source_pair,
            args.source_pair_sha256,
            args.reference,
            args.polarization,
            case_spec,
            case_payload,
            case_audit,
            material_law,
            material_law_audit,
        )
    except Exception as error:
        failure = {
            "status": STATUS_EXCEPTION,
            "ready": False,
            "reference": args.reference,
            "polarization": args.polarization,
            "case_contract_path": str(args.case_contract.expanduser().resolve()),
            "case_contract_expected_sha256": args.case_contract_sha256,
            "material_law_path": str(args.material_law.expanduser().resolve()),
            "material_law_expected_sha256": args.material_law_sha256,
            "source_pair_path": str(args.source_pair.expanduser().resolve()),
            "source_pair_expected_sha256": args.source_pair_sha256,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "optimizer_start_allowed": False,
        }
        output = args.output_dir.expanduser().resolve()
        if output.is_dir() and not any(output.iterdir()):
            _atomic_json(output / JSON_NAME, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    summary = {
        "status": result["status"],
        "ready": result["ready"],
        "reference": result["reference"],
        "polarization": result["polarization"],
        "case_contract_sha256": result["numerical_case_contract"][
            "case_contract_sha256"
        ],
        "material_law_contract_sha256": result[
            "candidate_material_law_contract"
        ]["material_law_contract_sha256"],
        "failed_gates": result["evaluation"]["failed_gates"],
        "maximum_complex_E_NRMSE": result["evaluation"]["field_stationarity"][
            "maximum_complex_E_NRMSE"
        ],
        "unscaled_Q_W": result["evaluation"]["Q"]["late"]["total_W"],
        "Q_vs_closed_phasor_relative": result["evaluation"]["flux"][
            "Q_vs_closed_phasor_symmetric_relative"
        ],
        "Q_vs_closed_td_relative": result["evaluation"]["flux"][
            "Q_vs_closed_td_symmetric_relative"
        ],
        "solve_runtime_s": result["solve_runtime_s"],
        "report": str(args.output_dir.expanduser().resolve() / JSON_NAME),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
