#!/usr/bin/env python3
"""Forward-only all-air source solve bound to one candidate material law."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_dependency import (
    configured_source,
    require_source,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    mesh_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_material import (
    coefficient_endpoint_matrix,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    FreshCaseSpec,
    case_contract,
    file_sha256,
    load_case_contract,
    realized_time_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_mesh import (
    build_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_only import (
    _atomic_json,
    _atomic_npz,
    _git,
    _output_directory,
    all_air_arrays,
    evaluate_output,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_material_contract import (
    load_material_law_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_runtime_preflight import (
    load_runtime_lock,
)


CASE_STATUS = "VALIDATED_FDTDX_FRESH_TWO_POLE_SOURCE_ONLY_CASE"
BLOCKED_STATUS = "BLOCKED_FDTDX_FRESH_TWO_POLE_SOURCE_ONLY_CASE"
EXCEPTION_STATUS = "BLOCKED_FDTDX_FRESH_TWO_POLE_SOURCE_ONLY_EXCEPTION"
SCOPE = (
    "all-air source-only bound to one candidate two-pole material law and "
    "hashed fresh numerical contract"
)
JSON_NAME = "FDTDX_FRESH_TWO_POLE_SOURCE_ONLY.json"
RAW_NAME = "FDTDX_FRESH_TWO_POLE_SOURCE_ONLY_FIELDS.npz"
HERE = Path(__file__).resolve().parent
IMPLEMENTATION_FILES = (
    HERE / "fdtdx_4um_model.py",
    HERE / "fdtdx_fresh_candidate_model_material.py",
    HERE / "fdtdx_fresh_mesh.py",
    HERE / "fdtdx_fresh_source_only.py",
    Path(__file__).resolve(),
)


def candidate_source_model_audit(
    model: dict[str, Any], law: dict[str, Any]
) -> dict[str, Any]:
    expected = {
        name: np.asarray(
            [
                [pole["c1"], pole["c2"], pole["c3"]]
                for pole in law["material_axes"][name]["candidate"]["poles"]
            ],
            dtype=np.float32,
        )
        for name in ("au", "a", "b", "c")
    }
    checks = {
        "candidate_model_mode_exact": (
            model.get("material_law_mode") == "candidate-two-pole-contract"
        ),
        "material_law_contract_sha256_exact": (
            model.get("material_law_contract_sha256")
            == law["material_law_contract_sha256"]
        ),
        "two_poles_declared": model.get("num_dispersive_poles") == 2,
        "all_endpoints_equal_contract_float32": all(
            np.array_equal(coefficient_endpoint_matrix(model, name), value)
            for name, value in expected.items()
        ),
        "all_fixed_candidate_material_arrays_zero_for_air": all(
            float(model["jnp"].max(model["jnp"].abs(model[name]))) == 0.0
            for name in ("fixed_c1", "fixed_c2", "fixed_c3")
        ),
        "lorentz_drude_c4_array_absent": model["base"].dispersive_c4 is None,
        "air_only_source_calibration_exact": (
            model.get("air_only_source_calibration") is True
        ),
        "adjoint_source_absent": "distributed_adjoint_source" not in model["slices"],
        "candidate_only_remains_true": law["promotion"]["candidate_only"] is True,
        "optimizer_remains_forbidden": (
            law["promotion"]["optimizer_start_allowed"] is False
        ),
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "devices": [str(device) for device in model["jax"].devices()],
    }


def run(
    output_directory: Path,
    source: Path,
    polarization: str,
    case_spec: FreshCaseSpec,
    case_payload: dict[str, Any],
    case_file_audit: dict[str, Any],
    material_law: dict[str, Any],
    material_law_file_audit: dict[str, Any],
) -> dict[str, Any]:
    output = _output_directory(output_directory)
    if not isinstance(case_spec, FreshCaseSpec):
        raise TypeError("case_spec must be a FreshCaseSpec")
    if case_contract(case_spec) != case_payload:
        raise RuntimeError("case payload is not the exact reconstructed request")
    source_audit = require_source(source)
    repository = Path(__file__).resolve().parents[3]
    repository_dirty = _git(
        repository, "status", "--porcelain", "--untracked-files=all"
    )
    if repository_dirty:
        raise RuntimeError("candidate source solve requires a clean repository")
    provenance = {
        "repository_commit": _git(repository, "rev-parse", "HEAD"),
        "repository_dirty_porcelain": repository_dirty,
        "fdtdx_source": source_audit["actual"],
        "runtime_lock": load_runtime_lock(),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "implementation_sha256": {
            str(path.relative_to(HERE)): file_sha256(path)
            for path in IMPLEMENTATION_FILES
        },
    }
    model = build_model(
        case_spec.mesh,
        polarization,
        total_periods=case_spec.time.total_periods,
        window_periods=case_spec.time.window_periods,
        courant_factor=case_spec.time.courant_factor,
        alpha_scale=case_spec.pml_alpha_scale,
        target_reflection=case_spec.pml_target_reflection,
        include_adjoint_source=False,
        air_only_source_calibration=True,
        material_law_contract=material_law,
    )
    model_audit = candidate_source_model_audit(model, material_law)
    arrays, air_audit = all_air_arrays(model)
    pre_solve_checks = {
        "candidate_model_audit_ready": model_audit["ready"],
        "all_air_material_readback_ready": air_audit["ready"],
        "mesh_contract_exact": model["fresh_mesh_audit"] == mesh_audit(case_spec.mesh),
        "time_step_exact": float(model["config"].time_step_duration)
        == float(material_law["case_binding"]["realized_float32_cfl"]["time_step_s"]),
        "pml_contract_exact": model["pml_face_parameters"]
        == case_payload["resolved_pml_face_parameters"],
    }
    if not all(pre_solve_checks.values()):
        raise RuntimeError(f"candidate source pre-solve audit failed: {pre_solve_checks}")

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
    evaluation, fields = evaluate_output(model, fdtd_output, polarization)
    fields.update(
        grid_x_edges_m=np.asarray(model["grid"].edges(0)),
        grid_y_edges_m=np.asarray(model["grid"].edges(1)),
        grid_z_edges_m=np.asarray(model["grid"].edges(2)),
    )
    raw_path = output / RAW_NAME
    _atomic_npz(raw_path, **fields)
    ready = all(pre_solve_checks.values()) and evaluation["ready"]
    payload = {
        "status": CASE_STATUS if ready else BLOCKED_STATUS,
        "ready": ready,
        "polarization": polarization,
        "scope": SCOPE,
        "numerical_case_contract": case_payload,
        "numerical_case_file_audit": case_file_audit,
        "candidate_material_law_contract": material_law,
        "candidate_material_law_file_audit": material_law_file_audit,
        "candidate_source_model_audit": model_audit,
        "pre_solve_checks": pre_solve_checks,
        "mesh": model["fresh_mesh_audit"],
        "time_contract": realized_time_contract(case_spec, model),
        "source_contract": model["source_contract"],
        "pml_face_parameters": model["pml_face_parameters"],
        "placement": model["placement"],
        "all_air_material_readback": air_audit,
        "evaluation": evaluation,
        "solve_runtime_s": solve_runtime,
        "reporting_incident_power_W": CONTRACT.reporting_incident_power_W,
        "per_case_scale_not_authorized_until_pair_comparison": True,
        "raw": {
            "path": str(raw_path),
            "sha256": file_sha256(raw_path),
            "arrays": {name: list(value.shape) for name, value in fields.items()},
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
            args.polarization,
            case_spec,
            case_payload,
            case_audit,
            material_law,
            material_law_audit,
        )
    except Exception as error:
        failure = {
            "status": EXCEPTION_STATUS,
            "ready": False,
            "polarization": args.polarization,
            "case_contract_path": str(args.case_contract.expanduser().resolve()),
            "case_contract_expected_sha256": args.case_contract_sha256,
            "material_law_path": str(args.material_law.expanduser().resolve()),
            "material_law_expected_sha256": args.material_law_sha256,
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
        "polarization": result["polarization"],
        "case_contract_sha256": result["numerical_case_contract"][
            "case_contract_sha256"
        ],
        "material_law_contract_sha256": result[
            "candidate_material_law_contract"
        ]["material_law_contract_sha256"],
        "incident_power_W": result["evaluation"]["flux"][
            "incident_plane_signed_W"
        ],
        "maximum_complex_E_NRMSE": result["evaluation"]["stationarity"][
            "maximum_complex_E_NRMSE"
        ],
        "gates": result["evaluation"]["gates"],
        "solve_runtime_s": result["solve_runtime_s"],
        "report": str(args.output_dir.expanduser().resolve() / JSON_NAME),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
