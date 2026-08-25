#!/usr/bin/env python3
"""Source-bound exact-binary Au reference on the user-balanced FDTDX mesh."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time
import traceback
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    REFERENCE_NAMES,
    reference_mask,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_material import (
    arrays_for_exact_binary,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    TimeSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
    _power_evaluation,
    material_stack_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_exact_binary_control import (
    EXPECTED_FDTDX_COMMIT,
    _json_default,
    _json_safe_memory_stats,
    _source_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_increment_state_source_only import (
    _atomic_json,
    _atomic_npz,
    _git,
    _output_directory,
    _sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_mesh import (
    UserBalancedMeshSpec,
    build_model,
    mesh_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_source_only import (
    balanced_case_contract,
    realized_time_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_source_pair import (
    validate_source_pair,
)


VERSION = "fdtdx-user-balanced-exact-binary-v1"
STATUS_READY = "VALIDATED_FDTDX_USER_BALANCED_EXACT_BINARY"
STATUS_BLOCKED = "BLOCKED_FDTDX_USER_BALANCED_EXACT_BINARY"
STATUS_EXCEPTION = "BLOCKED_FDTDX_USER_BALANCED_EXACT_BINARY_EXCEPTION"
SCOPE = (
    "one source-normalized forward-only exact-binary Au reference on the "
    "requested balanced mesh; no thermal/electrical/adjoint/optimizer"
)
DEFAULT_REFERENCE = "l_shape_4um_with_500nm_arms"
REPORT_NAME = "FDTDX_USER_BALANCED_EXACT_BINARY.json"
RAW_NAME = "FDTDX_USER_BALANCED_EXACT_BINARY_FIELDS.npz"


def _runtime_lock(model: dict[str, Any]) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "jax": model["jax"].__version__,
        "fdtdx_import": str(Path(model["fdtdx"].__file__).resolve()),
    }


def source_pair_contract_checks(
    source_pair: dict[str, Any],
    source_audit: dict[str, Any],
    model: dict[str, Any],
    time_spec: TimeSpec,
    polarization: str,
) -> dict[str, bool]:
    contracts = source_pair["source_case_contracts"]
    return {
        "numerical_case_exact": contracts["numerical_case_contract"]
        == balanced_case_contract(time_spec),
        "mesh_exact": contracts["mesh"] == model["fresh_mesh_audit"],
        "time_exact": contracts["time_contract"]
        == realized_time_contract(time_spec, model),
        "pml_exact": contracts["pml_face_parameters"] == model["pml_face_parameters"],
        "placement_exact": contracts["placement"] == model["placement"],
        "polarized_source_exact": contracts["source_contracts"][polarization]
        == model["source_contract"],
        "fdtdx_source_exact": contracts["fdtdx_source"] == source_audit,
        "runtime_lock_exact": contracts["runtime_lock"] == _runtime_lock(model),
    }


def _device_audit(model: dict[str, Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    devices = model["jax"].devices()
    audit = [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
            "memory_stats": _json_safe_memory_stats(device),
        }
        for device in devices
    ]
    return devices, audit


def run(
    output_directory: Path,
    source: Path,
    source_pair_path: Path,
    source_pair_sha256: str,
    polarization: str,
    reference: str,
    total_periods: int,
    window_periods: int,
    courant_factor: float,
) -> dict[str, Any]:
    started_total = time.perf_counter()
    output = _output_directory(output_directory)
    source_audit = _source_audit(source)
    if not source_audit["ready"]:
        raise RuntimeError(f"patched FDTDX source audit failed: {source_audit}")
    if Path(os.environ.get("FDTDX_SOURCE_DIR", "")).resolve() != Path(
        source_audit["path"]
    ):
        raise RuntimeError("FDTDX_SOURCE_DIR does not match --source")
    if polarization not in ("Ea", "Eb"):
        raise ValueError("polarization must be Ea or Eb")
    time_spec = TimeSpec(
        total_periods=total_periods,
        window_periods=window_periods,
        courant_factor=courant_factor,
    )
    source_pair, source_pair_audit = validate_source_pair(
        source_pair_path, source_pair_sha256, time_spec
    )
    if not source_pair_audit["ready"]:
        raise RuntimeError(f"source-pair artifact audit failed: {source_pair_audit}")

    repository = Path(__file__).resolve().parents[3]
    dirty_before = _git(repository, "status", "--porcelain", "--untracked-files=all")
    if dirty_before:
        raise RuntimeError("repository must be clean before balanced material solve")

    started_build = time.perf_counter()
    model = build_model(
        polarization,
        total_periods=time_spec.total_periods,
        window_periods=time_spec.window_periods,
        courant_factor=time_spec.courant_factor,
        include_adjoint_source=False,
        air_only_source_calibration=False,
        dispersive_state_representation="increment",
    )
    contract_checks = source_pair_contract_checks(
        source_pair, source_audit, model, time_spec, polarization
    )
    if not all(contract_checks.values()):
        raise RuntimeError(
            f"material/source numerical contract mismatch: {contract_checks}"
        )

    spec = UserBalancedMeshSpec()
    mask = np.asarray(reference_mask(reference), dtype=np.uint8)
    arrays = arrays_for_exact_binary(model, mask, spec)
    material = material_stack_audit(model, arrays, mask, spec)
    build_runtime_s = time.perf_counter() - started_build
    if not material["ready"]:
        raise RuntimeError(f"exact-binary material readback failed: {material}")

    devices, device_before = _device_audit(model)
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("exactly one visible GPU is required before solve")

    started_solve = time.perf_counter()
    _, fdtd_output = model["fdtdx"].run_fdtd(
        arrays,
        model["placed"],
        model["config"],
        model["key"],
        show_progress=False,
    )
    marker = fdtd_output.detector_states["target_field"]["phasor"]
    model["jax"].block_until_ready(marker)
    solve_runtime_s = time.perf_counter() - started_solve

    started_evaluation = time.perf_counter()
    evaluation, raw_arrays = _power_evaluation(
        model, fdtd_output, mask, source_pair, spec
    )
    raw_path = output / RAW_NAME
    _atomic_npz(raw_path, raw_arrays)
    evaluation_runtime_s = time.perf_counter() - started_evaluation

    dirty_after = _git(repository, "status", "--porcelain", "--untracked-files=all")
    provenance_checks = {
        "repository_clean_before_and_after": dirty_before == dirty_after == "",
        "source_pair_artifacts_ready": source_pair_audit["ready"],
        "source_pair_contracts_exact": all(contract_checks.values()),
        "fdtdx_commit_exact": source_audit["commit"] == EXPECTED_FDTDX_COMMIT,
        "fdtdx_source_clean": source_audit["dirty_porcelain"] == "",
        "one_visible_gpu": len(devices) == 1 and devices[0].platform == "gpu",
        "increment_state_selected": (
            model["config"].dispersive_state_representation == "increment"
        ),
        "physical_one_pole_selected": model["material_law_mode"]
        == "physical-one-pole-increment-state",
        "requested_mesh_exact": model["fresh_mesh_audit"] == mesh_audit(),
        "exact_binary_material_ready": material["ready"],
        "gray_material_law_not_used": material["exact_binary_au"][
            "gray_density_allowed"
        ]
        is False,
    }
    ready = all(provenance_checks.values()) and evaluation["ready"]
    payload = {
        "version": VERSION,
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "failed_provenance_checks": [
            name for name, passed in provenance_checks.items() if not passed
        ],
        "scope": SCOPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "polarization": polarization,
        "reference": reference,
        "numerical_case_contract": balanced_case_contract(time_spec),
        "mesh": model["fresh_mesh_audit"],
        "time_contract": realized_time_contract(time_spec, model),
        "source_contract": model["source_contract"],
        "pml_face_parameters": model["pml_face_parameters"],
        "placement": model["placement"],
        "source_pair": source_pair_audit,
        "source_pair_contract_checks": contract_checks,
        "material": material,
        "evaluation": evaluation,
        "normalization_policy": {
            "raw_fields_and_Q_are_unscaled": True,
            "per_polarization_power_matching_forbidden": True,
            "common_power_scale": source_pair["common_normalization"][
                "common_power_scale"
            ],
            "common_field_amplitude_scale": source_pair["common_normalization"][
                "common_field_amplitude_scale"
            ],
        },
        "runtime": {
            "cold_build_and_array_preparation_s": build_runtime_s,
            "cold_compile_and_forward_s": solve_runtime_s,
            "host_evaluation_and_raw_write_s": evaluation_runtime_s,
            "total_s": time.perf_counter() - started_total,
            "interpretation": (
                "first-call compile plus one forward only; not an adjoint or "
                "optimization-iteration timing"
            ),
        },
        "raw": {
            "path": str(raw_path),
            "sha256": _sha256(raw_path),
            "arrays": {
                name: list(np.asarray(value).shape)
                for name, value in raw_arrays.items()
            },
        },
        "device_before": device_before,
        "device_after": _device_audit(model)[1],
        "runtime_lock": _runtime_lock(model),
        "provenance": {
            "repository_commit": _git(repository, "rev-parse", "HEAD"),
            "repository_dirty_porcelain_before": dirty_before,
            "repository_dirty_porcelain_after": dirty_after,
            "fdtdx_source": source_audit,
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "lumerical_used": False,
        },
        "provenance_checks": provenance_checks,
        "optimizer_start_allowed": False,
    }
    report_path = output / REPORT_NAME
    _atomic_json(report_path, payload)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "ready": ready,
                "failed_evaluation_gates": evaluation["failed_gates"],
                "late_total_Q_W_unscaled": evaluation["Q"]["late"]["total_W"],
                **payload["runtime"],
            },
            default=_json_default,
        )
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-pair", type=Path, required=True)
    parser.add_argument("--source-pair-sha256", required=True)
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument(
        "--reference", choices=REFERENCE_NAMES, default=DEFAULT_REFERENCE
    )
    parser.add_argument("--total-periods", type=int, default=24)
    parser.add_argument("--window-periods", type=int, default=4)
    parser.add_argument("--courant-factor", type=float, default=0.5)
    args = parser.parse_args()
    try:
        payload = run(
            args.output_directory,
            args.source,
            args.source_pair,
            args.source_pair_sha256,
            args.polarization,
            args.reference,
            args.total_periods,
            args.window_periods,
            args.courant_factor,
        )
    except Exception as error:
        failure = {
            "version": VERSION,
            "status": STATUS_EXCEPTION,
            "ready": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "polarization": args.polarization,
            "reference": args.reference,
            "optimizer_start_allowed": False,
        }
        output = args.output_directory.expanduser().resolve()
        if output.is_dir() and not any(output.iterdir()):
            _atomic_json(output / REPORT_NAME, failure)
        print(json.dumps(failure, default=_json_default))
        return 2
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
