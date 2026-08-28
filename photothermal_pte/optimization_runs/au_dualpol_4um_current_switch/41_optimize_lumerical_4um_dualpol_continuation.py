#!/usr/bin/env python3
"""Run resumable Lumerical LD_MMA beta continuation to an exact Au mask."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import nlopt
import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_continuation import (
    BETA_SCHEDULE,
    DESIGN_CONSTRAINT_TOLERANCE,
    EPIGRAPH_CONSTRAINT_TOLERANCE,
    FINAL_GRAYNESS_CAP,
    INITIAL_MAXIMIN_WARM_MAXIMUM_CHANGE,
    MMA_INITIAL_STEP,
    STAGE_FTOL_REL,
    STAGE_MAXEVAL,
    STAGE_XTOL_REL,
    ContinuationEpigraphProblem,
    active_design_constraint_names,
    continuation_contract,
    grayness_value_gradient,
    linearized_maximin_box_warm_start,
    stage_objective_progress,
    stage_design_caps,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    OPTIMIZER_250NM_MAPPING,
    exact_binary_cell_candidate,
    smooth_lumerical_250nm_constraints,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_optimizer import (
    LumericalEvaluationDriver,
    OptimizerRuntime,
    artifact,
    uniform_initial_latent_density,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_maxwell_contract import (
    binary_mask_sha256,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
CHECKPOINT_SCHEMA = "au-lumerical-continuation-checkpoint-v3"
LEGACY_CHECKPOINT_SCHEMA = "au-lumerical-continuation-checkpoint-v2"
PREFLIGHT_STATUS = "PASSED_LUMERICAL_4UM_CONTINUATION_PREFLIGHT_ONLY"
FINAL_EXACT_BINARY_CERTIFICATE_SCHEMA = (
    "au-lumerical-exact-binary-lateral-pde-certificate-v3"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Write the solver-free continuation/runtime audit and exit.",
    )
    parser.add_argument(
        "--stop-after-beta",
        type=float,
        choices=BETA_SCHEDULE,
        help="Testing/controlled handoff gate; omit for full continuation.",
    )
    return parser.parse_args()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _failure_output_root() -> Path | None:
    """Return only an explicitly configured path outside the repository."""

    for name in ("AU_LUMERICAL_OPT_OUTPUT_ROOT", "EIDL_RUN_DIR"):
        raw = os.environ.get(name)
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        try:
            candidate.relative_to(REPOSITORY.resolve())
        except ValueError:
            return candidate
    return None


def _new_manifest(runtime: OptimizerRuntime) -> dict[str, Any]:
    physical_device = json.loads(
        (HERE / "physical_device_contract.json").read_text(encoding="utf-8")
    )
    return {
        "schema": "au-lumerical-dualpol-production-continuation-v2",
        "status": "RUNNING_LUMERICAL_4UM_DUALPOL_BETA_CONTINUATION",
        "passed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "optimizer": {
            "library": "NLopt",
            "version": nlopt.__version__,
            "algorithm": "LD_MMA",
        },
        "optimizer_lifecycle": {
            "normal_MMA_instances_per_beta": 1,
            "same_beta_restart": "crash recovery only",
            "MMA_internal_state_serialized": False,
            "recovery_semantics": "best successful density plus callback history; new MMA",
        },
        "physical_device_contract_status": physical_device.get("status"),
        "physical_device_assumptions_confirmed": physical_device.get(
            "assumptions_confirmed", False
        ),
        "runtime": runtime.audit(),
        "continuation_contract": continuation_contract(),
        "component_yee_independent_FD_cadence": {
            "schema": "component-yee-independent-fd-cadence-v1",
            "stage_policy": (
                "one full independent mapping FD at the first representative "
                "physics evaluation of each beta; later evaluations use the "
                "hash-bound certificate plus fresh construction/transpose gates"
            ),
            "final_policy": (
                "one final full independent mapping FD on the continuous "
                "precursor after its exact-binary physical certificate passes"
            ),
            "mapping_FD_relative_limit_unchanged": True,
            "stage_certificates": {},
        },
        "full_chain_current_AD_FD_cadence": {
            "schema": "full-chain-current-adfd-cadence-v1",
            "stage_policy": (
                "one independent centered latent-direction Ea/Eb current audit "
                "at each beta entry; no per-evaluation full-chain FD"
            ),
            "final_policy": (
                "one audit on the differentiable continuous precursor plus a "
                "separate fresh exact-binary physical certificate"
            ),
            "relative_error_limit": 0.01,
            "stage_certificates": {},
        },
        "active_stage": None,
        "stages": [],
        "final": None,
        "Lumerical_HEAT_or_CHARGE_solves": 0,
        "FDTDX_Maxwell_solves": 0,
    }


def _save_checkpoint(
    path: Path,
    *,
    latent: np.ndarray,
    beta_index: int,
    attempt: int,
    dfm_caps: np.ndarray,
    grayness_cap: float,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        schema=np.asarray(CHECKPOINT_SCHEMA),
        latent=np.asarray(latent, dtype=np.float64),
        beta_index=np.asarray(beta_index, dtype=np.int64),
        attempt=np.asarray(attempt, dtype=np.int64),
        dfm_caps=np.asarray(dfm_caps, dtype=np.float64),
        grayness_cap=np.asarray(grayness_cap, dtype=np.float64),
    )
    temporary.replace(path)


def _save_final_binary_mask(path: Path, mask: np.ndarray) -> None:
    value = np.asarray(mask, dtype=np.uint8)
    if value.shape != CONTRACT.design_shape or not np.all((value == 0) | (value == 1)):
        raise ValueError("final binary mask must be exact 80x80 zero/one")
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            existing = np.asarray(data["binary_mask"], dtype=np.uint8)
        if not np.array_equal(existing, value):
            raise RuntimeError("refusing to overwrite a different final binary mask")
        return
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, binary_mask=value)
    temporary.replace(path)


def _save_stage_progress_state(
    path: Path, *, latent: np.ndarray, projected: np.ndarray
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        latent=np.asarray(latent, dtype=np.float64),
        projected=np.asarray(projected, dtype=np.float64),
    )
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        schema = str(np.asarray(data["schema"]).item())
        if schema not in {CHECKPOINT_SCHEMA, LEGACY_CHECKPOINT_SCHEMA}:
            raise RuntimeError(f"checkpoint schema changed: {schema}")
        latent = np.asarray(data["latent"], dtype=np.float64)
        if latent.shape != CONTRACT.design_node_shape:
            raise RuntimeError("checkpoint latent shape changed")
        return {
            "latent": latent,
            "beta_index": int(np.asarray(data["beta_index"]).item()),
            "attempt": int(np.asarray(data["attempt"]).item()),
            "dfm_caps": np.asarray(data["dfm_caps"], dtype=np.float64),
            "grayness_cap": float(np.asarray(data["grayness_cap"]).item()),
        }


def _verified_artifact(
    record: object,
    *,
    label: str,
    relative_to: Path | None = None,
) -> Path:
    if not isinstance(record, dict):
        raise RuntimeError(f"completed manifest lacks {label} artifact")
    recorded = Path(str(record.get("path", ""))).expanduser()
    if not recorded.is_absolute() and relative_to is not None:
        recorded = relative_to / recorded
    path = recorded.resolve()
    if not path.is_file():
        raise RuntimeError(f"completed manifest {label} artifact is absent")
    actual = artifact(path)
    if (
        int(record.get("size_bytes", -1)) != actual["size_bytes"]
        or str(record.get("sha256", "")) != actual["sha256"]
    ):
        raise RuntimeError(f"completed manifest {label} artifact changed")
    return path


def _fd_beta_key(beta: float) -> str:
    return f"{float(beta):g}"


def _fd_cadence_record(manifest: dict[str, Any]) -> dict[str, Any]:
    cadence = manifest.get("component_yee_independent_FD_cadence")
    if not isinstance(cadence, dict):
        raise RuntimeError("production manifest lacks component-Yee FD cadence")
    if cadence.get("schema") != "component-yee-independent-fd-cadence-v1":
        raise RuntimeError("component-Yee FD cadence schema changed")
    certificates = cadence.get("stage_certificates")
    if not isinstance(certificates, dict):
        raise RuntimeError("component-Yee FD stage certificates are malformed")
    return cadence


def _stage_fd_certificate(manifest: dict[str, Any], beta: float) -> Path | None:
    cadence = _fd_cadence_record(manifest)
    record = cadence["stage_certificates"].get(_fd_beta_key(beta))
    if record is None:
        return None
    if not isinstance(record, dict) or float(record.get("beta", np.nan)) != float(beta):
        raise RuntimeError("component-Yee stage FD beta record is malformed")
    path = _verified_artifact(
        record.get("Jacobian_result"), label=f"beta-{beta:g} stage FD certificate"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    validation = result.get("validation")
    gates = {
        "result_passed": result.get("passed") is True,
        "all_result_gates_passed": bool(result.get("gates"))
        and all(result.get("gates", {}).values()),
        "full_independent_mapping_FD_performed": isinstance(validation, dict)
        and validation.get("independent_mapping_FD_performed") is True,
        "scope_is_stage_entry": result.get("validation_scope") == "stage-entry",
        "beta_matches": float(result.get("optimization_beta", np.nan)) == float(beta),
        "git_commit_matches": result.get("git_commit") == _git_commit(),
    }
    if not all(gates.values()):
        raise RuntimeError(
            f"beta-{beta:g} stage FD certificate failed revalidation: {gates}"
        )
    return path


def _record_stage_fd_certificate(
    *,
    manifest: dict[str, Any],
    beta: float,
    attempt: int,
    initial_physics: dict[str, Any],
) -> Path:
    cadence = _fd_cadence_record(manifest)
    key = _fd_beta_key(beta)
    if key in cadence["stage_certificates"]:
        raise RuntimeError(f"refusing to replace beta-{beta:g} FD certificate")
    validation_record = initial_physics.get("Jacobian_validation")
    if (
        not isinstance(validation_record, dict)
        or validation_record.get("independent_mapping_FD_performed") is not True
    ):
        raise RuntimeError("stage-entry representative did not run independent FD")
    path = _verified_artifact(
        initial_physics.get("Jacobian_result"),
        label=f"beta-{beta:g} representative Jacobian",
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    validation = result.get("validation")
    gates = {
        "result_passed": result.get("passed") is True,
        "all_result_gates_passed": bool(result.get("gates"))
        and all(result.get("gates", {}).values()),
        "full_independent_mapping_FD_performed": isinstance(validation, dict)
        and validation.get("independent_mapping_FD_performed") is True,
        "scope_is_stage_entry": result.get("validation_scope") == "stage-entry",
        "beta_matches": float(result.get("optimization_beta", np.nan)) == float(beta),
        "git_commit_matches": result.get("git_commit") == _git_commit(),
    }
    if not all(gates.values()):
        raise RuntimeError(f"beta-{beta:g} representative FD result failed: {gates}")
    cadence["stage_certificates"][key] = {
        "beta": float(beta),
        "attempt": int(attempt),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "representative_density_state": initial_physics.get("density_state"),
        "Jacobian_result": artifact(path),
        "mapping_FD_relative_limit_unchanged": True,
    }
    return path


def _full_chain_cadence_record(manifest: dict[str, Any]) -> dict[str, Any]:
    cadence = manifest.get("full_chain_current_AD_FD_cadence")
    if (
        not isinstance(cadence, dict)
        or cadence.get("schema") != "full-chain-current-adfd-cadence-v1"
        or not isinstance(cadence.get("stage_certificates"), dict)
    ):
        raise RuntimeError("production manifest lacks full-chain AD-FD cadence")
    return cadence


def _stage_full_chain_certificate(manifest: dict[str, Any], beta: float) -> Path | None:
    cadence = _full_chain_cadence_record(manifest)
    record = cadence["stage_certificates"].get(_fd_beta_key(beta))
    if record is None:
        return None
    path = _verified_artifact(
        record.get("result"), label=f"beta-{beta:g} full-chain AD-FD"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    gates = {
        "passed": result.get("passed") is True,
        "schema": result.get("schema")
        == "au-lumerical-full-chain-latent-current-adfd-v1",
        "beta_matches": float(result.get("optimization_beta", np.nan)) == float(beta),
        "scope_is_stage_entry": result.get("validation_scope") == "stage-entry",
        "all_physics_gates_passed": bool(result.get("gates"))
        and all(result.get("gates", {}).values()),
        "git_commit_matches": result.get("evaluation_contract", {}).get("git_commit")
        == _git_commit(),
    }
    if not all(gates.values()):
        raise RuntimeError(
            f"beta-{beta:g} full-chain AD-FD revalidation failed: {gates}"
        )
    return path


def _record_stage_full_chain_certificate(
    *,
    manifest: dict[str, Any],
    beta: float,
    attempt: int,
    result_path: Path,
    result: dict[str, Any],
) -> None:
    cadence = _full_chain_cadence_record(manifest)
    key = _fd_beta_key(beta)
    if key in cadence["stage_certificates"]:
        raise RuntimeError(f"refusing to replace beta-{beta:g} full-chain AD-FD")
    if result.get("passed") is not True:
        raise RuntimeError("cannot record a failed full-chain AD-FD result")
    cadence["stage_certificates"][key] = {
        "beta": float(beta),
        "attempt": int(attempt),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": artifact(result_path),
    }


def _restart_seed_from_environment() -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Verify an explicitly requested stopped-run checkpoint migration."""

    checkpoint_raw = os.environ.get("AU_LUMERICAL_RESTART_CHECKPOINT")
    manifest_raw = os.environ.get("AU_LUMERICAL_RESTART_MANIFEST")
    if not checkpoint_raw and not manifest_raw:
        return None
    if not checkpoint_raw or not manifest_raw:
        raise RuntimeError(
            "AU_LUMERICAL_RESTART_CHECKPOINT and "
            "AU_LUMERICAL_RESTART_MANIFEST must be set together"
        )
    checkpoint_path = Path(checkpoint_raw).expanduser().resolve()
    manifest_path = Path(manifest_raw).expanduser().resolve()
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("restart checkpoint or manifest is absent")

    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("passed") is not False or not str(
        source_manifest.get("status", "")
    ).startswith("STOPPED_"):
        raise RuntimeError(
            "restart source must be an explicitly stopped non-passed run"
        )
    state = _load_checkpoint(checkpoint_path)
    beta_index = int(state["beta_index"])
    if not 0 <= beta_index < len(BETA_SCHEDULE):
        raise RuntimeError("restart checkpoint beta index is terminal or invalid")
    latest = source_manifest.get("latest")
    if not isinstance(latest, dict):
        raise RuntimeError("restart manifest lacks latest stage evidence")
    if float(latest.get("beta", np.nan)) != BETA_SCHEDULE[beta_index]:
        raise RuntimeError("restart manifest beta differs from checkpoint")
    blocking_count = source_manifest.get("blocking_attempts")
    if blocking_count is None and "blocking_recovery_index" in source_manifest:
        blocking_count = int(source_manifest["blocking_recovery_index"]) + 1
    if int(blocking_count if blocking_count is not None else -1) != int(
        state["attempt"]
    ):
        raise RuntimeError("restart manifest recovery count differs from checkpoint")

    stages = source_manifest.get("stages")
    if not isinstance(stages, list) or not stages:
        raise RuntimeError("restart manifest lacks completed stage attempts")
    terminal_state_path = _verified_artifact(
        stages[-1].get("state_artifact"),
        label="restart terminal stage state",
        relative_to=manifest_path.parent,
    )
    with np.load(terminal_state_path, allow_pickle=False) as arrays:
        terminal_latent = np.asarray(arrays["latent_final"], dtype=np.float64)
    if not np.array_equal(terminal_latent, np.asarray(state["latent"])):
        raise RuntimeError(
            "restart checkpoint latent differs from terminal stage state"
        )

    provenance = {
        "schema": "au-lumerical-continuation-restart-provenance-v1",
        "reason": "explicit_cross_commit_density_recovery_from_stopped_run",
        "source_git_commit": source_manifest.get("git_commit"),
        "source_status": source_manifest.get("status"),
        "source_latest": latest,
        "source_manifest": artifact(manifest_path),
        "source_checkpoint": artifact(checkpoint_path),
        "source_terminal_state": artifact(terminal_state_path),
        "resumed_beta_index": beta_index,
        "resumed_attempt": int(state["attempt"]),
    }
    return state, provenance


def _completed_manifest_latent(manifest: dict[str, Any]) -> np.ndarray | None:
    """Verify and recover the terminal latent state after an interrupted commit."""

    if manifest.get("passed") is not True:
        return None
    if manifest.get("status") == PREFLIGHT_STATUS:
        if not (
            manifest.get("preflight_only") is True
            and manifest.get("stages") == []
            and manifest.get("final") is None
        ):
            raise RuntimeError("preflight-only continuation manifest is malformed")
        return None
    if not str(manifest.get("status", "")).startswith(
        "PASSED_LUMERICAL_4UM_DUALPOL_EXACT_BINARY_AU_"
    ):
        raise RuntimeError("passed continuation manifest has an invalid status")
    final = manifest.get("final")
    if not isinstance(final, dict):
        raise RuntimeError("passed continuation manifest lacks final evidence")
    exact = final.get("exact_binary_evaluation")
    if not isinstance(exact, dict) or exact.get("passed") is not True:
        raise RuntimeError("passed continuation manifest lacks a passed certificate")
    if exact.get("schema") != FINAL_EXACT_BINARY_CERTIFICATE_SCHEMA:
        raise RuntimeError("passed continuation exact-certificate schema is stale")
    if exact.get("git_commit") != manifest.get("git_commit"):
        raise RuntimeError("passed continuation exact-certificate commit differs")
    currents = exact.get("currents_A", {})
    if not (
        isinstance(currents, dict)
        and set(currents) == {"Ea", "Eb"}
        and float(currents["Ea"]) > 0.0
        and float(currents["Eb"]) < 0.0
    ):
        raise RuntimeError("passed continuation manifest lost the strict current signs")
    binary_path = _verified_artifact(final.get("binary_mask"), label="binary mask")
    with np.load(binary_path, allow_pickle=False) as arrays:
        mask = np.asarray(arrays["binary_mask"])
    if (
        mask.shape != CONTRACT.design_shape
        or not np.all((mask == 0) | (mask == 1))
        or exact.get("binary_mask_payload_sha256") != binary_mask_sha256(mask)
    ):
        raise RuntimeError("passed continuation binary-mask payload changed")
    stage = final.get("continuous_stage")
    if not isinstance(stage, dict):
        raise RuntimeError("passed continuation manifest lacks continuous stage")
    state_path = _verified_artifact(stage.get("state_artifact"), label="stage state")
    with np.load(state_path, allow_pickle=False) as arrays:
        latent = np.asarray(arrays["latent_final"], dtype=np.float64)
    if (
        latent.shape != CONTRACT.design_node_shape
        or not np.all(np.isfinite(latent))
        or np.min(latent) < 0.0
        or np.max(latent) > 1.0
    ):
        raise RuntimeError("passed continuation terminal latent state is invalid")
    return latent


def _attempt_directory(root: Path, beta: float, attempt: int) -> tuple[Path, int]:
    value = int(attempt)
    while True:
        candidate = root / "stages" / f"beta_{beta:03g}_recovery_{value:02d}"
        if not candidate.exists():
            return candidate, value
        value += 1


def _stage_constraints_satisfied(point: dict[str, Any]) -> bool:
    values = np.asarray(
        point["design_constraints"]["normalized_values"], dtype=np.float64
    )
    return bool(values.size == 0 or np.max(values) <= DESIGN_CONSTRAINT_TOLERANCE)


def _run_exact_binary_evaluation(
    *,
    runtime: OptimizerRuntime,
    binary_path: Path,
    output: Path,
) -> dict[str, Any]:
    if runtime.final_xy50_source_calibration is None:
        raise RuntimeError(
            "final 50-nm Ea/Eb source calibrations are required for promotion"
        )
    command = [
        sys.executable,
        str(HERE / "43_certify_lumerical_4um_exact_binary_lateral.py"),
        "--binary-mask-npz",
        str(binary_path),
        "--output-dir",
        str(output),
        "--gpu-index",
        str(runtime.gpu_index),
        "--accelerator-policy",
        runtime.accelerator_policy,
        "--threads",
        str(runtime.threads),
        "--ea-coarse-source-calibration",
        str(runtime.source_calibration["Ea"]),
        "--eb-coarse-source-calibration",
        str(runtime.source_calibration["Eb"]),
        "--ea-fine-source-calibration",
        str(runtime.final_xy50_source_calibration["Ea"]),
        "--eb-fine-source-calibration",
        str(runtime.final_xy50_source_calibration["Eb"]),
    ]
    completed = subprocess.run(command, cwd=REPOSITORY, check=False)
    result_path = output / "final_exact_binary_certificate.json"
    if not result_path.is_file():
        raise RuntimeError(
            "exact-binary 100/50-nm lateral/PDE certifier failed with exit "
            f"{completed.returncode}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if completed.returncode != 0 and result.get("error"):
        raise RuntimeError(
            f"exact-binary certifier execution failed: {result.get('error')}"
        )
    return result


def main() -> int:
    args = _parse_args()
    started = time.monotonic()
    output: Path | None = None
    manifest: dict[str, Any] = {
        "status": "FAILED_LUMERICAL_4UM_DUALPOL_BETA_CONTINUATION",
        "passed": False,
    }
    try:
        base_runtime = OptimizerRuntime.from_environment(
            require_smoke_beta=False,
            require_final_xy50_source_calibration=True,
        )
        output = base_runtime.output_root
        manifest_path = output / "production_manifest.json"
        checkpoint_path = output / "continuation_checkpoint.npz"
        if output.exists() and any(output.iterdir()):
            if not manifest_path.is_file() or not checkpoint_path.is_file():
                raise RuntimeError(
                    "non-empty output is not a resumable production continuation"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            state = _load_checkpoint(checkpoint_path)
            if manifest.get("git_commit") != _git_commit():
                raise RuntimeError(
                    "refusing resume under a different Git commit; use the committed run worktree"
                )
            completed_latent = _completed_manifest_latent(manifest)
            if completed_latent is not None:
                state.update(
                    latent=completed_latent,
                    beta_index=len(BETA_SCHEDULE),
                    attempt=0,
                )
                _save_checkpoint(checkpoint_path, **state)
                print(json.dumps(manifest["final"], indent=2, default=str))
                return 0
        else:
            output.mkdir(parents=True, exist_ok=True)
            (output / "stages").mkdir()
            (output / "checkpoints").mkdir()
            manifest = _new_manifest(base_runtime)
            restart_seed = _restart_seed_from_environment()
            if restart_seed is None:
                latent = uniform_initial_latent_density()
                projected = OPTIMIZER_250NM_MAPPING.physical(latent, BETA_SCHEDULE[0])
                if not np.array_equal(projected, latent):
                    raise RuntimeError(
                        "uniform rho=0.5 is not exactly preserved at beta=1"
                    )
                state = {
                    "latent": latent,
                    "beta_index": 0,
                    "attempt": 0,
                    "dfm_caps": np.full(2, np.inf, dtype=np.float64),
                    "grayness_cap": np.inf,
                }
            else:
                state, restart_provenance = restart_seed
                manifest["restart_provenance"] = restart_provenance
            _save_checkpoint(checkpoint_path, **state)
            _write_json(manifest_path, manifest)

        if args.preflight_only:
            manifest["status"] = PREFLIGHT_STATUS
            manifest["passed"] = True
            manifest["preflight_only"] = True
            manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(manifest_path, manifest)
            print(json.dumps(manifest, indent=2, default=str))
            return 0

        manifest["status"] = "RUNNING_LUMERICAL_4UM_DUALPOL_BETA_CONTINUATION"
        manifest["passed"] = False
        manifest.pop("preflight_only", None)
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, manifest)

        while state["beta_index"] < len(BETA_SCHEDULE):
            beta_index = int(state["beta_index"])
            beta = BETA_SCHEDULE[beta_index]
            attempt_dir, attempt = _attempt_directory(
                output, beta, int(state["attempt"])
            )
            attempt_dir.mkdir(parents=True)
            runtime = replace(base_runtime, output_root=attempt_dir, beta=beta)
            stage_fd_certificate = _stage_fd_certificate(manifest, beta)
            driver = LumericalEvaluationDriver(
                runtime,
                prune_heavy_intermediates=True,
                independent_fd_certificate=stage_fd_certificate,
                shared_evaluations_root=(
                    output / "evaluation_cache" / f"beta_{beta:03g}"
                ),
            )
            latent_initial = np.asarray(state["latent"], dtype=np.float64)
            initial_dfm, _, _ = smooth_lumerical_250nm_constraints(latent_initial, beta)
            initial_grayness = grayness_value_gradient(latent_initial, beta)[0]
            # Use the checkpoint logical recovery count, not the next free artifact
            # directory suffix. If a process dies before its first successful
            # callback, stage caps remain marked as uninitialized.
            # Persist those caps before the expensive solve so such a restart
            # cannot silently disable DFM or grayness constraints.
            if int(state["attempt"]) == 0:
                cap_record = stage_design_caps(
                    beta=beta,
                    baseline_dfm_values=initial_dfm,
                    baseline_grayness=initial_grayness,
                    previous_dfm_caps=np.asarray(state["dfm_caps"], dtype=np.float64),
                    previous_grayness_cap=float(state["grayness_cap"]),
                )
                state["dfm_caps"] = np.asarray(cap_record["DFM_caps"], dtype=np.float64)
                state["grayness_cap"] = float(cap_record["grayness_cap"])
                _save_checkpoint(checkpoint_path, **state)
            initial_physics = driver.evaluate(latent_initial)
            if stage_fd_certificate is None:
                stage_fd_certificate = _record_stage_fd_certificate(
                    manifest=manifest,
                    beta=beta,
                    attempt=attempt,
                    initial_physics=initial_physics,
                )
                driver.bind_independent_fd_certificate(stage_fd_certificate)
                manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                _write_json(manifest_path, manifest)
            stage_full_chain_certificate = _stage_full_chain_certificate(manifest, beta)
            if stage_full_chain_certificate is None:
                stage_adfd_output = attempt_dir / "stage_entry_full_chain_adfd"
                stage_adfd = driver.audit_full_chain_latent_adfd(
                    latent_initial,
                    initial_physics,
                    stage_adfd_output,
                    validation_scope="stage-entry",
                )
                stage_adfd_path = stage_adfd_output / "full_chain_adfd_result.json"
                _record_stage_full_chain_certificate(
                    manifest=manifest,
                    beta=beta,
                    attempt=attempt,
                    result_path=stage_adfd_path,
                    result=stage_adfd,
                )
                manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                _write_json(manifest_path, manifest)

            active_stage = manifest.get("active_stage")
            history_prefix = (
                list(active_stage.get("callback_history", []))
                if isinstance(active_stage, dict)
                and float(active_stage.get("beta", np.nan)) == beta
                else []
            )

            def persist_successful_callback(
                current_problem: ContinuationEpigraphProblem,
            ) -> None:
                selected_progress = current_problem.selected_candidate()
                latent_progress = np.asarray(
                    selected_progress["latent"], dtype=np.float64
                )
                projected_progress = OPTIMIZER_250NM_MAPPING.physical(
                    latent_progress, beta
                )
                progress_path = attempt_dir / "latest_successful_state.npz"
                _save_stage_progress_state(
                    progress_path,
                    latent=latent_progress,
                    projected=projected_progress,
                )
                state["latent"] = latent_progress
                state["attempt"] = attempt + 1
                _save_checkpoint(checkpoint_path, **state)
                selected_point = selected_progress["point"]
                manifest["active_stage"] = {
                    "status": "RUNNING_FIXED_BETA_MMA",
                    "beta": beta,
                    "beta_index": beta_index,
                    "recovery_index": attempt,
                    "MMA_internal_state_serialized": False,
                    "callback_history": current_problem.complete_callback_history,
                    "latest_best_currents_nA": {
                        "Ea": 1.0e9 * float(selected_point["current_a_A"]),
                        "Eb": 1.0e9 * float(selected_point["current_b_A"]),
                    },
                    "latest_best_balanced_utility_nA": (
                        1.0e9 * float(selected_point["balanced_utility_A"])
                    ),
                    "latest_best_grayness": float(
                        selected_point["design_constraints"]["grayness"]
                    ),
                    "state_artifact": artifact(progress_path),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                manifest["latest"] = {
                    "beta": beta,
                    "recovery_index": attempt,
                    "iteration": len(current_problem.complete_callback_history) - 1,
                    "currents_nA": manifest["active_stage"]["latest_best_currents_nA"],
                    "balanced_utility_nA": manifest["active_stage"][
                        "latest_best_balanced_utility_nA"
                    ],
                    "grayness": manifest["active_stage"]["latest_best_grayness"],
                }
                manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                _write_json(manifest_path, manifest)

            problem = ContinuationEpigraphProblem(
                driver.evaluate,
                beta=beta,
                dfm_caps=np.asarray(state["dfm_caps"], dtype=np.float64),
                grayness_cap=float(state["grayness_cap"]),
                history_prefix=history_prefix,
                progress_callback=persist_successful_callback,
            )
            epigraph_initial_nA = 1.0e9 * min(
                float(initial_physics["currents_A"]["Ea"]),
                -float(initial_physics["currents_A"]["Eb"]),
            )
            optimizer_latent = latent_initial
            optimizer_maxeval = STAGE_MAXEVAL[beta]
            warm_start_audit: dict[str, Any] | None = None
            if (
                beta_index == 0
                and int(state["attempt"]) == 0
                and np.array_equal(latent_initial, uniform_initial_latent_density())
            ):
                initial_point = problem.point(
                    np.r_[latent_initial.ravel(), epigraph_initial_nA]
                )
                warm = linearized_maximin_box_warm_start(
                    latent=latent_initial,
                    current_a_A=float(initial_point["current_a_A"]),
                    current_b_A=float(initial_point["current_b_A"]),
                    gradient_a_latent_A=np.asarray(
                        initial_point["gradient_a_latent_A"], dtype=np.float64
                    ),
                    gradient_b_latent_A=np.asarray(
                        initial_point["gradient_b_latent_A"], dtype=np.float64
                    ),
                    maximum_change=INITIAL_MAXIMIN_WARM_MAXIMUM_CHANGE,
                )
                optimizer_latent = np.asarray(warm["latent"], dtype=np.float64)
                warm_physics = driver.evaluate(optimizer_latent)
                epigraph_initial_nA = 1.0e9 * min(
                    float(warm_physics["currents_A"]["Ea"]),
                    -float(warm_physics["currents_A"]["Eb"]),
                )
                optimizer_maxeval = max(1, STAGE_MAXEVAL[beta] - 1)
                warm_start_audit = {
                    key: value
                    for key, value in warm.items()
                    if key not in {"latent", "delta"}
                }
                warm_start_audit["actual_currents_nA"] = {
                    key: 1.0e9 * float(value)
                    for key, value in warm_physics["currents_A"].items()
                }
                warm_start_audit["actual_balanced_utility_nA"] = epigraph_initial_nA
            variable_count = problem.variable_count
            optimizer = nlopt.opt(nlopt.LD_MMA, variable_count)
            optimizer.set_lower_bounds(
                np.r_[np.zeros(variable_count - 1, dtype=np.float64), -100.0]
            )
            optimizer.set_upper_bounds(
                np.r_[np.ones(variable_count - 1, dtype=np.float64), 1000.0]
            )
            optimizer.set_max_objective(problem.objective)
            tolerances = np.r_[
                np.full(2, EPIGRAPH_CONSTRAINT_TOLERANCE),
                np.full(
                    problem.design_constraint_count,
                    DESIGN_CONSTRAINT_TOLERANCE,
                ),
            ]
            optimizer.add_inequality_mconstraint(problem.constraints, tolerances)
            optimizer.set_initial_step(
                np.r_[
                    np.full(variable_count - 1, MMA_INITIAL_STEP[beta]),
                    0.1,
                ]
            )
            optimizer.set_ftol_rel(STAGE_FTOL_REL)
            optimizer.set_xtol_rel(STAGE_XTOL_REL)
            optimizer.set_maxeval(optimizer_maxeval)
            vector_initial = np.r_[optimizer_latent.ravel(), epigraph_initial_nA]
            problem.bind_force_stop(optimizer.force_stop)
            plateau_forced_stop = False
            try:
                vector_optimizer_terminal = optimizer.optimize(vector_initial)
            except nlopt.ForcedStop:
                if not problem.plateau_stop_requested:
                    raise
                plateau_forced_stop = True
                stopped_candidate = problem.selected_candidate()
                vector_optimizer_terminal = np.r_[
                    np.asarray(stopped_candidate["latent"]).ravel(),
                    float(stopped_candidate["point"]["epigraph_nA"]),
                ]
            objective_progress = stage_objective_progress(
                problem.complete_callback_history
            )
            selected = problem.selected_candidate()
            final_point = selected["point"]
            latent_final = np.asarray(selected["latent"], dtype=np.float64)
            latent_optimizer_terminal = vector_optimizer_terminal[:-1].reshape(
                CONTRACT.design_node_shape
            )
            projected_final = OPTIMIZER_250NM_MAPPING.physical(latent_final, beta)
            binary_mask, binary_audit = exact_binary_cell_candidate(projected_final)
            design_satisfied = _stage_constraints_satisfied(final_point)
            objective_converged = bool(
                objective_progress["converged"] and problem.plateau_stop_requested
            )
            switching = bool(
                float(final_point["current_a_A"]) > 0.0
                and float(final_point["current_b_A"]) < 0.0
            )
            stage_result = {
                "status": "COMPLETED_LUMERICAL_4UM_FIXED_BETA_MMA",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "beta": beta,
                "beta_index": beta_index,
                "recovery_index": attempt,
                "normal_MMA_instance_for_beta": attempt == 0,
                "same_beta_recovery_MMA": attempt > 0,
                "plateau_forced_stop": plateau_forced_stop,
                "active_design_constraints": list(active_design_constraint_names(beta)),
                "NLopt_total_constraint_count": problem.total_constraint_count,
                "safety_maxeval": STAGE_MAXEVAL[beta],
                "optimizer_requested_maxeval": optimizer_maxeval,
                "latent_bounds": [0.0, 1.0],
                "MMA_initial_step": MMA_INITIAL_STEP[beta],
                "initial_grayness": initial_grayness,
                "uniform_baseline_outside_optimizer": warm_start_audit is not None,
                "initial_maximin_warm_start": warm_start_audit,
                "reported_numevals": optimizer.get_numevals(),
                "result_code": optimizer.last_optimize_result(),
                "unique_physics_evaluations": len(driver.history),
                "selected_candidate": {
                    "callback_index": selected["callback_index"],
                    "reason": selected["reason"],
                    "optimizer_terminal_was_selected": bool(
                        np.array_equal(latent_final, latent_optimizer_terminal)
                    ),
                },
                "objective_progress": objective_progress,
                "objective_converged": objective_converged,
                "initial_currents_nA": {
                    key: 1.0e9 * float(value)
                    for key, value in initial_physics["currents_A"].items()
                },
                "final_currents_nA": {
                    "Ea": 1.0e9 * float(final_point["current_a_A"]),
                    "Eb": 1.0e9 * float(final_point["current_b_A"]),
                },
                "balanced_utility_nA": 1.0e9 * float(final_point["balanced_utility_A"]),
                "opposite_current_switching_achieved": switching,
                "design_constraints": {
                    "names": final_point["design_constraints"]["names"],
                    "normalized_values": final_point["design_constraints"][
                        "normalized_values"
                    ].tolist(),
                    "satisfied": design_satisfied,
                    "raw_DFM_values": final_point["design_constraints"][
                        "raw_DFM_values"
                    ].tolist(),
                    "DFM_caps": np.asarray(state["dfm_caps"]).tolist(),
                    "grayness": float(final_point["design_constraints"]["grayness"]),
                    "grayness_cap": float(state["grayness_cap"]),
                },
                "exact_binary_candidate_audit": {
                    key: value
                    for key, value in binary_audit.items()
                    if key not in {"binary", "bad_solid", "bad_void"}
                },
                "callback_history": problem.complete_callback_history,
                "wall_s": float(sum(row["wall_s"] for row in driver.history)),
            }
            state_path = attempt_dir / "stage_final_state.npz"
            np.savez_compressed(
                state_path,
                latent_initial=latent_initial,
                latent_final=latent_final,
                latent_optimizer_terminal=latent_optimizer_terminal,
                projected_final=projected_final,
                binary_candidate_cell_mask=binary_mask,
            )
            stage_result["state_artifact"] = artifact(state_path)
            _write_json(attempt_dir / "stage_result.json", stage_result)
            manifest["stages"].append(stage_result)
            manifest["active_stage"] = None
            manifest["latest"] = {
                "beta": beta,
                "recovery_index": attempt,
                "currents_nA": stage_result["final_currents_nA"],
                "balanced_utility_nA": stage_result["balanced_utility_nA"],
                "grayness": stage_result["design_constraints"]["grayness"],
                "design_constraints_satisfied": design_satisfied,
                "objective_converged": objective_converged,
                "opposite_current_switching_achieved": switching,
            }
            manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(manifest_path, manifest)

            final_beta = beta == BETA_SCHEDULE[-1]
            final_grayness_target_reached = bool(
                final_beta
                and float(state["grayness_cap"])
                <= FINAL_GRAYNESS_CAP * (1.0 + DESIGN_CONSTRAINT_TOLERANCE)
            )
            continuous_binary_gate = bool(
                final_beta
                and design_satisfied
                and objective_converged
                and final_grayness_target_reached
                and float(final_point["design_constraints"]["grayness"])
                <= float(state["grayness_cap"]) * (1.0 + DESIGN_CONSTRAINT_TOLERANCE)
                and binary_audit["solid_pass"]
                and binary_audit["void_pass"]
            )
            if final_beta and continuous_binary_gate and switching:
                attempt_binary_path = (
                    attempt_dir / "exact_binary_candidate_cell_mask.npz"
                )
                np.savez_compressed(attempt_binary_path, binary_mask=binary_mask)
                exact_result = _run_exact_binary_evaluation(
                    runtime=runtime,
                    binary_path=attempt_binary_path,
                    output=attempt_dir / "exact_binary_certificate",
                )
                exact_switching = bool(
                    exact_result.get("passed") is True
                    and exact_result["currents_A"]["Ea"] > 0.0
                    and exact_result["currents_A"]["Eb"] < 0.0
                )
                stage_result["exact_binary_certificate_attempt"] = {
                    "passed": bool(exact_result.get("passed")),
                    "opposite_current_switching_achieved": exact_switching,
                    "binary_mask": artifact(attempt_binary_path),
                    "result": exact_result,
                }
                _write_json(attempt_dir / "stage_result.json", stage_result)
                manifest["latest"]["exact_binary_certificate_passed"] = bool(
                    exact_result.get("passed")
                )
                manifest["latest"]["exact_binary_switching_achieved"] = exact_switching
                _write_json(manifest_path, manifest)
                if exact_switching:
                    final_full_chain_adfd = driver.audit_full_chain_latent_adfd(
                        latent_final,
                        final_point,
                        attempt_dir / "final_full_chain_adfd",
                        validation_scope="final-binary-continuous-precursor",
                    )
                    stage_result["final_full_chain_current_AD_FD"] = (
                        final_full_chain_adfd
                    )
                    final_precursor_fd_audit = (
                        driver.audit_final_binary_precursor_independent_fd(
                            latent_final,
                            attempt_dir / "final_binary_precursor_fd_audit",
                        )
                    )
                    stage_result["final_binary_precursor_independent_FD_audit"] = (
                        final_precursor_fd_audit
                    )
                    _write_json(attempt_dir / "stage_result.json", stage_result)
                    binary_path = output / "final_exact_binary_cell_mask.npz"
                    _save_final_binary_mask(binary_path, binary_mask)
                    manifest["status"] = (
                        "PASSED_LUMERICAL_4UM_DUALPOL_EXACT_BINARY_AU_"
                        "LATERAL_PDE_NUMERICAL_CERTIFICATE"
                    )
                    manifest["passed"] = True
                    manifest["final"] = {
                        "beta": beta,
                        "continuous_stage": stage_result,
                        "binary_mask": artifact(binary_path),
                        "exact_binary_evaluation": exact_result,
                        "final_full_chain_current_AD_FD": final_full_chain_adfd,
                        "final_binary_precursor_independent_FD_audit": (
                            final_precursor_fd_audit
                        ),
                        "ordinary_dispersive_Au": True,
                        "optical_xy100_to_xy50_converged_for_Ea_and_Eb": True,
                        "adaptive_custom_PDE_converged_for_Ea_and_Eb": True,
                        "same_PDE_grid_xy100_to_xy50_current_temperature_"
                        "converged_for_Ea_and_Eb": True,
                        "B200_promotion_certified": bool(
                            exact_result.get("B200_promotion_certified")
                        ),
                        "posthoc_morphology_repair": False,
                    }
                    state["beta_index"] = len(BETA_SCHEDULE)
                    state["attempt"] = 0
                    state["latent"] = latent_final
                    _write_json(manifest_path, manifest)
                    _save_checkpoint(checkpoint_path, **state)
                    print(json.dumps(manifest["final"], indent=2, default=str))
                    return 0

            may_advance = bool(
                not final_beta
                and design_satisfied
                and objective_converged
                and switching
            )
            if may_advance:
                state["latent"] = latent_final
                state["beta_index"] = beta_index + 1
                state["attempt"] = 0
                # The next stage computes a tighter cap while preserving the
                # monotone DFM and grayness limits already achieved.
                _save_checkpoint(checkpoint_path, **state)
                checkpoint_copy = (
                    output / "checkpoints" / f"beta_{beta:03g}_completed.npz"
                )
                _save_checkpoint(checkpoint_copy, **state)
                if args.stop_after_beta == beta:
                    manifest["status"] = "PAUSED_AFTER_REQUESTED_BETA"
                    _write_json(manifest_path, manifest)
                    return 0
                continue

            manifest["status"] = (
                "STOPPED_UNRESOLVED_FINAL_BINARY_OR_SWITCHING_GATES"
                if final_beta
                else "STOPPED_UNRESOLVED_STAGE_OBJECTIVE_OR_DESIGN_GATES"
            )
            manifest["passed"] = False
            manifest["blocking_beta"] = beta
            manifest["blocking_recovery_index"] = attempt
            manifest["blocking_reason"] = (
                "fixed-beta MMA ended without satisfying the audited plateau, "
                "current-sign, active-design, and promotion gates"
            )
            manifest["wall_s"] = time.monotonic() - started
            state["latent"] = latent_final
            state["attempt"] = attempt + 1
            _save_checkpoint(checkpoint_path, **state)
            _write_json(manifest_path, manifest)
            return 2

        raise RuntimeError("continuation exited without exact-binary promotion")
    except Exception as error:
        manifest.update(
            status="FAILED_LUMERICAL_4UM_DUALPOL_BETA_CONTINUATION",
            passed=False,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            wall_s=time.monotonic() - started,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        if output is None:
            output = _failure_output_root()
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            _write_json(output / "production_manifest.json", manifest)
        print(json.dumps(manifest, indent=2, default=str))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
