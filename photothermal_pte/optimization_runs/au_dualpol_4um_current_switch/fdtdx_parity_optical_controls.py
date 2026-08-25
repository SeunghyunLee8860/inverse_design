#!/usr/bin/env python3
"""Fail-closed full-Au optical controls for the fresh FDTDX parity route.

This module deliberately does not import the historical FDTDX model,
``material_fraction``, ``combined_4um``, or an optimizer checkpoint.  It checks
one simple density state first: constant 81x81 nodal rho=1, mapped by the exact
four-node average to constant 80x80 Au cells.

The physical forward is accepted only when late/previous fields are stable,
the target n-k-square and realized discrete-ADE absorption agree, and volume
integrated absorption agrees with inward power through a closed surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_ade import (
    realized_epsilon as realized_au_epsilon,
    target_epsilon as target_au_epsilon,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_fixed_materials import (
    TA_A,
    TA_B,
    realized_epsilon as realized_fixed_epsilon,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_microbenchmark import (
    REPOSITORY,
    _git_output,
    _write_new_external_json,
    query_and_require_idle_gpu,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_model import (
    model_plan,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_source_calibration import (
    ETA0_OHM,
    SCHEMA_AGGREGATE as SOURCE_SCHEMA_AGGREGATE,
    SCHEMA_CASE as SOURCE_SCHEMA_CASE,
    TARGET_POWER_W,
    _report_hash,
    normalized_flux_to_si_W,
)


SCHEMA_CASE = "fdtdx_4um_parity_full_au_optical_control_v1"
SCHEMA_AGGREGATE = "fdtdx_4um_parity_full_au_optical_control_aggregate_v1"
EPS0_F_PER_M = 8.854_187_8128e-12

# These limits are frozen before the physical-device runs.  The 0.5% temporal
# power gate matches the accepted source stationarity gate.  The 2% spatial and
# energy-closure gates allow detector/window discretization but are still much
# tighter than a useful-device signal comparison.  The target/discrete limit is
# tied to the independently certified 1e-5 complex-epsilon carrier tolerance.
MAX_PREVIOUS_LATE_POWER_MISMATCH = 5.0e-3
MAX_PREVIOUS_LATE_SPATIAL_NRMSE = 2.0e-2
MAX_TARGET_DISCRETE_Q_MISMATCH = 1.0e-5
MAX_TD_PHASOR_FLUX_MISMATCH = 2.0e-2
MAX_Q_FLUX_MISMATCH = 2.0e-2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: Any, *, label: str) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def relative_mismatch(first: float, second: float) -> float:
    denominator = max(abs(float(first)), abs(float(second)))
    if denominator == 0.0:
        return math.inf
    return abs(float(first) - float(second)) / denominator


def full_au_density() -> tuple[np.ndarray, np.ndarray]:
    """Return the only currently authorized control density."""

    nodes = np.ones((81, 81), dtype=np.float64)
    cells = 0.25 * (
        nodes[:-1, :-1]
        + nodes[1:, :-1]
        + nodes[:-1, 1:]
        + nodes[1:, 1:]
    )
    if cells.shape != (80, 80) or not np.array_equal(cells, np.ones((80, 80))):
        raise RuntimeError("constant nodal-to-cell full-Au control changed")
    return nodes, cells


def electric_yee_dual_volumes(
    edges: tuple[Any, Any, Any],
    grid_slice: tuple[slice, slice, slice],
) -> np.ndarray:
    """Return Ex/Ey/Ez dual volumes for FDTDX's Taflove Yee convention."""

    widths = [np.diff(np.asarray(axis, dtype=np.float64)) for axis in edges]
    if any(np.any(axis <= 0.0) for axis in widths):
        raise ValueError("grid edges must be strictly increasing")
    dual = [
        0.5 * (np.concatenate((axis[:1], axis[:-1])) + axis)
        for axis in widths
    ]
    result = []
    for component in range(3):
        metrics = [widths[axis] if axis == component else dual[axis] for axis in range(3)]
        local = [metrics[axis][grid_slice[axis]] for axis in range(3)]
        result.append(
            local[0][:, None, None]
            * local[1][None, :, None]
            * local[2][None, None, :]
        )
    return np.stack(result, axis=0)


def weighted_spatial_nrmse(
    late: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
    volumes: dict[str, np.ndarray],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for material in ("au", "tairte4"):
        difference = late[material] - previous[material]
        numerator += float(np.sum(difference**2 * volumes[material]))
        denominator += float(np.sum(late[material] ** 2 * volumes[material]))
    return math.sqrt(numerator) / max(math.sqrt(denominator), np.finfo(float).tiny)


def _validated_json(path: Path, *, expected_file_sha256: str | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = file_sha256(resolved)
    if expected_file_sha256 is not None and observed != expected_file_sha256:
        raise RuntimeError(
            f"file hash mismatch for {resolved}: expected {expected_file_sha256}, got {observed}"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    claimed = payload.get("report_sha256")
    unhashed = dict(payload)
    unhashed.pop("report_sha256", None)
    if claimed != _report_hash(unhashed):
        raise RuntimeError(f"report hash mismatch for {resolved}")
    return payload


def load_source_calibration(
    aggregate_path: Path,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    """Validate the aggregate and both raw all-air reports through their hashes."""

    aggregate_resolved = aggregate_path.expanduser().resolve()
    aggregate = _validated_json(
        aggregate_resolved, expected_file_sha256=expected_file_sha256
    )
    if aggregate.get("schema") != SOURCE_SCHEMA_AGGREGATE:
        raise RuntimeError("source aggregate schema mismatch")
    if aggregate.get("status") != "PASS_SOURCE_CALIBRATION":
        raise RuntimeError("source calibration is not PASS_SOURCE_CALIBRATION")
    if float(aggregate.get("target_incident_power_W", math.nan)) != TARGET_POWER_W:
        raise RuntimeError("source target power changed")

    cases: dict[str, dict[str, Any]] = {}
    for polarization in ("Ea", "Eb"):
        artifact = aggregate.get("input_files", {}).get(polarization)
        if not isinstance(artifact, dict):
            raise RuntimeError(f"source aggregate lacks {polarization} raw artifact")
        case = _validated_json(
            Path(str(artifact.get("path", ""))),
            expected_file_sha256=str(artifact.get("file_sha256", "")),
        )
        if case.get("schema") != SOURCE_SCHEMA_CASE:
            raise RuntimeError(f"{polarization} source case schema mismatch")
        if case.get("status") != "PASS_SOURCE_CASE":
            raise RuntimeError(f"{polarization} source case is not passing")
        if case.get("polarization") != polarization:
            raise RuntimeError(f"{polarization} source case label mismatch")
        if case.get("git_status_porcelain") != "":
            raise RuntimeError(f"{polarization} source case used a dirty worktree")
        if case.get("report_sha256") != aggregate["case_report_sha256"][polarization]:
            raise RuntimeError(f"{polarization} source report binding mismatch")
        if case.get("model_plan") != model_plan(polarization, air_only=True):
            raise RuntimeError(f"{polarization} source model no longer matches parity plan")
        case_power = float(case["metrics"]["incident_power_late_W"])
        aggregate_power = float(aggregate["incident_power_W"][polarization])
        if case_power != aggregate_power:
            raise RuntimeError(f"{polarization} source power binding mismatch")
        cases[polarization] = case

    scales = {
        polarization: float(aggregate["power_or_Q_scale_to_target"][polarization])
        for polarization in ("Ea", "Eb")
    }
    for polarization, scale in scales.items():
        expected = TARGET_POWER_W / float(cases[polarization]["metrics"]["incident_power_late_W"])
        if not math.isfinite(scale) or scale <= 0.0 or scale != expected:
            raise RuntimeError(f"invalid {polarization} source scale")
    return {
        "path": str(aggregate_resolved),
        "file_sha256": expected_file_sha256,
        "report_sha256": aggregate["report_sha256"],
        "git_commit": aggregate["git_commit"],
        "case_report_sha256": aggregate["case_report_sha256"],
        "unscaled_incident_power_W": aggregate["incident_power_W"],
        "power_or_Q_scale_to_target": scales,
        "target_incident_power_W": TARGET_POWER_W,
    }


def absorption_fields_and_powers(
    *,
    model: dict[str, Any],
    result: Any,
    rho_cell: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute target and discrete-ADE Q from material E phasors."""

    edges = tuple(np.asarray(model["grid"].edges(axis)) for axis in range(3))
    volumes = {
        "au": electric_yee_dual_volumes(edges, model["slices"]["au_design"]),
        "tairte4": electric_yee_dual_volumes(
            edges, model["slices"]["fixed_tairte4"]
        ),
    }
    fields = {
        material: {
            window: np.asarray(
                result.detector_states[f"{prefix}_{window}"]["phasor"][0, 0]
            )
            for window in ("previous", "late")
        }
        for material, prefix in (("au", "au"), ("tairte4", "tairte4"))
    }
    target_imag = {
        "au": np.asarray(target_au_epsilon(rho_cell).imag, dtype=np.float64)[
            None, :, :, None
        ],
        "tairte4": np.asarray(
            [TA_B.target_epsilon_imag, TA_A.target_epsilon_imag, TA_B.target_epsilon_imag],
            dtype=np.float64,
        )[:, None, None, None],
    }
    discrete_imag = {
        "au": np.asarray(realized_au_epsilon(rho_cell).imag, dtype=np.float64)[
            None, :, :, None
        ],
        "tairte4": np.asarray(
            [
                realized_fixed_epsilon(TA_B).imag,
                realized_fixed_epsilon(TA_A).imag,
                realized_fixed_epsilon(TA_B).imag,
            ],
            dtype=np.float64,
        )[:, None, None, None],
    }
    prefactor = 0.5 * float(model["omega_rad_s"]) * EPS0_F_PER_M * ETA0_OHM**2
    q: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    powers: dict[str, dict[str, float]] = {}
    for basis, imag in (("target", target_imag), ("discrete_ADE", discrete_imag)):
        q[basis] = {}
        powers[basis] = {}
        for window in ("previous", "late"):
            q[basis][window] = {
                material: prefactor * imag[material] * np.abs(fields[material][window]) ** 2
                for material in ("au", "tairte4")
            }
            powers[basis][window] = float(
                sum(
                    np.sum(q[basis][window][material] * volumes[material])
                    for material in ("au", "tairte4")
                )
            )
    payload = {
        "fields": fields,
        "volumes": volumes,
        "q": q,
        "powers_W": powers,
        "target_imag_epsilon": {
            "Au_min_max": [float(np.min(target_imag["au"])), float(np.max(target_imag["au"]))],
            "TaIrTe4_xyz_bac": target_imag["tairte4"].reshape(3).tolist(),
        },
        "discrete_imag_epsilon": {
            "Au_min_max": [float(np.min(discrete_imag["au"])), float(np.max(discrete_imag["au"]))],
            "TaIrTe4_xyz_bac": discrete_imag["tairte4"].reshape(3).tolist(),
        },
    }
    raw = {
        "rho_cell": np.asarray(rho_cell),
        "E_au_previous": fields["au"]["previous"],
        "E_au_late": fields["au"]["late"],
        "E_tairte4_previous": fields["tairte4"]["previous"],
        "E_tairte4_late": fields["tairte4"]["late"],
        "dual_volume_au": volumes["au"],
        "dual_volume_tairte4": volumes["tairte4"],
    }
    return payload, raw


def control_gate(metrics: dict[str, float | bool]) -> tuple[str, dict[str, bool]]:
    gates = {
        "finite": bool(metrics["finite"]),
        "positive_target_Q": float(metrics["target_Q_late_W"]) > 0.0,
        "positive_discrete_Q": float(metrics["discrete_ADE_Q_late_W"]) > 0.0,
        "positive_closed_td_flux": float(metrics["closed_td_flux_W"]) > 0.0,
        "positive_closed_phasor_flux": float(metrics["closed_phasor_flux_W"]) > 0.0,
        "previous_late_power": float(metrics["previous_late_Q_power_mismatch_relative"])
        < MAX_PREVIOUS_LATE_POWER_MISMATCH,
        "previous_late_spatial": float(metrics["previous_late_Q_spatial_NRMSE"])
        < MAX_PREVIOUS_LATE_SPATIAL_NRMSE,
        "target_discrete_Q": float(metrics["target_discrete_Q_mismatch_relative"])
        < MAX_TARGET_DISCRETE_Q_MISMATCH,
        "td_phasor_flux": float(metrics["td_phasor_flux_mismatch_relative"])
        < MAX_TD_PHASOR_FLUX_MISMATCH,
        "discrete_Q_td_flux": float(metrics["discrete_Q_td_flux_mismatch_relative"])
        < MAX_Q_FLUX_MISMATCH,
        "discrete_Q_phasor_flux": float(metrics["discrete_Q_phasor_flux_mismatch_relative"])
        < MAX_Q_FLUX_MISMATCH,
    }
    return (
        "PASS_FULL_AU_OPTICAL_CONTROL" if all(gates.values()) else "BLOCKED",
        gates,
    )


def _validate_new_external_path(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output == REPOSITORY or REPOSITORY in output.parents:
        raise RuntimeError("raw physical-control artifact must remain outside Git")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    return output


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _run_case(args: argparse.Namespace) -> dict[str, Any]:
    output_json = _validate_new_external_path(args.output_json)
    output_npz = _validate_new_external_path(args.output_npz)
    if output_json == output_npz:
        raise RuntimeError("JSON and NPZ outputs must be different paths")
    source = load_source_calibration(
        args.source_calibration_json,
        expected_file_sha256=args.source_calibration_sha256,
    )
    commit_before = _git_output(["rev-parse", "HEAD"])
    status_before = _git_output(["status", "--porcelain"])
    if status_before != "":
        raise RuntimeError("physical optical control requires a clean worktree")

    gpu_snapshot = query_and_require_idle_gpu(args.gpu_uuid)
    existing_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing_visibility not in {None, "", args.gpu_uuid}:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES conflicts with requested UUID: {existing_visibility!r}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # GPU-sensitive imports follow UUID isolation.
    import jax

    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_model import (
        arrays_for_density,
        build_model,
        setup_audit,
    )

    nodes, cells = full_au_density()
    print(f"phase=build polarization={args.polarization} density=full_au", flush=True)
    build_started = time.perf_counter()
    model = build_model(args.polarization, backend="gpu", air_only=False)
    setup = setup_audit(model)
    arrays = arrays_for_density(model, cells)
    jax.block_until_ready(arrays.fields.E)
    build_seconds = time.perf_counter() - build_started
    if setup["status"] != "PASS":
        raise RuntimeError(f"physical setup audit failed: {setup}")

    def forward(container):
        return model["fdtdx"].run_fdtd(
            arrays=container,
            objects=model["placed"],
            config=model["config"],
            key=model["key"],
            show_progress=False,
        )[1]

    print(f"phase=compile polarization={args.polarization}", flush=True)
    compile_started = time.perf_counter()
    executable = jax.jit(forward).lower(arrays).compile()
    compile_seconds = time.perf_counter() - compile_started
    print(f"phase=full_forward polarization={args.polarization}", flush=True)
    forward_started = time.perf_counter()
    result = executable(arrays)
    jax.block_until_ready(result.detector_states["tairte4_late"]["phasor"])
    forward_seconds = time.perf_counter() - forward_started

    absorption, raw = absorption_fields_and_powers(
        model=model, result=result, rho_cell=cells
    )
    powers = absorption["powers_W"]
    target_previous = powers["target"]["previous"]
    target_late = powers["target"]["late"]
    discrete_late = powers["discrete_ADE"]["late"]
    target_q = absorption["q"]["target"]
    volumes = absorption["volumes"]

    closed_phasor_raw = float(
        np.asarray(
            model["placed"]["material_flux"].compute_net_flux(
                result.detector_states["material_flux"]
            )
        )[0]
    )
    closed_td_raw_series = np.asarray(
        result.detector_states["material_flux_td"]["poynting_flux"],
        dtype=np.float64,
    ).reshape(-1)
    closed_td_raw = float(np.mean(closed_td_raw_series))
    closed_phasor_W = normalized_flux_to_si_W(closed_phasor_raw)
    closed_td_W = normalized_flux_to_si_W(closed_td_raw)

    incident_late_raw = float(
        np.asarray(
            model["placed"]["incident_plane"].compute_poynting_flux(
                result.detector_states["incident_plane"]
            )
        )[0]
    )
    metrics: dict[str, float | bool] = {
        "finite": bool(
            all(np.all(np.isfinite(value)) for value in raw.values())
            and np.all(np.isfinite(closed_td_raw_series))
            and all(
                math.isfinite(value)
                for value in (
                    target_previous,
                    target_late,
                    discrete_late,
                    closed_td_W,
                    closed_phasor_W,
                    incident_late_raw,
                )
            )
        ),
        "target_Q_previous_W": target_previous,
        "target_Q_late_W": target_late,
        "discrete_ADE_Q_late_W": discrete_late,
        "closed_td_flux_W": closed_td_W,
        "closed_phasor_flux_W": closed_phasor_W,
        "physical_incident_plane_net_flux_W": normalized_flux_to_si_W(incident_late_raw),
        "previous_late_Q_power_mismatch_relative": relative_mismatch(
            target_previous, target_late
        ),
        "previous_late_Q_spatial_NRMSE": weighted_spatial_nrmse(
            target_q["late"], target_q["previous"], volumes
        ),
        "target_discrete_Q_mismatch_relative": relative_mismatch(
            target_late, discrete_late
        ),
        "td_phasor_flux_mismatch_relative": relative_mismatch(
            closed_td_W, closed_phasor_W
        ),
        "discrete_Q_td_flux_mismatch_relative": relative_mismatch(
            discrete_late, closed_td_W
        ),
        "discrete_Q_phasor_flux_mismatch_relative": relative_mismatch(
            discrete_late, closed_phasor_W
        ),
    }
    status, gates = control_gate(metrics)
    power_scale = float(source["power_or_Q_scale_to_target"][args.polarization])
    scaled = {
        key: float(metrics[key]) * power_scale
        for key in (
            "target_Q_previous_W",
            "target_Q_late_W",
            "discrete_ADE_Q_late_W",
            "closed_td_flux_W",
            "closed_phasor_flux_W",
            "physical_incident_plane_net_flux_W",
        )
    }
    raw.update(
        rho_nodes=nodes,
        closed_td_normalized_flux_series=closed_td_raw_series,
    )
    _write_npz(output_npz, raw)

    commit_after = _git_output(["rev-parse", "HEAD"])
    status_after = _git_output(["status", "--porcelain"])
    if commit_after != commit_before or status_after != "":
        status = "BLOCKED"
        gates["stable_clean_git_state"] = False
    else:
        gates["stable_clean_git_state"] = True
    report: dict[str, Any] = {
        "schema": SCHEMA_CASE,
        "status": status,
        "scope": "fresh_exact_grid_full_au_optical_energy_control",
        "density_case": "81x81_constant_one_nodes_to_80x80_constant_one_cells",
        "full_au_optical_control_validated": status == "PASS_FULL_AU_OPTICAL_CONTROL",
        "optimizer_enabled": False,
        "thermal_electrical_validated": False,
        "polarization": args.polarization,
        "metrics_unscaled": metrics,
        "powers_scaled_to_285uW_incident_W": scaled,
        "gates": gates,
        "limits": {
            "previous_late_power_mismatch": MAX_PREVIOUS_LATE_POWER_MISMATCH,
            "previous_late_spatial_NRMSE": MAX_PREVIOUS_LATE_SPATIAL_NRMSE,
            "target_discrete_Q_mismatch": MAX_TARGET_DISCRETE_Q_MISMATCH,
            "td_phasor_flux_mismatch": MAX_TD_PHASOR_FLUX_MISMATCH,
            "Q_flux_mismatch": MAX_Q_FLUX_MISMATCH,
        },
        "epsilon_imag": {
            "target": absorption["target_imag_epsilon"],
            "discrete_ADE": absorption["discrete_imag_epsilon"],
        },
        "source_calibration": source,
        "density_sha256": {
            "rho_nodes": array_sha256(nodes, label="fdtdx-parity-rho-nodes-v1"),
            "rho_cells": array_sha256(cells, label="fdtdx-parity-rho-cells-v1"),
        },
        "raw_artifact": {
            "path": str(output_npz),
            "file_sha256": file_sha256(output_npz),
            "keys": sorted(raw),
            "tracked_by_git": False,
        },
        "full_forward_executed": True,
        "field_steps_executed": int(model["config"].time_steps_total),
        "build_seconds": build_seconds,
        "compile_seconds": compile_seconds,
        "forward_seconds": forward_seconds,
        "setup_audit": setup,
        "model_plan": model["plan"],
        "gpu_preflight": gpu_snapshot,
        "jax_devices": [str(device) for device in jax.devices()],
        "cublas_runtime_version": model["cublas_runtime_version"],
        "git_commit": commit_before,
        "git_status_porcelain_before": status_before,
        "git_commit_after": commit_after,
        "git_status_porcelain_after": status_after,
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "raw_result_in_git": False,
    }
    report["report_sha256"] = _report_hash(report)
    _write_new_external_json(output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def aggregate_cases(ea: dict[str, Any], eb: dict[str, Any]) -> dict[str, Any]:
    cases = {str(case.get("polarization")): case for case in (ea, eb)}
    if set(cases) != {"Ea", "Eb"}:
        raise RuntimeError("expected exactly one Ea and one Eb physical control")
    for polarization, case in cases.items():
        if case.get("schema") != SCHEMA_CASE:
            raise RuntimeError(f"{polarization} physical-control schema mismatch")
        claimed = case.get("report_sha256")
        unhashed = dict(case)
        unhashed.pop("report_sha256", None)
        if claimed != _report_hash(unhashed):
            raise RuntimeError(f"{polarization} physical-control report hash mismatch")
    invariant_keys = (
        "git_commit",
        "script_sha256",
        "density_sha256",
        "source_calibration",
    )
    invariants_match = all(
        cases["Ea"].get(key) == cases["Eb"].get(key) for key in invariant_keys
    )
    cases_pass = all(
        case.get("status") == "PASS_FULL_AU_OPTICAL_CONTROL"
        for case in cases.values()
    )
    status = (
        "PASS_FULL_AU_EA_EB_OPTICAL_CONTROLS"
        if invariants_match and cases_pass
        else "BLOCKED"
    )
    return {
        "schema": SCHEMA_AGGREGATE,
        "status": status,
        "scope": "fresh_exact_grid_full_au_Ea_Eb_optical_energy_controls",
        "both_polarizations_validated": status
        == "PASS_FULL_AU_EA_EB_OPTICAL_CONTROLS",
        "optimizer_enabled": False,
        "invariants_match": invariants_match,
        "case_gates_pass": cases_pass,
        "case_report_sha256": {
            polarization: cases[polarization]["report_sha256"]
            for polarization in ("Ea", "Eb")
        },
        "unscaled_metrics": {
            polarization: cases[polarization]["metrics_unscaled"]
            for polarization in ("Ea", "Eb")
        },
        "scaled_powers_W": {
            polarization: cases[polarization]["powers_scaled_to_285uW_incident_W"]
            for polarization in ("Ea", "Eb")
        },
        "raw_artifacts": {
            polarization: cases[polarization]["raw_artifact"]
            for polarization in ("Ea", "Eb")
        },
        "git_commit": cases["Ea"].get("git_commit"),
        "script_sha256": cases["Ea"].get("script_sha256"),
        "raw_result_in_git": False,
    }


def _aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output = _validate_new_external_path(args.output_json)
    paths = {
        "Ea": args.ea_json.expanduser().resolve(),
        "Eb": args.eb_json.expanduser().resolve(),
    }
    cases = {
        polarization: _validated_json(path)
        for polarization, path in paths.items()
    }
    report = aggregate_cases(cases["Ea"], cases["Eb"])
    report["input_files"] = {
        polarization: {
            "path": str(path),
            "file_sha256": file_sha256(path),
        }
        for polarization, path in paths.items()
    }
    report["report_sha256"] = _report_hash(report)
    _write_new_external_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    run.add_argument("--gpu-uuid", required=True)
    run.add_argument("--source-calibration-json", type=Path, required=True)
    run.add_argument("--source-calibration-sha256", required=True)
    run.add_argument("--output-json", type=Path, required=True)
    run.add_argument("--output-npz", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--ea-json", type=Path, required=True)
    aggregate.add_argument("--eb-json", type=Path, required=True)
    aggregate.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = _run_case(args) if args.command == "run" else _aggregate(args)
    return 0 if str(report["status"]).startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
