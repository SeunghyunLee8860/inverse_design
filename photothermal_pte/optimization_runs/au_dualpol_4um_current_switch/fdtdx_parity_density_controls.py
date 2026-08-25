#!/usr/bin/env python3
"""Empty and deterministic-gray optical controls on the fresh parity model."""

from __future__ import annotations

import argparse
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
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_design_mapping import (
    MAPPING,
    control_density,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_microbenchmark import (
    _git_output,
    _write_new_external_json,
    query_and_require_idle_gpu,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_optical_controls import (
    SCHEMA_CASE as FULL_CONTROL_SCHEMA,
    _report_hash,
    _validate_new_external_path,
    _validated_json,
    _write_npz,
    absorption_fields_and_powers,
    array_sha256,
    control_gate,
    file_sha256,
    load_source_calibration,
    normalized_flux_to_si_W,
    relative_mismatch,
    weighted_spatial_nrmse,
)


SCHEMA_CASE = "fdtdx_4um_parity_density_optical_control_v1"
SCHEMA_AGGREGATE = "fdtdx_4um_parity_density_optical_control_aggregate_v1"
ALLOWED_CASES = ("empty", "nonuniform_gray")
MAX_POINTWISE_AU_EPSILON_MISMATCH = 1.0e-5


def material_powers(absorption: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    q = absorption["q"]
    volumes = absorption["volumes"]
    return {
        basis: {
            window: {
                material: float(
                    np.sum(q[basis][window][material] * volumes[material])
                )
                for material in ("au", "tairte4")
            }
            for window in ("previous", "late")
        }
        for basis in ("target", "discrete_ADE")
    }


def pointwise_au_epsilon_mismatch(rho_cell: np.ndarray) -> float:
    target = np.asarray(target_au_epsilon(rho_cell), dtype=np.complex128)
    realized = np.asarray(realized_au_epsilon(rho_cell), dtype=np.complex128)
    return float(np.max(np.abs(realized - target) / np.maximum(np.abs(target), 1.0)))


def density_gate(
    case: str,
    *,
    cells: np.ndarray,
    metrics: dict[str, float | bool],
    by_material: dict[str, dict[str, dict[str, float]]],
) -> tuple[str, dict[str, bool]]:
    if case not in ALLOWED_CASES:
        raise ValueError(f"unsupported density optical control {case!r}")
    _, gates = control_gate(metrics)
    target_au = by_material["target"]["late"]["au"]
    discrete_au = by_material["discrete_ADE"]["late"]["au"]
    gates["pointwise_Au_epsilon"] = (
        float(metrics["pointwise_Au_epsilon_mismatch_relative"])
        < MAX_POINTWISE_AU_EPSILON_MISMATCH
    )
    if case == "empty":
        gates.update(
            exact_zero_cell_density=bool(np.all(cells == 0.0)),
            exact_zero_target_Au_Q=target_au == 0.0,
            exact_zero_discrete_Au_Q=discrete_au == 0.0,
            positive_TaIrTe4_Q=by_material["target"]["late"]["tairte4"] > 0.0,
        )
        passing = "PASS_EMPTY_OPTICAL_CONTROL"
    else:
        gates.update(
            strictly_gray_cell_density=bool(
                np.min(cells) > 0.0 and np.max(cells) < 1.0
            ),
            nonuniform_cell_density=float(np.ptp(cells)) > 0.25,
            positive_target_Au_Q=target_au > 0.0,
            positive_discrete_Au_Q=discrete_au > 0.0,
        )
        passing = "PASS_NONUNIFORM_GRAY_OPTICAL_CONTROL"
    return (passing if all(gates.values()) else "BLOCKED", gates)


def expected_case_status(case: str) -> str:
    if case == "empty":
        return "PASS_EMPTY_OPTICAL_CONTROL"
    if case == "nonuniform_gray":
        return "PASS_NONUNIFORM_GRAY_OPTICAL_CONTROL"
    raise ValueError(f"unsupported density optical control {case!r}")


def _run_case(args: argparse.Namespace) -> dict[str, Any]:
    output_json = _validate_new_external_path(args.output_json)
    output_npz = _validate_new_external_path(args.output_npz)
    if output_json == output_npz:
        raise RuntimeError("JSON and NPZ outputs must be different paths")
    source = load_source_calibration(
        args.source_calibration_json,
        expected_file_sha256=args.source_calibration_sha256,
    )
    density = control_density(args.density_case)
    latent = density["latent"]
    projected = density["projected_nodes"]
    cells = density["cells"]

    commit_before = _git_output(["rev-parse", "HEAD"])
    status_before = _git_output(["status", "--porcelain"])
    if status_before != "":
        raise RuntimeError("density optical control requires a clean worktree")
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

    print(
        f"phase=build polarization={args.polarization} density={args.density_case}",
        flush=True,
    )
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
    by_material = material_powers(absorption)
    target_previous = powers["target"]["previous"]
    target_late = powers["target"]["late"]
    discrete_late = powers["discrete_ADE"]["late"]

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
    closed_td_W = normalized_flux_to_si_W(float(np.mean(closed_td_raw_series)))
    closed_phasor_W = normalized_flux_to_si_W(closed_phasor_raw)
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
        "physical_incident_plane_net_flux_W": normalized_flux_to_si_W(
            incident_late_raw
        ),
        "previous_late_Q_power_mismatch_relative": relative_mismatch(
            target_previous, target_late
        ),
        "previous_late_Q_spatial_NRMSE": weighted_spatial_nrmse(
            absorption["q"]["target"]["late"],
            absorption["q"]["target"]["previous"],
            absorption["volumes"],
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
        "pointwise_Au_epsilon_mismatch_relative": pointwise_au_epsilon_mismatch(
            cells
        ),
    }
    status, gates = density_gate(
        args.density_case,
        cells=cells,
        metrics=metrics,
        by_material=by_material,
    )
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
    scaled_by_material = {
        basis: {
            window: {
                material: value * power_scale
                for material, value in materials.items()
            }
            for window, materials in windows.items()
        }
        for basis, windows in by_material.items()
    }
    raw.update(
        latent=latent,
        projected_nodes=projected,
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
        "scope": "fresh_exact_grid_density_dependent_optical_energy_control",
        "density_case": args.density_case,
        "density_control_validated": status == expected_case_status(args.density_case),
        "optimizer_enabled": False,
        "thermal_electrical_validated": False,
        "polarization": args.polarization,
        "metrics_unscaled": metrics,
        "material_powers_unscaled_W": by_material,
        "powers_scaled_to_285uW_incident_W": scaled,
        "material_powers_scaled_to_285uW_incident_W": scaled_by_material,
        "gates": gates,
        "pointwise_Au_epsilon_mismatch_limit": MAX_POINTWISE_AU_EPSILON_MISMATCH,
        "epsilon_imag": {
            "target": absorption["target_imag_epsilon"],
            "discrete_ADE": absorption["discrete_imag_epsilon"],
        },
        "mapping": MAPPING.audit(),
        "density_ranges": density["ranges"],
        "density_sha256": {
            "latent": array_sha256(latent, label="fdtdx-parity-latent-v1"),
            "projected_nodes": array_sha256(
                projected, label="fdtdx-parity-projected-nodes-v1"
            ),
            "cells": array_sha256(cells, label="fdtdx-parity-rho-cells-v1"),
        },
        "source_calibration": source,
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
        "full_control_schema_reference": FULL_CONTROL_SCHEMA,
        "raw_result_in_git": False,
    }
    report["report_sha256"] = _report_hash(report)
    _write_new_external_json(output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def aggregate_cases(ea: dict[str, Any], eb: dict[str, Any]) -> dict[str, Any]:
    cases = {str(case.get("polarization")): case for case in (ea, eb)}
    if set(cases) != {"Ea", "Eb"}:
        raise RuntimeError("expected exactly one Ea and one Eb density control")
    density_cases = {str(case.get("density_case")) for case in cases.values()}
    if len(density_cases) != 1:
        raise RuntimeError("Ea and Eb density controls do not use the same case")
    density_case = density_cases.pop()
    expected = expected_case_status(density_case)
    for polarization, case in cases.items():
        if case.get("schema") != SCHEMA_CASE:
            raise RuntimeError(f"{polarization} density-control schema mismatch")
        claimed = case.get("report_sha256")
        unhashed = dict(case)
        unhashed.pop("report_sha256", None)
        if claimed != _report_hash(unhashed):
            raise RuntimeError(f"{polarization} density-control report hash mismatch")
    invariant_keys = (
        "git_commit",
        "script_sha256",
        "density_sha256",
        "mapping",
        "source_calibration",
    )
    invariants_match = all(
        cases["Ea"].get(key) == cases["Eb"].get(key) for key in invariant_keys
    )
    cases_pass = all(case.get("status") == expected for case in cases.values())
    passing_status = f"PASS_{density_case.upper()}_EA_EB_OPTICAL_CONTROLS"
    status = passing_status if invariants_match and cases_pass else "BLOCKED"
    return {
        "schema": SCHEMA_AGGREGATE,
        "status": status,
        "scope": "fresh_exact_grid_Ea_Eb_density_dependent_optical_controls",
        "density_case": density_case,
        "both_polarizations_validated": status == passing_status,
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
        "mapping": cases["Ea"].get("mapping"),
        "density_sha256": cases["Ea"].get("density_sha256"),
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
    run.add_argument("--density-case", choices=ALLOWED_CASES, required=True)
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
