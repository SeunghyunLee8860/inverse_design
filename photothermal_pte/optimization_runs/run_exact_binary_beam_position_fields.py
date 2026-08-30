#!/usr/bin/env python3
"""Compute spatial fields at every fixed-flake beam-position scan point."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np

from photothermal_pte.optimization_runs.run_exact_binary_beam_response import (
    AU_INTERFACE_REFERENCE,
    AU_MATERIAL,
    AU_OPTICAL_REFERENCE,
    BEAM_RESPONSE_FDTD_THREADS,
    BEAM_RESPONSE_THERMAL_SOLVE_TOLERANCE,
    INPUT_SESSION_ATTEMPTS,
    OPTICAL_CLOSURE_GATE,
    configure_fixed_device,
    configure_source,
    json_write,
    open_fdtd_with_retry,
    run_maxwell,
    support_power,
    transient_session_error,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.beam_position_fields import (
    FIELD_SCHEMA,
    build_position_field_arrays,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.beam_response_contract import (
    AU_INDEX_AT_10UM,
    AU_TAIRTE4_INTERFACE_CONDUCTANCE_W_M2K,
    AU_THICKNESS_M,
    CASES,
    FLAKE_BOUNDS_M,
    TARGET_POWER_W,
    position_inputs,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.evaluate_objective_gradient import (
    load_rho,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.validate_combined_adfd import (
    CachedElectricalCuda,
    checked,
    polarization_angle,
    sha256,
    solve_coupled,
)


RESULT_SCHEMA = "exact-binary-fixed-flake-au-position-spatial-response-v1"
REFERENCE_CURRENT_RELATIVE_TOLERANCE = 1.0e-4
CURRENT_IDENTITY_RELATIVE_TOLERANCE = 1.0e-8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def scalar_position_rows(path: Path, run: int) -> dict[tuple[float, float], dict[str, object]]:
    payload = json.loads(path.read_text())
    if not payload.get("passed") or payload.get("run") != run:
        raise RuntimeError(f"invalid scalar beam-response reference: {path}")
    rows: dict[tuple[float, float], dict[str, object]] = {}
    for row in payload["responses"]:
        if row["kind"] == "position" or (
            row["kind"] == "waist"
            and np.isclose(row["target_waist_um"], 8.5)
        ):
            rows[(float(row["beam_x_um"]), float(row["beam_y_um"]))] = row
    if len(rows) != 25:
        raise RuntimeError(f"scalar reference does not contain 25 position points: {path}")
    return rows


def valid_completed_rows(checkpoint: Path) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    if not checkpoint.is_file():
        return [], None
    payload = json.loads(checkpoint.read_text())
    if payload.get("schema") != RESULT_SCHEMA:
        return [], None
    completed = []
    for row in payload.get("responses", []):
        raw = row.get("spatial_fields", {})
        path = Path(str(raw.get("path", "")))
        if (
            row.get("passed")
            and path.is_file()
            and raw.get("sha256") == sha256(path)
        ):
            completed.append(row)
    return completed, payload.get("geometry")


def evaluate_position(
    *,
    base_fsp: Path,
    rho: np.ndarray,
    case: object,
    item: dict[str, float | str],
    reference: dict[str, object],
    output: Path,
    gpu_device: str,
    cuda_device: int,
) -> tuple[dict[str, object], dict[str, object]]:
    fdtd = None
    try:
        fdtd, audit, runtime = open_fdtd_with_retry(gpu_device)
        fdtd.load(str(base_fsp))
        fdtd.switchtolayout()
        geometry = configure_fixed_device(fdtd, rho, case.contact_axis)
        source = configure_source(
            fdtd, item, polarization_angle(case.polarization)
        )
        forward = run_maxwell(
            fdtd, audit, runtime, output,
            f"run{case.run:03d}_{item['id']}_spatial",
        )
        coupled = solve_coupled(
            forward,
            rho,
            cuda_device,
            need_adjoint=False,
            thermal_state_kwargs={
                "au_contact_axis": case.contact_axis,
                "au_thickness_m": AU_THICKNESS_M,
                "au_tairte4_interface_conductance_W_m2K": (
                    AU_TAIRTE4_INTERFACE_CONDUCTANCE_W_M2K
                ),
            },
            thermal_relative_tolerance=BEAM_RESPONSE_THERMAL_SOLVE_TOLERANCE,
        )
        scale = TARGET_POWER_W / float(forward["source_power_W"])
        arrays, spatial = build_position_field_arrays(
            coupled,
            rho,
            scale,
            case.contact_axis,
            linear_solve=CachedElectricalCuda(cuda_device),
        )
        current_A = float(coupled["electrical"].current_A) * scale
        arrays.update(
            beam_x_um=np.asarray(float(item["x_um"])),
            beam_y_um=np.asarray(float(item["y_um"])),
            beam_waist_um=np.asarray(float(item["waist_um"])),
            incident_power_W=np.asarray(TARGET_POWER_W),
            terminal_current_A=np.asarray(current_A),
        )
        fields_path = output / "fields" / f"{item['id']}.npz"
        fields_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_savez(fields_path, arrays)

        reference_current = float(reference["terminal_current_A"])
        reference_error = abs(current_A - reference_current) / max(
            abs(reference_current), np.finfo(float).tiny
        )
        auto_shutoff = forward["log_audit"].get("final_auto_shutoff")
        gates = {
            "optical_closure_lt_2pct": bool(forward["closure"] < OPTICAL_CLOSURE_GATE),
            "Q_nonnegative_and_finite": bool(
                forward["Q_minimum_W_m3"] >= 0.0 and forward["Q_all_finite"]
            ),
            "auto_shutoff_lt_1e_5": bool(
                auto_shutoff is not None and auto_shutoff < 1.0e-5
            ),
            "Q_mapping_error_lt_0p5pct": bool(
                coupled["mapping"]["relative_mapping_error"] < 0.005
            ),
            "thermal_residual_lt_1e_8": bool(
                coupled["thermal_forward"].explicit_relative_residual < 1.0e-8
            ),
            "thermal_energy_error_lt_1pct": bool(coupled["energy"] < 0.01),
            "electrical_weighting_residual_lt_1e_8": bool(
                coupled["electrical"].weighting_residual < 1.0e-8
            ),
            "short_circuit_continuity_residual_lt_1e_8": bool(
                spatial["short_circuit_continuity_residual"] < 1.0e-8
            ),
            "pte_contribution_reintegrates_current": bool(
                spatial["pte_contribution_relative_error"]
                < CURRENT_IDENTITY_RELATIVE_TOLERANCE
            ),
            "short_circuit_flux_matches_current": bool(
                spatial["short_circuit_flux_relative_error"]
                < CURRENT_IDENTITY_RELATIVE_TOLERANCE
            ),
            "total_J_weighting_matches_current": bool(
                spatial["total_J_weighting_relative_error"]
                < CURRENT_IDENTITY_RELATIVE_TOLERANCE
            ),
            "matches_scalar_response_reference": bool(
                reference_error < REFERENCE_CURRENT_RELATIVE_TOLERANCE
            ),
            "all_spatial_fields_finite": bool(spatial["all_finite"]),
        }
        powers = {
            "Au_W": support_power(coupled, "Au_electrodes") * scale,
            "TaIrTe4_W": support_power(coupled, "flake_support") * scale,
            "SiO2_W": support_power(coupled, "SiO2") * scale,
            "Si_W": support_power(coupled, "Si") * scale,
        }
        row = {
            "id": item["id"],
            "beam_x_um": float(item["x_um"]),
            "beam_y_um": float(item["y_um"]),
            "target_waist_um": float(item["waist_um"]),
            "incident_power_W": TARGET_POWER_W,
            "terminal_current_A": current_A,
            "terminal_current_nA": current_A * 1.0e9,
            "reference_terminal_current_A": reference_current,
            "reference_current_relative_error": reference_error,
            "source": source,
            "linear_scale_from_raw_source": scale,
            "mapped_power_at_285uW": powers,
            "P_Q_at_285uW_W": float(forward["P_Q_W"]) * scale,
            "P_six_at_285uW_W": float(forward["P_six_W"]) * scale,
            "optical_closure": float(forward["closure"]),
            "auto_shutoff": auto_shutoff,
            "Q_mapping_error": float(coupled["mapping"]["relative_mapping_error"]),
            "thermal_residual": float(
                coupled["thermal_forward"].explicit_relative_residual
            ),
            "thermal_energy_error": float(coupled["energy"]),
            "electrical_weighting_residual": float(
                coupled["electrical"].weighting_residual
            ),
            "spatial_metrics": spatial,
            "spatial_fields": artifact(fields_path),
            "gpu_device": gpu_device,
            "cuda_device": cuda_device,
            "FDTD_threads": BEAM_RESPONSE_FDTD_THREADS,
            "Maxwell_wall_s": float(forward["wall_s"]),
            "gates": gates,
            "passed": all(gates.values()),
        }
        return row, geometry
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        working = output / "beam_response_working.fsp"
        if working.is_file():
            working.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, choices=sorted(CASES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scalar-reference-root", type=Path, required=True)
    parser.add_argument("--gpu-device", required=True)
    parser.add_argument("--cuda-device", type=int, required=True)
    parser.add_argument("--input-id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    case = CASES[args.run]
    if CONTRACT.geometry_mode != case.geometry_mode:
        raise RuntimeError(
            f"Run {case.run:03d} requires TAIRTE4_TOPOLOGY_GEOMETRY={case.geometry_mode}, "
            f"got {CONTRACT.geometry_mode}"
        )
    thermal = __import__(
        "photothermal_pte.optimization_runs.tairte4_flake_topology.thermal",
        fromlist=["TAIRTE4_SIO2_INTERFACE_SCENARIO"],
    )
    if case.interface_scenario != thermal.TAIRTE4_SIO2_INTERFACE_SCENARIO:
        raise RuntimeError("TaIrTe4/SiO2 interface scenario mismatch")

    base_fsp = checked(case.base_fsp, case.base_fsp_sha256)
    density_path = checked(case.density_path, case.density_sha256)
    rho = load_rho(density_path)
    if not np.array_equal(np.unique(rho), np.asarray((0.0, 1.0))):
        raise RuntimeError("position fields require an exact-binary density")
    scalar_path = (
        args.scalar_reference_root.expanduser().resolve()
        / f"run{case.run:03d}"
        / "beam_response_result.json"
    )
    references = scalar_position_rows(scalar_path, case.run)

    output = args.output_dir.expanduser().resolve()
    checkpoint = output / "position_fields_checkpoint.json"
    result_path = output / "position_fields_result.json"
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise RuntimeError(f"refusing non-empty output directory without --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    completed, geometry = valid_completed_rows(checkpoint) if args.resume else ([], None)
    completed_ids = {str(row["id"]) for row in completed}
    inputs = position_inputs()
    if args.input_id:
        inputs = [item for item in inputs if item["id"] == args.input_id]
        if not inputs:
            raise ValueError(f"unknown position input id: {args.input_id}")
    expected_ids = {str(item["id"]) for item in inputs}
    completed = [row for row in completed if str(row["id"]) in expected_ids]
    completed_ids = {str(row["id"]) for row in completed}

    started = time.monotonic()
    payload: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "field_schema": FIELD_SCHEMA,
        "status": "RUNNING",
        "run": case.run,
        "responses": completed,
    }
    try:
        for index, item in enumerate(inputs, start=1):
            if str(item["id"]) in completed_ids:
                continue
            errors: list[str] = []
            for attempt in range(1, INPUT_SESSION_ATTEMPTS + 1):
                try:
                    row, geometry = evaluate_position(
                        base_fsp=base_fsp,
                        rho=rho,
                        case=case,
                        item=item,
                        reference=references[(float(item["x_um"]), float(item["y_um"]))],
                        output=output,
                        gpu_device=args.gpu_device,
                        cuda_device=args.cuda_device,
                    )
                    break
                except Exception as error:
                    if not transient_session_error(error):
                        raise
                    errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
                    if attempt == INPUT_SESSION_ATTEMPTS:
                        raise RuntimeError("position input exhausted retries: " + " | ".join(errors)) from error
                    time.sleep(30.0 * attempt)
            else:
                raise AssertionError("unreachable")
            completed.append(row)
            completed_ids.add(str(row["id"]))
            payload.update(
                updated_at_utc=utc_now(),
                geometry=geometry,
                responses=completed,
                progress={"completed": len(completed), "total": len(inputs)},
            )
            json_write(checkpoint, payload)
            print(json.dumps({
                "run": case.run,
                "progress": f"{index}/{len(inputs)}",
                "id": item["id"],
                "current_nA": row["terminal_current_nA"],
                "passed": row["passed"],
            }), flush=True)
            if not row["passed"]:
                raise RuntimeError(f"Run {case.run:03d} {item['id']} failed a field gate")

        complete_grid = len(completed) == len(inputs) and completed_ids == expected_ids
        passed = bool(complete_grid and all(row["passed"] for row in completed))
        payload = {
            "schema": RESULT_SCHEMA,
            "field_schema": FIELD_SCHEMA,
            "status": "COMPLETED" if passed else "FAILED_INCOMPLETE_GRID",
            "passed": passed,
            "generated_at_utc": utc_now(),
            "run": case.run,
            "contact_axis": case.contact_axis,
            "geometry_mode": case.geometry_mode,
            "interface_scenario": case.interface_scenario,
            "polarization": case.polarization,
            "scope": "25-position spatial fields on one fixed exact-binary structure; no optimization",
            "axis_contract": "Lumerical x=b, y=a, z=c",
            "target_incident_power_W": TARGET_POWER_W,
            "geometry": geometry,
            "Au_contract": {
                "material": AU_MATERIAL,
                "thickness_m": AU_THICKNESS_M,
                "complex_index_at_10um": {
                    "real": AU_INDEX_AT_10UM.real,
                    "imag": AU_INDEX_AT_10UM.imag,
                },
                "thermal_conductivity_W_mK": 317.0,
                "Au_TaIrTe4_interface_conductance_W_m2K": (
                    AU_TAIRTE4_INTERFACE_CONDUCTANCE_W_M2K
                ),
                "optical_reference": AU_OPTICAL_REFERENCE,
                "interface_reference": AU_INTERFACE_REFERENCE,
            },
            "inputs": {
                "base_FSP": artifact(base_fsp),
                "exact_binary_density": artifact(density_path),
                "scalar_response_reference": artifact(scalar_path),
            },
            "responses": completed,
            "spatial_field_contents": [
                "temperature and temperature gradient",
                "short-circuit potential and electric field",
                "thermoelectric, conductive, and total local J",
                "signed terminal-current contribution and axis components",
                "weighting potential and gradient",
                "total and material-resolved absorbed-power density",
            ],
            "optimization_rerun": False,
            "flake_expanded_for_scan": False,
            "wall_s": time.monotonic() - started,
        }
    except Exception as error:
        payload.update(
            status="FAILED",
            passed=False,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            responses=completed,
            wall_s=time.monotonic() - started,
        )
    finally:
        json_write(result_path, payload)
        if checkpoint.is_file() and payload.get("passed"):
            checkpoint.unlink()
    print(json.dumps({
        "run": case.run,
        "status": payload.get("status"),
        "passed": payload.get("passed"),
        "responses": len(completed),
        "output": str(output),
    }, indent=2), flush=True)
    return 0 if payload.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
