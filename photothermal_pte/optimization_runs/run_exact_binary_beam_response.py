#!/usr/bin/env python3
"""Run fixed-geometry beam waist/position responses with explicit Au contacts."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time
import traceback

import numpy as np

from photothermal_pte.finite_inverse_design.native_yee_q import extract_native_yee_q
from photothermal_pte.optimization_runs.tairte4_flake_topology import optical
from photothermal_pte.optimization_runs.tairte4_flake_topology.beam_response_contract import (
    AU_INDEX_AT_10UM,
    AU_INTERFACE_REFERENCE,
    AU_OPTICAL_REFERENCE,
    AU_TAIRTE4_INTERFACE_CONDUCTANCE_W_M2K,
    AU_THICKNESS_M,
    CASES,
    FLAKE_BOUNDS_M,
    OPTICAL_DOMAIN_SPAN_M,
    RESPONSE_CONTROL_VOLUME_HALF_SPAN_M,
    SOURCE_OBJECT_WAIST_SCALE,
    TARGET_POWER_W,
    domain_center_m,
    electrode_bounds_m,
    source_bounds_m,
    sweep_inputs,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.evaluate_objective_gradient import (
    load_rho,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.validate_combined_adfd import (
    FIELD_REGION,
    FREQUENCY_HZ,
    checked,
    lumerical_gpu_engine_lock,
    open_fdtd,
    polarization_angle,
    set_density,
    sha256,
    solve_coupled,
)


AU_MATERIAL = "beam_response_Au_Ordal_10um"
AU_OBJECT_NAMES = ("beam_response_Au_low_terminal", "beam_response_Au_high_terminal")
SUBSTRATE_OBJECTS = ("run010_Si_substrate", "run010_bottom_SiO2")
OUTER_MESH = "run010_outer_coarse_xy_mesh"
FLAKE_MESH = "run010_flake_xy_z_mesh"
STACK_MESH = "run010_illuminated_stack_xy_mesh"
MAXWELL_RUN_ATTEMPTS = 3
MAXWELL_RETRY_WAIT_S = 20.0
RESULT_SCHEMA = "exact-binary-fixed-flake-au-beam-response-v6"
SESSION_OPEN_LOCK = Path("/tmp/seunghyun_exact_binary_beam_response_session_open.lock")
SESSION_OPEN_ATTEMPTS = 3
BEAM_RESPONSE_FDTD_THREADS = 3
OPTICAL_CLOSURE_GATE = 0.02
INPUT_SESSION_ATTEMPTS = 5
BEAM_RESPONSE_THERMAL_SOLVE_TOLERANCE = 1e-9


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def serialized_session_open():
    SESSION_OPEN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with SESSION_OPEN_LOCK.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def open_fdtd_with_retry(gpu_device: str):
    errors: list[str] = []
    for attempt in range(1, SESSION_OPEN_ATTEMPTS + 1):
        try:
            with serialized_session_open():
                session = open_fdtd(
                    gpu_device, fdtd_threads=BEAM_RESPONSE_FDTD_THREADS
                )
                # GPU field updates are unchanged; fewer host threads reduce
                # FlexNet HPC task demand from nine to four per solve.
                time.sleep(2.0)
            return session
        except Exception as error:
            errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            if attempt == SESSION_OPEN_ATTEMPTS:
                raise RuntimeError(
                    "Lumerical session open exhausted retries: " + " | ".join(errors)
                ) from error
            time.sleep(15.0 * attempt)
    raise AssertionError("unreachable")


def json_write(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


def load_valid_completed_result(path: Path, run: int) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    rows = payload.get("responses", [])
    expected_ids = {str(item["id"]) for item in sweep_inputs(smoke=False)}
    actual_ids = {str(row.get("id")) for row in rows}
    valid = (
        payload.get("schema") == RESULT_SCHEMA
        and payload.get("run") == run
        and payload.get("status") == "COMPLETED"
        and payload.get("passed") is True
        and len(rows) == len(expected_ids)
        and actual_ids == expected_ids
        and all(row.get("passed") is True for row in rows)
        and payload.get("flake_expanded_for_scan") is False
    )
    return payload if valid else None


def normalize_response_gates(row: dict[str, object]) -> dict[str, object]:
    normalized = dict(row)
    gates = dict(normalized.get("gates", {}))
    gates.pop("optical_closure_lt_0p5pct", None)
    gates.pop("optical_closure_lt_1pct", None)
    gates["optical_closure_lt_2pct"] = bool(
        float(normalized["optical_closure"]) < OPTICAL_CLOSURE_GATE
    )
    normalized["gates"] = gates
    normalized["passed"] = bool(gates and all(gates.values()))
    return normalized


def named_bounds(fdtd: object, name: str) -> dict[str, list[float]]:
    return {
        axis: [
            float(fdtd.getnamed(name, f"{axis} min")),
            float(fdtd.getnamed(name, f"{axis} max")),
        ]
        for axis in "xyz"
    }


def configure_response_control_volume(fdtd: object) -> dict[str, object]:
    lateral = (
        -RESPONSE_CONTROL_VOLUME_HALF_SPAN_M,
        RESPONSE_CONTROL_VOLUME_HALF_SPAN_M,
    )
    bounds = {"x": lateral, "y": lateral, "z": optical.Q_BOUNDS["z"]}
    for axis in "xy":
        fdtd.setnamed(optical.PABS_GROUP, axis, 0.0)
        fdtd.setnamed(optical.PABS_GROUP, f"{axis} span", lateral[1] - lateral[0])
        fdtd.setnamed(FIELD_REGION, f"{axis} min", lateral[0])
        fdtd.setnamed(FIELD_REGION, f"{axis} max", lateral[1])
    for normal_axis in "xyz":
        for side, position in zip(("min", "max"), bounds[normal_axis]):
            name = f"run010_flux_{normal_axis}_{side}"
            fdtd.setnamed(name, normal_axis, position)
            for transverse_axis in "xyz":
                if transverse_axis == normal_axis:
                    continue
                fdtd.setnamed(name, f"{transverse_axis} min", bounds[transverse_axis][0])
                fdtd.setnamed(name, f"{transverse_axis} max", bounds[transverse_axis][1])

    pabs_readback = {
        axis: [
            float(fdtd.getnamed(optical.PABS_GROUP, axis))
            - 0.5 * float(fdtd.getnamed(optical.PABS_GROUP, f"{axis} span")),
            float(fdtd.getnamed(optical.PABS_GROUP, axis))
            + 0.5 * float(fdtd.getnamed(optical.PABS_GROUP, f"{axis} span")),
        ]
        for axis in "xyz"
    }
    field_readback = named_bounds(fdtd, FIELD_REGION)
    flux_readback = {
        f"run010_flux_{axis}_{side}": named_bounds(
            fdtd, f"run010_flux_{axis}_{side}"
        )
        for axis in "xyz"
        for side in ("min", "max")
    }
    if not all(
        np.allclose(pabs_readback[axis], bounds[axis], rtol=0.0, atol=2e-18)
        and np.allclose(field_readback[axis], bounds[axis], rtol=0.0, atol=2e-18)
        for axis in "xyz"
    ):
        raise RuntimeError("response absorption/field control-volume readback mismatch")
    for normal_axis in "xyz":
        for side in ("min", "max"):
            monitor = flux_readback[f"run010_flux_{normal_axis}_{side}"]
            position = bounds[normal_axis][0 if side == "min" else 1]
            if not all(
                np.allclose(
                    monitor[axis],
                    [position, position] if axis == normal_axis else bounds[axis],
                    rtol=0.0,
                    atol=2e-18,
                )
                for axis in "xyz"
            ):
                raise RuntimeError("response flux control-volume readback mismatch")
    return {
        "requested_bounds_m": {axis: list(value) for axis, value in bounds.items()},
        "pabs_readback_m": pabs_readback,
        "field_readback_m": field_readback,
        "flux_readback_m": flux_readback,
    }


def configure_fixed_device(fdtd: object, rho: np.ndarray, contact_axis: str) -> dict[str, object]:
    if not np.isclose(CONTRACT.flake_span_m, 24.0e-6, rtol=0.0, atol=1.0e-18):
        raise RuntimeError("beam response requires the immutable 24 um flake")
    set_density(fdtd, rho)

    half_domain = 0.5 * OPTICAL_DOMAIN_SPAN_M
    fdtd.setnamed("FDTD", "x min", -half_domain)
    fdtd.setnamed("FDTD", "x max", half_domain)
    fdtd.setnamed("FDTD", "y min", -half_domain)
    fdtd.setnamed("FDTD", "y max", half_domain)
    material_half = half_domain + 1.0e-6
    for name in SUBSTRATE_OBJECTS:
        fdtd.setnamed(name, "x min", -material_half)
        fdtd.setnamed(name, "x max", material_half)
        fdtd.setnamed(name, "y min", -material_half)
        fdtd.setnamed(name, "y max", material_half)
    fdtd.setnamed(OUTER_MESH, "x min", -half_domain)
    fdtd.setnamed(OUTER_MESH, "x max", half_domain)
    fdtd.setnamed(OUTER_MESH, "y min", -half_domain)
    fdtd.setnamed(OUTER_MESH, "y max", half_domain)
    fdtd.setnamed(FLAKE_MESH, "z max", AU_THICKNESS_M + 10.0e-9)
    fdtd.setnamed(FLAKE_MESH, "dz", 5.0e-9)
    for axis in "xy":
        fdtd.setnamed(
            STACK_MESH, f"{axis} min", -RESPONSE_CONTROL_VOLUME_HALF_SPAN_M
        )
        fdtd.setnamed(
            STACK_MESH, f"{axis} max", RESPONSE_CONTROL_VOLUME_HALF_SPAN_M
        )

    material = fdtd.addmaterial("(n,k) Material")
    fdtd.setmaterial(material, "name", AU_MATERIAL)
    fdtd.setmaterial(AU_MATERIAL, "Refractive Index", AU_INDEX_AT_10UM.real)
    fdtd.setmaterial(AU_MATERIAL, "Imaginary Refractive Index", AU_INDEX_AT_10UM.imag)
    au_bounds = electrode_bounds_m(contact_axis)
    for name, bounds in zip(AU_OBJECT_NAMES, au_bounds):
        optical.add_rect(fdtd, name, AU_MATERIAL, bounds)
    response_control_volume = configure_response_control_volume(fdtd)
    stack_mesh_bounds = named_bounds(fdtd, STACK_MESH)
    if not all(
        np.allclose(
            stack_mesh_bounds[axis],
            [-RESPONSE_CONTROL_VOLUME_HALF_SPAN_M, RESPONSE_CONTROL_VOLUME_HALF_SPAN_M],
            rtol=0.0,
            atol=2e-18,
        )
        for axis in "xy"
    ):
        raise RuntimeError("illuminated-stack mesh does not cover response control volume")

    design_bounds = named_bounds(fdtd, optical.DESIGN_OBJECT)
    fixed_names = (
        ("run010_fixed_TaIrTe4_frame_left_contact", "run010_fixed_TaIrTe4_frame_right_contact")
        if contact_axis == "x"
        else ("run010_fixed_TaIrTe4_frame_bottom_contact", "run010_fixed_TaIrTe4_frame_top_contact")
    )
    fixed_bounds = {name: named_bounds(fdtd, name) for name in fixed_names}
    flake_low, flake_high = FLAKE_BOUNDS_M
    all_tairte4_bounds = (design_bounds, *fixed_bounds.values())
    flake_unchanged = all(
        bounds[axis][0] >= flake_low - 1e-18
        and bounds[axis][1] <= flake_high + 1e-18
        for bounds in all_tairte4_bounds
        for axis in "xy"
    )
    design_expected = {
        axis: list(CONTRACT.design_bounds_m[axis]) for axis in "xy"
    }
    design_unchanged = all(
        np.allclose(design_bounds[axis], design_expected[axis], rtol=0.0, atol=2e-18)
        for axis in "xy"
    )
    au_inside_flake = all(
        bounds[axis][0] >= flake_low - 1e-18
        and bounds[axis][1] <= flake_high + 1e-18
        for bounds in au_bounds
        for axis in "xy"
    )
    if not (flake_unchanged and design_unchanged and au_inside_flake):
        raise RuntimeError("fixed-flake/Au geometry audit failed")
    return {
        "flake_bounds_m": {"x": list(FLAKE_BOUNDS_M), "y": list(FLAKE_BOUNDS_M), "z": [-CONTRACT.flake_thickness_m, 0.0]},
        "design_bounds_readback_m": design_bounds,
        "fixed_TaIrTe4_bounds_readback_m": fixed_bounds,
        "Au_bounds_m": [{axis: list(value) for axis, value in bounds.items()} for bounds in au_bounds],
        "flake_geometry_unchanged": flake_unchanged,
        "design_geometry_unchanged": design_unchanged,
        "Au_entirely_inside_original_flake_xy": au_inside_flake,
        "optical_domain_span_m": OPTICAL_DOMAIN_SPAN_M,
        "response_control_volume": response_control_volume,
        "illuminated_stack_mesh_bounds_m": stack_mesh_bounds,
        "flake_expanded_for_scan": False,
    }


def configure_source(fdtd: object, item: dict[str, float | str], angle_deg: float) -> dict[str, object]:
    x_um = float(item["x_um"])
    y_um = float(item["y_um"])
    bounds = source_bounds_m(x_um, y_um)
    center = domain_center_m(x_um, y_um)
    half_domain = 0.5 * OPTICAL_DOMAIN_SPAN_M
    material_half = half_domain + 1.0e-6
    for axis in "xy":
        fdtd.setnamed("FDTD", f"{axis} min", center[axis] - half_domain)
        fdtd.setnamed("FDTD", f"{axis} max", center[axis] + half_domain)
        fdtd.setnamed(OUTER_MESH, f"{axis} min", center[axis] - half_domain)
        fdtd.setnamed(OUTER_MESH, f"{axis} max", center[axis] + half_domain)
        for name in SUBSTRATE_OBJECTS:
            fdtd.setnamed(name, f"{axis} min", center[axis] - material_half)
            fdtd.setnamed(name, f"{axis} max", center[axis] + material_half)
    source_object_waist_m = float(item["waist_um"]) * 1.0e-6 * SOURCE_OBJECT_WAIST_SCALE
    fdtd.setnamed(optical.SOURCE_NAME, "enabled", True)
    fdtd.setnamed(optical.SOURCE_NAME, "amplitude", 1.0)
    fdtd.setnamed(optical.SOURCE_NAME, "polarization angle", angle_deg)
    fdtd.setnamed(optical.SOURCE_NAME, "waist radius w0", source_object_waist_m)
    for axis in "xy":
        fdtd.setnamed(optical.SOURCE_NAME, f"{axis} min", bounds[axis][0])
        fdtd.setnamed(optical.SOURCE_NAME, f"{axis} max", bounds[axis][1])
    fdtd.setnamed(FIELD_REGION, "source mode", False)
    readback = {
        "polarization_angle_deg": float(fdtd.getnamed(optical.SOURCE_NAME, "polarization angle")),
        "source_object_waist_m": float(fdtd.getnamed(optical.SOURCE_NAME, "waist radius w0")),
        "x_bounds_m": [float(fdtd.getnamed(optical.SOURCE_NAME, "x min")), float(fdtd.getnamed(optical.SOURCE_NAME, "x max"))],
        "y_bounds_m": [float(fdtd.getnamed(optical.SOURCE_NAME, "y min")), float(fdtd.getnamed(optical.SOURCE_NAME, "y max"))],
    }
    if not np.isclose(readback["polarization_angle_deg"], angle_deg, rtol=0.0, atol=1e-12):
        raise RuntimeError("source polarization readback mismatch")
    if not np.isclose(readback["source_object_waist_m"], source_object_waist_m, rtol=0.0, atol=1e-15):
        raise RuntimeError("source waist readback mismatch")
    return {
        "requested": item,
        "aperture_bounds_m": bounds,
        "FDTD_domain_center_m": center,
        "readback": readback,
    }


def run_maxwell(fdtd: object, audit: object, runtime: object, output: Path, role: str) -> dict[str, object]:
    working = output / "beam_response_working.fsp"
    fdtd.save(str(working))
    started = time.monotonic()
    attempt_errors: list[str] = []
    attempts: list[dict[str, object]] = []
    for attempt in range(1, MAXWELL_RUN_ATTEMPTS + 1):
        resources = runtime.configure_session_resources(fdtd)
        try:
            with lumerical_gpu_engine_lock() as lock_metadata:
                resource_used = audit.strict_gpu_run(fdtd, f"beam_response_{role}")
            attempts.append({
                "attempt": attempt,
                "status": "COMPLETED",
                "resources": resources,
                "gpu_engine_lock": lock_metadata,
            })
            break
        except RuntimeError as error:
            attempt_errors.append(f"attempt {attempt}: {error}")
            attempts.append({
                "attempt": attempt,
                "status": "FAILED_TRANSIENT_GPU_RESOURCE",
                "resources": resources,
                "error": str(error),
            })
            if attempt == MAXWELL_RUN_ATTEMPTS:
                raise RuntimeError(
                    "GPU-only Maxwell run exhausted transient-resource retries: "
                    + " | ".join(attempt_errors)
                ) from error
            time.sleep(MAXWELL_RETRY_WAIT_S * attempt)
    wall_s = time.monotonic() - started

    fdtd.runanalysis(optical.PABS_GROUP)
    source_power = audit.scalar(
        fdtd.sourcepower(FREQUENCY_HZ, 2, optical.SOURCE_NAME), "sourcepower"
    )
    net_outward = 0.0
    faces: dict[str, dict[str, float]] = {}
    for axis in "xyz":
        for side, sign in (("min", -1.0), ("max", 1.0)):
            name = f"run010_flux_{axis}_{side}"
            signed = audit.scalar(fdtd.transmission(name), name) * source_power
            outward = sign * signed
            faces[name] = {"signed_axis_power_W": signed, "outward_power_W": outward}
            net_outward += outward
    p_six = -net_outward
    q = extract_native_yee_q(
        fdtd,
        field_monitor=optical.PABS_FIELD,
        index_monitor=optical.PABS_INDEX,
        wavelength_m=CONTRACT.wavelength_m,
    )
    p_q = float(q["P_Q_W"])
    closure = abs(p_q - p_six) / max(abs(p_six), np.finfo(float).tiny)
    q_min = min(float(np.min(np.asarray(q["Q_components"][axis], float))) for axis in "xyz")
    finite = all(np.all(np.isfinite(np.asarray(q["Q_components"][axis]))) for axis in "xyz")
    log = audit.log_audit(output)
    return {
        "q": q,
        "source_power_W": source_power,
        "P_Q_W": p_q,
        "P_six_W": p_six,
        "closure": closure,
        "Q_minimum_W_m3": q_min,
        "Q_all_finite": bool(finite),
        "faces": faces,
        "resources": resources,
        "run_attempts": attempts,
        "resource_used": resource_used,
        "wall_s": wall_s,
        "log_audit": log,
    }


def support_power(coupled: dict[str, object], mask_name: str) -> float:
    state = coupled["state"]
    q = np.asarray(coupled["mapped_q"], float)
    mask = state.masks[mask_name]
    return float(np.sum(q[mask] * state.system.cell_volume_m3[mask]))


def evaluate_input(
    fdtd: object,
    audit: object,
    runtime: object,
    output: Path,
    rho: np.ndarray,
    case: object,
    item: dict[str, float | str],
    gpu_device: str,
    cuda_device: int,
) -> dict[str, object]:
    fdtd.switchtolayout()
    source = configure_source(fdtd, item, polarization_angle(case.polarization))
    forward = run_maxwell(fdtd, audit, runtime, output, f"run{case.run:03d}_{item['id']}")
    coupled = solve_coupled(
        forward,
        rho,
        cuda_device,
        need_adjoint=False,
        thermal_state_kwargs={
            "au_contact_axis": case.contact_axis,
            "au_thickness_m": AU_THICKNESS_M,
            "au_tairte4_interface_conductance_W_m2K": AU_TAIRTE4_INTERFACE_CONDUCTANCE_W_M2K,
        },
        thermal_relative_tolerance=BEAM_RESPONSE_THERMAL_SOLVE_TOLERANCE,
    )
    scale = TARGET_POWER_W / float(forward["source_power_W"])
    current_A = float(coupled["electrical"].current_A) * scale
    temperature = np.asarray(coupled["temperature"], float) * scale
    powers = {
        "Au_W": support_power(coupled, "Au_electrodes") * scale,
        "TaIrTe4_W": support_power(coupled, "flake_support") * scale,
        "SiO2_W": support_power(coupled, "SiO2") * scale,
        "Si_W": support_power(coupled, "Si") * scale,
    }
    auto_shutoff = forward["log_audit"].get("final_auto_shutoff")
    gates = {
        "optical_closure_lt_2pct": bool(
            float(forward["closure"]) < OPTICAL_CLOSURE_GATE
        ),
        "Q_nonnegative_and_finite": bool(forward["Q_minimum_W_m3"] >= 0.0 and forward["Q_all_finite"]),
        "auto_shutoff_lt_1e_5": bool(auto_shutoff is not None and auto_shutoff < 1.0e-5),
        "Q_mapping_error_lt_0p5pct": bool(coupled["mapping"]["relative_mapping_error"] < 0.005),
        "thermal_residual_lt_1e_8": bool(coupled["thermal_forward"].explicit_relative_residual < 1.0e-8),
        "thermal_energy_error_lt_1pct": bool(coupled["energy"] < 0.01),
        "electrical_residual_lt_1e_8": bool(coupled["electrical"].weighting_residual < 1.0e-8),
        "finite_terminal_current": bool(np.isfinite(current_A)),
    }
    return {
        "id": item["id"],
        "kind": item["kind"],
        "target_waist_um": float(item["waist_um"]),
        "beam_x_um": float(item["x_um"]),
        "beam_y_um": float(item["y_um"]),
        "source": source,
        "incident_power_W": TARGET_POWER_W,
        "linear_scale_from_raw_source": scale,
        "terminal_current_A": current_A,
        "terminal_current_nA": current_A * 1.0e9,
        "terminal_conductance_S": float(coupled["electrical"].terminal_conductance_S),
        "Tmax_flake_K": float(np.nanmax(temperature)),
        "mapped_power_at_285uW": powers,
        "P_Q_at_285uW_W": float(forward["P_Q_W"]) * scale,
        "P_six_at_285uW_W": float(forward["P_six_W"]) * scale,
        "optical_closure": float(forward["closure"]),
        "auto_shutoff": auto_shutoff,
        "Q_mapping_error": float(coupled["mapping"]["relative_mapping_error"]),
        "thermal_residual": float(coupled["thermal_forward"].explicit_relative_residual),
        "thermal_solver_requested_relative_tolerance": BEAM_RESPONSE_THERMAL_SOLVE_TOLERANCE,
        "thermal_energy_error": float(coupled["energy"]),
        "electrical_residual": float(coupled["electrical"].weighting_residual),
        "gpu_device": gpu_device,
        "cuda_device": cuda_device,
        "FDTD_threads": BEAM_RESPONSE_FDTD_THREADS,
        "Maxwell_wall_s": float(forward["wall_s"]),
        "gates": gates,
        "passed": all(gates.values()),
    }


def evaluate_in_fresh_session(
    *,
    base_fsp: Path,
    rho: np.ndarray,
    case: object,
    item: dict[str, float | str],
    output: Path,
    gpu_device: str,
    cuda_device: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Use one Lumerical session per input to avoid stale GPU resource handles."""

    fdtd = None
    try:
        fdtd, audit, runtime = open_fdtd_with_retry(gpu_device)
        fdtd.load(str(base_fsp))
        fdtd.switchtolayout()
        geometry = configure_fixed_device(fdtd, rho, case.contact_axis)
        row = evaluate_input(
            fdtd, audit, runtime, output, rho, case, item,
            gpu_device, cuda_device,
        )
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


def transient_session_error(error: Exception) -> bool:
    message = str(error)
    return any(
        marker in message
        for marker in (
            "GPU-only Maxwell run exhausted transient-resource retries",
            "Lumerical session open exhausted retries",
            "Insufficient FlexNet Publisher",
            "Failed to set up Ansys license sharing",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, choices=sorted(CASES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", required=True)
    parser.add_argument("--cuda-device", type=int, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--smoke", action="store_true")
    selection.add_argument("--input-id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    case = CASES[args.run]
    if CONTRACT.geometry_mode != case.geometry_mode:
        raise RuntimeError(
            f"Run {case.run:03d} requires TAIRTE4_TOPOLOGY_GEOMETRY={case.geometry_mode}, "
            f"got {CONTRACT.geometry_mode}"
        )
    if case.interface_scenario != __import__(
        "photothermal_pte.optimization_runs.tairte4_flake_topology.thermal",
        fromlist=["TAIRTE4_SIO2_INTERFACE_SCENARIO"],
    ).TAIRTE4_SIO2_INTERFACE_SCENARIO:
        raise RuntimeError("TAIRTE4_SIO2_INTERFACE_SCENARIO does not match selected run")

    base_fsp = checked(case.base_fsp, case.base_fsp_sha256)
    density_path = checked(case.density_path, case.density_sha256)
    rho = load_rho(density_path)
    if not np.array_equal(np.unique(rho), np.asarray((0.0, 1.0))):
        raise RuntimeError("beam response requires an exact-binary density")

    output = args.output_dir.expanduser().resolve()
    checkpoint = output / "response_checkpoint.json"
    result_path = output / "beam_response_result.json"
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise RuntimeError(f"refusing non-empty output directory without --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and not args.smoke and args.input_id is None:
        existing = load_valid_completed_result(result_path, case.run)
        if existing is not None:
            print(json.dumps({
                "run": case.run,
                "status": "COMPLETED",
                "passed": True,
                "responses": len(existing["responses"]),
                "output": str(output),
                "event": "reused_valid_completed_result",
            }, indent=2), flush=True)
            return 0
    completed: list[dict[str, object]] = []
    checkpoint_payload: dict[str, object] = {}
    if args.resume and checkpoint.is_file():
        checkpoint_payload = json.loads(checkpoint.read_text())
        requested_bounds = (
            checkpoint_payload.get("geometry", {})
            .get("response_control_volume", {})
            .get("requested_bounds_m", {})
        )
        current_lateral = [
            -RESPONSE_CONTROL_VOLUME_HALF_SPAN_M,
            RESPONSE_CONTROL_VOLUME_HALF_SPAN_M,
        ]
        compatible_checkpoint = (
            checkpoint_payload.get("schema") == RESULT_SCHEMA
            and all(
                np.allclose(
                    requested_bounds.get(axis, []), current_lateral,
                    rtol=0.0, atol=2e-18,
                )
                for axis in "xy"
            )
        )
        if compatible_checkpoint:
            normalized_rows = [
                normalize_response_gates(row)
                for row in checkpoint_payload.get("responses", [])
            ]
            completed = [
                row for row in normalized_rows
                if row.get("passed")
            ]
        else:
            checkpoint_payload = {}
    completed_ids = {str(row["id"]) for row in completed if row.get("passed")}

    started = time.monotonic()
    payload: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": "RUNNING",
        "run": case.run,
        "generated_at_utc": utc_now(),
        "responses": completed,
    }
    try:
        geometry = checkpoint_payload.get("geometry")
        inputs = sweep_inputs(smoke=args.smoke)
        if args.input_id:
            inputs = [item for item in inputs if item["id"] == args.input_id]
            if not inputs:
                raise ValueError(f"unknown beam-response input id: {args.input_id}")
        for index, item in enumerate(inputs, start=1):
            if str(item["id"]) in completed_ids:
                continue
            input_errors: list[str] = []
            for input_attempt in range(1, INPUT_SESSION_ATTEMPTS + 1):
                try:
                    row, geometry = evaluate_in_fresh_session(
                        base_fsp=base_fsp,
                        rho=rho,
                        case=case,
                        item=item,
                        output=output,
                        gpu_device=args.gpu_device,
                        cuda_device=args.cuda_device,
                    )
                    break
                except Exception as error:
                    if not transient_session_error(error):
                        raise
                    input_errors.append(
                        f"attempt {input_attempt}: {type(error).__name__}: {error}"
                    )
                    print(json.dumps({
                        "run": case.run,
                        "id": item["id"],
                        "event": "transient_session_retry",
                        "attempt": input_attempt,
                    }), flush=True)
                    if input_attempt == INPUT_SESSION_ATTEMPTS:
                        raise RuntimeError(
                            "beam input exhausted fresh-session retries: "
                            + " | ".join(input_errors)
                        ) from error
                    time.sleep(30.0 * input_attempt)
            else:
                raise AssertionError("unreachable")
            completed.append(row)
            payload.update(
                status="RUNNING" if row["passed"] else "FAILED_NUMERICAL_GATE",
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
                raise RuntimeError(f"Run {case.run:03d} {item['id']} failed a numerical gate")

        payload = {
            "schema": RESULT_SCHEMA,
            "status": "COMPLETED" if all(row["passed"] for row in completed) else "FAILED_NUMERICAL_GATE",
            "passed": bool(completed and all(row["passed"] for row in completed)),
            "generated_at_utc": utc_now(),
            "run": case.run,
            "contact_axis": case.contact_axis,
            "geometry_mode": case.geometry_mode,
            "interface_scenario": case.interface_scenario,
            "polarization": case.polarization,
            "scope": "fixed exact-binary structure; forward Maxwell/thermal/electrical response only; no optimization",
            "axis_contract": "Lumerical x=b, y=a, z=c",
            "target_incident_power_W": TARGET_POWER_W,
            "geometry": geometry,
            "Au_contract": {
                "material": AU_MATERIAL,
                "thickness_m": AU_THICKNESS_M,
                "complex_index_at_10um": {"real": AU_INDEX_AT_10UM.real, "imag": AU_INDEX_AT_10UM.imag},
                "thermal_conductivity_W_mK": 317.0,
                "Au_TaIrTe4_interface_conductance_W_m2K": AU_TAIRTE4_INTERFACE_CONDUCTANCE_W_M2K,
                "optical_reference": AU_OPTICAL_REFERENCE,
                "interface_reference": AU_INTERFACE_REFERENCE,
            },
            "inputs": {
                "base_FSP": {"path": str(base_fsp), "sha256": sha256(base_fsp)},
                "exact_binary_density": {"path": str(density_path), "sha256": sha256(density_path)},
            },
            "responses": completed,
            "all_gates_passed": all(row["passed"] for row in completed),
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
        working = output / "beam_response_working.fsp"
        if working.is_file():
            working.unlink()
        concurrent_completion = load_valid_completed_result(result_path, case.run)
        if not payload.get("passed") and concurrent_completion is not None:
            payload = concurrent_completion
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
