"""Fresh all-air Ea/Eb source calibration for the 4-um parity grid."""

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

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_microbenchmark import (
    REPOSITORY,
    _git_output,
    _write_new_external_json,
    query_and_require_idle_gpu,
)


SCHEMA_CASE = "fdtdx_4um_parity_source_case_v1"
SCHEMA_AGGREGATE = "fdtdx_4um_parity_source_calibration_v1"
TARGET_POWER_W = 285.0e-6
ETA0_OHM = 4.0e-7 * math.pi * 299_792_458.0
MAX_TD_PHASOR_MISMATCH = 0.02
MAX_PREVIOUS_LATE_MISMATCH = 0.005
MAX_VACUUM_IMPEDANCE_ERROR = 0.02
MAX_CROSS_POLARIZATION_RATIO = 1.0e-5
MAX_EA_EB_POWER_MISMATCH = 0.005


def normalized_flux_to_si_W(normalized_flux: float) -> float:
    """Convert FDTDX E_stored x H_stored area integral to SI watts.

    Pinned FDTDX uses ``E_stored = E_SI / eta0`` and ``H_stored = H_SI``.
    The detector therefore returns physical power divided by ``eta0``.
    """

    return ETA0_OHM * float(normalized_flux)


def polarization_components(polarization: str) -> tuple[int, int]:
    if polarization == "Ea":
        return 1, 0  # Ey, Hx for propagation along -z
    if polarization == "Eb":
        return 0, 1  # Ex, Hy for propagation along -z
    raise ValueError(f"unknown polarization {polarization!r}")


def relative_mismatch(first: float, second: float) -> float:
    denominator = max(abs(float(first)), abs(float(second)))
    if denominator == 0.0:
        return math.inf
    return abs(float(first) - float(second)) / denominator


def case_gate(metrics: dict[str, float | bool]) -> str:
    passed = (
        bool(metrics["finite"])
        and float(metrics["incident_power_late_W"]) > 0.0
        and float(metrics["td_phasor_mismatch_relative"]) < MAX_TD_PHASOR_MISMATCH
        and float(metrics["previous_late_mismatch_relative"])
        < MAX_PREVIOUS_LATE_MISMATCH
        and float(metrics["vacuum_impedance_error_relative"])
        < MAX_VACUUM_IMPEDANCE_ERROR
        and float(metrics["cross_polarization_ratio"])
        < MAX_CROSS_POLARIZATION_RATIO
    )
    return "PASS_SOURCE_CASE" if passed else "BLOCKED"


def _validate_new_external_path(output_path: Path) -> Path:
    output = output_path.expanduser().resolve()
    if output == REPOSITORY or REPOSITORY in output.parents:
        raise RuntimeError("raw calibration JSON must remain outside Git")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    return output


def _report_hash(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _run_case(args: argparse.Namespace) -> dict[str, object]:
    output_path = _validate_new_external_path(args.output_json)
    gpu_snapshot = query_and_require_idle_gpu(args.gpu_uuid)
    existing_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing_visibility not in {None, "", args.gpu_uuid}:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES conflicts with requested UUID: "
            f"{existing_visibility!r}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # GPU-sensitive imports must follow UUID isolation.
    import jax

    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_model import (
        build_model,
        setup_audit,
    )

    print(f"phase=build polarization={args.polarization}", flush=True)
    build_started = time.perf_counter()
    model = build_model(args.polarization, backend="gpu", air_only=True)
    setup = setup_audit(model)
    arrays = model["base"].reset()
    jax.block_until_ready(arrays.fields.E)
    build_seconds = time.perf_counter() - build_started
    if setup["status"] != "PASS":
        raise RuntimeError(f"air-only setup audit failed: {setup}")

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
    jax.block_until_ready(result.detector_states["incident_plane"]["phasor"])
    forward_seconds = time.perf_counter() - forward_started

    late_detector = model["placed"]["incident_plane"]
    previous_detector = model["placed"]["incident_plane_previous"]
    late_raw = float(
        np.asarray(
            late_detector.compute_poynting_flux(
                result.detector_states["incident_plane"]
            )
        )[0]
    )
    previous_raw = float(
        np.asarray(
            previous_detector.compute_poynting_flux(
                result.detector_states["incident_plane_previous"]
            )
        )[0]
    )
    td_values = np.asarray(
        result.detector_states["incident_plane_td"]["poynting_flux"],
        dtype=np.float64,
    ).reshape(-1)
    td_raw = float(np.mean(td_values))

    incident_phasor = np.asarray(
        result.detector_states["incident_plane"]["phasor"][0, 0]
    )
    e_component, h_component = polarization_components(args.polarization)
    e_norm = float(np.linalg.norm(incident_phasor[e_component].ravel()))
    h_norm = float(np.linalg.norm(incident_phasor[3 + h_component].ravel()))
    normalized_impedance = e_norm / h_norm

    endpoint = np.asarray(
        result.detector_states["endpoint_field"]["phasor"][0, 0]
    )
    desired_endpoint_norm = float(np.linalg.norm(endpoint[e_component].ravel()))
    leakage_endpoint_norm = float(
        np.sqrt(
            sum(
                np.linalg.norm(endpoint[index].ravel()) ** 2
                for index in range(3)
                if index != e_component
            )
        )
    )
    cross_ratio = leakage_endpoint_norm / desired_endpoint_norm

    late_W = normalized_flux_to_si_W(late_raw)
    previous_W = normalized_flux_to_si_W(previous_raw)
    td_W = normalized_flux_to_si_W(td_raw)
    finite = bool(
        np.all(np.isfinite(incident_phasor))
        and np.all(np.isfinite(endpoint))
        and np.all(np.isfinite(td_values))
        and all(
            math.isfinite(value)
            for value in (
                late_W,
                previous_W,
                td_W,
                normalized_impedance,
                cross_ratio,
            )
        )
    )
    metrics: dict[str, float | bool] = {
        "finite": finite,
        "incident_power_late_W": late_W,
        "incident_power_previous_W": previous_W,
        "incident_power_td_mean_W": td_W,
        "td_phasor_mismatch_relative": relative_mismatch(td_W, late_W),
        "previous_late_mismatch_relative": relative_mismatch(previous_W, late_W),
        "normalized_vacuum_impedance": normalized_impedance,
        "vacuum_impedance_error_relative": abs(normalized_impedance - 1.0),
        "cross_polarization_ratio": cross_ratio,
        "desired_endpoint_phasor_l2": desired_endpoint_norm,
    }
    power_scale = TARGET_POWER_W / late_W
    report: dict[str, object] = {
        "schema": SCHEMA_CASE,
        "status": case_gate(metrics),
        "scope": "all_air_source_only_on_exact_parity_grid",
        "physics_validated": False,
        "source_calibration_case_validated": case_gate(metrics)
        == "PASS_SOURCE_CASE",
        "polarization": args.polarization,
        "metrics": metrics,
        "target_incident_power_W": TARGET_POWER_W,
        "power_or_Q_scale_to_target": power_scale,
        "field_amplitude_scale_to_target": math.sqrt(power_scale),
        "stored_field_convention": "E_stored=E_SI/eta0; H_stored=H_SI",
        "normalized_flux_to_W": "multiply_by_eta0",
        "eta0_ohm": ETA0_OHM,
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
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "git_status_porcelain": _git_output(["status", "--porcelain"]),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "raw_result_in_git": False,
    }
    report["report_sha256"] = _report_hash(report)
    _write_new_external_json(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def aggregate_cases(
    ea: dict[str, Any],
    eb: dict[str, Any],
) -> dict[str, object]:
    cases = {str(case["polarization"]): case for case in (ea, eb)}
    if set(cases) != {"Ea", "Eb"}:
        raise RuntimeError(f"expected one Ea and one Eb case, got {set(cases)}")
    for polarization, case in cases.items():
        if case.get("schema") != SCHEMA_CASE:
            raise RuntimeError(f"{polarization} case schema mismatch")
        claimed_hash = case.get("report_sha256")
        unhashed = dict(case)
        unhashed.pop("report_sha256", None)
        if claimed_hash != _report_hash(unhashed):
            raise RuntimeError(f"{polarization} case report hash mismatch")
    invariant_keys = (
        "git_commit",
        "script_sha256",
        "cublas_runtime_version",
    )
    invariants_match = all(
        cases["Ea"].get(key) == cases["Eb"].get(key) for key in invariant_keys
    )
    clean_cases = all(case.get("git_status_porcelain") == "" for case in cases.values())
    case_gates = all(case.get("status") == "PASS_SOURCE_CASE" for case in cases.values())
    powers = {
        polarization: float(case["metrics"]["incident_power_late_W"])
        for polarization, case in cases.items()
    }
    mismatch = relative_mismatch(powers["Ea"], powers["Eb"])
    status = (
        "PASS_SOURCE_CALIBRATION"
        if invariants_match
        and clean_cases
        and case_gates
        and mismatch < MAX_EA_EB_POWER_MISMATCH
        else "BLOCKED"
    )
    return {
        "schema": SCHEMA_AGGREGATE,
        "status": status,
        "scope": "all_air_Ea_Eb_source_power_only",
        "physics_device_validated": False,
        "invariants_match": invariants_match,
        "clean_case_worktrees": clean_cases,
        "case_gates_pass": case_gates,
        "incident_power_W": powers,
        "Ea_Eb_incident_power_mismatch_relative": mismatch,
        "Ea_Eb_mismatch_limit": MAX_EA_EB_POWER_MISMATCH,
        "target_incident_power_W": TARGET_POWER_W,
        "power_or_Q_scale_to_target": {
            polarization: TARGET_POWER_W / power
            for polarization, power in powers.items()
        },
        "normalization_contract": (
            "scale each polarization independently by 285e-6/P_all_air_pol; "
            "never match Ea and Eb empirically"
        ),
        "case_report_sha256": {
            polarization: case["report_sha256"]
            for polarization, case in cases.items()
        },
        "git_commit": cases["Ea"].get("git_commit"),
        "script_sha256": cases["Ea"].get("script_sha256"),
        "raw_result_in_git": False,
    }


def _aggregate(args: argparse.Namespace) -> dict[str, object]:
    output_path = _validate_new_external_path(args.output_json)
    paths = {
        "Ea": args.ea_json.expanduser().resolve(),
        "Eb": args.eb_json.expanduser().resolve(),
    }
    payloads = {
        polarization: json.loads(path.read_text(encoding="utf-8"))
        for polarization, path in paths.items()
    }
    report = aggregate_cases(payloads["Ea"], payloads["Eb"])
    report["input_files"] = {
        polarization: {
            "path": str(path),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for polarization, path in paths.items()
    }
    report["report_sha256"] = _report_hash(report)
    _write_new_external_json(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    run.add_argument("--gpu-uuid", required=True)
    run.add_argument("--output-json", type=Path, required=True)
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
