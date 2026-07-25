#!/usr/bin/env python3
"""Fail-closed controls for the isolated 2 um TaIrTe4 steady-state HEAT model.

This stage never runs Maxwell and never changes the validated production Q
artifact.  Full-device HEAT cases are prohibited until the Q-volume,
anisotropic-kappa, interface-G, and analytic-control gates all pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import config_stage1 as config
from lumerical_api import (
    jsonable,
    open_device,
    select_installation,
    selected_properties,
    utc_timestamp,
    write_json,
)


BASELINE_COMMIT = "be2cbc2c9c77bbcc0265ce2c293affdbb08105de"
BATH_TEMPERATURE_K = 300.0
ACTIVE_SPAN_M = 2.0e-6
TAIRTE4_THICKNESS_M = 100.0e-9
BOTTOM_SIO2_THICKNESS_M = 285.0e-9
DESIGN_HEIGHT_M = 600.0e-9

TAIRTE4_K_W_MK = (14.4, 3.8, 1.0)
SIO2_K_W_MK = 1.38
SI_K_W_MK = 145.0
TAIRTE4_KZ_NOTE = "estimated value"

THERMAL_LATERAL_SPANS_M = tuple(v * 1.0e-6 for v in (4.0, 8.0, 16.0, 32.0))
SI_DEPTHS_M = tuple(v * 1.0e-6 for v in (2.0, 5.0, 10.0, 20.0))
G_BOTTOM_W_M2K = (1.0e6, 3.0e6, 7.37e6, 1.5e7, 3.0e7, 1.0e8, None)
G_TOP_W_M2K = (7.37e4, 7.37e5, 7.37e6, 7.37e7, None)
G_OXIDE_SI_W_M2K = 1.1e9

POWER_IMPORT_LIMIT = 0.005
ENERGY_BALANCE_LIMIT = 0.01
DOMAIN_CONVERGENCE_LIMIT = 0.01

BLOCKED_ANISOTROPIC = "BLOCKED_ANISOTROPIC_K_UNSUPPORTED"
BLOCKED_Q_FOOTPRINT = "BLOCKED_Q_ARTIFACT_INCOMPATIBLE_WITH_2UM_FOOTPRINT"
BLOCKED_INTERFACE_G = "BLOCKED_INTERFACE_G_UNVERIFIED"
BLOCKED_LICENSE = "BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE"


def parse_args() -> argparse.Namespace:
    default_run = (
        config.OUTPUT_ROOT
        / "20260725T_validated_qon_heat_steady"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--q-artifact",
        default=str(default_run / "fdtd_qon" / "q_on_physical.npz"),
    )
    parser.add_argument(
        "--fdtd-summary",
        default=str(default_run / "fdtd_qon" / "fdtd_absorption_summary.json"),
    )
    parser.add_argument(
        "--api-evidence",
        default=str(default_run / "tensor_retry" / "api_probe" / "thermal_material.json"),
    )
    parser.add_argument(
        "--isotropic-slab-evidence",
        default=str(default_run / "analytic_heat" / "analytic_summary.json"),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--lumerical-version", choices=("auto", "v261"), default="v261")
    parser.add_argument("--hide-gui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--live-api-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attempt a fresh v261 conductivity and interface-property probe.",
    )
    return parser.parse_args()


def trapz3(values: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    trap = np.trapezoid
    return float(trap(trap(trap(values, z, axis=2), y, axis=1), x, axis=0))


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path, generation_command: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "server_path": str(path.resolve()),
        "exists": path.is_file(),
        "generation_command": generation_command,
    }
    if path.is_file():
        record.update({"size_bytes": path.stat().st_size, "sha256": sha256(path)})
    return record


def scalar_from_npz(data: Any, name: str) -> float:
    if name not in data:
        raise ValueError(f"Q artifact lacks required scalar {name!r}")
    value = np.asarray(data[name]).reshape(-1)
    if value.size != 1:
        raise ValueError(f"Q artifact {name!r} is not scalar")
    result = float(value[0])
    if not np.isfinite(result):
        raise ValueError(f"Q artifact {name!r} is not finite")
    return result


def audit_q_artifact(path: Path, fdtd_summary_path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"validated Q artifact is missing: {path}")
    if not fdtd_summary_path.is_file():
        raise FileNotFoundError(f"validated FDTD summary is missing: {fdtd_summary_path}")
    fdtd_summary = json.loads(fdtd_summary_path.read_text())
    if not fdtd_summary.get("validated", False):
        raise ValueError("FDTD summary does not mark the production Q as validated")

    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(data["x_m"], float).reshape(-1)
        y = np.asarray(data["y_m"], float).reshape(-1)
        z = np.asarray(data["z_m"], float).reshape(-1)
        q = np.asarray(data["Q_on_W_m3"], float)
        if q.shape != (x.size, y.size, z.size):
            raise ValueError(f"Q shape {q.shape} does not match coordinates")
        if any(np.any(np.diff(axis) <= 0) for axis in (x, y, z)):
            raise ValueError("Q coordinates are not strictly increasing")
        if not np.all(np.isfinite(q)):
            raise ValueError("Q contains NaN or Inf")
        if np.any(q < 0.0):
            raise ValueError("validated Q contains negative samples")

        total_power = trapz3(q, x, y, z)
        artifact_power = scalar_from_npz(data, "P_abs_volume_W")
        incident_intensity = scalar_from_npz(data, "incident_intensity_W_m2")
        unit_response = bool(np.asarray(data["unit_response_mode"]).reshape(-1)[0])

        half = 0.5 * ACTIVE_SPAN_M
        ix = (x >= -half) & (x <= half)
        iy = (y >= -half) & (y <= half)
        iz = (z >= -TAIRTE4_THICKNESS_M) & (z <= 0.0)
        if min(np.count_nonzero(ix), np.count_nonzero(iy), np.count_nonzero(iz)) < 2:
            raise ValueError("Q grid does not resolve the requested TaIrTe4 volume")
        # Retain the complete original z grid. The validated artifact already
        # contains exact zero padding outside the flake; removing those padding
        # samples would change trapezoidal boundary weights and undercount Q.
        inside_power = trapz3(
            q[np.ix_(ix, iy, np.arange(z.size))],
            x[ix],
            y[iy],
            z,
        )
        outside_z = np.zeros_like(q)
        outside_z[:, :, ~iz] = q[:, :, ~iz]
        outside_z_power = trapz3(outside_z, x, y, z)

    reintegration_error = abs(total_power - artifact_power) / abs(artifact_power)
    predicted_import_error = abs(total_power - inside_power) / abs(total_power)
    return {
        "status": (
            "compatible"
            if predicted_import_error < POWER_IMPORT_LIMIT
            else BLOCKED_Q_FOOTPRINT
        ),
        "artifact": artifact_record(path),
        "fdtd_summary": artifact_record(fdtd_summary_path),
        "shape_xyz": list(q.shape),
        "coordinate_ranges_m": {
            "x": [float(x[0]), float(x[-1])],
            "y": [float(y[0]), float(y[-1])],
            "z": [float(z[0]), float(z[-1])],
        },
        "requested_TaIrTe4_bounds_m": {
            "x": [-half, half],
            "y": [-half, half],
            "z": [-TAIRTE4_THICKNESS_M, 0.0],
        },
        "P_Q_full_validated_grid_W": total_power,
        "P_Q_artifact_metadata_W": artifact_power,
        "full_grid_reintegration_relative_error": reintegration_error,
        "P_Q_inside_requested_2um_TaIrTe4_W": inside_power,
        "P_Q_outside_requested_TaIrTe4_z_W": outside_z_power,
        "inside_power_fraction": inside_power / total_power,
        "outside_power_fraction": predicted_import_error,
        "predicted_FDTD_to_HEAT_import_relative_error": predicted_import_error,
        "acceptance_limit": POWER_IMPORT_LIMIT,
        "compatible_without_clipping_or_rescaling": predicted_import_error < POWER_IMPORT_LIMIT,
        "normalization": {
            "unit_response_mode": unit_response,
            "incident_intensity_W_m2": incident_intensity,
            "reporting_quantity": (
                "DeltaT / incident intensity"
                if unit_response
                else "DeltaT at explicitly supplied incident intensity"
            ),
            "is_experimental_laser_temperature": False if unit_response else None,
        },
        "prohibited_operations_applied": {
            "gain": False,
            "clipping": False,
            "smoothing": False,
            "rescaling": False,
            "periodic_tiling": False,
        },
    }


def isotropic_slab_reference() -> dict[str, Any]:
    length = 1.0e-6
    k = 10.0
    q_volume = 1.0e14
    area = 1.0e-12
    cells = 400
    dz = length / cells
    matrix = np.zeros((cells, cells), float)
    rhs = np.full(cells, q_volume * area * dz, float)
    conductance = k * area / dz
    for index in range(cells):
        if index == 0:
            matrix[index, index] += 2.0 * conductance
        else:
            matrix[index, index] += conductance
            matrix[index, index - 1] -= conductance
        if index < cells - 1:
            matrix[index, index] += conductance
            matrix[index, index + 1] -= conductance
    delta_numeric = np.linalg.solve(matrix, rhs)
    z_center = (np.arange(cells) + 0.5) * dz
    delta_exact = q_volume / k * (length * z_center - 0.5 * z_center**2)
    nrmse = float(
        np.sqrt(np.mean((delta_numeric - delta_exact) ** 2))
        / (q_volume * length**2 / (2.0 * k))
    )
    top_numeric = float(delta_numeric[-1] + q_volume * dz**2 / (8.0 * k))
    top_exact = q_volume * length**2 / (2.0 * k)
    top_error = abs(top_numeric - top_exact) / top_exact
    return {
        "name": "single_isotropic_slab",
        "method": "independent cell-centered 1D finite-volume reference",
        "solver_verified": False,
        "offline_reference_passed": top_error < 1e-5 and nrmse < 1e-5,
        "cells": cells,
        "deltaT_top_exact_K": top_exact,
        "deltaT_top_numeric_K": top_numeric,
        "relative_error_deltaT_top": top_error,
        "normalized_rmse": nrmse,
    }


def multilayer_reference() -> dict[str, Any]:
    heat_flux = 1.0
    oxide_cells = 57
    silicon_cells = 400
    oxide_dz = BOTTOM_SIO2_THICKNESS_M / oxide_cells
    silicon_depth = 20.0e-6
    silicon_dz = silicon_depth / silicon_cells
    numeric_resistance = (
        np.sum(np.full(oxide_cells, oxide_dz / SIO2_K_W_MK))
        + np.sum(np.full(silicon_cells, silicon_dz / SI_K_W_MK))
    )
    exact_resistance = (
        BOTTOM_SIO2_THICKNESS_M / SIO2_K_W_MK
        + silicon_depth / SI_K_W_MK
    )
    relative_error = abs(numeric_resistance - exact_resistance) / exact_resistance
    return {
        "name": "multilayer_SiO2_Si_1D_resistance",
        "method": "layer-aligned finite-volume resistance sum",
        "solver_verified": False,
        "offline_reference_passed": relative_error < 1e-12,
        "heat_flux_W_m2": heat_flux,
        "thermal_resistance_exact_m2K_W": exact_resistance,
        "thermal_resistance_numeric_m2K_W": float(numeric_resistance),
        "deltaT_exact_K": heat_flux * exact_resistance,
        "relative_error": float(relative_error),
    }


def interface_g_reference() -> dict[str, Any]:
    heat_flux = 2.5e5
    conductance = 7.37e6
    layer_1_resistance = 0.4e-6 / 5.0
    layer_2_resistance = 0.8e-6 / 20.0
    interface_resistance = 1.0 / conductance
    total_resistance = (
        layer_1_resistance + interface_resistance + layer_2_resistance
    )
    jump = heat_flux * interface_resistance
    reconstructed = (
        heat_flux * layer_1_resistance
        + jump
        + heat_flux * layer_2_resistance
    )
    exact = heat_flux * total_resistance
    relative_error = abs(reconstructed - exact) / exact
    return {
        "name": "thermal_impedance_interface_G",
        "method": "series resistance with explicit 1/G interface",
        "solver_verified": False,
        "offline_reference_passed": relative_error < 1e-14,
        "G_W_m2K": conductance,
        "heat_flux_W_m2": heat_flux,
        "expected_interface_temperature_jump_K": jump,
        "total_deltaT_exact_K": exact,
        "total_deltaT_reconstructed_K": reconstructed,
        "relative_error": relative_error,
    }


def load_prior_api_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "available": False,
            "path": str(path),
            "diagonal_round_trip": False,
            "status": BLOCKED_ANISOTROPIC,
        }
    evidence = json.loads(path.read_text())
    diagonal_ok = bool(evidence.get("diagonal_round_trip", False))
    return {
        "available": True,
        "artifact": artifact_record(path),
        "scalar_request_W_mK": evidence.get("scalar_request_W_mK"),
        "scalar_return": evidence.get("scalar_return"),
        "scalar_round_trip": evidence.get("scalar_round_trip"),
        "diagonal_request_W_mK": evidence.get("diagonal_request_W_mK"),
        "diagonal_return": evidence.get("diagonal_return"),
        "diagonal_round_trip": diagonal_ok,
        "status": "passed" if diagonal_ok else BLOCKED_ANISOTROPIC,
    }


def attach_isotropic_solver_evidence(
    control: dict[str, Any], path: Path
) -> dict[str, Any]:
    result = dict(control)
    if not path.is_file():
        result["solver_evidence"] = {
            "available": False,
            "path": str(path),
        }
        return result
    evidence = json.loads(path.read_text())
    validated = bool(evidence.get("validated", False))
    result["solver_verified"] = validated
    result["solver_evidence"] = {
        "available": True,
        "artifact": artifact_record(path),
        "status": evidence.get("status"),
        "validated": validated,
        "selected_version": evidence.get("selected_version"),
        "finest": evidence.get("finest"),
    }
    return result


def live_api_probe(version: str, hide: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempted": True,
        "conductivity": {},
        "interface_G": {},
    }
    try:
        installation = select_installation(version)
        with open_device(installation, hide=hide) as device:
            result["solver_version"] = str(device.version())
            device.addmodelmaterial()
            device.set("name", "isolated 2um tensor probe")
            device.addhtmaterialproperty("Solid")
            device.set("name", "isolated 2um tensor probe thermal")
            device.set("thermal conductivity.active model", "constant")
            device.set("thermal conductivity.constant", 10.0)
            scalar_return = np.asarray(
                device.get("thermal conductivity.constant"), float
            ).reshape(-1)
            diagonal_request = np.asarray(TAIRTE4_K_W_MK, float)
            device.set("thermal conductivity.constant", diagonal_request)
            diagonal_return = np.asarray(
                device.get("thermal conductivity.constant"), float
            ).reshape(-1)
            result["conductivity"] = {
                "scalar_request_W_mK": 10.0,
                "scalar_return": scalar_return.tolist(),
                "scalar_round_trip": bool(
                    scalar_return.size == 1
                    and np.isclose(scalar_return[0], 10.0)
                ),
                "diagonal_request_W_mK": diagonal_request.tolist(),
                "diagonal_return": diagonal_return.tolist(),
                "diagonal_round_trip": bool(
                    diagonal_return.size == 3
                    and np.allclose(diagonal_return, diagonal_request)
                ),
            }

        with open_device(installation, hide=hide) as device:
            device.addsimulationregion()
            device.addheatsolver()
            boundary = device.addtemperaturebc("HEAT")
            del boundary
            properties = selected_properties(device)
            normalized = {" ".join(k.lower().split()): k for k in properties}
            impedance_property = normalized.get("thermal impedance")
            result["interface_G"] = {
                "temperature_bc_properties": properties,
                "thermal_impedance_property": impedance_property,
                "internal_domain_to_domain_G_solver_test": "not executed",
                "verified": False,
                "status": BLOCKED_INTERFACE_G,
            }
        result["status"] = "completed"
    except Exception as exc:
        result.update(
            {
                "status": BLOCKED_LICENSE,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return result


def sweep_contract() -> dict[str, Any]:
    return {
        "active_device": {
            "TaIrTe4_span_m": [ACTIVE_SPAN_M, ACTIVE_SPAN_M],
            "TaIrTe4_thickness_m": TAIRTE4_THICKNESS_M,
            "design_material": "SiO2",
            "design_height_m": DESIGN_HEIGHT_M,
            "bottom_SiO2_thickness_m": BOTTOM_SIO2_THICKNESS_M,
        },
        "materials_W_mK": {
            "TaIrTe4_diagonal": list(TAIRTE4_K_W_MK),
            "TaIrTe4_kz_note": TAIRTE4_KZ_NOTE,
            "SiO2": SIO2_K_W_MK,
            "Si": SI_K_W_MK,
        },
        "lateral_domain_spans_m": list(THERMAL_LATERAL_SPANS_M),
        "Si_depths_m": list(SI_DEPTHS_M),
        "G_bottom_W_m2K": ["perfect" if v is None else v for v in G_BOTTOM_W_M2K],
        "G_top_W_m2K": ["perfect" if v is None else v for v in G_TOP_W_M2K],
        "G_oxide_Si_W_m2K": G_OXIDE_SI_W_M2K,
        "oxide_Si_perfect_contact_comparison": True,
        "boundaries": {
            "bottom_Si": "T=300 K",
            "far_x_y_Si_and_SiO2": "T=300 K",
            "top_exposed_baseline": "adiabatic",
            "top_exposed_sensitivity": "h=10 W/(m2 K), Tamb=300 K",
            "thermal_periodic": False,
            "thermal_PML": False,
        },
        "acceptance": {
            "Q_import_relative_error": POWER_IMPORT_LIMIT,
            "energy_balance_relative_error": ENERGY_BALANCE_LIMIT,
            "domain_deltaT_max_relative_change": DOMAIN_CONVERGENCE_LIMIT,
            "domain_TaIrTe4_average_relative_change": DOMAIN_CONVERGENCE_LIMIT,
        },
    }


def raw_artifact_manifest(
    *,
    q_artifact: Path,
    fdtd_summary: Path,
    api_evidence: Path,
    isotropic_evidence: Path,
    control_results: Path,
    current_command: str,
) -> dict[str, Any]:
    validated_run = q_artifact.parents[1]
    disk_case = (
        config.REPOSITORY_ROOT
        / "reports"
        / "production_optical"
        / "cases"
        / "disk_x.json"
    )
    postprocess_project = None
    if disk_case.is_file():
        try:
            postprocess_project = json.loads(disk_case.read_text()).get("project")
        except Exception:
            postprocess_project = None
    q_command = (
        f"{sys.executable} {config.HERE / '02_export_fdtd_qon.py'} "
        f"--lumerical-version v261 --output-dir {validated_run} "
        f"--fixed-geometry centered-disk "
        f"--postprocess-project {postprocess_project or '<validated-production-disk.fsp>'} "
        f"--validated-case-json {disk_case}"
    )
    analytic_command = (
        f"{sys.executable} {config.HERE / '01_validate_heat_analytic.py'} "
        f"--lumerical-version v261 --output-dir {validated_run}"
    )
    large: dict[str, Any] = {}
    for path in sorted(validated_run.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {
            ".fsp",
            ".ldev",
            ".mat",
            ".npz",
            ".npy",
        }:
            continue
        generation = (
            analytic_command
            if "analytic_heat" in path.parts
            else q_command
            if "fdtd_qon" in path.parts
            else "legacy artifact; generation command unavailable"
        )
        large[str(path.relative_to(validated_run))] = artifact_record(
            path, generation
        )
    return {
        "policy": (
            "Large solver projects and raw 3-D fields are not committed; "
            "record SHA-256, byte size, server path, and generation command."
        ),
        "control_generation_command": current_command,
        "new_full_device_raw_artifacts": [],
        "new_full_device_raw_artifacts_reason": (
            "mandatory controls failed before a full-device project or field was created"
        ),
        "compact_control_artifacts": {
            "control_results": artifact_record(
                control_results, current_command
            ),
            "validated_FDTD_summary": artifact_record(fdtd_summary, q_command),
            "prior_API_evidence": artifact_record(api_evidence),
            "isotropic_slab_solver_evidence": artifact_record(
                isotropic_evidence, analytic_command
            ),
        },
        "pre_existing_large_artifacts": large,
    }


def main() -> int:
    args = parse_args()
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else config.OUTPUT_ROOT
        / f"{utc_timestamp()}_isolated_2um_heat_steady_controls"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory {output}")
    output.mkdir(parents=True, exist_ok=True)

    q_artifact = Path(args.q_artifact).expanduser().resolve()
    fdtd_summary = Path(args.fdtd_summary).expanduser().resolve()
    api_evidence_path = Path(args.api_evidence).expanduser().resolve()
    isotropic_evidence_path = (
        Path(args.isotropic_slab_evidence).expanduser().resolve()
    )
    command = shlex.join([sys.executable, *sys.argv])

    controls = {
        "single_isotropic_slab": attach_isotropic_solver_evidence(
            isotropic_slab_reference(), isotropic_evidence_path
        ),
        "multilayer_SiO2_Si": multilayer_reference(),
        "interface_G_analytic": interface_g_reference(),
    }
    q_audit = audit_q_artifact(q_artifact, fdtd_summary)
    prior_api = load_prior_api_evidence(api_evidence_path)
    live_api = (
        live_api_probe(args.lumerical_version, args.hide_gui)
        if args.live_api_probe
        else {"attempted": False, "status": "not_requested"}
    )

    diagonal_ok = bool(prior_api.get("diagonal_round_trip", False))
    if live_api.get("status") == "completed":
        diagonal_ok = bool(
            live_api.get("conductivity", {}).get("diagonal_round_trip", False)
        )
    interface_ok = bool(
        live_api.get("interface_G", {}).get("verified", False)
    )
    q_ok = bool(q_audit["compatible_without_clipping_or_rescaling"])
    solver_controls_ok = all(
        bool(item["solver_verified"]) for item in controls.values()
    )

    blockers: list[str] = []
    if not diagonal_ok:
        blockers.append(BLOCKED_ANISOTROPIC)
    if not q_ok:
        blockers.append(BLOCKED_Q_FOOTPRINT)
    if not interface_ok:
        blockers.append(BLOCKED_INTERFACE_G)
    if live_api.get("status") == BLOCKED_LICENSE:
        blockers.append(BLOCKED_LICENSE)
    blockers = list(dict.fromkeys(blockers))

    status = "READY_FOR_FULL_DEVICE" if (
        diagonal_ok and q_ok and interface_ok and solver_controls_ok
    ) else "BLOCKED"
    summary = {
        "status": status,
        "validated": False,
        "baseline_optical_commit": BASELINE_COMMIT,
        "scope": (
            "non-periodic isolated 2 um TaIrTe4 steady-state HEAT controls only"
        ),
        "prohibited_stages_executed": {
            "Maxwell": False,
            "transient": False,
            "PTE_current": False,
            "adjoint": False,
            "gradient": False,
            "optimization": False,
        },
        "blockers": blockers,
        "q_import_control": q_audit,
        "prior_v261_api_evidence": prior_api,
        "live_v261_api_probe": live_api,
        "analytic_controls": controls,
        "all_offline_analytic_references_passed": all(
            bool(item["offline_reference_passed"]) for item in controls.values()
        ),
        "all_required_solver_controls_passed": solver_controls_ok,
        "full_device_executed": False,
        "full_device_reason": (
            "All mandatory controls must pass before any full-device case."
        ),
        "sweep_contract": sweep_contract(),
        "generation_command": command,
        "output_directory": str(output),
    }
    write_json(output / "control_results.json", summary)

    manifest = raw_artifact_manifest(
        q_artifact=q_artifact,
        fdtd_summary=fdtd_summary,
        api_evidence=api_evidence_path,
        isotropic_evidence=isotropic_evidence_path,
        control_results=output / "control_results.json",
        current_command=command,
    )
    write_json(output / "artifact_manifest.json", manifest)
    print(json.dumps(jsonable(summary), indent=2), flush=True)
    return 0 if status == "READY_FOR_FULL_DEVICE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
