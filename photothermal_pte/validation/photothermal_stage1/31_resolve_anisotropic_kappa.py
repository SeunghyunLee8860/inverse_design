#!/usr/bin/env python3
"""Probe native v261 routes and validate a conservative anisotropic fallback."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import config_stage1 as config
from anisotropic_heat_fvm import solve_steady_diagonal_kappa
from lumerical_api import (
    flatten_strings,
    jsonable,
    open_device,
    select_installation,
    utc_timestamp,
    write_json,
)


KAPPA_W_MK = np.asarray([14.4, 3.8, 1.0], float)
ERROR_LIMIT = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("native-probe", "fvm-controls", "all"),
        default="all",
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--hide-gui", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def output_directory(explicit: str | None) -> Path:
    output = (
        Path(explicit).expanduser().resolve()
        if explicit
        else config.OUTPUT_ROOT
        / f"{utc_timestamp()}_resolve_anisotropic_kappa"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def lsf_tensor_round_trip(
    device: Any,
    label: str,
    expression: str,
    expected: np.ndarray,
) -> dict[str, Any]:
    material_name = f"lsf {label}"
    thermal_name = f"{material_name} thermal"
    device.addmodelmaterial()
    device.set("name", material_name)
    device.addhtmaterialproperty("Solid")
    device.set("name", thermal_name)
    path = f"::model::materials::{material_name}::{thermal_name}"
    variable = "anisotropy_route_return"
    script = (
        f'select("{path}");'
        'set("thermal conductivity.active model","constant");'
        f'set("thermal conductivity.constant",{expression});'
        f'{variable}=get("thermal conductivity.constant");'
    )
    item: dict[str, Any] = {
        "material_path": path,
        "lsf_expression": expression,
        "requested": expected.tolist(),
        "requested_shape": list(expected.shape),
    }
    try:
        device.eval(script)
        returned = np.asarray(device.getv(variable), float)
        item.update(
            {
                "returned": returned.tolist(),
                "returned_shape": list(returned.shape),
                "round_trip_passed": bool(
                    returned.shape == expected.shape
                    and np.allclose(returned, expected, rtol=0.0, atol=0.0)
                ),
            }
        )
    except Exception as exc:
        item.update(
            {
                "round_trip_passed": False,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            }
        )
    return item


def native_v261_probe(output: Path, *, hide: bool) -> dict[str, Any]:
    installation = select_installation("v261")
    result: dict[str, Any] = {
        "status": "BLOCKED_ANISOTROPIC_K_UNSUPPORTED",
        "passed": False,
        "official_thermal_property_types": [
            "Solid",
            "Solid Alloy",
            "Fluid",
        ],
        "probe_scope": (
            "fresh v261 DEVICE session; LSF-native tensor expressions, "
            "candidate hidden properties, and every HT database material"
        ),
        "lsf_tensor_round_trips": {},
        "hidden_property_candidates": {},
        "thermal_database_scan": {},
    }
    try:
        with open_device(installation, hide=hide) as device:
            result["v261_DEVICE_version"] = str(device.version())
            result["installation_root"] = str(installation.root)
            tensor_encodings = {
                "column_3x1": (
                    "[14.4;3.8;1.0]",
                    KAPPA_W_MK.reshape(3, 1),
                ),
                "row_1x3": (
                    "[14.4,3.8,1.0]",
                    KAPPA_W_MK.reshape(1, 3),
                ),
                "diagonal_3x3": (
                    "[14.4,0,0;0,3.8,0;0,0,1.0]",
                    np.diag(KAPPA_W_MK),
                ),
            }
            for label, (expression, expected) in tensor_encodings.items():
                result["lsf_tensor_round_trips"][label] = (
                    lsf_tensor_round_trip(
                        device, label, expression, expected
                    )
                )

            device.addmodelmaterial()
            device.set("name", "hidden property probe")
            device.addhtmaterialproperty("Solid")
            device.set("name", "hidden property probe thermal")
            for property_name, value in {
                "anisotropy": "diagonal",
                "thermal conductivity.anisotropy": "diagonal",
                "thermal conductivity.constant.x": 14.4,
                "thermal conductivity.constant.y": 3.8,
                "thermal conductivity.constant.z": 1.0,
                "thermal conductivity.constant.kx": 14.4,
                "thermal conductivity.constant.ky": 3.8,
                "thermal conductivity.constant.kz": 1.0,
                "thermal conductivity.constant.xx": 14.4,
                "thermal conductivity.constant.yy": 3.8,
                "thermal conductivity.constant.zz": 1.0,
            }.items():
                try:
                    device.set(property_name, value)
                    result["hidden_property_candidates"][property_name] = {
                        "write_succeeded": True,
                        "readback": jsonable(device.get(property_name)),
                    }
                except Exception as exc:
                    result["hidden_property_candidates"][property_name] = {
                        "write_succeeded": False,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    }

            device.addmodelmaterial()
            device.set("name", "HT database list probe")
            database_names = flatten_strings(
                device.addmaterialproperties("HT")
            )
            result["thermal_database_material_count"] = len(database_names)
            result["thermal_database_material_names"] = database_names
            nonscalar: dict[str, Any] = {}
            scan_errors: dict[str, str] = {}
            scalar_count = 0
            for index, database_name in enumerate(database_names):
                model_name = f"HT scan {index}"
                try:
                    device.addmodelmaterial()
                    device.set("name", model_name)
                    device.addmaterialproperties("HT", database_name)
                    value = np.asarray(
                        device.get("thermal conductivity.constant"), float
                    )
                    if value.size == 1:
                        scalar_count += 1
                    else:
                        nonscalar[database_name] = {
                            "shape": list(value.shape),
                            "value": value.tolist(),
                        }
                except Exception as exc:
                    scan_errors[database_name] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            result["thermal_database_scan"] = {
                "scalar_conductivity_material_count": scalar_count,
                "nonscalar_conductivity_materials": nonscalar,
                "errors": scan_errors,
            }
            result["passed"] = bool(
                any(
                    item["round_trip_passed"]
                    for item in result["lsf_tensor_round_trips"].values()
                )
                or nonscalar
                or any(
                    item["write_succeeded"]
                    for item in result["hidden_property_candidates"].values()
                )
            )
            if result["passed"]:
                result["status"] = "NATIVE_ANISOTROPIC_ROUTE_CANDIDATE_FOUND"
    except Exception as exc:
        result.update(
            {
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    write_json(output / "native_v261_anisotropy_probe.json", result)
    return result


def directional_control(
    output: Path, *, axis: str, axis_index: int
) -> dict[str, Any]:
    case_dir = output / f"fvm_{axis}"
    case_dir.mkdir(parents=True, exist_ok=True)
    length_m = 2.0e-6
    cross_m = 1.0e-6
    active_cells = 24
    cross_cells = 8
    counts = [cross_cells, cross_cells, cross_cells]
    spans = [cross_m, cross_m, cross_m]
    counts[axis_index] = active_cells
    spans[axis_index] = length_m
    edges = [
        np.linspace(-0.5 * span, 0.5 * span, count + 1)
        for span, count in zip(spans, counts)
    ]
    shape = tuple(counts)
    kappa = np.broadcast_to(KAPPA_W_MK, (*shape, 3)).copy()
    cold_K, hot_K = 300.0, 310.0
    result = solve_steady_diagonal_kappa(
        x_edges_m=edges[0],
        y_edges_m=edges[1],
        z_edges_m=edges[2],
        kappa_W_mK=kappa,
        dirichlet_temperature_K={
            f"{axis}_min": cold_K,
            f"{axis}_max": hot_K,
        },
    )
    centers = 0.5 * (
        edges[axis_index][:-1] + edges[axis_index][1:]
    )
    transverse_axes = tuple(
        index for index in range(3) if index != axis_index
    )
    profile = np.mean(result.temperature_K, axis=transverse_axes)
    exact_profile = cold_K + (hot_K - cold_K) * (
        centers - edges[axis_index][0]
    ) / length_m
    expected_flux = (
        KAPPA_W_MK[axis_index] * (hot_K - cold_K) / length_m
    )
    area_m2 = cross_m**2
    numerical_flux = 0.5 * (
        abs(result.boundary_power_out_W[f"{axis}_min"])
        + abs(result.boundary_power_out_W[f"{axis}_max"])
    ) / area_m2
    flux_error = abs(numerical_flux - expected_flux) / expected_flux
    profile_error = float(
        np.max(np.abs(profile - exact_profile)) / (hot_K - cold_K)
    )
    passed = bool(
        flux_error < ERROR_LIMIT
        and profile_error < ERROR_LIMIT
        and result.energy_balance_relative_error < ERROR_LIMIT
        and result.linear_residual_relative < 1.0e-9
    )
    case = {
        "case_id": f"fvm_{axis}",
        "status": "PASSED" if passed else "FAILED",
        "passed": passed,
        "axis": axis,
        "requested_tensor_W_mK": KAPPA_W_MK.tolist(),
        "expected_kappa_W_mK": float(KAPPA_W_MK[axis_index]),
        "effective_kappa_W_mK": (
            numerical_flux * length_m / (hot_K - cold_K)
        ),
        "analytic_heat_flux_W_m2": expected_flux,
        "numerical_heat_flux_W_m2": numerical_flux,
        "heat_flux_relative_error": flux_error,
        "temperature_profile_max_relative_error": profile_error,
        "energy_balance_relative_error": (
            result.energy_balance_relative_error
        ),
        "linear_residual_relative": result.linear_residual_relative,
        "boundary_power_out_W": result.boundary_power_out_W,
        "grid_shape": list(shape),
        "solver": result.solver,
        "iterations": result.iterations,
    }
    np.savez_compressed(
        case_dir / "temperature_profile.npz",
        x_edges_m=edges[0],
        y_edges_m=edges[1],
        z_edges_m=edges[2],
        kappa_diagonal_W_mK=KAPPA_W_MK,
        temperature_3d_K=result.temperature_K,
        coordinate_m=centers,
        temperature_K=profile,
        exact_temperature_K=exact_profile,
    )
    write_json(case_dir / "case_result.json", case)
    return case


def fvm_controls(output: Path) -> dict[str, Any]:
    cases = [
        directional_control(output, axis=axis, axis_index=index)
        for index, axis in enumerate(("x", "y", "z"))
    ]
    passed = all(case["passed"] for case in cases)
    result = {
        "status": (
            "VALIDATED_DIAGONAL_KAPPA_FVM_CONTROLS"
            if passed
            else "FAILED_DIAGONAL_KAPPA_FVM_CONTROLS"
        ),
        "passed": passed,
        "solver_scope": (
            "independent conservative Python finite-volume solver; "
            "not a v261 HEAT result"
        ),
        "discretization": (
            "cell-centered Cartesian finite volume with exact half-cell "
            "series resistance and adiabatic unspecified boundaries"
        ),
        "requested_tensor_W_mK": KAPPA_W_MK.tolist(),
        "isotropic_average_used": False,
        "cases": cases,
    }
    write_json(output / "anisotropic_kappa_fvm_controls.json", result)
    return result


def main() -> int:
    args = parse_args()
    output = output_directory(args.output_dir)
    result: dict[str, Any] = {
        "phase": args.phase,
        "generation_command": shlex.join([sys.executable, *sys.argv]),
        "full_device_HEAT_executed": False,
    }
    if args.phase in ("native-probe", "all"):
        result["native_v261_probe"] = native_v261_probe(
            output, hide=args.hide_gui
        )
    if args.phase in ("fvm-controls", "all"):
        result["fvm_controls"] = fvm_controls(output)
    if args.phase == "native-probe":
        result["status"] = result["native_v261_probe"]["status"]
    elif args.phase == "fvm-controls":
        result["status"] = result["fvm_controls"]["status"]
    else:
        result["status"] = (
            "VALIDATED_ANISOTROPIC_KAPPA_FALLBACK"
            if result["fvm_controls"]["passed"]
            else "FAILED_ANISOTROPIC_KAPPA_FALLBACK"
        )
    write_json(output / "anisotropic_kappa_resolution.json", result)
    print(json.dumps(jsonable(result), indent=2))
    return 0 if result.get("fvm_controls", {}).get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
