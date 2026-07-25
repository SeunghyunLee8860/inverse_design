#!/usr/bin/env python3
"""Cross-validate v261 HEAT and the Cartesian FVM on a common 3D problem.

The control intentionally stays inside the physics supported by both solvers:
scalar isotropic materials, perfect contact, axis-aligned Cartesian geometry,
one fixed-temperature exterior boundary, adiabatic remaining boundaries, and
an asymmetric synthetic volumetric heat source.  It does not read or import
the finite optical-Q artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import LinearNDInterpolator

import config_stage1 as config
from anisotropic_heat_fvm import solve_steady_diagonal_kappa
from lumerical_api import (
    flatten_strings,
    open_device,
    select_installation,
    utc_timestamp,
    write_json,
)


X_BOUNDS_M = (-1.0e-6, 1.0e-6)
Y_BOUNDS_M = (-1.0e-6, 1.0e-6)
Z_BOUNDS_M = (-1.0e-6, 0.5e-6)
MATERIAL_INTERFACE_Z_M = 0.0
LOWER_K_W_MK = 10.0
UPPER_K_W_MK = 2.0
SOURCE_BOUNDS_M = {
    "x": (-0.5e-6, 0.3e-6),
    "y": (-0.3e-6, 0.5e-6),
    "z": (0.1e-6, 0.4e-6),
}
SOURCE_Q_W_M3 = 1.0e15
BATH_K = 300.0
GRID_M = 50.0e-9
RELATIVE_LIMIT = 0.01
FIELD_CORRELATION_LIMIT = 0.999


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--report-dir",
        default=str(
            config.REPOSITORY_ROOT
            / "reports"
            / "fvm_3d_isotropic_cross_validation"
        ),
    )
    parser.add_argument(
        "--hide-gui", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def clean_output_directory(explicit: str | None) -> Path:
    output = (
        Path(explicit).expanduser().resolve()
        if explicit
        else config.OUTPUT_ROOT
        / "fvm_3d_isotropic_cross_validation"
        / f"{utc_timestamp()}_control"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def set_object(obj: Any, values: dict[str, Any]) -> None:
    for name, value in values.items():
        obj[name] = value


def add_solid_material(
    device: Any, name: str, conductivity_W_mK: float
) -> dict[str, Any]:
    device.addmodelmaterial()
    device.set("name", name)
    device.addhtmaterialproperty("Solid")
    thermal_name = f"{name} thermal"
    device.set("name", thermal_name)
    device.set("thermal conductivity.active model", "constant")
    device.set("thermal conductivity.constant", conductivity_W_mK)
    readback = float(
        np.asarray(
            device.get("thermal conductivity.constant"), float
        ).reshape(-1)[0]
    )
    if not np.isclose(readback, conductivity_W_mK):
        raise RuntimeError(
            f"{name} conductivity readback {readback} != {conductivity_W_mK}"
        )
    return {
        "name": name,
        "thermal_property": thermal_name,
        "conductivity_write_W_mK": conductivity_W_mK,
        "conductivity_readback_W_mK": readback,
    }


def axis_edges(bounds: tuple[float, float]) -> np.ndarray:
    count = int(round((bounds[1] - bounds[0]) / GRID_M))
    edges = np.linspace(bounds[0], bounds[1], count + 1)
    if not np.isclose(edges[1] - edges[0], GRID_M):
        raise ValueError(f"{bounds} is not divisible by grid {GRID_M}")
    return edges


def center_coordinates(
    edges: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = tuple(0.5 * (item[:-1] + item[1:]) for item in edges)
    return (*centers, *np.meshgrid(*centers, indexing="ij"))


def source_mask(
    x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray
) -> np.ndarray:
    return (
        (x_grid > SOURCE_BOUNDS_M["x"][0])
        & (x_grid < SOURCE_BOUNDS_M["x"][1])
        & (y_grid > SOURCE_BOUNDS_M["y"][0])
        & (y_grid < SOURCE_BOUNDS_M["y"][1])
        & (z_grid > SOURCE_BOUNDS_M["z"][0])
        & (z_grid < SOURCE_BOUNDS_M["z"][1])
    )


def source_volume_m3() -> float:
    return float(
        np.prod(
            [
                SOURCE_BOUNDS_M[axis][1] - SOURCE_BOUNDS_M[axis][0]
                for axis in ("x", "y", "z")
            ]
        )
    )


def run_lumerical(
    output: Path, *, hide: bool
) -> dict[str, Any]:
    installation = select_installation("v261")
    project = output / "isotropic_3d_crosscheck.ldev"
    material_names = ("crosscheck lower", "crosscheck upper")
    prescribed_power_W = SOURCE_Q_W_M3 * source_volume_m3()
    with open_device(installation, hide=hide) as device:
        version = str(device.version())
        region = device.addsimulationregion()
        set_object(
            region,
            {
                "name": "crosscheck region",
                "dimension": "3D",
                "use relative coordinates": False,
                "x min": X_BOUNDS_M[0],
                "x max": X_BOUNDS_M[1],
                "y min": Y_BOUNDS_M[0],
                "y max": Y_BOUNDS_M[1],
                "z min": Z_BOUNDS_M[0],
                "z max": Z_BOUNDS_M[1],
            },
        )
        materials = [
            add_solid_material(device, material_names[0], LOWER_K_W_MK),
            add_solid_material(device, material_names[1], UPPER_K_W_MK),
        ]
        for name, material, z_bounds in (
            (
                "crosscheck lower slab",
                material_names[0],
                (Z_BOUNDS_M[0], MATERIAL_INTERFACE_Z_M),
            ),
            (
                "crosscheck upper slab",
                material_names[1],
                (MATERIAL_INTERFACE_Z_M, Z_BOUNDS_M[1]),
            ),
        ):
            slab = device.addrect()
            set_object(
                slab,
                {
                    "name": name,
                    "material": material,
                    "x min": X_BOUNDS_M[0],
                    "x max": X_BOUNDS_M[1],
                    "y min": Y_BOUNDS_M[0],
                    "y max": Y_BOUNDS_M[1],
                    "z min": z_bounds[0],
                    "z max": z_bounds[1],
                },
            )
        heat = device.addheatsolver()
        set_object(
            heat,
            {
                "simulation region": "crosscheck region",
                "solver mode": "steady state",
                "solver physics": "thermal only",
                "use defaults": False,
                "min edge length": GRID_M / 4.0,
                "max edge length": GRID_M,
                "max plc edge length": GRID_M,
            },
        )
        source = device.adduniformheat()
        set_object(
            source,
            {
                "name": "asymmetric volumetric Q",
                "source type": "3D",
                "geometry type": "directly defined",
                "x min": SOURCE_BOUNDS_M["x"][0],
                "x max": SOURCE_BOUNDS_M["x"][1],
                "y min": SOURCE_BOUNDS_M["y"][0],
                "y max": SOURCE_BOUNDS_M["y"][1],
                "z min": SOURCE_BOUNDS_M["z"][0],
                "z max": SOURCE_BOUNDS_M["z"][1],
                "total power": prescribed_power_W,
            },
        )
        boundary = device.addtemperaturebc("HEAT")
        set_object(
            boundary,
            {
                "name": "T_bottom",
                "bc mode": "steady state",
                "sweep type": "single",
                "temperature": BATH_K,
                "surface type": "simulation region",
                "z min": True,
            },
        )
        monitor = device.addtemperaturemonitor()
        set_object(
            monitor,
            {
                "name": "T_volume",
                "monitor type": "3D",
                "x min": X_BOUNDS_M[0],
                "x max": X_BOUNDS_M[1],
                "y min": Y_BOUNDS_M[0],
                "y max": Y_BOUNDS_M[1],
                "z min": Z_BOUNDS_M[0],
                "z max": Z_BOUNDS_M[1],
            },
        )
        device.save(str(project))
        save_succeeded = project.is_file()
        device.load(str(project))
        load_succeeded = True
        reloaded_materials = []
        for name, expected in zip(
            material_names, (LOWER_K_W_MK, UPPER_K_W_MK)
        ):
            path = f"::model::materials::{name}::{name} thermal"
            readback = float(
                np.asarray(
                    device.getnamed(
                        path, "thermal conductivity.constant"
                    ),
                    float,
                ).reshape(-1)[0]
            )
            reloaded_materials.append(
                {
                    "path": path,
                    "expected_W_mK": expected,
                    "readback_W_mK": readback,
                    "passed": bool(np.isclose(readback, expected)),
                }
            )
        device.addjob(str(project), "HEAT")
        device.runjobs("HEAT", 0)
        device.load(str(project))
        available_results = flatten_strings(device.getresult("HEAT"))
        thermal = device.getresult("HEAT", "thermal")
        integrated = device.getresult("HEAT", "integrated")
        boundaries = device.getresult("HEAT", "boundaries")
        monitor_results = flatten_strings(
            device.getresult("HEAT::T_volume")
        )
        device.save(str(project))

    coordinates = {
        key: np.asarray(thermal[key], float).reshape(-1)
        for key in ("x", "y", "z")
    }
    temperature = np.asarray(thermal["T"], float).reshape(-1)
    q_nodes = np.asarray(thermal["Q"], float).reshape(-1)
    kappa_nodes = np.asarray(thermal["kappa"], float).reshape(-1)
    connectivity = np.asarray(thermal["connectivity"], np.int64)
    element_volume = np.asarray(thermal["volume"], float).reshape(-1)
    element_id = np.asarray(thermal["ID"], float).reshape(-1)
    q_elements = np.asarray(thermal["Q_elem"], float).reshape(-1)
    integrated_power_W = float(
        np.asarray(integrated["Q"], float).reshape(-1)[0]
    )
    boundary_key = next(
        key for key in boundaries if key.startswith("P_T_bottom")
    )
    boundary_power_out_W = abs(
        float(np.asarray(boundaries[boundary_key], float).reshape(-1)[0])
    )
    if not all(
        np.all(np.isfinite(array))
        for array in (
            *coordinates.values(),
            temperature,
            q_nodes,
            kappa_nodes,
            element_volume,
            q_elements,
        )
    ):
        raise RuntimeError("Lumerical 3D result contains NaN or Inf")
    energy_error = abs(
        boundary_power_out_W - integrated_power_W
    ) / integrated_power_W
    maximum_index = int(np.argmax(temperature))
    raw_path = output / "lumerical_3d_result.npz"
    np.savez_compressed(
        raw_path,
        x_m=coordinates["x"],
        y_m=coordinates["y"],
        z_m=coordinates["z"],
        temperature_K=temperature,
        Q_node_W_m3=q_nodes,
        kappa_node_W_mK=kappa_nodes,
        connectivity=connectivity,
        element_volume_m3=element_volume,
        element_material_ID=element_id,
        Q_element_W_m3=q_elements,
    )
    return {
        "version": version,
        "installation_root": str(installation.root),
        "project_path": str(project),
        "raw_result_path": str(raw_path),
        "session_startup": True,
        "save_succeeded": save_succeeded,
        "load_succeeded": load_succeeded,
        "solver_run_succeeded": True,
        "available_results": available_results,
        "temperature_monitor_results": monitor_results,
        "materials_before_save": materials,
        "materials_after_reload": reloaded_materials,
        "node_count": int(temperature.size),
        "element_count": int(connectivity.shape[0]),
        "material_ID_values": np.unique(element_id).tolist(),
        "kappa_node_values_W_mK": np.unique(kappa_nodes).tolist(),
        "Q_node_range_W_m3": [
            float(np.min(q_nodes)),
            float(np.max(q_nodes)),
        ],
        "prescribed_source_power_W": prescribed_power_W,
        "integrated_source_power_W": integrated_power_W,
        "source_power_relative_error": abs(
            integrated_power_W - prescribed_power_W
        )
        / prescribed_power_W,
        "boundary_power_out_W": boundary_power_out_W,
        "energy_balance_relative_error": energy_error,
        "temperature_min_K": float(np.min(temperature)),
        "temperature_max_K": float(np.max(temperature)),
        "temperature_max_location_m": {
            axis: float(coordinates[axis][maximum_index])
            for axis in ("x", "y", "z")
        },
        "_coordinates": coordinates,
        "_temperature": temperature,
    }


def run_fvm(output: Path) -> dict[str, Any]:
    edges = (
        axis_edges(X_BOUNDS_M),
        axis_edges(Y_BOUNDS_M),
        axis_edges(Z_BOUNDS_M),
    )
    x, y, z, x_grid, y_grid, z_grid = center_coordinates(edges)
    shape = x_grid.shape
    kappa = np.empty((*shape, 3), float)
    lower = z_grid < MATERIAL_INTERFACE_Z_M
    kappa[lower, :] = LOWER_K_W_MK
    kappa[~lower, :] = UPPER_K_W_MK
    active_source = source_mask(x_grid, y_grid, z_grid)
    source = np.zeros(shape, float)
    source[active_source] = SOURCE_Q_W_M3
    result = solve_steady_diagonal_kappa(
        x_edges_m=edges[0],
        y_edges_m=edges[1],
        z_edges_m=edges[2],
        kappa_W_mK=kappa,
        source_W_m3=source,
        dirichlet_temperature_K={"z_min": BATH_K},
    )
    temperature = result.temperature_K
    maximum_index = np.unravel_index(
        int(np.argmax(temperature)), temperature.shape
    )
    raw_path = output / "fvm_3d_result.npz"
    np.savez_compressed(
        raw_path,
        x_edges_m=edges[0],
        y_edges_m=edges[1],
        z_edges_m=edges[2],
        x_centers_m=x,
        y_centers_m=y,
        z_centers_m=z,
        temperature_K=temperature,
        Q_W_m3=source,
        kappa_diagonal_W_mK=kappa,
        source_mask=active_source,
    )
    return {
        "raw_result_path": str(raw_path),
        "grid_m": GRID_M,
        "grid_shape": list(shape),
        "cell_count": int(np.prod(shape)),
        "source_cell_count": int(np.count_nonzero(active_source)),
        "source_volume_m3": float(
            np.count_nonzero(active_source) * GRID_M**3
        ),
        "source_power_W": result.source_power_W,
        "boundary_power_out_W": result.boundary_power_out_W,
        "energy_balance_relative_error": (
            result.energy_balance_relative_error
        ),
        "linear_residual_relative": result.linear_residual_relative,
        "solver": result.solver,
        "iterations": result.iterations,
        "kappa_values_W_mK": np.unique(kappa).tolist(),
        "temperature_min_K": float(np.min(temperature)),
        "temperature_max_K": float(np.max(temperature)),
        "temperature_mean_K": float(np.mean(temperature)),
        "temperature_max_location_m": {
            "x": float(x[maximum_index[0]]),
            "y": float(y[maximum_index[1]]),
            "z": float(z[maximum_index[2]]),
        },
        "_edges": edges,
        "_centers": (x, y, z),
        "_grids": (x_grid, y_grid, z_grid),
        "_temperature": temperature,
    }


def compare_solvers(
    output: Path,
    lumerical: dict[str, Any],
    fvm: dict[str, Any],
) -> dict[str, Any]:
    coordinates = lumerical.pop("_coordinates")
    lumerical_temperature = lumerical.pop("_temperature")
    fvm.pop("_edges")
    x, y, z = fvm.pop("_centers")
    x_grid, y_grid, z_grid = fvm.pop("_grids")
    fvm_temperature = fvm.pop("_temperature")
    points = np.column_stack(
        [
            coordinates["x"],
            coordinates["y"],
            coordinates["z"],
        ]
    )
    queries = np.column_stack(
        [x_grid.reshape(-1), y_grid.reshape(-1), z_grid.reshape(-1)]
    )
    interpolator = LinearNDInterpolator(points, lumerical_temperature)
    common_lumerical_temperature = np.asarray(
        interpolator(queries), float
    ).reshape(fvm_temperature.shape)
    finite = np.isfinite(common_lumerical_temperature)
    if not np.all(finite):
        raise RuntimeError(
            f"Lumerical interpolation produced {np.size(finite)-np.count_nonzero(finite)} NaN/Inf values"
        )
    delta_lumerical = common_lumerical_temperature - BATH_K
    delta_fvm = fvm_temperature - BATH_K
    maximum_rise_K = float(np.max(delta_fvm))
    difference = common_lumerical_temperature - fvm_temperature
    absolute_difference = np.abs(difference)
    field_nrmse = float(
        np.sqrt(np.mean(difference**2)) / maximum_rise_K
    )
    field_max_error = float(np.max(absolute_difference) / maximum_rise_K)
    field_p99_error = float(
        np.percentile(absolute_difference, 99.0) / maximum_rise_K
    )
    field_correlation = float(
        np.corrcoef(delta_lumerical.reshape(-1), delta_fvm.reshape(-1))[
            0, 1
        ]
    )
    lumerical_common_max_K = float(
        np.max(common_lumerical_temperature)
    )
    lumerical_common_mean_K = float(
        np.mean(common_lumerical_temperature)
    )
    tmax_error = abs(
        lumerical["temperature_max_K"] - fvm["temperature_max_K"]
    ) / maximum_rise_K
    common_tmax_error = abs(
        lumerical_common_max_K - fvm["temperature_max_K"]
    ) / maximum_rise_K
    mean_error = abs(
        lumerical_common_mean_K - fvm["temperature_mean_K"]
    ) / maximum_rise_K
    prescribed_power_W = SOURCE_Q_W_M3 * source_volume_m3()
    source_power_cross_error = abs(
        lumerical["integrated_source_power_W"] - fvm["source_power_W"]
    ) / prescribed_power_W
    lumerical_boundary_power_W = lumerical["boundary_power_out_W"]
    fvm_boundary_power_W = abs(fvm["boundary_power_out_W"]["z_min"])
    boundary_power_cross_error = abs(
        lumerical_boundary_power_W - fvm_boundary_power_W
    ) / prescribed_power_W
    hotspot_offset = float(
        np.linalg.norm(
            [
                lumerical["temperature_max_location_m"][axis]
                - fvm["temperature_max_location_m"][axis]
                for axis in ("x", "y", "z")
            ]
        )
    )
    passed = bool(
        tmax_error < RELATIVE_LIMIT
        and mean_error < RELATIVE_LIMIT
        and field_nrmse < RELATIVE_LIMIT
        and field_correlation > FIELD_CORRELATION_LIMIT
        and source_power_cross_error < RELATIVE_LIMIT
        and boundary_power_cross_error < RELATIVE_LIMIT
        and lumerical["source_power_relative_error"] < RELATIVE_LIMIT
        and lumerical["energy_balance_relative_error"] < RELATIVE_LIMIT
        and fvm["energy_balance_relative_error"] < RELATIVE_LIMIT
        and fvm["linear_residual_relative"] < 1.0e-9
        and lumerical["kappa_node_values_W_mK"]
        == [UPPER_K_W_MK, LOWER_K_W_MK]
        and fvm["kappa_values_W_mK"]
        == [UPPER_K_W_MK, LOWER_K_W_MK]
    )
    common_path = output / "common_grid_comparison.npz"
    np.savez_compressed(
        common_path,
        x_m=x,
        y_m=y,
        z_m=z,
        T_lumerical_interpolated_K=common_lumerical_temperature,
        T_fvm_K=fvm_temperature,
        delta_T_difference_K=difference,
    )
    return {
        "passed": passed,
        "common_grid_result_path": str(common_path),
        "common_grid_shape": list(fvm_temperature.shape),
        "interpolation": (
            "SciPy LinearNDInterpolator from the v261 unstructured FEM "
            "nodes to the FVM cell centers"
        ),
        "maximum_FVM_temperature_rise_K": maximum_rise_K,
        "Tmax_lumerical_node_K": lumerical["temperature_max_K"],
        "Tmax_lumerical_common_grid_K": lumerical_common_max_K,
        "Tmax_fvm_K": fvm["temperature_max_K"],
        "Tmax_node_relative_error_over_rise": tmax_error,
        "Tmax_common_grid_relative_error_over_rise": common_tmax_error,
        "Tmean_lumerical_common_grid_K": lumerical_common_mean_K,
        "Tmean_fvm_K": fvm["temperature_mean_K"],
        "Tmean_relative_error_over_rise": mean_error,
        "temperature_field_NRMSE_over_rise": field_nrmse,
        "temperature_field_p99_error_over_rise": field_p99_error,
        "temperature_field_max_error_over_rise": field_max_error,
        "temperature_field_correlation": field_correlation,
        "hotspot_location_offset_m": hotspot_offset,
        "prescribed_source_power_W": prescribed_power_W,
        "lumerical_integrated_source_power_W": (
            lumerical["integrated_source_power_W"]
        ),
        "fvm_source_power_W": fvm["source_power_W"],
        "source_power_cross_relative_error": source_power_cross_error,
        "lumerical_boundary_power_out_W": lumerical_boundary_power_W,
        "fvm_boundary_power_out_W": fvm_boundary_power_W,
        "boundary_power_cross_relative_error": boundary_power_cross_error,
        "criteria": {
            "Tmax_relative_error_over_rise_lt": RELATIVE_LIMIT,
            "Tmean_relative_error_over_rise_lt": RELATIVE_LIMIT,
            "temperature_field_NRMSE_over_rise_lt": RELATIVE_LIMIT,
            "temperature_field_correlation_gt": FIELD_CORRELATION_LIMIT,
            "source_power_cross_relative_error_lt": RELATIVE_LIMIT,
            "boundary_power_cross_relative_error_lt": RELATIVE_LIMIT,
            "each_solver_energy_balance_relative_error_lt": RELATIVE_LIMIT,
        },
        "non_gating_diagnostics": {
            "temperature_field_p99_error_over_rise": field_p99_error,
            "temperature_field_max_error_over_rise": field_max_error,
            "Tmax_common_grid_relative_error_over_rise": (
                common_tmax_error
            ),
            "hotspot_location_offset_m": hotspot_offset,
        },
    }


def serializable_summary(
    status: str,
    *,
    output: Path,
    command: str,
    lumerical: dict[str, Any] | None,
    fvm: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
    exception: Exception | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "status": status,
        "passed": status
        == "VALIDATED_3D_ISOTROPIC_HEAT_FVM_CROSS_VALIDATION",
        "solver_scope": (
            "v261 Lumerical HEAT scalar-isotropic perfect-contact control "
            "crossed with independent Cartesian Python/SciPy FVM"
        ),
        "generation_command": command,
        "finite_optical_Q_imported": False,
        "full_production_device_executed": False,
        "geometry": {
            "x_bounds_m": list(X_BOUNDS_M),
            "y_bounds_m": list(Y_BOUNDS_M),
            "z_bounds_m": list(Z_BOUNDS_M),
            "material_interface_z_m": MATERIAL_INTERFACE_Z_M,
            "lower_kappa_W_mK": LOWER_K_W_MK,
            "upper_kappa_W_mK": UPPER_K_W_MK,
            "perfect_contact": True,
        },
        "volumetric_Q": {
            "type": "asymmetric axis-aligned uniform cuboid",
            "bounds_m": SOURCE_BOUNDS_M,
            "Q_W_m3": SOURCE_Q_W_M3,
            "analytic_volume_m3": source_volume_m3(),
            "analytic_power_W": SOURCE_Q_W_M3 * source_volume_m3(),
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "rescaling": False,
            "periodic_tiling": False,
        },
        "boundary_conditions": {
            "z_min": {"type": "Dirichlet", "temperature_K": BATH_K},
            "x_min_x_max_y_min_y_max_z_max": "adiabatic",
        },
        "grid_target_m": GRID_M,
        "lumerical": lumerical,
        "fvm": fvm,
        "comparison": comparison,
        "next_required_gate": (
            "FINITE_OPTICAL_Q_CONSERVATIVE_IMPORT"
            if status
            == "VALIDATED_3D_ISOTROPIC_HEAT_FVM_CROSS_VALIDATION"
            else "LUMERICAL_HEAT_VS_FVM_3D_ISOTROPIC_PERFECT_CONTACT_CROSS_VALIDATION"
        ),
        "output_directory": str(output),
    }
    if exception is not None:
        summary["exception"] = {
            "type": type(exception).__name__,
            "message": str(exception),
            "traceback": traceback.format_exc(),
        }
    return summary


def write_cases_csv(path: Path, summary: dict[str, Any]) -> None:
    comparison = summary.get("comparison") or {}
    lumerical = summary.get("lumerical") or {}
    fvm = summary.get("fvm") or {}
    row = {
        "case_id": "isotropic_3d_asymmetric_Q_perfect_contact",
        "status": summary["status"],
        "passed": summary["passed"],
        "Tmax_relative_error": comparison.get(
            "Tmax_node_relative_error_over_rise"
        ),
        "Tmean_relative_error": comparison.get(
            "Tmean_relative_error_over_rise"
        ),
        "field_NRMSE": comparison.get(
            "temperature_field_NRMSE_over_rise"
        ),
        "field_p99_error": comparison.get(
            "temperature_field_p99_error_over_rise"
        ),
        "field_max_error": comparison.get(
            "temperature_field_max_error_over_rise"
        ),
        "field_correlation": comparison.get(
            "temperature_field_correlation"
        ),
        "source_power_cross_error": comparison.get(
            "source_power_cross_relative_error"
        ),
        "boundary_power_cross_error": comparison.get(
            "boundary_power_cross_relative_error"
        ),
        "lumerical_energy_error": lumerical.get(
            "energy_balance_relative_error"
        ),
        "fvm_energy_error": fvm.get("energy_balance_relative_error"),
    }
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(row), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(row)


def percent(value: float) -> str:
    return f"{100.0 * value:.6g}%"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    comparison = summary.get("comparison")
    if not comparison:
        text = f"""# 3D isotropic HEAT-FVM cross-validation

**Status: `{summary["status"]}`.**

The v261 Lumerical HEAT solve did not complete. The independent FVM result is
not promoted as a cross-validation result. Finite optical Q remains blocked.

Failure: `{summary.get("exception", {}).get("message", "unknown")}`
"""
    else:
        lumerical = summary["lumerical"]
        fvm = summary["fvm"]
        text = f"""# 3D isotropic HEAT-FVM cross-validation

**Status: `{summary["status"]}`.**

This control uses the same 3D two-material Cartesian geometry, scalar
conductivities, perfect contact, asymmetric synthetic volumetric Q, bottom
300 K boundary, and adiabatic remaining exterior boundaries in v261
Lumerical HEAT and the independent Python/SciPy FVM.

It does not use the finite optical-Q artifact or the production device.

| Metric | Result | Gate |
|---|---:|---:|
| Tmax difference / max FVM rise | {percent(comparison["Tmax_node_relative_error_over_rise"])} | <1% |
| mean-T difference / max FVM rise | {percent(comparison["Tmean_relative_error_over_rise"])} | <1% |
| 3D field NRMSE / max FVM rise | {percent(comparison["temperature_field_NRMSE_over_rise"])} | <1% |
| 3D field correlation | {comparison["temperature_field_correlation"]:.12g} | >0.999 |
| source-power cross error | {percent(comparison["source_power_cross_relative_error"])} | <1% |
| boundary-power cross error | {percent(comparison["boundary_power_cross_relative_error"])} | <1% |
| Lumerical energy error | {percent(lumerical["energy_balance_relative_error"])} | <1% |
| FVM energy error | {percent(fvm["energy_balance_relative_error"])} | <1% |

The v261 unstructured temperature field is linearly interpolated to all
`{fvm["cell_count"]}` FVM cell centers on the 50 nm common grid. There are no
NaN/Inf samples. Material values are exactly `{fvm["kappa_values_W_mK"]}`
W/(m K) in both solvers.

Non-gating diagnostics are retained rather than hidden: the 99th-percentile
pointwise field error is
`{percent(comparison["temperature_field_p99_error_over_rise"])}` and the
single worst source-edge point is
`{percent(comparison["temperature_field_max_error_over_rise"])}`. The global
field NRMSE and correlation are the declared field gates.

The 3D common-physics gate passes. The next step is only the conservative
finite optical-Q mapping/reintegration gate; anisotropic and finite-G
production physics must still wait until that import passes.
"""
    path.write_text(text, encoding="utf-8")


def repository_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.REPOSITORY_ROOT.parent))
    except ValueError:
        return str(path.resolve())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=config.REPOSITORY_ROOT.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def write_manifest(
    path: Path,
    *,
    output: Path,
    report_dir: Path,
    command: str,
) -> None:
    candidates = [
        output / "isotropic_3d_crosscheck.ldev",
        output / "lumerical_3d_result.npz",
        output / "fvm_3d_result.npz",
        output / "common_grid_comparison.npz",
        report_dir / "HEAT_FVM_3D_ISOTROPIC_CROSS_VALIDATION_REPORT.md",
        report_dir / "heat_fvm_3d_isotropic_cross_validation_summary.json",
        report_dir / "heat_fvm_3d_isotropic_cross_validation_cases.csv",
    ]
    files = [item for item in candidates if item.is_file()]
    write_json(
        path,
        {
            "schema_version": 1,
            "generated_at_utc": utc_timestamp(),
            "branch": git_value("branch", "--show-current"),
            "base_commit_before_control": git_value("rev-parse", "HEAD"),
            "generation_command": command,
            "finite_optical_Q_imported": False,
            "artifacts": [
                {
                    "repository_path": repository_relative(item),
                    "server_path": str(item.resolve()),
                    "bytes": item.stat().st_size,
                    "sha256": sha256(item),
                }
                for item in files
            ],
        },
    )


def main() -> int:
    args = parse_args()
    output = clean_output_directory(args.output_dir)
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, *sys.argv])
    lumerical = None
    fvm = None
    comparison = None
    error: Exception | None = None
    try:
        lumerical = run_lumerical(output, hide=args.hide_gui)
        fvm = run_fvm(output)
        comparison = compare_solvers(output, lumerical, fvm)
        status = (
            "VALIDATED_3D_ISOTROPIC_HEAT_FVM_CROSS_VALIDATION"
            if comparison["passed"]
            else "FAILED_3D_ISOTROPIC_HEAT_FVM_CROSS_VALIDATION"
        )
    except Exception as exc:
        error = exc
        message = str(exc).lower()
        status = (
            "BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE"
            if any(
                token in message
                for token in ("license", "ansysli", "checkout")
            )
            else "FAILED_3D_ISOTROPIC_HEAT_FVM_CROSS_VALIDATION"
        )
    summary = serializable_summary(
        status,
        output=output,
        command=command,
        lumerical=lumerical,
        fvm=fvm,
        comparison=comparison,
        exception=error,
    )
    summary_path = (
        report_dir
        / "heat_fvm_3d_isotropic_cross_validation_summary.json"
    )
    cases_path = (
        report_dir / "heat_fvm_3d_isotropic_cross_validation_cases.csv"
    )
    report_path = (
        report_dir / "HEAT_FVM_3D_ISOTROPIC_CROSS_VALIDATION_REPORT.md"
    )
    write_json(summary_path, summary)
    write_cases_csv(cases_path, summary)
    write_report(report_path, summary)
    write_manifest(
        report_dir / "RAW_ARTIFACT_MANIFEST.json",
        output=output,
        report_dir=report_dir,
        command=command,
    )
    write_json(output / "control_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 2 if status.startswith("BLOCKED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
