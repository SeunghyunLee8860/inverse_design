#!/usr/bin/env python3
"""Validate v261/v251 HEAT against a 3-D uniformly heated analytic slab."""

from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config_stage1 as config
from lumerical_api import (
    Stage1Error,
    flatten_strings,
    jsonable,
    open_device,
    select_installation,
    selected_properties,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumerical-version", choices=("auto", "v261", "v251"), default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hide-gui", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def set_object(obj: Any, values: dict[str, Any]) -> None:
    """Set SimObject properties by the exact names confirmed by the API probe."""
    for name, value in values.items():
        obj[name] = value


def add_constant_solid_material(device: Any, name: str, k_w_mk: float) -> dict[str, Any]:
    device.addmodelmaterial()
    device.set("name", name)
    device.addhtmaterialproperty("Solid")
    thermal_property_name = f"{name} thermal"
    device.set("name", thermal_property_name)
    # Steady state does not use mass density or specific heat.  They are
    # intentionally left unset instead of receiving arbitrary placeholders.
    device.set("thermal conductivity.active model", "constant")
    device.set("thermal conductivity.constant", float(k_w_mk))
    return {
        "model_material": name,
        "thermal_property": thermal_property_name,
        "thermal_conductivity_W_mK": float(k_w_mk),
        "mass_density": "not set; unused by steady-state equation",
        "specific_heat": "not set; unused by steady-state equation",
    }


def result_names(device: Any, object_name: str) -> list[str]:
    try:
        return flatten_strings(device.getresult(object_name))
    except Exception:
        return []


def get_result_candidate(device: Any, object_name: str, candidates: tuple[str, ...]):
    available = result_names(device, object_name)
    normalized = {name.lower(): name for name in available}
    errors = []
    for candidate in candidates:
        actual = normalized.get(candidate.lower(), candidate)
        try:
            return actual, device.getresult(object_name, actual), available
        except Exception as exc:
            errors.append(f"{actual}: {type(exc).__name__}: {exc}")
    raise Stage1Error(
        f"no usable result for {object_name}; available={available}; errors={errors}"
    )


def scalar_attribute(dataset: dict[str, Any], prefixes: tuple[str, ...]) -> tuple[str, float]:
    keys = list(dataset)
    for prefix in prefixes:
        for key in keys:
            if key.lower() == prefix.lower() or key.lower().startswith(prefix.lower()):
                value = np.asarray(dataset[key], float).reshape(-1)
                if value.size:
                    return key, float(value[0])
    raise Stage1Error(f"no scalar attribute matching {prefixes}; keys={keys}")


def extract_temperature_line(dataset: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "T" not in dataset:
        matches = [key for key in dataset if key.lower() in {"temperature", "t"}]
        if not matches:
            raise Stage1Error(f"temperature dataset has no T attribute; keys={list(dataset)}")
        t_key = matches[0]
    else:
        t_key = "T"
    if "z" not in dataset:
        raise Stage1Error(f"temperature line has no z coordinate; keys={list(dataset)}")
    z = np.asarray(dataset["z"], float).reshape(-1)
    temperature = np.asarray(dataset[t_key], float).squeeze()
    while temperature.ndim > 1:
        # All non-z monitor dimensions are singleton in this linear-z monitor.
        singleton = next((axis for axis, size in enumerate(temperature.shape) if size == 1), None)
        if singleton is None:
            break
        temperature = np.squeeze(temperature, axis=singleton)
    temperature = np.asarray(temperature, float).reshape(-1)
    if temperature.size != z.size:
        raise Stage1Error(f"T size {temperature.size} != z size {z.size}")
    order = np.argsort(z)
    z, temperature = z[order], temperature[order]
    if np.any(np.diff(z) <= 0):
        raise Stage1Error("temperature-monitor z coordinates are not strictly increasing")
    return z, temperature


def analytic_temperature(z: np.ndarray) -> np.ndarray:
    return config.BATH_TEMPERATURE_K + (
        config.ANALYTIC_Q_W_M3 / config.ANALYTIC_K_W_MK
    ) * (config.ANALYTIC_LENGTH_M * z - 0.5 * z**2)


def run_level(
    installation,
    output: Path,
    *,
    label: str,
    target_edge_m: float,
    hide: bool,
) -> dict[str, Any]:
    level_dir = output / "analytic_heat" / label
    level_dir.mkdir(parents=True, exist_ok=True)
    project_path = level_dir / "heat_analytic.ldev"
    volume = (
        config.ANALYTIC_X_SPAN_M
        * config.ANALYTIC_Y_SPAN_M
        * config.ANALYTIC_LENGTH_M
    )
    input_power = config.ANALYTIC_Q_W_M3 * volume

    with open_device(installation, hide=hide) as device:
        region = device.addsimulationregion()
        set_object(
            region,
            {
                "name": "stage1 region",
                "dimension": "3D",
                "use relative coordinates": False,
                "x": 0.0,
                "x span": config.ANALYTIC_X_SPAN_M,
                "y": 0.0,
                "y span": config.ANALYTIC_Y_SPAN_M,
                "z min": 0.0,
                "z max": config.ANALYTIC_LENGTH_M,
            },
        )
        material = add_constant_solid_material(
            device, "stage1 analytic material", config.ANALYTIC_K_W_MK
        )
        slab = device.addrect()
        set_object(
            slab,
            {
                "name": "analytic slab",
                "material": "stage1 analytic material",
                "x": 0.0,
                "x span": config.ANALYTIC_X_SPAN_M,
                "y": 0.0,
                "y span": config.ANALYTIC_Y_SPAN_M,
                "z min": 0.0,
                "z max": config.ANALYTIC_LENGTH_M,
            },
        )
        heat = device.addheatsolver()
        set_object(
            heat,
            {
                "simulation region": "stage1 region",
                "solver mode": "steady state",
                "solver physics": "thermal only",
                "use defaults": False,
                "min edge length": min(target_edge_m / 4.0, target_edge_m),
                "max edge length": target_edge_m,
                "max plc edge length": target_edge_m,
            },
        )
        source = device.adduniformheat()
        set_object(
            source,
            {
                "name": "uniform volumetric heat",
                "source type": "3D",
                "geometry type": "directly defined",
                "x": 0.0,
                "x span": config.ANALYTIC_X_SPAN_M,
                "y": 0.0,
                "y span": config.ANALYTIC_Y_SPAN_M,
                "z min": 0.0,
                "z max": config.ANALYTIC_LENGTH_M,
                "total power": input_power,
            },
        )
        boundary = device.addtemperaturebc("HEAT")
        set_object(
            boundary,
            {
                "name": "T_bottom",
                "bc mode": "steady state",
                "sweep type": "single",
                "temperature": config.BATH_TEMPERATURE_K,
                "surface type": "simulation region",
                "z min": True,
            },
        )
        temperature_monitor = device.addtemperaturemonitor()
        set_object(
            temperature_monitor,
            {
                "name": "T_centerline",
                "monitor type": "linear z",
                "x": 0.0,
                "y": 0.0,
                "z min": 0.0,
                "z max": config.ANALYTIC_LENGTH_M,
            },
        )
        flux_monitor = device.addheatfluxmonitor()
        set_object(
            flux_monitor,
            {
                "name": "bottom_flux",
                "monitor type": "2D z-normal",
                "x": 0.0,
                "x span": config.ANALYTIC_X_SPAN_M,
                "y": 0.0,
                "y span": config.ANALYTIC_Y_SPAN_M,
                "z": 0.0,
            },
        )
        mesh = device.addheatmesh()
        set_object(
            mesh,
            {
                "name": f"mesh_{label}",
                "geometry type": "directly defined",
                "max edge length": target_edge_m,
                "x": 0.0,
                "x span": config.ANALYTIC_X_SPAN_M,
                "y": 0.0,
                "y span": config.ANALYTIC_Y_SPAN_M,
                "z min": 0.0,
                "z max": config.ANALYTIC_LENGTH_M,
            },
        )
        device.save(str(project_path))
        device.addjob(str(project_path), "HEAT")
        device.runjobs("HEAT", 0)
        device.load(str(project_path))

        temp_result_name, temp_dataset, temp_available = get_result_candidate(
            device, "HEAT::T_centerline", ("temperature", "thermal")
        )
        boundary_result_name, boundaries, boundary_available = get_result_candidate(
            device, "HEAT", ("boundaries",)
        )
        z, temperature = extract_temperature_line(temp_dataset)
        power_key, power_value = scalar_attribute(
            boundaries, ("P_T_bottom", "P_T bottom", "P_")
        )
        device.save(str(project_path))

        probe_snapshot = {
            "simulation_region": selected_properties(device) if False else "see api_probe",
            "temperature_monitor_results": temp_available,
            "heat_solver_results": boundary_available,
            "temperature_result_name": temp_result_name,
            "boundary_result_name": boundary_result_name,
            "bottom_power_attribute": power_key,
        }

    exact = analytic_temperature(z)
    error = temperature - exact
    delta_exact = config.ANALYTIC_Q_W_M3 * config.ANALYTIC_LENGTH_M**2 / (
        2.0 * config.ANALYTIC_K_W_MK
    )
    delta_numeric = float(np.max(temperature) - config.BATH_TEMPERATURE_K)
    max_abs_error = float(np.max(np.abs(error)))
    relative_delta_error = abs(delta_numeric - delta_exact) / delta_exact
    nrmse = float(np.sqrt(np.mean(error**2)) / delta_exact)
    output_power = abs(float(power_value))
    energy_error = abs(output_power - input_power) / input_power
    passed = (
        relative_delta_error < config.ANALYTIC_RELATIVE_DT_MAX_LIMIT
        and nrmse < config.ANALYTIC_NRMSE_LIMIT
        and energy_error < config.ANALYTIC_ENERGY_ERROR_LIMIT
    )
    np.savez_compressed(
        level_dir / "analytic_temperature_profile.npz",
        z_m=z,
        temperature_K=temperature,
        exact_temperature_K=exact,
        error_K=error,
    )
    with (level_dir / "analytic_temperature_profile.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("z_m", "temperature_K", "exact_temperature_K", "error_K"))
        writer.writerows(zip(z, temperature, exact, error))
    write_json(level_dir / "result_probe.json", probe_snapshot)
    summary = {
        "mesh_label": label,
        "target_max_edge_length_m": target_edge_m,
        "material": material,
        "input_power_W": input_power,
        "bottom_outflow_power_W": output_power,
        "bottom_power_raw_signed_W": float(power_value),
        "bottom_power_attribute": power_key,
        "deltaT_max_exact_K": delta_exact,
        "deltaT_max_numeric_K": delta_numeric,
        "T_max_exact_K": config.BATH_TEMPERATURE_K + delta_exact,
        "T_max_numeric_K": float(np.max(temperature)),
        "maximum_absolute_error_K": max_abs_error,
        "relative_error_deltaT_max": relative_delta_error,
        "normalized_rmse": nrmse,
        "energy_error": energy_error,
        "pass": passed,
    }
    write_json(level_dir / "analytic_summary.json", summary)
    return {**summary, "z": z, "temperature": temperature, "exact": exact}


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    installation = select_installation(args.lumerical_version)
    labels = ("coarse", "medium", "fine")
    rows = []
    try:
        for label, edge in zip(labels, config.ANALYTIC_MESH_TARGETS_M):
            print(f"[analytic] {label}: max edge {edge:.3e} m", flush=True)
            rows.append(
                run_level(
                    installation, output, label=label, target_edge_m=edge, hide=args.hide_gui
                )
            )
    except Exception as exc:
        failure = {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "validated": False,
        }
        write_json(output / "analytic_heat" / "analytic_summary.json", failure)
        print(json.dumps(failure, indent=2))
        return 2

    convergence_path = output / "analytic_heat" / "mesh_convergence.csv"
    with convergence_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "mesh_label",
                "target_max_edge_length_m",
                "deltaT_max_numeric_K",
                "relative_error_deltaT_max",
                "normalized_rmse",
                "energy_error",
                "pass",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row["mesh_label"],
                    row["target_max_edge_length_m"],
                    row["deltaT_max_numeric_K"],
                    row["relative_error_deltaT_max"],
                    row["normalized_rmse"],
                    row["energy_error"],
                    row["pass"],
                )
            )

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for row in rows:
        ax.plot(row["z"] * 1e6, row["temperature"], marker="o", ms=3, label=row["mesh_label"])
    ax.plot(rows[-1]["z"] * 1e6, rows[-1]["exact"], "k--", label="analytic")
    ax.set(xlabel="z (um)", ylabel="Temperature (K)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "analytic_heat" / "analytic_temperature_profile.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    edge_nm = np.array([row["target_max_edge_length_m"] for row in rows]) * 1e9
    for ax, key, title in zip(
        axes,
        ("relative_error_deltaT_max", "normalized_rmse", "energy_error"),
        ("DeltaT max relative error", "Normalized RMSE", "Energy error"),
    ):
        ax.loglog(edge_nm, [row[key] for row in rows], "o-")
        ax.set(xlabel="Target max edge (nm)", title=title)
        ax.grid(True, which="both", alpha=0.3)
        ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(output / "analytic_heat" / "mesh_convergence.png", dpi=180)
    plt.close(fig)

    finest = rows[-1]
    validated = bool(finest["pass"])
    summary = {
        "status": "passed" if validated else "failed_acceptance_criteria",
        "selected_version": installation.version_key,
        "criteria": {
            "relative_error_deltaT_max_limit": config.ANALYTIC_RELATIVE_DT_MAX_LIMIT,
            "normalized_rmse_limit": config.ANALYTIC_NRMSE_LIMIT,
            "energy_error_limit": config.ANALYTIC_ENERGY_ERROR_LIMIT,
        },
        "mesh_levels": [
            {key: value for key, value in row.items() if key not in {"z", "temperature", "exact"}}
            for row in rows
        ],
        "finest": {
            key: value for key, value in finest.items() if key not in {"z", "temperature", "exact"}
        },
        "validated": validated,
    }
    write_json(output / "analytic_heat" / "analytic_summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2))
    return 0 if validated else 3


if __name__ == "__main__":
    raise SystemExit(main())
