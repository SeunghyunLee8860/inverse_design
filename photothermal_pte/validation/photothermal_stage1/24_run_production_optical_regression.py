#!/usr/bin/env python3
"""Run one fresh optical regression from the production entrypoint.

The case is built from ``eqc_lib.build_control_base(force=True)``; no bandwidth
sweep FSP is loaded. HEAT, optimization, adjoint solves, clipping, gains,
Q-channel deletion, and Q rescaling are not implemented here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

import config_stage1 as config
from lumerical_api import select_installation, write_json


C0 = 299792458.0
EPS0 = 8.8541878128e-12
TARGET_WAVELENGTH_M = 4.0e-6
TARGET_FREQUENCY_HZ = C0 / TARGET_WAVELENGTH_M
MATERIAL_WAVELENGTH_RANGE_M = (2.7e-6, 13.2e-6)
MATERIAL_SAMPLES = 600
MATERIAL_NAME = "TaIrTe4_ani"
PABS_GROUP = "stage1_pabs_adv"
PABS_FIELD = f"{PABS_GROUP}::field"
PABS_INDEX = f"{PABS_GROUP}::index"
REFLECTION_MONITOR = "stage1_reflection"
TRANSMISSION_MONITOR = "stage1_transmission"
LOCAL_TOP_MONITOR = "stage1_flux_top"
LOCAL_BOTTOM_MONITOR = "stage1_flux_bottom"
SIDE_MONITORS = {
    "x_min": "bandwidth_flux_xmin",
    "x_max": "bandwidth_flux_xmax",
    "y_min": "bandwidth_flux_ymin",
    "y_max": "bandwidth_flux_ymax",
}
EXPECTED_SIMULATION_TIME_S = 4.0e-12
EXPECTED_AUTO_SHUTOFF_MIN = 1.0e-8
EXPECTED_FLAKE_DZ_M = 5.0e-9
PABS_Z_PADDING_M = 50.0e-9
BOX_Z_MIN_M = -200e-9
BOX_Z_MAX_M = 100e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case", required=True, choices=("flat", "disk"))
    parser.add_argument("--polarization-deg", required=True, type=float)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument("--threads", default="8")
    parser.add_argument("--baseline-result")
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    if args.case == "disk" and not np.isclose(args.polarization_deg, 0.0):
        parser.error("the fixed-disk regression is defined for x polarization")
    return args


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def scalar(value: Any, label: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise RuntimeError(f"{label} is not scalar: shape={array.shape}")
    result = float(np.real(array[0]))
    if not np.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def safe_get(fdtd: Any, name: str, prop: str) -> Any:
    try:
        value = fdtd.getnamed(name, prop)
        array = np.asarray(value)
        if array.size == 1:
            item = array.reshape(-1)[0]
            if isinstance(item, (str, np.str_)):
                return str(item)
            if np.iscomplexobj(item):
                return jsonable(complex(item))
            return float(item)
        return jsonable(array)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def safe_set(fdtd: Any, name: str, prop: str, value: Any) -> bool:
    try:
        fdtd.setnamed(name, prop, value)
        return True
    except Exception:
        return False


def object_exists(fdtd: Any, name: str) -> bool:
    try:
        return int(fdtd.getnamednumber(name)) > 0
    except Exception:
        return False


def configure_single_frequency(fdtd: Any, name: str) -> None:
    if not object_exists(fdtd, name):
        return
    for prop, value in (
        ("override global monitor settings", True),
        ("use source limits", False),
        ("use wavelength spacing", True),
        ("wavelength center", TARGET_WAVELENGTH_M),
        ("wavelength span", 0.0),
        ("frequency points", 1),
    ):
        safe_set(fdtd, name, prop, value)


def source_properties(fdtd: Any) -> dict[str, Any]:
    props = (
        "wavelength start",
        "wavelength stop",
        "wavelength span",
        "center wavelength",
        "frequency start",
        "frequency stop",
        "frequency span",
        "center frequency",
        "frequency",
        "pulse type",
        "pulselength",
        "offset",
        "optimize for short pulse",
        "eliminate dc",
        "eliminate discontinuities",
        "set time domain",
        "set wavelength",
        "set frequency",
        "maximum convolution time window",
        "set maximum convolution time window",
        "amplitude",
        "phase",
        "polarization angle",
        "direction",
        "injection axis",
        "use global source settings",
        "override global source settings",
    )
    result = {prop: safe_get(fdtd, "source", prop) for prop in props}
    for prop in ("wavelength start", "wavelength stop"):
        try:
            result[f"global {prop}"] = scalar(fdtd.getglobalsource(prop), prop)
        except Exception as exc:
            result[f"global {prop}"] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def invariant_snapshot(fdtd: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "solver": {
            prop: safe_get(fdtd, "FDTD", prop)
            for prop in (
                "x min bc", "x max bc", "y min bc", "y max bc",
                "z min bc", "z max bc", "x", "x span", "y", "y span",
                "z", "z span", "pml layers",
            )
        },
        "objects": {},
    }
    object_props = {
        "source": ("x", "x span", "y", "y span", "z", "direction", "injection axis"),
        REFLECTION_MONITOR: ("x", "x span", "y", "y span", "z", "monitor type"),
        TRANSMISSION_MONITOR: ("x", "x span", "y", "y span", "z", "monitor type"),
        LOCAL_TOP_MONITOR: ("x", "x span", "y", "y span", "z", "monitor type"),
        LOCAL_BOTTOM_MONITOR: ("x", "x span", "y", "y span", "z", "monitor type"),
        "TaIrTe4_flake": ("x", "x span", "y", "y span", "z min", "z max", "material"),
        "SiO2_spacer": ("x", "x span", "y", "y span", "z min", "z max", "material"),
        "Si_substrate": ("x", "x span", "y", "y span", "z min", "z max", "material"),
        "design": ("x", "x span", "y", "y span", "z min", "z max", "material"),
    }
    for name, props in object_props.items():
        if object_exists(fdtd, name):
            result["objects"][name] = {prop: safe_get(fdtd, name, prop) for prop in props}
    return result


def assert_same_invariants(before: dict[str, Any], after: dict[str, Any]) -> None:
    if before != after:
        raise RuntimeError(
            "geometry/boundary/source/monitor-position contract changed: "
            + json.dumps({"before": before, "after": after}, indent=2)
        )


def fixed_material_table(repository_root: Path) -> np.ndarray:
    data = np.loadtxt(repository_root / "bundle" / "perm_data.txt")
    order = np.argsort(data[:, 0])
    wavelengths_nm = data[order, 0]
    eps_x_data = data[order, 1] + 1j * data[order, 2]
    eps_y_data = data[order, 3] + 1j * data[order, 4]
    sample_wavelengths_m = np.linspace(
        MATERIAL_WAVELENGTH_RANGE_M[0],
        MATERIAL_WAVELENGTH_RANGE_M[1],
        MATERIAL_SAMPLES,
    )
    sample_nm = sample_wavelengths_m * 1e9

    def interp(values: np.ndarray) -> np.ndarray:
        return (
            np.interp(sample_nm, wavelengths_nm, values.real)
            + 1j * np.interp(sample_nm, wavelengths_nm, values.imag)
        )

    eps_x = interp(eps_x_data)
    eps_y = interp(eps_y_data)
    eps_z = np.full_like(eps_x, 16.0 + 0.0j)
    frequency_hz = C0 / sample_wavelengths_m
    return np.column_stack((frequency_hz, eps_x, eps_y, eps_z))


def add_side_monitor(
    fdtd: Any,
    name: str,
    normal: str,
    position_m: float,
    transverse_span_m: float,
) -> None:
    if object_exists(fdtd, name):
        fdtd.eval(f'select("{name}"); delete;')
    monitor = fdtd.addpower()
    monitor["name"] = name
    monitor["monitor type"] = f"2D {normal.upper()}-normal"
    monitor[normal] = position_m
    transverse = "y" if normal == "x" else "x"
    monitor[transverse] = 0.0
    monitor[f"{transverse} span"] = transverse_span_m
    monitor["z"] = 0.5 * (BOX_Z_MIN_M + BOX_Z_MAX_M)
    monitor["z span"] = BOX_Z_MAX_M - BOX_Z_MIN_M
    configure_single_frequency(fdtd, name)


def trapezoid_weights(coordinates: np.ndarray) -> np.ndarray:
    x = np.asarray(coordinates, float).reshape(-1)
    if x.size < 2 or np.any(np.diff(x) <= 0):
        raise RuntimeError("quadrature coordinates are not strictly increasing")
    weights = np.empty_like(x)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


def integrate_xyz(values: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    return float(
        np.einsum(
            "i,j,k,ijk->",
            trapezoid_weights(x),
            trapezoid_weights(y),
            trapezoid_weights(z),
            np.asarray(values, float),
            optimize=True,
        )
    )


def component_absorption(fdtd: Any, frequency_hz: float) -> dict[str, float]:
    x = np.asarray(fdtd.getdata(PABS_FIELD, "x", 1), float).reshape(-1)
    y = np.asarray(fdtd.getdata(PABS_FIELD, "y", 1), float).reshape(-1)
    z = np.asarray(fdtd.getdata(PABS_FIELD, "z", 1), float).reshape(-1)
    deltas = {
        axis: np.asarray(fdtd.getdata(PABS_FIELD, f"delta_{axis}", 1), float).reshape(-1)
        for axis in "xyz"
    }
    source_power = scalar(fdtd.sourcepower(frequency_hz), "sourcepower")
    omega = 2.0 * np.pi * frequency_hz
    result: dict[str, float] = {}
    for axis_index, component in enumerate("xyz"):
        electric = np.asarray(fdtd.getdata(PABS_FIELD, f"E{component}", 1)).squeeze()
        epsilon = np.asarray(fdtd.getdata(PABS_INDEX, f"index_{component}", 1)).squeeze() ** 2
        q_native = 0.5 * EPS0 * omega * np.abs(electric) ** 2 * np.imag(epsilon)
        coordinates = [x, y, z]
        coordinates[axis_index] = coordinates[axis_index] + deltas[component]
        result[f"A_Q{component}_native"] = integrate_xyz(q_native, *coordinates) / source_power
        del electric, epsilon, q_native
    result["A_Q_total_native"] = sum(result[f"A_Q{axis}_native"] for axis in "xyz")
    return result


def complex_interp(x: np.ndarray, y: np.ndarray, target: float) -> complex:
    order = np.argsort(np.asarray(x, float))
    xx = np.asarray(x, float)[order]
    yy = np.asarray(y, complex)[order]
    return complex(np.interp(target, xx, yy.real) + 1j * np.interp(target, xx, yy.imag))


def multilayer_tmm(epsilon_film: complex) -> dict[str, float]:
    n0 = 1.0 + 0.0j
    ns = 3.425 + 0.0j
    n_film = np.sqrt(complex(epsilon_film))
    if n_film.imag < 0:
        n_film = -n_film
    layers = ((n_film, 100e-9), (1.38 + 0.0j, 285e-9))
    matrix = np.eye(2, dtype=complex)
    for refractive_index, thickness_m in layers:
        phase = 2.0 * np.pi * refractive_index * thickness_m / TARGET_WAVELENGTH_M
        layer = np.array(
            [
                [np.cos(phase), -1j * np.sin(phase) / refractive_index],
                [-1j * refractive_index * np.sin(phase), np.cos(phase)],
            ],
            dtype=complex,
        )
        matrix = matrix @ layer
    denominator = (
        n0 * matrix[0, 0] + n0 * ns * matrix[0, 1]
        + matrix[1, 0] + ns * matrix[1, 1]
    )
    reflection = (
        n0 * matrix[0, 0] + n0 * ns * matrix[0, 1]
        - matrix[1, 0] - ns * matrix[1, 1]
    ) / denominator
    transmission = 2.0 * n0 / denominator
    r_power = float(abs(reflection) ** 2)
    t_power = float((ns.real / n0.real) * abs(transmission) ** 2)
    return {"R": r_power, "T": t_power, "A": 1.0 - r_power - t_power}


def epsilon_contract(fdtd: Any, material_table: np.ndarray, dt_s: float) -> dict[str, Any]:
    source_start = scalar(fdtd.getnamed("source", "wavelength start"), "source start")
    source_stop = scalar(fdtd.getnamed("source", "wavelength stop"), "source stop")
    frequency_ends = C0 / np.asarray([source_start, source_stop], float)
    fmin = float(np.min(frequency_ends))
    fmax = float(np.max(frequency_ends))
    axes = []
    for component, axis in zip((1, 2, 3), "xyz"):
        fitted_n = np.asarray(
            fdtd.getfdtdindex(
                MATERIAL_NAME,
                np.asarray([TARGET_FREQUENCY_HZ]),
                fmin,
                fmax,
                component,
            )
        ).reshape(-1)[0]
        fitted_epsilon = complex(fitted_n) ** 2
        numerical = np.asarray(
            fdtd.getnumericalpermittivity(
                MATERIAL_NAME,
                np.asarray([TARGET_FREQUENCY_HZ]),
                fmin,
                fmax,
                dt_s,
                component,
            )
        ).reshape(-1)[0]
        raw = complex_interp(material_table[:, 0].real, material_table[:, component], TARGET_FREQUENCY_HZ)
        axes.append(
            {
                "axis": axis,
                "raw_imported_epsilon": jsonable(raw),
                "fitted_epsilon": jsonable(fitted_epsilon),
                "finite_dt_numerical_permittivity": jsonable(complex(numerical)),
                "fit_relative_error": float(abs(fitted_epsilon - raw) / max(abs(raw), np.finfo(float).tiny)),
                "numerical_relative_difference": float(abs(complex(numerical) - fitted_epsilon) / max(abs(fitted_epsilon), np.finfo(float).tiny)),
            }
        )

    x = np.asarray(fdtd.getdata(PABS_INDEX, "x", 1), float).reshape(-1)
    y = np.asarray(fdtd.getdata(PABS_INDEX, "y", 1), float).reshape(-1)
    z = np.asarray(fdtd.getdata(PABS_INDEX, "z", 1), float).reshape(-1)
    ix = int(np.argmin(np.abs(x)))
    iy = int(np.argmin(np.abs(y)))
    iz = int(np.argmin(np.abs(z + 50e-9)))
    for row, axis in zip(axes, "xyz"):
        index = np.asarray(fdtd.getdata(PABS_INDEX, f"index_{axis}", 1)).squeeze()
        monitored = complex(index[ix, iy, iz]) ** 2
        row["index_monitor_epsilon"] = jsonable(monitored)
        fitted = complex(row["fitted_epsilon"]["real"], row["fitted_epsilon"]["imag"])
        row["index_monitor_relative_error_vs_fitted"] = float(
            abs(monitored - fitted) / max(abs(fitted), np.finfo(float).tiny)
        )
    sampled_wavelengths = C0 / np.asarray(material_table[:, 0].real, float)
    return {
        "material_name": MATERIAL_NAME,
        "sampled_material_wavelength_range_m": [
            float(np.min(sampled_wavelengths)), float(np.max(sampled_wavelengths))
        ],
        "sample_count": int(material_table.shape[0]),
        "fit_frequency_range_Hz": [fmin, fmax],
        "axes": axes,
    }


def axis_contract(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, float).reshape(-1)
    delta = np.diff(values)
    return {
        "count": int(values.size),
        "min_m": float(values[0]),
        "max_m": float(values[-1]),
        "minimum_step_m": float(np.min(delta)),
        "maximum_step_m": float(np.max(delta)),
        "coordinates_m": values.tolist(),
    }


def mesh_contract(fdtd: Any, output: Path) -> dict[str, Any]:
    axes = {
        axis: np.asarray(fdtd.getdata(PABS_FIELD, axis, 1), float).reshape(-1)
        for axis in "xyz"
    }
    np.savez(output / "mesh_coordinates.npz", **{f"{k}_m": v for k, v in axes.items()})
    z = axes["z"]
    internal = z[(z >= -100e-9 - 1e-15) & (z <= 0.0 + 1e-15)]
    internal_dz = np.diff(internal)
    interface_targets = (-100e-9, 0.0)
    interface_planes = []
    for target in interface_targets:
        nearest = float(z[np.argmin(np.abs(z - target))])
        interface_planes.append(
            {
                "target_m": target,
                "nearest_mesh_plane_m": nearest,
                "absolute_error_m": abs(nearest - target),
            }
        )
    mesh_objects = {}
    for name in ("global_uniform_mesh", "design_mesh", "flake_mesh", "fixed_stack_z_mesh", "source_z_registration_mesh"):
        mesh_objects[name] = {
            "exists": object_exists(fdtd, name),
            "enabled": safe_get(fdtd, name, "enabled") if object_exists(fdtd, name) else None,
            "dx": safe_get(fdtd, name, "dx") if object_exists(fdtd, name) else None,
            "dy": safe_get(fdtd, name, "dy") if object_exists(fdtd, name) else None,
            "dz": safe_get(fdtd, name, "dz") if object_exists(fdtd, name) else None,
        }
    return {
        "solver": {
            prop: safe_get(fdtd, "FDTD", prop)
            for prop in (
                "mesh type", "mesh refinement", "mesh accuracy", "min mesh step",
                "simulation time", "auto shutoff min", "dt",
            )
        },
        "axes": {axis: axis_contract(values) for axis, values in axes.items()},
        "TaIrTe4_internal_dz_m": internal_dz.tolist(),
        "TaIrTe4_internal_dz_min_m": float(np.min(internal_dz)),
        "TaIrTe4_internal_dz_max_m": float(np.max(internal_dz)),
        "interface_mesh_planes": interface_planes,
        "mesh_objects": mesh_objects,
    }


def assert_numerical_contract(fdtd: Any, expected_start_m: float, expected_stop_m: float) -> None:
    required_strings = {
        "mesh type": "auto non-uniform",
        "mesh refinement": "conformal variant 1",
    }
    for prop, expected in required_strings.items():
        actual = str(fdtd.getnamed("FDTD", prop)).strip().lower()
        if actual != expected:
            raise RuntimeError(f"FDTD.{prop}={actual!r}, expected {expected!r}")
    if int(round(scalar(fdtd.getnamed("FDTD", "mesh accuracy"), "mesh accuracy"))) != 5:
        raise RuntimeError("mesh accuracy is not 5")
    if not np.isclose(scalar(fdtd.getnamed("FDTD", "simulation time"), "simulation time"), EXPECTED_SIMULATION_TIME_S):
        raise RuntimeError("simulation time changed from corrected contract")
    if not np.isclose(scalar(fdtd.getnamed("FDTD", "auto shutoff min"), "auto shutoff"), EXPECTED_AUTO_SHUTOFF_MIN):
        raise RuntimeError("auto shutoff changed from corrected contract")
    if object_exists(fdtd, "global_uniform_mesh"):
        raise RuntimeError("global_uniform_mesh exists")
    if not object_exists(fdtd, "flake_mesh"):
        raise RuntimeError("flake_mesh is missing")
    flake_dz = scalar(fdtd.getnamed("flake_mesh", "dz"), "flake_mesh.dz")
    if not np.isclose(flake_dz, EXPECTED_FLAKE_DZ_M, rtol=0.0, atol=1e-15):
        raise RuntimeError(f"flake dz={flake_dz}, expected {EXPECTED_FLAKE_DZ_M}")
    actual_start = scalar(fdtd.getnamed("source", "wavelength start"), "source start")
    actual_stop = scalar(fdtd.getnamed("source", "wavelength stop"), "source stop")
    if not np.isclose(actual_start, expected_start_m, rtol=0.0, atol=1e-15):
        raise RuntimeError(f"source start={actual_start}, expected {expected_start_m}")
    if not np.isclose(actual_stop, expected_stop_m, rtol=0.0, atol=1e-15):
        raise RuntimeError(f"source stop={actual_stop}, expected {expected_stop_m}")


def save_source_spectrum(fdtd: Any, output: Path) -> dict[str, Any]:
    props = source_properties(fdtd)
    f_start = scalar(fdtd.getnamed("source", "frequency start"), "frequency start")
    f_stop = scalar(fdtd.getnamed("source", "frequency stop"), "frequency stop")
    frequencies = np.linspace(min(f_start, f_stop), max(f_start, f_stop), 513)
    try:
        spectrum = np.asarray(fdtd.sourcenorm(frequencies, "source"), complex).reshape(-1)
    except Exception:
        spectrum = np.asarray(fdtd.sourcenorm(frequencies), complex).reshape(-1)
    target_norm = np.interp(TARGET_FREQUENCY_HZ, frequencies, np.abs(spectrum))
    target_power = scalar(fdtd.sourcepower(TARGET_FREQUENCY_HZ), "sourcepower at 4um")
    target_intensity = scalar(fdtd.sourceintensity(TARGET_FREQUENCY_HZ), "sourceintensity at 4um")
    with (output / "source_spectrum.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("frequency_Hz", "wavelength_m", "sourcenorm_real", "sourcenorm_imag", "sourcenorm_magnitude"))
        for frequency, value in zip(frequencies, spectrum):
            writer.writerow((frequency, C0 / frequency, value.real, value.imag, abs(value)))

    # Complex-envelope reconstruction from the actual solver normalization
    # spectrum.  This is explicitly diagnostic; it is not fed back to FDTD.
    df = float(frequencies[1] - frequencies[0])
    envelope = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(spectrum)))
    times = np.fft.fftshift(np.fft.fftfreq(frequencies.size, d=df))
    peak = float(np.max(np.abs(envelope)))
    if peak > 0:
        envelope = envelope / peak
    with (output / "source_time_envelope_from_sourcenorm.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "normalized_real", "normalized_imag", "normalized_magnitude"))
        for time_s, value in zip(times, envelope):
            writer.writerow((time_s, value.real, value.imag, abs(value)))
    props.update(
        {
            "sourcenorm_magnitude_at_4um": float(target_norm),
            "sourcepower_W_at_4um": target_power,
            "sourceintensity_W_m2_at_4um": target_intensity,
            "spectrum_csv": str(output / "source_spectrum.csv"),
            "time_envelope_csv": str(output / "source_time_envelope_from_sourcenorm.csv"),
            "time_envelope_definition": (
                "normalized complex envelope reconstructed by inverse DFT of the "
                "actual v261 sourcenorm spectrum; diagnostic only"
            ),
        }
    )
    return props


def enable_periodic_correction(fdtd: Any, name: str) -> dict[str, Any]:
    script = str(fdtd.getnamed(name, "analysis script"))
    marker = "Periodic boundary condition correction"
    marker_position = script.find(marker)
    if marker_position < 0:
        raise RuntimeError("installed pabs_adv lacks periodic-correction code")
    tail = script[marker_position:]
    old = "if (0) {"
    position = tail.find(old)
    if position < 0:
        raise RuntimeError("cannot enable pabs_adv periodic correction")
    absolute = marker_position + position
    fdtd.setnamed(
        name, "analysis script",
        script[:absolute] + "if (1) {" + script[absolute + len(old):],
    )
    return {"x_periodic": True, "y_periodic": True}


def fixed_disk_density(sim: Any) -> np.ndarray:
    x = np.asarray(sim.design_x, float)
    y = np.asarray(sim.design_y, float)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    radius = float(config.FIXED_DESIGN_DISK_RADIUS_M)
    disk = (xx**2 + yy**2) <= radius**2
    density = np.repeat(
        disk[:, :, None], int(sim.design_grids[2]), axis=2
    ).astype(float)
    density[-1, :, :] = density[0, :, :]
    density[:, -1, :] = density[:, 0, :]
    return density


def add_power_monitor(fdtd: Any, name: str, z_m: float, x_span_m: float, y_span_m: float) -> None:
    if object_exists(fdtd, name):
        fdtd.eval(f'select("{name}"); delete;')
    monitor = fdtd.addpower()
    monitor["name"] = name
    monitor["monitor type"] = "2D Z-normal"
    monitor["x"] = 0.0
    monitor["x span"] = x_span_m
    monitor["y"] = 0.0
    monitor["y span"] = y_span_m
    monitor["z"] = z_m
    configure_single_frequency(fdtd, name)


def fitted_axis(epsilon: dict[str, Any], axis: str) -> complex:
    for row in epsilon["axes"]:
        if row["axis"] == axis:
            value = row["fitted_epsilon"]
            return complex(value["real"], value["imag"])
    raise KeyError(axis)


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    project = output / "production_case.fsp"
    result_path = output / "case_result.json"

    installation = select_installation("v261")
    os.environ["VC_LUMERICAL_ROOT"] = str(installation.root)
    os.environ["LUMERICAL_ROOT"] = str(installation.root)
    os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = args.gpu_device
    os.environ["CL_GPU_DEVICE"] = args.gpu_device
    os.environ["FDTD_THREADS"] = str(args.threads)
    os.environ["TARGET_WL_UM"] = "4.0"
    os.environ["SOURCE_WL_START_UM"] = "3.0"
    os.environ["SOURCE_WL_STOP_UM"] = "6.0"
    os.environ["MATERIAL_FIT_START_UM"] = "2.7"
    os.environ["MATERIAL_FIT_STOP_UM"] = "13.2"
    os.environ["BULK_MESH_MODE"] = "auto"
    os.environ["MESH_ACCURACY"] = "5"
    os.environ["VC_MESH_REFINEMENT"] = "conformal variant 1"
    for path in (config.REPOSITORY_ROOT, config.REPOSITORY_ROOT / "bundle"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import eqc_lib as runtime

    runtime.CONTROL_BASE = project
    runtime.DATA = output / "data"
    incident = "y" if np.isclose(args.polarization_deg, 90.0) else "x"
    model = runtime.load_model()
    sim = runtime.build_control_base(
        model, force=True, incident_polarization=incident
    )

    fdtd = runtime.open_control(project)
    try:
        fdtd.switchtolayout()
        sim.fdtd = fdtd
        fdtd.setnamed("source", "polarization angle", float(args.polarization_deg))
        if args.case == "flat":
            fdtd.setnamed("design", "enabled", False)
        else:
            sim.update_design_density(fixed_disk_density(sim))
            fdtd.setnamed("design", "enabled", True)

        if object_exists(fdtd, PABS_GROUP):
            fdtd.eval(f'select("{PABS_GROUP}"); delete;')
        pabs = fdtd.addobject("pabs_adv")
        pabs["name"] = PABS_GROUP
        pabs["x"] = 0.0
        pabs["x span"] = float(model.Sx) * 1e-6
        pabs["y"] = 0.0
        pabs["y span"] = float(model.Sy) * 1e-6
        pabs["z"] = -0.5 * float(model.flake_h) * 1e-6
        pabs["z span"] = (
            float(model.flake_h) * 1e-6 + 2.0 * PABS_Z_PADDING_M
        )
        periodic = enable_periodic_correction(fdtd, PABS_GROUP)

        add_power_monitor(
            fdtd, REFLECTION_MONITOR,
            (float(model.src_c[2]) + 0.15) * 1e-6,
            float(model.Sx) * 1e-6, float(model.Sy) * 1e-6,
        )
        add_power_monitor(
            fdtd, TRANSMISSION_MONITOR,
            -(float(model.flake_h) + float(model.sio2_h) + 0.40) * 1e-6,
            float(model.Sx) * 1e-6, float(model.Sy) * 1e-6,
        )
        add_power_monitor(
            fdtd, LOCAL_TOP_MONITOR, BOX_Z_MAX_M,
            float(model.Sx) * 1e-6, float(model.Sy) * 1e-6,
        )
        add_power_monitor(
            fdtd, LOCAL_BOTTOM_MONITOR, BOX_Z_MIN_M,
            float(model.Sx) * 1e-6, float(model.Sy) * 1e-6,
        )
        x_face = 0.5 * scalar(fdtd.getnamed("FDTD", "x span"), "x span")
        y_face = 0.5 * scalar(fdtd.getnamed("FDTD", "y span"), "y span")
        add_side_monitor(fdtd, SIDE_MONITORS["x_min"], "x", -x_face, 2.0 * y_face)
        add_side_monitor(fdtd, SIDE_MONITORS["x_max"], "x", x_face, 2.0 * y_face)
        add_side_monitor(fdtd, SIDE_MONITORS["y_min"], "y", -y_face, 2.0 * x_face)
        add_side_monitor(fdtd, SIDE_MONITORS["y_max"], "y", y_face, 2.0 * x_face)

        monitor_names = (
            runtime.FIELD_REGION,
            sim.design_monitor_name,
            sim.design_index_monitor_name,
            REFLECTION_MONITOR,
            TRANSMISSION_MONITOR,
            LOCAL_TOP_MONITOR,
            LOCAL_BOTTOM_MONITOR,
            *SIDE_MONITORS.values(),
            PABS_FIELD,
            PABS_INDEX,
        )
        for name in monitor_names[3:]:
            configure_single_frequency(fdtd, name)
        runtime.pin_solver(fdtd)
        fdtd.setnamed("source", "use global source settings", True)
        fdtd.setnamed("source", "override global source settings", False)
        fdtd.setglobalsource("wavelength start", runtime.SOURCE_WL_START)
        fdtd.setglobalsource("wavelength stop", runtime.SOURCE_WL_STOP)
        runtime.configure_session_resources(fdtd)
        pre_run_contract = runtime.assert_production_contract(
            fdtd, monitor_names, run_setup=True
        )
        fdtd.save(str(project))
        if args.contract_only:
            write_json(
                output / "pre_run_contract.json",
                jsonable({
                    "entrypoint": "eqc_lib.build_control_base(force=True)",
                    "contract": pre_run_contract,
                    "heat_run": False,
                    "solver_run": False,
                }),
            )
            print(json.dumps(jsonable(pre_run_contract), indent=2))
            return 0

        resource = runtime.run_session(
            fdtd, f"production_{args.case}_{args.polarization_deg:g}"
        )
        fdtd.runanalysis(PABS_GROUP)
        solver_log = output / "production_case_p0.log"
        log_text = solver_log.read_text(errors="replace") if solver_log.is_file() else ""
        diverged = "electromagnetic fields are diverging" in log_text.lower()
        if diverged:
            raise RuntimeError("production FDTD solver diverged")

        result_frequency = scalar(fdtd.getdata(PABS_FIELD, "f"), "Pabs frequency")
        if not np.isclose(result_frequency, TARGET_FREQUENCY_HZ, rtol=2e-12):
            raise RuntimeError("Pabs result is not at 4 um")
        r_power = scalar(fdtd.transmission(REFLECTION_MONITOR), "R")
        t_power = -scalar(fdtd.transmission(TRANSMISSION_MONITOR), "T")
        global_absorption = 1.0 - r_power - t_power
        top = scalar(fdtd.transmission(LOCAL_TOP_MONITOR), "local top")
        bottom = scalar(fdtd.transmission(LOCAL_BOTTOM_MONITOR), "local bottom")
        local_absorption = bottom - top
        side = {
            key: scalar(fdtd.transmission(name), key)
            for key, name in SIDE_MONITORS.items()
        }
        a_x = side["x_min"] - side["x_max"]
        a_y = side["y_min"] - side["y_max"]
        six_face = local_absorption + a_x + a_y
        pabs_total = scalar(
            fdtd.getresult(PABS_GROUP, "Pabs_total")["Pabs_total"],
            "Pabs_total",
        )
        components = component_absorption(fdtd, result_frequency)
        dt_s = scalar(fdtd.getnamed("FDTD", "dt"), "FDTD dt")
        material_table = np.asarray(fdtd.getmaterial(MATERIAL_NAME, "sampled data"))
        epsilon = epsilon_contract(fdtd, material_table, dt_s)
        mesh = mesh_contract(fdtd, output)
        post_run_source = save_source_spectrum(fdtd, output)
        post_run_contract = runtime.assert_production_contract(
            fdtd, monitor_names, run_setup=False, validate_resources=False
        )
        fdtd.save(str(project))

        if not np.allclose(
            [mesh["TaIrTe4_internal_dz_min_m"], mesh["TaIrTe4_internal_dz_max_m"]],
            [EXPECTED_FLAKE_DZ_M, EXPECTED_FLAKE_DZ_M],
            rtol=0.0, atol=1e-15,
        ):
            raise RuntimeError("realized TaIrTe4 z mesh is not 5 nm")
        closure = abs(pabs_total - local_absorption) / max(
            abs(local_absorption), np.finfo(float).tiny
        )
        component_closure = abs(
            components["A_Q_total_native"] - local_absorption
        ) / max(abs(local_absorption), np.finfo(float).tiny)
        tmm = None
        tmm_error = None
        if args.case == "flat":
            tmm_x = multilayer_tmm(fitted_axis(epsilon, "x"))
            tmm_y = multilayer_tmm(fitted_axis(epsilon, "y"))
            theta = np.deg2rad(float(args.polarization_deg))
            tmm_a = float(
                np.cos(theta) ** 2 * tmm_x["A"]
                + np.sin(theta) ** 2 * tmm_y["A"]
            )
            tmm = {"x": tmm_x, "y": tmm_y, "polarization_weighted_A": tmm_a}
            tmm_error = abs(local_absorption - tmm_a) / max(
                abs(tmm_a), np.finfo(float).tiny
            )

        baseline_comparison = None
        if args.baseline_result:
            baseline_path = Path(args.baseline_result).expanduser().resolve()
            baseline = json.loads(baseline_path.read_text())
            old_abs = baseline["absorption"]
            current = {
                "A_Q_pabs_adv": pabs_total,
                "A_local_flux": local_absorption,
                "delta_closure_pabs_adv": closure,
                "A_Qx_native": components["A_Qx_native"],
                "A_Qy_native": components["A_Qy_native"],
                "A_Qz_native": components["A_Qz_native"],
            }
            baseline_comparison = {
                "baseline_result": str(baseline_path),
                "absolute_differences": {
                    key: abs(current[key] - float(old_abs[key])) for key in current
                },
                "dt_absolute_difference_s": abs(
                    dt_s - float(baseline["mesh_contract"]["solver"]["dt"])
                ),
                "mesh_coordinates_identical": all(
                    mesh["axes"][axis]["coordinates_m"]
                    == baseline["mesh_contract"]["axes"][axis]["coordinates_m"]
                    for axis in "xyz"
                ),
                "fitted_epsilon_relative_differences": {
                    axis: float(abs(
                        fitted_axis(epsilon, axis)
                        - fitted_axis(baseline["epsilon_contract"], axis)
                    ) / max(abs(fitted_axis(baseline["epsilon_contract"], axis)), np.finfo(float).tiny))
                    for axis in "xyz"
                },
                "pulse_type_equal": (
                    post_run_source["pulse type"]
                    == baseline["source_contract_post_run"]["pulse type"]
                ),
                "source_norm_relative_difference": float(abs(
                    post_run_source["sourcenorm_magnitude_at_4um"]
                    - baseline["source_contract_post_run"]["sourcenorm_magnitude_at_4um"]
                ) / baseline["source_contract_post_run"]["sourcenorm_magnitude_at_4um"]),
            }

        result = {
            "status": "completed",
            "entrypoint": "eqc_lib.build_control_base(force=True)",
            "case": args.case,
            "polarization_angle_deg": float(args.polarization_deg),
            "source_range_um": [3.0, 6.0],
            "analysis_wavelength_um": 4.0,
            "resource": resource,
            "solver_run": {
                "converged": True,
                "diverged": False,
                "log": str(solver_log),
            },
            "project": str(project),
            "pre_run_contract": pre_run_contract,
            "post_run_contract": post_run_contract,
            "source_contract_post_run": post_run_source,
            "pabs_periodic_correction": periodic,
            "absorption": {
                "R": r_power,
                "T": t_power,
                "A_global": global_absorption,
                "A_local_flux": local_absorption,
                "A_six_face": six_face,
                "A_x_pair": a_x,
                "A_y_pair": a_y,
                "side_monitor_signed_transmission": side,
                "A_Q_pabs_adv": pabs_total,
                **components,
                "delta_closure_pabs_adv": closure,
                "delta_closure_native_components": component_closure,
                "TMM": tmm,
                "delta_TMM": tmm_error,
            },
            "epsilon_contract": epsilon,
            "mesh_contract": mesh,
            "baseline_comparison": baseline_comparison,
            "acceptance": {
                "closure_lt_0p5_percent": closure < 0.005,
                "flat_TMM_lt_0p5_percent": tmm_error is None or tmm_error < 0.005,
                "all_pass": closure < 0.005 and (
                    tmm_error is None or tmm_error < 0.005
                ),
            },
            "unchanged": [
                "geometry", "PBC/PML", "source and monitor positions",
                "TaIrTe4 tensor axes", "pabs_adv formula", "Qy",
                "normalization", "HEAT code", "optimizer and mapping logic",
            ],
            "heat_run": False,
            "optimizer_run": False,
            "adjoint_run": False,
            "Qy_deleted": False,
            "Q_clipped": False,
            "flux_gain": False,
            "Q_rescaled": False,
        }
        write_json(result_path, jsonable(result))
        print(json.dumps(jsonable(result), indent=2))
    finally:
        fdtd.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
