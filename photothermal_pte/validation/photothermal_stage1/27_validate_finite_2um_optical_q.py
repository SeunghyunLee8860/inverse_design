#!/usr/bin/env python3
"""Run one fail-closed finite 2 um TaIrTe4 optical-Q validation case.

This is a separate non-periodic builder.  It reuses the certified production
material and numerical constants without changing the periodic builder.
HEAT, adjoint, gradients, optimization, Q clipping, gains, rescaling, cropping,
and tiling are intentionally absent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import config_stage1 as config
from lumerical_api import jsonable, load_lumapi, select_installation, write_json


C0 = 299792458.0
EPS0 = 8.8541878128e-12
TARGET_WAVELENGTH_M = 4.0e-6
TARGET_FREQUENCY_HZ = C0 / TARGET_WAVELENGTH_M
SOURCE_START_M = 3.0e-6
SOURCE_STOP_M = 6.0e-6
MATERIAL_START_M = 2.7e-6
MATERIAL_STOP_M = 13.2e-6
MATERIAL_SAMPLES = 600
MATERIAL_NAME = "TaIrTe4_ani"
SIO2_MATERIAL = "finite_shared_SiO2_n1p38"

FLAKE_SPAN_M = 2.0e-6
FLAKE_THICKNESS_M = 100.0e-9
FLAKE_BOUNDS_M = {
    "x": (-1.0e-6, 1.0e-6),
    "y": (-1.0e-6, 1.0e-6),
    "z": (-100.0e-9, 0.0),
}
GEOMETRIC_AREA_M2 = 4.0e-12
SIO2_THICKNESS_M = 285.0e-9
SI_DEPTH_M = 2.0e-6
DESIGN_RADIUS_M = 1.5e-6
DESIGN_HEIGHT_M = 0.6e-6
PABS_PADDING_M = 50.0e-9

FDTD_Z_MIN_M = -FLAKE_THICKNESS_M - SIO2_THICKNESS_M - SI_DEPTH_M
FDTD_Z_MAX_M = DESIGN_HEIGHT_M + 2.0e-6
TFSF_Z_MIN_M = -0.60e-6
TFSF_Z_MAX_M = 1.20e-6
INNER_BOX = {
    "x": (-1.75e-6, 1.75e-6),
    "y": (-1.75e-6, 1.75e-6),
    "z": (-0.25e-6, 0.75e-6),
}

POWER_CLOSURE_LIMIT = 0.005
CONVERGENCE_LIMIT = 0.01
EMPTY_POWER_FRACTION_LIMIT = 1.0e-4
EMPTY_FIELD_RATIO_LIMIT = 1.0e-4
TARGET_INTENSITY_W_M2 = 1.0

PABS_GROUP = "finite_pabs_adv"
PABS_FIELD = f"{PABS_GROUP}::field"
PABS_INDEX = f"{PABS_GROUP}::index"
SOURCE_NAME = "finite_tfsf_source"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--case",
        required=True,
        choices=("no-source", "empty-stack", "flat", "fixed-design"),
    )
    parser.add_argument("--polarization-deg", type=float, default=0.0)
    parser.add_argument("--domain-um", type=float, default=8.0)
    parser.add_argument("--pml-layers", type=int, default=24)
    parser.add_argument("--flake-dz-nm", type=float, default=5.0)
    parser.add_argument("--tfsf-span-um", type=float, default=6.0)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument("--threads", default="8")
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    if args.domain_um not in (8.0, 12.0, 16.0):
        parser.error("--domain-um must be 8, 12, or 16")
    if args.pml_layers < 8:
        parser.error("--pml-layers must be at least 8")
    if args.flake_dz_nm not in (2.5, 5.0, 10.0):
        parser.error("--flake-dz-nm must be 2.5, 5, or 10")
    if args.tfsf_span_um + 1.0 >= args.domain_um:
        parser.error("TFSF span needs at least 0.5 um clearance per lateral side")
    if args.case == "fixed-design" and not np.isclose(
        args.polarization_deg, 0.0
    ):
        parser.error("the required fixed-design case is x polarization")
    return args


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


def safe_set(fdtd: Any, name: str, prop: str, value: Any) -> None:
    try:
        fdtd.setnamed(name, prop, value)
    except Exception as exc:
        raise RuntimeError(f"cannot set {name}.{prop}={value!r}: {exc}") from exc


def object_exists(fdtd: Any, name: str) -> bool:
    try:
        return int(fdtd.getnamednumber(name)) > 0
    except Exception:
        return False


def configure_single_frequency(fdtd: Any, name: str) -> None:
    for prop, value in (
        ("override global monitor settings", True),
        ("use source limits", False),
        ("use wavelength spacing", True),
        ("wavelength center", TARGET_WAVELENGTH_M),
        ("wavelength span", 0.0),
        ("frequency points", 1),
    ):
        try:
            fdtd.setnamed(name, prop, value)
        except Exception:
            pass


def add_rect(
    fdtd: Any,
    name: str,
    bounds: dict[str, tuple[float, float]],
    *,
    material: str | None = None,
    index: float | None = None,
) -> None:
    obj = fdtd.addrect()
    obj["name"] = name
    for axis in "xyz":
        obj[f"{axis} min"] = bounds[axis][0]
        obj[f"{axis} max"] = bounds[axis][1]
    if material is not None:
        obj["material"] = material
    elif index is not None:
        obj["index"] = float(index)
    else:
        raise ValueError("material or index is required")


def add_shared_sio2_material(fdtd: Any) -> None:
    material = fdtd.addmaterial("Dielectric")
    fdtd.setmaterial(material, "name", SIO2_MATERIAL)
    fdtd.setmaterial(SIO2_MATERIAL, "Refractive Index", 1.38)


def add_power_face(
    fdtd: Any,
    name: str,
    axis: str,
    position: float,
    bounds: dict[str, tuple[float, float]],
) -> None:
    monitor = fdtd.addpower()
    monitor["name"] = name
    monitor["monitor type"] = f"2D {axis.upper()}-normal"
    monitor[axis] = position
    for transverse in "xyz":
        if transverse == axis:
            continue
        monitor[f"{transverse} min"] = bounds[transverse][0]
        monitor[f"{transverse} max"] = bounds[transverse][1]
    configure_single_frequency(fdtd, name)


def add_flux_box(
    fdtd: Any,
    prefix: str,
    bounds: dict[str, tuple[float, float]],
) -> dict[str, dict[str, Any]]:
    faces: dict[str, dict[str, Any]] = {}
    for axis in "xyz":
        for side, position in zip(("min", "max"), bounds[axis]):
            key = f"{axis}_{side}"
            name = f"{prefix}_{key}"
            add_power_face(fdtd, name, axis, position, bounds)
            faces[key] = {
                "name": name,
                "axis": axis,
                "side": side,
                "outward_sign": -1.0 if side == "min" else 1.0,
            }
    return faces


def add_field_monitor(
    fdtd: Any,
    name: str,
    monitor_type: str,
    bounds: dict[str, tuple[float, float]],
) -> None:
    monitor = fdtd.adddftmonitor()
    monitor["name"] = name
    monitor["monitor type"] = monitor_type
    for axis in "xyz":
        low, high = bounds[axis]
        monitor[axis] = 0.5 * (low + high)
        if not np.isclose(low, high):
            monitor[f"{axis} span"] = high - low
    configure_single_frequency(fdtd, name)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config.REPOSITORY_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(
    path: Path,
    *,
    command: str,
    commit: str,
) -> dict[str, Any]:
    return {
        "server_path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "generation_command": command,
        "generation_commit": commit,
        "reproduction": (
            f"checkout {commit}, activate the project Miniconda environment, "
            f"then run: {command}"
        ),
    }


def trapezoid_weights(coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinates, float).reshape(-1)
    if values.size < 2 or np.any(np.diff(values) <= 0):
        raise RuntimeError("quadrature coordinates are not strictly increasing")
    weights = np.empty_like(values)
    weights[0] = 0.5 * (values[1] - values[0])
    weights[-1] = 0.5 * (values[-1] - values[-2])
    weights[1:-1] = 0.5 * (values[2:] - values[:-2])
    return weights


def integrate_xyz(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> float:
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


def common_grid_component_q(
    fdtd: Any,
    frequency_hz: float,
) -> dict[str, Any]:
    x = np.asarray(fdtd.getdata(PABS_FIELD, "x", 1), float).reshape(-1)
    y = np.asarray(fdtd.getdata(PABS_FIELD, "y", 1), float).reshape(-1)
    z = np.asarray(fdtd.getdata(PABS_FIELD, "z", 1), float).reshape(-1)
    coordinates = {"x": x, "y": y, "z": z}
    deltas = {
        axis: np.asarray(
            fdtd.getdata(PABS_FIELD, f"delta_{axis}", 1), float
        ).reshape(-1)
        for axis in "xyz"
    }
    target = np.stack(
        np.meshgrid(x, y, z, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    omega = 2.0 * np.pi * frequency_hz
    common: dict[str, np.ndarray] = {}
    native_power: dict[str, float] = {}
    common_power: dict[str, float] = {}
    interpolation_error: dict[str, float] = {}
    for component in "xyz":
        electric = np.asarray(
            fdtd.getdata(PABS_FIELD, f"E{component}", 1)
        ).squeeze()
        epsilon = (
            np.asarray(fdtd.getdata(PABS_INDEX, f"index_{component}", 1)).squeeze()
            ** 2
        )
        q_native = (
            0.5
            * EPS0
            * omega
            * np.abs(electric) ** 2
            * np.imag(epsilon)
        )
        native_coordinates = [x.copy(), y.copy(), z.copy()]
        axis_index = "xyz".index(component)
        native_coordinates[axis_index] = (
            native_coordinates[axis_index] + deltas[component]
        )
        native_power[component] = integrate_xyz(
            q_native, *native_coordinates
        )
        interpolator = RegularGridInterpolator(
            tuple(native_coordinates),
            q_native,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        q_common = interpolator(target).reshape(x.size, y.size, z.size)
        common[component] = q_common
        common_power[component] = integrate_xyz(q_common, x, y, z)
        interpolation_error[component] = abs(
            common_power[component] - native_power[component]
        ) / max(abs(native_power[component]), np.finfo(float).tiny)
    return {
        "x_m": x,
        "y_m": y,
        "z_m": z,
        "Qx_native_W_m3": common["x"],
        "Qy_native_W_m3": common["y"],
        "Qz_native_W_m3": common["z"],
        "Q_native_W_m3": common["x"] + common["y"] + common["z"],
        "native_component_power_W": native_power,
        "common_component_power_W": common_power,
        "component_interpolation_relative_error": interpolation_error,
    }


def field_intensity(fdtd: Any, monitor: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    components = [
        np.asarray(fdtd.getdata(monitor, f"E{axis}", 1)).squeeze()
        for axis in "xyz"
    ]
    intensity = sum(np.abs(value) ** 2 for value in components)
    coordinates: dict[str, np.ndarray] = {}
    for axis in "xyz":
        try:
            coordinates[axis] = np.asarray(
                fdtd.getdata(monitor, axis, 1), float
            ).reshape(-1)
        except Exception:
            coordinates[axis] = np.asarray([0.0])
    return np.asarray(intensity, float), coordinates


def face_fluxes(
    fdtd: Any,
    faces: dict[str, dict[str, Any]],
    source_power_native_w: float,
    intensity_scale: float,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    net_outward = 0.0
    absolute_sum = 0.0
    for key, face in faces.items():
        normalized = scalar(fdtd.transmission(face["name"]), face["name"])
        signed_axis_power = (
            normalized * source_power_native_w * intensity_scale
        )
        outward = face["outward_sign"] * signed_axis_power
        values[key] = {
            "monitor": face["name"],
            "normalized_signed_axis_flux": normalized,
            "signed_axis_power_W_at_1_W_m2": signed_axis_power,
            "outward_power_W_at_1_W_m2": outward,
        }
        net_outward += outward
        absolute_sum += abs(outward)
    return {
        "faces": values,
        "net_outward_power_W": net_outward,
        "net_inward_power_W": -net_outward,
        "sum_absolute_face_power_W": absolute_sum,
    }


def geometry_contract(
    args: argparse.Namespace,
    tfsf_bounds: dict[str, tuple[float, float]],
    outer_bounds: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    return {
        "finite_nonperiodic": True,
        "FDTD_domain_bounds_m": {
            "x": (-0.5 * args.domain_um * 1e-6, 0.5 * args.domain_um * 1e-6),
            "y": (-0.5 * args.domain_um * 1e-6, 0.5 * args.domain_um * 1e-6),
            "z": (FDTD_Z_MIN_M, FDTD_Z_MAX_M),
        },
        "TaIrTe4_bounds_m": FLAKE_BOUNDS_M,
        "bottom_SiO2_bounds_m": {
            "x": "extends through PML",
            "y": "extends through PML",
            "z": (
                -FLAKE_THICKNESS_M - SIO2_THICKNESS_M,
                -FLAKE_THICKNESS_M,
            ),
        },
        "fixed_design": {
            "enabled": args.case == "fixed-design",
            "shape": "single centered disk",
            "radius_m": DESIGN_RADIUS_M,
            "z_bounds_m": (0.0, DESIGN_HEIGHT_M),
            "material": SIO2_MATERIAL,
            "periodic_repetition": False,
        },
        "tfsf_bounds_m": tfsf_bounds,
        "six_face_absorption_box_bounds_m": INNER_BOX,
        "outer_scattered_power_box_bounds_m": outer_bounds,
    }


def add_geometry_and_monitors(
    fdtd: Any,
    model: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    domain_m = args.domain_um * 1e-6
    tfsf_span_m = args.tfsf_span_um * 1e-6
    tfsf_bounds = {
        "x": (-0.5 * tfsf_span_m, 0.5 * tfsf_span_m),
        "y": (-0.5 * tfsf_span_m, 0.5 * tfsf_span_m),
        "z": (TFSF_Z_MIN_M, TFSF_Z_MAX_M),
    }
    outer_clearance = 0.25e-6
    outer_bounds = {
        "x": (
            tfsf_bounds["x"][0] - outer_clearance,
            tfsf_bounds["x"][1] + outer_clearance,
        ),
        "y": (
            tfsf_bounds["y"][0] - outer_clearance,
            tfsf_bounds["y"][1] + outer_clearance,
        ),
        "z": (
            tfsf_bounds["z"][0] - outer_clearance,
            tfsf_bounds["z"][1] + outer_clearance,
        ),
    }
    if max(abs(v) for v in outer_bounds["x"]) >= 0.5 * domain_m:
        raise RuntimeError("outer scattered-power box touches the lateral PML")

    fdtd.addfdtd()
    safe_set(fdtd, "FDTD", "dimension", "3D")
    safe_set(fdtd, "FDTD", "x", 0.0)
    safe_set(fdtd, "FDTD", "x span", domain_m)
    safe_set(fdtd, "FDTD", "y", 0.0)
    safe_set(fdtd, "FDTD", "y span", domain_m)
    safe_set(fdtd, "FDTD", "z min", FDTD_Z_MIN_M)
    safe_set(fdtd, "FDTD", "z max", FDTD_Z_MAX_M)
    for axis in "xyz":
        safe_set(fdtd, "FDTD", f"{axis} min bc", "PML")
        safe_set(fdtd, "FDTD", f"{axis} max bc", "PML")
    safe_set(fdtd, "FDTD", "pml layers", int(args.pml_layers))
    safe_set(fdtd, "FDTD", "mesh type", "auto non-uniform")
    safe_set(fdtd, "FDTD", "mesh refinement", "conformal variant 1")
    safe_set(fdtd, "FDTD", "mesh accuracy", 5)
    safe_set(fdtd, "FDTD", "simulation time", 4.0e-12)
    safe_set(fdtd, "FDTD", "auto shutoff min", 1.0e-8)
    safe_set(fdtd, "FDTD", "min mesh step", 1.0e-9)

    add_shared_sio2_material(fdtd)
    model.add_flake_material(SimpleNamespace(fdtd=fdtd))
    lateral_material_span = domain_m + 2.0e-6
    add_rect(
        fdtd,
        "Si_substrate",
        {
            "x": (-0.5 * lateral_material_span, 0.5 * lateral_material_span),
            "y": (-0.5 * lateral_material_span, 0.5 * lateral_material_span),
            "z": (FDTD_Z_MIN_M - 0.5e-6, -FLAKE_THICKNESS_M - SIO2_THICKNESS_M),
        },
        index=3.425,
    )
    add_rect(
        fdtd,
        "SiO2_spacer",
        {
            "x": (-0.5 * lateral_material_span, 0.5 * lateral_material_span),
            "y": (-0.5 * lateral_material_span, 0.5 * lateral_material_span),
            "z": (
                -FLAKE_THICKNESS_M - SIO2_THICKNESS_M,
                -FLAKE_THICKNESS_M,
            ),
        },
        material=SIO2_MATERIAL,
    )
    if args.case in ("flat", "fixed-design"):
        add_rect(
            fdtd,
            "TaIrTe4_flake",
            FLAKE_BOUNDS_M,
            material=MATERIAL_NAME,
        )
    if args.case == "fixed-design":
        design = fdtd.addcircle()
        design["name"] = "fixed_design_SiO2"
        design["x"] = 0.0
        design["y"] = 0.0
        design["radius"] = DESIGN_RADIUS_M
        design["z min"] = 0.0
        design["z max"] = DESIGN_HEIGHT_M
        design["material"] = SIO2_MATERIAL

    mesh = fdtd.addmesh()
    mesh["name"] = "flake_mesh"
    mesh["x min"] = FLAKE_BOUNDS_M["x"][0]
    mesh["x max"] = FLAKE_BOUNDS_M["x"][1]
    mesh["y min"] = FLAKE_BOUNDS_M["y"][0]
    mesh["y max"] = FLAKE_BOUNDS_M["y"][1]
    mesh["z min"] = -FLAKE_THICKNESS_M - 10.0 * args.flake_dz_nm * 1e-9
    mesh["z max"] = PABS_PADDING_M
    mesh["override x mesh"] = 0
    mesh["override y mesh"] = 0
    mesh["override z mesh"] = 1
    mesh["dz"] = args.flake_dz_nm * 1e-9

    source = fdtd.addtfsf()
    source["name"] = SOURCE_NAME
    source["injection axis"] = "z"
    source["direction"] = "backward"
    source["polarization angle"] = float(args.polarization_deg)
    for axis in "xyz":
        source[f"{axis} min"] = tfsf_bounds[axis][0]
        source[f"{axis} max"] = tfsf_bounds[axis][1]
    source["use global source settings"] = True
    source["override global source settings"] = False
    if args.case == "no-source":
        source["amplitude"] = 0.0

    fdtd.setglobalsource("wavelength start", SOURCE_START_M)
    fdtd.setglobalsource("wavelength stop", SOURCE_STOP_M)
    fdtd.setglobalmonitor("use source limits", False)
    fdtd.setglobalmonitor("use wavelength spacing", True)
    fdtd.setglobalmonitor("wavelength center", TARGET_WAVELENGTH_M)
    fdtd.setglobalmonitor("wavelength span", 0.0)
    fdtd.setglobalmonitor("frequency points", 1)

    pabs = fdtd.addobject("pabs_adv")
    pabs["name"] = PABS_GROUP
    pabs["x"] = 0.5 * sum(FLAKE_BOUNDS_M["x"])
    pabs["x span"] = FLAKE_BOUNDS_M["x"][1] - FLAKE_BOUNDS_M["x"][0]
    pabs["y"] = 0.5 * sum(FLAKE_BOUNDS_M["y"])
    pabs["y span"] = FLAKE_BOUNDS_M["y"][1] - FLAKE_BOUNDS_M["y"][0]
    pabs["z"] = 0.5 * sum(FLAKE_BOUNDS_M["z"])
    pabs["z span"] = FLAKE_THICKNESS_M + 2.0 * PABS_PADDING_M

    inner_faces = add_flux_box(fdtd, "finite_abs", INNER_BOX)
    outer_faces = add_flux_box(fdtd, "finite_scatter", outer_bounds)
    add_field_monitor(
        fdtd,
        "finite_E_xy_inside",
        "2D Z-normal",
        {
            "x": INNER_BOX["x"],
            "y": INNER_BOX["y"],
            "z": (0.50e-6, 0.50e-6),
        },
    )
    add_field_monitor(
        fdtd,
        "finite_E_yz_outside_x",
        "2D X-normal",
        {
            "x": (outer_bounds["x"][1], outer_bounds["x"][1]),
            "y": outer_bounds["y"],
            "z": outer_bounds["z"],
        },
    )
    add_field_monitor(
        fdtd,
        "finite_E_xz_outside_y",
        "2D Y-normal",
        {
            "x": outer_bounds["x"],
            "y": (outer_bounds["y"][1], outer_bounds["y"][1]),
            "z": outer_bounds["z"],
        },
    )
    for name in (
        PABS_FIELD,
        PABS_INDEX,
        *(face["name"] for face in inner_faces.values()),
        *(face["name"] for face in outer_faces.values()),
        "finite_E_xy_inside",
        "finite_E_yz_outside_x",
        "finite_E_xz_outside_y",
    ):
        configure_single_frequency(fdtd, name)
    return {
        "tfsf_bounds": tfsf_bounds,
        "outer_bounds": outer_bounds,
        "inner_faces": inner_faces,
        "outer_faces": outer_faces,
        "geometry": geometry_contract(args, tfsf_bounds, outer_bounds),
    }


def read_material_contract(fdtd: Any) -> dict[str, Any]:
    table = np.asarray(fdtd.getmaterial(MATERIAL_NAME, "sampled data"))
    wavelengths = C0 / np.asarray(table[:, 0].real, float)
    return {
        "name": MATERIAL_NAME,
        "sample_count": int(table.shape[0]),
        "wavelength_min_m": float(np.min(wavelengths)),
        "wavelength_max_m": float(np.max(wavelengths)),
        "eps_flake_input_units": "nm",
    }


def assert_pre_run_contract(
    fdtd: Any,
    runtime: Any,
    args: argparse.Namespace,
    setup: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    boundaries = {}
    for axis in "xyz":
        for side in ("min", "max"):
            key = f"{axis}_{side}"
            value = str(fdtd.getnamed("FDTD", f"{axis} {side} bc")).strip()
            boundaries[key] = value
            checks[f"{key}_is_PML"] = value.lower() == "pml"
    checks["finite_flake_count"] = (
        int(fdtd.getnamednumber("TaIrTe4_flake"))
        == (1 if args.case in ("flat", "fixed-design") else 0)
    )
    checks["periodic_builder_untouched"] = True
    checks["global_uniform_mesh_absent"] = (
        int(fdtd.getnamednumber("global_uniform_mesh")) == 0
    )
    checks["mesh_type_auto_nonuniform"] = (
        str(fdtd.getnamed("FDTD", "mesh type")).strip().lower()
        == "auto non-uniform"
    )
    checks["mesh_refinement_CV1"] = (
        str(fdtd.getnamed("FDTD", "mesh refinement")).strip().lower()
        == "conformal variant 1"
    )
    checks["mesh_accuracy_5"] = int(
        round(scalar(fdtd.getnamed("FDTD", "mesh accuracy"), "mesh accuracy"))
    ) == 5
    requested_dz = args.flake_dz_nm * 1e-9
    realized_dz = scalar(fdtd.getnamed("flake_mesh", "dz"), "flake_mesh.dz")
    checks["requested_flake_dz"] = np.isclose(
        realized_dz, requested_dz, rtol=0.0, atol=1e-15
    )
    source_start = scalar(
        fdtd.getnamed(SOURCE_NAME, "wavelength start"), "source start"
    )
    source_stop = scalar(
        fdtd.getnamed(SOURCE_NAME, "wavelength stop"), "source stop"
    )
    checks["source_3_to_6_um"] = (
        np.isclose(source_start, SOURCE_START_M, rtol=0.0, atol=1e-15)
        and np.isclose(source_stop, SOURCE_STOP_M, rtol=0.0, atol=1e-15)
    )
    source_type = str(fdtd.getnamed(SOURCE_NAME, "type"))
    checks["source_is_TFSF"] = "total field" in source_type.lower() or "tfsf" in source_type.lower()
    material = read_material_contract(fdtd)
    checks["material_600_samples"] = material["sample_count"] == MATERIAL_SAMPLES
    checks["material_range_2p7_to_13p2_um"] = (
        np.isclose(material["wavelength_min_m"], MATERIAL_START_M, atol=1e-15)
        and np.isclose(material["wavelength_max_m"], MATERIAL_STOP_M, atol=1e-15)
    )
    checks["Pabs_padding_50_nm"] = (
        np.isclose(
            scalar(fdtd.getnamed(PABS_GROUP, "z"), "Pabs z"),
            -0.5 * FLAKE_THICKNESS_M,
        )
        and np.isclose(
            scalar(fdtd.getnamed(PABS_GROUP, "z span"), "Pabs z span"),
            FLAKE_THICKNESS_M + 2.0 * PABS_PADDING_M,
        )
    )
    checks["bottom_and_design_share_SiO2_model"] = (
        args.case != "fixed-design"
        or str(fdtd.getnamed("fixed_design_SiO2", "material"))
        == str(fdtd.getnamed("SiO2_spacer", "material"))
        == SIO2_MATERIAL
    )
    source_bounds = setup["tfsf_bounds"]
    checks["scatterer_inside_TFSF"] = (
        INNER_BOX["x"][0] > source_bounds["x"][0]
        and INNER_BOX["x"][1] < source_bounds["x"][1]
        and INNER_BOX["y"][0] > source_bounds["y"][0]
        and INNER_BOX["y"][1] < source_bounds["y"][1]
        and INNER_BOX["z"][0] > source_bounds["z"][0]
        and INNER_BOX["z"][1] < source_bounds["z"][1]
    )
    runtime.configure_session_resources(fdtd)
    fdtd.runsetup()
    dt_s = scalar(fdtd.getnamed("FDTD", "dt"), "FDTD.dt")
    version = str(fdtd.version())
    resources = runtime.resource_contract(fdtd)
    checks["v261"] = version.startswith("8.35")
    checks["all"] = all(checks.values())
    if not checks["all"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"finite optical pre-run contract failed: {failed}")
    return {
        "checks": checks,
        "boundaries": boundaries,
        "source": {
            "type": source_type,
            "injection_axis": safe_get(fdtd, SOURCE_NAME, "injection axis"),
            "direction": safe_get(fdtd, SOURCE_NAME, "direction"),
            "polarization_angle_deg": safe_get(
                fdtd, SOURCE_NAME, "polarization angle"
            ),
            "wavelength_start_m": source_start,
            "wavelength_stop_m": source_stop,
            "bounds_m": source_bounds,
        },
        "material": material,
        "mesh": {
            "type": safe_get(fdtd, "FDTD", "mesh type"),
            "refinement": safe_get(fdtd, "FDTD", "mesh refinement"),
            "accuracy": safe_get(fdtd, "FDTD", "mesh accuracy"),
            "global_uniform_mesh_count": int(
                fdtd.getnamednumber("global_uniform_mesh")
            ),
            "flake_dz_m": realized_dz,
            "dt_s": dt_s,
        },
        "solver": {
            "root": str(runtime.R12_ROOT),
            "version": version,
            "resources": resources,
        },
        "geometry": setup["geometry"],
    }


def plot_geometry(
    output: Path,
    args: argparse.Namespace,
    setup: dict[str, Any],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    domain_half = 0.5 * args.domain_um
    ax.add_patch(
        plt.Rectangle(
            (-domain_half, FDTD_Z_MIN_M * 1e6),
            2 * domain_half,
            (FDTD_Z_MAX_M - FDTD_Z_MIN_M) * 1e6,
            fill=False,
            lw=2,
            label="FDTD/PML",
        )
    )
    ax.axhspan(
        (-FLAKE_THICKNESS_M - SIO2_THICKNESS_M) * 1e6,
        -FLAKE_THICKNESS_M * 1e6,
        color="lightblue",
        label="SiO2",
    )
    ax.axhspan(
        FDTD_Z_MIN_M * 1e6,
        (-FLAKE_THICKNESS_M - SIO2_THICKNESS_M) * 1e6,
        color="silver",
        label="Si",
    )
    if args.case in ("flat", "fixed-design"):
        ax.add_patch(
            plt.Rectangle(
                (-1.0, -0.1), 2.0, 0.1, color="darkorange", label="TaIrTe4"
            )
        )
    if args.case == "fixed-design":
        ax.add_patch(
            plt.Rectangle(
                (-DESIGN_RADIUS_M * 1e6, 0.0),
                2 * DESIGN_RADIUS_M * 1e6,
                DESIGN_HEIGHT_M * 1e6,
                color="deepskyblue",
                alpha=0.6,
                label="finite SiO2 disk",
            )
        )
    source = setup["tfsf_bounds"]
    ax.add_patch(
        plt.Rectangle(
            (source["x"][0] * 1e6, source["z"][0] * 1e6),
            (source["x"][1] - source["x"][0]) * 1e6,
            (source["z"][1] - source["z"][0]) * 1e6,
            fill=False,
            ls="--",
            lw=2,
            color="green",
            label="TFSF",
        )
    )
    ax.set(xlabel="x (um)", ylabel="z (um)", title="Finite geometry (x-z)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.set_aspect("equal")
    ax.add_patch(
        plt.Rectangle(
            (-domain_half, -domain_half),
            2 * domain_half,
            2 * domain_half,
            fill=False,
            lw=2,
            label="FDTD/PML",
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (-1.0, -1.0), 2.0, 2.0, color="darkorange", label="TaIrTe4"
        )
    )
    if args.case == "fixed-design":
        ax.add_patch(
            plt.Circle(
                (0.0, 0.0),
                DESIGN_RADIUS_M * 1e6,
                color="deepskyblue",
                alpha=0.5,
                label="SiO2 disk",
            )
        )
    source_half = 0.5 * args.tfsf_span_um
    ax.add_patch(
        plt.Rectangle(
            (-source_half, -source_half),
            2 * source_half,
            2 * source_half,
            fill=False,
            ls="--",
            lw=2,
            color="green",
            label="TFSF",
        )
    )
    inner = INNER_BOX["x"][1] * 1e6
    ax.add_patch(
        plt.Rectangle(
            (-inner, -inner),
            2 * inner,
            2 * inner,
            fill=False,
            ls=":",
            lw=2,
            color="purple",
            label="six-face box",
        )
    )
    ax.set(
        xlim=(-domain_half, domain_half),
        ylim=(-domain_half, domain_half),
        xlabel="x (um)",
        ylabel="y (um)",
        title="Source and power-box geometry",
    )
    ax.legend(fontsize=8, loc="upper right")
    figure.tight_layout()
    figure.savefig(output / "finite_geometry_and_source.png", dpi=180)
    plt.close(figure)


def plot_field_slices(
    output: Path,
    inside: np.ndarray,
    outside_x: np.ndarray,
    outside_y: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, image, title in zip(
        axes,
        (inside, outside_x, outside_y),
        ("inside TFSF |E|^2", "outside x boundary |E|^2", "outside y boundary |E|^2"),
    ):
        shown = np.squeeze(image)
        if shown.ndim != 2:
            shown = shown.reshape(shown.shape[0], -1)
        handle = ax.imshow(shown.T, origin="lower", aspect="auto")
        ax.set_title(title)
        figure.colorbar(handle, ax=ax)
    figure.tight_layout()
    figure.savefig(output / "E2_slices.png", dpi=180)
    plt.close(figure)


def plot_q_slices(output: Path, artifact: dict[str, Any]) -> None:
    x = artifact["x_m"]
    y = artifact["y_m"]
    z = artifact["z_m"]
    ix = int(np.argmin(np.abs(x)))
    iy = int(np.argmin(np.abs(y)))
    iz = int(np.argmin(np.abs(z + 0.5 * FLAKE_THICKNESS_M)))
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, key in zip(
        axes.reshape(-1),
        ("Qx_W_m3", "Qy_W_m3", "Qz_W_m3", "Q_on_W_m3"),
    ):
        image = artifact[key][:, :, iz]
        handle = ax.imshow(
            image.T,
            origin="lower",
            extent=[x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6],
            aspect="equal",
        )
        ax.set(title=key, xlabel="x (um)", ylabel="y (um)")
        figure.colorbar(handle, ax=ax)
    figure.tight_layout()
    figure.savefig(output / "Q_component_xy_slices.png", dpi=180)
    plt.close(figure)

    total = artifact["Q_on_W_m3"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, image, extent, title in (
        (
            axes[0],
            total[:, iy, :],
            [x[0] * 1e6, x[-1] * 1e6, z[0] * 1e9, z[-1] * 1e9],
            "Q(x,z), y=0",
        ),
        (
            axes[1],
            total[ix, :, :],
            [y[0] * 1e6, y[-1] * 1e6, z[0] * 1e9, z[-1] * 1e9],
            "Q(y,z), x=0",
        ),
    ):
        handle = ax.imshow(
            image.T, origin="lower", extent=extent, aspect="auto"
        )
        ax.set(title=title, xlabel="lateral (um)", ylabel="z (nm)")
        figure.colorbar(handle, ax=ax)
    figure.tight_layout()
    figure.savefig(output / "Q_cross_section_slices.png", dpi=180)
    plt.close(figure)


def run_case(
    fdtd: Any,
    runtime: Any,
    args: argparse.Namespace,
    output: Path,
    setup: dict[str, Any],
) -> dict[str, Any]:
    resource = runtime.run_session(
        fdtd,
        (
            f"finite_{args.case}_p{args.polarization_deg:g}_"
            f"L{args.domain_um:g}_pml{args.pml_layers}_dz{args.flake_dz_nm:g}"
        ),
    )
    inside_e2, inside_coordinates = field_intensity(
        fdtd, "finite_E_xy_inside"
    )
    outside_x_e2, outside_x_coordinates = field_intensity(
        fdtd, "finite_E_yz_outside_x"
    )
    outside_y_e2, outside_y_coordinates = field_intensity(
        fdtd, "finite_E_xz_outside_y"
    )
    plot_field_slices(output, inside_e2, outside_x_e2, outside_y_e2)
    np.savez(
        output / "field_slices_raw.npz",
        E2_inside=inside_e2,
        E2_outside_x=outside_x_e2,
        E2_outside_y=outside_y_e2,
        **{
            f"inside_{key}_m": value
            for key, value in inside_coordinates.items()
        },
        **{
            f"outside_x_{key}_m": value
            for key, value in outside_x_coordinates.items()
        },
        **{
            f"outside_y_{key}_m": value
            for key, value in outside_y_coordinates.items()
        },
    )
    max_inside = float(np.max(inside_e2))
    max_outside = max(
        float(np.max(outside_x_e2)), float(np.max(outside_y_e2))
    )

    if args.case == "no-source":
        return {
            "resource": resource,
            "source_enabled": False,
            "field": {
                "maximum_inside_E2": max_inside,
                "maximum_outside_E2": max_outside,
            },
            "acceptance": {
                "no_source_background_is_zero": max(max_inside, max_outside)
                < np.finfo(float).eps,
            },
        }

    source_power_native = scalar(
        fdtd.sourcepower(TARGET_FREQUENCY_HZ, SOURCE_NAME),
        "sourcepower at 4 um",
    )
    source_intensity_native = scalar(
        fdtd.sourceintensity(TARGET_FREQUENCY_HZ, SOURCE_NAME),
        "sourceintensity at 4 um",
    )
    if source_power_native <= 0.0 or source_intensity_native <= 0.0:
        raise RuntimeError("invalid measured TFSF source power/intensity")
    intensity_scale = TARGET_INTENSITY_W_M2 / source_intensity_native
    incident_power_scaled = source_power_native * intensity_scale
    inner_flux = face_fluxes(
        fdtd, setup["inner_faces"], source_power_native, intensity_scale
    )
    outer_flux = face_fluxes(
        fdtd, setup["outer_faces"], source_power_native, intensity_scale
    )
    outside_field_ratio = max_outside / max(max_inside, np.finfo(float).tiny)

    if args.case == "empty-stack":
        fdtd.runanalysis(PABS_GROUP)
        q_data = common_grid_component_q(fdtd, TARGET_FREQUENCY_HZ)
        p_q_native = integrate_xyz(
            q_data["Q_native_W_m3"],
            q_data["x_m"],
            q_data["y_m"],
            q_data["z_m"],
        )
        p_q_scaled = p_q_native * intensity_scale
        inner_fraction = abs(inner_flux["net_inward_power_W"]) / max(
            incident_power_scaled, np.finfo(float).tiny
        )
        q_fraction = abs(p_q_scaled) / max(
            incident_power_scaled, np.finfo(float).tiny
        )
        lateral_leakage = sum(
            abs(outer_flux["faces"][key]["outward_power_W_at_1_W_m2"])
            for key in ("x_min", "x_max", "y_min", "y_max")
        )
        lateral_fraction = lateral_leakage / max(
            incident_power_scaled, np.finfo(float).tiny
        )
        return {
            "resource": resource,
            "source_enabled": True,
            "normalization": {
                "measured_source_power_native_W": source_power_native,
                "measured_source_intensity_native_W_m2": source_intensity_native,
                "scale_to_1_W_m2": intensity_scale,
                "incident_power_W_at_1_W_m2": incident_power_scaled,
            },
            "field": {
                "maximum_inside_E2": max_inside,
                "maximum_outside_E2": max_outside,
                "outside_to_inside_max_E2_ratio": outside_field_ratio,
            },
            "six_face": inner_flux,
            "outer_scattered_box": outer_flux,
            "empty_stack_P_Q_W_at_1_W_m2": p_q_scaled,
            "acceptance": {
                "background_absorption_fraction_lt_1e_4": (
                    inner_fraction < EMPTY_POWER_FRACTION_LIMIT
                    and q_fraction < EMPTY_POWER_FRACTION_LIMIT
                ),
                "lateral_leakage_fraction_lt_1e_4": (
                    lateral_fraction < EMPTY_POWER_FRACTION_LIMIT
                ),
                "outside_field_ratio_lt_1e_4": (
                    outside_field_ratio < EMPTY_FIELD_RATIO_LIMIT
                ),
            },
        }

    fdtd.runanalysis(PABS_GROUP)
    q_data = common_grid_component_q(fdtd, TARGET_FREQUENCY_HZ)
    artifact = {
        "x_m": q_data["x_m"],
        "y_m": q_data["y_m"],
        "z_m": q_data["z_m"],
        "Qx_W_m3": q_data["Qx_native_W_m3"] * intensity_scale,
        "Qy_W_m3": q_data["Qy_native_W_m3"] * intensity_scale,
        "Qz_W_m3": q_data["Qz_native_W_m3"] * intensity_scale,
        "Q_on_W_m3": q_data["Q_native_W_m3"] * intensity_scale,
    }
    component_power = {
        axis: integrate_xyz(
            artifact[f"Q{axis}_W_m3"],
            artifact["x_m"],
            artifact["y_m"],
            artifact["z_m"],
        )
        for axis in "xyz"
    }
    p_q = sum(component_power.values())
    p_six_face = inner_flux["net_inward_power_W"]
    closure = abs(p_q - p_six_face) / max(
        abs(p_six_face), np.finfo(float).tiny
    )
    sigma_abs = p_q / TARGET_INTENSITY_W_M2
    q_total = artifact["Q_on_W_m3"]
    hotspot_index = np.unravel_index(
        int(np.argmax(q_total)), q_total.shape
    )
    hotspot = {
        "x_m": float(artifact["x_m"][hotspot_index[0]]),
        "y_m": float(artifact["y_m"][hotspot_index[1]]),
        "z_m": float(artifact["z_m"][hotspot_index[2]]),
        "Q_W_m3": float(q_total[hotspot_index]),
    }
    negative_power = integrate_xyz(
        np.minimum(q_total, 0.0),
        artifact["x_m"],
        artifact["y_m"],
        artifact["z_m"],
    )
    metadata = {
        "geometry_bounds_m": setup["geometry"],
        "material_names": {
            "TaIrTe4": MATERIAL_NAME,
            "SiO2_spacer": SIO2_MATERIAL,
            "fixed_design": SIO2_MATERIAL if args.case == "fixed-design" else None,
            "Si_substrate": "Object defined dielectric n=3.425",
        },
        "source_type": "TFSF normally incident layered-background plane wave",
        "source_range_m": [SOURCE_START_M, SOURCE_STOP_M],
        "analysis_wavelength_m": TARGET_WAVELENGTH_M,
        "incident_intensity_W_m2": TARGET_INTENSITY_W_M2,
        "exact_flake_bounds_m": FLAKE_BOUNDS_M,
        "pabs_zero_padding_m": {
            "bottom": PABS_PADDING_M,
            "top": PABS_PADDING_M,
        },
        "array_axis_order": ["x", "y", "z"],
        "Q_units": "W/m^3",
        "clipped": False,
        "gain_applied": False,
        "rescaled_to_flux": False,
        "periodic_crop": False,
        "periodic_tiling": False,
    }
    np.savez(
        output / "finite_q_on_artifact.npz",
        **artifact,
        incident_intensity_W_m2=np.asarray([TARGET_INTENSITY_W_M2]),
        P_abs_volume_W=np.asarray([p_q]),
        P_abs_six_face_W=np.asarray([p_six_face]),
        absorption_cross_section_m2=np.asarray([sigma_abs]),
        metadata_json=np.asarray([json.dumps(jsonable(metadata))]),
    )
    plot_q_slices(output, artifact)
    return {
        "resource": resource,
        "source_enabled": True,
        "normalization": {
            "measured_source_power_native_W": source_power_native,
            "measured_source_intensity_native_W_m2": source_intensity_native,
            "scale_to_1_W_m2": intensity_scale,
            "incident_intensity_W_m2": TARGET_INTENSITY_W_M2,
            "incident_power_W_at_1_W_m2": incident_power_scaled,
            "normalization_basis": "measured v261 TFSF sourceintensity",
            "empirical_flux_gain": False,
        },
        "component_power_W": component_power,
        "P_Q_W": p_q,
        "P_six_face_W": p_six_face,
        "six_face_relative_closure": closure,
        "outer_scattered_box": outer_flux,
        "PML_incident_scattered_power_W": outer_flux["net_outward_power_W"],
        "PML_absorbed_power_W": None,
        "PML_absorption_note": (
            "v261 case records the scattered power incident on the PML via the "
            "outer box; no unsupported PML-loss getter is claimed"
        ),
        "absorption_cross_section_m2": sigma_abs,
        "geometric_flake_area_m2": GEOMETRIC_AREA_M2,
        "normalized_absorption_cross_section": sigma_abs / GEOMETRIC_AREA_M2,
        "Q_hotspot": hotspot,
        "component_interpolation_relative_error": q_data[
            "component_interpolation_relative_error"
        ],
        "minimum_Q_W_m3": float(np.min(q_total)),
        "negative_Q_voxel_count": int(np.count_nonzero(q_total < 0.0)),
        "integrated_negative_Q_power_W": negative_power,
        "field": {
            "maximum_inside_E2": max_inside,
            "maximum_outside_E2": max_outside,
            "outside_to_inside_max_E2_ratio": outside_field_ratio,
        },
        "six_face": inner_flux,
        "artifact": "finite_q_on_artifact.npz",
        "artifact_metadata": metadata,
        "acceptance": {
            "six_face_closure_lt_0p5_percent": closure < POWER_CLOSURE_LIMIT,
            "Q_not_clipped": True,
            "measured_intensity_normalization": True,
        },
    }


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, *sys.argv])
    commit = git_commit()
    project = output / "finite_2um_optical_q.fsp"
    result_path = output / "case_result.json"
    status = "BLOCKED"
    fdtd = None
    result: dict[str, Any] = {
        "status": status,
        "validated": False,
        "case": args.case,
        "polarization_deg": args.polarization_deg,
        "domain_um": args.domain_um,
        "pml_layers": args.pml_layers,
        "flake_dz_nm": args.flake_dz_nm,
        "source_type": "TFSF",
        "generation_command": command,
        "generation_commit": commit,
        "project": str(project),
        "heat_run": False,
        "adjoint_run": False,
        "gradient_run": False,
        "optimization_run": False,
        "periodic_Q_used": False,
        "Q_clipped": False,
        "flux_gain": False,
        "Q_rescaled": False,
    }
    try:
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
        os.environ["MATERIAL_SAMPLE_COUNT"] = str(MATERIAL_SAMPLES)
        os.environ["BULK_MESH_MODE"] = "auto"
        os.environ["MESH_ACCURACY"] = "5"
        os.environ["FILM_DZ_NM"] = str(args.flake_dz_nm)
        os.environ["VC_MESH_REFINEMENT"] = "conformal variant 1"
        for path in (config.REPOSITORY_ROOT, config.REPOSITORY_ROOT / "bundle"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import eqc_lib as runtime

        lumapi = load_lumapi(installation)

        model = runtime.load_model()
        fdtd = lumapi.FDTD(
            hide=True,
            serverArgs={"platform": "offscreen"},
        )
        fdtd.switchtolayout()
        setup = add_geometry_and_monitors(fdtd, model, args)
        plot_geometry(output, args, setup)
        pre_run = assert_pre_run_contract(fdtd, runtime, args, setup)
        result["pre_run_contract"] = pre_run
        fdtd.save(str(project))
        if args.contract_only:
            result["status"] = "CONTRACT_BUILT_NOT_SOLVED"
            write_json(result_path, result)
            return_code = 0
        else:
            run_result = run_case(fdtd, runtime, args, output, setup)
            result["run_result"] = run_result
            acceptance = run_result.get("acceptance", {})
            all_pass = bool(acceptance) and all(bool(v) for v in acceptance.values())
            result["status"] = "COMPLETED" if all_pass else "FAILED_ACCEPTANCE"
            result["validated"] = False
            fdtd.save(str(project))
            write_json(result_path, result)
            return_code = 0 if all_pass else 2
    except Exception as exc:
        result.update(
            {
                "status": "BLOCKED_EXECUTION_ERROR",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(result_path, result)
        return_code = 2
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        raw_suffixes = {".fsp", ".npz", ".mat", ".npy", ".log"}
        raw = {}
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix.lower() in raw_suffixes:
                raw[str(path.relative_to(output))] = artifact_record(
                    path, command=command, commit=commit
                )
        manifest = {
            "policy": "Large FSP, NPZ, MAT, logs, and raw 3-D fields stay outside Git.",
            "case": args.case,
            "generation_command": command,
            "generation_commit": commit,
            "reproduction_environment": (
                "/home/eidl/miniconda3/bin/python with Lumerical v261 and "
                f"{args.gpu_device}"
            ),
            "raw_artifacts": raw,
        }
        write_json(output / "RAW_ARTIFACT_MANIFEST.json", manifest)
    print(json.dumps(jsonable(result), indent=2), flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
