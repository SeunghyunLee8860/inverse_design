#!/usr/bin/env python3
"""GPU-only finite-array Maxwell/Q wrapper for the T 11x15 and Z 1x3 cases.

The finite source, layer stack, PML, material and Q-closure contracts are
inherited without modification from stage 78.  Only the number and lateral
placement of the floating top-Au resonators changes.  The Z array uses the
paper-period reconstruction (P1=5.1 um along b, P2=2.6 um along a), not the
expanded square project supercell used by the earlier isolated-Z control.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
BASE_RUNNER = HERE / "78_run_v261_finite_t_z_gaussian_q.py"
GEOMETRY_MODULE = HERE / "05_actual_metasurface_geometry.py"
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_array_Q")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def array_contract(architecture: str) -> dict[str, object]:
    geometry = load(GEOMETRY_MODULE, "finite_array_geometry_contract")
    if architecture == "T":
        primitive = geometry.inverse_t_mir_4750nm()
        nx, ny = 11, 15
        identity = "finite T 11x15 array using the figure-digitized periodic-cell pitch"
    elif architecture == "Z":
        primitive = geometry.z_m2_5300nm_figure_period_corrected_tairte4_v3("LH")
        nx, ny = 1, 3
        identity = "finite Z 1x3 column using paper P1/P2 directions and pitches"
    else:
        raise ValueError("ARRAY_ARCHITECTURE must be T or Z")
    return {
        "architecture": architecture,
        "nx_along_b": nx,
        "ny_along_a": ny,
        "period_b_m": float(primitive.period_x_nm) * 1e-9,
        "period_a_m": float(primitive.period_y_nm) * 1e-9,
        "primitive": primitive.as_dict(),
        "identity": identity,
        "finite_not_periodic": True,
        "top_Au_is_floating_in_electrical_stage": True,
    }


def shifted_polygons(contract: dict[str, object]) -> list[dict[str, object]]:
    primitive = contract["primitive"]
    nx = int(contract["nx_along_b"])
    ny = int(contract["ny_along_a"])
    px_nm = float(contract["period_b_m"]) * 1e9
    py_nm = float(contract["period_a_m"]) * 1e9
    output: list[dict[str, object]] = []
    for ix in range(nx):
        x_shift = (ix - 0.5 * (nx - 1)) * px_nm
        for iy in range(ny):
            y_shift = (iy - 0.5 * (ny - 1)) * py_nm
            for primitive_index, item in enumerate(primitive["polygons"]):
                vertices = np.asarray(item["vertices_nm"], dtype=float)
                vertices[:, 0] += x_shift
                vertices[:, 1] += y_shift
                shifted = dict(item)
                shifted["name"] = (
                    f"{item['name']}_array_ix{ix:02d}_iy{iy:02d}_p{primitive_index:02d}"
                )
                shifted["vertices_nm"] = vertices.tolist()
                shifted["array_index"] = [ix, iy]
                shifted["cell_center_nm"] = [x_shift, y_shift]
                output.append(shifted)
    return output


def add_array_geometry(runner, fdtd, architecture, include_top_au, contract, base):
    if not include_top_au:
        raise RuntimeError("finite-array certificate requires top Au")
    # Add the identical finite layer stack and flake, but suppress the original
    # single top-Au object before adding the finite array below.
    result = runner._original_add_geometry(fdtd, architecture, False, contract, base)
    array = array_contract(architecture)
    polygons = shifted_polygons(array)
    names: list[str] = []
    all_vertices = []
    for item in polygons:
        polygon = fdtd.addpoly()
        polygon["name"] = f"finite_array_{architecture}_{item['name']}"
        polygon["material"] = base.AU_MATERIAL
        vertices_m = np.asarray(item["vertices_nm"], dtype=float) * 1e-9
        polygon["vertices"] = vertices_m
        polygon["z min"] = float(item["z_min_nm"]) * 1e-9
        polygon["z max"] = float(item["z_max_nm"]) * 1e-9
        names.append(str(polygon["name"]))
        all_vertices.append(vertices_m)

    vertices = np.concatenate(all_vertices, axis=0)
    margin = 0.10e-6
    bounds = {
        "x": [float(np.min(vertices[:, 0]) - margin), float(np.max(vertices[:, 0]) + margin)],
        "y": [float(np.min(vertices[:, 1]) - margin), float(np.max(vertices[:, 1]) + margin)],
        "z": [0.0, max(float(item["z_max_nm"]) for item in polygons) * 1e-9 + 20e-9],
    }
    mesh = fdtd.addmesh()
    mesh["name"] = f"finite_array_{architecture}_top_Au_mesh_25nm"
    for axis in "xyz":
        mesh[f"{axis} min"], mesh[f"{axis} max"] = bounds[axis]
        mesh[f"override {axis} mesh"] = True
    mesh["dx"] = 25e-9
    mesh["dy"] = 25e-9
    mesh["dz"] = 5e-9

    result.update(
        {
            "top_Au_present": True,
            "top_Au_objects": names,
            "top_Au_polygons": polygons,
            "array_contract": array,
            "array_mesh_override": {
                "name": str(mesh["name"]),
                "bounds_m": bounds,
                "dx_m": 25e-9,
                "dy_m": 25e-9,
                "dz_m": 5e-9,
            },
        }
    )
    return result


def main() -> int:
    architecture = os.environ.get("ARRAY_ARCHITECTURE", "T").strip().upper()
    polarization = os.environ.get("ARRAY_POLARIZATION", "Ea").strip()
    allowed_polarizations = {
        "Ea", "Eb", "linear_plus_45", "linear_minus_45",
    }
    if architecture not in {"T", "Z"} or polarization not in allowed_polarizations:
        raise ValueError(
            "ARRAY_ARCHITECTURE=T/Z and ARRAY_POLARIZATION in "
            "Ea/Eb/linear_plus_45/linear_minus_45 are required"
        )
    label = "T11x15" if architecture == "T" else "Z1x3"
    output = Path(
        os.environ.get("ARRAY_Q_OUTPUT", str(RAW_ROOT / f"{label}_{polarization}_Au_on"))
    ).expanduser().resolve()
    os.environ["FINITE_Q_ARCHITECTURE"] = architecture
    os.environ["FINITE_Q_POLARIZATION"] = polarization
    os.environ["FINITE_Q_TOP_AU"] = "on"
    os.environ["FINITE_Q_OUTPUT"] = str(output)

    runner = load(BASE_RUNNER, "finite_array_base_q_runner")
    runner._original_add_geometry = runner._add_geometry
    runner._add_geometry = lambda fdtd, arch, include, contract, base: add_array_geometry(
        runner, fdtd, arch, include, contract, base
    )
    exit_code = int(runner.main())

    result_path = output / f"FINITE_{architecture}_{polarization}_Au_on_Q.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text())
        payload["architecture_variant"] = label
        payload["array_contract"] = array_contract(architecture)
        if exit_code == 0:
            payload["status"] = f"VALIDATED_FINITE_{label}_{polarization}_VOLUMETRIC_Q"
            payload["classification"] = (
                "finite nonperiodic Gaussian Maxwell/Q array certificate; "
                "no thermal/electrical/PTE/adjoint/optimization"
            )
        result_path.write_text(json.dumps(payload, indent=2) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
