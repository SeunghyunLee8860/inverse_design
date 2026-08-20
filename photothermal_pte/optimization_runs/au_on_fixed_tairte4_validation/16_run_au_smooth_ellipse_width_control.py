#!/usr/bin/env python3
"""Exact scalar-Au smooth-ellipse width control at 10 um.

The in-plane Au boundary is a high-resolution counter-clockwise polygonal
approximation to an ellipse.  Varying the x semi-axis moves the entire smooth
closed boundary without the four 90-degree lateral corners of the rejected
rectangular control.  The Au thickness, material, source and six-PML contract
remain unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
BINARY_CONTROL = HERE / "04_run_au_binary_representation_control.py"
ELLIPSE_HALF_Y_UM = 18.0
ELLIPSE_VERTICES = 512


def load_binary_control():
    spec = importlib.util.spec_from_file_location("au_ellipse_binary_base", BINARY_CONTROL)
    if spec is None or spec.loader is None:
        raise ImportError(BINARY_CONTROL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option_present(arguments: list[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in arguments)


def option_value(arguments: list[str], option: str) -> str | None:
    for index, value in enumerate(arguments):
        if value.startswith(f"{option}="):
            return value.split("=", 1)[1]
        if value == option and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def ellipse_vertices(half_x_m: float, half_y_m: float, count: int) -> np.ndarray:
    if count < 64 or count % 4:
        raise ValueError("ellipse vertex count must be a multiple of four and >=64")
    theta = np.arange(count, dtype=float) * (2.0 * np.pi / count)
    vertices = np.column_stack((half_x_m * np.cos(theta), half_y_m * np.sin(theta)))
    signed_twice_area = np.sum(
        vertices[:, 0] * np.roll(vertices[:, 1], -1)
        - vertices[:, 1] * np.roll(vertices[:, 0], -1)
    )
    if signed_twice_area <= 0.0:
        raise RuntimeError("ellipse vertices are not counter-clockwise")
    return vertices


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--au-half-x-um", type=float, required=True)
    parser.add_argument("--au-half-y-um", type=float, default=ELLIPSE_HALF_Y_UM)
    parser.add_argument("--ellipse-vertices", type=int, default=ELLIPSE_VERTICES)
    parser.add_argument("--ellipse-dxy-nm", type=float, default=25.0)
    parser.add_argument(
        "--mesh-half-x-um",
        type=float,
        help=(
            "fixed x half-span for both the local mesh and field/flux monitors; "
            "omit only for legacy moving-box diagnostics"
        ),
    )
    parser.add_argument(
        "--mesh-half-y-um",
        type=float,
        help=(
            "fixed y half-span for both the local mesh and field/flux monitors; "
            "omit only for legacy moving-box diagnostics"
        ),
    )
    parsed, remaining = parser.parse_known_args()
    if not 4.0 <= parsed.au_half_x_um <= 10.0:
        raise ValueError("--au-half-x-um must remain within [4,10] um")
    if not 8.0 <= parsed.au_half_y_um <= 20.0:
        raise ValueError("--au-half-y-um must remain within [8,20] um")
    if not 12.5 <= parsed.ellipse_dxy_nm <= 100.0:
        raise ValueError("--ellipse-dxy-nm must remain within [12.5,100] nm")
    if (parsed.mesh_half_x_um is None) != (parsed.mesh_half_y_um is None):
        raise ValueError("--mesh-half-x-um and --mesh-half-y-um must be paired")

    half_x_m = float(parsed.au_half_x_um) * 1.0e-6
    half_y_m = float(parsed.au_half_y_um) * 1.0e-6
    fixed_mesh_bounds = parsed.mesh_half_x_um is not None
    mesh_half_x_m = (
        float(parsed.mesh_half_x_um) * 1.0e-6
        if fixed_mesh_bounds
        else half_x_m + 0.5e-6
    )
    mesh_half_y_m = (
        float(parsed.mesh_half_y_um) * 1.0e-6
        if fixed_mesh_bounds
        else half_y_m + 0.5e-6
    )
    if mesh_half_x_m < half_x_m + 0.49e-6:
        raise ValueError("fixed mesh x bounds must clear Au by at least 0.49 um")
    if mesh_half_y_m < half_y_m + 0.49e-6:
        raise ValueError("fixed mesh y bounds must clear Au by at least 0.49 um")
    vertices = ellipse_vertices(half_x_m, half_y_m, int(parsed.ellipse_vertices))
    dxy_m = float(parsed.ellipse_dxy_nm) * 1.0e-9
    base = load_binary_control()
    original_load_legacy = base.load_legacy

    base.AU_BOUNDS = {
        "x": (-half_x_m, half_x_m),
        "y": (-half_y_m, half_y_m),
        "z": (0.05e-6, 0.10e-6),
    }
    base.FLUX_BOUNDS = {
        "x": (-mesh_half_x_m, mesh_half_x_m),
        "y": (-mesh_half_y_m, mesh_half_y_m),
        "z": (-0.45e-6, 0.60e-6),
    }
    base.AU_DZ_M = 5.0e-9
    base.AU_DXY_M = dxy_m
    base.EDGE_DXY_M = None

    def load_ellipse_legacy():
        legacy = original_load_legacy()

        def add_ellipse_design(fdtd, *, rho: float, representation: str):
            if representation != "scalar" or float(rho) != 1.0:
                raise ValueError("smooth-ellipse control requires scalar rho=1 Au")
            epsilon, index = legacy.complex_index(rho)
            name = f"rho{rho:g}_{representation}_complex_block"
            material_name = f"rho{rho:g}_single_frequency_nk"
            material = fdtd.addmaterial("(n,k) Material")
            fdtd.setmaterial(material, "name", material_name)
            fdtd.setmaterial(material_name, "Refractive Index", float(index.real))
            fdtd.setmaterial(
                material_name, "Imaginary Refractive Index", float(index.imag)
            )
            polygon = fdtd.addpoly()
            polygon["name"] = name
            polygon["material"] = material_name
            polygon["vertices"] = vertices
            polygon["z min"], polygon["z max"] = base.AU_BOUNDS["z"]
            return {
                "name": name,
                "representation": representation,
                "rho": rho,
                "requested_epsilon": [epsilon.real, epsilon.imag],
                "requested_nk": [index.real, index.imag],
                "bounds_m": {axis: list(base.AU_BOUNDS[axis]) for axis in "xyz"},
                "import_sample_shape": None,
                "lateral_geometry": "counter-clockwise polygonal ellipse",
                "ellipse_semi_axes_m": [half_x_m, half_y_m],
                "ellipse_vertex_count": int(parsed.ellipse_vertices),
            }

        legacy.add_design = add_ellipse_design
        return legacy

    def add_ellipse_mesh(fdtd) -> None:
        mesh = fdtd.addmesh()
        mesh["name"] = "au_ellipse_full_local_mesh"
        mesh["x min"], mesh["x max"] = base.FLUX_BOUNDS["x"]
        mesh["y min"], mesh["y max"] = base.FLUX_BOUNDS["y"]
        mesh["z min"], mesh["z max"] = (0.0, 0.15e-6)
        mesh["override x mesh"] = True
        mesh["override y mesh"] = True
        mesh["override z mesh"] = True
        mesh["dx"] = dxy_m
        mesh["dy"] = dxy_m
        mesh["dz"] = 5.0e-9

    base.load_legacy = load_ellipse_legacy
    base.add_local_mesh = add_ellipse_mesh

    if not option_present(remaining, "--rho"):
        remaining.extend(("--rho", "1"))
    if not option_present(remaining, "--representation"):
        remaining.extend(("--representation", "scalar"))
    if option_value(remaining, "--rho") != "1":
        raise ValueError("smooth-ellipse control requires rho=1")
    if option_value(remaining, "--representation") != "scalar":
        raise ValueError("smooth-ellipse control requires scalar Au")
    output_value = option_value(remaining, "--output-dir")
    if output_value is None:
        raise ValueError("--output-dir is required")
    result_path = Path(output_value).expanduser().resolve() / "case_result.json"

    sys.argv = [sys.argv[0], *remaining]
    return_code = int(base.main())
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        result["geometry_representation"] = "smooth_closed_binary_scalar_Au_ellipse"
        result["shape_parameter"] = {
            "name": "Au_ellipse_x_semi_axis",
            "value_m": half_x_m,
            "value_um": float(parsed.au_half_x_um),
            "fixed_y_semi_axis_m": half_y_m,
            "fixed_depth_m": 50.0e-9,
            "ellipse_vertex_count": int(parsed.ellipse_vertices),
            "maximum_vertex_angle_rad": 2.0 * np.pi / parsed.ellipse_vertices,
            "moved_boundary": "complete smooth closed lateral ellipse",
            "fixed_boundaries": ["z_min", "z_max"],
            "mesh_and_monitor_bounds_move_with_shape": not fixed_mesh_bounds,
            "mesh_and_monitor_bounds_m": {
                "x": [-mesh_half_x_m, mesh_half_x_m],
                "y": [-mesh_half_y_m, mesh_half_y_m],
                "z": [0.0, 0.15e-6],
            },
        }
        result["gray_Au_air_material_used"] = False
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
