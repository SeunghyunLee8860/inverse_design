#!/usr/bin/env python3
"""Exact scalar-Au, fully smooth 3-D ellipsoid width control at 10 um.

This deliberately changes the vertical geometry from the 50-nm production
film.  It is a mathematical root-cause control: an ellipsoid has no lateral
corners and no top/bottom rims.  Passing this control would isolate the failed
film derivative to its non-smooth 3-D rim; it would not promote the ellipsoid
as a physical electrode model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
STAGE04 = HERE / "04_run_au_binary_representation_control.py"

from photothermal_pte.finite_inverse_design.native_yee_q import frequency_slice


def load_stage04():
    spec = importlib.util.spec_from_file_location("au_ellipsoid_stage04", STAGE04)
    if spec is None or spec.loader is None:
        raise ImportError(STAGE04)
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


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--au-half-x-um", type=float, required=True)
    parser.add_argument("--au-half-y-um", type=float, default=18.0)
    parser.add_argument("--au-half-z-um", type=float, default=1.0)
    parser.add_argument("--ellipsoid-dxy-nm", type=float, default=50.0)
    parser.add_argument("--ellipsoid-dz-nm", type=float, default=25.0)
    parsed, remaining = parser.parse_known_args()
    if not 7.5 <= parsed.au_half_x_um <= 8.5:
        raise ValueError("--au-half-x-um must remain within [7.5,8.5] um")
    if parsed.au_half_y_um <= 0.0 or parsed.au_half_z_um <= 0.0:
        raise ValueError("ellipsoid semi-axes must be positive")

    a = float(parsed.au_half_x_um) * 1.0e-6
    b = float(parsed.au_half_y_um) * 1.0e-6
    c = float(parsed.au_half_z_um) * 1.0e-6
    center_z = 0.075e-6
    dxy = float(parsed.ellipsoid_dxy_nm) * 1.0e-9
    dz = float(parsed.ellipsoid_dz_nm) * 1.0e-9
    base = load_stage04()
    original_load_legacy = base.load_legacy

    base.AU_BOUNDS = {
        "x": (-a, a),
        "y": (-b, b),
        "z": (center_z - c, center_z + c),
    }
    base.FLUX_BOUNDS = {
        "x": (-8.6e-6, 8.6e-6),
        "y": (-18.5e-6, 18.5e-6),
        "z": (-2.20e-6, 1.60e-6),
    }
    base.AU_DXY_M = dxy
    base.AU_DZ_M = dz
    base.EDGE_DXY_M = None

    def load_ellipsoid_legacy():
        legacy = original_load_legacy()

        def add_ellipsoid(fdtd, *, rho: float, representation: str):
            if float(rho) != 1.0 or representation != "scalar":
                raise ValueError("smooth-3D control requires scalar rho=1 Au")
            epsilon, index = legacy.complex_index(rho)
            material_name = "rho1_single_frequency_nk"
            material = fdtd.addmaterial("(n,k) Material")
            fdtd.setmaterial(material, "name", material_name)
            fdtd.setmaterial(material_name, "Refractive Index", float(index.real))
            fdtd.setmaterial(
                material_name, "Imaginary Refractive Index", float(index.imag)
            )
            sphere = fdtd.addsphere()
            sphere["name"] = "rho1_scalar_complex_block"
            sphere["material"] = material_name
            sphere["make ellipsoid"] = True
            sphere["radius"] = a
            sphere["radius 2"] = b
            sphere["radius 3"] = c
            sphere["x"] = 0.0
            sphere["y"] = 0.0
            sphere["z"] = center_z
            return {
                "name": "rho1_scalar_complex_block",
                "representation": "scalar_exact_Au_smooth_3D_ellipsoid",
                "rho": 1.0,
                "requested_epsilon": [float(epsilon.real), float(epsilon.imag)],
                "requested_nk": [float(index.real), float(index.imag)],
                "bounds_m": {axis: list(base.AU_BOUNDS[axis]) for axis in "xyz"},
                "ellipsoid_semi_axes_m": [a, b, c],
                "center_z_m": center_z,
                "no_lateral_corners": True,
                "no_top_bottom_rims": True,
            }

        legacy.add_design = add_ellipsoid
        return legacy

    def add_mesh(fdtd):
        mesh = fdtd.addmesh()
        mesh["name"] = "au_smooth_3d_ellipsoid_fixed_mesh"
        mesh["x min"], mesh["x max"] = (-8.6e-6, 8.6e-6)
        mesh["y min"], mesh["y max"] = (-18.5e-6, 18.5e-6)
        mesh["z min"], mesh["z max"] = (center_z - c - 0.15e-6, center_z + c + 0.15e-6)
        mesh["override x mesh"] = True
        mesh["override y mesh"] = True
        mesh["override z mesh"] = True
        mesh["dx"] = dxy
        mesh["dy"] = dxy
        mesh["dz"] = dz

    def ellipsoid_readback(_configured_base, fdtd, q):
        # Stage 04 installs this callback through a compatibility lambda whose
        # first argument is the configured legacy module.  Keep that argument
        # explicit even though the ellipsoid readback only needs fdtd and q.
        spatial_shape = tuple(np.asarray(q["base_coordinates"][axis]).size for axis in "xyz")
        result = {}
        for component in "xyz":
            index = frequency_slice(
                np.asarray(fdtd.getdata(base.PABS_INDEX, f"index_{component}", 1)),
                spatial_shape,
                int(q["frequency_index_zero_based"]),
                int(q["frequency_count"]),
                f"index_{component}",
            )
            epsilon = index**2
            x = np.asarray(q["native_coordinates"][component]["x"], float)
            y = np.asarray(q["native_coordinates"][component]["y"], float)
            z = np.asarray(q["native_coordinates"][component]["z"], float)
            normalized_radius = (
                (x[:, None, None] / a) ** 2
                + (y[None, :, None] / b) ** 2
                + ((z[None, None, :] - center_z) / c) ** 2
            )
            interior = normalized_radius <= 0.80**2
            values = epsilon[interior & np.isfinite(epsilon)]
            if values.size == 0:
                raise RuntimeError(f"empty ellipsoid interior readback for {component}")
            result[component] = {
                "shape": list(epsilon.shape),
                "epsilon_interior_median": [
                    float(np.median(values.real)),
                    float(np.median(values.imag)),
                ],
                "interior_sample_count": int(values.size),
                "all_finite": bool(np.all(np.isfinite(epsilon))),
                "interior_normalized_radius_max": 0.80,
            }
        return result

    base.load_legacy = load_ellipsoid_legacy
    base.add_local_mesh = add_mesh
    base.component_epsilon_readback = ellipsoid_readback

    defaults = [
        ("--rho", "1"),
        ("--representation", "scalar"),
        ("--mesh-refinement", "precise volume average"),
        ("--meshing-refinement", "5"),
        ("--dt-stability-factor", "0.95"),
        ("--mesh-wavelength-um", "10"),
    ]
    for option, value in defaults:
        if not option_present(remaining, option):
            remaining.extend((option, value))
    sys.argv = [sys.argv[0], *remaining]
    code = int(base.main())
    output_value = option_value(remaining, "--output-dir")
    if output_value is None:
        raise ValueError("--output-dir is required")
    result_path = Path(output_value).expanduser().resolve() / "case_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        result["geometry_representation"] = "fully_smooth_3D_Au_ellipsoid"
        result["shape_parameter"] = {
            "name": "ellipsoid_x_semi_axis",
            "value_m": a,
            "value_um": a * 1.0e6,
            "fixed_y_semi_axis_m": b,
            "fixed_z_semi_axis_m": c,
            "normal_velocity": "dr/da dot outward_unit_normal",
        }
        result["physical_promotion_permitted"] = False
        result["purpose"] = "mathematical smooth-boundary root-cause control only"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
