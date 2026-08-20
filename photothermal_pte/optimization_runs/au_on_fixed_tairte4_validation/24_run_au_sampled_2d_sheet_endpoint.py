#!/usr/bin/env python3
"""GPU endpoint control for a rim-free sampled-2D Au sheet.

The production candidate is *not* promoted by construction.  It converts the
10-um Ordal bulk permittivity to a 50-nm surface conductivity, represents the
same smooth ellipse by a true Lumerical 2-D polygon, and runs the same finite
Gaussian/six-PML/PVA5 fixed-grid contract as the resolved 3-D film control.

The 2-D absorbed-power density is evaluated independently from

    q_s,c = 0.5 Re(sigma_s) |E_c|^2,  c in {x,y},

on the component-specific Yee coordinates.  It is stored as a conservative
pseudo-volume layer solely so the existing artifact schema and comparison
tools can integrate it.  No clipping, gain, smoothing, or power rescaling is
used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
STAGE04 = HERE / "04_run_au_binary_representation_control.py"
STAGE12 = HERE / "12_run_au_sharp_interface_external_field_adjoint.py"
STAGE16 = HERE / "16_run_au_smooth_ellipse_width_control.py"

C0 = 299792458.0
EPS0 = 8.8541878128e-12
WAVELENGTH_M = 10.0e-6
FREQUENCY_HZ = C0 / WAVELENGTH_M
AU_EPSILON = (12.1 + 69.2j) ** 2
AIR_EPSILON = 1.0 + 0.0j
PHYSICAL_THICKNESS_M = 50.0e-9
SHEET_Z_M = 75.0e-9
MATERIAL_NAME = "Au_Ordal_10um_equivalent_sampled_2D_50nm"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage04 = load("au_sheet_stage04", STAGE04)
stage12 = load("au_sheet_stage12", STAGE12)
stage16 = load("au_sheet_stage16", STAGE16)


def trapezoid_weights(values: np.ndarray) -> np.ndarray:
    coordinate = np.asarray(values, float).reshape(-1)
    if coordinate.size < 2 or not np.all(np.diff(coordinate) > 0.0):
        raise ValueError("integration coordinate must be strictly increasing")
    weights = np.empty_like(coordinate)
    weights[0] = 0.5 * (coordinate[1] - coordinate[0])
    weights[-1] = 0.5 * (coordinate[-1] - coordinate[-2])
    weights[1:-1] = 0.5 * (coordinate[2:] - coordinate[:-2])
    return weights


def ellipse_overlap_fraction(
    x: np.ndarray,
    y: np.ndarray,
    *,
    half_x_m: float,
    half_y_m: float,
    samples_per_axis: int = 8,
) -> np.ndarray:
    """Sub-cell area fraction for the analytic ellipse on a Yee surface grid."""

    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    wx = trapezoid_weights(x)
    wy = trapezoid_weights(y)
    offsets = (np.arange(samples_per_axis, dtype=float) + 0.5) / samples_per_axis - 0.5
    fraction = np.zeros((x.size, y.size), float)
    for ox in offsets:
        xx = x[:, None] + ox * wx[:, None]
        for oy in offsets:
            yy = y[None, :] + oy * wy[None, :]
            fraction += (xx / half_x_m) ** 2 + (yy / half_y_m) ** 2 <= 1.0
    return fraction / float(samples_per_axis**2)


def bulk_conductivity(frequency_hz: np.ndarray | float) -> np.ndarray:
    frequency = np.asarray(frequency_hz, float)
    return -1j * (2.0 * np.pi * frequency) * EPS0 * (AU_EPSILON - AIR_EPSILON)


def add_sampled_sheet_material(fdtd) -> dict[str, object]:
    frequencies = np.linspace(0.95 * FREQUENCY_HZ, 1.05 * FREQUENCY_HZ,  nine := 9)
    conductivity = bulk_conductivity(frequencies)
    sampled = np.column_stack((frequencies, conductivity))
    handle = fdtd.addmaterial("Sampled 2D data")
    fdtd.setmaterial(handle, "name", MATERIAL_NAME)
    fdtd.setmaterial(MATERIAL_NAME, "layer thickness enabled", True)
    fdtd.setmaterial(MATERIAL_NAME, "layer thickness", PHYSICAL_THICKNESS_M)
    fdtd.setmaterial(MATERIAL_NAME, "sampled data", sampled)
    fdtd.setmaterial(MATERIAL_NAME, "tolerance", 0.0)
    fdtd.setmaterial(MATERIAL_NAME, "max coefficients", 20)
    fdtd.setmaterial(MATERIAL_NAME, "make fit passive", True)
    direct = complex(
        np.asarray(fdtd.getsurfaceconductivity(MATERIAL_NAME, FREQUENCY_HZ, 1))
        .reshape(-1)[0]
    )
    fitted = complex(
        np.asarray(
            fdtd.getfdtdsurfaceconductivity(
                MATERIAL_NAME,
                np.asarray([FREQUENCY_HZ]),
                float(frequencies.min()),
                float(frequencies.max()),
                1,
            )
        ).reshape(-1)[0]
    )
    expected = complex(bulk_conductivity(FREQUENCY_HZ) * PHYSICAL_THICKNESS_M)
    return {
        "sample_count": int(nine),
        "frequency_bounds_Hz": [float(frequencies.min()), float(frequencies.max())],
        "bulk_epsilon_at_10um": [float(AU_EPSILON.real), float(AU_EPSILON.imag)],
        "expected_surface_conductivity_S": [float(expected.real), float(expected.imag)],
        "direct_surface_conductivity_S": [float(direct.real), float(direct.imag)],
        "fitted_surface_conductivity_S": [float(fitted.real), float(fitted.imag)],
        "fit_relative_error": float(abs(fitted - expected) / abs(expected)),
    }


def sheet_extractor(fdtd, *, field_monitor: str, index_monitor: str, wavelength_m: float):
    del index_monitor
    if abs(float(wavelength_m) - WAVELENGTH_M) > 1.0e-15:
        raise ValueError("sheet endpoint is frozen at 10 um")
    electric, grid = stage12.monitor_electric(fdtd, field_monitor)
    frequency = np.asarray(fdtd.getdata(field_monitor, "f", 1), float).reshape(-1)
    frequency_index = int(np.argmin(abs(frequency - FREQUENCY_HZ)))
    fitted_sigma = complex(
        np.asarray(
            fdtd.getfdtdsurfaceconductivity(
                MATERIAL_NAME,
                np.asarray([frequency[frequency_index]]),
                0.95 * FREQUENCY_HZ,
                1.05 * FREQUENCY_HZ,
                1,
            )
        ).reshape(-1)[0]
    )
    if fitted_sigma.real <= 0.0:
        raise RuntimeError(f"non-passive fitted sheet conductivity {fitted_sigma}")

    components: dict[str, np.ndarray] = {}
    coordinates: dict[str, dict[str, np.ndarray]] = {}
    power: dict[str, float] = {}
    hotspots: dict[str, dict[str, float]] = {}
    raw_shapes: dict[str, object] = {}
    for component, label in enumerate("xyz"):
        x, y, z = stage12.component_coordinates(grid, component)
        field = np.asarray(electric[..., 0, component], complex)
        raw_shapes[label] = {"E": list(field.shape), "index": None}
        if label == "z":
            q_volume = np.zeros_like(field.real)
            chosen_z_index = int(np.argmin(abs(z - SHEET_Z_M)))
            overlap = np.zeros((x.size, y.size), float)
        else:
            chosen_z_index = int(np.argmin(abs(z - SHEET_Z_M)))
            overlap = ellipse_overlap_fraction(
                x,
                y,
                half_x_m=8.0e-6,
                half_y_m=18.0e-6,
            )
            q_surface = (
                0.5
                * fitted_sigma.real
                * np.abs(field[:, :, chosen_z_index]) ** 2
                * overlap
            )
            wz = trapezoid_weights(z)
            q_volume = np.zeros_like(field.real)
            q_volume[:, :, chosen_z_index] = q_surface / wz[chosen_z_index]
        if not np.all(np.isfinite(q_volume)) or np.any(q_volume < 0.0):
            raise RuntimeError(f"invalid sheet Q{label}")
        components[label] = q_volume
        coordinates[label] = {"x": x, "y": y, "z": z}
        power[label] = float(
            np.einsum(
                "i,j,k,ijk->",
                trapezoid_weights(x),
                trapezoid_weights(y),
                trapezoid_weights(z),
                q_volume,
                optimize=True,
            )
        )
        maximum = np.unravel_index(int(np.argmax(q_volume)), q_volume.shape)
        hotspots[label] = {
            "x_m": float(x[maximum[0]]),
            "y_m": float(y[maximum[1]]),
            "z_m": float(z[maximum[2]]),
            "Q_W_m3": float(q_volume[maximum]),
            "sheet_z_target_m": SHEET_Z_M,
            "sheet_z_sample_m": float(z[chosen_z_index]),
            "ellipse_overlap_fraction_range": [
                float(np.min(overlap)),
                float(np.max(overlap)),
            ],
        }
    return {
        "frequency_hz": float(frequency[frequency_index]),
        "wavelength_m": float(C0 / frequency[frequency_index]),
        "frequency_index_zero_based": frequency_index,
        "frequency_count": int(frequency.size),
        "base_coordinates": {axis: np.asarray(grid[axis], float) for axis in "xyz"},
        "delta_coordinates": {
            axis: np.asarray(grid[f"delta_{axis}"], float) for axis in "xyz"
        },
        "native_coordinates": coordinates,
        "Q_components": components,
        "component_power_W": power,
        "P_Q_W": float(sum(power.values())),
        "component_hotspots": hotspots,
        "raw_array_shapes": raw_shapes,
        "surface_conductivity_fitted_S": [
            float(fitted_sigma.real),
            float(fitted_sigma.imag),
        ],
        "surface_loss_formula": "0.5*Re(sigma_s)*(|Ex|^2+|Ey|^2)",
        "pseudo_volume_storage_only": True,
        "operations_forbidden_and_absent": [
            "clipping",
            "smoothing",
            "gain",
            "global_rescaling",
            "tiling",
            "old_artifact_crop",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-device", default="GPU 0")
    parsed, remaining = parser.parse_known_args()

    base = stage16.load_binary_control()
    vertices = stage16.ellipse_vertices(8.0e-6, 18.0e-6, 512)
    base.AU_BOUNDS = {
        "x": (-8.0e-6, 8.0e-6),
        "y": (-18.0e-6, 18.0e-6),
        "z": (SHEET_Z_M, SHEET_Z_M),
    }
    base.FLUX_BOUNDS = {
        "x": (-8.6e-6, 8.6e-6),
        "y": (-18.5e-6, 18.5e-6),
        "z": (-0.45e-6, 0.60e-6),
    }
    base.AU_DXY_M = 50.0e-9
    base.AU_DZ_M = 5.0e-9
    material_metadata: dict[str, object] = {}

    original_load_legacy = base.load_legacy

    def load_sheet_legacy():
        legacy = original_load_legacy()

        def add_sheet(fdtd, *, rho: float, representation: str):
            if float(rho) != 1.0 or representation != "scalar":
                raise ValueError("sampled-2D endpoint requires rho=1 scalar selector")
            material_metadata.update(add_sampled_sheet_material(fdtd))
            polygon = fdtd.add2dpoly()
            polygon["name"] = "rho1_scalar_complex_block"
            polygon["surface normal"] = 3
            polygon["vertices"] = vertices
            polygon["z"] = SHEET_Z_M
            polygon["material"] = MATERIAL_NAME
            return {
                "name": "rho1_scalar_complex_block",
                "representation": "sampled_2D_surface_conductivity",
                "rho": 1.0,
                "requested_epsilon": [float(AU_EPSILON.real), float(AU_EPSILON.imag)],
                "requested_nk": [12.1, 69.2],
                "bounds_m": {
                    "x": [-8.0e-6, 8.0e-6],
                    "y": [-18.0e-6, 18.0e-6],
                    "z": [SHEET_Z_M, SHEET_Z_M],
                },
                "physical_thickness_m": PHYSICAL_THICKNESS_M,
                "surface_conductivity": material_metadata,
                "lateral_geometry": "counter-clockwise polygonal ellipse",
                "ellipse_semi_axes_m": [8.0e-6, 18.0e-6],
                "ellipse_vertex_count": 512,
            }

        legacy.add_design = add_sheet
        return legacy

    def add_mesh(fdtd):
        mesh = fdtd.addmesh()
        mesh["name"] = "au_sheet_fixed_local_mesh"
        mesh["x min"], mesh["x max"] = base.FLUX_BOUNDS["x"]
        mesh["y min"], mesh["y max"] = base.FLUX_BOUNDS["y"]
        mesh["z min"], mesh["z max"] = (0.0, 0.15e-6)
        mesh["override x mesh"] = True
        mesh["override y mesh"] = True
        mesh["override z mesh"] = True
        mesh["dx"] = 50.0e-9
        mesh["dy"] = 50.0e-9
        mesh["dz"] = 5.0e-9

    def readback(_fdtd, q):
        fitted = q["surface_conductivity_fitted_S"]
        finite = bool(np.all(np.isfinite(fitted)))
        return {
            label: {
                "shape": list(q["Q_components"][label].shape),
                "all_finite": finite,
                "surface_conductivity_fitted_S": fitted,
                "material_dimensionality": "2D sheet",
            }
            for label in "xyz"
        }

    base.load_legacy = load_sheet_legacy
    base.add_local_mesh = add_mesh
    base.extract_native_yee_q = sheet_extractor
    base.component_epsilon_readback = readback

    required = [
        "--rho",
        "1",
        "--representation",
        "scalar",
        "--output-dir",
        parsed.output_dir,
        "--gpu-device",
        parsed.gpu_device,
        "--mesh-refinement",
        "precise volume average",
        "--meshing-refinement",
        "5",
        "--dt-stability-factor",
        "0.95",
        "--mesh-wavelength-um",
        "10",
    ]
    sys.argv = [sys.argv[0], *required, *remaining]
    code = int(base.main())
    result_path = Path(parsed.output_dir).expanduser().resolve() / "case_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        result["geometry_representation"] = "rim_free_sampled_2D_Au_sheet"
        result["surface_conductivity_conversion"] = material_metadata
        result["promotion_contract"] = {
            "promoted": False,
            "required_next_gate": (
                "endpoint equivalence to exact 50-nm volumetric Au in P_six, "
                "face powers, external field, and lateral absorbed-power profile"
            ),
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
