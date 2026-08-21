#!/usr/bin/env python3
"""GPU endpoint controls for a temperature-attribute Au-density carrier.

The temperature attribute is only a numerical optical carrier.  The forward
and reverse interpolation directions, and linear/table material models, are
diagnostics for the v261 GPU material path; none of their attribute values are
physical temperatures or may be exported to the thermal solver.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "legacy_v261_optical_support" / "run_complex_material_control.py"
T_REF_K = 300.0
N_AU = 12.1
K_AU = 69.2
BASE_N = 1.0
TEMPERATURE_MAT: Path | None = None
REVERSE_AU_BASE = False
USE_TEMPERATURE_TABLE = False
CARRIER_SPAN_K = 1.0
USE_SAMPLED_ORDAL_BASE = False
SAMPLED_MAX_COEFFICIENTS = 12
ENABLE_GRID_ATTRIBUTE_CONFORMAL = True


def json_safe(value):
    """Convert Lumerical material readback values to JSON-safe objects."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    return value


def load_base():
    spec = importlib.util.spec_from_file_location("au_temperature_density_base", LEGACY)
    if spec is None or spec.loader is None:
        raise ImportError(LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def box_tetrahedra(bounds: dict[str, tuple[float, float]]):
    xmin, xmax = bounds["x"]
    ymin, ymax = bounds["y"]
    zmin, zmax = bounds["z"]
    vertices = np.asarray(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmin, ymax, zmax],
            [xmax, ymax, zmax],
        ],
        dtype=float,
    )
    # Six tetrahedra around the body diagonal from vertex 1 to vertex 8.
    connectivity = np.asarray(
        [
            [1, 2, 4, 8],
            [1, 4, 3, 8],
            [1, 3, 7, 8],
            [1, 7, 5, 8],
            [1, 5, 6, 8],
            [1, 6, 2, 8],
        ],
        dtype=float,
    )
    return vertices, connectivity


def component_epsilon_readback(base, fdtd, q: dict[str, object]):
    """Read epsilon on each component grid, including thin-film interiors."""

    spatial_shape = tuple(
        np.asarray(q["base_coordinates"][axis]).size for axis in "xyz"
    )
    thickness = base.BLOCK_BOUNDS["z"][1] - base.BLOCK_BOUNDS["z"][0]
    margins = {
        "x": min(0.2e-6, 0.1 * (base.BLOCK_BOUNDS["x"][1] - base.BLOCK_BOUNDS["x"][0])),
        "y": min(0.2e-6, 0.1 * (base.BLOCK_BOUNDS["y"][1] - base.BLOCK_BOUNDS["y"][0])),
        "z": min(10.0e-9, 0.2 * thickness),
    }
    result = {}
    for component in "xyz":
        index = base.frequency_slice(
            np.asarray(fdtd.getdata(base.PABS_INDEX, f"index_{component}", 1)),
            spatial_shape,
            int(q["frequency_index_zero_based"]),
            int(q["frequency_count"]),
            f"index_{component}",
        )
        epsilon = index**2
        finite = np.isfinite(epsilon)
        x = np.asarray(q["base_coordinates"]["x"], float)
        y = np.asarray(q["base_coordinates"]["y"], float)
        z = np.asarray(q["base_coordinates"]["z"], float)
        interior = (
            (x[:, None, None] >= base.BLOCK_BOUNDS["x"][0] + margins["x"])
            & (x[:, None, None] <= base.BLOCK_BOUNDS["x"][1] - margins["x"])
            & (y[None, :, None] >= base.BLOCK_BOUNDS["y"][0] + margins["y"])
            & (y[None, :, None] <= base.BLOCK_BOUNDS["y"][1] - margins["y"])
            & (z[None, None, :] >= base.BLOCK_BOUNDS["z"][0] + margins["z"])
            & (z[None, None, :] <= base.BLOCK_BOUNDS["z"][1] - margins["z"])
            & finite
        )
        interior_values = epsilon[interior]
        if interior_values.size == 0:
            raise RuntimeError(f"empty temperature-density interior for {component}")
        values = epsilon[finite]
        result[component] = {
            "shape": list(epsilon.shape),
            "epsilon_interior_median": [
                float(np.median(interior_values.real)),
                float(np.median(interior_values.imag)),
            ],
            "interior_sample_count": int(interior_values.size),
            "epsilon_real_range": [float(np.min(values.real)), float(np.max(values.real))],
            "epsilon_imag_range": [float(np.min(values.imag)), float(np.max(values.imag))],
            "all_finite": bool(np.all(finite)),
            "interior_margins_m": margins,
        }
    return result


def add_temperature_density_design(base, fdtd, *, rho: float, representation: str):
    del representation
    density = float(rho)
    expected_nk = BASE_N + density * ((N_AU + 1j * K_AU) - BASE_N)
    expected_epsilon = expected_nk**2

    if REVERSE_AU_BASE:
        if USE_SAMPLED_ORDAL_BASE:
            base_material_name = "rho_temperature_Ordal_sampled_passive_stable_base"
            base_material_id = fdtd.addmaterial("Sampled data")
            fdtd.setmaterial(base_material_id, "name", base_material_name)
            table = np.genfromtxt(
                HERE / "data" / "au_ordal_1987_nk.csv",
                delimiter=",",
                names=True,
            )
            wavelength_m = np.asarray(table["wavelength_um"], float) * 1.0e-6
            sampled_index = np.asarray(table["n"], float) + 1j * np.asarray(
                table["k"], float
            )
            sampled_frequency = 299792458.0 / wavelength_m
            fdtd.setmaterial(
                base_material_name,
                "sampled data",
                np.column_stack((sampled_frequency, sampled_index**2)),
            )
            fdtd.setmaterial(
                base_material_name,
                "max coefficients",
                int(SAMPLED_MAX_COEFFICIENTS),
            )
            fdtd.setmaterial(base_material_name, "tolerance", 0.0)
            fdtd.setmaterial(base_material_name, "make fit passive", True)
            sampled_stability_error = None
            try:
                fdtd.setmaterial(base_material_name, "improve stability", True)
                sampled_stability_setting = True
            except Exception as exc:
                # v261 documents this Material Explorer option but does not
                # expose it for every project-local Sampled-data material.
                # Preserve that capability result rather than silently
                # claiming the stability restriction was active.
                sampled_stability_setting = False
                sampled_stability_error = f"{type(exc).__name__}: {exc}"
            base_model = "Ordal_sampled_data_passive_fit"
        else:
            base_material_name = "rho_temperature_exact_target_base"
            base_material_id = fdtd.addmaterial("(n,k) Material")
            fdtd.setmaterial(base_material_id, "name", base_material_name)
            fdtd.setmaterial(base_material_name, "Refractive Index", N_AU)
            fdtd.setmaterial(base_material_name, "Imaginary Refractive Index", K_AU)
            base_model = "single_frequency_constant_nk"
            sampled_stability_setting = None
            sampled_stability_error = None
        carrier_value = CARRIER_SPAN_K * (1.0 - density)
        dn_dt = (BASE_N - N_AU) / CARRIER_SPAN_K
        dk_dt = -K_AU / CARRIER_SPAN_K
        carrier_law = (
            f"T_attribute_K = 300 K + {CARRIER_SPAN_K:.17g}*(1-rho) K"
        )
        interpolation_direction = "exact_target_base_toward_air"
    else:
        base_material_name = "rho_temperature_vacuum_base"
        base_material_id = fdtd.addmaterial("Dielectric")
        fdtd.setmaterial(base_material_id, "name", base_material_name)
        fdtd.setmaterial(base_material_name, "Refractive Index", BASE_N)
        carrier_value = CARRIER_SPAN_K * density
        dn_dt = (N_AU - BASE_N) / CARRIER_SPAN_K
        dk_dt = K_AU / CARRIER_SPAN_K
        carrier_law = (
            f"T_attribute_K = 300 K + {CARRIER_SPAN_K:.17g}*rho K"
        )
        interpolation_direction = "air_base_toward_exact_target"
        base_model = "nondispersive_dielectric"
        sampled_stability_setting = None
        sampled_stability_error = None

    material_name = f"rho{density:g}_temperature_attribute_density"
    material_id = fdtd.addmaterial("Index perturbation")
    fdtd.setmaterial(material_id, "name", material_name)
    fdtd.setmaterial(material_name, "base material", base_material_name)
    fdtd.setmaterial(material_name, "include np density", False)
    fdtd.setmaterial(material_name, "include temperature effects", True)
    fdtd.setmaterial(material_name, "linear sensitivity", not USE_TEMPERATURE_TABLE)
    fdtd.setmaterial(material_name, "table of values", USE_TEMPERATURE_TABLE)
    if USE_TEMPERATURE_TABLE:
        sensitivity_table = np.asarray(
            [
                [T_REF_K, 0.0, 0.0],
                [
                    T_REF_K + 0.5 * CARRIER_SPAN_K,
                    0.5 * CARRIER_SPAN_K * dn_dt,
                    0.5 * CARRIER_SPAN_K * dk_dt,
                ],
                [
                    T_REF_K + CARRIER_SPAN_K,
                    CARRIER_SPAN_K * dn_dt,
                    CARRIER_SPAN_K * dk_dt,
                ],
            ],
            dtype=float,
        )
        fdtd.setmaterial(
            material_name, "temperature sensitivity table", sensitivity_table
        )
    else:
        fdtd.setmaterial(material_name, "Tref", T_REF_K)
        fdtd.setmaterial(material_name, "dn/dt", dn_dt)
        fdtd.setmaterial(material_name, "dk/dt", dk_dt)

    vertices, connectivity = box_tetrahedra(base.BLOCK_BOUNDS)
    attribute_origin = np.asarray(
        [
            0.5 * sum(base.BLOCK_BOUNDS[axis])
            for axis in "xyz"
        ],
        dtype=float,
    )
    relative_vertices = vertices - attribute_origin[None, :]
    # Keep the canonical object name used by the v261 temperature-grid path.
    # Arbitrary renaming is accepted by the layout API but the forward solver
    # controls below verify whether it is honored in the material update.
    attribute_name = "temperature"
    if TEMPERATURE_MAT is None:
        fdtd.putv("rho_attr_x", relative_vertices[:, 0])
        fdtd.putv("rho_attr_y", relative_vertices[:, 1])
        fdtd.putv("rho_attr_z", relative_vertices[:, 2])
        fdtd.putv("rho_attr_C", connectivity)
        fdtd.putv(
            "rho_attr_N",
            np.full(vertices.shape[0], T_REF_K + carrier_value, dtype=float),
        )
        script_steps = (
            (
                "construct unstructured dataset",
                "rho_attr_data=unstructureddataset('thermal',"
                "rho_attr_x,rho_attr_y,rho_attr_z,rho_attr_C);",
            ),
            (
                "attach singleton temperature parameter",
                "rho_attr_data.addparameter('rho_parameter',1);",
            ),
            (
                "attach vertex scalar T",
                "rho_attr_data.addattribute('T',rho_attr_N,'vertex');",
            ),
            (
                "add populated temperature attribute",
                "addgridattribute('temperature',rho_attr_data);",
            ),
            ("name temperature attribute", f"set('name','{attribute_name}');"),
            ("relative coordinates", "set('use relative coordinates',1);"),
            ("position x", f"set('x',{attribute_origin[0]:.17g});"),
            ("position y", f"set('y',{attribute_origin[1]:.17g});"),
            ("position z", f"set('z',{attribute_origin[2]:.17g});"),
        )
        attribute_source = "hand_built_unstructured_dataset"
    else:
        escaped_mat = str(TEMPERATURE_MAT).replace("\\", "/").replace("'", "''")
        script_steps = (
            ("add empty temperature attribute", "addgridattribute('temperature');"),
            ("name temperature attribute", f"set('name','{attribute_name}');"),
            (
                "import exact HEAT-exported dataset",
                f"select('{attribute_name}');importdataset('{escaped_mat}');",
            ),
            ("absolute coordinates", "set('use relative coordinates',0);"),
        )
        attribute_source = "exact_heat_exported_mat"
    for label, script in script_steps:
        try:
            fdtd.eval(script)
        except Exception as exc:
            raise RuntimeError(f"temperature-density step failed: {label}: {exc}") from exc

    # Probe rather than assume this capability.  The Ansys grid-attribute page
    # explicitly says its conformal-toggle tips do not apply to np-density and
    # Temperature attributes.  The installed v261 Temperature object indeed
    # lacks that property; recording the complete property list makes the
    # unavailable route reproducible instead of silently claiming it was used.
    fdtd.select(attribute_name)
    attribute_property_names = [
        line.strip() for line in str(fdtd.get()).splitlines() if line.strip()
    ]
    grid_attribute_conformal_control_available = (
        "enable conformal meshing" in attribute_property_names
    )
    grid_attribute_conformal_readback = None
    if not ENABLE_GRID_ATTRIBUTE_CONFORMAL:
        if not grid_attribute_conformal_control_available:
            raise RuntimeError(
                "v261 temperature grid attribute does not expose the requested "
                "'enable conformal meshing' property; available properties are "
                f"{attribute_property_names}. Use global conformal variant 0 as "
                "the next fail-closed metal-interface control."
            )
        fdtd.set("enable conformal meshing", False)
        grid_attribute_conformal_readback = bool(
            fdtd.get("enable conformal meshing")
        )
        if grid_attribute_conformal_readback:
            raise RuntimeError(
                "temperature grid-attribute conformal-meshing disable did not "
                "survive readback"
            )

    block_name = f"rho{density:g}_temperature_attribute_complex_block"
    block = fdtd.addrect()
    block["name"] = block_name
    block["material"] = material_name
    block["grid attribute name"] = attribute_name
    for axis in "xyz":
        block[f"{axis} min"], block[f"{axis} max"] = base.BLOCK_BOUNDS[axis]

    configured_keys = [
        "include np density",
        "include temperature effects",
        "linear sensitivity",
        "table of values",
    ]
    configured_keys.extend(
        ["temperature sensitivity table"]
        if USE_TEMPERATURE_TABLE
        else ["Tref", "dn/dt", "dk/dt"]
    )
    configured = {
        key: json_safe(fdtd.getmaterial(material_name, key))
        for key in configured_keys
    }
    return {
        "name": block_name,
        "material_name": material_name,
        "base_material_name": base_material_name,
        "base_material_model": base_model,
        "sampled_Ordal_make_fit_passive": (
            True if USE_SAMPLED_ORDAL_BASE else None
        ),
        "sampled_Ordal_improve_stability": sampled_stability_setting,
        "sampled_Ordal_improve_stability_error": sampled_stability_error,
        "sampled_Ordal_max_coefficients": (
            int(SAMPLED_MAX_COEFFICIENTS) if USE_SAMPLED_ORDAL_BASE else None
        ),
        "representation": "temperature_attribute_density",
        "rho": density,
        "temperature_attribute_name": attribute_name,
        "temperature_attribute_is_physical_temperature": False,
        "temperature_attribute_value_K": T_REF_K + carrier_value,
        "temperature_attribute_numerical_span_K": CARRIER_SPAN_K,
        "temperature_attribute_use_relative_coordinates": TEMPERATURE_MAT is None,
        "temperature_attribute_origin_m": (
            attribute_origin.tolist() if TEMPERATURE_MAT is None else [0.0, 0.0, 0.0]
        ),
        "temperature_attribute_source": attribute_source,
        "temperature_attribute_enable_conformal_meshing_requested": bool(
            ENABLE_GRID_ATTRIBUTE_CONFORMAL
        ),
        "temperature_attribute_enable_conformal_meshing_readback": (
            grid_attribute_conformal_readback
        ),
        "temperature_attribute_conformal_control_available": (
            grid_attribute_conformal_control_available
        ),
        "temperature_attribute_property_names": attribute_property_names,
        "temperature_attribute_mat_path": (
            None if TEMPERATURE_MAT is None else str(TEMPERATURE_MAT)
        ),
        "rho_to_attribute": carrier_law,
        "interpolation_direction": interpolation_direction,
        "temperature_interpolation_model": (
            "nonlinear_table" if USE_TEMPERATURE_TABLE else "linear_sensitivity"
        ),
        "index_law": (
            f"n+ik = {BASE_N:.17g} + rho*(("
            f"{N_AU:.17g}+{K_AU:.17g}i)-{BASE_N:.17g})"
        ),
        "base_refractive_index": BASE_N,
        "requested_nk": [expected_nk.real, expected_nk.imag],
        "requested_epsilon": [expected_epsilon.real, expected_epsilon.imag],
        "bounds_m": {axis: list(base.BLOCK_BOUNDS[axis]) for axis in "xyz"},
        "unstructured_vertex_count": int(vertices.shape[0]),
        "unstructured_tetrahedron_count": int(connectivity.shape[0]),
        "material_readback": configured,
    }


def main() -> int:
    global N_AU, K_AU, BASE_N, TEMPERATURE_MAT, REVERSE_AU_BASE, USE_TEMPERATURE_TABLE, CARRIER_SPAN_K, USE_SAMPLED_ORDAL_BASE, SAMPLED_MAX_COEFFICIENTS, ENABLE_GRID_ATTRIBUTE_CONFORMAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rho", type=float, required=True, choices=(0.0, 0.5, 1.0))
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument("--duration-ps", type=float, default=8.0)
    parser.add_argument("--auto-shutoff-min", type=float, default=1.0e-7)
    parser.add_argument("--target-n", type=float, default=N_AU)
    parser.add_argument("--target-k", type=float, default=K_AU)
    parser.add_argument("--base-n", type=float, default=BASE_N)
    parser.add_argument("--reverse-target-base", action="store_true")
    parser.add_argument(
        "--sampled-ordal-base",
        action="store_true",
        help=(
            "Use a passive, stability-restricted Sampled-data fit of the "
            "published Ordal Au table as the reverse-direction base."
        ),
    )
    parser.add_argument("--sampled-max-coefficients", type=int, default=12)
    parser.add_argument(
        "--disable-grid-attribute-conformal",
        action="store_true",
        help=(
            "Request a Temperature-attribute conformal toggle only if the "
            "installed object exposes it. v261 normally fails closed because "
            "that attribute property is unavailable; this does not change "
            "the global FDTD mesh-refinement setting."
        ),
    )
    parser.add_argument("--temperature-table", action="store_true")
    parser.add_argument(
        "--carrier-span-k",
        type=float,
        default=CARRIER_SPAN_K,
        help=(
            "Purely numerical temperature-attribute span used to carry rho. "
            "It is never a physical temperature or a thermal-solver input."
        ),
    )
    parser.add_argument(
        "--mesh-refinement",
        choices=("conformal variant 0", "conformal variant 1", "precise volume average"),
        default="conformal variant 1",
    )
    parser.add_argument("--dt-stability-factor", type=float, default=0.99)
    parser.add_argument(
        "--boundary-mode",
        choices=("PML", "Metal"),
        default="PML",
        help="All-Metal is a divergence-classification control only.",
    )
    parser.add_argument("--block-half-x-um", type=float, default=5.0)
    parser.add_argument("--block-half-y-um", type=float, default=5.0)
    parser.add_argument("--block-z-min-nm", type=float, default=50.0)
    parser.add_argument("--block-thickness-nm", type=float, default=1000.0)
    parser.add_argument("--local-dxy-nm", type=float, default=100.0)
    parser.add_argument("--local-dz-nm", type=float, default=25.0)
    parser.add_argument(
        "--temperature-mat",
        type=Path,
        help=(
            "Import an exact HEAT-exported thermal dataset instead of the "
            "synthetic density carrier. This is a coupling diagnostic only."
        ),
    )
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    N_AU = float(args.target_n)
    K_AU = float(args.target_k)
    BASE_N = float(args.base_n)
    REVERSE_AU_BASE = bool(args.reverse_target_base)
    USE_SAMPLED_ORDAL_BASE = bool(args.sampled_ordal_base)
    SAMPLED_MAX_COEFFICIENTS = int(args.sampled_max_coefficients)
    ENABLE_GRID_ATTRIBUTE_CONFORMAL = not bool(
        args.disable_grid_attribute_conformal
    )
    if USE_SAMPLED_ORDAL_BASE and not REVERSE_AU_BASE:
        parser.error("--sampled-ordal-base requires --reverse-target-base")
    if SAMPLED_MAX_COEFFICIENTS < 1:
        parser.error("--sampled-max-coefficients must be positive")
    USE_TEMPERATURE_TABLE = bool(args.temperature_table)
    CARRIER_SPAN_K = float(args.carrier_span_k)
    if not np.isfinite(CARRIER_SPAN_K) or CARRIER_SPAN_K <= 0.0:
        raise ValueError("carrier span must be finite and positive")
    TEMPERATURE_MAT = (
        args.temperature_mat.expanduser().resolve()
        if args.temperature_mat is not None
        else None
    )
    if TEMPERATURE_MAT is not None and not TEMPERATURE_MAT.is_file():
        raise FileNotFoundError(TEMPERATURE_MAT)
    positive_geometry = (
        args.block_half_x_um,
        args.block_half_y_um,
        args.block_thickness_nm,
        args.local_dxy_nm,
        args.local_dz_nm,
    )
    if any(value <= 0.0 for value in positive_geometry):
        raise ValueError("block spans, thickness, and local mesh steps must be positive")

    base = load_base()
    z_min_m = args.block_z_min_nm * 1.0e-9
    z_max_m = z_min_m + args.block_thickness_nm * 1.0e-9
    base.BLOCK_BOUNDS = {
        "x": (-args.block_half_x_um * 1.0e-6, args.block_half_x_um * 1.0e-6),
        "y": (-args.block_half_y_um * 1.0e-6, args.block_half_y_um * 1.0e-6),
        "z": (z_min_m, z_max_m),
    }
    base.FLUX_BOUNDS = {
        "x": (base.BLOCK_BOUNDS["x"][0] - 0.5e-6, base.BLOCK_BOUNDS["x"][1] + 0.5e-6),
        "y": (base.BLOCK_BOUNDS["y"][0] - 0.5e-6, base.BLOCK_BOUNDS["y"][1] + 0.5e-6),
        "z": (z_min_m - 0.5e-6, z_max_m + 0.5e-6),
    }

    def add_local_mesh(fdtd):
        mesh = fdtd.addmesh()
        mesh["name"] = "temperature_density_block_local_mesh"
        mesh["x min"], mesh["x max"] = base.FLUX_BOUNDS["x"]
        mesh["y min"], mesh["y max"] = base.FLUX_BOUNDS["y"]
        mesh["z min"] = z_min_m - max(0.1e-6, 2.0 * args.local_dz_nm * 1.0e-9)
        mesh["z max"] = z_max_m + max(0.1e-6, 2.0 * args.local_dz_nm * 1.0e-9)
        mesh["override x mesh"] = True
        mesh["override y mesh"] = True
        mesh["override z mesh"] = True
        mesh["dx"] = args.local_dxy_nm * 1.0e-9
        mesh["dy"] = args.local_dxy_nm * 1.0e-9
        mesh["dz"] = args.local_dz_nm * 1.0e-9

    base.add_local_mesh = add_local_mesh
    base.component_epsilon_readback = lambda fdtd, q: component_epsilon_readback(
        base, fdtd, q
    )
    base.add_design = lambda fdtd, rho, representation: add_temperature_density_design(
        base, fdtd, rho=rho, representation=representation
    )
    forwarded = [
        sys.argv[0],
        "--output-dir",
        args.output_dir,
        "--rho",
        str(args.rho),
        "--representation",
        "scalar",
        "--gpu-device",
        args.gpu_device,
        "--duration-ps",
        str(args.duration_ps),
        "--auto-shutoff-min",
        str(args.auto_shutoff_min),
        "--mesh-refinement",
        args.mesh_refinement,
        "--meshing-refinement",
        "5",
        "--mesh-wavelength-um",
        "10.0",
        "--dt-stability-factor",
        str(args.dt_stability_factor),
        "--boundary-mode",
        args.boundary_mode,
    ]
    if args.contract_only:
        forwarded.append("--contract-only")
    sys.argv = forwarded
    return_code = base.main()

    result_path = Path(args.output_dir).expanduser().resolve() / "case_result.json"
    result = json.loads(result_path.read_text())
    result["representation"] = "temperature_attribute_density"
    result["temperature_attribute_is_physical_temperature"] = False
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
