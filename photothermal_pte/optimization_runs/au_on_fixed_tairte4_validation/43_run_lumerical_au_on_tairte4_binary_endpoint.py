#!/usr/bin/env python3
"""Run an exact-binary v261 Au-on-fixed-TaIrTe4 endpoint control.

Au is an optical nanostructure material, not an electrode.  This wrapper uses
the already validated 10-um scalar-Gaussian/six-PML GPU machinery and adds a
fixed anisotropic TaIrTe4 slab plus an optional exact scalar-Au block.  It
does not use importnk2, a gray metal, a moving boundary, or a Lumerical
adjoint.  Raw FSP/NPZ files stay outside Git.
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
STAGE04 = HERE / "04_run_au_binary_representation_control.py"
PERMITTIVITY_PATH = REPOSITORY / "photothermal_pte" / "bundle" / "perm_data.txt"
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.native_yee_q import (  # noqa: E402
    frequency_slice,
    integrate_xyz,
)
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_INDEX,
)


TAIRTE4_BOUNDS = {
    "x": (-10.0e-6, 10.0e-6),
    "y": (-10.0e-6, 10.0e-6),
    "z": (-0.10e-6, 0.0),
}
AU_BOUNDS = {
    "x": (-5.0e-6, 5.0e-6),
    "y": (-5.0e-6, 5.0e-6),
    "z": (0.0, 0.05e-6),
}
FLUX_BOUNDS = {
    "x": (-10.5e-6, 10.5e-6),
    "y": (-10.5e-6, 10.5e-6),
    "z": (-0.50e-6, 0.55e-6),
}
LOCAL_DXY_M = 100.0e-9
LOCAL_DZ_M = 5.0e-9
WAVELENGTH_NM = 10_000.0


def load_stage04():
    spec = importlib.util.spec_from_file_location("au_tairte4_stage04", STAGE04)
    if spec is None or spec.loader is None:
        raise ImportError(STAGE04)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _epsilon_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(PERMITTIVITY_PATH)
    data = data[np.argsort(data[:, 0])]
    wavelength_nm = data[:, 0]
    epsilon_a = data[:, 1] + 1j * data[:, 2]
    epsilon_b = data[:, 3] + 1j * data[:, 4]
    epsilon_c = data[:, 5] + 1j * data[:, 6]
    if not np.array_equal(epsilon_b, epsilon_c):
        raise RuntimeError("perm_data.txt no longer satisfies epsilon_c=epsilon_b")
    return wavelength_nm, epsilon_a, epsilon_b, epsilon_c


def _epsilon_at_10um() -> dict[str, complex]:
    wavelength_nm, epsilon_a, epsilon_b, epsilon_c = _epsilon_table()
    result = {}
    for axis, values in (("a", epsilon_a), ("b", epsilon_b), ("c", epsilon_c)):
        result[axis] = complex(
            np.interp(WAVELENGTH_NM, wavelength_nm, values.real),
            np.interp(WAVELENGTH_NM, wavelength_nm, values.imag),
        )
    return result


def _inside(coords: dict[str, np.ndarray], bounds: dict[str, tuple[float, float]], *, upper_z: bool) -> np.ndarray:
    x = np.asarray(coords["x"], float)
    y = np.asarray(coords["y"], float)
    z = np.asarray(coords["z"], float)
    z_low, z_high = bounds["z"]
    z_condition = (z >= z_low) & (z <= z_high) if upper_z else (z >= z_low) & (z < z_high)
    return (
        (x[:, None, None] >= bounds["x"][0])
        & (x[:, None, None] <= bounds["x"][1])
        & (y[None, :, None] >= bounds["y"][0])
        & (y[None, :, None] <= bounds["y"][1])
        & z_condition[None, None, :]
    )


def _readback_and_partition(base, fdtd, q: dict[str, object], *, au_present: bool) -> dict[str, object]:
    spatial_shape = tuple(np.asarray(q["base_coordinates"][axis]).size for axis in "xyz")
    target_ta = _epsilon_at_10um()
    target_au = complex(base.au_complex_index(1.0)[0])
    result: dict[str, object] = {}
    for component, crystal_axis in zip("xyz", ("b", "a", "c")):
        raw_index = np.asarray(fdtd.getdata(PABS_INDEX, f"index_{component}", 1))
        refractive_index = frequency_slice(
            raw_index,
            spatial_shape,
            int(q["frequency_index_zero_based"]),
            int(q["frequency_count"]),
            f"index_{component}",
        )
        epsilon = refractive_index**2
        coords = q["native_coordinates"][component]
        ta_mask = _inside(coords, TAIRTE4_BOUNDS, upper_z=False)
        au_mask = _inside(coords, AU_BOUNDS, upper_z=True) if au_present else np.zeros(spatial_shape, dtype=bool)
        q_component = np.asarray(q["Q_components"][component], float)
        p_ta = integrate_xyz(q_component * ta_mask, coords["x"], coords["y"], coords["z"])
        p_au = integrate_xyz(q_component * au_mask, coords["x"], coords["y"], coords["z"])
        p_total = float(q["component_power_W"][component])

        ta_interior = _inside(
            coords,
            {
                "x": (TAIRTE4_BOUNDS["x"][0] + 0.3e-6, TAIRTE4_BOUNDS["x"][1] - 0.3e-6),
                "y": (TAIRTE4_BOUNDS["y"][0] + 0.3e-6, TAIRTE4_BOUNDS["y"][1] - 0.3e-6),
                "z": (TAIRTE4_BOUNDS["z"][0] + 15e-9, TAIRTE4_BOUNDS["z"][1] - 15e-9),
            },
            upper_z=True,
        )
        ta_values = epsilon[ta_interior & np.isfinite(epsilon)]
        if ta_values.size == 0:
            raise RuntimeError(f"empty TaIrTe4 interior readback for {component}")
        au_values = np.asarray([], complex)
        if au_present:
            au_interior = _inside(
                coords,
                {
                    "x": (AU_BOUNDS["x"][0] + 0.3e-6, AU_BOUNDS["x"][1] - 0.3e-6),
                    "y": (AU_BOUNDS["y"][0] + 0.3e-6, AU_BOUNDS["y"][1] - 0.3e-6),
                    "z": (AU_BOUNDS["z"][0] + 10e-9, AU_BOUNDS["z"][1] - 10e-9),
                },
                upper_z=True,
            )
            au_values = epsilon[au_interior & np.isfinite(epsilon)]
            if au_values.size == 0:
                raise RuntimeError(f"empty Au interior readback for {component}")
        result[component] = {
            "all_finite": bool(np.all(np.isfinite(epsilon)) and np.all(np.isfinite(q_component))),
            "crystal_axis": crystal_axis,
            "requested_tairte4_epsilon": [target_ta[crystal_axis].real, target_ta[crystal_axis].imag],
            "tairte4_epsilon_interior_median": [float(np.median(ta_values.real)), float(np.median(ta_values.imag))],
            "requested_au_epsilon": [target_au.real, target_au.imag] if au_present else None,
            "au_epsilon_interior_median": (
                [float(np.median(au_values.real)), float(np.median(au_values.imag))]
                if au_present
                else None
            ),
            "geometric_power_partition_W": {
                "TaIrTe4": p_ta,
                "Au": p_au,
                "component_total": p_total,
                "unassigned_or_interface_residual": p_total - p_ta - p_au,
            },
            "partition_note": "component-specific native Yee coordinates; z=0 assigned to upper Au when present; no Q deletion or rescaling",
        }
    return result


def _option_present(arguments: list[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in arguments)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--au-endpoint", type=int, required=True, choices=(0, 1))
    parsed, remaining = parser.parse_known_args()
    au_present = bool(parsed.au_endpoint)
    stage04 = load_stage04()
    original_load_legacy = stage04.load_legacy
    stage04.AU_BOUNDS = AU_BOUNDS
    stage04.FLUX_BOUNDS = FLUX_BOUNDS
    stage04.AU_DXY_M = LOCAL_DXY_M
    stage04.AU_DZ_M = LOCAL_DZ_M
    stage04.EDGE_DXY_M = None

    def load_fixed_stack_legacy():
        base = original_load_legacy()
        base.BLOCK_BOUNDS = AU_BOUNDS
        base.FLUX_BOUNDS = FLUX_BOUNDS

        def add_fixed_stack(fdtd, *, rho: float, representation: str):
            if representation != "scalar" or float(rho) != float(parsed.au_endpoint):
                raise ValueError("binary endpoint wrapper only accepts the requested scalar endpoint")
            wavelength_nm, epsilon_a, epsilon_b, epsilon_c = _epsilon_table()
            frequencies_hz = base.C0 / (wavelength_nm * 1e-9)
            material_id = fdtd.addmaterial("Sampled 3D data")
            fdtd.setmaterial(material_id, "name", "TaIrTe4_10um_anisotropic")
            fdtd.setmaterial("TaIrTe4_10um_anisotropic", "anisotropy", 1)
            fdtd.setmaterial("TaIrTe4_10um_anisotropic", "max coefficients", 20)
            fdtd.setmaterial(
                "TaIrTe4_10um_anisotropic",
                "sampled data",
                np.column_stack((frequencies_hz, epsilon_b, epsilon_a, epsilon_c)),
            )
            flake = fdtd.addrect()
            flake["name"] = "fixed_TaIrTe4"
            flake["material"] = "TaIrTe4_10um_anisotropic"
            for axis in "xyz":
                flake[f"{axis} min"], flake[f"{axis} max"] = TAIRTE4_BOUNDS[axis]

            requested_au = None
            if au_present:
                epsilon, index = stage04.au_complex_index(1.0)
                material = fdtd.addmaterial("(n,k) Material")
                fdtd.setmaterial(material, "name", "exact_Au_10um")
                fdtd.setmaterial("exact_Au_10um", "Refractive Index", float(index.real))
                fdtd.setmaterial("exact_Au_10um", "Imaginary Refractive Index", float(index.imag))
                au = fdtd.addrect()
                au["name"] = "exact_binary_Au_nanostructure"
                au["material"] = "exact_Au_10um"
                for axis in "xyz":
                    au[f"{axis} min"], au[f"{axis} max"] = AU_BOUNDS[axis]
                requested_au = [float(epsilon.real), float(epsilon.imag)]
            return {
                "name": "fixed_TaIrTe4_plus_optional_exact_Au",
                "representation": "exact_binary_scalar_materials",
                "au_endpoint": int(au_present),
                "au_is_nanostructure_not_electrode": True,
                "requested_au_epsilon": requested_au,
                "axis_mapping": {"x": "b", "y": "a", "z": "c=b closure"},
                "tairte4_bounds_m": TAIRTE4_BOUNDS,
                "au_bounds_m": AU_BOUNDS if au_present else None,
                "direct_optical_contact": au_present,
            }

        base.add_design = add_fixed_stack
        return base

    def add_stack_mesh(fdtd):
        mesh = fdtd.addmesh()
        mesh["name"] = "Au_TaIrTe4_local_mesh"
        mesh["x min"], mesh["x max"] = FLUX_BOUNDS["x"]
        mesh["y min"], mesh["y max"] = FLUX_BOUNDS["y"]
        mesh["z min"], mesh["z max"] = (-0.16e-6, 0.11e-6)
        mesh["override x mesh"] = True
        mesh["override y mesh"] = True
        mesh["override z mesh"] = True
        mesh["dx"] = LOCAL_DXY_M
        mesh["dy"] = LOCAL_DXY_M
        mesh["dz"] = LOCAL_DZ_M

    def readback(configured_base, fdtd, q):
        return _readback_and_partition(configured_base, fdtd, q, au_present=au_present)

    stage04.load_legacy = load_fixed_stack_legacy
    stage04.add_local_mesh = add_stack_mesh
    stage04.component_epsilon_readback = readback
    defaults = [
        ("--rho", str(parsed.au_endpoint)),
        ("--representation", "scalar"),
        ("--mesh-refinement", "conformal variant 1"),
        ("--dt-stability-factor", "0.5"),
        ("--mesh-wavelength-um", "10"),
        ("--duration-ps", "8"),
        ("--auto-shutoff-min", "1e-7"),
    ]
    for option, value in defaults:
        if not _option_present(remaining, option):
            remaining.extend((option, value))
    sys.argv = [sys.argv[0], *remaining]
    code = int(stage04.main())

    output_dir = None
    for index, value in enumerate(remaining):
        if value == "--output-dir" and index + 1 < len(remaining):
            output_dir = Path(remaining[index + 1]).expanduser().resolve()
        elif value.startswith("--output-dir="):
            output_dir = Path(value.split("=", 1)[1]).expanduser().resolve()
    if output_dir is not None:
        result_path = output_dir / "case_result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["endpoint_crosscheck_contract"] = {
                "Au_is_nanostructure_not_electrode": True,
                "Au_endpoint": parsed.au_endpoint,
                "TaIrTe4_fixed": True,
                "axis_mapping": "Lumerical x=b, y=a, z=c=b closure",
                "raw_files_committed": False,
                "no_gray_or_imported_metal": True,
                "no_adjoint": True,
            }
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
