#!/usr/bin/env python3
"""GPU-only v261 optical smoke for the 2024 MIR inverse-T TaIrTe4 control.

This is a paper-derived *scenario*, not a reproduction of the graphene
experiment.  The 1500 x 1000 nm unit cell and 4.75-um target are disclosed in
Supplementary Fig. 14.  Arm vertices are digitized from its physical axes and
the active graphene is replaced only by fixed 100-nm anisotropic TaIrTe4.

No thermal, electrical, PTE, adjoint, or optimization calculation is run.
Raw FSP/NPZ artifacts are written outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.interpolate import RegularGridInterpolator


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
PERMITTIVITY_PATH = REPOSITORY / "photothermal_pte" / "bundle" / "perm_data.txt"
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.native_yee_q import (  # noqa: E402
    extract_native_yee_q,
    frequency_slice,
)
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_FIELD,
    PABS_GROUP,
    PABS_INDEX,
)
from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    validate_paper_ir_source_only_gpu as audit,
)


C0 = 299_792_458.0
WAVELENGTH_M = 4.75e-6
FREQUENCY_HZ = C0 / WAVELENGTH_M
PERIOD_X_M = 1.5e-6
PERIOD_Y_M = 1.0e-6
FDTD_Z_MIN_AU_TRUNCATED_M = -1.0e-6
FDTD_Z_MIN_SIO2_SI_REDUCED_M = -1.2e-6
FDTD_Z_MIN_SIO2_SI_FULL_M = -2.5e-6
FDTD_Z_MAX_M = 1.2e-6
SOURCE_Z_M = 0.8e-6
TOP_MONITOR_Z_M = 0.45e-6
BOTTOM_MONITOR_AU_TRUNCATED_Z_M = -0.65e-6
BOTTOM_MONITOR_SIO2_SI_REDUCED_Z_M = -0.70e-6
BOTTOM_MONITOR_SIO2_SI_FULL_Z_M = -2.0e-6
AU_MIRROR_TOP_M = -35.0e-9
AU_MIRROR_BOTTOM_M = -235.0e-9
SIO2_REDUCED_THICKNESS_M = 285.0e-9
SIO2_FULL_THICKNESS_M = 1.5e-6
AL2O3_N = 1.62 + 0.0j
AU_MATERIAL = "Au (Gold) - CRC"
SIO2_MATERIAL = "SiO2 (Glass) - Palik"
SI_MATERIAL = "Si (Silicon) - Palik"
TAIRTE4_MATERIAL = "TaIrTe4_100nm_2024T_substitution"
AL2O3_MATERIAL = "Al2O3_lossless_n1p62_explicit_closure"
SOURCE_NAME = "T2024_normal_incidence_plane_wave"
TOP_MONITOR = "T2024_flux_top"
BOTTOM_MONITOR = "T2024_flux_bottom"


def common_field_slices(fdtd: object, q: dict[str, object]) -> dict[str, np.ndarray]:
    """Collocate complex E on three compact physical cross sections.

    The native Yee components are first paired with their own shifted physical
    coordinates. Interpolation is restricted to the common coordinate
    intersection, so no boundary extrapolation or same-index pairing is used.
    """

    frequency_hz = np.asarray(fdtd.getdata(PABS_FIELD, "f", 1), float).reshape(-1)
    frequency_index = int(q["frequency_index_zero_based"])
    base = {
        axis: np.asarray(q["base_coordinates"][axis], float).reshape(-1)
        for axis in "xyz"
    }
    native = q["native_coordinates"]
    interpolators: dict[str, RegularGridInterpolator] = {}
    for component in "xyz":
        shape = tuple(base[axis].size for axis in "xyz")
        electric = frequency_slice(
            np.asarray(fdtd.getdata(PABS_FIELD, f"E{component}", 1)),
            shape,
            frequency_index,
            frequency_hz.size,
            f"E{component}",
        )
        interpolators[component] = RegularGridInterpolator(
            tuple(np.asarray(native[component][axis], float) for axis in "xyz"),
            np.asarray(electric, complex),
            method="linear",
            bounds_error=True,
        )

    common_bounds = {
        axis: (
            max(float(np.min(native[component][axis])) for component in "xyz"),
            min(float(np.max(native[component][axis])) for component in "xyz"),
        )
        for axis in "xyz"
    }
    common = {
        axis: base[axis][
            (base[axis] >= common_bounds[axis][0])
            & (base[axis] <= common_bounds[axis][1])
        ]
        for axis in "xyz"
    }
    if min(values.size for values in common.values()) < 2:
        raise RuntimeError(f"empty common Yee intersection: {common_bounds}")

    result: dict[str, np.ndarray] = {}

    def evaluate_plane(
        label: str,
        first_axis: str,
        first: np.ndarray,
        second_axis: str,
        second: np.ndarray,
        fixed_axis: str,
        fixed_value: float,
    ) -> None:
        mesh_first, mesh_second = np.meshgrid(first, second, indexing="ij")
        coordinates = {
            first_axis: mesh_first,
            second_axis: mesh_second,
            fixed_axis: np.full_like(mesh_first, fixed_value),
        }
        points = np.column_stack(
            tuple(coordinates[axis].reshape(-1) for axis in "xyz")
        )
        intensity = np.zeros(mesh_first.shape, float)
        for component in "xyz":
            electric = interpolators[component](points).reshape(mesh_first.shape)
            result[f"field_{label}_E{component}"] = electric
            intensity += np.abs(electric) ** 2
        result[f"field_{label}_E2_V2_m2"] = intensity
        result[f"field_{label}_{first_axis}_m"] = np.asarray(first, float)
        result[f"field_{label}_{second_axis}_m"] = np.asarray(second, float)
        result[f"field_{label}_{fixed_axis}_m"] = np.asarray([fixed_value], float)

    z_mid = 50.0e-9
    if not common_bounds["z"][0] <= z_mid <= common_bounds["z"][1]:
        raise RuntimeError("TaIrTe4 midplane lies outside common Yee support")
    near_z = common["z"][(common["z"] >= -0.30e-6) & (common["z"] <= 0.35e-6)]
    evaluate_plane("xy", "x", common["x"], "y", common["y"], "z", z_mid)
    evaluate_plane("xz", "x", common["x"], "z", near_z, "y", 0.0)
    evaluate_plane("yz", "y", common["y"], "z", near_z, "x", 0.0)
    result["field_common_bounds_m"] = np.asarray(
        [common_bounds[axis] for axis in "xyz"], float
    )
    return result


def load_local_module(filename: str, name: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value: object, label: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise RuntimeError(f"{label} is not scalar: {array.shape}")
    result = float(np.real(array[0]))
    if not np.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def complex_record(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def epsilon_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(PERMITTIVITY_PATH)
    data = data[np.argsort(data[:, 0])]
    wavelength_nm = data[:, 0]
    epsilon_a = data[:, 1] + 1j * data[:, 2]
    epsilon_b = data[:, 3] + 1j * data[:, 4]
    epsilon_c = data[:, 5] + 1j * data[:, 6]
    if not np.array_equal(epsilon_b, epsilon_c):
        raise RuntimeError("perm_data.txt no longer satisfies epsilon_c=epsilon_b")
    return wavelength_nm, epsilon_a, epsilon_b, epsilon_c


def epsilon_at_wavelength() -> dict[str, complex]:
    wavelength_nm, epsilon_a, epsilon_b, epsilon_c = epsilon_table()
    target_nm = WAVELENGTH_M * 1.0e9
    result: dict[str, complex] = {}
    for key, values in (("a", epsilon_a), ("b", epsilon_b), ("c", epsilon_c)):
        result[key] = complex(
            np.interp(target_nm, wavelength_nm, values.real),
            np.interp(target_nm, wavelength_nm, values.imag),
        )
    return result


def add_tairte4_material(fdtd: object) -> dict[str, object]:
    wavelength_nm, epsilon_a, epsilon_b, epsilon_c = epsilon_table()
    frequencies_hz = C0 / (wavelength_nm * 1.0e-9)
    material = fdtd.addmaterial("Sampled 3D data")
    fdtd.setmaterial(material, "name", TAIRTE4_MATERIAL)
    fdtd.setmaterial(TAIRTE4_MATERIAL, "anisotropy", 1)
    fdtd.setmaterial(TAIRTE4_MATERIAL, "max coefficients", 20)
    fdtd.setmaterial(
        TAIRTE4_MATERIAL,
        "sampled data",
        np.column_stack((frequencies_hz, epsilon_b, epsilon_a, epsilon_c)),
    )
    requested = epsilon_at_wavelength()
    return {
        "name": TAIRTE4_MATERIAL,
        "axis_mapping": {"x": "b", "y": "a", "z": "c=b closure"},
        "requested_epsilon": {
            "x": complex_record(requested["b"]),
            "y": complex_record(requested["a"]),
            "z": complex_record(requested["c"]),
        },
        "permittivity_source": str(PERMITTIVITY_PATH),
        "permittivity_sha256": sha256(PERMITTIVITY_PATH),
    }


def add_constant_nk(fdtd: object, name: str, value: complex) -> None:
    material = fdtd.addmaterial("(n,k) Material")
    fdtd.setmaterial(material, "name", name)
    fdtd.setmaterial(name, "Refractive Index", float(value.real))
    fdtd.setmaterial(name, "Imaginary Refractive Index", float(value.imag))


def add_rect(fdtd: object, name: str, material: str, z_min: float, z_max: float) -> None:
    rectangle = fdtd.addrect()
    rectangle["name"] = name
    rectangle["material"] = material
    rectangle["x min"] = -0.5 * PERIOD_X_M
    rectangle["x max"] = 0.5 * PERIOD_X_M
    rectangle["y min"] = -0.5 * PERIOD_Y_M
    rectangle["y max"] = 0.5 * PERIOD_Y_M
    rectangle["z min"] = z_min
    rectangle["z max"] = z_max


def add_power_monitor(fdtd: object, name: str, z_m: float) -> None:
    monitor = fdtd.addpower()
    monitor["name"] = name
    monitor["monitor type"] = "2D Z-normal"
    monitor["x min"] = -0.5 * PERIOD_X_M
    monitor["x max"] = 0.5 * PERIOD_X_M
    monitor["y min"] = -0.5 * PERIOD_Y_M
    monitor["y max"] = 0.5 * PERIOD_Y_M
    monitor["z"] = z_m
    monitor["override global monitor settings"] = True
    monitor["use source limits"] = False
    monitor["use wavelength spacing"] = True
    monitor["wavelength center"] = WAVELENGTH_M
    monitor["wavelength span"] = 0.0
    monitor["frequency points"] = 1


def setup(
    fdtd: object,
    polarization: str,
    duration_ps: float,
    *,
    include_top_t: bool = True,
    substrate_mode: str = "sio2_si_reduced_285nm",
) -> dict[str, object]:
    geometry_module = load_local_module(
        "05_actual_metasurface_geometry.py", "paper_actual_metasurface_geometry"
    )
    backplane_module = load_local_module(
        "02_run_v261_backplane_truncation_control.py", "paper_backplane_helpers"
    )
    geometry = geometry_module.inverse_t_mir_4750nm()

    if substrate_mode == "sio2_si_reduced_285nm":
        fdtd_z_min_m = FDTD_Z_MIN_SIO2_SI_REDUCED_M
        bottom_monitor_z_m = BOTTOM_MONITOR_SIO2_SI_REDUCED_Z_M
        sio2_thickness_m = SIO2_REDUCED_THICKNESS_M
        oxide_bottom_m = AU_MIRROR_BOTTOM_M - sio2_thickness_m
    elif substrate_mode in ("sio2_si", "sio2_si_full_1500nm"):
        fdtd_z_min_m = FDTD_Z_MIN_SIO2_SI_FULL_M
        bottom_monitor_z_m = BOTTOM_MONITOR_SIO2_SI_FULL_Z_M
        sio2_thickness_m = SIO2_FULL_THICKNESS_M
        oxide_bottom_m = AU_MIRROR_BOTTOM_M - sio2_thickness_m
    elif substrate_mode == "au_truncated":
        fdtd_z_min_m = FDTD_Z_MIN_AU_TRUNCATED_M
        bottom_monitor_z_m = BOTTOM_MONITOR_AU_TRUNCATED_Z_M
        sio2_thickness_m = None
        oxide_bottom_m = None
    else:
        raise ValueError(f"unsupported substrate mode: {substrate_mode}")

    solver = fdtd.addfdtd()
    solver["dimension"] = "3D"
    solver["x min"] = -0.5 * PERIOD_X_M
    solver["x max"] = 0.5 * PERIOD_X_M
    solver["y min"] = -0.5 * PERIOD_Y_M
    solver["y max"] = 0.5 * PERIOD_Y_M
    solver["z min"] = fdtd_z_min_m
    solver["z max"] = FDTD_Z_MAX_M
    solver["x min bc"] = "Periodic"
    solver["x max bc"] = "Periodic"
    solver["y min bc"] = "Periodic"
    solver["y max bc"] = "Periodic"
    solver["z min bc"] = "PML"
    solver["z max bc"] = "PML"
    solver["pml layers"] = 24
    solver["mesh type"] = "auto non-uniform"
    solver["mesh refinement"] = "conformal variant 1"
    solver["mesh accuracy"] = 3
    solver["simulation time"] = duration_ps * 1.0e-12
    solver["auto shutoff min"] = 1.0e-6
    solver["override simulation bandwidth for mesh generation"] = True
    solver["mesh wavelength min"] = WAVELENGTH_M
    solver["mesh wavelength max"] = WAVELENGTH_M

    source = fdtd.addplane()
    source["name"] = SOURCE_NAME
    source["injection axis"] = "z"
    source["direction"] = "backward"
    source["polarization angle"] = 0.0 if polarization == "x_b" else 90.0
    source["angle theta"] = 0.0
    source["angle phi"] = 0.0
    source["plane wave type"] = "Bloch/Periodic"
    source["x min"] = -0.5 * PERIOD_X_M
    source["x max"] = 0.5 * PERIOD_X_M
    source["y min"] = -0.5 * PERIOD_Y_M
    source["y max"] = 0.5 * PERIOD_Y_M
    source["z"] = SOURCE_Z_M
    source["override global source settings"] = True
    source["wavelength start"] = 4.5e-6
    source["wavelength stop"] = 5.0e-6

    fdtd.setglobalmonitor("use source limits", False)
    fdtd.setglobalmonitor("use wavelength spacing", True)
    fdtd.setglobalmonitor("wavelength center", WAVELENGTH_M)
    fdtd.setglobalmonitor("wavelength span", 0.0)
    fdtd.setglobalmonitor("frequency points", 1)

    tairte4 = add_tairte4_material(fdtd)
    add_constant_nk(fdtd, AL2O3_MATERIAL, AL2O3_N)
    au_index = complex(np.asarray(fdtd.getindex(AU_MATERIAL, FREQUENCY_HZ)).reshape(-1)[0])
    explicit_sio2_si = substrate_mode != "au_truncated"
    if explicit_sio2_si:
        sio2_index = complex(
            np.asarray(fdtd.getindex(SIO2_MATERIAL, FREQUENCY_HZ)).reshape(-1)[0]
        )
        si_index = complex(
            np.asarray(fdtd.getindex(SI_MATERIAL, FREQUENCY_HZ)).reshape(-1)[0]
        )
        add_rect(
            fdtd,
            "paper_Au_mirror_200nm",
            AU_MATERIAL,
            AU_MIRROR_BOTTOM_M,
            AU_MIRROR_TOP_M,
        )
        add_rect(
            fdtd,
            f"optical_SiO2_{sio2_thickness_m * 1e9:.0f}nm",
            SIO2_MATERIAL,
            oxide_bottom_m,
            AU_MIRROR_BOTTOM_M,
        )
        add_rect(
            fdtd,
            "paper_intrinsic_Si_substrate",
            SI_MATERIAL,
            fdtd_z_min_m,
            oxide_bottom_m,
        )
    else:
        sio2_index = None
        si_index = None
        add_rect(
            fdtd,
            "Au_mirror_extended_through_bottom_PML",
            AU_MATERIAL,
            fdtd_z_min_m,
            AU_MIRROR_TOP_M,
        )
    add_rect(fdtd, "paper_Al2O3_35nm", AL2O3_MATERIAL, -35.0e-9, 0.0)
    add_rect(fdtd, "TaIrTe4_active_100nm", TAIRTE4_MATERIAL, 0.0, 100.0e-9)

    polygon_contract = geometry.polygons[0]
    if include_top_t:
        polygon = fdtd.addpoly()
        polygon["name"] = polygon_contract.name
        polygon["material"] = AU_MATERIAL
        polygon["vertices"] = np.asarray(polygon_contract.vertices_nm, float) * 1.0e-9
        polygon["z min"] = polygon_contract.z_min_nm * 1.0e-9
        polygon["z max"] = polygon_contract.z_max_nm * 1.0e-9

    mesh = fdtd.addmesh()
    mesh["name"] = "T2024_local_structure_mesh"
    mesh["x min"] = -0.5 * PERIOD_X_M
    mesh["x max"] = 0.5 * PERIOD_X_M
    mesh["y min"] = -0.5 * PERIOD_Y_M
    mesh["y max"] = 0.5 * PERIOD_Y_M
    mesh["z min"] = -0.30e-6
    mesh["z max"] = 0.20e-6
    mesh["override x mesh"] = True
    mesh["override y mesh"] = True
    mesh["override z mesh"] = True
    mesh["dx"] = 10.0e-9
    mesh["dy"] = 10.0e-9
    mesh["dz"] = 5.0e-9

    pabs = fdtd.addobject("pabs_adv")
    pabs["name"] = PABS_GROUP
    pabs["x"] = 0.0
    pabs["x span"] = PERIOD_X_M
    pabs["y"] = 0.0
    pabs["y span"] = PERIOD_Y_M
    pabs["z"] = 0.5 * (TOP_MONITOR_Z_M + bottom_monitor_z_m)
    pabs["z span"] = TOP_MONITOR_Z_M - bottom_monitor_z_m
    pabs_contract = backplane_module.enable_pabs_periodic_correction(fdtd)
    add_power_monitor(fdtd, TOP_MONITOR, TOP_MONITOR_Z_M)
    add_power_monitor(fdtd, BOTTOM_MONITOR, bottom_monitor_z_m)

    return {
        "geometry": geometry.as_dict(),
        "source": {
            "type": "normal-incidence Bloch/Periodic plane wave",
            "polarization": polarization,
            "wavelength_m": WAVELENGTH_M,
            "z_m": SOURCE_Z_M,
        },
        "top_Au_T_present": include_top_t,
        "case_identity": (
            "paper_derived_inverse_T_TaIrTe4_substitution"
            if include_top_t
            else "matched_bare_TaIrTe4_control_without_top_T"
        ),
        "materials": {
            "TaIrTe4": tairte4,
            "Au": {
                "Lumerical_material": AU_MATERIAL,
                "readback_n_plus_ik": complex_record(au_index),
                "paper_dataset_limit": "2024 paper does not identify the optical Au dataset",
            },
            "Al2O3": {
                "n_plus_ik": complex_record(AL2O3_N),
                "status": "explicit_lossless_optical_closure_not_paper_certified_dataset",
            },
            "SiO2": {
                "Lumerical_material": SIO2_MATERIAL if sio2_index is not None else None,
                "readback_n_plus_ik": complex_record(sio2_index) if sio2_index is not None else None,
                "thickness_m": sio2_thickness_m,
                "provenance": (
                    "reduced 285-nm optical closure below an opaque 200-nm Au mirror; "
                    "not the physical 1.5-um oxide thickness"
                    if substrate_mode == "sio2_si_reduced_285nm"
                    else "2024 main Methods: 1.5-um thermally grown SiO2"
                ),
            },
            "Si": {
                "Lumerical_material": SI_MATERIAL if si_index is not None else None,
                "readback_n_plus_ik": complex_record(si_index) if si_index is not None else None,
                "provenance": "2024 main Methods: intrinsic Si substrate",
            },
        },
        "substrate": {
            "mode": substrate_mode,
            "domain_z_min_m": fdtd_z_min_m,
            "bottom_flux_monitor_z_m": bottom_monitor_z_m,
            "Au_mirror_bounds_m": [
                AU_MIRROR_BOTTOM_M if explicit_sio2_si else fdtd_z_min_m,
                AU_MIRROR_TOP_M,
            ],
            "SiO2_bounds_m": [oxide_bottom_m, AU_MIRROR_BOTTOM_M]
            if explicit_sio2_si
            else None,
            "Si_bounds_m": [fdtd_z_min_m, oxide_bottom_m]
            if explicit_sio2_si
            else None,
            "paper_identity": (
                (
                    "reduced 285-nm optical-SiO2 / intrinsic-Si closure; "
                    "physical thermal thickness is not implied"
                    if substrate_mode == "sio2_si_reduced_285nm"
                    else "explicit 1.5-um thermal-SiO2 / intrinsic-Si stack"
                )
                if explicit_sio2_si
                else "legacy numerical Au-to-bottom-PML closure"
            ),
        },
        "pabs_contract": pabs_contract,
        "scope": "optical forward smoke only; no thermal/PTE/adjoint/optimization",
    }


def mesh_metrics(mesh: dict[str, object]) -> dict[str, object]:
    if not mesh.get("available"):
        return {"available": False}
    coordinates = mesh["coordinate_arrays"]
    x = np.asarray(coordinates["x"], float)
    y = np.asarray(coordinates["y"], float)
    z = np.asarray(coordinates["z"], float)
    z_steps = np.diff(z)
    structure_steps = z_steps[
        (0.5 * (z[:-1] + z[1:]) >= -0.30e-6)
        & (0.5 * (z[:-1] + z[1:]) <= 0.20e-6)
    ]
    return {
        "available": True,
        "shape": [int(x.size), int(y.size), int(z.size)],
        "yee_cell_estimate": int(max(x.size - 1, 0) * max(y.size - 1, 0) * max(z.size - 1, 0)),
        "max_dx_m": float(np.max(np.diff(x))),
        "max_dy_m": float(np.max(np.diff(y))),
        "max_structure_dz_m": float(np.max(structure_steps)),
        "min_dx_m": float(np.min(np.diff(x))),
        "min_dy_m": float(np.min(np.diff(y))),
        "min_dz_m": float(np.min(np.diff(z))),
        "bounds_m": {"x": [float(x[0]), float(x[-1])], "y": [float(y[0]), float(y[-1])], "z": [float(z[0]), float(z[-1])]},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument("--polarization", choices=("x_b", "y_a"), default="x_b")
    parser.add_argument("--duration-ps", type=float, default=1.0)
    parser.add_argument(
        "--substrate-mode",
        choices=(
            "sio2_si_reduced_285nm",
            "sio2_si_full_1500nm",
            "sio2_si",
            "au_truncated",
        ),
        default="sio2_si_reduced_285nm",
        help=(
            "Use the reduced 285-nm optical SiO2/Si closure by default. "
            "The full physical 1.5-um oxide remains an explicit control."
        ),
    )
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument(
        "--omit-top-t-control",
        action="store_true",
        help="Remove only the top inverse-T while retaining the matched mirror/spacer/TaIrTe4 stack.",
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "T2024_TaIrTe4_optical_smoke.json"
    fsp_path = output / "T2024_TaIrTe4_optical_smoke.fsp"
    npz_path = output / "T2024_TaIrTe4_native_q.npz"
    result: dict[str, object] = {"status": "BLOCKED_T2024_TAIRTE4_OPTICAL_SMOKE"}
    fdtd = None
    try:
        os.environ["VC_LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
        os.environ["LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
        os.environ["LUMERICAL_PYTHONPATH"] = str(audit.APPROVED_API)
        os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = args.gpu_device
        os.environ["CL_GPU_DEVICE"] = args.gpu_device
        os.environ["FDTD_THREADS"] = "8"
        os.environ["PATH"] = f"{audit.APPROVED_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
        sys.path.insert(0, str(audit.APPROVED_API))
        import lumapi

        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        contract = setup(
            fdtd,
            args.polarization,
            args.duration_ps,
            include_top_t=not args.omit_top_t_control,
            substrate_mode=args.substrate_mode,
        )
        fdtd.setresource("FDTD", 1, "active", 0)
        fdtd.setresource("FDTD", 2, "active", 1)
        fdtd.setresource("FDTD", 2, "processes", "1")
        fdtd.setresource("FDTD", 2, "threads", "8")
        fdtd.setresource("FDTD", 2, "device type", args.gpu_device)
        fdtd.setresource("FDTD", 2, "solver extra command line options", "-gpu")
        fdtd.runsetup()
        mesh = audit.mesh_readback(fdtd)
        metrics = mesh_metrics(mesh)
        if not metrics.get("available"):
            raise RuntimeError("native mesh unavailable after runsetup")
        if metrics["min_dx_m"] > 10.0e-9 + 1.0e-12 or metrics["min_dy_m"] > 10.0e-9 + 1.0e-12:
            raise RuntimeError("10-nm in-plane structure mesh was not realized")
        if metrics["max_structure_dz_m"] > 5.0e-9 + 1.0e-12:
            raise RuntimeError("5-nm structure dz was not realized")
        result.update(
            {
                "contract": contract,
                "solver_version": str(fdtd.version()),
                "mesh_runsetup": metrics,
                "resource": {
                    prop: str(fdtd.getresource("FDTD", 2, prop))
                    for prop in ("active", "device type", "processes", "threads", "solver extra command line options")
                },
            }
        )
        fdtd.save(str(fsp_path))
        if args.contract_only:
            result["status"] = "COMPLETED_T2024_TAIRTE4_RUNSETUP_AUDIT"
        else:
            started = time.monotonic()
            result["GPU_resource_used"] = audit.strict_gpu_run(fdtd, "T2024_TaIrTe4_optical_smoke")
            result["solver_wall_time_s"] = time.monotonic() - started
            source_power = scalar(fdtd.sourcepower(FREQUENCY_HZ, 2, SOURCE_NAME), "sourcepower")
            top_signed = scalar(fdtd.transmission(TOP_MONITOR), TOP_MONITOR) * source_power
            bottom_signed = scalar(fdtd.transmission(BOTTOM_MONITOR), BOTTOM_MONITOR) * source_power
            p_flux = bottom_signed - top_signed
            fdtd.runanalysis(PABS_GROUP)
            pabs_normalized = scalar(fdtd.getresult(PABS_GROUP, "Pabs_total")["Pabs_total"], "Pabs_total")
            p_q = pabs_normalized * source_power
            q = extract_native_yee_q(fdtd, field_monitor=PABS_FIELD, index_monitor=PABS_INDEX, wavelength_m=WAVELENGTH_M)
            p_q_native = float(q["P_Q_W"])
            closure = abs(p_q - p_flux) / max(abs(p_q), abs(p_flux), 1.0e-300)
            negative = {component: int(np.count_nonzero(np.asarray(q["Q_components"][component]) < 0.0)) for component in "xyz"}
            finite = bool(all(np.all(np.isfinite(np.asarray(q["Q_components"][component]))) for component in "xyz"))
            arrays: dict[str, np.ndarray] = {}
            for component in "xyz":
                arrays[f"Q{component}_W_m3"] = np.asarray(q["Q_components"][component])
                for axis in "xyz":
                    arrays[f"Q{component}_{axis}_m"] = np.asarray(q["native_coordinates"][component][axis])
                arrays[f"top_E{component}"] = np.asarray(fdtd.getdata(TOP_MONITOR, f"E{component}", 1)).squeeze()
            arrays["top_x_m"] = np.asarray(fdtd.getdata(TOP_MONITOR, "x", 1), float)
            arrays["top_y_m"] = np.asarray(fdtd.getdata(TOP_MONITOR, "y", 1), float)
            arrays.update(common_field_slices(fdtd, q))
            np.savez_compressed(npz_path, **arrays)
            result.update(
                {
                    "source_power_W": source_power,
                    "P_flux_absorbed_W": p_flux,
                    "P_Q_pabs_periodic_W": p_q,
                    "P_Q_native_uncorrected_W": p_q_native,
                    "Q_component_power_native_W": q["component_power_W"],
                    "closure_relative": closure,
                    "reflection": 1.0 + top_signed / source_power,
                    "transmission_bottom_monitor": -bottom_signed / source_power,
                    "transmission_inside_Au_diagnostic": (
                        -bottom_signed / source_power
                        if args.substrate_mode == "au_truncated"
                        else None
                    ),
                    "all_Q_arrays_finite": finite,
                    "negative_Q_cell_count": negative,
                    "log_audit": audit.log_audit(output),
                }
            )
            gates = {
                "GPU_completed": bool(result["log_audit"]["simulation_completed_successfully"]),
                "auto_shutoff_lt_1e_5": result["log_audit"]["final_auto_shutoff"] is not None and result["log_audit"]["final_auto_shutoff"] < 1.0e-5,
                "closure_lt_0p5pct": closure < 0.005,
                "all_Q_arrays_finite": finite,
                "no_negative_Q": sum(negative.values()) == 0,
            }
            result["gates"] = gates
            result["status"] = "COMPLETED_T2024_TAIRTE4_OPTICAL_SMOKE" if all(gates.values()) else "FAILED_T2024_TAIRTE4_OPTICAL_SMOKE_GATE"
            fdtd.save(str(fsp_path))
        result["raw_artifacts"] = [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (fsp_path, npz_path)
            if path.is_file()
        ]
    except Exception as exc:
        result["status"] = "BLOCKED_T2024_TAIRTE4_OPTICAL_SMOKE"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0 if str(result["status"]).startswith("COMPLETED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
