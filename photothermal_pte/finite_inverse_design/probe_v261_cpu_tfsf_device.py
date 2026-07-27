#!/usr/bin/env python3
"""CPU TFSF gate for the finite TaIrTe4/SiO2 device geometry.

This is a forward optical gate only.  It never runs HEAT, an adjoint,
finite differences, or optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .probe_v261_cpu_tfsf_roi import (
    add_single_frequency_monitor_settings,
    optional_mesh_contract,
    run_on_cpu,
)
from .probe_v261_gpu_plane_wave_roi import (
    APPROVED_API,
    APPROVED_ROOT,
    json_default,
    load_lumapi,
    resource_snapshot,
    scalar,
    source_readback,
)


C0 = 299792458.0
EPS0 = 8.8541878128e-12
WAVELENGTH_M = 4.0e-6
FREQUENCY_HZ = C0 / WAVELENGTH_M
FLAKE_SPAN_M = 4.0e-6
FLAKE_Z = (-100.0e-9, 0.0)
DESIGN_SPAN_M = 2.0e-6
DESIGN_Z = (0.0, 600.0e-9)
SIO2_Z = (-385.0e-9, -100.0e-9)
SI_Z_TOP_M = -385.0e-9
FDTD_Z = (-2.385e-6, 2.0e-6)
TFSF_Z = (-1.8e-6, 1.5e-6)
PABS_GROUP = "finite_device_pabs"
PABS_FIELD = f"{PABS_GROUP}::field"
PABS_INDEX = f"{PABS_GROUP}::index"
SOURCE_NAME = "finite_device_tfsf"
TAIRTE4_MATERIAL = "TaIrTe4_ani"
SIO2_MATERIAL = "finite_actual_SiO2_n1p38"
# Measured once from the all-PML d=6 um, PML-24 layered empty-stack gate:
# analytic vacuum E amplitude produced sourceintensity=0.9946285135316062
# W/m2 at 4 um in v261.  This pre-solve amplitude calibration is held fixed
# for every device and FD run; fields/Q are never rescaled after a solve.
V261_TFSF_AMPLITUDE_CALIBRATION = 1.0026966117201364


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--case",
        required=True,
        choices=("empty-stack", "flat-flake", "full-sio2"),
    )
    parser.add_argument("--domain-um", type=float, default=6.0)
    parser.add_argument("--tfsf-span-um", type=float, default=4.6)
    parser.add_argument("--pml-layers", type=int, default=24)
    parser.add_argument("--flake-dz-nm", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--simulation-time-ps", type=float, default=4.0)
    args = parser.parse_args()
    if args.domain_um <= args.tfsf_span_um:
        parser.error("FDTD domain must contain the TFSF box")
    if args.tfsf_span_um <= FLAKE_SPAN_M * 1e6:
        parser.error("TFSF box must contain the complete finite flake")
    if args.pml_layers < 8 or args.flake_dz_nm <= 0 or args.threads < 1:
        parser.error("invalid numerical setting")
    return args


def add_rect(
    fdtd: object,
    name: str,
    *,
    x_span: float,
    y_span: float,
    z_min: float,
    z_max: float,
    material: str | None = None,
    index: float | None = None,
) -> None:
    item = fdtd.addrect()
    item["name"] = name
    item["x span"] = x_span
    item["y span"] = y_span
    item["z min"] = z_min
    item["z max"] = z_max
    if material is not None:
        item["material"] = material
    elif index is not None:
        item["index"] = float(index)
    else:
        raise ValueError("material or index is required")


def add_sio2_material(fdtd: object) -> None:
    material = fdtd.addmaterial("Dielectric")
    fdtd.setmaterial(material, "name", SIO2_MATERIAL)
    fdtd.setmaterial(SIO2_MATERIAL, "Refractive Index", 1.38)


def add_tairte4_material(fdtd: object) -> None:
    """Install the same anisotropic sampled data without importing a builder."""
    data_path = (
        Path(__file__).resolve().parents[1] / "bundle" / "perm_data.txt"
    )
    data = np.loadtxt(data_path)
    order = np.argsort(data[:, 0])
    wavelength_nm = data[order, 0]
    epsilon_a = (data[:, 1] + 1j * data[:, 2])[order]
    epsilon_b = (data[:, 3] + 1j * data[:, 4])[order]
    sample_wavelength_nm = np.linspace(2700.0, 13200.0, 600)
    epsilon_x = np.interp(
        sample_wavelength_nm, wavelength_nm, epsilon_a.real
    ) + 1j * np.interp(
        sample_wavelength_nm, wavelength_nm, epsilon_a.imag
    )
    epsilon_y = np.interp(
        sample_wavelength_nm, wavelength_nm, epsilon_b.real
    ) + 1j * np.interp(
        sample_wavelength_nm, wavelength_nm, epsilon_b.imag
    )
    epsilon_z = np.full_like(epsilon_x, 16.0)
    frequency_hz = C0 / (sample_wavelength_nm * 1e-9)
    material = fdtd.addmaterial("Sampled 3D data")
    fdtd.setmaterial(material, "name", TAIRTE4_MATERIAL)
    fdtd.setmaterial(TAIRTE4_MATERIAL, "anisotropy", 1)
    fdtd.setmaterial(TAIRTE4_MATERIAL, "max coefficients", 20)
    fdtd.setmaterial(
        TAIRTE4_MATERIAL,
        "sampled data",
        np.column_stack(
            (frequency_hz, epsilon_x, epsilon_y, epsilon_z)
        ),
    )


def configure_one_frequency(fdtd: object, name: str) -> None:
    add_single_frequency_monitor_settings(fdtd.getnamed(name))


def add_power_face(
    fdtd: object,
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
        if transverse != axis:
            monitor[f"{transverse} min"] = bounds[transverse][0]
            monitor[f"{transverse} max"] = bounds[transverse][1]
    add_single_frequency_monitor_settings(monitor)


def add_flux_box(
    fdtd: object,
    bounds: dict[str, tuple[float, float]],
) -> dict[str, float]:
    signs = {}
    for axis in "xyz":
        for side, position in zip(("min", "max"), bounds[axis]):
            name = f"device_flux_{axis}_{side}"
            add_power_face(fdtd, name, axis, position, bounds)
            signs[name] = -1.0 if side == "min" else 1.0
    return signs


def trapezoid_weights(values: np.ndarray) -> np.ndarray:
    coordinate = np.asarray(values, float).reshape(-1)
    weights = np.empty_like(coordinate)
    weights[0] = 0.5 * (coordinate[1] - coordinate[0])
    weights[-1] = 0.5 * (coordinate[-1] - coordinate[-2])
    weights[1:-1] = 0.5 * (coordinate[2:] - coordinate[:-2])
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


def extract_q(fdtd: object) -> dict[str, object]:
    fdtd.runanalysis(PABS_GROUP)
    x = np.asarray(fdtd.getdata(PABS_FIELD, "x", 1), float).reshape(-1)
    y = np.asarray(fdtd.getdata(PABS_FIELD, "y", 1), float).reshape(-1)
    z = np.asarray(fdtd.getdata(PABS_FIELD, "z", 1), float).reshape(-1)
    deltas = {
        axis: np.asarray(
            fdtd.getdata(PABS_FIELD, f"delta_{axis}", 1), float
        ).reshape(-1)
        for axis in "xyz"
    }
    target = np.stack(
        np.meshgrid(x, y, z, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    components = {}
    native_power = {}
    omega = 2.0 * np.pi * FREQUENCY_HZ
    for component in "xyz":
        electric = np.asarray(
            fdtd.getdata(PABS_FIELD, f"E{component}", 1)
        ).squeeze()
        epsilon = (
            np.asarray(
                fdtd.getdata(PABS_INDEX, f"index_{component}", 1)
            ).squeeze()
            ** 2
        )
        q_native = (
            0.5
            * EPS0
            * omega
            * np.imag(epsilon)
            * np.abs(electric) ** 2
        )
        native_coordinates = [x.copy(), y.copy(), z.copy()]
        axis_index = "xyz".index(component)
        native_coordinates[axis_index] += deltas[component]
        native_power[component] = integrate_xyz(
            q_native, *native_coordinates
        )
        components[component] = RegularGridInterpolator(
            tuple(native_coordinates),
            q_native,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )(target).reshape(x.size, y.size, z.size)
    q = sum(components.values())
    return {
        "x_m": x,
        "y_m": y,
        "z_m": z,
        "Qx_W_m3": components["x"],
        "Qy_W_m3": components["y"],
        "Qz_W_m3": components["z"],
        "Q_W_m3": q,
        "P_Q_W": integrate_xyz(q, x, y, z),
        "native_component_power_W": native_power,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    fsp_path = output / f"{args.case}.fsp"
    npz_path = output / f"{args.case}_raw.npz"
    result_path = output / "case_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_FINITE_DEVICE_TFSF_NOT_RUN",
        "case": args.case,
        "engine": "CPU",
        "solver_root": str(APPROVED_ROOT),
        "lumapi_path": str(APPROVED_API),
        "periodic": False,
        "six_boundaries": "PML",
        "thermal_run": False,
        "adjoint_run": False,
        "finite_difference_run": False,
        "geometry": {
            "FDTD_lateral_span_m": args.domain_um * 1e-6,
            "FDTD_z_bounds_m": list(FDTD_Z),
            "TFSF_lateral_span_m": args.tfsf_span_um * 1e-6,
            "TFSF_z_bounds_m": list(TFSF_Z),
            "finite_flake_bounds_m": {
                "x": [-0.5 * FLAKE_SPAN_M, 0.5 * FLAKE_SPAN_M],
                "y": [-0.5 * FLAKE_SPAN_M, 0.5 * FLAKE_SPAN_M],
                "z": list(FLAKE_Z),
            },
            "design_bounds_m": {
                "x": [-0.5 * DESIGN_SPAN_M, 0.5 * DESIGN_SPAN_M],
                "y": [-0.5 * DESIGN_SPAN_M, 0.5 * DESIGN_SPAN_M],
                "z": list(DESIGN_Z),
            },
            "bottom_SiO2_and_Si_extend_through_lateral_PML": True,
            "flake_extends_to_PML": False,
        },
        "settings": {
            "pml_layers": args.pml_layers,
            "mesh_type": "auto non-uniform",
            "mesh_refinement": "conformal variant 1",
            "mesh_accuracy": 5,
            "flake_dz_m": args.flake_dz_nm * 1e-9,
            "source_wavelength_m": [3.0e-6, 6.0e-6],
            "analysis_wavelength_m": WAVELENGTH_M,
            "target_incident_intensity_W_m2": 1.0,
            "pre_solve_amplitude_calibration": (
                V261_TFSF_AMPLITUDE_CALIBRATION
            ),
            "post_solve_field_or_Q_rescaling": False,
        },
    }
    fdtd = None
    total_started = time.monotonic()
    try:
        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        fdtd.eval("selectall; delete;")
        domain_m = args.domain_um * 1e-6
        tfsf_span_m = args.tfsf_span_um * 1e-6
        solver = fdtd.addfdtd()
        solver["dimension"] = "3D"
        solver["x span"] = domain_m
        solver["y span"] = domain_m
        solver["z min"] = FDTD_Z[0]
        solver["z max"] = FDTD_Z[1]
        for axis in "xyz":
            solver[f"{axis} min bc"] = "PML"
            solver[f"{axis} max bc"] = "PML"
        solver["pml layers"] = args.pml_layers
        solver["mesh type"] = "auto non-uniform"
        solver["mesh refinement"] = "conformal variant 1"
        solver["mesh accuracy"] = 5
        solver["min mesh step"] = 1.0e-9
        solver["simulation time"] = args.simulation_time_ps * 1e-12
        solver["auto shutoff min"] = 1.0e-8

        add_sio2_material(fdtd)
        material_span = domain_m + 2.0e-6
        add_rect(
            fdtd,
            "Si_substrate",
            x_span=material_span,
            y_span=material_span,
            z_min=FDTD_Z[0] - 0.5e-6,
            z_max=SI_Z_TOP_M,
            index=3.425,
        )
        add_rect(
            fdtd,
            "SiO2_spacer",
            x_span=material_span,
            y_span=material_span,
            z_min=SIO2_Z[0],
            z_max=SIO2_Z[1],
            material=SIO2_MATERIAL,
        )
        if args.case != "empty-stack":
            add_tairte4_material(fdtd)
            add_rect(
                fdtd,
                "TaIrTe4_flake",
                x_span=FLAKE_SPAN_M,
                y_span=FLAKE_SPAN_M,
                z_min=FLAKE_Z[0],
                z_max=FLAKE_Z[1],
                material=TAIRTE4_MATERIAL,
            )
        if args.case == "full-sio2":
            add_rect(
                fdtd,
                "design_full_SiO2",
                x_span=DESIGN_SPAN_M,
                y_span=DESIGN_SPAN_M,
                z_min=DESIGN_Z[0],
                z_max=DESIGN_Z[1],
                material=SIO2_MATERIAL,
            )
        mesh = fdtd.addmesh()
        mesh["name"] = "flake_mesh"
        mesh["x span"] = FLAKE_SPAN_M + 0.2e-6
        mesh["y span"] = FLAKE_SPAN_M + 0.2e-6
        mesh["z min"] = FLAKE_Z[0] - 50.0e-9
        mesh["z max"] = 50.0e-9
        mesh["override x mesh"] = 0
        mesh["override y mesh"] = 0
        mesh["override z mesh"] = 1
        mesh["dz"] = args.flake_dz_nm * 1e-9

        source = fdtd.addtfsf()
        source["name"] = SOURCE_NAME
        source["injection axis"] = "z"
        source["direction"] = "backward"
        source["polarization angle"] = 0.0
        source["angle theta"] = 0.0
        source["angle phi"] = 0.0
        source["x span"] = tfsf_span_m
        source["y span"] = tfsf_span_m
        source["z min"] = TFSF_Z[0]
        source["z max"] = TFSF_Z[1]
        source["override global source settings"] = True
        source["wavelength start"] = 3.0e-6
        source["wavelength stop"] = 6.0e-6
        # Lumerical's source amplitude is incident electric-field amplitude.
        # Set the analytic vacuum plane-wave value before solve.  v261 does
        # not expose sourceintensity for a newly created TFSF until the
        # project has been saved/meshed, so the API readback is gated after
        # the run rather than used for any post-solve scaling.
        source["amplitude"] = (
            np.sqrt(2.0 / (EPS0 * C0))
            * V261_TFSF_AMPLITUDE_CALIBRATION
        )

        pabs = fdtd.addobject("pabs_adv")
        pabs["name"] = PABS_GROUP
        pabs["x span"] = FLAKE_SPAN_M + 0.1e-6
        pabs["y span"] = FLAKE_SPAN_M + 0.1e-6
        pabs["z"] = -50.0e-9
        pabs["z span"] = 200.0e-9
        flux_bounds = {
            "x": (-2.15e-6, 2.15e-6),
            "y": (-2.15e-6, 2.15e-6),
            "z": (-0.25e-6, 0.75e-6),
        }
        flux_signs = add_flux_box(fdtd, flux_bounds)
        roi = fdtd.addpower()
        roi["name"] = "protected_roi_field"
        roi["monitor type"] = "2D Z-normal"
        roi["x min"] = -1.0e-6
        roi["x max"] = 1.0e-6
        roi["y min"] = -1.0e-6
        roi["y max"] = 1.0e-6
        roi["z"] = 50.0e-9
        add_single_frequency_monitor_settings(roi)

        fdtd.setresource("FDTD", 1, "active", 1)
        fdtd.setresource("FDTD", 1, "processes", "1")
        fdtd.setresource("FDTD", 1, "threads", str(args.threads))
        fdtd.setresource("FDTD", 2, "active", 0)
        result["resources_before_run"] = resource_snapshot(fdtd)
        result["source_readback_before_save"] = source_readback(
            fdtd, SOURCE_NAME
        )
        fdtd.save(str(fsp_path))
        run_started = time.monotonic()
        result["cpu_resource_name"] = run_on_cpu(fdtd)
        result["run_wall_s"] = time.monotonic() - run_started
        result["mesh_contract"] = optional_mesh_contract(fdtd)
        result["source_intensity_W_m2"] = scalar(
            fdtd.sourceintensity(FREQUENCY_HZ, 2, SOURCE_NAME),
            "post-run source intensity",
        )
        source_power = scalar(
            fdtd.sourcepower(FREQUENCY_HZ, 2, SOURCE_NAME),
            "source power",
        )
        face_power = {}
        net_outward = 0.0
        for name, sign in flux_signs.items():
            power = scalar(fdtd.transmission(name), name) * source_power
            face_power[name] = {
                "signed_axis_power_W": power,
                "outward_power_W": sign * power,
            }
            net_outward += sign * power
        p_six = -net_outward
        ex = np.asarray(
            fdtd.getdata("protected_roi_field", "Ex", 1)
        ).squeeze()
        ey = np.asarray(
            fdtd.getdata("protected_roi_field", "Ey", 1)
        ).squeeze()
        ez = np.asarray(
            fdtd.getdata("protected_roi_field", "Ez", 1)
        ).squeeze()
        e2 = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
        result["protected_roi_field"] = {
            "mean_E2_V2_m2": float(np.mean(e2)),
            "peak_to_peak_over_mean": float(
                (np.max(e2) - np.min(e2)) / np.mean(e2)
            ),
        }
        result["six_face"] = {
            "faces": face_power,
            "P_six_W": p_six,
        }
        arrays = {
            "roi_Ex_V_m": ex,
            "roi_Ey_V_m": ey,
            "roi_Ez_V_m": ez,
        }
        if args.case == "empty-stack":
            p_q = 0.0
            closure = abs(p_six) / max(abs(source_power), np.finfo(float).tiny)
            q_result = None
        else:
            q_result = extract_q(fdtd)
            p_q = float(q_result["P_Q_W"])
            closure = abs(p_q - p_six) / max(
                abs(p_six), np.finfo(float).tiny
            )
            arrays.update(
                {
                    key: value
                    for key, value in q_result.items()
                    if isinstance(value, np.ndarray)
                }
            )
            result["Q"] = {
                "P_Q_W": p_q,
                "native_component_power_W": q_result[
                    "native_component_power_W"
                ],
            }
        result["six_face"]["closure_relative_error"] = closure
        np.savez_compressed(npz_path, **arrays)
        result["artifacts"] = {
            "fsp": {
                "path": str(fsp_path),
                "bytes": fsp_path.stat().st_size,
                "sha256": sha256(fsp_path),
            },
            "npz": {
                "path": str(npz_path),
                "bytes": npz_path.stat().st_size,
                "sha256": sha256(npz_path),
            },
        }
        passed = bool(
            np.isfinite(p_six)
            and abs(result["source_intensity_W_m2"] - 1.0) < 1.0e-4
            and (
                closure < 5.0e-3
                if args.case != "empty-stack"
                else closure < 1.0e-4
            )
        )
        result["status"] = (
            "PASSED_FINITE_DEVICE_TFSF_FORWARD_GATE"
            if passed
            else "FAILED_FINITE_DEVICE_TFSF_FORWARD_GATE"
        )
        result["passed"] = passed
    except Exception as exc:
        result["status"] = "BLOCKED_FINITE_DEVICE_TFSF_EXECUTION"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["passed"] = False
    finally:
        result["total_wall_s"] = time.monotonic() - total_started
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(
            json.dumps(result, indent=2, default=json_default) + "\n"
        )
    print(json.dumps(result, indent=2, default=json_default), flush=True)
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
