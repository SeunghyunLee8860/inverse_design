#!/usr/bin/env python3
"""CPU-TFSF forward gate for the large-TaIrTe4 optical background."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .audit_v261_large_background_tfsf_geometry import (
    _mesh_coordinate,
    add_large_background_geometry,
)
from .large_background_contract import (
    Bounds3D,
    baseline_contract,
    infer_realized_pml_bounds,
)
from .native_yee_q import extract_native_yee_q, frequency_slice
from .probe_v261_cpu_tfsf_device import (
    FREQUENCY_HZ,
    PABS_FIELD,
    PABS_GROUP,
    PABS_INDEX,
    SIO2_MATERIAL,
    SOURCE_NAME,
    add_flux_box,
)
from .probe_v261_cpu_tfsf_roi import (
    add_single_frequency_monitor_settings,
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


WAVELENGTH_M = 4.0e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--case",
        required=True,
        choices=("source-off", "flat", "rho0", "rho1", "gray"),
    )
    parser.add_argument("--gray-rho", type=float, default=0.5)
    parser.add_argument("--domain-um", type=float, default=6.4)
    parser.add_argument("--tfsf-span-um", type=float, default=2.6)
    parser.add_argument("--pml-layers", type=int, default=24)
    parser.add_argument("--transverse-mesh-nm", type=float, default=50.0)
    parser.add_argument("--flake-dz-nm", type=float, default=5.0)
    parser.add_argument("--z-min-um", type=float, default=-3.2)
    parser.add_argument("--z-max-um", type=float, default=3.0)
    parser.add_argument("--tfsf-z-min-um", type=float, default=-1.8)
    parser.add_argument("--tfsf-z-max-um", type=float, default=1.5)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--simulation-time-ps", type=float, default=4.0)
    args = parser.parse_args()
    if not 0.0 <= args.gray_rho <= 1.0:
        parser.error("gray-rho must lie in [0,1]")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_design(fdtd: object, case: str, gray_rho: float) -> None:
    name = "finite_design_placeholder_air"
    if case == "flat" or case == "source-off":
        fdtd.select(name)
        fdtd.delete()
        return
    if case == "rho0":
        fdtd.setnamed(name, "index", 1.0)
        fdtd.setnamed(name, "name", "finite_design_rho0_air")
        return
    if case == "rho1":
        fdtd.setnamed(name, "material", SIO2_MATERIAL)
        fdtd.setnamed(name, "name", "finite_design_rho1_SiO2")
        return
    if case == "gray":
        epsilon = 1.0 + gray_rho * (1.38**2 - 1.0)
        fdtd.setnamed(name, "index", float(np.sqrt(epsilon)))
        fdtd.setnamed(name, "name", "finite_design_gray")
        return
    raise ValueError(case)


def add_monitors(fdtd: object, contract: object) -> dict[str, float]:
    pabs = fdtd.addobject("pabs_adv")
    pabs["name"] = PABS_GROUP
    pabs["x"] = 0.5 * sum(contract.omega_q.x)
    pabs["x span"] = contract.omega_q.x[1] - contract.omega_q.x[0]
    pabs["y"] = 0.5 * sum(contract.omega_q.y)
    pabs["y span"] = contract.omega_q.y[1] - contract.omega_q.y[0]
    pabs["z"] = 0.5 * sum(contract.omega_q.z)
    pabs["z span"] = contract.omega_q.z[1] - contract.omega_q.z[0]

    flux_signs = add_flux_box(
        fdtd,
        {
            "x": contract.omega_q.x,
            "y": contract.omega_q.y,
            "z": contract.omega_q.z,
        },
    )
    roi = fdtd.addpower()
    roi["name"] = "certificate_total_field_roi"
    roi["monitor type"] = "2D Z-normal"
    roi["x min"] = contract.design.x[0]
    roi["x max"] = contract.design.x[1]
    roi["y min"] = contract.design.y[0]
    roi["y max"] = contract.design.y[1]
    roi["z"] = 0.70e-6
    add_single_frequency_monitor_settings(roi)

    leakage = fdtd.addpower()
    leakage["name"] = "certificate_scattered_leakage"
    leakage["monitor type"] = "2D X-normal"
    leakage["x"] = contract.tfsf.x[1] + 0.10e-6
    leakage["y min"] = -1.0e-6
    leakage["y max"] = 1.0e-6
    leakage["z min"] = 0.65e-6
    leakage["z max"] = 0.75e-6
    add_single_frequency_monitor_settings(leakage)

    reference = fdtd.addindex()
    reference["name"] = "tfsf_reference_corner_index"
    reference["monitor type"] = "3D"
    corner = contract.tfsf.x[0] + 0.05e-6
    reference["x min"] = corner - 0.01e-6
    reference["x max"] = corner + 0.01e-6
    reference["y min"] = corner - 0.01e-6
    reference["y max"] = corner + 0.01e-6
    reference["z min"] = contract.tfsf.z[0]
    reference["z max"] = contract.tfsf.z[1]
    return flux_signs


def _field_cube(fdtd: object, monitor: str, component: str) -> np.ndarray:
    return np.asarray(fdtd.getdata(monitor, f"E{component}", 1)).squeeze()


def field_metrics(fdtd: object, monitor: str) -> dict[str, object]:
    components = {
        component: _field_cube(fdtd, monitor, component)
        for component in "xyz"
    }
    e2 = sum(np.abs(field) ** 2 for field in components.values())
    mean = float(np.mean(e2))
    return {
        "mean_E2_V2_m2": mean,
        "max_E2_V2_m2": float(np.max(e2)),
        "all_finite": bool(np.all(np.isfinite(e2))),
        "x_reflection_NRMSE": float(
            np.linalg.norm(e2 - np.flip(e2, axis=0))
            / max(np.linalg.norm(e2), np.finfo(float).tiny)
        ),
        "y_reflection_NRMSE": float(
            np.linalg.norm(e2 - np.flip(e2, axis=1))
            / max(np.linalg.norm(e2), np.finfo(float).tiny)
        ),
        "shape": list(e2.shape),
    }


def reference_stack(fdtd: object) -> dict[str, object]:
    monitor = "tfsf_reference_corner_index"
    z = np.asarray(fdtd.getdata(monitor, "z", 1), float).reshape(-1)
    f = np.asarray(fdtd.getdata(monitor, "f", 1), float).reshape(-1)
    frequency_index = int(np.argmin(np.abs(f - FREQUENCY_HZ)))
    result: dict[str, object] = {
        "z_min_m": float(np.min(z)),
        "z_max_m": float(np.max(z)),
        "z_count": int(z.size),
        "analysis_frequency_index_zero_based": frequency_index,
    }
    for component in "xyz":
        raw = np.asarray(
            fdtd.getdata(monitor, f"index_{component}", 1)
        )
        spatial_shape = raw.shape[:3]
        cube = frequency_slice(
            raw,
            spatial_shape,
            frequency_index,
            f.size,
            f"reference index_{component}",
        )
        profile = np.median(cube, axis=(0, 1))
        result[f"epsilon_{component}_profile_real"] = np.real(
            profile**2
        ).tolist()
        result[f"epsilon_{component}_profile_imag"] = np.imag(
            profile**2
        ).tolist()
    result["z_m"] = z.tolist()
    return result


def save_native_npz(path: Path, q: dict[str, object]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for axis in "xyz":
        arrays[f"base_{axis}_m"] = q["base_coordinates"][axis]
        arrays[f"delta_{axis}_m"] = q["delta_coordinates"][axis]
    for component in "xyz":
        arrays[f"Q{component}_W_m3"] = q["Q_components"][component]
        for axis in "xyz":
            arrays[f"Q{component}_{axis}_m"] = q[
                "native_coordinates"
            ][component][axis]
    np.savez_compressed(path, **arrays)


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    project_path = output / f"large_background_{args.case}.fsp"
    npz_path = output / f"large_background_{args.case}_native_q.npz"
    result_path = output / "case_result.json"
    contract = baseline_contract(
        lateral_domain_um=args.domain_um,
        tfsf_span_um=args.tfsf_span_um,
        pml_layers=args.pml_layers,
        transverse_mesh_nm=args.transverse_mesh_nm,
        flake_dz_nm=args.flake_dz_nm,
        z_min_um=args.z_min_um,
        z_max_um=args.z_max_um,
        tfsf_z_min_um=args.tfsf_z_min_um,
        tfsf_z_max_um=args.tfsf_z_max_um,
    )
    result: dict[str, object] = {
        "status": "BLOCKED_LARGE_BACKGROUND_FORWARD_NOT_RUN",
        "case": args.case,
        "gray_rho": args.gray_rho if args.case == "gray" else None,
        "solver_root": str(APPROVED_ROOT),
        "lumapi_path": str(APPROVED_API),
        "engine": "CPU_TFSF_FORWARD",
        "periodic_or_bloch": False,
        "thermal_run": False,
        "adjoint_run": False,
        "finite_difference_run": False,
        "geometry": contract.geometry_audit_input(),
        "source_normalization": {
            "target_incident_intensity_W_m2": 1.0,
            "amplitude_definition": "sqrt(2/(epsilon0*c0))",
            "source_intensity_relative_error_limit": 5.0e-3,
            "post_solve_field_or_Q_rescaling": False,
        },
    }
    fdtd = None
    started = time.monotonic()
    try:
        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        add_large_background_geometry(fdtd, contract)
        configure_design(fdtd, args.case, args.gray_rho)
        simulation_time_ps = (
            min(args.simulation_time_ps, 0.1)
            if args.case == "source-off"
            else args.simulation_time_ps
        )
        fdtd.setnamed("FDTD", "simulation time", simulation_time_ps * 1e-12)
        result["simulation_time_ps"] = simulation_time_ps
        flux_signs = add_monitors(fdtd, contract)
        if args.case == "source-off":
            fdtd.setnamed(SOURCE_NAME, "enabled", False)
        fdtd.setresource("FDTD", 1, "active", 1)
        fdtd.setresource("FDTD", 1, "processes", "1")
        fdtd.setresource("FDTD", 1, "threads", str(args.threads))
        fdtd.setresource("FDTD", 2, "active", 0)
        result["resources_before_run"] = resource_snapshot(fdtd)
        result["source_readback_before_run"] = source_readback(
            fdtd, SOURCE_NAME
        )
        fdtd.save(str(project_path))
        coordinates = {
            axis: _mesh_coordinate(fdtd, axis)
            for axis in "xyz"
        }
        result["realized_mesh"] = infer_realized_pml_bounds(
            coordinates, contract.fdtd_outer, contract.pml_layers
        )
        realized_inner = Bounds3D(
            **{
                axis: tuple(
                    result["realized_mesh"][axis][
                        "pml_inner_from_layer_count_m"
                    ]
                )
                for axis in "xyz"
            }
        )
        if not realized_inner.contains(contract.tfsf, strict=True):
            raise RuntimeError("TFSF enters realized PML; refusing solver run")
        run_started = time.monotonic()
        result["cpu_resource_name"] = run_on_cpu(fdtd)
        result["run_wall_s"] = time.monotonic() - run_started
        result["resources_after_run"] = resource_snapshot(fdtd)
        result["roi_total_field"] = field_metrics(
            fdtd, "certificate_total_field_roi"
        )
        result["outside_tfsf_field"] = field_metrics(
            fdtd, "certificate_scattered_leakage"
        )
        result["outside_to_inside_mean_E2_ratio"] = (
            result["outside_tfsf_field"]["mean_E2_V2_m2"]
            / max(
                result["roi_total_field"]["mean_E2_V2_m2"],
                np.finfo(float).tiny,
            )
        )
        if args.case == "source-off":
            result["source_off_max_E2_V2_m2"] = max(
                result["roi_total_field"]["max_E2_V2_m2"],
                result["outside_tfsf_field"]["max_E2_V2_m2"],
            )
            result["passed"] = bool(
                result["source_off_max_E2_V2_m2"] < 1.0e-20
            )
            result["status"] = (
                "VALIDATED_LARGE_BACKGROUND_SOURCE_OFF_CONTROL"
                if result["passed"]
                else "FAILED_LARGE_BACKGROUND_SOURCE_OFF_CONTROL"
            )
        else:
            result["source_intensity_W_m2"] = scalar(
                fdtd.sourceintensity(FREQUENCY_HZ, 2, SOURCE_NAME),
                "source intensity",
            )
            result["source_intensity_relative_error"] = abs(
                result["source_intensity_W_m2"] - 1.0
            )
            source_power = scalar(
                fdtd.sourcepower(FREQUENCY_HZ, 2, SOURCE_NAME),
                "source power",
            )
            face_power: dict[str, object] = {}
            net_outward = 0.0
            for name, sign in flux_signs.items():
                signed_axis_power = (
                    scalar(fdtd.transmission(name), name) * source_power
                )
                outward_power = sign * signed_axis_power
                face_power[name] = {
                    "signed_axis_power_W": signed_axis_power,
                    "outward_power_W": outward_power,
                }
                net_outward += outward_power
            p_six = -net_outward
            fdtd.runanalysis(PABS_GROUP)
            q = extract_native_yee_q(
                fdtd,
                field_monitor=PABS_FIELD,
                index_monitor=PABS_INDEX,
                wavelength_m=WAVELENGTH_M,
            )
            p_q = float(q["P_Q_W"])
            closure = abs(p_q - p_six) / max(
                abs(p_six), np.finfo(float).tiny
            )
            result["six_face"] = {
                "bounds_m": contract.omega_q.to_dict(),
                "faces": face_power,
                "P_six_W": p_six,
            }
            result["Q"] = {
                "integration_bounds_m": {
                    "x": list(contract.omega_q.x),
                    "y": list(contract.omega_q.y),
                    "z": list(contract.omega_q.z),
                },
                "P_Q_W": p_q,
                "component_power_W": q["component_power_W"],
                "component_hotspots": q["component_hotspots"],
                "raw_array_shapes": q["raw_array_shapes"],
                "frequency_count": q["frequency_count"],
                "frequency_index_zero_based": q[
                    "frequency_index_zero_based"
                ],
                "operations_forbidden_and_absent": q[
                    "operations_forbidden_and_absent"
                ],
            }
            result["six_face"]["closure_relative_error"] = closure
            result["reference_corner_stack"] = reference_stack(fdtd)
            save_native_npz(npz_path, q)
            result["passed"] = bool(
                np.isfinite(p_q)
                and np.isfinite(p_six)
                and result["source_intensity_relative_error"] < 5.0e-3
                and closure < 5.0e-3
                and result["roi_total_field"]["all_finite"]
            )
            result["status"] = (
                "VALIDATED_LARGE_BACKGROUND_FORWARD_Q_CLOSURE"
                if result["passed"]
                else "FAILED_LARGE_BACKGROUND_FORWARD_Q_CLOSURE"
            )
        fdtd.save(str(project_path))
        result["artifacts"] = {
            "fsp": {
                "path": str(project_path),
                "byte_size": project_path.stat().st_size,
                "sha256": sha256(project_path),
            }
        }
        if npz_path.is_file():
            result["artifacts"]["npz"] = {
                "path": str(npz_path),
                "byte_size": npz_path.stat().st_size,
                "sha256": sha256(npz_path),
            }
    except Exception as exc:
        result["status"] = "BLOCKED_LARGE_BACKGROUND_FORWARD_EXECUTION"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["passed"] = False
    finally:
        result["total_wall_s"] = time.monotonic() - started
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
