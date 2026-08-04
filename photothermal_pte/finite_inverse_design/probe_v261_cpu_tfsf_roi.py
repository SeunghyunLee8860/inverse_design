#!/usr/bin/env python3
"""CPU TFSF source-integrity probe for the protected central 2 um ROI.

This is deliberately a source gate, not a device, thermal, AD, or FD run.
The finite TFSF box remains strictly inside the six-PML FDTD domain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np

from .probe_v261_gpu_plane_wave_roi import (
    APPROVED_API,
    APPROVED_ROOT,
    ROI_MAX_M,
    ROI_MIN_M,
    WAVELENGTH_M,
    json_default,
    load_lumapi,
    phase_deviation_rad,
    resource_snapshot,
    scalar,
    source_readback,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domain-um", type=float, default=4.0)
    parser.add_argument("--tfsf-span-um", type=float, default=2.6)
    parser.add_argument("--tfsf-z-span-um", type=float, default=2.0)
    parser.add_argument("--z-min-um", type=float, default=-1.5)
    parser.add_argument("--z-max-um", type=float, default=1.5)
    parser.add_argument("--monitor-z-um", type=float, default=0.5)
    parser.add_argument("--pml-layers", type=int, default=24)
    parser.add_argument("--mesh-accuracy", type=int, default=5)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--simulation-time-ps", type=float, default=1.0)
    parser.add_argument("--sampling-dxy-nm", type=float, default=100.0)
    args = parser.parse_args()
    if args.domain_um <= 2.0:
        parser.error("domain must be larger than the protected 2 um ROI")
    if not 2.0 < args.tfsf_span_um < args.domain_um:
        parser.error("TFSF span must contain the ROI and stay inside the domain")
    if not 0.0 < args.tfsf_z_span_um < args.z_max_um - args.z_min_um:
        parser.error("TFSF z span must stay inside the z domain")
    if (
        abs(args.monitor_z_um) >= 0.5 * args.tfsf_z_span_um
        or not args.z_min_um < args.monitor_z_um < args.z_max_um
    ):
        parser.error("ROI monitor must be inside the total-field region")
    if args.pml_layers < 8 or args.mesh_accuracy < 1 or args.threads < 1:
        parser.error("invalid numerical setting")
    return args


def optional_mesh_contract(fdtd: object) -> dict[str, object]:
    result: dict[str, object] = {}
    try:
        mesh = fdtd.getresult("FDTD", "mesh")
        counts = {}
        for axis in "xyz":
            coordinate = np.asarray(mesh[axis], float).reshape(-1)
            counts[axis] = int(coordinate.size)
            result[f"{axis}_min_m"] = float(np.min(coordinate))
            result[f"{axis}_max_m"] = float(np.max(coordinate))
        result["node_counts"] = counts
        result["estimated_yee_cells"] = int(
            max(counts["x"] - 1, 0)
            * max(counts["y"] - 1, 0)
            * max(counts["z"] - 1, 0)
        )
    except Exception as exc:
        result["read_error"] = f"{type(exc).__name__}: {exc}"
    return result


def add_single_frequency_monitor_settings(monitor: object) -> None:
    monitor["override global monitor settings"] = True
    monitor["use source limits"] = False
    monitor["use wavelength spacing"] = True
    monitor["wavelength center"] = WAVELENGTH_M
    monitor["wavelength span"] = 0.0
    monitor["frequency points"] = 1


def add_flux_box(fdtd: object) -> dict[str, int]:
    """Add a source-free closed box strictly inside the TFSF total-field zone."""
    half_xy = 0.8e-6
    half_z = 0.35e-6
    definitions = {
        "flux_x_min": ("2D X-normal", "x", -half_xy),
        "flux_x_max": ("2D X-normal", "x", half_xy),
        "flux_y_min": ("2D Y-normal", "y", -half_xy),
        "flux_y_max": ("2D Y-normal", "y", half_xy),
        "flux_z_min": ("2D Z-normal", "z", -half_z),
        "flux_z_max": ("2D Z-normal", "z", half_z),
    }
    outward_sign = {
        "flux_x_min": -1,
        "flux_x_max": 1,
        "flux_y_min": -1,
        "flux_y_max": 1,
        "flux_z_min": -1,
        "flux_z_max": 1,
    }
    for name, (monitor_type, fixed_axis, fixed_value) in definitions.items():
        monitor = fdtd.addpower()
        monitor["name"] = name
        monitor["monitor type"] = monitor_type
        monitor[fixed_axis] = fixed_value
        if fixed_axis != "x":
            monitor["x min"] = -half_xy
            monitor["x max"] = half_xy
        if fixed_axis != "y":
            monitor["y min"] = -half_xy
            monitor["y max"] = half_xy
        if fixed_axis != "z":
            monitor["z min"] = -half_z
            monitor["z max"] = half_z
        add_single_frequency_monitor_settings(monitor)
    return outward_sign


def run_on_cpu(fdtd: object) -> str:
    errors = []
    for resource_name in ("Local Host", "localhost", "Local Computer"):
        try:
            fdtd.run("FDTD", "CPU", resource_name)
            return resource_name
        except Exception as exc:
            errors.append(f"{resource_name}: {exc}")
    raise RuntimeError("CPU session run failed: " + " | ".join(errors))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "probe_result.json"
    project_path = output / "cpu_tfsf_roi_probe.fsp"
    result: dict[str, object] = {
        "status": "BLOCKED_CPU_TFSF_NOT_RUN",
        "purpose": "empty-air TFSF source gate only",
        "solver_root": str(APPROVED_ROOT),
        "lumapi_path": str(APPROVED_API),
        "engine": "CPU",
        "periodic_boundary": False,
        "all_six_boundaries": "PML",
        "device_geometry_present": False,
        "thermal_run": False,
        "adjoint_run": False,
        "finite_difference_run": False,
        "roi_bounds_m": {
            "x": [ROI_MIN_M, ROI_MAX_M],
            "y": [ROI_MIN_M, ROI_MAX_M],
        },
        "domain_um": args.domain_um,
        "tfsf_span_um": args.tfsf_span_um,
        "tfsf_z_span_um": args.tfsf_z_span_um,
        "z_bounds_um": [args.z_min_um, args.z_max_um],
        "monitor_z_um": args.monitor_z_um,
        "pml_layers": args.pml_layers,
        "mesh_accuracy": args.mesh_accuracy,
        "threads": args.threads,
        "simulation_time_ps": args.simulation_time_ps,
        "sampling_dxy_nm": args.sampling_dxy_nm,
    }
    fdtd = None
    total_started = time.monotonic()
    try:
        lumapi = load_lumapi()
        session_started = time.monotonic()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        result["session_startup_wall_s"] = time.monotonic() - session_started
        fdtd.switchtolayout()
        # Some v261 sessions open with a default solver object while others
        # open an empty model.  Start from an explicitly empty CAD tree so the
        # named-object readbacks below can never bind to a stale/default FDTD.
        fdtd.eval("selectall; delete;")

        domain_m = args.domain_um * 1e-6
        tfsf_span_m = args.tfsf_span_um * 1e-6
        tfsf_z_span_m = args.tfsf_z_span_um * 1e-6
        monitor_z_m = args.monitor_z_um * 1e-6

        solver = fdtd.addfdtd()
        solver["dimension"] = "3D"
        solver["x span"] = domain_m
        solver["y span"] = domain_m
        solver["z min"] = args.z_min_um * 1e-6
        solver["z max"] = args.z_max_um * 1e-6
        for axis in "xyz":
            solver[f"{axis} min bc"] = "PML"
            solver[f"{axis} max bc"] = "PML"
        solver["pml layers"] = args.pml_layers
        solver["mesh type"] = "auto non-uniform"
        solver["mesh accuracy"] = args.mesh_accuracy
        solver["simulation time"] = args.simulation_time_ps * 1e-12
        solver["auto shutoff min"] = 1e-7

        source_name = "cpu_tfsf_probe"
        source = fdtd.addtfsf()
        source["name"] = source_name
        source["injection axis"] = "z"
        source["direction"] = "backward"
        source["polarization angle"] = 0.0
        source["angle theta"] = 0.0
        source["angle phi"] = 0.0
        source["x span"] = tfsf_span_m
        source["y span"] = tfsf_span_m
        source["z"] = 0.0
        source["z span"] = tfsf_z_span_m
        source["override global source settings"] = True
        source["wavelength start"] = 3.0e-6
        source["wavelength stop"] = 6.0e-6

        sampling_mesh = fdtd.addmesh()
        sampling_mesh["name"] = "roi_sampling_mesh"
        sampling_mesh["x min"] = ROI_MIN_M
        sampling_mesh["x max"] = ROI_MAX_M
        sampling_mesh["y min"] = ROI_MIN_M
        sampling_mesh["y max"] = ROI_MAX_M
        sampling_mesh["z min"] = monitor_z_m - 0.1e-6
        sampling_mesh["z max"] = monitor_z_m + 0.1e-6
        sampling_mesh["override x mesh"] = 1
        sampling_mesh["override y mesh"] = 1
        sampling_mesh["override z mesh"] = 0
        sampling_mesh["dx"] = args.sampling_dxy_nm * 1e-9
        sampling_mesh["dy"] = args.sampling_dxy_nm * 1e-9

        roi_monitor = fdtd.addpower()
        roi_monitor["name"] = "roi_xy"
        roi_monitor["monitor type"] = "2D Z-normal"
        roi_monitor["x min"] = ROI_MIN_M
        roi_monitor["x max"] = ROI_MAX_M
        roi_monitor["y min"] = ROI_MIN_M
        roi_monitor["y max"] = ROI_MAX_M
        roi_monitor["z"] = monitor_z_m
        add_single_frequency_monitor_settings(roi_monitor)
        outward_sign = add_flux_box(fdtd)

        fdtd.setresource("FDTD", 1, "active", 1)
        fdtd.setresource("FDTD", 1, "processes", "1")
        fdtd.setresource("FDTD", 1, "threads", str(args.threads))
        fdtd.setresource("FDTD", 2, "active", 0)
        result["resources_before_run"] = resource_snapshot(fdtd)
        cpu = result["resources_before_run"]["1"]
        gpu = result["resources_before_run"]["2"]
        if cpu["active"].strip() != "1" or cpu["device type"].strip().upper() != "CPU":
            raise RuntimeError(f"CPU resource contract failed: {cpu}")
        if gpu["active"].strip() != "0":
            raise RuntimeError(f"GPU resource must be inactive: {gpu}")

        result["source_property_readback_before_save"] = source_readback(
            fdtd, source_name
        )
        save_started = time.monotonic()
        fdtd.save(str(project_path))
        result["initial_save_wall_s"] = time.monotonic() - save_started
        result["source_property_readback_after_save"] = source_readback(
            fdtd, source_name
        )

        run_started = time.monotonic()
        result["cpu_session_resource_name"] = run_on_cpu(fdtd)
        result["fdtd_run_wall_s"] = time.monotonic() - run_started
        result["resources_after_run"] = resource_snapshot(fdtd)
        result["mesh_contract"] = optional_mesh_contract(fdtd)

        post_started = time.monotonic()
        ex = np.asarray(fdtd.getdata("roi_xy", "Ex", 1)).squeeze()
        ey = np.asarray(fdtd.getdata("roi_xy", "Ey", 1)).squeeze()
        ez = np.asarray(fdtd.getdata("roi_xy", "Ez", 1)).squeeze()
        x_m = np.asarray(fdtd.getdata("roi_xy", "x", 1), float).reshape(-1)
        y_m = np.asarray(fdtd.getdata("roi_xy", "y", 1), float).reshape(-1)
        x_inside = np.flatnonzero(
            (x_m >= ROI_MIN_M - 1e-15) & (x_m <= ROI_MAX_M + 1e-15)
        )
        y_inside = np.flatnonzero(
            (y_m >= ROI_MIN_M - 1e-15) & (y_m <= ROI_MAX_M + 1e-15)
        )
        if x_inside.size < 2 or y_inside.size < 2:
            raise RuntimeError("ROI monitor does not contain enough samples")
        ex = ex[np.ix_(x_inside, y_inside)]
        ey = ey[np.ix_(x_inside, y_inside)]
        ez = ez[np.ix_(x_inside, y_inside)]
        x_roi_m = x_m[x_inside]
        y_roi_m = y_m[y_inside]
        e2 = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
        mean_e2 = float(np.mean(e2))
        phase = phase_deviation_rad(ex)
        frequency_hz = 299792458.0 / WAVELENGTH_M
        source_intensity = scalar(
            fdtd.sourceintensity(frequency_hz, 2, source_name),
            "source intensity",
        )
        flux = {
            name: scalar(fdtd.transmission(name), f"{name} transmission")
            for name in outward_sign
        }
        outward_flux = {
            name: outward_sign[name] * value for name, value in flux.items()
        }
        net_outward = float(sum(outward_flux.values()))
        flux_scale = max(
            sum(value for value in outward_flux.values() if value > 0.0),
            -sum(value for value in outward_flux.values() if value < 0.0),
            np.finfo(float).tiny,
        )
        metrics = {
            "x_sample_min_m": float(x_roi_m[0]),
            "x_sample_max_m": float(x_roi_m[-1]),
            "y_sample_min_m": float(y_roi_m[0]),
            "y_sample_max_m": float(y_roi_m[-1]),
            "mean_E2": mean_e2,
            "mean_E2_relative_error_from_unit_source": abs(mean_e2 - 1.0),
            "min_E2": float(np.min(e2)),
            "max_E2": float(np.max(e2)),
            "intensity_relative_rms": float(np.std(e2))
            / max(mean_e2, np.finfo(float).tiny),
            "intensity_relative_peak_to_peak": float(np.ptp(e2))
            / max(mean_e2, np.finfo(float).tiny),
            "phase_rms_deg": float(np.sqrt(np.mean(phase**2)) * 180.0 / np.pi),
            "phase_max_abs_deg": float(np.max(np.abs(phase)) * 180.0 / np.pi),
            "Ey_to_Ex_L2": float(np.linalg.norm(ey))
            / max(float(np.linalg.norm(ex)), np.finfo(float).tiny),
            "Ez_to_Ex_L2": float(np.linalg.norm(ez))
            / max(float(np.linalg.norm(ex)), np.finfo(float).tiny),
            "source_intensity_native_W_m2": source_intensity,
            "all_fields_finite": bool(
                np.all(np.isfinite(ex))
                and np.all(np.isfinite(ey))
                and np.all(np.isfinite(ez))
            ),
        }
        result["roi_metrics"] = metrics
        result["closed_flux_box"] = {
            "raw_monitor_transmission": flux,
            "outward_signed_transmission": outward_flux,
            "net_outward_transmission": net_outward,
            "relative_energy_closure_error": abs(net_outward) / flux_scale,
            "box_bounds_um": {
                "x": [-0.8, 0.8],
                "y": [-0.8, 0.8],
                "z": [-0.35, 0.35],
            },
        }
        result["acceptance"] = {
            "cpu_resource_active": (
                result["resources_after_run"]["1"]["active"].strip() == "1"
            ),
            "gpu_resource_inactive": (
                result["resources_after_run"]["2"]["active"].strip() == "0"
            ),
            "exact_roi_monitor_bounds": (
                abs(x_roi_m[0] - ROI_MIN_M) <= 0.051e-6
                and abs(x_roi_m[-1] - ROI_MAX_M) <= 0.051e-6
                and abs(y_roi_m[0] - ROI_MIN_M) <= 0.051e-6
                and abs(y_roi_m[-1] - ROI_MAX_M) <= 0.051e-6
            ),
            "field_finite": metrics["all_fields_finite"],
            "mean_intensity_error_lt_0p5_percent": (
                metrics["mean_E2_relative_error_from_unit_source"] < 0.005
            ),
            "intensity_rms_lt_0p5_percent": (
                metrics["intensity_relative_rms"] < 0.005
            ),
            "intensity_peak_to_peak_lt_1_percent": (
                metrics["intensity_relative_peak_to_peak"] < 0.01
            ),
            "phase_max_lt_1_degree": metrics["phase_max_abs_deg"] < 1.0,
            "cross_polarized_components_lt_0p1_percent": (
                max(metrics["Ey_to_Ex_L2"], metrics["Ez_to_Ex_L2"]) < 0.001
            ),
            "closed_box_energy_error_lt_1_percent": (
                result["closed_flux_box"]["relative_energy_closure_error"]
                < 0.01
            ),
        }
        passed = all(bool(value) for value in result["acceptance"].values())
        result["status"] = (
            "VALIDATED_CPU_TFSF_4UM_DOMAIN_2UM_ROI_SOURCE_GATE"
            if passed
            else "FAILED_CPU_TFSF_4UM_DOMAIN_2UM_ROI_SOURCE_GATE"
        )
        result["postprocess_wall_s"] = time.monotonic() - post_started
        save_started = time.monotonic()
        fdtd.save(str(project_path))
        result["final_save_wall_s"] = time.monotonic() - save_started
        result["fsp"] = {
            "path": str(project_path),
            "byte_size": project_path.stat().st_size,
            "sha256": sha256(project_path),
        }
        return_code = 0 if passed else 2
    except Exception as exc:
        result["status"] = "BLOCKED_CPU_TFSF_EXECUTION"
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        return_code = 3
    finally:
        result["total_wall_s"] = time.monotonic() - total_started
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=json_default)
            + "\n",
            encoding="utf-8",
        )
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
