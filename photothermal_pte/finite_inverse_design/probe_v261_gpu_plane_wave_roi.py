#!/usr/bin/env python3
"""GPU-only plane-wave integrity probe for the exact central 2 um ROI."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


APPROVED_ROOT = Path("/home/seunghyun/lumerical_r12/opt/lumerical/v261")
APPROVED_API = APPROVED_ROOT / "api" / "python" / "lumapi.py"
WAVELENGTH_M = 4.0e-6
ROI_MIN_M = -1.0e-6
ROI_MAX_M = 1.0e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domain-um", type=float, required=True)
    parser.add_argument("--pml-layers", type=int, choices=(16, 24, 32), default=16)
    parser.add_argument(
        "--pml-profile",
        choices=("standard", "steep angle"),
        default="standard",
    )
    parser.add_argument(
        "--source-kind",
        choices=("plane-wave", "imported-flat-top", "tfsf"),
        default="plane-wave",
    )
    parser.add_argument(
        "--plane-wave-type",
        choices=("bloch-periodic", "diffracting"),
        default="bloch-periodic",
        help="Explicit Lumerical source type; never rely on the GUI default.",
    )
    parser.add_argument(
        "--source-span-um",
        type=float,
        default=None,
        help="Finite aperture span. Default: lateral FDTD span + 1 um.",
    )
    parser.add_argument("--source-z-um", type=float, default=0.8)
    parser.add_argument("--monitor-z-um", type=float, default=0.65)
    parser.add_argument("--z-min-um", type=float, default=-1.5)
    parser.add_argument("--z-max-um", type=float, default=1.5)
    parser.add_argument(
        "--flat-plateau-half-span-um",
        type=float,
        default=1.25,
        help="Imported flat-top field is exactly constant inside this half-span.",
    )
    parser.add_argument(
        "--flat-zero-half-span-um",
        type=float,
        default=2.5,
        help="Raised-cosine taper reaches exactly zero at this half-span.",
    )
    parser.add_argument("--flat-source-samples", type=int, default=81)
    parser.add_argument(
        "--flat-h-model",
        choices=("automatic", "local-plane-wave"),
        default="automatic",
        help="Let Lumerical derive H, or import the local plane-impedance H.",
    )
    parser.add_argument("--gpu-device", default="GPU 1")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--simulation-time-ps", type=float, default=1.0)
    parser.add_argument(
        "--sampling-mesh-span-um",
        type=float,
        default=None,
        help="Lateral span of the 100 nm sampling mesh; default is full domain.",
    )
    args = parser.parse_args()
    if args.domain_um <= 2.0:
        parser.error("domain span must exceed the exact 2 um ROI")
    if args.simulation_time_ps <= 0.0:
        parser.error("simulation-time-ps must be positive")
    if (
        args.sampling_mesh_span_um is not None
        and args.sampling_mesh_span_um < 2.0
    ):
        parser.error("sampling mesh must contain the exact 2 um ROI")
    if args.source_span_um is not None and args.source_span_um <= 2.0:
        parser.error("source aperture span must exceed the exact 2 um ROI")
    if args.source_kind == "imported-flat-top":
        if args.flat_plateau_half_span_um < 1.0:
            parser.error("flat plateau must contain the exact [-1,1] um ROI")
        if args.flat_zero_half_span_um <= args.flat_plateau_half_span_um:
            parser.error("flat zero half-span must exceed the plateau")
        if args.flat_zero_half_span_um >= 0.5 * args.domain_um:
            parser.error("flat source must reach zero before lateral PML")
        if args.flat_source_samples < 9 or args.flat_source_samples % 2 == 0:
            parser.error("flat-source-samples must be an odd integer >= 9")
    if args.z_min_um >= args.z_max_um:
        parser.error("z-min-um must be smaller than z-max-um")
    if not args.z_min_um < args.source_z_um < args.z_max_um:
        parser.error("source z must be inside the FDTD z bounds")
    if not args.z_min_um < args.monitor_z_um < args.z_max_um:
        parser.error("monitor z must be inside the FDTD z bounds")
    if abs(args.monitor_z_um - args.source_z_um) < 1.0e-6:
        parser.error("monitor and source planes must differ")
    return args


def load_lumapi() -> object:
    if not APPROVED_API.is_file():
        raise FileNotFoundError(f"approved lumapi missing: {APPROVED_API}")
    os.environ["VC_LUMERICAL_ROOT"] = str(APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(APPROVED_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(APPROVED_API.parent)
    os.environ["PATH"] = f"{APPROVED_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
    if str(APPROVED_API.parent) not in sys.path:
        sys.path.insert(0, str(APPROVED_API.parent))
    import lumapi

    actual = Path(lumapi.__file__).resolve()
    if actual != APPROVED_API.resolve():
        raise RuntimeError(f"wrong lumapi loaded: {actual}")
    return lumapi


def resource_snapshot(fdtd: object) -> dict[str, dict[str, str]]:
    properties = (
        "active",
        "device type",
        "processes",
        "threads",
        "solver extra command line options",
    )
    result: dict[str, dict[str, str]] = {}
    for index in (1, 2):
        result[str(index)] = {}
        for prop in properties:
            try:
                result[str(index)][prop] = str(
                    fdtd.getresource("FDTD", index, prop)
                )
            except Exception as exc:
                result[str(index)][prop] = f"{type(exc).__name__}: {exc}"
    return result


def scalar(value: object, label: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise RuntimeError(f"{label} is not scalar: {array.shape}")
    result = float(np.real(array[0]))
    if not np.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def phase_deviation_rad(field: np.ndarray) -> np.ndarray:
    reference = np.mean(field)
    if abs(reference) <= np.finfo(float).tiny:
        raise RuntimeError("mean Ex is zero")
    return np.angle(field / reference)


def raised_cosine_flat_top(
    coordinates_m: np.ndarray,
    plateau_half_span_m: float,
    zero_half_span_m: float,
) -> np.ndarray:
    """Unit plateau with a C1 half-cosine taper to exact zero."""
    radius = np.abs(np.asarray(coordinates_m, float))
    result = np.ones_like(radius)
    outside = radius >= zero_half_span_m
    taper = (radius > plateau_half_span_m) & ~outside
    phase = (
        radius[taper] - plateau_half_span_m
    ) / (zero_half_span_m - plateau_half_span_m)
    result[taper] = 0.5 * (1.0 + np.cos(np.pi * phase))
    result[outside] = 0.0
    return result


def optional_readback(fdtd: object, name: str, prop: str) -> object:
    try:
        value = fdtd.getnamed(name, prop)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    array = np.asarray(value)
    if array.size == 1:
        item = array.reshape(-1)[0]
        return item.item() if isinstance(item, np.generic) else item
    return array.tolist()


def source_readback(fdtd: object, name: str) -> dict[str, object]:
    return {
        prop: optional_readback(fdtd, name, prop)
        for prop in (
            "source shape",
            "plane wave type",
            "injection axis",
            "direction",
            "x span",
            "y span",
            "z",
            "z0",
            "wavelength start",
            "wavelength stop",
        )
    }


def run_on_gpu(fdtd: object) -> str:
    errors = []
    for resource_name in (
        "Local GPU",
        "local GPU",
        "Local Host",
        "localhost",
        "Local Computer",
    ):
        try:
            fdtd.run("FDTD", "GPU", resource_name)
            return resource_name
        except Exception as exc:
            errors.append(f"{resource_name}: {exc}")
    raise RuntimeError("GPU session run failed: " + " | ".join(errors))


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "probe_result.json"
    domain_m = args.domain_um * 1.0e-6
    if args.source_kind == "imported-flat-top":
        source_span_m = 2.0 * args.flat_zero_half_span_um * 1.0e-6
    else:
        source_span_m = (
            args.source_span_um * 1.0e-6
            if args.source_span_um is not None
            else domain_m + 1.0e-6
        )
    source_z_m = args.source_z_um * 1.0e-6
    monitor_z_m = args.monitor_z_um * 1.0e-6
    z_min_m = args.z_min_um * 1.0e-6
    z_max_m = args.z_max_um * 1.0e-6
    sampling_mesh_span_m = (
        args.sampling_mesh_span_um * 1.0e-6
        if args.sampling_mesh_span_um is not None
        else domain_m
    )
    result: dict[str, object] = {
        "status": "BLOCKED_GPU_PLANE_WAVE_NOT_RUN",
        "roi_bounds_m": {
            "x": [ROI_MIN_M, ROI_MAX_M],
            "y": [ROI_MIN_M, ROI_MAX_M],
        },
        "domain_um": args.domain_um,
        "source_span_um": source_span_m * 1.0e6,
        "source_z_um": args.source_z_um,
        "monitor_z_um": args.monitor_z_um,
        "z_min_um": args.z_min_um,
        "z_max_um": args.z_max_um,
        "source_to_monitor_distance_um": args.source_z_um - args.monitor_z_um,
        "source_kind": args.source_kind,
        "requested_plane_wave_type": args.plane_wave_type,
        "source_extends_beyond_lateral_fdtd_boundary": (
            source_span_m > domain_m
        ),
        "pml_layers": args.pml_layers,
        "pml_profile": args.pml_profile,
        "simulation_time_ps": args.simulation_time_ps,
        "sampling_mesh_span_um": sampling_mesh_span_m * 1.0e6,
        "gpu_device": args.gpu_device,
        "periodic_boundary": False,
        "tfsf": args.source_kind == "tfsf",
        "gaussian": False,
        "imported_flat_top": args.source_kind == "imported-flat-top",
        "device_geometry_present": False,
        "adjoint_run": False,
        "gradient_run": False,
    }
    fdtd = None
    started = time.monotonic()
    try:
        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        fdtd.switchtolayout()

        fdtd.addfdtd()
        fdtd.setnamed("FDTD", "dimension", "3D")
        fdtd.setnamed("FDTD", "x span", domain_m)
        fdtd.setnamed("FDTD", "y span", domain_m)
        fdtd.setnamed("FDTD", "z min", z_min_m)
        fdtd.setnamed("FDTD", "z max", z_max_m)
        for axis in "xyz":
            fdtd.setnamed("FDTD", f"{axis} min bc", "PML")
            fdtd.setnamed("FDTD", f"{axis} max bc", "PML")
        fdtd.setnamed("FDTD", "pml layers", args.pml_layers)
        result["pml_profile_matrix_before"] = np.asarray(
            fdtd.getnamed("FDTD", "pml profile")
        ).tolist()
        if args.pml_profile == "steep angle":
            fdtd.setnamed("FDTD", "same settings on all boundaries", 0)
            # v261 stores the six boundary profiles as a 1-based enum matrix:
            # standard=1, stabilized=2, steep-angle=3, custom=4.  Apply steep
            # angle only to x/y; retain standard on propagation-axis z PML.
            fdtd.setnamed(
                "FDTD",
                "pml profile",
                np.asarray([[3], [3], [3], [3], [1], [1]], float),
            )
        result["pml_profile_matrix_after"] = np.asarray(
            fdtd.getnamed("FDTD", "pml profile")
        ).tolist()
        fdtd.setnamed("FDTD", "mesh type", "auto non-uniform")
        fdtd.setnamed("FDTD", "mesh accuracy", 2)
        fdtd.setnamed(
            "FDTD", "simulation time", args.simulation_time_ps * 1.0e-12
        )
        fdtd.setnamed("FDTD", "auto shutoff min", 1.0e-7)

        source_name = "gpu_plane_wave_probe"
        if args.source_kind == "plane-wave":
            source = fdtd.addplane()
            source["name"] = source_name
            source["injection axis"] = "z"
            source["direction"] = "backward"
            source["polarization angle"] = 0.0
            source["angle theta"] = 0.0
            source["angle phi"] = 0.0
            source["plane wave type"] = (
                "Diffracting"
                if args.plane_wave_type == "diffracting"
                else "Bloch/Periodic"
            )
            source["x span"] = source_span_m
            source["y span"] = source_span_m
            source["z"] = source_z_m
            source["override global source settings"] = True
            source["wavelength start"] = 3.0e-6
            source["wavelength stop"] = 6.0e-6
        elif args.source_kind == "imported-flat-top":
            source = fdtd.addimportedsource()
            source["name"] = source_name
            source["injection axis"] = "z"
            source["direction"] = "backward"
            source["override global source settings"] = True
            source["wavelength start"] = 3.0e-6
            source["wavelength stop"] = 6.0e-6
            coordinate_m = np.linspace(
                -args.flat_zero_half_span_um * 1.0e-6,
                args.flat_zero_half_span_um * 1.0e-6,
                args.flat_source_samples,
            )
            one_d = raised_cosine_flat_top(
                coordinate_m,
                args.flat_plateau_half_span_um * 1.0e-6,
                args.flat_zero_half_span_um * 1.0e-6,
            )
            amplitude = np.ascontiguousarray(
                (one_d[:, None] * one_d[None, :])[:, :, None]
            )
            zeros = np.zeros_like(amplitude)
            impedance_v_per_a = 376.730313668
            fdtd.putv("flat_x", coordinate_m)
            fdtd.putv("flat_y", coordinate_m)
            fdtd.putv("flat_z", np.asarray([source_z_m]))
            fdtd.putv("flat_Ex", amplitude)
            fdtd.putv("flat_Ey", zeros)
            fdtd.putv("flat_Ez", zeros)
            fdtd.putv("flat_Hx", zeros)
            fdtd.putv("flat_Hy", -amplitude / impedance_v_per_a)
            fdtd.putv("flat_Hz", zeros)
            dataset_script = (
                'flat_ds=rectilineardataset("flat top fields",'
                "flat_x,flat_y,flat_z);"
                f'flat_ds.addparameter("lambda",{WAVELENGTH_M},'
                f'"f",{299792458.0 / WAVELENGTH_M});'
                'flat_ds.addattribute("E",flat_Ex,flat_Ey,flat_Ez);'
            )
            if args.flat_h_model == "local-plane-wave":
                dataset_script += (
                    'flat_ds.addattribute("H",flat_Hx,flat_Hy,flat_Hz);'
                )
            dataset_script += (
                f'select("{source_name}");importdataset(flat_ds);'
            )
            fdtd.eval(dataset_script)
            # Importing a single-frequency profile resets the temporal source
            # limits to that one frequency. Restore the approved broadband
            # pulse after import; the spatial profile itself is evaluated at
            # the 4 um certificate wavelength.
            fdtd.setnamed(source_name, "override global source settings", 1)
            fdtd.setnamed(source_name, "wavelength start", 3.0e-6)
            fdtd.setnamed(source_name, "wavelength stop", 6.0e-6)
            result["flat_top_contract"] = {
                "plateau_half_span_um": args.flat_plateau_half_span_um,
                "zero_half_span_um": args.flat_zero_half_span_um,
                "samples_per_axis": args.flat_source_samples,
                "central_plateau_exactly_one": bool(
                    np.all(
                        amplitude[
                            np.ix_(
                                np.flatnonzero(
                                    np.abs(coordinate_m)
                                    <= args.flat_plateau_half_span_um * 1.0e-6
                                    + 1.0e-15
                                ),
                                np.flatnonzero(
                                    np.abs(coordinate_m)
                                    <= args.flat_plateau_half_span_um * 1.0e-6
                                    + 1.0e-15
                                ),
                                np.asarray([0]),
                            )
                        ]
                        == 1.0
                    )
                ),
                "edge_amplitude_exactly_zero": bool(
                    np.all(amplitude[[0, -1], :, :] == 0.0)
                    and np.all(amplitude[:, [0, -1], :] == 0.0)
                ),
                "electric_polarization": "Ex only",
                "magnetic_field_model": args.flat_h_model,
                "magnetic_polarization": (
                    "derived by Lumerical"
                    if args.flat_h_model == "automatic"
                    else "Hy=-Ex/Z0"
                ),
                "propagation": "-z",
            }
        else:
            source = fdtd.addtfsf()
            source["name"] = source_name
            source["injection axis"] = "z"
            source["direction"] = "backward"
            source["polarization angle"] = 0.0
            source["angle theta"] = 0.0
            source["angle phi"] = 0.0
            source["x span"] = source_span_m
            source["y span"] = source_span_m
            source["z"] = 0.0
            source["z span"] = 2.0e-6
            source["override global source settings"] = True
            source["wavelength start"] = 3.0e-6
            source["wavelength stop"] = 6.0e-6
        result["source_property_readback_before_save"] = source_readback(
            fdtd, source_name
        )
        try:
            source_profile = fdtd.getresult(source_name, "fields")
            profile_e = np.asarray(source_profile["E"])
            result["source_profile_readback"] = {
                "E_shape": list(profile_e.shape),
                "E_all_finite": bool(np.all(np.isfinite(profile_e))),
                "E_max_abs": float(np.max(np.abs(profile_e))),
                "E_min_abs": float(np.min(np.abs(profile_e))),
                "E_component_L2": [
                    float(np.linalg.norm(profile_e[..., component]))
                    for component in range(3)
                ],
                "x_min_m": float(
                    np.min(np.asarray(source_profile["x"], float))
                ),
                "x_max_m": float(
                    np.max(np.asarray(source_profile["x"], float))
                ),
                "y_min_m": float(
                    np.min(np.asarray(source_profile["y"], float))
                ),
                "y_max_m": float(
                    np.max(np.asarray(source_profile["y"], float))
                ),
                "z_min_m": float(
                    np.min(np.asarray(source_profile["z"], float))
                ),
                "z_max_m": float(
                    np.max(np.asarray(source_profile["z"], float))
                ),
            }
        except Exception as exc:
            result["source_profile_readback"] = {
                "error": f"{type(exc).__name__}: {exc}"
            }

        roi_mesh = fdtd.addmesh()
        roi_mesh["name"] = "roi_sampling_mesh"
        roi_mesh["x span"] = sampling_mesh_span_m
        roi_mesh["y span"] = sampling_mesh_span_m
        roi_mesh["z min"] = monitor_z_m - 0.1e-6
        roi_mesh["z max"] = monitor_z_m + 0.1e-6
        roi_mesh["override x mesh"] = 1
        roi_mesh["override y mesh"] = 1
        roi_mesh["override z mesh"] = 0
        roi_mesh["dx"] = 0.1e-6
        roi_mesh["dy"] = 0.1e-6

        monitor = fdtd.addpower()
        monitor["name"] = "roi_xy"
        monitor["monitor type"] = "2D Z-normal"
        monitor["x min"] = ROI_MIN_M
        monitor["x max"] = ROI_MAX_M
        monitor["y min"] = ROI_MIN_M
        monitor["y max"] = ROI_MAX_M
        monitor["z"] = monitor_z_m
        monitor["override global monitor settings"] = True
        monitor["use source limits"] = False
        monitor["use wavelength spacing"] = True
        monitor["wavelength center"] = WAVELENGTH_M
        monitor["wavelength span"] = 0.0
        monitor["frequency points"] = 1

        fdtd.setresource("FDTD", 1, "active", 0)
        fdtd.setresource("FDTD", 2, "active", 1)
        fdtd.setresource("FDTD", 2, "processes", "1")
        fdtd.setresource("FDTD", 2, "threads", str(args.threads))
        fdtd.setresource("FDTD", 2, "device type", args.gpu_device)
        fdtd.setresource(
            "FDTD", 2, "solver extra command line options", "-gpu"
        )
        result["resources_before_run"] = resource_snapshot(fdtd)

        project = output / "gpu_plane_wave_roi_probe.fsp"
        fdtd.save(str(project))
        result["source_property_readback_after_save"] = source_readback(
            fdtd, source_name
        )
        result["gpu_session_resource_name"] = run_on_gpu(fdtd)
        fdtd.save(str(project))
        result["resources_after_run"] = resource_snapshot(fdtd)

        ex = np.asarray(fdtd.getdata("roi_xy", "Ex", 1)).squeeze()
        ey = np.asarray(fdtd.getdata("roi_xy", "Ey", 1)).squeeze()
        ez = np.asarray(fdtd.getdata("roi_xy", "Ez", 1)).squeeze()
        x_m = np.asarray(fdtd.getdata("roi_xy", "x", 1), float).reshape(-1)
        y_m = np.asarray(fdtd.getdata("roi_xy", "y", 1), float).reshape(-1)
        x_inside = np.flatnonzero(
            (x_m >= ROI_MIN_M - 1.0e-15)
            & (x_m <= ROI_MAX_M + 1.0e-15)
        )
        y_inside = np.flatnonzero(
            (y_m >= ROI_MIN_M - 1.0e-15)
            & (y_m <= ROI_MAX_M + 1.0e-15)
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
        intensity = scalar(
            fdtd.sourceintensity(
                frequency_hz, 2, "gpu_plane_wave_probe"
            ),
            "source intensity",
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
            "intensity_relative_rms": (
                float(np.std(e2)) / max(mean_e2, np.finfo(float).tiny)
            ),
            "intensity_relative_peak_to_peak": (
                float(np.max(e2) - np.min(e2))
                / max(mean_e2, np.finfo(float).tiny)
            ),
            "phase_rms_deg": float(np.sqrt(np.mean(phase**2)) * 180.0 / np.pi),
            "phase_max_abs_deg": float(np.max(np.abs(phase)) * 180.0 / np.pi),
            "Ey_to_Ex_L2": float(np.linalg.norm(ey))
            / max(float(np.linalg.norm(ex)), np.finfo(float).tiny),
            "Ez_to_Ex_L2": float(np.linalg.norm(ez))
            / max(float(np.linalg.norm(ex)), np.finfo(float).tiny),
            "source_intensity_native_W_m2": intensity,
            "all_fields_finite": bool(
                np.all(np.isfinite(ex))
                and np.all(np.isfinite(ey))
                and np.all(np.isfinite(ez))
            ),
        }
        result["roi_metrics"] = metrics
        result["acceptance"] = {
            "gpu_resource_active": (
                result["resources_after_run"]["2"]["active"].strip() == "1"
            ),
            "cpu_resource_inactive": (
                result["resources_after_run"]["1"]["active"].strip() == "0"
            ),
            "exact_roi_monitor_bounds": (
                abs(x_roi_m[0] - ROI_MIN_M) <= 0.051e-6
                and abs(x_roi_m[-1] - ROI_MAX_M) <= 0.051e-6
                and abs(y_roi_m[0] - ROI_MIN_M) <= 0.051e-6
                and abs(y_roi_m[-1] - ROI_MAX_M) <= 0.051e-6
            ),
            "field_finite": metrics["all_fields_finite"],
            "intensity_rms_lt_0p5_percent": (
                metrics["intensity_relative_rms"] < 0.005
            ),
            "mean_intensity_error_lt_0p5_percent": (
                metrics["mean_E2_relative_error_from_unit_source"] < 0.005
            ),
            "intensity_peak_to_peak_lt_1_percent": (
                metrics["intensity_relative_peak_to_peak"] < 0.01
            ),
            "phase_max_lt_1_degree": (
                metrics["phase_max_abs_deg"] < 1.0
            ),
            "cross_polarized_components_lt_0p1_percent": (
                max(metrics["Ey_to_Ex_L2"], metrics["Ez_to_Ex_L2"]) < 0.001
            ),
        }
        passed = all(bool(value) for value in result["acceptance"].values())
        result["status"] = (
            "GPU_PLANE_WAVE_ROI_PROBE_PASSED"
            if passed
            else "FAILED_GPU_PLANE_WAVE_ROI_PROBE"
        )
        return_code = 0 if passed else 2
    except Exception as exc:
        message = str(exc)
        result["status"] = (
            "BLOCKED_GPU_TFSF_UNSUPPORTED"
            if "GPU simulation does not support the use of TFSF sources"
            in message
            else "BLOCKED_GPU_PLANE_WAVE_UNAVAILABLE"
        )
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        return_code = 3
    finally:
        result["elapsed_s"] = time.monotonic() - started
        result_path.write_text(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
                default=json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
