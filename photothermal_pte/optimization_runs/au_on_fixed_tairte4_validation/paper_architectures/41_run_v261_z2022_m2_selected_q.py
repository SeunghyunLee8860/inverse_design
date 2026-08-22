#!/usr/bin/env python3
"""GPU selected-wavelength volumetric-Q certificate for reconstructed Z M2."""

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


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.native_yee_q import extract_native_yee_q
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (
    PABS_FIELD,
    PABS_GROUP,
    PABS_INDEX,
)
from photothermal_pte.validation.paper_ir_sanity import (
    validate_paper_ir_source_only_gpu as audit,
)


C0 = 299_792_458.0
DEFAULT_WAVELENGTH_UM = 5.25


def load_module(filename: str, name: str):
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


def scalar(value: object, name: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"{name} is not scalar: {array.shape}")
    return float(np.real(array[0]))


def max_spacing_inside(coordinates: np.ndarray, lower: float, upper: float) -> float:
    coordinates = np.asarray(coordinates, float).reshape(-1)
    selected = coordinates[(coordinates >= lower - 1.0e-15) & (coordinates <= upper + 1.0e-15)]
    if selected.size < 2:
        raise RuntimeError(f"insufficient mesh coordinates in [{lower}, {upper}]")
    return float(np.max(np.diff(selected)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default=os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", "GPU 0"))
    parser.add_argument("--handedness", choices=("LH", "RH"), default="LH")
    parser.add_argument(
        "--polarization",
        choices=("x_b", "y_a", "CP_plus", "CP_minus"),
        default="CP_plus",
    )
    parser.add_argument(
        "--geometry-variant",
        choices=(
            "legacy_axis_swapped_v1",
            "figure_axis_corrected_v2",
            "figure_period_corrected_v3",
        ),
        default="legacy_axis_swapped_v1",
    )
    parser.add_argument("--wavelength-um", type=float, default=DEFAULT_WAVELENGTH_UM)
    parser.add_argument("--duration-ps", type=float, default=1.5)
    parser.add_argument(
        "--omit-top-au-control",
        action="store_true",
        help="diagnostic planar-stack control; preserves all other settings",
    )
    parser.add_argument("--top-au-edge-mesh-nm", type=float)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    wavelength_m = args.wavelength_um * 1e-6
    frequency_hz = C0 / wavelength_m
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "Z2022_M2_selected_Q.json"
    npz_path = output / "Z2022_M2_selected_Q.npz"
    fsp_path = output / "Z2022_M2_selected_Q.fsp"
    result: dict[str, object] = {"status": "BLOCKED_Z2022_M2_SELECTED_Q"}
    fdtd = None
    try:
        zrun = load_module("19_run_v261_z2022_m2_periodic_broadband_rta.py", "z_selected_setup")
        base = load_module("07_run_v261_t2024_tairte4_optical_smoke.py", "z_selected_helpers")
        backplane = load_module("02_run_v261_backplane_truncation_control.py", "z_selected_pabs")
        # `run`/`runres` creates an isolated, licensed Lumerical environment.
        # Preserve that root instead of replacing it with a user-local install,
        # because mixing the two roots breaks ANSYSLI license sharing.
        lumerical_root = Path(os.environ.get("LUMERICAL_ROOT", str(audit.APPROVED_ROOT)))
        lumerical_api = Path(os.environ.get("LUMERICAL_PYTHONPATH", str(audit.APPROVED_API)))
        os.environ.setdefault("VC_LUMERICAL_ROOT", str(lumerical_root))
        os.environ.setdefault("LUMERICAL_ROOT", str(lumerical_root))
        os.environ.setdefault("LUMERICAL_PYTHONPATH", str(lumerical_api))
        os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = args.gpu_device
        os.environ["CL_GPU_DEVICE"] = args.gpu_device
        os.environ["FDTD_THREADS"] = "8"
        os.environ["PATH"] = f"{lumerical_root / 'bin'}:{os.environ.get('PATH', '')}"
        sys.path.insert(0, str(lumerical_api))
        import lumapi

        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        contract = zrun.setup(
            fdtd,
            args.handedness,
            args.polarization,
            args.duration_ps,
            geometry_variant=args.geometry_variant,
            top_au_edge_mesh_nm=args.top_au_edge_mesh_nm,
        )
        if args.omit_top_au_control:
            for polygon in contract["geometry"]["polygons"]:
                fdtd.select(str(polygon["name"]))
                fdtd.delete()
        source_names = (
            ("Z2022_source_linear",)
            if args.polarization in ("x_b", "y_a")
            else ("Z2022_source_x", "Z2022_source_y")
        )
        source_start = max(4.0e-6, wavelength_m * 0.95)
        source_stop = min(12.0e-6, wavelength_m * 1.05)
        for name in source_names:
            fdtd.setnamed(name, "wavelength start", source_start)
            fdtd.setnamed(name, "wavelength stop", source_stop)
        fdtd.setglobalmonitor("use source limits", False)
        fdtd.setglobalmonitor("use wavelength spacing", True)
        fdtd.setglobalmonitor("wavelength center", wavelength_m)
        fdtd.setglobalmonitor("wavelength span", 0.0)
        fdtd.setglobalmonitor("frequency points", 1)
        for monitor_name in ("Z2022_flux_top", "Z2022_flux_bottom"):
            fdtd.setnamed(monitor_name, "wavelength center", wavelength_m)
            fdtd.setnamed(monitor_name, "wavelength span", 0.0)
            fdtd.setnamed(monitor_name, "frequency points", 1)
        geometry = contract["geometry"]
        period_x = float(geometry["period_x_nm"]) * 1e-9
        period_y = float(geometry["period_y_nm"]) * 1e-9
        pabs = fdtd.addobject("pabs_adv")
        pabs["name"] = PABS_GROUP
        pabs["x"] = 0.0
        pabs["x span"] = period_x
        pabs["y"] = 0.0
        pabs["y span"] = period_y
        pabs["z"] = 0.5 * (zrun.TOP_MONITOR_Z_M + zrun.BOTTOM_MONITOR_Z_M)
        pabs["z span"] = zrun.TOP_MONITOR_Z_M - zrun.BOTTOM_MONITOR_Z_M
        pabs_contract = backplane.enable_pabs_periodic_correction(fdtd)

        fdtd.setresource("FDTD", 1, "active", 0)
        fdtd.setresource("FDTD", 2, "active", 1)
        fdtd.setresource("FDTD", 2, "processes", "1")
        fdtd.setresource("FDTD", 2, "threads", "8")
        fdtd.setresource("FDTD", 2, "device type", args.gpu_device)
        fdtd.setresource("FDTD", 2, "solver extra command line options", "-gpu")
        fdtd.runsetup()
        initial_mesh_readback = audit.mesh_readback(fdtd)
        if not initial_mesh_readback.get("available"):
            raise RuntimeError("native mesh unavailable before control-volume snap")
        native_z = np.asarray(
            initial_mesh_readback["coordinate_arrays"]["z"], float
        ).reshape(-1)
        snapped_bottom = float(
            native_z[int(np.argmin(np.abs(native_z - zrun.BOTTOM_MONITOR_Z_M)))]
        )
        snapped_top = float(
            native_z[int(np.argmin(np.abs(native_z - zrun.TOP_MONITOR_Z_M)))]
        )
        if not snapped_top > snapped_bottom:
            raise RuntimeError("invalid snapped Z control-volume bounds")
        # The two flux faces and the volumetric-loss group must use exactly the
        # same realized Yee planes.  Leaving the requested analytic positions
        # to be snapped independently gives a false closure error.
        fdtd.setnamed(PABS_GROUP, "z", 0.5 * (snapped_bottom + snapped_top))
        fdtd.setnamed(PABS_GROUP, "z span", snapped_top - snapped_bottom)
        fdtd.setnamed("Z2022_flux_top", "z", snapped_top)
        fdtd.setnamed("Z2022_flux_bottom", "z", snapped_bottom)
        fdtd.runsetup()
        mesh_readback = audit.mesh_readback(fdtd)
        mesh = base.mesh_metrics(mesh_readback)
        if not mesh.get("available"):
            raise RuntimeError("native mesh unavailable after runsetup")
        if mesh["min_dx_m"] > 25e-9 + 1e-12 or mesh["min_dy_m"] > 25e-9 + 1e-12:
            raise RuntimeError("25-nm selected-Z in-plane mesh was not realized")
        if mesh["max_structure_dz_m"] > 5e-9 + 1e-12:
            raise RuntimeError("5-nm selected-Z structure dz was not realized")
        edge_mesh_realized: list[dict[str, object]] = []
        if args.top_au_edge_mesh_nm is not None:
            coordinates = mesh_readback["coordinate_arrays"]
            requested_edge = args.top_au_edge_mesh_nm * 1.0e-9
            for item in contract["top_Au_edge_mesh"]["objects"]:
                xmin, xmax, ymin, ymax, zmin, zmax = item["bounds_m"]
                realized = {
                    "name": item["name"],
                    "max_dx_m": max_spacing_inside(coordinates["x"], xmin, xmax),
                    "max_dy_m": max_spacing_inside(coordinates["y"], ymin, ymax),
                    "max_dz_m": max_spacing_inside(coordinates["z"], zmin, zmax),
                }
                if realized["max_dx_m"] > requested_edge + 1.0e-12:
                    raise RuntimeError(f"edge dx was not realized: {realized}")
                if realized["max_dy_m"] > requested_edge + 1.0e-12:
                    raise RuntimeError(f"edge dy was not realized: {realized}")
                if realized["max_dz_m"] > 5.0e-9 + 1.0e-12:
                    raise RuntimeError(f"edge dz was not realized: {realized}")
                edge_mesh_realized.append(realized)
        result.update(
            {
                "classification": (
                    "published M2 scalar dimensions plus Fig. 1b-corrected periods/axes; "
                    "edge-joined figure-constrained reconstruction; not author CAD"
                    if args.geometry_variant == "figure_period_corrected_v3"
                    else
                    "published M2 scalar dimensions plus Fig. 1b axis-corrected "
                    "corner-joined reconstruction; not author CAD"
                    if args.geometry_variant == "figure_axis_corrected_v2"
                    else "legacy axis-swapped v1 diagnostic; not the Fig. 1b geometry"
                ),
                "geometry_variant": args.geometry_variant,
                "geometry": geometry,
                "wavelength_um": args.wavelength_um,
                "handedness": args.handedness,
                "polarization": args.polarization,
                "phase_definition": contract["source"],
                "pabs_contract": pabs_contract,
                "matched_lossy_control_volume": {
                    "method": "post-runsetup nearest native z planes",
                    "requested_z_bounds_m": [
                        zrun.BOTTOM_MONITOR_Z_M,
                        zrun.TOP_MONITOR_Z_M,
                    ],
                    "realized_z_bounds_m": [snapped_bottom, snapped_top],
                    "maximum_absolute_shift_m": max(
                        abs(snapped_bottom - zrun.BOTTOM_MONITOR_Z_M),
                        abs(snapped_top - zrun.TOP_MONITOR_Z_M),
                    ),
                    "Pabs_and_both_flux_faces_updated_together": True,
                },
                "mesh_runsetup": mesh,
                "solver_version": str(fdtd.version()),
                "scope": "periodic selected-wavelength volumetric-Q certificate; no thermal/PTE",
                "top_Au_included": not args.omit_top_au_control,
                "top_Au_edge_mesh": contract["top_Au_edge_mesh"],
                "top_Au_edge_mesh_realized": edge_mesh_realized,
            }
        )
        fdtd.save(str(fsp_path))
        if args.contract_only:
            result["status"] = "COMPLETED_Z2022_M2_SELECTED_Q_RUNSETUP"
        else:
            started = time.monotonic()
            result["GPU_resource_used"] = audit.strict_gpu_run(fdtd, "Z2022_M2_selected_Q")
            result["solver_wall_time_s"] = time.monotonic() - started
            source_power = sum(
                scalar(fdtd.sourcepower(frequency_hz, 2, name), f"sourcepower:{name}")
                for name in source_names
            )
            top_signed = scalar(fdtd.transmission("Z2022_flux_top"), "top") * source_power
            bottom_signed = scalar(fdtd.transmission("Z2022_flux_bottom"), "bottom") * source_power
            p_flux = bottom_signed - top_signed
            fdtd.runanalysis(PABS_GROUP)
            pabs_normalized = scalar(
                fdtd.getresult(PABS_GROUP, "Pabs_total")["Pabs_total"], "Pabs_total"
            )
            p_q = pabs_normalized * source_power
            q = extract_native_yee_q(
                fdtd,
                field_monitor=PABS_FIELD,
                index_monitor=PABS_INDEX,
                wavelength_m=wavelength_m,
            )
            arrays: dict[str, np.ndarray] = {}
            negative: dict[str, int] = {}
            for component in "xyz":
                values = np.asarray(q["Q_components"][component], float)
                arrays[f"Q{component}_W_m3"] = values
                negative[component] = int(np.count_nonzero(values < 0.0))
                for axis in "xyz":
                    arrays[f"Q{component}_{axis}_m"] = np.asarray(
                        q["native_coordinates"][component][axis], float
                    )
            np.savez_compressed(npz_path, **arrays)
            closure = abs(p_q - p_flux) / max(abs(p_q), abs(p_flux), np.finfo(float).tiny)
            log = audit.log_audit(output)
            gates = {
                "GPU_completed": bool(log["simulation_completed_successfully"]),
                "auto_shutoff_lt_1e_5": log["final_auto_shutoff"] is not None and log["final_auto_shutoff"] < 1e-5,
                "closure_lt_0p5pct": closure < 0.005,
                "all_Q_arrays_finite": all(np.all(np.isfinite(arrays[f"Q{c}_W_m3"])) for c in "xyz"),
                "no_negative_Q": sum(negative.values()) == 0,
            }
            result.update(
                {
                    "source_power_W": source_power,
                    "P_flux_absorbed_W": p_flux,
                    "P_Q_pabs_periodic_W": p_q,
                    "P_Q_native_uncorrected_W": float(q["P_Q_W"]),
                    "Q_component_power_native_W": q["component_power_W"],
                    "closure_relative": closure,
                    "negative_Q_cell_count": negative,
                    "log_audit": log,
                    "gates": gates,
                    "status": (
                        "COMPLETED_Z2022_M2_FIGURE_PERIOD_CORRECTED_SELECTED_Q"
                        if all(gates.values()) and args.geometry_variant == "figure_period_corrected_v3"
                        else "COMPLETED_Z2022_M2_FIGURE_CORRECTED_SELECTED_Q"
                        if all(gates.values()) and args.geometry_variant == "figure_axis_corrected_v2"
                        else "COMPLETED_Z2022_M2_RECONSTRUCTED_SELECTED_Q"
                        if all(gates.values())
                        else "FAILED_Z2022_M2_SELECTED_Q_GATE"
                    ),
                }
            )
            fdtd.save(str(fsp_path))
        result["raw_artifacts"] = [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (fsp_path, npz_path)
            if path.is_file()
        ]
    except Exception as exc:
        result["status"] = "BLOCKED_Z2022_M2_SELECTED_Q"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        json_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0 if str(result["status"]).startswith("COMPLETED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
