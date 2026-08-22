#!/usr/bin/env python3
"""GPU periodic 4--12 um R/T/A screen for the reconstructed 2022 M2 Z cell."""

from __future__ import annotations

import argparse
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

from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    validate_paper_ir_source_only_gpu as audit,
)


C0 = 299_792_458.0
WAVELENGTH_MIN_M = 4.0e-6
WAVELENGTH_MAX_M = 12.0e-6
WAVELENGTH_POINTS = 321
Z_MIN_M = -1.5e-6
Z_MAX_M = 1.5e-6
SOURCE_Z_M = 1.0e-6
TOP_MONITOR_Z_M = 0.65e-6
BOTTOM_MONITOR_Z_M = -1.0e-6


def load_module(filename: str, name: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_flux_monitor(fdtd: object, name: str, x_span: float, y_span: float, z: float) -> None:
    monitor = fdtd.addpower()
    monitor["name"] = name
    monitor["monitor type"] = "2D Z-normal"
    monitor["x span"] = x_span
    monitor["y span"] = y_span
    monitor["z"] = z
    monitor["override global monitor settings"] = True
    monitor["use source limits"] = False
    monitor["use wavelength spacing"] = True
    monitor["wavelength center"] = 0.5 * (WAVELENGTH_MIN_M + WAVELENGTH_MAX_M)
    monitor["wavelength span"] = WAVELENGTH_MAX_M - WAVELENGTH_MIN_M
    monitor["frequency points"] = WAVELENGTH_POINTS


def add_plane_source(
    fdtd: object,
    name: str,
    x_span: float,
    y_span: float,
    polarization_angle_deg: float,
    phase_deg: float,
    amplitude: float,
) -> None:
    source = fdtd.addplane()
    source["name"] = name
    source["injection axis"] = "z"
    source["direction"] = "backward"
    source["plane wave type"] = "Bloch/Periodic"
    source["angle theta"] = 0.0
    source["angle phi"] = 0.0
    source["polarization angle"] = polarization_angle_deg
    source["phase"] = phase_deg
    source["amplitude"] = amplitude
    source["x span"] = x_span
    source["y span"] = y_span
    source["z"] = SOURCE_Z_M
    source["override global source settings"] = True
    source["wavelength start"] = WAVELENGTH_MIN_M
    source["wavelength stop"] = WAVELENGTH_MAX_M


def setup(fdtd: object, handedness: str, polarization: str, duration_ps: float) -> dict[str, object]:
    geometry_module = load_module("05_actual_metasurface_geometry.py", "z2022_geometry")
    base = load_module("07_run_v261_t2024_tairte4_optical_smoke.py", "z2022_material_helpers")
    geometry = geometry_module.z_m2_5300nm_corner_joined_tairte4(handedness)
    period_x = geometry.period_x_nm * 1e-9
    period_y = geometry.period_y_nm * 1e-9

    solver = fdtd.addfdtd()
    solver["dimension"] = "3D"
    solver["x span"] = period_x
    solver["y span"] = period_y
    solver["z min"] = Z_MIN_M
    solver["z max"] = Z_MAX_M
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
    solver["simulation time"] = duration_ps * 1e-12
    solver["auto shutoff min"] = 1e-6
    solver["override simulation bandwidth for mesh generation"] = True
    solver["mesh wavelength min"] = WAVELENGTH_MIN_M
    solver["mesh wavelength max"] = WAVELENGTH_MAX_M

    if polarization in ("x_b", "y_a"):
        add_plane_source(
            fdtd,
            "Z2022_source_linear",
            period_x,
            period_y,
            0.0 if polarization == "x_b" else 90.0,
            0.0,
            1.0,
        )
    else:
        phase = 90.0 if polarization == "CP_plus" else -90.0
        amplitude = 1.0 / np.sqrt(2.0)
        add_plane_source(fdtd, "Z2022_source_x", period_x, period_y, 0.0, 0.0, amplitude)
        add_plane_source(fdtd, "Z2022_source_y", period_x, period_y, 90.0, phase, amplitude)

    tairte4 = base.add_tairte4_material(fdtd)
    base.add_constant_nk(fdtd, base.AL2O3_MATERIAL, base.AL2O3_N)
    base.add_rect(fdtd, "Z2022_Au_mirror_200nm", base.AU_MATERIAL, -400e-9, -200e-9)
    base.add_rect(fdtd, "Z2022_Al2O3_200nm", base.AL2O3_MATERIAL, -200e-9, 0.0)
    base.add_rect(fdtd, "Z2022_TaIrTe4_100nm", base.TAIRTE4_MATERIAL, 0.0, 100e-9)
    base.add_rect(fdtd, "Z2022_optical_SiO2_285nm", base.SIO2_MATERIAL, -685e-9, -400e-9)
    base.add_rect(fdtd, "Z2022_Si_substrate", base.SI_MATERIAL, Z_MIN_M, -685e-9)
    # The helper uses the T-cell spans; expand every planar layer to this Z cell.
    for name in (
        "Z2022_Au_mirror_200nm",
        "Z2022_Al2O3_200nm",
        "Z2022_TaIrTe4_100nm",
        "Z2022_optical_SiO2_285nm",
        "Z2022_Si_substrate",
    ):
        fdtd.setnamed(name, "x span", period_x)
        fdtd.setnamed(name, "y span", period_y)
    for item in geometry.polygons:
        polygon = fdtd.addpoly()
        polygon["name"] = item.name
        polygon["material"] = base.AU_MATERIAL
        polygon["vertices"] = np.asarray(item.vertices_nm, float) * 1e-9
        polygon["z min"] = item.z_min_nm * 1e-9
        polygon["z max"] = item.z_max_nm * 1e-9

    mesh = fdtd.addmesh()
    mesh["name"] = "Z2022_local_structure_mesh"
    mesh["x span"] = period_x
    mesh["y span"] = period_y
    mesh["z min"] = -0.45e-6
    mesh["z max"] = 0.20e-6
    mesh["override x mesh"] = True
    mesh["override y mesh"] = True
    mesh["override z mesh"] = True
    mesh["dx"] = 25e-9
    mesh["dy"] = 25e-9
    mesh["dz"] = 5e-9
    add_flux_monitor(fdtd, "Z2022_flux_top", period_x, period_y, TOP_MONITOR_Z_M)
    add_flux_monitor(fdtd, "Z2022_flux_bottom", period_x, period_y, BOTTOM_MONITOR_Z_M)
    return {
        "geometry": geometry.as_dict(),
        "materials": {"TaIrTe4": tairte4, "Au": base.AU_MATERIAL, "Al2O3_n": [base.AL2O3_N.real, base.AL2O3_N.imag]},
        "source": {
            "polarization": polarization,
            "propagation": "-z",
            "CP_plus_definition": "Ex phase 0 deg; Ey phase +90 deg",
            "CP_minus_definition": "Ex phase 0 deg; Ey phase -90 deg",
            "LCP_RCP_name_assignment": "not promoted until propagation/time-convention audit",
        },
        "scope": "periodic flux R/T/A only; no volumetric Q/thermal/PTE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument("--handedness", choices=("LH", "RH"), default="LH")
    parser.add_argument("--polarization", choices=("x_b", "y_a", "CP_plus", "CP_minus"), default="CP_plus")
    parser.add_argument("--duration-ps", type=float, default=2.0)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "Z2022_M2_periodic_broadband_rta.json"
    npz_path = output / "Z2022_M2_periodic_broadband_rta.npz"
    fsp_path = output / "Z2022_M2_periodic_broadband_rta.fsp"
    result: dict[str, object] = {"status": "BLOCKED_Z2022_M2_PERIODIC_BROADBAND_RTA"}
    fdtd = None
    try:
        base = load_module("07_run_v261_t2024_tairte4_optical_smoke.py", "z2022_raw_helpers")
        spectrum = load_module("17_run_v261_t2024_periodic_broadband_rta.py", "z2022_rta_helpers")
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
        contract = setup(fdtd, args.handedness, args.polarization, args.duration_ps)
        fdtd.setresource("FDTD", 1, "active", 0)
        fdtd.setresource("FDTD", 2, "active", 1)
        fdtd.setresource("FDTD", 2, "processes", "1")
        fdtd.setresource("FDTD", 2, "threads", "8")
        fdtd.setresource("FDTD", 2, "device type", args.gpu_device)
        fdtd.setresource("FDTD", 2, "solver extra command line options", "-gpu")
        fdtd.runsetup()
        mesh = base.mesh_metrics(audit.mesh_readback(fdtd))
        if not mesh.get("available"):
            raise RuntimeError("native mesh unavailable after runsetup")
        result.update({"contract": contract, "solver_version": str(fdtd.version()), "mesh_runsetup": mesh})
        fdtd.save(str(fsp_path))
        if args.contract_only:
            result["status"] = "COMPLETED_Z2022_M2_PERIODIC_RUNSETUP"
        else:
            started = time.monotonic()
            result["GPU_resource_used"] = audit.strict_gpu_run(fdtd, "Z2022_M2_periodic_broadband_rta")
            result["solver_wall_time_s"] = time.monotonic() - started
            frequency = np.asarray(fdtd.getdata("Z2022_flux_top", "f", 1), float).reshape(-1)
            top = np.real(np.asarray(fdtd.transmission("Z2022_flux_top"))).reshape(-1)
            bottom = np.real(np.asarray(fdtd.transmission("Z2022_flux_bottom"))).reshape(-1)
            wavelength = C0 / frequency
            order = np.argsort(wavelength)
            wavelength, top, bottom = wavelength[order], top[order], bottom[order]
            reflection, transmission, absorption = spectrum.rta_from_signed_transmission(top, bottom)
            finite = bool(np.all(np.isfinite(reflection)) and np.all(np.isfinite(transmission)) and np.all(np.isfinite(absorption)))
            np.savez_compressed(npz_path, wavelength_m=wavelength, frequency_hz=frequency[order], R=reflection, T=transmission, A=absorption)
            log = audit.log_audit(output)
            gates = {
                "GPU_completed": bool(log["simulation_completed_successfully"]),
                "auto_shutoff_lt_1e_5": log["final_auto_shutoff"] is not None and log["final_auto_shutoff"] < 1e-5,
                "finite_RTA": finite,
                "no_large_negative_absorption": float(np.min(absorption)) > -0.01,
            }
            peak = int(np.argmax(absorption))
            result.update(
                {
                    "log_audit": log,
                    "gates": gates,
                    "peak_total_absorption": {"wavelength_m": float(wavelength[peak]), "A": float(absorption[peak]), "R": float(reflection[peak]), "T": float(transmission[peak])},
                    "status": "COMPLETED_Z2022_M2_PERIODIC_BROADBAND_RTA" if all(gates.values()) else "FAILED_Z2022_M2_PERIODIC_BROADBAND_RTA_GATE",
                }
            )
            fdtd.save(str(fsp_path))
        result["raw_artifacts"] = [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": base.sha256(path)}
            for path in (fsp_path, npz_path)
            if path.is_file()
        ]
    except Exception as exc:
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
