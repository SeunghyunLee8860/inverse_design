#!/usr/bin/env python3
"""GPU-only periodic broadband R/T/A screening for the 2024 inverse-T.

This deliberately solves only the inexpensive periodic optical screening
problem.  It does not construct a finite array, a Gaussian beam, volumetric
Q, temperature, PTE current, an adjoint, or an inverse design.
"""

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
WAVELENGTH_MIN_M = 3.0e-6
WAVELENGTH_MAX_M = 8.0e-6
WAVELENGTH_POINTS = 251


def load_base():
    path = HERE / "07_run_v261_t2024_tairte4_optical_smoke.py"
    spec = importlib.util.spec_from_file_location("t2024_single_frequency_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_broadband(fdtd: object, base: object, duration_ps: float) -> None:
    """Remove heavy 3-D Q monitors and retain only two spectral flux planes."""

    fdtd.select(base.PABS_GROUP)
    fdtd.delete()
    fdtd.setnamed(base.SOURCE_NAME, "wavelength start", WAVELENGTH_MIN_M)
    fdtd.setnamed(base.SOURCE_NAME, "wavelength stop", WAVELENGTH_MAX_M)
    fdtd.setnamed("FDTD", "simulation time", duration_ps * 1.0e-12)
    fdtd.setnamed("FDTD", "mesh wavelength min", WAVELENGTH_MIN_M)
    fdtd.setnamed("FDTD", "mesh wavelength max", WAVELENGTH_MAX_M)

    center = 0.5 * (WAVELENGTH_MIN_M + WAVELENGTH_MAX_M)
    span = WAVELENGTH_MAX_M - WAVELENGTH_MIN_M
    for name in (base.TOP_MONITOR, base.BOTTOM_MONITOR):
        fdtd.setnamed(name, "override global monitor settings", True)
        fdtd.setnamed(name, "use source limits", False)
        fdtd.setnamed(name, "use wavelength spacing", True)
        fdtd.setnamed(name, "wavelength center", center)
        fdtd.setnamed(name, "wavelength span", span)
        fdtd.setnamed(name, "frequency points", WAVELENGTH_POINTS)


def rta_from_signed_transmission(
    top_signed: np.ndarray, bottom_signed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Lumerical signed z-normal transmissions into R, T and A."""

    reflection = 1.0 + np.asarray(top_signed, float)
    transmission = -np.asarray(bottom_signed, float)
    absorption = 1.0 - reflection - transmission
    return reflection, transmission, absorption


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument("--polarization", choices=("x_b", "y_a"), default="x_b")
    parser.add_argument("--duration-ps", type=float, default=2.0)
    parser.add_argument("--omit-top-t-control", action="store_true")
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "T2024_periodic_broadband_rta.json"
    spectrum_path = output / "T2024_periodic_broadband_rta.npz"
    fsp_path = output / "T2024_periodic_broadband_rta.fsp"
    result: dict[str, object] = {
        "status": "BLOCKED_T2024_PERIODIC_BROADBAND_RTA",
        "scope": "periodic broadband flux screening only; no Q/thermal/PTE/adjoint/optimization",
    }
    fdtd = None
    try:
        base = load_base()
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
        contract = base.setup(
            fdtd,
            args.polarization,
            args.duration_ps,
            include_top_t=not args.omit_top_t_control,
            substrate_mode="sio2_si_reduced_285nm",
        )
        configure_broadband(fdtd, base, args.duration_ps)
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
        result.update(
            {
                "contract": contract,
                "solver_version": str(fdtd.version()),
                "mesh_runsetup": mesh,
                "spectrum_contract": {
                    "wavelength_bounds_m": [WAVELENGTH_MIN_M, WAVELENGTH_MAX_M],
                    "wavelength_points": WAVELENGTH_POINTS,
                    "quantity": "flux-derived R, T, A=1-R-T",
                    "periodic_xy": True,
                    "plane_wave_normal_incidence": True,
                    "volumetric_Q_computed": False,
                },
            }
        )
        fdtd.save(str(fsp_path))
        if args.contract_only:
            result["status"] = "COMPLETED_T2024_PERIODIC_BROADBAND_RUNSETUP"
        else:
            started = time.monotonic()
            result["GPU_resource_used"] = audit.strict_gpu_run(fdtd, "T2024_periodic_broadband_rta")
            result["solver_wall_time_s"] = time.monotonic() - started
            frequency = np.asarray(fdtd.getdata(base.TOP_MONITOR, "f", 1), float).reshape(-1)
            top_signed = np.real(np.asarray(fdtd.transmission(base.TOP_MONITOR))).reshape(-1)
            bottom_signed = np.real(np.asarray(fdtd.transmission(base.BOTTOM_MONITOR))).reshape(-1)
            if not (frequency.size == top_signed.size == bottom_signed.size == WAVELENGTH_POINTS):
                raise RuntimeError(
                    f"unexpected spectral shapes: f={frequency.shape}, top={top_signed.shape}, bottom={bottom_signed.shape}"
                )
            wavelength = C0 / frequency
            order = np.argsort(wavelength)
            wavelength = wavelength[order]
            top_signed = top_signed[order]
            bottom_signed = bottom_signed[order]
            reflection, transmission, absorption = rta_from_signed_transmission(top_signed, bottom_signed)
            closure = np.abs(reflection + transmission + absorption - 1.0)
            finite = bool(
                np.all(np.isfinite(wavelength))
                and np.all(np.isfinite(reflection))
                and np.all(np.isfinite(transmission))
                and np.all(np.isfinite(absorption))
            )
            np.savez_compressed(
                spectrum_path,
                wavelength_m=wavelength,
                frequency_hz=frequency[order],
                R=reflection,
                T=transmission,
                A=absorption,
                top_signed_transmission=top_signed,
                bottom_signed_transmission=bottom_signed,
            )
            audit_log = audit.log_audit(output)
            gates = {
                "GPU_completed": bool(audit_log["simulation_completed_successfully"]),
                "auto_shutoff_lt_1e_5": audit_log["final_auto_shutoff"] is not None
                and audit_log["final_auto_shutoff"] < 1.0e-5,
                "all_spectral_arrays_finite": finite,
                "algebraic_RTA_closure_lt_1e_12": float(np.max(closure)) < 1.0e-12,
                "no_large_negative_absorption": float(np.min(absorption)) > -0.01,
                "R_and_T_physical_tolerance": float(np.min(reflection)) > -0.01
                and float(np.min(transmission)) > -0.01,
            }
            peak = int(np.argmax(absorption))
            result.update(
                {
                    "log_audit": audit_log,
                    "gates": gates,
                    "peak_total_absorption": {
                        "wavelength_m": float(wavelength[peak]),
                        "A": float(absorption[peak]),
                        "R": float(reflection[peak]),
                        "T": float(transmission[peak]),
                    },
                    "spectral_extrema": {
                        "R_min_max": [float(np.min(reflection)), float(np.max(reflection))],
                        "T_min_max": [float(np.min(transmission)), float(np.max(transmission))],
                        "A_min_max": [float(np.min(absorption)), float(np.max(absorption))],
                        "max_algebraic_closure": float(np.max(closure)),
                    },
                }
            )
            result["status"] = (
                "COMPLETED_T2024_PERIODIC_BROADBAND_RTA"
                if all(gates.values())
                else "FAILED_T2024_PERIODIC_BROADBAND_RTA_GATE"
            )
            fdtd.save(str(fsp_path))
        result["raw_artifacts"] = [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": base.sha256(path)}
            for path in (fsp_path, spectrum_path)
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
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0 if str(result["status"]).startswith("COMPLETED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
