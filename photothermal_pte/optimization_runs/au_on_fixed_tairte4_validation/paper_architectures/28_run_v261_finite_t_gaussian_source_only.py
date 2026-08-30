#!/usr/bin/env python3
"""GPU source-only gate for the finite multi-T scalar Gaussian contract."""

from __future__ import annotations

import argparse
import hashlib
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


WAVELENGTH_M = 11.825e-6
FREQUENCY_HZ = 299_792_458.0 / WAVELENGTH_M
W0_M = 4.0e-6
TARGET_W0_M = 4.0e-6
DOMAIN_X_M = 28.5e-6
DOMAIN_Y_M = 29.0e-6
ARRAY_X_M = 16.5e-6
ARRAY_Y_M = 17.0e-6
SOURCE_SPAN_M = 16.0e-6
SOURCE_Z_M = 0.8e-6
FOCUS_Z_M = 0.05e-6
MONITOR_NAME = "finite_T_Gaussian_target"
SOURCE_NAME = "finite_T_scalar_Gaussian"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup(fdtd) -> None:
    solver = fdtd.addfdtd()
    solver["dimension"] = "3D"
    solver["x min"], solver["x max"] = -0.5 * DOMAIN_X_M, 0.5 * DOMAIN_X_M
    solver["y min"], solver["y max"] = -0.5 * DOMAIN_Y_M, 0.5 * DOMAIN_Y_M
    solver["z min"], solver["z max"] = -1.2e-6, 1.2e-6
    for axis in "xyz":
        solver[f"{axis} min bc"] = "PML"
        solver[f"{axis} max bc"] = "PML"
    solver["pml layers"] = 24
    solver["mesh type"] = "auto non-uniform"
    solver["mesh refinement"] = "conformal variant 1"
    solver["mesh accuracy"] = 3
    solver["simulation time"] = 1.5e-12
    solver["auto shutoff min"] = 1.0e-6
    solver["override simulation bandwidth for mesh generation"] = True
    solver["mesh wavelength min"] = WAVELENGTH_M
    solver["mesh wavelength max"] = WAVELENGTH_M

    source = fdtd.addgaussian()
    source["name"] = SOURCE_NAME
    source["injection axis"] = "z"
    source["direction"] = "backward"
    source["polarization angle"] = 0.0
    source["source shape"] = "Gaussian"
    source["use scalar approximation"] = True
    source["beam parameters"] = "Waist size and position"
    source["waist radius w0"] = W0_M
    source["distance from waist"] = -(SOURCE_Z_M - FOCUS_Z_M)
    source["x min"], source["x max"] = -0.5 * SOURCE_SPAN_M, 0.5 * SOURCE_SPAN_M
    source["y min"], source["y max"] = -0.5 * SOURCE_SPAN_M, 0.5 * SOURCE_SPAN_M
    source["z"] = SOURCE_Z_M
    source["override global source settings"] = True
    source["wavelength start"] = 0.95 * WAVELENGTH_M
    source["wavelength stop"] = 12.0e-6

    outer = fdtd.addmesh()
    outer["name"] = "finite_T_outer_coarse_xy_mesh"
    outer["x min"], outer["x max"] = -0.5 * DOMAIN_X_M, 0.5 * DOMAIN_X_M
    outer["y min"], outer["y max"] = -0.5 * DOMAIN_Y_M, 0.5 * DOMAIN_Y_M
    outer["z min"], outer["z max"] = -0.30e-6, 0.20e-6
    outer["override x mesh"] = True
    outer["override y mesh"] = True
    outer["override z mesh"] = False
    outer["dx"] = 250e-9
    outer["dy"] = 250e-9

    local = fdtd.addmesh()
    local["name"] = "finite_T_array_local_mesh"
    local["x min"], local["x max"] = -0.5 * ARRAY_X_M, 0.5 * ARRAY_X_M
    local["y min"], local["y max"] = -0.5 * ARRAY_Y_M, 0.5 * ARRAY_Y_M
    local["z min"], local["z max"] = -0.30e-6, 0.20e-6
    local["override x mesh"] = True
    local["override y mesh"] = True
    local["override z mesh"] = True
    local["dx"], local["dy"], local["dz"] = 50e-9, 50e-9, 5e-9

    monitor = fdtd.addpower()
    monitor["name"] = MONITOR_NAME
    monitor["monitor type"] = "2D Z-normal"
    monitor["x min"], monitor["x max"] = -0.5 * SOURCE_SPAN_M, 0.5 * SOURCE_SPAN_M
    monitor["y min"], monitor["y max"] = -0.5 * SOURCE_SPAN_M, 0.5 * SOURCE_SPAN_M
    monitor["z"] = FOCUS_Z_M
    monitor["override global monitor settings"] = True
    monitor["use source limits"] = False
    monitor["use wavelength spacing"] = True
    monitor["wavelength center"] = WAVELENGTH_M
    monitor["wavelength span"] = 0.0
    monitor["frequency points"] = 1


def main() -> int:
    global W0_M, TARGET_W0_M, DOMAIN_X_M, DOMAIN_Y_M, ARRAY_X_M, ARRAY_Y_M, SOURCE_SPAN_M
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_finite_T_w0_4um_source_only"),
    )
    parser.add_argument("--gpu-device", default="GPU 5")
    parser.add_argument("--w0-um", type=float, default=4.0)
    parser.add_argument(
        "--target-w0-um",
        type=float,
        default=None,
        help="Physical target-plane waist; --w0-um is the Lumerical source-object input.",
    )
    parser.add_argument("--domain-x-um", type=float, default=28.5)
    parser.add_argument("--domain-y-um", type=float, default=29.0)
    parser.add_argument("--array-x-um", type=float, default=16.5)
    parser.add_argument("--array-y-um", type=float, default=17.0)
    parser.add_argument("--source-span-um", type=float, default=16.0)
    args = parser.parse_args()
    W0_M = args.w0_um * 1.0e-6
    TARGET_W0_M = (
        args.target_w0_um * 1.0e-6
        if args.target_w0_um is not None
        else W0_M
    )
    DOMAIN_X_M = args.domain_x_um * 1.0e-6
    DOMAIN_Y_M = args.domain_y_um * 1.0e-6
    ARRAY_X_M = args.array_x_um * 1.0e-6
    ARRAY_Y_M = args.array_y_um * 1.0e-6
    SOURCE_SPAN_M = args.source_span_um * 1.0e-6
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "FINITE_T_GAUSSIAN_SOURCE_ONLY.json"
    fsp_path = output / "finite_T_gaussian_source_only.fsp"
    npz_path = output / "finite_T_gaussian_source_only_fields.npz"
    result: dict[str, object] = {"status": "BLOCKED_FINITE_T_GAUSSIAN_SOURCE_ONLY"}
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
        setup(fdtd)
        fdtd.setresource("FDTD", 1, "active", 0)
        fdtd.setresource("FDTD", 2, "active", 1)
        fdtd.setresource("FDTD", 2, "processes", "1")
        fdtd.setresource("FDTD", 2, "threads", "8")
        fdtd.setresource("FDTD", 2, "device type", args.gpu_device)
        fdtd.setresource("FDTD", 2, "solver extra command line options", "-gpu")
        fdtd.runsetup()
        mesh = audit.mesh_readback(fdtd)
        fdtd.save(str(fsp_path))
        started = time.monotonic()
        resource = audit.strict_gpu_run(fdtd, "finite_T_Gaussian_source_only")
        wall_time = time.monotonic() - started
        source_power = audit.scalar(fdtd.sourcepower(FREQUENCY_HZ, 2, SOURCE_NAME), "sourcepower")
        x = np.asarray(fdtd.getdata(MONITOR_NAME, "x", 1), float).reshape(-1)
        y = np.asarray(fdtd.getdata(MONITOR_NAME, "y", 1), float).reshape(-1)
        fields = {
            component: np.asarray(fdtd.getdata(MONITOR_NAME, component, 1)).squeeze()
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy")
        }
        ex_down = 0.5 * (fields["Ex"] - audit.ETA0 * fields["Hy"])
        ey_down = 0.5 * (fields["Ey"] + audit.ETA0 * fields["Hx"])
        intensity_proxy = np.abs(ex_down) ** 2 + np.abs(ey_down) ** 2
        if intensity_proxy.shape != (x.size, y.size):
            raise RuntimeError(f"unexpected target-plane shape: {intensity_proxy.shape}, {(x.size, y.size)}")
        fit = audit.fit_gaussian(x, y, intensity_proxy)
        transmitted_fraction = float(np.real(np.asarray(fdtd.transmission(MONITOR_NAME)).reshape(-1)[0]))
        np.savez_compressed(
            npz_path,
            x_m=x,
            y_m=y,
            E2_V2_m2=intensity_proxy,
            Ex_down=ex_down,
            Ey_down=ey_down,
            **fields,
        )
        log = audit.log_audit(output)
        gates = {
            "GPU_completed": bool(log["simulation_completed_successfully"]),
            "auto_shutoff_lt_1e_5": log["final_auto_shutoff"] is not None and log["final_auto_shutoff"] < 1.0e-5,
            "waist_x_within_0p5pct": abs(fit["fitted_waist_x_m"] - TARGET_W0_M) / TARGET_W0_M < 0.005,
            "waist_y_within_0p5pct": abs(fit["fitted_waist_y_m"] - TARGET_W0_M) / TARGET_W0_M < 0.005,
            "Gaussian_fit_NRMSE_lt_0p5pct": fit["Gaussian_fit_NRMSE"] < 0.005,
            "ellipticity_lt_0p5pct": fit["fitted_xy_ellipticity"] < 0.005,
            "center_displacement_lt_50nm": float(np.hypot(fit["fitted_center_x_m"], fit["fitted_center_y_m"])) < 50e-9,
            "target_transmission_magnitude_within_0p5pct": abs(abs(transmitted_fraction) - 1.0) < 0.005,
        }
        result = {
            "status": "VALIDATED_FINITE_T_GAUSSIAN_SOURCE_ONLY" if all(gates.values()) else "FAILED_FINITE_T_GAUSSIAN_SOURCE_ONLY_GATE",
            "classification": "homogeneous-air source-only gate; not finite-array Q or PTE",
            "solver_version": str(fdtd.version()),
            "GPU_resource_used": resource,
            "solver_wall_time_s": wall_time,
            "source": {
                "wavelength_um": 11.825,
                "target_realized_w0_um": TARGET_W0_M * 1e6,
                "Lumerical_source_object_w0_um": W0_M * 1e6,
                "source_object_calibration_is_power_or_Q_rescaling": False,
                "span_um": args.source_span_um,
                "source_z_um": 0.8,
                "focus_z_um": 0.05,
                "polarization": "E||b",
                "field_comparator": "downward transverse E/H decomposition",
            },
            "domain": {"x_um": args.domain_x_um, "y_um": args.domain_y_um, "z_um": [-1.2, 1.2], "boundaries": "six PML, 24 layers"},
            "mesh_shape": [int(np.asarray(mesh["coordinate_arrays"][a]).size) for a in "xyz"],
            "source_power_W": source_power,
            "target_plane_transmitted_fraction": transmitted_fraction,
            "target_plane_fit": fit,
            "ideal_square_aperture_boundary_intensity_over_peak": float(np.exp(-2.0 * (0.5 * SOURCE_SPAN_M / TARGET_W0_M) ** 2)),
            "ideal_nearest_PML_intensity_over_peak": float(np.exp(-2.0 * (0.5 * DOMAIN_X_M / TARGET_W0_M) ** 2)),
            "log_audit": log,
            "gates": gates,
            "raw_artifacts": [
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in (fsp_path, npz_path)
            ],
        }
        fdtd.save(str(fsp_path))
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
    return 0 if str(result["status"]).startswith("VALIDATED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
