#!/usr/bin/env python3
"""GPU Maxwell forward smoke for the compact TaIrTe4 flake contract."""

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

from photothermal_pte.finite_inverse_design.native_yee_q import extract_native_yee_q
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import PABS_GROUP
from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology import optical


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
RUN002 = REPOSITORY / "photothermal_pte" / "optimization_runs" / "legacy_v261_optical_support"
STAGE1 = REPOSITORY / "photothermal_pte" / "validation" / "photothermal_stage1"
FIELD_REGION = "run010_component_yee_adjoint_region"
C0 = 299792458.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_fieldregion(fdtd) -> None:
    if int(fdtd.getnamednumber(FIELD_REGION)) == 1:
        return
    region = fdtd.addfieldregion()
    region["name"] = FIELD_REGION
    region["monitor type"] = "3D"
    for axis in "xyz":
        region[f"{axis} min"], region[f"{axis} max"] = optical.Q_BOUNDS[axis]
    region["source mode"] = False
    region["override global monitor settings"] = True
    region["use source limits"] = False
    region["use wavelength spacing"] = True
    region["wavelength center"] = CONTRACT.wavelength_m
    region["wavelength span"] = 0.0
    region["frequency points"] = 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-project", required=True, type=Path)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-device", default="GPU 5")
    parser.add_argument("--fdtd-threads", type=int, default=3)
    parser.add_argument("--polarization", choices=("a", "b"), required=True)
    parser.add_argument("--without-fieldregion", action="store_true")
    args = parser.parse_args()
    if args.fdtd_threads <= 0:
        parser.error("--fdtd-threads must be positive")
    CONTRACT.validate()
    source_project = args.input_project.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    project = output / f"tairte4_flake_forward_E{args.polarization}.fsp"
    npz_path = output / f"tairte4_flake_native_Q_E{args.polarization}.npz"
    result_path = output / f"tairte4_flake_forward_E{args.polarization}.json"
    result: dict[str, object] = {
        "status": "BLOCKED_TAIRTE4_FLAKE_FORWARD",
        "passed": False,
        "thermal_solve": False,
        "electrical_solve": False,
        "adjoint_solve": False,
        "optimization_run": False,
    }
    fdtd = None
    try:
        if not source_project.is_file():
            raise FileNotFoundError(source_project)
        actual_sha = sha256(source_project)
        if actual_sha != args.input_sha256:
            raise RuntimeError(f"input FSP SHA mismatch: {actual_sha}")
        for helper_path in (RUN002, STAGE1, REPOSITORY / "photothermal_pte"):
            helper_string = str(helper_path)
            if helper_string not in sys.path:
                sys.path.insert(0, helper_string)
        import run_complex_material_control as material_control

        wrapper = material_control.load_source_wrapper()
        audit = wrapper.source_audit
        optical.configure_source(audit)
        os.environ["VC_LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
        os.environ["LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
        os.environ["LUMERICAL_PYTHONPATH"] = str(audit.APPROVED_API)
        os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = args.gpu_device
        os.environ["CL_GPU_DEVICE"] = args.gpu_device
        os.environ["FDTD_THREADS"] = str(args.fdtd_threads)
        os.environ["PATH"] = f"{audit.APPROVED_ROOT / 'bin'}:{os.environ.get('PATH','')}"
        helper = audit.load_module(audit.API_HELPER, "run010_forward_api")
        installation = type(
            "Installation",
            (),
            {
                "version_key": "v261",
                "root": audit.APPROVED_ROOT,
                "lumapi_path": audit.APPROVED_API / "lumapi.py",
                "device_executable": audit.APPROVED_ROOT / "bin" / "device",
            },
        )()
        lumapi = helper.load_lumapi(installation)
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        # eqc_lib is registered on the embedded API path when the first
        # Lumerical session is created; importing it before FDTD() is an
        # initialization-order error in a clean process.
        import eqc_lib as runtime

        fdtd.load(str(source_project))
        fdtd.switchtolayout()
        if not args.without_fieldregion:
            add_fieldregion(fdtd)
        polarization_angle = 90.0 if args.polarization == "a" else 0.0
        fdtd.setnamed(optical.SOURCE_NAME, "enabled", True)
        fdtd.setnamed(optical.SOURCE_NAME, "polarization angle", polarization_angle)
        resources = runtime.configure_session_resources(fdtd)
        fdtd.save(str(project))
        started = time.monotonic()
        resource_used = audit.strict_gpu_run(
            fdtd, f"run010_uniform_rho0p5_E{args.polarization}"
        )
        wall_time = time.monotonic() - started
        fdtd.save(str(project))
        frequency = C0 / CONTRACT.wavelength_m
        source_power = audit.scalar(
            fdtd.sourcepower(frequency, 2, optical.SOURCE_NAME), "sourcepower"
        )
        face_power: dict[str, dict[str, float]] = {}
        net_outward = 0.0
        for axis in "xyz":
            for side, sign in (("min", -1.0), ("max", 1.0)):
                name = f"run010_flux_{axis}_{side}"
                signed = audit.scalar(fdtd.transmission(name), name) * source_power
                outward = sign * signed
                face_power[name] = {
                    "signed_axis_power_W": signed,
                    "outward_power_W": outward,
                }
                net_outward += outward
        p_six = -net_outward
        fdtd.runanalysis(PABS_GROUP)
        q = extract_native_yee_q(
            fdtd,
            field_monitor=optical.PABS_FIELD,
            index_monitor=optical.PABS_INDEX,
            wavelength_m=CONTRACT.wavelength_m,
        )
        p_q = float(q["P_Q_W"])
        closure = abs(p_q - p_six) / max(abs(p_six), np.finfo(float).tiny)
        arrays: dict[str, np.ndarray] = {}
        minimum = float("inf")
        maximum = -float("inf")
        finite = True
        for component in "xyz":
            values = np.asarray(q["Q_components"][component], dtype=np.float64)
            arrays[f"Q{component}_W_m3"] = values
            minimum = min(minimum, float(np.min(values)))
            maximum = max(maximum, float(np.max(values)))
            finite = finite and bool(np.all(np.isfinite(values)))
            for axis in "xyz":
                arrays[f"Q{component}_{axis}_m"] = np.asarray(
                    q["native_coordinates"][component][axis], dtype=np.float64
                )
        np.savez_compressed(npz_path, **arrays)
        log = audit.log_audit(output)
        passed = bool(
            closure < 0.005
            and log["final_auto_shutoff"] is not None
            and log["final_auto_shutoff"] < 1.0e-5
            and finite
            and minimum >= 0.0
            and p_q > 0.0
        )
        result = {
            "status": "VALIDATED_TAIRTE4_FLAKE_FORWARD" if passed else "FAILED_TAIRTE4_FLAKE_FORWARD",
            "passed": passed,
            "scope": "uniform rho=0.5 GPU Maxwell forward only",
            "axis_contract": "Lumerical x=b, y=a, z=c",
            "device_geometry": (
                "24 um x 24 um local import primitive rotated +45 degrees; "
                "two opposite full-edge 2 um terminal strips"
                if CONTRACT.geometry_mode == "diagonal_45_contact_anchored"
                else CONTRACT.geometry_mode
            ),
            "Au_electrodes": (
                {
                    "objects": list(optical.AU_OBJECT_NAMES),
                    "thickness_m": optical.AU_THICKNESS_M,
                    "n_at_10um": optical.AU_INDEX_AT_10UM.real,
                    "k_at_10um": optical.AU_INDEX_AT_10UM.imag,
                }
                if CONTRACT.geometry_mode == "diagonal_45_contact_anchored"
                else None
            ),
            "polarization": f"E_parallel_{args.polarization}",
            "polarization_angle_deg": polarization_angle,
            "input_project": {
                "path": str(source_project), "size_bytes": source_project.stat().st_size,
                "sha256": actual_sha,
            },
            "output_project": {
                "path": str(project), "size_bytes": project.stat().st_size,
                "sha256": sha256(project),
            },
            "native_Q_artifact": {
                "path": str(npz_path), "size_bytes": npz_path.stat().st_size,
                "sha256": sha256(npz_path),
            },
            "GPU_device_requested": args.gpu_device,
            "FDTD_host_threads_requested": args.fdtd_threads,
            "GPU_resource_used": resource_used,
            "resources": resources,
            "solver_wall_time_s": wall_time,
            "source_power_W": source_power,
            "P_Q_W": p_q,
            "P_six_W": p_six,
            "six_face_closure_relative": closure,
            "face_power": face_power,
            "Q_component_power_W": q["component_power_W"],
            "Q_minimum_W_m3": minimum,
            "Q_maximum_W_m3": maximum,
            "Q_all_finite": finite,
            "Q_clipping_smoothing_gain_or_rescaling": False,
            "fieldregion_saved": not args.without_fieldregion,
            "log_audit": log,
            "thermal_solve": False,
            "electrical_solve": False,
            "adjoint_solve": False,
            "optimization_run": False,
        }
    except Exception as exc:
        result.update({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
