#!/usr/bin/env python3
"""GPU forward gate for the matched-volume Run-002 production candidate."""

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
REPOSITORY = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.native_yee_q import (  # noqa: E402
    extract_native_yee_q,
)
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_GROUP,
)

import audit_production_candidate_geometry as geometry  # noqa: E402
import run_complex_material_control as material_control  # noqa: E402


C0 = 299792458.0
FREQUENCY_HZ = C0 / 10.0e-6
FIELD_REGION = "run002_component_yee_adjoint_region"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_fieldregion(fdtd: object) -> None:
    if int(fdtd.getnamednumber(FIELD_REGION)) == 1:
        return
    region = fdtd.addfieldregion()
    region["name"] = FIELD_REGION
    region["monitor type"] = "3D"
    for axis in "xyz":
        region[f"{axis} min"], region[f"{axis} max"] = geometry.Q_BOUNDS[axis]
    region["source mode"] = False
    region["override global monitor settings"] = True
    region["use source limits"] = False
    region["use wavelength spacing"] = True
    region["wavelength center"] = 10.0e-6
    region["wavelength span"] = 0.0
    region["frequency points"] = 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-project", required=True, type=Path)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-device", default="GPU 3")
    args = parser.parse_args()
    source_project = args.input_project.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    project = output / "production_candidate_forward.fsp"
    npz_path = output / "production_candidate_native_q.npz"
    result_path = output / "production_candidate_forward_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_RUN002_PRODUCTION_CANDIDATE_FORWARD",
        "thermal_solve": False,
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
        wrapper = material_control.load_source_wrapper()
        audit = wrapper.source_audit
        os.environ["VC_LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
        os.environ["LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
        os.environ["LUMERICAL_PYTHONPATH"] = str(audit.APPROVED_API)
        os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = args.gpu_device
        os.environ["CL_GPU_DEVICE"] = args.gpu_device
        os.environ["FDTD_THREADS"] = "8"
        os.environ["PATH"] = f"{audit.APPROVED_ROOT / 'bin'}:{os.environ.get('PATH','')}"
        for path in (audit.STAGE1, REPOSITORY / "photothermal_pte"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        helper = audit.load_module(audit.API_HELPER, "run002_production_forward_api")
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
        import eqc_lib as runtime

        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        fdtd.load(str(source_project))
        fdtd.switchtolayout()
        add_fieldregion(fdtd)
        fdtd.setnamed(audit.SOURCE_NAME, "enabled", True)
        resources = runtime.configure_session_resources(fdtd)
        fdtd.save(str(project))
        started = time.monotonic()
        resource_used = audit.strict_gpu_run(fdtd, "run002_production_candidate_rho0p5")
        wall_time = time.monotonic() - started
        fdtd.save(str(project))
        source_power = audit.scalar(
            fdtd.sourcepower(FREQUENCY_HZ, 2, audit.SOURCE_NAME), "sourcepower"
        )
        face_power = {}
        net_outward = 0.0
        for axis in "xyz":
            for side, sign in (("min", -1.0), ("max", 1.0)):
                name = f"run002_production_flux_{axis}_{side}"
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
            field_monitor=geometry.PABS_FIELD,
            index_monitor=geometry.PABS_INDEX,
            wavelength_m=10.0e-6,
        )
        p_q = float(q["P_Q_W"])
        closure = abs(p_q - p_six) / max(abs(p_q), abs(p_six), 1e-300)
        arrays = {}
        q_minimum = float("inf")
        q_maximum = -float("inf")
        all_finite = True
        for component in "xyz":
            values = np.asarray(q["Q_components"][component], float)
            arrays[f"Q{component}_W_m3"] = values
            q_minimum = min(q_minimum, float(np.min(values)))
            q_maximum = max(q_maximum, float(np.max(values)))
            all_finite = all_finite and bool(np.all(np.isfinite(values)))
            for axis in "xyz":
                arrays[f"Q{component}_{axis}_m"] = np.asarray(
                    q["native_coordinates"][component][axis], float
                )
        np.savez_compressed(npz_path, **arrays)
        log_audit = audit.log_audit(output)
        passed = bool(
            closure < 0.005
            and log_audit["final_auto_shutoff"] < 1e-5
            and all_finite
            and q_minimum >= 0.0
            and p_q > 0.0
        )
        result = {
            "status": "VALIDATED_RUN002_PRODUCTION_CANDIDATE_FORWARD" if passed else "FAILED_RUN002_PRODUCTION_CANDIDATE_FORWARD",
            "passed": passed,
            "scope": "rho=0.5 coarse production-candidate GPU forward; no thermal/PTE/adjoint/optimization",
            "input_project": {
                "path": str(source_project),
                "size_bytes": source_project.stat().st_size,
                "sha256": actual_sha,
            },
            "output_project": {
                "path": str(project),
                "size_bytes": project.stat().st_size,
                "sha256": sha256(project),
            },
            "native_Q_artifact": {
                "path": str(npz_path),
                "size_bytes": npz_path.stat().st_size,
                "sha256": sha256(npz_path),
            },
            "resources": resources,
            "GPU_resource_used": resource_used,
            "solver_wall_time_s": wall_time,
            "source_power_W": source_power,
            "P_Q_W": p_q,
            "P_six_W": p_six,
            "six_face_closure_relative": closure,
            "face_power": face_power,
            "Q_component_power_W": q["component_power_W"],
            "Q_minimum_W_m3": q_minimum,
            "Q_maximum_W_m3": q_maximum,
            "Q_all_finite": all_finite,
            "Q_clipping_smoothing_gain_or_rescaling": False,
            "log_audit": log_audit,
            "thermal_solve": False,
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
    return 0 if result.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
