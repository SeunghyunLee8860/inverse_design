#!/usr/bin/env python3
"""Recover Q/readback from an already-solved smooth-ellipsoid FSP.

This narrowly handles a post-solve Python callback failure.  It never calls
``run`` or ``runsetup`` and leaves the original ``case_result.json`` intact as
failure provenance.  Recovered metadata is written to a separate JSON file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.native_yee_q import (
    extract_native_yee_q,
    frequency_slice,
)
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (
    PABS_FIELD,
    PABS_GROUP,
    PABS_INDEX,
)


STAGE12 = HERE / "12_run_au_sharp_interface_external_field_adjoint.py"
WAVELENGTH_M = 10.0e-6
FREQUENCY_HZ = 299792458.0 / WAVELENGTH_M
def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage12 = load("au_ellipsoid_recovery_stage12", STAGE12)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ellipsoid_readback(fdtd, q: dict[str, object], material: dict) -> dict:
    a, b, c = map(float, material["ellipsoid_semi_axes_m"])
    center_z = float(material["center_z_m"])
    spatial_shape = tuple(np.asarray(q["base_coordinates"][axis]).size for axis in "xyz")
    result = {}
    for component in "xyz":
        index = frequency_slice(
            np.asarray(fdtd.getdata(PABS_INDEX, f"index_{component}", 1)),
            spatial_shape,
            int(q["frequency_index_zero_based"]),
            int(q["frequency_count"]),
            f"index_{component}",
        )
        epsilon = index**2
        x = np.asarray(q["native_coordinates"][component]["x"], float)
        y = np.asarray(q["native_coordinates"][component]["y"], float)
        z = np.asarray(q["native_coordinates"][component]["z"], float)
        radius = (
            (x[:, None, None] / a) ** 2
            + (y[None, :, None] / b) ** 2
            + ((z[None, None, :] - center_z) / c) ** 2
        )
        interior = radius <= 0.80**2
        values = epsilon[interior & np.isfinite(epsilon)]
        if values.size == 0:
            raise RuntimeError(f"empty ellipsoid interior readback for {component}")
        result[component] = {
            "shape": list(epsilon.shape),
            "epsilon_interior_median": [
                float(np.median(values.real)),
                float(np.median(values.imag)),
            ],
            "interior_sample_count": int(values.size),
            "all_finite": bool(np.all(np.isfinite(epsilon))),
            "interior_normalized_radius_max": 0.80,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 0")
    args = parser.parse_args()
    case_dir = args.case_dir.expanduser().resolve()
    original_path = case_dir / "case_result.json"
    project_path = case_dir / "complex_material_control.fsp"
    result_path = case_dir / "case_result_recovered.json"
    npz_path = case_dir / "complex_material_control_q.npz"
    if result_path.exists() or npz_path.exists():
        raise RuntimeError("refusing to overwrite recovered ellipsoid artifacts")
    original = json.loads(original_path.read_text())
    result: dict[str, object] = {
        "status": "BLOCKED_ELLIPSOID_FORWARD_POSTPROCESSING_RECOVERY",
        "passed": False,
        "Maxwell_forward_solves_this_invocation": 0,
        "Maxwell_adjoint_solves_this_invocation": 0,
        "runsetup_calls_this_invocation": 0,
        "CPU_FDTD_fallback": False,
        "original_failure_provenance": {
            "path": str(original_path),
            "sha256": sha256(original_path),
            "status": original.get("status"),
            "error": original.get("error"),
        },
    }
    fdtd = None
    try:
        if "ellipsoid" not in str(original.get("material", {}).get("representation", "")):
            raise RuntimeError("input is not the smooth-ellipsoid control")
        fdtd, audit, _runtime = stage12.open_fdtd(args.gpu_device)
        fdtd.load(str(project_path))
        source_power = audit.scalar(
            fdtd.sourcepower(FREQUENCY_HZ, 2, audit.SOURCE_NAME), "sourcepower"
        )
        face_power = {}
        net_outward = 0.0
        for axis in "xyz":
            for side in ("min", "max"):
                name = f"run002_flux_{axis}_{side}"
                sign = -1.0 if side == "min" else 1.0
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
            field_monitor=PABS_FIELD,
            index_monitor=PABS_INDEX,
            wavelength_m=WAVELENGTH_M,
        )
        p_q = float(q["P_Q_W"])
        closure = abs(p_q - p_six) / max(abs(p_q), abs(p_six), np.finfo(float).tiny)
        readback = ellipsoid_readback(fdtd, q, original["material"])
        arrays = {}
        for component in "xyz":
            arrays[f"Q{component}_W_m3"] = q["Q_components"][component]
            for axis in "xyz":
                arrays[f"Q{component}_{axis}_m"] = q["native_coordinates"][component][axis]
        np.savez_compressed(npz_path, **arrays)
        finite = all(row["all_finite"] for row in readback.values())
        passed = bool(finite and closure < 0.005)
        result.update(original)
        result.pop("error", None)
        result.pop("traceback", None)
        result.update(
            {
                "status": (
                    "COMPLETED_COMPLEX_MATERIAL_FORWARD_CONTROL_RECOVERED"
                    if passed
                    else "FAILED_COMPLEX_MATERIAL_FORWARD_CONTROL_RECOVERED"
                ),
                "passed": passed,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "postprocessing_recovered_without_Maxwell_rerun": True,
                "Maxwell_forward_solves_this_invocation": 0,
                "Maxwell_adjoint_solves_this_invocation": 0,
                "runsetup_calls_this_invocation": 0,
                "source_power_W": source_power,
                "P_Q_W": p_q,
                "P_six_W": p_six,
                "six_face_closure_relative": closure,
                "face_power": face_power,
                "Q_component_power_W": q["component_power_W"],
                "epsilon_component_readback": readback,
                "raw_artifacts": [
                    {
                        "path": str(project_path),
                        "size_bytes": project_path.stat().st_size,
                        "sha256": sha256(project_path),
                    },
                    {
                        "path": str(npz_path),
                        "size_bytes": npz_path.stat().st_size,
                        "sha256": sha256(npz_path),
                    },
                ],
            }
        )
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
    return 0 if bool(result.get("passed", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
