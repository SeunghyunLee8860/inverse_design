#!/usr/bin/env python3
"""GPU-only finite 187-inverse-T Gaussian volumetric-Q certificate."""

from __future__ import annotations

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

from photothermal_pte.finite_inverse_design.native_yee_q import (  # noqa: E402
    extract_native_yee_q,
)
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_FIELD,
    PABS_GROUP,
    PABS_INDEX,
)
from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    validate_paper_ir_source_only_gpu as audit,
)


WAVELENGTH_M = 11.825e-6
FREQUENCY_HZ = 299_792_458.0 / WAVELENGTH_M
TARGET_W0_UM = 12.0
SOURCE_OBJECT_W0_UM = 11.85757138436561
DOMAIN_UM = 60.0
SOURCE_SPAN_UM = 50.0
CONTROL_BOUNDS_M = {
    "x": (-27.0e-6, 27.0e-6),
    "y": (-27.0e-6, 27.0e-6),
    "z": (-0.85e-6, 0.45e-6),
}
RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "paper_tairte4_finite_187T_w12_Q_11p825um_Eb"
)
SOURCE_GATE = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "paper_tairte4_finite_T_target_w0_12um_calibrated_source_only/"
    "FINITE_T_GAUSSIAN_SOURCE_ONLY.json"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value: object, label: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise RuntimeError(f"{label} is not scalar: {array.shape}")
    result = float(np.real(array[0]))
    if not np.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def configure_single_frequency(fdtd: object, name: str) -> None:
    for prop, value in (
        ("override global monitor settings", True),
        ("use source limits", False),
        ("use wavelength spacing", True),
        ("wavelength center", WAVELENGTH_M),
        ("wavelength span", 0.0),
        ("frequency points", 1),
    ):
        try:
            fdtd.setnamed(name, prop, value)
        except Exception:
            pass


def add_flux_box(fdtd: object) -> dict[str, dict[str, object]]:
    faces: dict[str, dict[str, object]] = {}
    for axis in "xyz":
        for side, position in zip(("min", "max"), CONTROL_BOUNDS_M[axis]):
            key = f"{axis}_{side}"
            name = f"finite_187T_flux_{key}"
            monitor = fdtd.addpower()
            monitor["name"] = name
            monitor["monitor type"] = f"2D {axis.upper()}-normal"
            monitor[axis] = position
            for transverse in "xyz":
                if transverse != axis:
                    monitor[f"{transverse} min"] = CONTROL_BOUNDS_M[transverse][0]
                    monitor[f"{transverse} max"] = CONTROL_BOUNDS_M[transverse][1]
            configure_single_frequency(fdtd, name)
            faces[key] = {
                "name": name,
                "axis": axis,
                "side": side,
                "outward_sign": -1.0 if side == "min" else 1.0,
            }
    return faces


def face_fluxes(fdtd: object, faces: dict[str, dict[str, object]], source_power: float) -> dict[str, object]:
    values: dict[str, object] = {}
    net_outward = 0.0
    for key, face in faces.items():
        normalized = scalar(fdtd.transmission(str(face["name"])), str(face["name"]))
        signed_axis = normalized * source_power
        outward = float(face["outward_sign"]) * signed_axis
        values[key] = {
            "normalized_signed_axis_flux": normalized,
            "signed_axis_power_W": signed_axis,
            "outward_power_W": outward,
        }
        net_outward += outward
    return {
        "faces": values,
        "net_outward_power_W": net_outward,
        "net_inward_power_W": -net_outward,
    }


def main() -> int:
    output = RAW
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    fsp = output / "finite_187T_w12_Q.fsp"
    npz = output / "finite_187T_w12_Q.npz"
    result_path = output / "FINITE_187T_W12_Q.json"
    result: dict[str, object] = {"status": "BLOCKED_FINITE_187T_W12_Q"}
    fdtd = None
    try:
        source_gate = json.loads(SOURCE_GATE.read_text())
        if source_gate.get("status") != "VALIDATED_FINITE_T_GAUSSIAN_SOURCE_ONLY":
            raise RuntimeError("calibrated source-only gate is not validated")
        if abs(float(source_gate["source"]["target_realized_w0_um"]) - TARGET_W0_UM) > 1e-12:
            raise RuntimeError("source-only target waist mismatch")
        if abs(float(source_gate["source"]["Lumerical_source_object_w0_um"]) - SOURCE_OBJECT_W0_UM) > 1e-9:
            raise RuntimeError("source-object calibration mismatch")

        os.environ["VC_LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
        os.environ["LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
        os.environ["LUMERICAL_PYTHONPATH"] = str(audit.APPROVED_API)
        gpu = os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", "GPU 1")
        os.environ["CL_GPU_DEVICE"] = gpu
        os.environ["FDTD_THREADS"] = "8"
        os.environ["PATH"] = f"{audit.APPROVED_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
        sys.path.insert(0, str(audit.APPROVED_API))
        import lumapi

        runsetup = load_module(HERE / "27_audit_v261_finite_t_gaussian_runsetup.py", "finite_187T_runsetup")
        runsetup.W0_UM = SOURCE_OBJECT_W0_UM
        runsetup.DOMAIN_X_UM = DOMAIN_UM
        runsetup.DOMAIN_Y_UM = DOMAIN_UM
        runsetup.SOURCE_SPAN_UM = SOURCE_SPAN_UM

        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        contract = runsetup.setup(fdtd)
        pabs = fdtd.addobject("pabs_adv")
        pabs["name"] = PABS_GROUP
        for axis in "xyz":
            pabs[axis] = 0.5 * sum(CONTROL_BOUNDS_M[axis])
            pabs[f"{axis} span"] = CONTROL_BOUNDS_M[axis][1] - CONTROL_BOUNDS_M[axis][0]
        faces = add_flux_box(fdtd)
        for name in (PABS_FIELD, PABS_INDEX):
            configure_single_frequency(fdtd, name)

        fdtd.setresource("FDTD", 1, "active", 0)
        fdtd.setresource("FDTD", 2, "active", 1)
        fdtd.setresource("FDTD", 2, "processes", "1")
        fdtd.setresource("FDTD", 2, "threads", "8")
        fdtd.setresource("FDTD", 2, "device type", gpu)
        fdtd.setresource("FDTD", 2, "solver extra command line options", "-gpu")
        fdtd.runsetup()
        raw_mesh = audit.mesh_readback(fdtd)
        metrics = runsetup.local_mesh_metrics(raw_mesh)
        pre_gates = {
            "source_gate_validated": True,
            "finite_T_count_187": contract["array"]["count"] == 187,
            "local_dx_le_50nm": metrics["local_max_step_m"]["x"] <= 50e-9 + 1e-12,
            "local_dy_le_50nm": metrics["local_max_step_m"]["y"] <= 50e-9 + 1e-12,
            "local_dz_le_5nm": metrics["local_max_step_m"]["z"] <= 5e-9 + 1e-12,
            "control_box_inside_PML": max(abs(v) for v in CONTROL_BOUNDS_M["x"] + CONTROL_BOUNDS_M["y"]) < 0.5 * DOMAIN_UM * 1e-6,
        }
        if not all(pre_gates.values()):
            raise RuntimeError(f"pre-run gates failed: {pre_gates}")
        fdtd.save(str(fsp))
        started = time.monotonic()
        resource = audit.strict_gpu_run(fdtd, "finite_187T_w12_Q")
        wall_time = time.monotonic() - started
        source_power = scalar(fdtd.sourcepower(FREQUENCY_HZ, 2, "finite_T_scalar_Gaussian"), "sourcepower")
        six_face = face_fluxes(fdtd, faces, source_power)
        fdtd.runanalysis(PABS_GROUP)
        q = extract_native_yee_q(
            fdtd,
            field_monitor=PABS_FIELD,
            index_monitor=PABS_INDEX,
            wavelength_m=WAVELENGTH_M,
        )
        common_module = load_module(
            REPOSITORY / "photothermal_pte/validation/photothermal_stage1/27_validate_finite_2um_optical_q.py",
            "finite_187T_common_q",
        )
        common = common_module.common_grid_component_q(fdtd, FREQUENCY_HZ)
        p_native = float(q["P_Q_W"])
        p_pabs = scalar(fdtd.getresult(PABS_GROUP, "Pabs_total")["Pabs_total"], "Pabs_total") * source_power
        p_six = float(six_face["net_inward_power_W"])
        closure = abs(p_native - p_six) / max(abs(p_six), np.finfo(float).tiny)
        pabs_delta = abs(p_native - p_pabs) / max(abs(p_pabs), np.finfo(float).tiny)
        negative = {
            component: int(np.count_nonzero(np.asarray(q["Q_components"][component]) < 0.0))
            for component in "xyz"
        }
        finite = all(np.all(np.isfinite(np.asarray(q["Q_components"][component]))) for component in "xyz")
        q_total_common = np.asarray(common["Q_native_W_m3"], float)
        hotspot_index = np.unravel_index(int(np.argmax(q_total_common)), q_total_common.shape)
        hotspot = {
            "x_m": float(common["x_m"][hotspot_index[0]]),
            "y_m": float(common["y_m"][hotspot_index[1]]),
            "z_m": float(common["z_m"][hotspot_index[2]]),
            "Q_W_m3": float(q_total_common[hotspot_index]),
            "classification": "linearly collocated common-grid diagnostic; power gate uses native Yee components",
        }
        arrays: dict[str, np.ndarray] = {
            "common_x_m": np.asarray(common["x_m"]),
            "common_y_m": np.asarray(common["y_m"]),
            "common_z_m": np.asarray(common["z_m"]),
            "Q_common_W_m3": q_total_common,
        }
        for component in "xyz":
            arrays[f"Q{component}_W_m3"] = np.asarray(q["Q_components"][component])
            for axis in "xyz":
                arrays[f"Q{component}_{axis}_m"] = np.asarray(q["native_coordinates"][component][axis])
        np.savez_compressed(npz, **arrays)
        log = audit.log_audit(output)
        gates = {
            **pre_gates,
            "GPU_completed": bool(log["simulation_completed_successfully"]),
            "auto_shutoff_lt_1e_5": log["final_auto_shutoff"] is not None and log["final_auto_shutoff"] < 1e-5,
            "six_face_closure_lt_0p5pct": closure < 0.005,
            "native_vs_pabs_lt_0p5pct": pabs_delta < 0.005,
            "all_Q_arrays_finite": bool(finite),
            "no_negative_Q": sum(negative.values()) == 0,
        }
        result = {
            "status": "VALIDATED_FINITE_187T_W12_VOLUMETRIC_Q" if all(gates.values()) else "FAILED_FINITE_187T_W12_VOLUMETRIC_Q_GATE",
            "classification": "finite nonperiodic 187 inverse-T Gaussian Maxwell/Q certificate; not thermal or PTE",
            "solver_version": str(fdtd.version()),
            "GPU_resource_used": resource,
            "solver_wall_time_s": wall_time,
            "contract": contract,
            "source": {
                "wavelength_um": 11.825,
                "polarization": "E||b; Lumerical x=b",
                "target_realized_w0_um": TARGET_W0_UM,
                "Lumerical_source_object_w0_um": SOURCE_OBJECT_W0_UM,
                "span_um": SOURCE_SPAN_UM,
                "source_gate_path": str(SOURCE_GATE),
                "source_gate_sha256": sha256(SOURCE_GATE),
            },
            "control_volume_bounds_m": CONTROL_BOUNDS_M,
            "mesh_runsetup": metrics,
            "source_power_W": source_power,
            "P_Q_native_W": p_native,
            "P_Q_pabs_W": p_pabs,
            "P_six_face_W": p_six,
            "six_face_closure_relative": closure,
            "native_vs_pabs_relative": pabs_delta,
            "Q_component_power_native_W": q["component_power_W"],
            "common_grid_component_power_W": common["common_component_power_W"],
            "common_grid_interpolation_relative_error": common["component_interpolation_relative_error"],
            "hotspot": hotspot,
            "negative_Q_cell_count": negative,
            "all_Q_arrays_finite": bool(finite),
            "six_face": six_face,
            "log_audit": log,
            "gates": gates,
            "scope_exclusions": ["thermal", "weighting potential", "PTE", "adjoint", "optimization"],
            "raw_artifacts": [],
        }
        fdtd.save(str(fsp))
        result["raw_artifacts"] = [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (fsp, npz)
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
    return 0 if str(result["status"]).startswith("VALIDATED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
