#!/usr/bin/env python3
"""GPU-only finite T/Z Gaussian volumetric-Q certificate.

Eight immutable cases are supported through environment variables:

    architecture T/Z x polarization Ea/Eb x top-Au on/off.

The source parameters are inherited from the validated source-only gate.  The
matched six-face/Q control volume lies below the source plane and inside every
PML.  No thermal, electrical, PTE, adjoint, or optimization solve is performed
here.  Raw FSP/NPZ files are written outside Git and are never rescaled.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

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


C0 = 299_792_458.0
CONTRACT_JSON = (
    HERE
    / "results_finite_T_Z_thermal_electrical_contract"
    / "FINITE_T_Z_THERMAL_ELECTRICAL_CONTRACT.json"
)
SOURCE_SUMMARY = (
    HERE
    / "results_finite_T_Z_gaussian_source_only"
    / "FINITE_T_Z_GAUSSIAN_SOURCE_ONLY_SUMMARY.json"
)
SOURCE_MODULE = HERE / "76_run_v261_finite_t_z_gaussian_source_only.py"
BASE_MODULE = HERE / "07_run_v261_t2024_tairte4_optical_smoke.py"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(value: object, label: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise RuntimeError(f"{label} is not scalar: {array.shape}")
    result = float(np.real(array[0]))
    if not np.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def _case() -> tuple[str, str, bool]:
    architecture = os.environ.get("FINITE_Q_ARCHITECTURE", "T").strip().upper()
    polarization = os.environ.get("FINITE_Q_POLARIZATION", "Ea").strip()
    au_token = os.environ.get("FINITE_Q_TOP_AU", "on").strip().lower()
    if architecture not in {"T", "Z"}:
        raise ValueError("FINITE_Q_ARCHITECTURE must be T or Z")
    if polarization not in {"Ea", "Eb"}:
        raise ValueError("FINITE_Q_POLARIZATION must be Ea or Eb")
    if au_token not in {"on", "off"}:
        raise ValueError("FINITE_Q_TOP_AU must be on or off")
    return architecture, polarization, au_token == "on"


def _configure_frequency(item: Any, wavelength_m: float) -> None:
    item["override global monitor settings"] = True
    item["use source limits"] = False
    item["use wavelength spacing"] = True
    item["wavelength center"] = wavelength_m
    item["wavelength span"] = 0.0
    item["frequency points"] = 1


def _control_bounds(architecture: str) -> dict[str, tuple[float, float]]:
    return {
        "x": (-10.5e-6, 10.5e-6),
        "y": (-10.5e-6, 10.5e-6),
        # Source is at +0.50 um.  The upper face is deliberately below it.
        "z": (-2.20e-6, 0.30e-6) if architecture == "T" else (-1.50e-6, 0.30e-6),
    }


def _add_flux_box(
    fdtd: Any, bounds: dict[str, tuple[float, float]], wavelength_m: float
) -> dict[str, dict[str, Any]]:
    faces: dict[str, dict[str, Any]] = {}
    for axis in "xyz":
        for side, position in zip(("min", "max"), bounds[axis], strict=True):
            name = f"finite_Q_flux_{axis}_{side}"
            monitor = fdtd.addpower()
            monitor["name"] = name
            monitor["monitor type"] = f"2D {axis.upper()}-normal"
            monitor[axis] = position
            for transverse in "xyz":
                if transverse != axis:
                    monitor[f"{transverse} min"] = bounds[transverse][0]
                    monitor[f"{transverse} max"] = bounds[transverse][1]
            _configure_frequency(monitor, wavelength_m)
            faces[f"{axis}_{side}"] = {
                "name": name,
                "axis": axis,
                "side": side,
                "outward_sign": -1.0 if side == "min" else 1.0,
            }
    return faces


def _face_fluxes(
    fdtd: Any, faces: dict[str, dict[str, Any]], source_power_w: float
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    net_outward = 0.0
    for key, face in faces.items():
        normalized = _scalar(fdtd.transmission(face["name"]), face["name"])
        axis_power = normalized * source_power_w
        outward = float(face["outward_sign"]) * axis_power
        result[key] = {
            "normalized_signed_axis_flux": normalized,
            "signed_axis_power_W": axis_power,
            "outward_power_W": outward,
        }
        net_outward += outward
    return {
        "faces": result,
        "net_outward_power_W": net_outward,
        "net_inward_power_W": -net_outward,
    }


def _add_rect(
    fdtd: Any,
    name: str,
    material: str,
    z_min: float,
    z_max: float,
    *,
    x_bounds: tuple[float, float] = (-12.0e-6, 12.0e-6),
    y_bounds: tuple[float, float] = (-12.0e-6, 12.0e-6),
) -> None:
    rectangle = fdtd.addrect()
    rectangle["name"] = name
    rectangle["material"] = material
    rectangle["x min"], rectangle["x max"] = x_bounds
    rectangle["y min"], rectangle["y max"] = y_bounds
    rectangle["z min"], rectangle["z max"] = z_min, z_max


def _add_geometry(
    fdtd: Any,
    architecture: str,
    include_top_au: bool,
    contract: dict[str, Any],
    base: Any,
) -> dict[str, Any]:
    arch = contract["architectures"][architecture]
    wavelength_m = float(arch["wavelength_um"]) * 1.0e-6
    base.configure_wavelength(float(arch["wavelength_um"]))
    tairte4 = base.add_tairte4_material(fdtd)
    base.add_constant_nk(fdtd, base.AL2O3_MATERIAL, base.AL2O3_N)

    al_m = float(arch["Al2O3_thickness_nm"]) * 1.0e-9
    mirror_m = float(arch["Au_mirror_thickness_nm"]) * 1.0e-9
    oxide_m = float(arch["SiO2_thickness_nm"]) * 1.0e-9
    mirror_top = -al_m
    mirror_bottom = mirror_top - mirror_m
    oxide_bottom = mirror_bottom - oxide_m
    z_min = float(arch["optical_z_bounds_um"][0]) * 1.0e-6

    _add_rect(fdtd, f"finite_{architecture}_Si", base.SI_MATERIAL, z_min, oxide_bottom)
    _add_rect(fdtd, f"finite_{architecture}_SiO2", base.SIO2_MATERIAL, oxide_bottom, mirror_bottom)
    _add_rect(fdtd, f"finite_{architecture}_Au_mirror", base.AU_MATERIAL, mirror_bottom, mirror_top)
    _add_rect(fdtd, f"finite_{architecture}_Al2O3", base.AL2O3_MATERIAL, mirror_top, 0.0)
    _add_rect(
        fdtd,
        f"finite_{architecture}_TaIrTe4_flake",
        base.TAIRTE4_MATERIAL,
        0.0,
        100.0e-9,
        x_bounds=(-10.0e-6, 10.0e-6),
        y_bounds=(-10.0e-6, 10.0e-6),
    )
    top_names: list[str] = []
    if include_top_au:
        for item in arch["geometry"]["polygons"]:
            polygon = fdtd.addpoly()
            polygon["name"] = f"finite_{architecture}_{item['name']}"
            polygon["material"] = base.AU_MATERIAL
            polygon["vertices"] = np.asarray(item["vertices_nm"], float) * 1.0e-9
            polygon["z min"] = float(item["z_min_nm"]) * 1.0e-9
            polygon["z max"] = float(item["z_max_nm"]) * 1.0e-9
            top_names.append(str(polygon["name"]))
    return {
        "wavelength_m": wavelength_m,
        "TaIrTe4": tairte4,
        "layers_z_m": {
            "Si": [z_min, oxide_bottom],
            "SiO2": [oxide_bottom, mirror_bottom],
            "Au_mirror": [mirror_bottom, mirror_top],
            "Al2O3": [mirror_top, 0.0],
            "TaIrTe4": [0.0, 100.0e-9],
        },
        "top_Au_present": include_top_au,
        "top_Au_objects": top_names,
        "top_Au_polygons": arch["geometry"]["polygons"] if include_top_au else [],
    }


def main() -> int:
    architecture, polarization, include_top_au = _case()
    au_label = "Au_on" if include_top_au else "Au_off"
    case_label = f"{architecture}_{polarization}_{au_label}"
    output = Path(
        os.environ.get(
            "FINITE_Q_OUTPUT",
            f"/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_Q/{case_label}",
        )
    ).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    fsp = output / f"finite_{case_label}.fsp"
    npz = output / f"finite_{case_label}_Q.npz"
    result_path = output / f"FINITE_{case_label}_Q.json"
    for protected in (fsp, npz, result_path):
        if protected.exists():
            raise RuntimeError(f"refusing to overwrite: {protected}")

    result: dict[str, Any] = {"status": f"BLOCKED_FINITE_{case_label}_Q"}
    fdtd = None
    try:
        source_summary = json.loads(SOURCE_SUMMARY.read_text())
        if source_summary.get("status") != "VALIDATED_FINITE_T_Z_GAUSSIAN_SOURCE_ONLY":
            raise RuntimeError("finite T/Z source-only gate is not validated")
        source_case = source_summary["validated_cases"][architecture]
        if not str(source_case["status"]).startswith("VALIDATED"):
            raise RuntimeError(f"source-only architecture gate failed: {architecture}")

        contract = json.loads(CONTRACT_JSON.read_text())
        source_module = _load(SOURCE_MODULE, "finite_q_source_contract")
        base = _load(BASE_MODULE, "finite_q_material_helpers")

        os.environ["VC_LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
        os.environ["LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
        os.environ["LUMERICAL_PYTHONPATH"] = str(audit.APPROVED_API)
        os.environ["PATH"] = f"{audit.APPROVED_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
        sys.path.insert(0, str(audit.APPROVED_API))
        import lumapi

        gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        gpu_device = f"GPU {gpu_index}"
        os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = gpu_device
        os.environ["CL_GPU_DEVICE"] = gpu_device
        os.environ["FDTD_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8")

        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        setup = source_module._setup(fdtd, architecture, contract)
        geometry = _add_geometry(fdtd, architecture, include_top_au, contract, base)
        wavelength_m = float(geometry["wavelength_m"])
        frequency_hz = C0 / wavelength_m
        source_name = source_module.SOURCE_NAME
        fdtd.setnamed(source_name, "polarization angle", 90.0 if polarization == "Ea" else 0.0)
        bounds = _control_bounds(architecture)

        pabs = fdtd.addobject("pabs_adv")
        pabs["name"] = PABS_GROUP
        for axis in "xyz":
            pabs[axis] = 0.5 * sum(bounds[axis])
            pabs[f"{axis} span"] = bounds[axis][1] - bounds[axis][0]
        for name in (PABS_FIELD, PABS_INDEX):
            try:
                fdtd.setnamed(name, "override global monitor settings", True)
                fdtd.setnamed(name, "use source limits", False)
                fdtd.setnamed(name, "use wavelength spacing", True)
                fdtd.setnamed(name, "wavelength center", wavelength_m)
                fdtd.setnamed(name, "wavelength span", 0.0)
                fdtd.setnamed(name, "frequency points", 1)
            except Exception:
                pass
        faces = _add_flux_box(fdtd, bounds, wavelength_m)

        fdtd.setresource("FDTD", 1, "active", 0)
        fdtd.setresource("FDTD", 2, "active", 1)
        fdtd.setresource("FDTD", 2, "processes", "1")
        fdtd.setresource("FDTD", 2, "threads", os.environ["FDTD_THREADS"])
        fdtd.setresource("FDTD", 2, "device type", gpu_device)
        fdtd.setresource("FDTD", 2, "solver extra command line options", "-gpu")

        fdtd.runsetup()
        mesh = audit.mesh_readback(fdtd)
        if not mesh.get("available"):
            raise RuntimeError(f"mesh readback unavailable: {mesh}")
        coordinates = mesh["coordinate_arrays"]
        mesh_audit = {
            "flake_max_dx_m": source_module._interval_max_step(coordinates["x"], -10e-6, 10e-6),
            "flake_max_dy_m": source_module._interval_max_step(coordinates["y"], -10e-6, 10e-6),
            "structure_max_dz_m": source_module._interval_max_step(coordinates["z"], -50e-9, 225e-9),
        }
        pre_gates = {
            "validated_source_gate": True,
            "source_plane_outside_control_volume": float(setup["source_bounds_m"]["z"]) > bounds["z"][1],
            "control_volume_inside_PML": all(
                bounds[axis][0] > setup["domain_bounds_m"][axis][0]
                and bounds[axis][1] < setup["domain_bounds_m"][axis][1]
                for axis in "xyz"
            ),
            "flake_dx_le_100nm": mesh_audit["flake_max_dx_m"] <= 100e-9 + 1e-12,
            "flake_dy_le_100nm": mesh_audit["flake_max_dy_m"] <= 100e-9 + 1e-12,
            "structure_dz_le_5nm": mesh_audit["structure_max_dz_m"] <= 5e-9 + 1e-12,
        }
        if not all(pre_gates.values()):
            raise RuntimeError(f"pre-run gate failed: {pre_gates}")
        fdtd.save(str(fsp))

        started = time.monotonic()
        resource = audit.strict_gpu_run(fdtd, f"finite_{case_label}_Q")
        wall_time = time.monotonic() - started
        fdtd.save(str(fsp))

        source_power = _scalar(fdtd.sourcepower(frequency_hz, 2, source_name), "sourcepower")
        six_face = _face_fluxes(fdtd, faces, source_power)
        fdtd.runanalysis(PABS_GROUP)
        q = extract_native_yee_q(
            fdtd,
            field_monitor=PABS_FIELD,
            index_monitor=PABS_INDEX,
            wavelength_m=wavelength_m,
        )
        stage1 = REPOSITORY / "photothermal_pte/validation/photothermal_stage1"
        if str(stage1) not in sys.path:
            sys.path.insert(0, str(stage1))
        common_module = _load(
            stage1 / "27_validate_finite_2um_optical_q.py",
            f"finite_{case_label}_common_q",
        )
        common_module.PABS_FIELD = PABS_FIELD
        common_module.PABS_INDEX = PABS_INDEX
        common = common_module.common_grid_component_q(fdtd, frequency_hz)

        p_native = float(q["P_Q_W"])
        p_pabs = _scalar(fdtd.getresult(PABS_GROUP, "Pabs_total")["Pabs_total"], "Pabs_total") * source_power
        p_six = float(six_face["net_inward_power_W"])
        closure = abs(p_native - p_six) / max(abs(p_six), np.finfo(float).tiny)
        native_pabs_error = abs(p_native - p_pabs) / max(abs(p_pabs), np.finfo(float).tiny)
        negative = {
            component: int(np.count_nonzero(np.asarray(q["Q_components"][component]) < 0.0))
            for component in "xyz"
        }
        finite = all(
            np.all(np.isfinite(np.asarray(q["Q_components"][component])))
            for component in "xyz"
        )
        q_common = np.asarray(common["Q_native_W_m3"], float)
        hotspot_index = np.unravel_index(int(np.argmax(q_common)), q_common.shape)
        hotspot = {
            "x_m": float(common["x_m"][hotspot_index[0]]),
            "y_m": float(common["y_m"][hotspot_index[1]]),
            "z_m": float(common["z_m"][hotspot_index[2]]),
            "Q_W_m3": float(q_common[hotspot_index]),
        }
        arrays: dict[str, np.ndarray] = {
            "common_x_m": np.asarray(common["x_m"]),
            "common_y_m": np.asarray(common["y_m"]),
            "common_z_m": np.asarray(common["z_m"]),
            "Q_common_W_m3": q_common,
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
            "native_vs_pabs_lt_0p5pct": native_pabs_error < 0.005,
            "all_Q_arrays_finite": bool(finite),
            "no_negative_Q": sum(negative.values()) == 0,
        }
        result = {
            "status": f"VALIDATED_FINITE_{case_label}_VOLUMETRIC_Q" if all(gates.values()) else f"FAILED_FINITE_{case_label}_VOLUMETRIC_Q_GATE",
            "classification": "finite nonperiodic Maxwell/Q certificate; no thermal/electrical/PTE/adjoint/optimization",
            "solver_version": str(fdtd.version()),
            "GPU_resource_used": resource,
            "solver_wall_time_s": wall_time,
            "architecture": architecture,
            "polarization": polarization,
            "axis_mapping": {"x": "b", "y": "a", "z": "c=b optical closure"},
            "top_Au_present": include_top_au,
            "source_gate_path": str(SOURCE_SUMMARY),
            "source_gate_sha256": _sha256(SOURCE_SUMMARY),
            "setup": setup,
            "geometry": geometry,
            "control_volume_bounds_m": bounds,
            "mesh_region_audit": mesh_audit,
            "source_power_W": source_power,
            "P_Q_native_W": p_native,
            "P_Q_pabs_W": p_pabs,
            "P_six_face_W": p_six,
            "six_face_closure_relative": closure,
            "native_vs_pabs_relative": native_pabs_error,
            "Q_component_power_native_W": q["component_power_W"],
            "common_grid_component_power_W": common["common_component_power_W"],
            "common_grid_interpolation_relative_error": common["component_interpolation_relative_error"],
            "hotspot": hotspot,
            "negative_Q_cell_count": negative,
            "all_Q_arrays_finite": bool(finite),
            "six_face": six_face,
            "log_audit": log,
            "gates": gates,
            "Q_processing": {
                "clipping": False,
                "smoothing": False,
                "gain": False,
                "global_rescaling": False,
                "tiling": False,
            },
            "raw_artifacts": [],
        }
        fdtd.save(str(fsp))
        result["raw_artifacts"] = [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
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
        _write_json(result_path, result)
    print(json.dumps(result, indent=2, default=str))
    return 0 if str(result["status"]).startswith("VALIDATED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
