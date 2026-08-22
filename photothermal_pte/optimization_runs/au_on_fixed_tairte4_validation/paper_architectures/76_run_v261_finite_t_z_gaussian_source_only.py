#!/usr/bin/env python3
"""GPU-only source gate for the finite T/Z Gaussian Maxwell contract.

The architecture and output directory are supplied through environment
variables so this file can be launched by the site ``runres`` wrapper, which
accepts a Python script but no script-specific command-line arguments.

There is no material, heat source, thermal solve, electrical solve, adjoint or
optimization in this program.  CPU FDTD fallback is prohibited.
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

from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    validate_paper_ir_source_only_gpu as audit,
)


C0 = 299_792_458.0
SOURCE_NAME = "finite_T_Z_scalar_Gaussian"
TARGET_MONITOR = "finite_T_Z_target_plane"
CONTRACT_JSON = (
    HERE
    / "results_finite_T_Z_thermal_electrical_contract"
    / "FINITE_T_Z_THERMAL_ELECTRICAL_CONTRACT.json"
)
GEOMETRY_FILE = HERE / "05_actual_metasurface_geometry.py"


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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _architecture() -> tuple[str, dict[str, Any]]:
    key = os.environ.get("FINITE_SOURCE_ARCHITECTURE", "T").strip().upper()
    if key not in {"T", "Z"}:
        raise ValueError("FINITE_SOURCE_ARCHITECTURE must be T or Z")
    contract = json.loads(CONTRACT_JSON.read_text(encoding="utf-8"))
    return key, contract


def _add_single_frequency_monitor(
    fdtd: Any, name: str, z_m: float, wavelength_m: float
) -> None:
    monitor = fdtd.addpower()
    monitor["name"] = name
    monitor["monitor type"] = "2D Z-normal"
    monitor["x min"], monitor["x max"] = -9.0e-6, 9.0e-6
    monitor["y min"], monitor["y max"] = -9.0e-6, 9.0e-6
    monitor["z"] = z_m
    monitor["override global monitor settings"] = True
    monitor["use source limits"] = False
    monitor["use wavelength spacing"] = True
    monitor["wavelength center"] = wavelength_m
    monitor["wavelength span"] = 0.0
    monitor["frequency points"] = 1
    try:
        monitor["spatial interpolation"] = "specified position"
    except Exception:
        pass


def _polygon_bounds(architecture: dict[str, Any]) -> list[tuple[float, ...]]:
    bounds: list[tuple[float, ...]] = []
    for polygon in architecture["geometry"]["polygons"]:
        vertices = np.asarray(polygon["vertices_nm"], float) * 1.0e-9
        bounds.append(
            (
                float(vertices[:, 0].min()),
                float(vertices[:, 0].max()),
                float(vertices[:, 1].min()),
                float(vertices[:, 1].max()),
            )
        )
    return bounds


def _setup(fdtd: Any, key: str, contract: dict[str, Any]) -> dict[str, Any]:
    optical = contract["optical"]
    architecture = contract["architectures"][key]
    wavelength_m = float(architecture["wavelength_um"]) * 1.0e-6
    z_min_m, z_max_m = [
        float(value) * 1.0e-6 for value in architecture["optical_z_bounds_um"]
    ]
    w0_m = float(optical["source"]["physical_target_waist_um"]) * 1.0e-6
    source_object_w0_m = float(
        os.environ.get(
            "FINITE_SOURCE_OBJECT_W0_UM",
            optical["source"]["physical_target_waist_um"],
        )
    ) * 1.0e-6
    source_z_m = float(optical["source"]["source_z_um"]) * 1.0e-6
    waist_z_m = float(optical["source"]["waist_plane_z_um"]) * 1.0e-6

    solver = fdtd.addfdtd()
    solver["dimension"] = "3D"
    solver["x min"], solver["x max"] = -12.0e-6, 12.0e-6
    solver["y min"], solver["y max"] = -12.0e-6, 12.0e-6
    solver["z min"], solver["z max"] = z_min_m, z_max_m
    for axis in "xyz":
        solver[f"{axis} min bc"] = "PML"
        solver[f"{axis} max bc"] = "PML"
    solver["pml layers"] = int(optical["PML_layers"])
    solver["mesh type"] = "auto non-uniform"
    solver["mesh refinement"] = "conformal variant 1"
    solver["mesh accuracy"] = int(optical["mesh"]["accuracy"])
    solver["simulation time"] = 3.0e-12
    solver["auto shutoff min"] = 1.0e-6
    solver["override simulation bandwidth for mesh generation"] = True
    solver["mesh wavelength min"] = wavelength_m
    solver["mesh wavelength max"] = wavelength_m

    source = fdtd.addgaussian()
    source["name"] = SOURCE_NAME
    source["injection axis"] = "z"
    source["direction"] = "backward"
    source["polarization angle"] = 0.0
    source["source shape"] = "Gaussian"
    source["use scalar approximation"] = True
    source["beam parameters"] = "Waist size and position"
    source["waist radius w0"] = source_object_w0_m
    source["distance from waist"] = -(source_z_m - waist_z_m)
    source["x min"], source["x max"] = -9.0e-6, 9.0e-6
    source["y min"], source["y max"] = -9.0e-6, 9.0e-6
    source["z"] = source_z_m
    source["override global source settings"] = True
    source["wavelength start"] = 0.90 * wavelength_m
    source["wavelength stop"] = 1.10 * wavelength_m

    outer = fdtd.addmesh()
    outer["name"] = "finite_outer_250nm_xy"
    outer["x min"], outer["x max"] = -12.0e-6, 12.0e-6
    outer["y min"], outer["y max"] = -12.0e-6, 12.0e-6
    outer["z min"], outer["z max"] = -0.5e-6, 0.3e-6
    outer["override x mesh"] = True
    outer["override y mesh"] = True
    outer["override z mesh"] = False
    outer["dx"], outer["dy"] = 250.0e-9, 250.0e-9

    flake = fdtd.addmesh()
    flake["name"] = "finite_flake_Q_100nm_xy_5nm_z"
    flake["x min"], flake["x max"] = -10.0e-6, 10.0e-6
    flake["y min"], flake["y max"] = -10.0e-6, 10.0e-6
    flake["z min"], flake["z max"] = -50.0e-9, 200.0e-9
    flake["override x mesh"] = True
    flake["override y mesh"] = True
    flake["override z mesh"] = True
    flake["dx"], flake["dy"], flake["dz"] = 100.0e-9, 100.0e-9, 5.0e-9

    top_bounds = _polygon_bounds(architecture)
    # A square union keeps the source-only x/y discretization symmetric while
    # covering every top-Au edge.  The actual Au shape is still the published
    # T or Z polygon; this object changes only the mesh, not the geometry.
    fine_half_extent = max(
        abs(value)
        for bounds in top_bounds
        for value in bounds
    ) + 100.0e-9
    local = fdtd.addmesh()
    local["name"] = f"finite_{key}_top_Au_symmetric_local"
    local["x min"], local["x max"] = -fine_half_extent, fine_half_extent
    local["y min"], local["y max"] = -fine_half_extent, fine_half_extent
    local["z min"], local["z max"] = 75.0e-9, 225.0e-9
    local["override x mesh"] = True
    local["override y mesh"] = True
    local["override z mesh"] = True
    local["dx"], local["dy"], local["dz"] = 25.0e-9, 25.0e-9, 5.0e-9

    _add_single_frequency_monitor(fdtd, TARGET_MONITOR, waist_z_m, wavelength_m)
    return {
        "architecture": key,
        "wavelength_m": wavelength_m,
        "domain_bounds_m": {
            "x": [-12.0e-6, 12.0e-6],
            "y": [-12.0e-6, 12.0e-6],
            "z": [z_min_m, z_max_m],
        },
        "source_bounds_m": {
            "x": [-9.0e-6, 9.0e-6],
            "y": [-9.0e-6, 9.0e-6],
            "z": source_z_m,
        },
        "waist_plane_z_m": waist_z_m,
        "requested_w0_m": w0_m,
        "Lumerical_source_object_w0_m": source_object_w0_m,
        "source_object_calibration_is_Q_rescaling": False,
        "top_Au_mesh_bounds_m": top_bounds,
        "top_Au_symmetric_mesh_half_extent_m": fine_half_extent,
    }


def _interval_max_step(coordinate: np.ndarray, lower: float, upper: float) -> float:
    centers = 0.5 * (coordinate[:-1] + coordinate[1:])
    selected = np.diff(coordinate)[(centers >= lower) & (centers <= upper)]
    if selected.size == 0:
        raise RuntimeError(f"no mesh intervals in [{lower}, {upper}]")
    return float(np.max(selected))


def main() -> int:
    key, contract = _architecture()
    default_output = (
        Path("/home/seunghyun/tairte4/raw_artifacts")
        / "finite_T_Z_source_only"
        / key
    )
    output = Path(
        os.environ.get("FINITE_SOURCE_OUTPUT", str(default_output))
    ).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"FINITE_{key}_GAUSSIAN_SOURCE_ONLY.json"
    fsp_path = output / f"finite_{key}_gaussian_source_only.fsp"
    npz_path = output / f"finite_{key}_gaussian_source_only_fields.npz"
    for protected in (json_path, fsp_path, npz_path):
        if protected.exists():
            raise RuntimeError(f"refusing to overwrite: {protected}")

    result: dict[str, Any] = {
        "status": f"BLOCKED_FINITE_{key}_GAUSSIAN_SOURCE_ONLY"
    }
    fdtd = None
    try:
        os.environ["VC_LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
        os.environ["LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
        os.environ["LUMERICAL_PYTHONPATH"] = str(audit.APPROVED_API)
        os.environ["PATH"] = (
            f"{audit.APPROVED_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
        )
        sys.path.insert(0, str(audit.APPROVED_API))
        import lumapi

        gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        gpu_device = f"GPU {gpu_index}"
        os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = gpu_device
        os.environ["CL_GPU_DEVICE"] = gpu_device
        os.environ["FDTD_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8")

        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        setup = _setup(fdtd, key, contract)
        fdtd.setresource("FDTD", 1, "active", 0)
        fdtd.setresource("FDTD", 2, "active", 1)
        fdtd.setresource("FDTD", 2, "processes", "1")
        fdtd.setresource("FDTD", 2, "threads", os.environ["FDTD_THREADS"])
        fdtd.setresource("FDTD", 2, "device type", gpu_device)
        fdtd.setresource(
            "FDTD", 2, "solver extra command line options", "-gpu"
        )

        fdtd.runsetup()
        pre_mesh = audit.mesh_readback(fdtd)
        if not pre_mesh.get("available"):
            raise RuntimeError(f"pre-run mesh readback unavailable: {pre_mesh}")
        coordinates = pre_mesh["coordinate_arrays"]
        mesh_regions = {
            "flake_max_dx_m": _interval_max_step(
                coordinates["x"], -10.0e-6, 10.0e-6
            ),
            "flake_max_dy_m": _interval_max_step(
                coordinates["y"], -10.0e-6, 10.0e-6
            ),
            "structure_max_dz_m": _interval_max_step(
                coordinates["z"], -50.0e-9, 225.0e-9
            ),
        }
        fdtd.save(str(fsp_path))

        started = time.monotonic()
        resource = audit.strict_gpu_run(fdtd, f"finite_{key}_source_only")
        wall_time_s = time.monotonic() - started
        fdtd.save(str(fsp_path))

        wavelength_m = float(setup["wavelength_m"])
        frequency_hz = C0 / wavelength_m
        source_power_w = audit.scalar(
            fdtd.sourcepower(frequency_hz, 2, SOURCE_NAME), "sourcepower"
        )
        fields = audit.monitor_fields(fdtd, TARGET_MONITOR)
        target_metrics, target_arrays = audit.plane_metrics(
            fields, source_power_w
        )
        source_result = fdtd.getresult(SOURCE_NAME, "fields")
        source_metrics, source_arrays = audit.source_profile_from_arrays(
            source_result["x"], source_result["y"], source_result["E"]
        )
        post_mesh = audit.mesh_readback(fdtd)
        launcher_run_dir = Path(
            os.environ.get("EIDL_RUN_DIR", str(output))
        ).expanduser().resolve()
        # The engine log is written beside the FSP.  The wrapper log is only
        # finalized after this Python process exits and therefore cannot be
        # the in-process source for shutoff/memory gates.
        log = audit.log_audit(output)

        np.savez_compressed(
            npz_path,
            **{f"target_{name}": value for name, value in target_arrays.items()},
            **source_arrays,
        )
        w0_m = float(setup["requested_w0_m"])
        fitted_wx = float(target_metrics["fitted_waist_x_m"])
        fitted_wy = float(target_metrics["fitted_waist_y_m"])
        center_error = float(target_metrics["beam_center_error_m"])
        gates = {
            "GPU_completed": bool(log["simulation_completed_successfully"]),
            "no_CPU_fallback": True,
            "auto_shutoff_lt_1e_5": (
                log["final_auto_shutoff"] is not None
                and log["final_auto_shutoff"] < 1.0e-5
            ),
            "waist_x_within_0p5pct": abs(fitted_wx - w0_m) / w0_m < 0.005,
            "waist_y_within_0p5pct": abs(fitted_wy - w0_m) / w0_m < 0.005,
            "Gaussian_fit_NRMSE_lt_0p5pct": (
                target_metrics["Gaussian_fit_NRMSE"] < 0.005
            ),
            "ellipticity_lt_0p5pct": (
                target_metrics["fitted_xy_ellipticity"] < 0.005
            ),
            "center_displacement_lt_50nm": center_error < 50.0e-9,
            "incident_power_closure_lt_0p5pct": (
                abs(target_metrics["downward_Poynting_power_over_sourcepower"] - 1.0)
                < 0.005
            ),
            "actual_mesh_readback_available": bool(post_mesh.get("available")),
            "GPU_memory_readback_available": (
                log["precise_GPU_memory_GiB"] is not None
            ),
            "all_fields_finite": bool(target_metrics["all_fields_finite"]),
        }
        result = {
            "status": (
                f"VALIDATED_FINITE_{key}_GAUSSIAN_SOURCE_ONLY"
                if all(gates.values())
                else f"FAILED_FINITE_{key}_GAUSSIAN_SOURCE_ONLY_GATE"
            ),
            "classification": (
                "homogeneous-air source-only gate; no material, Q, thermal, "
                "electrical, PTE, adjoint or optimization result"
            ),
            "solver_version": str(fdtd.version()),
            "GPU_resource_used": resource,
            "solver_wall_time_s": wall_time_s,
            "launcher_run_directory": str(launcher_run_dir),
            "setup": setup,
            "source_power_W": source_power_w,
            "source_object_metrics": source_metrics,
            "target_plane_metrics": target_metrics,
            "mesh_region_audit": mesh_regions,
            "mesh_readback": {
                key: value
                for key, value in post_mesh.items()
                if key != "coordinate_arrays"
            },
            "log_audit": log,
            "gates": gates,
            "Q_processing": {
                "clipping": False,
                "smoothing": False,
                "gain": False,
                "global_rescaling": False,
            },
            "raw_artifacts": [],
        }
        result["raw_artifacts"] = [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (fsp_path, npz_path)
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
        _write_json(json_path, result)

    print(json.dumps(result, indent=2, default=_json_default))
    return 0 if str(result["status"]).startswith("VALIDATED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
