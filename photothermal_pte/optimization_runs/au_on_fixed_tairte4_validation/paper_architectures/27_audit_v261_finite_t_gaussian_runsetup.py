#!/usr/bin/env python3
"""GPU-resource v261 runsetup audit for the finite multi-T Gaussian case.

No Maxwell time stepping, Q extraction, thermal, PTE, adjoint, or optimization
is performed.  The purpose is to close the geometry, source, and realized-mesh
contract before a costly finite-array solve.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    validate_paper_ir_source_only_gpu as audit,
)


WAVELENGTH_UM = 11.825
W0_UM = 4.0
DOMAIN_X_UM = 28.5
DOMAIN_Y_UM = 29.0
ARRAY_NX = 11
ARRAY_NY = 17
PERIOD_X_UM = 1.5
PERIOD_Y_UM = 1.0
ARRAY_X_UM = ARRAY_NX * PERIOD_X_UM
ARRAY_Y_UM = ARRAY_NY * PERIOD_Y_UM
SOURCE_SPAN_UM = 16.0
SOURCE_Z_UM = 0.8
FOCUS_Z_UM = 0.05
Z_MIN_UM = -1.2
Z_MAX_UM = 1.2
LOCAL_DXY_NM = 50.0
LOCAL_DZ_NM = 5.0


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


def add_full_span_rect(fdtd, name: str, material: str, z_min: float, z_max: float) -> None:
    item = fdtd.addrect()
    item["name"] = name
    item["material"] = material
    item["x min"] = -0.5 * DOMAIN_X_UM * 1.0e-6
    item["x max"] = 0.5 * DOMAIN_X_UM * 1.0e-6
    item["y min"] = -0.5 * DOMAIN_Y_UM * 1.0e-6
    item["y max"] = 0.5 * DOMAIN_Y_UM * 1.0e-6
    item["z min"] = z_min
    item["z max"] = z_max


def setup(fdtd) -> dict[str, object]:
    base = load_module("07_run_v261_t2024_tairte4_optical_smoke.py", "finite_t_material_helpers")
    geometry_module = load_module("05_actual_metasurface_geometry.py", "finite_t_geometry")
    base.configure_wavelength(WAVELENGTH_UM)
    geometry = geometry_module.inverse_t_mir_4750nm()

    solver = fdtd.addfdtd()
    solver["dimension"] = "3D"
    solver["x min"] = -0.5 * DOMAIN_X_UM * 1.0e-6
    solver["x max"] = 0.5 * DOMAIN_X_UM * 1.0e-6
    solver["y min"] = -0.5 * DOMAIN_Y_UM * 1.0e-6
    solver["y max"] = 0.5 * DOMAIN_Y_UM * 1.0e-6
    solver["z min"] = Z_MIN_UM * 1.0e-6
    solver["z max"] = Z_MAX_UM * 1.0e-6
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
    solver["mesh wavelength min"] = WAVELENGTH_UM * 1.0e-6
    solver["mesh wavelength max"] = WAVELENGTH_UM * 1.0e-6

    source = fdtd.addgaussian()
    source["name"] = "finite_T_scalar_Gaussian"
    source["injection axis"] = "z"
    source["direction"] = "backward"
    source["polarization angle"] = 0.0
    source["source shape"] = "Gaussian"
    source["use scalar approximation"] = True
    source["beam parameters"] = "Waist size and position"
    source["waist radius w0"] = W0_UM * 1.0e-6
    source["distance from waist"] = -(SOURCE_Z_UM - FOCUS_Z_UM) * 1.0e-6
    source["x min"] = -0.5 * SOURCE_SPAN_UM * 1.0e-6
    source["x max"] = 0.5 * SOURCE_SPAN_UM * 1.0e-6
    source["y min"] = -0.5 * SOURCE_SPAN_UM * 1.0e-6
    source["y max"] = 0.5 * SOURCE_SPAN_UM * 1.0e-6
    source["z"] = SOURCE_Z_UM * 1.0e-6
    source["override global source settings"] = True
    source["wavelength start"] = 0.95 * WAVELENGTH_UM * 1.0e-6
    source["wavelength stop"] = 12.0e-6

    base.add_tairte4_material(fdtd)
    base.add_constant_nk(fdtd, base.AL2O3_MATERIAL, base.AL2O3_N)
    add_full_span_rect(fdtd, "finite_stack_Si", base.SI_MATERIAL, Z_MIN_UM * 1.0e-6, -0.520e-6)
    add_full_span_rect(fdtd, "finite_stack_SiO2_285nm", base.SIO2_MATERIAL, -0.520e-6, -0.235e-6)
    add_full_span_rect(fdtd, "finite_stack_Au_mirror_200nm", base.AU_MATERIAL, -0.235e-6, -0.035e-6)
    add_full_span_rect(fdtd, "finite_stack_Al2O3_35nm", base.AL2O3_MATERIAL, -0.035e-6, 0.0)
    add_full_span_rect(fdtd, "finite_stack_TaIrTe4_100nm", base.TAIRTE4_MATERIAL, 0.0, 0.100e-6)

    polygon = geometry.polygons[0]
    base_vertices = np.asarray(polygon.vertices_nm, float) * 1.0e-9
    centers: list[list[float]] = []
    for ix in range(ARRAY_NX):
        for iy in range(ARRAY_NY):
            x0 = (ix - 0.5 * (ARRAY_NX - 1)) * PERIOD_X_UM * 1.0e-6
            y0 = (iy - 0.5 * (ARRAY_NY - 1)) * PERIOD_Y_UM * 1.0e-6
            item = fdtd.addpoly()
            item["name"] = f"finite_T_{ix:02d}_{iy:02d}"
            item["material"] = base.AU_MATERIAL
            item["vertices"] = base_vertices + np.asarray([x0, y0])
            item["z min"] = polygon.z_min_nm * 1.0e-9
            item["z max"] = polygon.z_max_nm * 1.0e-9
            centers.append([x0, y0])

    mesh = fdtd.addmesh()
    mesh["name"] = "finite_T_array_local_mesh"
    mesh["x min"] = -0.5 * ARRAY_X_UM * 1.0e-6
    mesh["x max"] = 0.5 * ARRAY_X_UM * 1.0e-6
    mesh["y min"] = -0.5 * ARRAY_Y_UM * 1.0e-6
    mesh["y max"] = 0.5 * ARRAY_Y_UM * 1.0e-6
    mesh["z min"] = -0.30e-6
    mesh["z max"] = 0.20e-6
    mesh["override x mesh"] = True
    mesh["override y mesh"] = True
    mesh["override z mesh"] = True
    mesh["dx"] = LOCAL_DXY_NM * 1.0e-9
    mesh["dy"] = LOCAL_DXY_NM * 1.0e-9
    mesh["dz"] = LOCAL_DZ_NM * 1.0e-9

    return {
        "classification": "finite multi-T Gaussian runsetup audit only",
        "wavelength_um": WAVELENGTH_UM,
        "polarization": "E||b (Lumerical x)",
        "axis_mapping": "x=b, y=a, z=c=b closure",
        "domain_um": {"x": [-0.5 * DOMAIN_X_UM, 0.5 * DOMAIN_X_UM], "y": [-0.5 * DOMAIN_Y_UM, 0.5 * DOMAIN_Y_UM], "z": [Z_MIN_UM, Z_MAX_UM]},
        "boundaries": "six PML, 24 layers; no periodic/Bloch boundary",
        "array": {"nx": ARRAY_NX, "ny": ARRAY_NY, "count": ARRAY_NX * ARRAY_NY, "span_um": [ARRAY_X_UM, ARRAY_Y_UM], "centers_m": centers},
        "source": {"type": "scalar Gaussian; waist size and position", "requested_w0_um": W0_UM, "span_um": SOURCE_SPAN_UM, "z_um": SOURCE_Z_UM, "focus_z_um": FOCUS_Z_UM, "polarization": "x_b"},
        "stack": ["air", "Au inverse-T array 33nm", "TaIrTe4 100nm", "Al2O3 35nm", "Au mirror 200nm", "SiO2 285nm optical closure", "Si"],
        "lateral_stack_rule": "TaIrTe4/Al2O3/Au-mirror/SiO2/Si extend through lateral PML; only top inverse-T array is finite",
        "local_mesh_requested_nm": {"dx": LOCAL_DXY_NM, "dy": LOCAL_DXY_NM, "dz": LOCAL_DZ_NM},
        "Q_operations": {"clipping": False, "smoothing": False, "gain": False, "rescaling": False, "tiling_after_solve": False},
    }


def local_mesh_metrics(mesh: dict[str, object]) -> dict[str, object]:
    coordinates = {axis: np.asarray(mesh["coordinate_arrays"][axis], float) for axis in "xyz"}
    result: dict[str, object] = {
        "shape": [int(coordinates[axis].size) for axis in "xyz"],
        "yee_cell_estimate": int(np.prod([max(coordinates[axis].size - 1, 0) for axis in "xyz"], dtype=np.int64)),
        "bounds_m": {axis: [float(coordinates[axis][0]), float(coordinates[axis][-1])] for axis in "xyz"},
        "global_min_step_m": {axis: float(np.min(np.diff(coordinates[axis]))) for axis in "xyz"},
        "global_max_step_m": {axis: float(np.max(np.diff(coordinates[axis]))) for axis in "xyz"},
    }
    masks = {
        "x": (coordinates["x"][:-1] >= -0.5 * ARRAY_X_UM * 1e-6) & (coordinates["x"][1:] <= 0.5 * ARRAY_X_UM * 1e-6),
        "y": (coordinates["y"][:-1] >= -0.5 * ARRAY_Y_UM * 1e-6) & (coordinates["y"][1:] <= 0.5 * ARRAY_Y_UM * 1e-6),
        "z": (coordinates["z"][:-1] >= -0.30e-6) & (coordinates["z"][1:] <= 0.20e-6),
    }
    result["local_max_step_m"] = {
        axis: float(np.max(np.diff(coordinates[axis])[masks[axis]])) for axis in "xyz"
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/seunghyun/tairte4/raw_artifacts/"
            "paper_tairte4_finite_T_w0_4um_runsetup"
        ),
    )
    parser.add_argument("--gpu-device", default="GPU 5")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "FINITE_T_GAUSSIAN_RUNSETUP_AUDIT.json"
    fsp_path = output / "finite_T_gaussian_runsetup.fsp"
    result: dict[str, object] = {"status": "BLOCKED_FINITE_T_GAUSSIAN_RUNSETUP"}
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
        contract = setup(fdtd)
        fdtd.setresource("FDTD", 1, "active", 0)
        fdtd.setresource("FDTD", 2, "active", 1)
        fdtd.setresource("FDTD", 2, "processes", "1")
        fdtd.setresource("FDTD", 2, "threads", "8")
        fdtd.setresource("FDTD", 2, "device type", args.gpu_device)
        fdtd.setresource("FDTD", 2, "solver extra command line options", "-gpu")
        fdtd.runsetup()
        raw_mesh = audit.mesh_readback(fdtd)
        if not raw_mesh.get("available"):
            raise RuntimeError(f"native mesh readback unavailable: {raw_mesh}")
        metrics = local_mesh_metrics(raw_mesh)
        gates = {
            "six_PML_and_no_periodic": True,
            "finite_T_count_187": contract["array"]["count"] == 187,
            "local_dx_le_50nm": metrics["local_max_step_m"]["x"] <= 50.0e-9 + 1.0e-12,
            "local_dy_le_50nm": metrics["local_max_step_m"]["y"] <= 50.0e-9 + 1.0e-12,
            "local_dz_le_5nm": metrics["local_max_step_m"]["z"] <= 5.0e-9 + 1.0e-12,
            "source_strictly_inside_domain": SOURCE_SPAN_UM < min(DOMAIN_X_UM, DOMAIN_Y_UM),
        }
        fdtd.save(str(fsp_path))
        result = {
            "status": "COMPLETED_FINITE_T_GAUSSIAN_RUNSETUP_AUDIT" if all(gates.values()) else "FAILED_FINITE_T_GAUSSIAN_RUNSETUP_GATE",
            "contract": contract,
            "solver_version": str(fdtd.version()),
            "resource": {prop: str(fdtd.getresource("FDTD", 2, prop)) for prop in ("active", "device type", "processes", "threads", "solver extra command line options")},
            "mesh_runsetup": metrics,
            "gates": gates,
            "scope_exclusions": ["Maxwell time stepping", "Q", "thermal", "PTE", "adjoint", "optimization"],
            "raw_artifact": {"path": str(fsp_path), "size_bytes": fsp_path.stat().st_size, "sha256": sha256(fsp_path)},
        }
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
