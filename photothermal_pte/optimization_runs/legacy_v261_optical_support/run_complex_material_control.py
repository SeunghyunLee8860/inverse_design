#!/usr/bin/env python3
"""GPU scalar/imported complex-material control at the 10 um FOM frequency."""

from __future__ import annotations

import argparse
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
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.native_yee_q import (  # noqa: E402
    extract_native_yee_q,
    frequency_slice,
)
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_FIELD,
    PABS_GROUP,
    PABS_INDEX,
)
from photothermal_pte.optimization_runs.gaussian10_contract import (  # noqa: E402
    silica_10um,
)


C0 = 299792458.0
WAVELENGTH_M = 10.0e-6
FREQUENCY_HZ = C0 / WAVELENGTH_M
BLOCK_BOUNDS = {"x": (-5e-6, 5e-6), "y": (-5e-6, 5e-6), "z": (0.05e-6, 1.05e-6)}
FLUX_BOUNDS = {"x": (-5.5e-6, 5.5e-6), "y": (-5.5e-6, 5.5e-6), "z": (-0.45e-6, 1.55e-6)}


def load_source_wrapper():
    path = HERE / "audit_source_only_gpu.py"
    spec = importlib.util.spec_from_file_location("run002_source_wrapper_material", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure_source_audit()
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def imported_nodes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.linspace(*BLOCK_BOUNDS["x"], 101),
        np.linspace(*BLOCK_BOUNDS["y"], 101),
        np.linspace(*BLOCK_BOUNDS["z"], 21),
    )


def complex_index(rho: float) -> tuple[complex, complex]:
    silica = silica_10um()
    epsilon_sio2 = complex(silica["epsilon_real"], silica["epsilon_imag"])
    epsilon = 1.0 + float(rho) * (epsilon_sio2 - 1.0)
    index = complex(np.sqrt(epsilon))
    if index.imag < 0.0:
        index = -index
    return epsilon, index


def add_design(fdtd, *, rho: float, representation: str) -> dict[str, object]:
    epsilon, index = complex_index(rho)
    name = f"rho{rho:g}_{representation}_complex_block"
    if representation == "scalar":
        material_name = f"rho{rho:g}_single_frequency_nk"
        material = fdtd.addmaterial("(n,k) Material")
        fdtd.setmaterial(material, "name", material_name)
        fdtd.setmaterial(material_name, "Refractive Index", float(index.real))
        fdtd.setmaterial(
            material_name, "Imaginary Refractive Index", float(index.imag)
        )
        block = fdtd.addrect()
        block["name"] = name
        block["material"] = material_name
        for axis in "xyz":
            block[f"{axis} min"], block[f"{axis} max"] = BLOCK_BOUNDS[axis]
        sample_shape = None
    elif representation == "imported":
        x, y, z = imported_nodes()
        values = np.full((x.size, y.size, z.size), index, complex)
        fdtd.addimport({"name": name, "x": 0.0, "y": 0.0, "z": 0.0})
        if int(fdtd.importnk2(values, x, y, z)) != 1:
            raise RuntimeError("importnk2 returned failure")
        sample_shape = list(values.shape)
    else:
        raise ValueError(representation)
    return {
        "name": name,
        "representation": representation,
        "rho": rho,
        "requested_epsilon": [epsilon.real, epsilon.imag],
        "requested_nk": [index.real, index.imag],
        "bounds_m": {axis: list(BLOCK_BOUNDS[axis]) for axis in "xyz"},
        "import_sample_shape": sample_shape,
    }


def add_local_mesh(fdtd) -> None:
    mesh = fdtd.addmesh()
    mesh["name"] = "complex_block_local_mesh"
    mesh["x min"], mesh["x max"] = (-5.5e-6, 5.5e-6)
    mesh["y min"], mesh["y max"] = (-5.5e-6, 5.5e-6)
    mesh["z min"], mesh["z max"] = (-0.1e-6, 1.2e-6)
    mesh["override x mesh"] = True
    mesh["override y mesh"] = True
    mesh["override z mesh"] = True
    mesh["dx"] = 100e-9
    mesh["dy"] = 100e-9
    mesh["dz"] = 25e-9


def add_absorption_monitors(fdtd) -> dict[str, float]:
    pabs = fdtd.addobject("pabs_adv")
    pabs["name"] = PABS_GROUP
    for axis in "xyz":
        # Match the six-face control volume, including conformal interface
        # half-cells outside the nominal block boundary.
        low, high = FLUX_BOUNDS[axis]
        pabs[axis] = 0.5 * (low + high)
        pabs[f"{axis} span"] = high - low
    signs = {}
    for axis in "xyz":
        for side, position in zip(("min", "max"), FLUX_BOUNDS[axis]):
            name = f"run002_flux_{axis}_{side}"
            monitor = fdtd.addpower()
            monitor["name"] = name
            monitor["monitor type"] = f"2D {axis.upper()}-normal"
            monitor[axis] = position
            for transverse in "xyz":
                if transverse != axis:
                    monitor[f"{transverse} min"] = FLUX_BOUNDS[transverse][0]
                    monitor[f"{transverse} max"] = FLUX_BOUNDS[transverse][1]
            monitor["override global monitor settings"] = True
            monitor["use source limits"] = False
            monitor["use wavelength spacing"] = True
            monitor["wavelength center"] = WAVELENGTH_M
            monitor["wavelength span"] = 0.0
            monitor["frequency points"] = 1
            signs[name] = -1.0 if side == "min" else 1.0
    return signs


def component_epsilon_readback(fdtd, q: dict[str, object]) -> dict[str, object]:
    spatial_shape = tuple(
        np.asarray(q["base_coordinates"][axis]).size for axis in "xyz"
    )
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
        finite = np.isfinite(epsilon)
        values = epsilon[finite]
        x = np.asarray(q["base_coordinates"]["x"], float)
        y = np.asarray(q["base_coordinates"]["y"], float)
        z = np.asarray(q["base_coordinates"]["z"], float)
        margin = {"x": 0.2e-6, "y": 0.2e-6, "z": 0.1e-6}
        interior = (
            (x[:, None, None] >= BLOCK_BOUNDS["x"][0] + margin["x"])
            & (x[:, None, None] <= BLOCK_BOUNDS["x"][1] - margin["x"])
            & (y[None, :, None] >= BLOCK_BOUNDS["y"][0] + margin["y"])
            & (y[None, :, None] <= BLOCK_BOUNDS["y"][1] - margin["y"])
            & (z[None, None, :] >= BLOCK_BOUNDS["z"][0] + margin["z"])
            & (z[None, None, :] <= BLOCK_BOUNDS["z"][1] - margin["z"])
            & finite
        )
        interior_values = epsilon[interior]
        if interior_values.size == 0:
            raise RuntimeError(f"empty interior epsilon readback for {component}")
        result[component] = {
            "shape": list(epsilon.shape),
            "epsilon_interior_median": [
                float(np.median(interior_values.real)),
                float(np.median(interior_values.imag)),
            ],
            "interior_sample_count": int(interior_values.size),
            "epsilon_real_range": [float(np.min(values.real)), float(np.max(values.real))],
            "epsilon_imag_range": [float(np.min(values.imag)), float(np.max(values.imag))],
            "all_finite": bool(np.all(finite)),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rho", type=float, required=True, choices=(0.0, 0.5, 1.0))
    parser.add_argument("--representation", required=True, choices=("scalar", "imported"))
    parser.add_argument("--gpu-device", default="GPU 4")
    parser.add_argument("--duration-ps", type=float, default=8.0)
    parser.add_argument("--auto-shutoff-min", type=float, default=1.0e-7)
    parser.add_argument(
        "--mesh-refinement",
        choices=(
            "conformal variant 0",
            "conformal variant 1",
            "precise volume average",
        ),
        default="conformal variant 1",
    )
    parser.add_argument("--meshing-refinement", type=int, default=5)
    parser.add_argument("--dt-stability-factor", type=float, default=0.99)
    parser.add_argument(
        "--mesh-wavelength-um",
        type=float,
        help=(
            "When provided, override the mesh-generation bandwidth with this "
            "single wavelength. This is required to evaluate dispersive "
            "materials at 10 um under precise volume average."
        ),
    )
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.meshing_refinement <= 12:
        raise ValueError("--meshing-refinement must be in [1,12]")
    if not 0.0 < args.dt_stability_factor < 1.0:
        raise ValueError("--dt-stability-factor must be in (0,1)")
    if args.mesh_wavelength_um is not None and args.mesh_wavelength_um <= 0.0:
        raise ValueError("--mesh-wavelength-um must be positive")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "case_result.json"
    project_path = output / "complex_material_control.fsp"
    npz_path = output / "complex_material_control_q.npz"
    result: dict[str, object] = {
        "status": "BLOCKED_COMPLEX_MATERIAL_CONTROL",
        "rho": args.rho,
        "representation": args.representation,
        "contract_only": args.contract_only,
        "scope": "isolated finite complex-index block in air; no thermal/PTE/adjoint/optimization",
    }
    fdtd = None
    try:
        wrapper = load_source_wrapper()
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
        helper = audit.load_module(audit.API_HELPER, "run002_material_lumerical_api")
        installation = type("Installation", (), {
            "version_key": "v261",
            "root": audit.APPROVED_ROOT,
            "lumapi_path": audit.APPROVED_API / "lumapi.py",
            "device_executable": audit.APPROVED_ROOT / "bin" / "device",
        })()
        lumapi = helper.load_lumapi(installation)
        import eqc_lib as runtime

        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        built = audit.setup(
            fdtd,
            args.duration_ps,
            args.auto_shutoff_min,
            3,
            8.36043075475035e-6,
            8.5e-6,
        )
        fdtd.setnamed("FDTD", "mesh refinement", args.mesh_refinement)
        fdtd.setnamed("FDTD", "meshing refinement", args.meshing_refinement)
        fdtd.setnamed("FDTD", "dt stability factor", args.dt_stability_factor)
        if args.mesh_wavelength_um is not None:
            mesh_wavelength_m = args.mesh_wavelength_um * 1.0e-6
            fdtd.setnamed(
                "FDTD", "override simulation bandwidth for mesh generation", True
            )
            fdtd.setnamed("FDTD", "mesh wavelength min", mesh_wavelength_m)
            fdtd.setnamed("FDTD", "mesh wavelength max", mesh_wavelength_m)
        built["mesh_contract"].update(
            {
                "mesh_refinement": str(
                    fdtd.getnamed("FDTD", "mesh refinement")
                ),
                "meshing_refinement": int(
                    round(float(fdtd.getnamed("FDTD", "meshing refinement")))
                ),
                "dt_stability_factor": float(
                    fdtd.getnamed("FDTD", "dt stability factor")
                ),
                "override_simulation_bandwidth_for_mesh_generation": bool(
                    round(
                        float(
                            fdtd.getnamed(
                                "FDTD",
                                "override simulation bandwidth for mesh generation",
                            )
                        )
                    )
                ),
                "mesh_wavelength_bounds_m": (
                    [
                        float(fdtd.getnamed("FDTD", "mesh wavelength min")),
                        float(fdtd.getnamed("FDTD", "mesh wavelength max")),
                    ]
                    if args.mesh_wavelength_um is not None
                    else None
                ),
            }
        )
        built["source"]["model"] = (
            "Run-002 scalar Gaussian at 10 um; single-frequency FOM contract"
        )
        built["source"]["source_object_waist_calibration"] = {
            "method": "one_step_from_run002_homogeneous_air_source_only_measurement",
            "input_source_object_waist_m": 8.36043075475035e-6,
            "measured_target_plane_effective_waist_m": 8.506397119765702e-6,
            "source_only_field_NPZ_sha256": (
                "d95e5cd2758c2d0cfba52a00da955619f65af2ca31befc4479d2730ae96006f6"
            ),
            "legacy_11um_12um_calibration_reused": False,
            "Q_clipping_smoothing_gain_or_rescaling": False,
        }
        material = add_design(fdtd, rho=args.rho, representation=args.representation)
        add_local_mesh(fdtd)
        flux_signs = add_absorption_monitors(fdtd)
        resources = runtime.configure_session_resources(fdtd)
        fdtd.runsetup()
        mesh = audit.mesh_readback(fdtd)
        result.update({
            "built_source_contract": built,
            "material": material,
            "resources": resources,
            "mesh_after_runsetup": {k: v for k, v in mesh.items() if k != "coordinate_arrays"},
            "solver_version": str(fdtd.version()),
        })
        fdtd.save(str(project_path))
        if args.contract_only:
            result["status"] = "COMPLETED_COMPLEX_MATERIAL_RUNSETUP_AUDIT"
        else:
            started = time.monotonic()
            result["GPU_resource_used"] = audit.strict_gpu_run(
                fdtd, f"run002_complex_rho{args.rho:g}_{args.representation}"
            )
            result["solver_wall_time_s"] = time.monotonic() - started
            source_power = audit.scalar(
                fdtd.sourcepower(FREQUENCY_HZ, 2, audit.SOURCE_NAME), "sourcepower"
            )
            net_outward = 0.0
            face_power = {}
            for name, sign in flux_signs.items():
                signed = audit.scalar(fdtd.transmission(name), name) * source_power
                outward = sign * signed
                face_power[name] = {"signed_axis_power_W": signed, "outward_power_W": outward}
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
            closure = abs(p_q - p_six) / max(abs(p_six), abs(p_q), 1e-300)
            readback = component_epsilon_readback(fdtd, q)
            arrays = {}
            for component in "xyz":
                arrays[f"Q{component}_W_m3"] = q["Q_components"][component]
                for axis in "xyz":
                    arrays[f"Q{component}_{axis}_m"] = q["native_coordinates"][component][axis]
            np.savez_compressed(npz_path, **arrays)
            result.update({
                "source_power_W": source_power,
                "P_Q_W": p_q,
                "P_six_W": p_six,
                "six_face_closure_relative": closure,
                "face_power": face_power,
                "Q_component_power_W": q["component_power_W"],
                "epsilon_component_readback": readback,
                "log_audit": audit.log_audit(output),
            })
            finite = all(row["all_finite"] for row in readback.values())
            pass_closure = (args.rho == 0.0 and abs(p_q) < 1e-20) or closure < 0.005
            result["passed"] = bool(finite and pass_closure)
            result["status"] = (
                "COMPLETED_COMPLEX_MATERIAL_FORWARD_CONTROL"
                if result["passed"]
                else "FAILED_COMPLEX_MATERIAL_FORWARD_CONTROL"
            )
            fdtd.save(str(project_path))
        artifacts = []
        for path in (project_path, npz_path):
            if path.is_file():
                artifacts.append({
                    "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)
                })
        result["raw_artifacts"] = artifacts
    except Exception as exc:
        result["status"] = "BLOCKED_COMPLEX_MATERIAL_CONTROL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(json.dumps(result, indent=2, default=audit.json_default if 'audit' in locals() else str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"].startswith("COMPLETED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
