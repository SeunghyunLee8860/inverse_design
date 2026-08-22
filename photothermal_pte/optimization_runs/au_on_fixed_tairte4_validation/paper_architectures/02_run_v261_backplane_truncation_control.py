#!/usr/bin/env python3
"""GPU-only v261 discriminator for substrate truncation below an Au mirror.

This is intentionally a planar mirror control.  It asks only whether the
fields above a 200-nm Au backplane depend on the SiO2/Si layers below it.  It
does not claim to be a T or Z device calculation, and it does not run thermal,
PTE, adjoint, or optimization code.

The full cases use the substrate stacks disclosed by the papers.  The
truncated case extends Au through the bottom PML.  Both use the same domain,
source, monitors, mesh near the upper Au interface, and native-Yee absorption
extraction.  CPU FDTD fallback is prohibited.
"""

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

from photothermal_pte.finite_inverse_design.native_yee_q import (  # noqa: E402
    extract_native_yee_q,
)
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_FIELD,
    PABS_GROUP,
    PABS_INDEX,
)
from photothermal_pte.optimization_runs.gaussian10_contract import (  # noqa: E402
    silica_10um,
)
from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    validate_paper_ir_source_only_gpu as audit,
)


C0 = 299_792_458.0
WAVELENGTH_M = 10.0e-6
FREQUENCY_HZ = C0 / WAVELENGTH_M
PERIOD_M = 1.0e-6
FDTD_Z_MIN_M = -2.5e-6
FDTD_Z_MAX_M = 1.5e-6
SOURCE_Z_M = 0.9e-6
TOP_MONITOR_Z_M = 0.5e-6
BOTTOM_MONITOR_Z_M = -2.0e-6
AU_TOP_M = 0.0
AU_PAPER_BOTTOM_M = -0.2e-6
AU_N_10UM = complex(12.1, 69.2)
PALIK_SI = "Si (Silicon) - Palik"
SOURCE_NAME = "paper_backplane_plane_wave"
TOP_MONITOR = "backplane_flux_top"
BOTTOM_MONITOR = "backplane_flux_bottom"


OXIDE_THICKNESS_M = {
    "z2022_285nm": 285.0e-9,
    "t2024_main_1500nm": 1500.0e-9,
    "t2024_si_rf_1000nm": 1000.0e-9,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complex_record(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def scalar(value: object, label: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise RuntimeError(f"{label} is not scalar: {array.shape}")
    result = float(np.real(array[0]))
    if not np.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def add_nk_material(fdtd: object, name: str, index: complex) -> None:
    material = fdtd.addmaterial("(n,k) Material")
    fdtd.setmaterial(material, "name", name)
    fdtd.setmaterial(name, "Refractive Index", float(index.real))
    fdtd.setmaterial(name, "Imaginary Refractive Index", float(index.imag))


def add_rect(
    fdtd: object,
    *,
    name: str,
    material: str,
    z_min_m: float,
    z_max_m: float,
) -> None:
    block = fdtd.addrect()
    block["name"] = name
    block["material"] = material
    block["x min"] = -0.5 * PERIOD_M
    block["x max"] = 0.5 * PERIOD_M
    block["y min"] = -0.5 * PERIOD_M
    block["y max"] = 0.5 * PERIOD_M
    block["z min"] = z_min_m
    block["z max"] = z_max_m


def add_power_monitor(fdtd: object, name: str, z_m: float) -> None:
    monitor = fdtd.addpower()
    monitor["name"] = name
    monitor["monitor type"] = "2D Z-normal"
    monitor["x min"] = -0.5 * PERIOD_M
    monitor["x max"] = 0.5 * PERIOD_M
    monitor["y min"] = -0.5 * PERIOD_M
    monitor["y max"] = 0.5 * PERIOD_M
    monitor["z"] = z_m
    monitor["override global monitor settings"] = True
    monitor["use source limits"] = False
    monitor["use wavelength spacing"] = True
    monitor["wavelength center"] = WAVELENGTH_M
    monitor["wavelength span"] = 0.0
    monitor["frequency points"] = 1


def setup_case(
    fdtd: object,
    *,
    architecture: str,
    substrate_mode: str,
    duration_ps: float,
    auto_shutoff_min: float,
) -> dict[str, object]:
    fdtd.switchtolayout()
    solver = fdtd.addfdtd()
    solver["dimension"] = "3D"
    solver["x min"] = -0.5 * PERIOD_M
    solver["x max"] = 0.5 * PERIOD_M
    solver["y min"] = -0.5 * PERIOD_M
    solver["y max"] = 0.5 * PERIOD_M
    solver["z min"] = FDTD_Z_MIN_M
    solver["z max"] = FDTD_Z_MAX_M
    solver["x min bc"] = "Periodic"
    solver["x max bc"] = "Periodic"
    solver["y min bc"] = "Periodic"
    solver["y max bc"] = "Periodic"
    solver["z min bc"] = "PML"
    solver["z max bc"] = "PML"
    solver["pml layers"] = 24
    solver["mesh type"] = "auto non-uniform"
    solver["mesh refinement"] = "conformal variant 1"
    solver["mesh accuracy"] = 3
    solver["simulation time"] = duration_ps * 1.0e-12
    solver["auto shutoff min"] = auto_shutoff_min
    solver["override simulation bandwidth for mesh generation"] = True
    solver["mesh wavelength min"] = WAVELENGTH_M
    solver["mesh wavelength max"] = WAVELENGTH_M

    source = fdtd.addplane()
    source["name"] = SOURCE_NAME
    source["injection axis"] = "z"
    source["direction"] = "backward"
    source["polarization angle"] = 0.0
    source["angle theta"] = 0.0
    source["angle phi"] = 0.0
    source["plane wave type"] = "Bloch/Periodic"
    source["x min"] = -0.5 * PERIOD_M
    source["x max"] = 0.5 * PERIOD_M
    source["y min"] = -0.5 * PERIOD_M
    source["y max"] = 0.5 * PERIOD_M
    source["z"] = SOURCE_Z_M
    source["override global source settings"] = True
    source["wavelength start"] = 9.5e-6
    source["wavelength stop"] = 10.5e-6
    fdtd.setglobalmonitor("use source limits", False)
    fdtd.setglobalmonitor("use wavelength spacing", True)
    fdtd.setglobalmonitor("wavelength center", WAVELENGTH_M)
    fdtd.setglobalmonitor("wavelength span", 0.0)
    fdtd.setglobalmonitor("frequency points", 1)

    add_nk_material(fdtd, "Ordal_Au_10um_exact_nk", AU_N_10UM)
    if substrate_mode == "full":
        silica = silica_10um()
        sio2_n = complex(silica["n_real"], silica["n_imag"])
        si_n = complex(
            np.asarray(fdtd.getindex(PALIK_SI, FREQUENCY_HZ)).reshape(-1)[0]
        )
        add_nk_material(fdtd, "Kitamura_SiO2_10um_exact_nk", sio2_n)
        add_nk_material(fdtd, "Palik_Si_10um_readback_nk", si_n)
        au_bottom_m = AU_PAPER_BOTTOM_M
        oxide_bottom_m = au_bottom_m - OXIDE_THICKNESS_M[architecture]
        if oxide_bottom_m <= FDTD_Z_MIN_M + 0.25e-6:
            raise RuntimeError("explicit oxide leaves no resolved Si before bottom PML")
        add_rect(
            fdtd,
            name="paper_Au_backplane_200nm",
            material="Ordal_Au_10um_exact_nk",
            z_min_m=au_bottom_m,
            z_max_m=AU_TOP_M,
        )
        add_rect(
            fdtd,
            name="paper_thermal_SiO2",
            material="Kitamura_SiO2_10um_exact_nk",
            z_min_m=oxide_bottom_m,
            z_max_m=au_bottom_m,
        )
        add_rect(
            fdtd,
            name="paper_Si_substrate",
            material="Palik_Si_10um_readback_nk",
            z_min_m=FDTD_Z_MIN_M,
            z_max_m=oxide_bottom_m,
        )
    elif substrate_mode == "au_truncated":
        sio2_n = None
        si_n = None
        oxide_bottom_m = None
        au_bottom_m = FDTD_Z_MIN_M
        add_rect(
            fdtd,
            name="Au_extended_through_bottom_PML",
            material="Ordal_Au_10um_exact_nk",
            z_min_m=FDTD_Z_MIN_M,
            z_max_m=AU_TOP_M,
        )
    else:
        raise ValueError(substrate_mode)

    mesh = fdtd.addmesh()
    mesh["name"] = "Au_interface_5nm_z_mesh"
    mesh["x min"] = -0.5 * PERIOD_M
    mesh["x max"] = 0.5 * PERIOD_M
    mesh["y min"] = -0.5 * PERIOD_M
    mesh["y max"] = 0.5 * PERIOD_M
    mesh["z min"] = -0.30e-6
    mesh["z max"] = 0.10e-6
    mesh["override x mesh"] = True
    mesh["override y mesh"] = True
    mesh["override z mesh"] = True
    mesh["dx"] = 100.0e-9
    mesh["dy"] = 100.0e-9
    mesh["dz"] = 5.0e-9

    pabs = fdtd.addobject("pabs_adv")
    pabs["name"] = PABS_GROUP
    pabs["x"] = 0.0
    pabs["x span"] = PERIOD_M
    pabs["y"] = 0.0
    pabs["y span"] = PERIOD_M
    pabs["z"] = 0.5 * (TOP_MONITOR_Z_M + BOTTOM_MONITOR_Z_M)
    pabs["z span"] = TOP_MONITOR_Z_M - BOTTOM_MONITOR_Z_M
    add_power_monitor(fdtd, TOP_MONITOR, TOP_MONITOR_Z_M)
    add_power_monitor(fdtd, BOTTOM_MONITOR, BOTTOM_MONITOR_Z_M)

    return {
        "architecture": architecture,
        "substrate_mode": substrate_mode,
        "period_m": PERIOD_M,
        "domain_bounds_m": {
            "x": [-0.5 * PERIOD_M, 0.5 * PERIOD_M],
            "y": [-0.5 * PERIOD_M, 0.5 * PERIOD_M],
            "z": [FDTD_Z_MIN_M, FDTD_Z_MAX_M],
        },
        "source": {
            "type": "normal-incidence Bloch/Periodic plane wave",
            "z_m": SOURCE_Z_M,
            "polarization": "Ex",
            "wavelength_m": WAVELENGTH_M,
        },
        "Au_backplane": {
            "n_plus_ik": complex_record(AU_N_10UM),
            "top_m": AU_TOP_M,
            "bottom_m": au_bottom_m,
            "paper_thickness_m": 200.0e-9,
        },
        "oxide": {
            "thickness_m": OXIDE_THICKNESS_M[architecture]
            if substrate_mode == "full"
            else None,
            "bottom_m": oxide_bottom_m,
            "n_plus_ik": complex_record(sio2_n) if sio2_n is not None else None,
        },
        "Si": {
            "material": PALIK_SI if substrate_mode == "full" else None,
            "readback_n_plus_ik": complex_record(si_n) if si_n is not None else None,
        },
        "scope": (
            "planar Au-backplane substrate discriminator only; no TaIrTe4/T/Z, "
            "thermal, PTE, adjoint, or optimization"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=tuple(OXIDE_THICKNESS_M), required=True)
    parser.add_argument("--substrate-mode", choices=("full", "au_truncated"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 4")
    parser.add_argument("--duration-ps", type=float, default=1.0)
    parser.add_argument("--auto-shutoff-min", type=float, default=1.0e-6)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "backplane_case_result.json"
    fsp_path = output / "backplane_control.fsp"
    npz_path = output / "backplane_native_q.npz"
    result: dict[str, object] = {
        "status": "BLOCKED_BACKPLANE_TRUNCATION_CONTROL",
        "contract_only": args.contract_only,
    }
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
        geometry = setup_case(
            fdtd,
            architecture=args.architecture,
            substrate_mode=args.substrate_mode,
            duration_ps=args.duration_ps,
            auto_shutoff_min=args.auto_shutoff_min,
        )
        fdtd.setresource("FDTD", 1, "active", 0)
        fdtd.setresource("FDTD", 2, "active", 1)
        fdtd.setresource("FDTD", 2, "processes", "1")
        fdtd.setresource("FDTD", 2, "threads", "8")
        fdtd.setresource("FDTD", 2, "device type", args.gpu_device)
        fdtd.setresource("FDTD", 2, "solver extra command line options", "-gpu")
        fdtd.runsetup()
        mesh = audit.mesh_readback(fdtd)
        result.update(
            {
                "geometry": geometry,
                "solver_version": str(fdtd.version()),
                "mesh_after_runsetup": {
                    key: value
                    for key, value in mesh.items()
                    if key != "coordinate_arrays"
                },
                "resource": {
                    prop: str(fdtd.getresource("FDTD", 2, prop))
                    for prop in (
                        "active",
                        "device type",
                        "processes",
                        "threads",
                        "solver extra command line options",
                    )
                },
            }
        )
        fdtd.save(str(fsp_path))
        if args.contract_only:
            result["status"] = "COMPLETED_BACKPLANE_RUNSETUP_AUDIT"
        else:
            started = time.monotonic()
            result["GPU_resource_used"] = audit.strict_gpu_run(
                fdtd,
                f"backplane_{args.architecture}_{args.substrate_mode}",
            )
            result["solver_wall_time_s"] = time.monotonic() - started
            source_power = scalar(
                fdtd.sourcepower(FREQUENCY_HZ, 2, SOURCE_NAME), "sourcepower"
            )
            top_signed = scalar(fdtd.transmission(TOP_MONITOR), TOP_MONITOR) * source_power
            bottom_signed = scalar(fdtd.transmission(BOTTOM_MONITOR), BOTTOM_MONITOR) * source_power
            p_flux_absorbed = bottom_signed - top_signed
            fdtd.runanalysis(PABS_GROUP)
            q = extract_native_yee_q(
                fdtd,
                field_monitor=PABS_FIELD,
                index_monitor=PABS_INDEX,
                wavelength_m=WAVELENGTH_M,
            )
            p_q = float(q["P_Q_W"])
            closure = abs(p_q - p_flux_absorbed) / max(
                abs(p_q), abs(p_flux_absorbed), 1.0e-300
            )
            top_fields = {
                component: np.asarray(
                    fdtd.getdata(TOP_MONITOR, f"E{component}", 1)
                ).squeeze()
                for component in "xyz"
            }
            finite = bool(
                all(np.all(np.isfinite(value)) for value in top_fields.values())
                and all(
                    np.all(np.isfinite(np.asarray(q["Q_components"][component])))
                    for component in "xyz"
                )
            )
            negative_q_cells = {
                component: int(
                    np.count_nonzero(np.asarray(q["Q_components"][component]) < 0.0)
                )
                for component in "xyz"
            }
            arrays: dict[str, np.ndarray] = {}
            for component in "xyz":
                arrays[f"top_E{component}"] = top_fields[component]
                arrays[f"Q{component}_W_m3"] = np.asarray(q["Q_components"][component])
                for axis in "xyz":
                    arrays[f"Q{component}_{axis}_m"] = np.asarray(
                        q["native_coordinates"][component][axis]
                    )
            arrays["top_x_m"] = np.asarray(fdtd.getdata(TOP_MONITOR, "x", 1), float)
            arrays["top_y_m"] = np.asarray(fdtd.getdata(TOP_MONITOR, "y", 1), float)
            np.savez_compressed(npz_path, **arrays)
            reflection = 1.0 + top_signed / source_power
            transmission = -bottom_signed / source_power
            absorptance = p_q / source_power
            result.update(
                {
                    "source_power_W": source_power,
                    "top_signed_z_power_W": top_signed,
                    "bottom_signed_z_power_W": bottom_signed,
                    "P_flux_absorbed_W": p_flux_absorbed,
                    "P_Q_W": p_q,
                    "Q_component_power_W": q["component_power_W"],
                    "closure_relative": closure,
                    "reflection": reflection,
                    "transmission": transmission,
                    "absorptance": absorptance,
                    "R_plus_T_plus_A_minus_1": reflection + transmission + absorptance - 1.0,
                    "all_arrays_finite": finite,
                    "negative_Q_cell_count": negative_q_cells,
                    "log_audit": audit.log_audit(output),
                }
            )
            gates = {
                "GPU_resource_selected": str(result["GPU_resource_used"]) != "",
                "closure_lt_0p5pct": closure < 0.005,
                "all_arrays_finite": finite,
                "no_negative_Q": sum(negative_q_cells.values()) == 0,
                "auto_shutoff_lt_1e_5": (
                    result["log_audit"]["final_auto_shutoff"] is not None
                    and result["log_audit"]["final_auto_shutoff"] < 1.0e-5
                ),
            }
            result["gates"] = gates
            result["status"] = (
                "COMPLETED_BACKPLANE_TRUNCATION_FORWARD"
                if all(gates.values())
                else "FAILED_BACKPLANE_TRUNCATION_FORWARD"
            )
            fdtd.save(str(fsp_path))
        artifacts = []
        for path in (fsp_path, npz_path):
            if path.is_file():
                artifacts.append(
                    {
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
        result["raw_artifacts"] = artifacts
    except Exception as exc:
        result["status"] = "BLOCKED_BACKPLANE_TRUNCATION_CONTROL"
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
    return 0 if result["status"].startswith("COMPLETED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
