#!/usr/bin/env python3
"""Runsetup-only audit of the Run-002 coarse production candidate geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_FIELD,
    PABS_GROUP,
    PABS_INDEX,
)
from photothermal_pte.optimization_runs.gaussian10_contract import (  # noqa: E402
    silica_10um,
)

import run_complex_material_control as material_control  # noqa: E402


C0 = 299792458.0
WAVELENGTH_M = 10.0e-6
TAIRTE4_MATERIAL = "run002_TaIrTe4_paper_abc"
SIO2_MATERIAL = "run002_Kitamura_SiO2_10um"
SI_MATERIAL = "run002_Palik_Si_10um"
DESIGN_OBJECT = "run002_coarse_design_import"
DESIGN_BOUNDS = {"x": (-10e-6, 10e-6), "y": (-10e-6, 10e-6), "z": (0.0, 1e-6)}
Q_BOUNDS = {
    "x": (-20e-6, 20e-6),
    "y": (-20e-6, 20e-6),
    "z": (-1.25e-6, 1.25e-6),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complex_json(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def material_epsilon() -> dict[str, complex]:
    data = np.loadtxt(REPOSITORY / "photothermal_pte" / "bundle" / "perm_data.txt")
    order = np.argsort(data[:, 0])
    wavelength_nm = data[order, 0]
    eps_a_data = (data[:, 1] + 1j * data[:, 2])[order]
    eps_b_data = (data[:, 3] + 1j * data[:, 4])[order]

    def interpolate(values: np.ndarray, target_nm: float) -> complex:
        return complex(
            np.interp(target_nm, wavelength_nm, values.real),
            np.interp(target_nm, wavelength_nm, values.imag),
        )

    eps_a = interpolate(eps_a_data, 1e4)
    eps_b = interpolate(eps_b_data, 1e4)
    return {"x": eps_b, "y": eps_a, "z": eps_b}


def add_tairte4_material(fdtd: object) -> dict[str, object]:
    data = np.loadtxt(REPOSITORY / "photothermal_pte" / "bundle" / "perm_data.txt")
    order = np.argsort(data[:, 0])
    wavelength_nm = data[order, 0]
    eps_a_data = (data[:, 1] + 1j * data[:, 2])[order]
    eps_b_data = (data[:, 3] + 1j * data[:, 4])[order]
    samples_nm = np.linspace(2700.0, 13200.0, 600)

    def interpolate(values: np.ndarray) -> np.ndarray:
        return np.interp(samples_nm, wavelength_nm, values.real) + 1j * np.interp(
            samples_nm, wavelength_nm, values.imag
        )

    eps_a = interpolate(eps_a_data)
    eps_b = interpolate(eps_b_data)
    material = fdtd.addmaterial("Sampled 3D data")
    fdtd.setmaterial(material, "name", TAIRTE4_MATERIAL)
    fdtd.setmaterial(TAIRTE4_MATERIAL, "anisotropy", 1)
    fdtd.setmaterial(TAIRTE4_MATERIAL, "max coefficients", 20)
    fdtd.setmaterial(
        TAIRTE4_MATERIAL,
        "sampled data",
        np.column_stack((C0 / (samples_nm * 1e-9), eps_b, eps_a, eps_b)),
    )
    requested = material_epsilon()
    return {
        "name": TAIRTE4_MATERIAL,
        "axis_mapping": "x=b, y=a, z=c=b closure",
        "epsilon_at_10um": {
            axis: complex_json(value) for axis, value in requested.items()
        },
        "out_of_plane_limit": (
            "epsilon_c=epsilon_b is the explicit paper-consistent 3D closure; "
            "not an independently measured c-axis property"
        ),
    }


def add_isotropic_nk_material(fdtd: object, name: str, index: complex) -> None:
    material = fdtd.addmaterial("(n,k) Material")
    fdtd.setmaterial(material, "name", name)
    fdtd.setmaterial(name, "Refractive Index", float(index.real))
    fdtd.setmaterial(name, "Imaginary Refractive Index", float(index.imag))


def add_substrate_materials(fdtd: object) -> dict[str, object]:
    silica = silica_10um()
    silica_n = complex(silica["n_real"], silica["n_imag"])
    silicon_n = complex(
        np.asarray(fdtd.getindex("Si (Silicon) - Palik", C0 / WAVELENGTH_M)).reshape(-1)[0]
    )
    add_isotropic_nk_material(fdtd, SIO2_MATERIAL, silica_n)
    add_isotropic_nk_material(fdtd, SI_MATERIAL, silicon_n)
    return {
        "SiO2": {
            "material": SIO2_MATERIAL,
            "n_at_10um": complex_json(silica_n),
            "epsilon_at_10um": complex_json(silica_n**2),
            "provenance": "Kitamura-2007 closure already frozen in Run 002",
        },
        "Si": {
            "material": SI_MATERIAL,
            "n_at_10um": complex_json(silicon_n),
            "provenance": "installed Lumerical v261 Palik database readback at 10 um",
        },
    }


def add_rect(fdtd: object, name: str, material: str, bounds: dict[str, tuple[float, float]]) -> None:
    item = fdtd.addrect()
    item["name"] = name
    item["material"] = material
    for axis in "xyz":
        item[f"{axis} min"], item[f"{axis} max"] = bounds[axis]


def design_nodes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.linspace(*DESIGN_BOUNDS["x"], 201),
        np.linspace(*DESIGN_BOUNDS["y"], 201),
        np.linspace(*DESIGN_BOUNDS["z"], 21),
    )


def add_design(fdtd: object, rho: float = 0.5) -> dict[str, object]:
    x, y, z = design_nodes()
    epsilon_sio2 = complex(
        silica_10um()["epsilon_real"], silica_10um()["epsilon_imag"]
    )
    epsilon = 1.0 + rho * (epsilon_sio2 - 1.0)
    index = complex(np.sqrt(epsilon))
    values = np.full((x.size, y.size, z.size), index, complex)
    fdtd.addimport({"name": DESIGN_OBJECT, "x": 0.0, "y": 0.0, "z": 0.0})
    if int(fdtd.importnk2(values, x, y, z)) != 1:
        raise RuntimeError("production-candidate importnk2 failed")
    return {
        "name": DESIGN_OBJECT,
        "rho": rho,
        "node_shape": list(values.shape),
        "node_spacing_m": {
            "x": float(x[1] - x[0]),
            "y": float(y[1] - y[0]),
            "z": float(z[1] - z[0]),
        },
        "bounds_m": {axis: list(values) for axis, values in DESIGN_BOUNDS.items()},
        "epsilon": complex_json(epsilon),
        "index": complex_json(index),
    }


def add_mesh(fdtd: object, name: str, bounds: dict[str, tuple[float, float]], **steps: float) -> None:
    mesh = fdtd.addmesh()
    mesh["name"] = name
    for axis in "xyz":
        mesh[f"{axis} min"], mesh[f"{axis} max"] = bounds[axis]
        mesh[f"override {axis} mesh"] = axis in steps
        if axis in steps:
            mesh[f"d{axis}"] = steps[axis]


def add_absorption_and_flux(fdtd: object) -> dict[str, float]:
    pabs = fdtd.addobject("pabs_adv")
    pabs["name"] = PABS_GROUP
    for axis in "xyz":
        low, high = Q_BOUNDS[axis]
        pabs[axis] = 0.5 * (low + high)
        pabs[f"{axis} span"] = high - low
    signs = {}
    for axis in "xyz":
        for side, position in zip(("min", "max"), Q_BOUNDS[axis]):
            name = f"run002_production_flux_{axis}_{side}"
            monitor = fdtd.addpower()
            monitor["name"] = name
            monitor["monitor type"] = f"2D {axis.upper()}-normal"
            monitor[axis] = position
            for transverse in "xyz":
                if transverse != axis:
                    monitor[f"{transverse} min"] = Q_BOUNDS[transverse][0]
                    monitor[f"{transverse} max"] = Q_BOUNDS[transverse][1]
            monitor["override global monitor settings"] = True
            monitor["use source limits"] = False
            monitor["use wavelength spacing"] = True
            monitor["wavelength center"] = WAVELENGTH_M
            monitor["wavelength span"] = 0.0
            monitor["frequency points"] = 1
            signs[name] = -1.0 if side == "min" else 1.0
    return signs


def named_bounds(fdtd: object, name: str) -> dict[str, list[float]]:
    return {
        axis: [
            float(fdtd.getnamed(name, f"{axis} min")),
            float(fdtd.getnamed(name, f"{axis} max")),
        ]
        for axis in "xyz"
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    project = output / "production_candidate_runsetup.fsp"
    result_path = output / "production_candidate_geometry_audit.json"
    result: dict[str, object] = {
        "status": "BLOCKED_RUN002_PRODUCTION_CANDIDATE_GEOMETRY",
        "Maxwell_solve": False,
        "thermal_solve": False,
        "adjoint_solve": False,
        "optimization_run": False,
    }
    fdtd = None
    try:
        wrapper = material_control.load_source_wrapper()
        audit = wrapper.source_audit
        os.environ["VC_LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
        os.environ["LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
        os.environ["LUMERICAL_PYTHONPATH"] = str(audit.APPROVED_API)
        os.environ["PATH"] = f"{audit.APPROVED_ROOT / 'bin'}:{os.environ.get('PATH','')}"
        for path in (audit.STAGE1, REPOSITORY / "photothermal_pte"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        helper = audit.load_module(audit.API_HELPER, "run002_production_geometry_api")
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
        source_contract = audit.setup(fdtd, 8.0, 1e-7, 3, 8.36043075475035e-6, 8.5e-6)
        source_contract["source"]["model"] = (
            "Run-002 calibrated scalar Gaussian at 10 um; single-frequency FOM contract"
        )
        source_contract["source"]["source_object_waist_calibration"] = {
            "method": "one_step_from_run002_homogeneous_air_source_only_measurement",
            "input_source_object_waist_m": 8.36043075475035e-6,
            "measured_target_plane_effective_waist_m": 8.506397119765702e-6,
            "source_only_field_NPZ_sha256": (
                "d95e5cd2758c2d0cfba52a00da955619f65af2ca31befc4479d2730ae96006f6"
            ),
            "legacy_11um_12um_calibration_reused": False,
            "Q_clipping_smoothing_gain_or_rescaling": False,
        }
        tairte4 = add_tairte4_material(fdtd)
        substrate = add_substrate_materials(fdtd)
        add_rect(
            fdtd,
            "run002_Si_substrate",
            SI_MATERIAL,
            {"x": (-30e-6, 30e-6), "y": (-30e-6, 30e-6), "z": (-8e-6, -0.385e-6)},
        )
        add_rect(
            fdtd,
            "run002_bottom_SiO2",
            SIO2_MATERIAL,
            {"x": (-30e-6, 30e-6), "y": (-30e-6, 30e-6), "z": (-0.385e-6, -0.100e-6)},
        )
        add_rect(
            fdtd,
            "run002_extended_TaIrTe4",
            TAIRTE4_MATERIAL,
            {"x": (-30e-6, 30e-6), "y": (-30e-6, 30e-6), "z": (-0.100e-6, 0.0)},
        )
        design = add_design(fdtd)
        add_mesh(
            fdtd,
            "run002_coarse_design_xy_mesh",
            {"x": (-10.1e-6, 10.1e-6), "y": (-10.1e-6, 10.1e-6), "z": (0.0, 1.0e-6)},
            x=100e-9,
            y=100e-9,
            z=50e-9,
        )
        add_mesh(
            fdtd,
            "run002_illuminated_xy_mesh",
            {"x": (-20e-6, 20e-6), "y": (-20e-6, 20e-6), "z": (-0.5e-6, 0.1e-6)},
            x=200e-9,
            y=200e-9,
        )
        add_mesh(
            fdtd,
            "run002_flake_z_mesh",
            {"x": (-20e-6, 20e-6), "y": (-20e-6, 20e-6), "z": (-0.100e-6, 0.0)},
            z=10e-9,
        )
        flux_signs = add_absorption_and_flux(fdtd)
        fdtd.runsetup()
        mesh = audit.mesh_readback(fdtd)
        fdtd.save(str(project))
        geometry = {
            name: named_bounds(fdtd, name)
            for name in (
                "FDTD",
                audit.SOURCE_NAME,
                "run002_Si_substrate",
                "run002_bottom_SiO2",
                "run002_extended_TaIrTe4",
                DESIGN_OBJECT,
                PABS_FIELD,
                PABS_INDEX,
            )
        }
        minimum = mesh["minimum_step_m"]
        pabs_bounds_match = all(
            np.allclose(
                geometry[name][axis],
                Q_BOUNDS[axis],
                rtol=0.0,
                atol=2e-18,
            )
            for name in (PABS_FIELD, PABS_INDEX)
            for axis in "xyz"
        )
        layer_interfaces_match = bool(
            abs(geometry["run002_Si_substrate"]["z"][1] + 0.385e-6) < 2e-18
            and abs(geometry["run002_bottom_SiO2"]["z"][0] + 0.385e-6) < 2e-18
            and abs(geometry["run002_bottom_SiO2"]["z"][1] + 0.100e-6) < 2e-18
            and abs(geometry["run002_extended_TaIrTe4"]["z"][0] + 0.100e-6) < 2e-18
            and geometry["run002_extended_TaIrTe4"]["z"][1] == 0.0
            and geometry[DESIGN_OBJECT]["z"][0] == 0.0
        )
        passed = bool(
            design["node_shape"] == [201, 201, 21]
            and minimum["x"] <= 100e-9 + 1e-18
            and minimum["y"] <= 100e-9 + 1e-18
            and minimum["z"] <= 10e-9 + 1e-18
            and geometry["run002_extended_TaIrTe4"]["x"] == [-30e-6, 30e-6]
            and geometry["run002_extended_TaIrTe4"]["y"] == [-30e-6, 30e-6]
            and pabs_bounds_match
            and layer_interfaces_match
        )
        result = {
            "status": "VALIDATED_RUN002_PRODUCTION_CANDIDATE_RUNSETUP" if passed else "FAILED_RUN002_PRODUCTION_CANDIDATE_RUNSETUP",
            "passed": passed,
            "scope": "runsetup/geometry/mesh/material audit only; no field solve",
            "source_contract": source_contract,
            "materials": {"TaIrTe4": tairte4, **substrate},
            "design": design,
            "geometry_readback_m": geometry,
            "Q_control_volume_m": {axis: list(values) for axis, values in Q_BOUNDS.items()},
            "pabs_field_index_bounds_match_Q_control_volume": pabs_bounds_match,
            "layer_interfaces_exactly_contiguous": layer_interfaces_match,
            "flux_monitor_signs": flux_signs,
            "mesh_readback": {key: value for key, value in mesh.items() if key != "coordinate_arrays"},
            "project": {
                "path": str(project),
                "size_bytes": project.stat().st_size,
                "sha256": sha256(project),
            },
            "Maxwell_solve": False,
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
