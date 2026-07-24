#!/usr/bin/env python3
"""Run one optical-only constitutive/Pabs diagnostic case.

This wrapper deliberately leaves the installed pabs_adv expression unchanged.
It varies only source polarization, TaIrTe4 material representation, the
imaginary part of epsilon_y, simulation time, geometry, and solver version.
HEAT, clipping, and flux/Pabs gain are never invoked.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from lumerical_api import write_json


EPS0 = 8.8541878128e-12
WAVELENGTH_M = 4.0e-6
SIDE_MONITORS = {
    "x_min": "stage1_flux_xmin",
    "x_max": "stage1_flux_xmax",
    "y_min": "stage1_flux_ymin",
    "y_max": "stage1_flux_ymax",
}


def scalar(value: object) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise RuntimeError(f"expected scalar, got {array.shape}")
    return float(np.real(array[0]))


def complex_json(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def add_side_monitor(
    fdtd: object,
    exporter: object,
    *,
    name: str,
    normal: str,
    position_m: float,
    transverse_span_m: float,
    z_min_m: float,
    z_max_m: float,
) -> None:
    monitor = fdtd.addpower()
    monitor["name"] = name
    monitor["monitor type"] = f"2D {normal.upper()}-normal"
    monitor[normal] = position_m
    transverse = "y" if normal == "x" else "x"
    monitor[transverse] = 0.0
    monitor[f"{transverse} span"] = transverse_span_m
    monitor["z"] = 0.5 * (z_min_m + z_max_m)
    monitor["z span"] = z_max_m - z_min_m
    exporter.configure_single_frequency(fdtd, name)


def sampled_epsilon(model: object, scale_y_imag: float) -> tuple[np.ndarray, list[complex]]:
    wavelengths_nm = np.linspace(
        min(model.target_wl) * 900.0,
        max(model.target_wl) * 1100.0,
        400,
    )
    frequencies_hz = model.c0 / (wavelengths_nm * 1e-9)
    eps_x = np.asarray(model.eps_flake(wavelengths_nm, "a"), complex)
    eps_y_raw = np.asarray(model.eps_flake(wavelengths_nm, "b"), complex)
    eps_y = eps_y_raw.real + 1j * scale_y_imag * eps_y_raw.imag
    eps_z = np.full_like(eps_x, complex(model.eps_c_flake))
    table = np.column_stack((frequencies_hz, eps_x, eps_y, eps_z))
    target = [
        complex(model.eps_flake(4000.0, "a")),
        complex(model.eps_flake(4000.0, "b").real)
        + 1j * scale_y_imag * complex(model.eps_flake(4000.0, "b")).imag,
        complex(model.eps_c_flake),
    ]
    return table, target


def install_analytic_diagonal(fdtd: object, epsilon: list[complex]) -> dict[str, object]:
    """Install a non-fitted diagonal Conductive material exact at 4 um."""
    omega = 2.0 * np.pi * 299792458.0 / WAVELENGTH_M
    material = fdtd.addmaterial("Conductive")
    name = "TaIrTe4_diag_analytic_4um"
    fdtd.setmaterial(material, "name", name)
    fdtd.setmaterial(name, "anisotropy", 1)
    relative_permittivity = np.asarray([v.real for v in epsilon], float)
    conductivity_s_m = omega * EPS0 * np.asarray([v.imag for v in epsilon], float)
    fdtd.setmaterial(name, "permittivity", relative_permittivity)
    fdtd.setmaterial(name, "conductivity", conductivity_s_m)
    fdtd.setnamed("TaIrTe4_flake", "material", name)
    return {
        "name": name,
        "model": "Conductive diagonal analytic",
        "relative_permittivity_real": relative_permittivity.tolist(),
        "conductivity_S_m": conductivity_s_m.tolist(),
        "target_relative_permittivity_at_4um": [
            complex_json(v) for v in epsilon
        ],
        "frequency_exact_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--fixed-geometry",
        required=True,
        choices=("none", "centered-disk", "centered-square"),
    )
    parser.add_argument("--polarization-deg", type=float, default=0.0)
    parser.add_argument(
        "--material-control",
        choices=("sampled", "analytic-diagonal"),
        default="sampled",
    )
    parser.add_argument("--eps-y-imag-scale", type=float, default=1.0)
    parser.add_argument("--simulation-time-ps", type=float, default=4.0)
    parser.add_argument(
        "--disable-auto-shutoff",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--lumerical-version", choices=("v251", "v261"), default="v261")
    parser.add_argument("--solver-engine", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--global-resolution", type=int, default=20)
    parser.add_argument("--design-resolution-xy", type=int, default=40)
    parser.add_argument("--flake-dz-nm", type=float, default=5.0)
    args = parser.parse_args()
    if args.simulation_time_ps <= 0:
        parser.error("--simulation-time-ps must be positive")
    if args.eps_y_imag_scale < 0:
        parser.error("--eps-y-imag-scale must be nonnegative")

    exporter_path = Path(__file__).with_name("02_export_fdtd_qon.py")
    spec = importlib.util.spec_from_file_location("fdtd_qon_exporter", exporter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {exporter_path}")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)

    original_enable = exporter.enable_periodic_correction
    box_z_min_m = -200e-9
    box_z_max_m = 100e-9
    control_contract: dict[str, object] = {}

    def configure_control(fdtd: object, name: str) -> dict[str, object]:
        import tairte4_volume_model as model
        import eqc_lib as runtime

        if args.solver_engine == "CPU":
            def run_cpu(session: object, run_name: str) -> str:
                errors = []
                for resource in ("Local Host", "localhost", "Local Computer"):
                    try:
                        print(f"[run] {run_name} on CPU resource {resource!r}", flush=True)
                        session.run("FDTD", "CPU", resource)
                        return f"CPU:{resource}"
                    except Exception as exc:
                        errors.append(f"{resource}: {exc}")
                raise RuntimeError("CPU session run failed: " + " | ".join(errors))
            runtime.run_session = run_cpu

        fdtd.setnamed("source", "polarization angle", float(args.polarization_deg))
        fdtd.setnamed(
            "FDTD", "simulation time", float(args.simulation_time_ps) * 1e-12
        )
        auto_shutoff_setting: object = "unchanged"
        if args.disable_auto_shutoff:
            fdtd.setnamed("FDTD", "auto shutoff min", 0.0)
            auto_shutoff_setting = 0.0

        sampled, target_epsilon = sampled_epsilon(
            model, float(args.eps_y_imag_scale)
        )
        material_contract: dict[str, object]
        if args.material_control == "sampled":
            fdtd.setmaterial("TaIrTe4_ani", "anisotropy", 1)
            fdtd.setmaterial("TaIrTe4_ani", "sampled data", sampled)
            material_contract = {
                "name": "TaIrTe4_ani",
                "model": "Sampled 3D data",
                "max_coefficients": int(model.MAX_COEFFS),
                "eps_y_imag_scale": float(args.eps_y_imag_scale),
                "target_relative_permittivity_at_4um": [
                    complex_json(v) for v in target_epsilon
                ],
            }
        else:
            material_contract = install_analytic_diagonal(
                fdtd, target_epsilon
            )
            material_contract["eps_y_imag_scale"] = float(
                args.eps_y_imag_scale
            )

        x_face = 0.5 * float(model.Sx) * 1e-6
        y_face = 0.5 * float(model.Sy) * 1e-6
        for key, normal, position, span in (
            ("x_min", "x", -x_face, float(model.Sy) * 1e-6),
            ("x_max", "x", x_face, float(model.Sy) * 1e-6),
            ("y_min", "y", -y_face, float(model.Sx) * 1e-6),
            ("y_max", "y", y_face, float(model.Sx) * 1e-6),
        ):
            add_side_monitor(
                fdtd,
                exporter,
                name=SIDE_MONITORS[key],
                normal=normal,
                position_m=position,
                transverse_span_m=span,
                z_min_m=box_z_min_m,
                z_max_m=box_z_max_m,
            )
        control_contract.update(
            {
                "source_polarization_angle_deg": float(args.polarization_deg),
                "solver_engine": args.solver_engine,
                "simulation_time_s": float(args.simulation_time_ps) * 1e-12,
                "auto_shutoff_min": auto_shutoff_setting,
                "material": material_contract,
            }
        )
        return original_enable(fdtd, name)

    exporter.enable_periodic_correction = configure_control
    sys.argv = [
        str(exporter_path),
        "--lumerical-version",
        args.lumerical_version,
        "--output-dir",
        args.output_dir,
        "--fixed-geometry",
        args.fixed_geometry,
        "--global-resolution",
        str(args.global_resolution),
        "--design-resolution-xy",
        str(args.design_resolution_xy),
        "--flake-dz-nm",
        str(args.flake_dz_nm),
        "--pabs-z-padding-nm",
        "50",
        "--allow-nonpositive-diagnostic-flux",
        "--hide-gui",
    ]
    exporter.main()

    output = Path(args.output_dir).expanduser().resolve() / "fdtd_qon"
    project = output / "fdtd_test.fsp"
    summary_path = output / "fdtd_absorption_summary.json"
    summary = json.loads(summary_path.read_text())

    import eqc_lib as runtime
    import tairte4_volume_model as model

    fdtd = runtime.open_control(project)
    try:
        side = {
            key: scalar(fdtd.transmission(monitor))
            for key, monitor in SIDE_MONITORS.items()
        }
        actual_source_angle = scalar(
            fdtd.getnamed("source", "polarization angle")
        )
        actual_simulation_time = scalar(
            fdtd.getnamed("FDTD", "simulation time")
        )
        actual_auto_shutoff = scalar(
            fdtd.getnamed("FDTD", "auto shutoff min")
        )
    finally:
        fdtd.close()

    absorption = summary["absorption"]
    a_z = (
        float(absorption["local_flux_bottom_signed"])
        - float(absorption["local_flux_top_signed"])
    )
    a_x = side["x_min"] - side["x_max"]
    a_y = side["y_min"] - side["y_max"]
    a_six = a_x + a_y + a_z
    absorption["six_face_flux"] = {
        "signed_transmission": side,
        "A_x_pair": a_x,
        "A_y_pair": a_y,
        "A_top_bottom": a_z,
        "A_six_face_net": a_six,
        "six_face_minus_top_bottom": a_six - a_z,
        "box": {
            "x_min_m": -0.5 * float(model.Sx) * 1e-6,
            "x_max_m": 0.5 * float(model.Sx) * 1e-6,
            "y_min_m": -0.5 * float(model.Sy) * 1e-6,
            "y_max_m": 0.5 * float(model.Sy) * 1e-6,
            "z_min_m": box_z_min_m,
            "z_max_m": box_z_max_m,
        },
    }
    polarization_label = (
        "x" if np.isclose(args.polarization_deg, 0.0)
        else "y" if np.isclose(args.polarization_deg, 90.0)
        else f"linear_{args.polarization_deg:g}deg"
    )
    summary["source"]["polarization"] = polarization_label
    summary["source"]["polarization_angle_deg"] = actual_source_angle
    summary["simulation_time_s"] = actual_simulation_time
    summary["fixed_geometry"]["flake_material_control"] = (
        args.material_control
    )
    summary["control_wrapper"] = {
        **control_contract,
        "actual_source_polarization_angle_deg": actual_source_angle,
        "actual_simulation_time_s": actual_simulation_time,
        "actual_auto_shutoff_min": actual_auto_shutoff,
        "six_face_flux_enabled": True,
        "maxwell_run": True,
        "heat_run": False,
        "clipping": False,
        "flux_gain": False,
        "pabs_script_modified_beyond_periodic_switch": False,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
