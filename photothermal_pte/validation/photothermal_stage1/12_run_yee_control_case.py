#!/usr/bin/env python3
"""Run an optical control with optional isotropic TaIrTe4 and six-face flux."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

import config_stage1 as config
from lumerical_api import write_json


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--fixed-geometry",
        required=True,
        choices=("centered-disk", "centered-square"),
    )
    parser.add_argument("--flake-isotropic-axis", choices=("a", "b"))
    parser.add_argument("--lumerical-version", default="v261")
    parser.add_argument("--global-resolution", type=int, default=20)
    parser.add_argument("--design-resolution-xy", type=int, default=40)
    parser.add_argument("--flake-dz-nm", type=float, default=5.0)
    args = parser.parse_args()

    exporter_path = Path(__file__).with_name("02_export_fdtd_qon.py")
    spec = importlib.util.spec_from_file_location("fdtd_qon_exporter", exporter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {exporter_path}")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)

    original_enable = exporter.enable_periodic_correction
    box_z_min_m = -200e-9
    box_z_max_m = 100e-9

    def configure_control(fdtd: object, name: str) -> dict[str, object]:
        import tairte4_volume_model as model

        if args.flake_isotropic_axis:
            wavelengths_nm = np.linspace(
                min(model.target_wl) * 900,
                max(model.target_wl) * 1100,
                400,
            )
            frequencies_hz = model.c0 / (wavelengths_nm * 1e-9)
            epsilon = model.eps_flake(
                wavelengths_nm, args.flake_isotropic_axis
            )
            sampled = np.column_stack(
                (frequencies_hz, epsilon, epsilon, epsilon)
            )
            fdtd.setmaterial("TaIrTe4_ani", "anisotropy", 1)
            fdtd.setmaterial("TaIrTe4_ani", "sampled data", sampled)

        x_face = 0.5 * float(model.Sx) * 1e-6
        y_face = 0.5 * float(model.Sy) * 1e-6
        add_side_monitor(
            fdtd,
            exporter,
            name=SIDE_MONITORS["x_min"],
            normal="x",
            position_m=-x_face,
            transverse_span_m=float(model.Sy) * 1e-6,
            z_min_m=box_z_min_m,
            z_max_m=box_z_max_m,
        )
        add_side_monitor(
            fdtd,
            exporter,
            name=SIDE_MONITORS["x_max"],
            normal="x",
            position_m=x_face,
            transverse_span_m=float(model.Sy) * 1e-6,
            z_min_m=box_z_min_m,
            z_max_m=box_z_max_m,
        )
        add_side_monitor(
            fdtd,
            exporter,
            name=SIDE_MONITORS["y_min"],
            normal="y",
            position_m=-y_face,
            transverse_span_m=float(model.Sx) * 1e-6,
            z_min_m=box_z_min_m,
            z_max_m=box_z_max_m,
        )
        add_side_monitor(
            fdtd,
            exporter,
            name=SIDE_MONITORS["y_max"],
            normal="y",
            position_m=y_face,
            transverse_span_m=float(model.Sx) * 1e-6,
            z_min_m=box_z_min_m,
            z_max_m=box_z_max_m,
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
            key: scalar(fdtd.transmission(name))
            for key, name in SIDE_MONITORS.items()
        }
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
    axis = args.flake_isotropic_axis
    summary["fixed_geometry"]["flake_material_control"] = (
        "original_anisotropic" if axis is None else f"isotropic_eps_{axis}"
    )
    if axis is not None:
        epsilon = complex(model.eps_flake(4000.0, axis))
        index = complex(np.sqrt(epsilon))
        summary["diagnostics"]["material_at_4um"][
            "TaIrTe4_epsilon_diagonal"
        ] = [
            {"real": epsilon.real, "imag": epsilon.imag}
            for _ in range(3)
        ]
        summary["diagnostics"]["material_at_4um"][
            "TaIrTe4_refractive_index_diagonal"
        ] = [
            {"real": index.real, "imag": index.imag}
            for _ in range(3)
        ]
    summary["control_wrapper"] = {
        "six_face_flux_enabled": True,
        "flake_isotropic_axis": axis,
        "maxwell_run": True,
        "heat_run": False,
        "clipping": False,
        "flux_gain": False,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
