#!/usr/bin/env python3
"""Add read-only mesh/material diagnostics to an existing completed FDTD case.

The completed Maxwell project is opened only to read its stored index-monitor
data.  No FDTD or HEAT solve is launched, and the project itself is not saved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

import config_stage1 as config
from lumerical_api import jsonable, select_installation, write_json
from pabs_diagnostic_utils import (
    mesh_and_material_metadata,
    save_epsilon_and_material_slices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument(
        "--geometry",
        required=True,
        choices=("none", "full-film", "centered-square", "centered-disk",
                 "centered-disk-gap"),
    )
    parser.add_argument("--lumerical-version", default="v261")
    parser.add_argument("--global-resolution", type=int, default=20)
    parser.add_argument("--design-resolution-xy", type=int, default=40)
    parser.add_argument("--flake-dz-nm", type=float, default=5.0)
    parser.add_argument("--design-gap-nm", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_dir = Path(args.case_dir).expanduser().resolve()
    project = case_dir / "fdtd_test.fsp"
    data_path = case_dir / "q_on_physical.npz"
    output_path = case_dir / "pabs_case_diagnostics.json"
    fdtd = None
    try:
        if not project.is_file() or not data_path.is_file():
            raise FileNotFoundError(f"missing completed case under {case_dir}")

        installation = select_installation(args.lumerical_version)
        os.environ["VC_LUMERICAL_ROOT"] = str(installation.root)
        os.environ["LUMERICAL_ROOT"] = str(installation.root)
        os.environ["TARGET_WL_UM"] = "4.0"
        os.environ["RES"] = str(args.global_resolution)
        os.environ["DESIGN_RES_XY"] = str(args.design_resolution_xy)
        os.environ["FILM_DZ_NM"] = str(args.flake_dz_nm)
        os.environ["ENABLE_LIVE_PLOTTING"] = "0"
        for path in (config.REPOSITORY_ROOT, config.REPOSITORY_ROOT / "bundle"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        import eqc_lib as runtime
        import tairte4_volume_model as model

        data = np.load(data_path)
        x = np.asarray(data["x_m"], float).reshape(-1)
        y = np.asarray(data["y_m"], float).reshape(-1)
        z = np.asarray(data["z_m"], float).reshape(-1)
        design_gap_m = args.design_gap_nm * 1e-9
        fdtd = runtime.open_control(project)
        metadata = mesh_and_material_metadata(
            fdtd, model, geometry=args.geometry, design_gap_m=design_gap_m
        )
        slices = save_epsilon_and_material_slices(
            fdtd,
            case_dir,
            x=x,
            y=y,
            z=z,
            geometry=args.geometry,
            disk_radius_m=config.FIXED_DESIGN_DISK_RADIUS_M,
            square_half_width_m=config.FIXED_DESIGN_DISK_RADIUS_M,
            design_gap_m=design_gap_m,
            design_height_m=float(model.design_h) * 1e-6,
        )
        result = {
            "source_project": str(project),
            "read_only_postprocess": True,
            "maxwell_rerun": False,
            "heat_run": False,
            "diagnostics": {
                **metadata,
                "solver_epsilon_material_slices": slices,
            },
        }
        write_json(output_path, result)
        print(json.dumps(jsonable(result), indent=2), flush=True)
        return 0
    except Exception as exc:
        write_json(
            output_path,
            {
                "status": "failed",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "maxwell_rerun": False,
                "heat_run": False,
            },
        )
        raise
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
