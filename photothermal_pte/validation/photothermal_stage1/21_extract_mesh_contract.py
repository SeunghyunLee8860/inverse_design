#!/usr/bin/env python3
"""Export the exact FDTD mesh/conformal/override contract from a completed case."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

import config_stage1 as config
from lumerical_api import select_installation, write_json


def safe(fdtd: Any, object_name: str, prop: str) -> Any:
    try:
        value = fdtd.getnamed(object_name, prop)
        array = np.asarray(value)
        if array.size == 1:
            item = array.reshape(-1)[0]
            if np.iscomplexobj(item):
                return {"real": float(np.real(item)), "imag": float(np.imag(item))}
            if isinstance(item, (str, np.str_)):
                return str(item)
            return float(item)
        return array.tolist()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def mesh_object(fdtd: Any, name: str) -> dict[str, Any]:
    exists = int(fdtd.getnamednumber(name))
    result: dict[str, Any] = {"name": name, "exists": bool(exists)}
    if not exists:
        return result
    for prop in (
        "enabled",
        "x min", "x max", "y min", "y max", "z min", "z max",
        "x", "x span", "y", "y span", "z", "z span",
        "override x mesh", "override y mesh", "override z mesh",
        "dx", "dy", "dz",
    ):
        result[prop] = safe(fdtd, name, prop)
    return result


def axis_contract(axis: np.ndarray) -> dict[str, Any]:
    values = np.asarray(axis, float).reshape(-1)
    spacing = np.diff(values)
    rounded = np.unique(np.round(spacing * 1e12, 6)) * 1e-12
    return {
        "count": int(values.size),
        "min_m": float(values[0]),
        "max_m": float(values[-1]),
        "minimum_step_m": float(np.min(spacing)),
        "maximum_step_m": float(np.max(spacing)),
        "unique_steps_m_rounded_1e-18": rounded.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lumerical-version", choices=("v251", "v261"), default="v261")
    args = parser.parse_args()
    project = Path(args.project).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    installation = select_installation(args.lumerical_version)
    os.environ["VC_LUMERICAL_ROOT"] = str(installation.root)
    os.environ["LUMERICAL_ROOT"] = str(installation.root)
    for path in (config.REPOSITORY_ROOT, config.REPOSITORY_ROOT / "bundle"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import eqc_lib as runtime

    fdtd = runtime.open_control(project)
    try:
        field = "stage1_pabs_adv::field"
        x = np.asarray(fdtd.getdata(field, "x", 1), float).reshape(-1)
        y = np.asarray(fdtd.getdata(field, "y", 1), float).reshape(-1)
        z = np.asarray(fdtd.getdata(field, "z", 1), float).reshape(-1)
        solver = {
            prop: safe(fdtd, "FDTD", prop)
            for prop in (
                "mesh type",
                "mesh refinement",
                "mesh accuracy",
                "min mesh step",
                "simulation time",
                "auto shutoff min",
            )
        }
        objects = [
            mesh_object(fdtd, name)
            for name in (
                "global_uniform_mesh",
                "design_mesh",
                "flake_mesh",
                "fixed_stack_z_mesh",
                "air_bulk_z_mesh",
                "si_bulk_z_mesh",
                "source_z_registration_mesh",
            )
        ]
        geometry = {}
        for name in ("design", "TaIrTe4_flake"):
            geometry[name] = {
                "mesh order": safe(fdtd, name, "mesh order"),
                "override mesh order from material database": safe(
                    fdtd, name, "override mesh order from material database"
                ),
                "material": safe(fdtd, name, "material"),
                "z min": safe(fdtd, name, "z min"),
                "z max": safe(fdtd, name, "z max"),
            }
    finally:
        fdtd.close()
    result = {
        "project": str(project),
        "solver": solver,
        "mesh_objects": objects,
        "raw_native_pabs_field_axes": {
            "x": axis_contract(x),
            "y": axis_contract(y),
            "z": axis_contract(z),
        },
        "geometry_mesh_order": geometry,
        "interpretation": {
            "global_uniform_mesh": "whole-domain 50 nm override",
            "design_mesh": "local design-grid override; 25 nm x/y and 50 nm z in baseline",
            "flake_mesh": "z-only 5 nm override from 50 nm below the flake through z=0",
            "conformal_setting": (
                "The FDTD 'mesh refinement' string is the active conformal "
                "algorithm. 'precise volume average' is not conformal variant 1 or 2."
            ),
            "density_grid_warning": (
                "The imported design density grid is not by itself the complete "
                "Yee mesh; the mesh-object overrides and raw native axes are authoritative."
            ),
        },
        "heat_run": False,
        "clipping": False,
        "flux_gain": False,
    }
    write_json(output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
