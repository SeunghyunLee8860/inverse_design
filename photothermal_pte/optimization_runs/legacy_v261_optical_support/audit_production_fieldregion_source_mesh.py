#!/usr/bin/env python3
"""Runsetup-only audit of production FieldRegion/source mesh coordinates."""

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
for path in (HERE, REPOSITORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_complex_material_control as material_control  # noqa: E402


FIELD_REGION = "run002_component_yee_adjoint_region"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mesh_coordinate(fdtd, axis: str) -> np.ndarray:
    raw = fdtd.getresult("FDTD", axis)
    if isinstance(raw, dict):
        raw = raw[axis]
    value = np.asarray(raw, float).reshape(-1)
    if value.size < 3 or np.any(np.diff(value) <= 0.0):
        raise RuntimeError(f"invalid FDTD {axis} mesh")
    return value


def nearest_record(recorded: np.ndarray, mesh: np.ndarray) -> dict:
    insertion = np.searchsorted(mesh, recorded)
    insertion = np.clip(insertion, 1, mesh.size - 1)
    left = mesh[insertion - 1]
    right = mesh[insertion]
    nearest = np.where(np.abs(recorded - left) <= np.abs(recorded - right), left, right)
    difference = recorded - nearest
    return {
        "recorded_count": int(recorded.size),
        "recorded_bounds_m": [float(recorded[0]), float(recorded[-1])],
        "mesh_count": int(mesh.size),
        "mesh_bounds_m": [float(mesh[0]), float(mesh[-1])],
        "maximum_nearest_mesh_mismatch_m": float(np.max(np.abs(difference))),
        "nonmatching_recorded_count_at_2e-18m": int(
            np.count_nonzero(np.abs(difference) > 2.0e-18)
        ),
        "nearest_mesh_coordinates_m": nearest,
        "difference_m": difference,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completed-fsp", type=Path, required=True)
    parser.add_argument("--completed-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project = args.completed_fsp.expanduser().resolve()
    if sha256(project) != args.completed_sha256:
        raise RuntimeError("completed FSP SHA mismatch")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing non-empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "production_fieldregion_source_mesh_audit.json"
    result = {
        "status": "BLOCKED_PRODUCTION_FIELDREGION_SOURCE_MESH_AUDIT",
        "passed": False,
        "Maxwell_timestepping_solves": 0,
        "thermal_solves": 0,
        "optimization_iterations": 0,
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
        helper = audit.load_module(audit.API_HELPER, "run002_fieldregion_mesh_api")
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
        import eqc_lib as runtime

        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        fdtd.load(str(project))
        recorded_dataset = fdtd.getresult(FIELD_REGION, "E")
        recorded_grid = {
            axis: np.asarray(recorded_dataset[axis], float).reshape(-1)
            for axis in "xyz"
        }
        bounds = {
            axis: [
                float(fdtd.getnamed(FIELD_REGION, f"{axis} min")),
                float(fdtd.getnamed(FIELD_REGION, f"{axis} max")),
            ]
            for axis in "xyz"
        }
        fdtd.switchtolayout()
        runtime.configure_session_resources(fdtd)
        fdtd.runsetup()
        mesh = {axis: mesh_coordinate(fdtd, axis) for axis in "xyz"}
        axes = {
            axis: nearest_record(
                np.asarray(recorded_grid[axis], float), mesh[axis]
            )
            for axis in "xyz"
        }
        exact_subsets = {}
        for axis in "xyz":
            low, high = bounds[axis]
            selected = mesh[axis][
                (mesh[axis] >= low - 2.0e-18)
                & (mesh[axis] <= high + 2.0e-18)
            ]
            exact_subsets[axis] = {
                "count": int(selected.size),
                "bounds_m": (
                    [float(selected[0]), float(selected[-1])]
                    if selected.size
                    else None
                ),
            }
        npz_path = output / "production_fieldregion_and_fdtd_mesh_coordinates.npz"
        np.savez_compressed(
            npz_path,
            **{f"recorded_{axis}_m": recorded_grid[axis] for axis in "xyz"},
            **{f"mesh_{axis}_m": mesh[axis] for axis in "xyz"},
            **{
                f"recorded_{axis}_nearest_mesh_m": axes[axis][
                    "nearest_mesh_coordinates_m"
                ]
                for axis in "xyz"
            },
        )
        for axis in "xyz":
            axes[axis].pop("nearest_mesh_coordinates_m")
            axes[axis].pop("difference_m")
        result = {
            "status": "COMPLETED_PRODUCTION_FIELDREGION_SOURCE_MESH_AUDIT",
            "passed": True,
            "completed_FSP": {
                "path": str(project),
                "size_bytes": project.stat().st_size,
                "sha256": sha256(project),
            },
            "fieldregion_bounds_m": bounds,
            "recorded_to_layout_mesh": axes,
            "layout_mesh_points_inside_fieldregion_bounds": exact_subsets,
            "artifact": {
                "path": str(npz_path),
                "size_bytes": npz_path.stat().st_size,
                "sha256": sha256(npz_path),
            },
            "Maxwell_timestepping_solves": 0,
            "thermal_solves": 0,
            "optimization_iterations": 0,
        }
    except Exception as exc:
        result.update(
            {
                "status": "FAILED_PRODUCTION_FIELDREGION_SOURCE_MESH_AUDIT",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
