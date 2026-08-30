#!/usr/bin/env python3
"""Inspect a failed adjoint template without Maxwell timestepping."""

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
    return np.asarray(raw, float).reshape(-1)


def nearest(coordinate: np.ndarray, candidate: np.ndarray) -> dict:
    index = np.searchsorted(candidate, coordinate)
    index = np.clip(index, 1, candidate.size - 1)
    left = candidate[index - 1]
    right = candidate[index]
    selected = np.where(
        np.abs(coordinate - left) <= np.abs(coordinate - right), left, right
    )
    delta = coordinate - selected
    return {
        "candidate_count": int(candidate.size),
        "candidate_bounds_m": [float(candidate[0]), float(candidate[-1])],
        "maximum_mismatch_m": float(np.max(np.abs(delta))),
        "nonmatching_count_2e-18m": int(np.count_nonzero(np.abs(delta) > 2e-18)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    template = args.template.expanduser().resolve()
    if sha256(template) != args.template_sha256:
        raise RuntimeError("template SHA mismatch")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing non-empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "production_adjoint_template_mesh_audit.json"
    result = {
        "status": "BLOCKED_PRODUCTION_ADJOINT_TEMPLATE_MESH_AUDIT",
        "passed": False,
        "Maxwell_timestepping_solves": 0,
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
        helper = audit.load_module(audit.API_HELPER, "run002_adjoint_template_mesh_api")
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
        fdtd.load(str(template))
        source_before = fdtd.getresult(FIELD_REGION, "source profile")
        before = {
            axis: np.asarray(source_before[axis], float).reshape(-1)
            for axis in "xyz"
        }
        source_shape = list(np.asarray(source_before["E"]).shape)
        bounds = {
            axis: [
                float(fdtd.getnamed(FIELD_REGION, f"{axis} min")),
                float(fdtd.getnamed(FIELD_REGION, f"{axis} max")),
            ]
            for axis in "xyz"
        }
        runtime.configure_session_resources(fdtd)
        fdtd.runsetup()
        mesh = {axis: mesh_coordinate(fdtd, axis) for axis in "xyz"}
        object_coordinates = {}
        object_coordinate_errors = {}
        for axis in "xyz":
            try:
                object_coordinates[axis] = np.asarray(
                    fdtd.getdata(FIELD_REGION, axis, 1), float
                ).reshape(-1)
            except Exception as exc:
                object_coordinate_errors[axis] = f"{type(exc).__name__}: {exc}"
        comparisons = {}
        mesh_inside_bounds = {}
        for axis in "xyz":
            midpoints = 0.5 * (mesh[axis][:-1] + mesh[axis][1:])
            low, high = bounds[axis]
            nodes_inside = mesh[axis][
                (mesh[axis] >= low - 2.0e-18)
                & (mesh[axis] <= high + 2.0e-18)
            ]
            centers_inside = midpoints[
                (midpoints >= low - 2.0e-18)
                & (midpoints <= high + 2.0e-18)
            ]
            mesh_inside_bounds[axis] = {
                "node_count": int(nodes_inside.size),
                "node_bounds_m": (
                    [float(nodes_inside[0]), float(nodes_inside[-1])]
                    if nodes_inside.size
                    else None
                ),
                "cell_center_count": int(centers_inside.size),
                "cell_center_bounds_m": (
                    [float(centers_inside[0]), float(centers_inside[-1])]
                    if centers_inside.size
                    else None
                ),
            }
            comparisons[axis] = {
                "source_count": int(before[axis].size),
                "source_bounds_m": [float(before[axis][0]), float(before[axis][-1])],
                "to_mesh_nodes": nearest(before[axis], mesh[axis]),
                "to_mesh_cell_centers": nearest(before[axis], midpoints),
            }
        npz = output / "production_adjoint_template_source_and_mesh.npz"
        np.savez_compressed(
            npz,
            **{f"source_{axis}_m": before[axis] for axis in "xyz"},
            **{f"mesh_{axis}_m": mesh[axis] for axis in "xyz"},
            **{
                f"object_{axis}_m": object_coordinates[axis]
                for axis in object_coordinates
            },
        )
        result = {
            "status": "COMPLETED_PRODUCTION_ADJOINT_TEMPLATE_MESH_AUDIT",
            "passed": True,
            "template": {
                "path": str(template),
                "size_bytes": template.stat().st_size,
                "sha256": sha256(template),
            },
            "fieldregion_bounds_m": bounds,
            "source_E_shape": source_shape,
            "coordinate_comparisons": comparisons,
            "source_mode_mesh_inside_fieldregion_bounds": mesh_inside_bounds,
            "fieldregion_getdata_coordinates": {
                axis: {
                    "count": int(value.size),
                    "bounds_m": [float(value[0]), float(value[-1])],
                }
                for axis, value in object_coordinates.items()
            },
            "fieldregion_getdata_errors": object_coordinate_errors,
            "artifact": {
                "path": str(npz),
                "size_bytes": npz.stat().st_size,
                "sha256": sha256(npz),
            },
            "Maxwell_timestepping_solves": 0,
            "thermal_solves": 0,
            "optimization_iterations": 0,
        }
    except Exception as exc:
        result.update(
            {
                "status": "FAILED_PRODUCTION_ADJOINT_TEMPLATE_MESH_AUDIT",
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
