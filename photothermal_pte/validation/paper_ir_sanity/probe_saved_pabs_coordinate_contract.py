#!/usr/bin/env python3
"""Probe independent E/index/face coordinates from a saved Lumerical FSP.

The script is read-only: it does not call FDTD ``run`` or ``runanalysis``.
It is used to determine which component-detail coordinate arrays v261
actually exposes before defining a collocation or control-volume gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "run_lumerical_device_a_ir_q.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "paper_ir_coordinate_probe_runner",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def coordinate_summary(values: np.ndarray) -> dict[str, Any]:
    coordinate = np.asarray(values, float).reshape(-1)
    result: dict[str, Any] = {
        "count": int(coordinate.size),
        "bounds_m": [float(np.min(coordinate)), float(np.max(coordinate))],
        "strictly_increasing": bool(
            coordinate.size < 2 or np.all(np.diff(coordinate) > 0.0)
        ),
    }
    if coordinate.size > 1:
        steps = np.diff(coordinate)
        result["minimum_step_m"] = float(np.min(steps))
        result["median_step_m"] = float(np.median(steps))
        result["maximum_step_m"] = float(np.max(steps))
    return result


def try_getdata(
    fdtd: Any,
    monitor: str,
    quantity: str,
) -> dict[str, Any]:
    try:
        values = np.asarray(fdtd.getdata(monitor, quantity, 1)).squeeze()
    except Exception as exc:
        return {
            "available": False,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }
    result: dict[str, Any] = {
        "available": True,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
    }
    if quantity in ("x", "y", "z") or quantity.startswith("delta_"):
        result["coordinate"] = coordinate_summary(values)
        result["values_m"] = np.asarray(values, float).reshape(-1).tolist()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsp", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fsp = Path(args.fsp).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not fsp.is_file():
        raise FileNotFoundError(fsp)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite coordinate probe: {output}")

    runner = load_runner()
    base = runner.load_base()
    os.environ["VC_LUMERICAL_ROOT"] = str(runner.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(runner.APPROVED_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(runner.APPROVED_API)
    if str(runner.APPROVED_API) not in sys.path:
        sys.path.insert(0, str(runner.APPROVED_API))
    installation = SimpleNamespace(
        version_key="v261",
        root=runner.APPROVED_ROOT.resolve(),
        lumapi_path=(runner.APPROVED_API / "lumapi.py").resolve(),
        device_executable=(runner.APPROVED_ROOT / "bin" / "device").resolve(),
    )
    lumapi = base.load_lumapi(installation)

    fdtd = None
    try:
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        fdtd.load(str(fsp))
        monitors = {
            "field_detail": base.PABS_FIELD,
            "index_detail": base.PABS_INDEX,
            **{
                f"face_{axis}_{side}": f"paper_ir_abs_{axis}_{side}"
                for axis in "xyz"
                for side in ("min", "max")
            },
        }
        probe = {
            label: {
                "monitor": monitor,
                "quantities": {
                    quantity: try_getdata(fdtd, monitor, quantity)
                    for quantity in (
                        "x",
                        "y",
                        "z",
                        "delta_x",
                        "delta_y",
                        "delta_z",
                        "Ex",
                        "Ey",
                        "Ez",
                        "index_x",
                        "index_y",
                        "index_z",
                    )
                },
            }
            for label, monitor in monitors.items()
        }

        field = probe["field_detail"]["quantities"]
        index = probe["index_detail"]["quantities"]
        component_pairing: dict[str, Any] = {}
        for component in "xyz":
            field_common = {
                axis: np.asarray(field[axis].get("values_m", []), float)
                for axis in "xyz"
            }
            index_common = {
                axis: np.asarray(index[axis].get("values_m", []), float)
                for axis in "xyz"
            }
            field_delta = np.asarray(
                field[f"delta_{component}"].get("values_m", []),
                float,
            )
            index_delta = np.asarray(
                index[f"delta_{component}"].get("values_m", []),
                float,
            )
            comparisons: dict[str, Any] = {}
            for axis in "xyz":
                field_coordinate = np.array(field_common[axis], copy=True)
                index_coordinate = np.array(index_common[axis], copy=True)
                if axis == component:
                    if field_delta.size == field_coordinate.size:
                        field_coordinate += field_delta
                    if index_delta.size == index_coordinate.size:
                        index_coordinate += index_delta
                same_shape = field_coordinate.shape == index_coordinate.shape
                comparisons[axis] = {
                    "field_coordinate_count": int(field_coordinate.size),
                    "index_coordinate_count": int(index_coordinate.size),
                    "same_shape": same_shape,
                    "maximum_coordinate_mismatch_m": (
                        float(
                            np.max(
                                np.abs(
                                    field_coordinate - index_coordinate
                                )
                            )
                        )
                        if same_shape and field_coordinate.size
                        else None
                    ),
                    "independent_index_delta_available": bool(
                        index_delta.size
                    ),
                }
            component_pairing[component] = comparisons

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                jsonable(
                    {
                        "classification": (
                            "READ_ONLY_INDEPENDENT_SAVED_MONITOR_"
                            "COORDINATE_PROBE"
                        ),
                        "FDTD_solve_called": False,
                        "runanalysis_called": False,
                        "input_fsp": {
                            "path": str(fsp),
                            "size_bytes": fsp.stat().st_size,
                            "sha256": sha256(fsp),
                        },
                        "probe": probe,
                        "component_pairing": component_pairing,
                    }
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        if fdtd is not None:
            fdtd.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
