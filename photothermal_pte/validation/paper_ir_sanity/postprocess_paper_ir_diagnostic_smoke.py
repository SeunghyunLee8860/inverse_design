#!/usr/bin/env python3
"""Read-only post-processing of a completed paper-IR diagnostic FSP.

This script never calls ``run`` or ``runanalysis``.  It reopens already saved
monitor data to recover the six individual face fluxes and native/common-grid
component absorption powers after a diagnostic acceptance failure.
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
        "paper_ir_diagnostic_runner",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsp", required=True)
    parser.add_argument("--case-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fsp = Path(args.fsp).expanduser().resolve()
    case_result_path = Path(args.case_result).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not fsp.is_file() or not case_result_path.is_file():
        raise FileNotFoundError("completed FSP and case_result.json are required")

    case_result = json.loads(case_result_path.read_text(encoding="utf-8"))
    if case_result.get("status") != "FAILED_ACCEPTANCE":
        raise RuntimeError("postprocessor is restricted to the failed smoke")
    if (
        case_result.get("run_result", {}).get("classification")
        != "DIAGNOSTIC_ONE_POL_GPU_SMOKE_NOT_PRODUCTION_OR_PAPER_RESULT"
    ):
        raise RuntimeError("input is not the reduced diagnostic smoke")

    runner = load_runner()
    base = runner.load_base()
    base.TARGET_FREQUENCY_HZ = runner.C0 / runner.WAVELENGTH_M

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
        source_power = base.scalar(
            fdtd.sourcepower(
                base.TARGET_FREQUENCY_HZ,
                2,
                base.SOURCE_NAME,
            ),
            "saved native source power",
        )
        faces = {
            f"{axis}_{side}": {
                "name": f"paper_ir_abs_{axis}_{side}",
                "axis": axis,
                "side": side,
                "outward_sign": -1.0 if side == "min" else 1.0,
            }
            for axis in "xyz"
            for side in ("min", "max")
        }
        six_face = base.face_fluxes(
            fdtd,
            faces,
            source_power,
            1.0,
        )
        q_data = base.common_grid_component_q(
            fdtd,
            base.TARGET_FREQUENCY_HZ,
        )
        native_component_power = {
            axis: float(q_data["native_component_power_W"][axis])
            for axis in "xyz"
        }
        common_component_power = {
            axis: float(q_data["common_component_power_W"][axis])
            for axis in "xyz"
        }
        p_native = float(sum(native_component_power.values()))
        p_common = float(sum(common_component_power.values()))
        p_six = float(six_face["net_inward_power_W"])
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            output,
            {
                "classification": (
                    "READ_ONLY_POSTPROCESS_OF_FAILED_DIAGNOSTIC_SMOKE"
                ),
                "FDTD_solve_called": False,
                "runanalysis_called": False,
                "input": {
                    "fsp_path": str(fsp),
                    "fsp_size_bytes": fsp.stat().st_size,
                    "fsp_sha256": sha256(fsp),
                    "case_result_path": str(case_result_path),
                    "case_result_size_bytes": case_result_path.stat().st_size,
                    "case_result_sha256": sha256(case_result_path),
                    "generation_commit": case_result.get(
                        "generation_commit"
                    ),
                },
                "source_power_native_W": source_power,
                "six_face_native": six_face,
                "native_component_power_W": native_component_power,
                "common_component_power_W": common_component_power,
                "P_Q_native_component_grid_W": p_native,
                "P_Q_common_grid_W": p_common,
                "P_six_face_W": p_six,
                "native_component_vs_six_face_relative_closure": (
                    abs(p_native - p_six)
                    / max(abs(p_six), np.finfo(float).tiny)
                ),
                "common_grid_vs_six_face_relative_closure": (
                    abs(p_common - p_six)
                    / max(abs(p_six), np.finfo(float).tiny)
                ),
                "native_to_common_relative_difference": (
                    abs(p_native - p_common)
                    / max(abs(p_native), np.finfo(float).tiny)
                ),
                "component_interpolation_relative_error": q_data[
                    "component_interpolation_relative_error"
                ],
            },
        )
    finally:
        if fdtd is not None:
            fdtd.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
