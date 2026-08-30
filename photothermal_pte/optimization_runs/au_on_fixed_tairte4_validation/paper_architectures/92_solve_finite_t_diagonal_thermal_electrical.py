#!/usr/bin/env python3
"""Solve unchanged explicit 3-D thermal/electrical physics for T +/-45 Q."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "87_solve_finite_t_z_array_thermal_electrical.py"
MAPPING = HERE / "results_finite_T_diagonal_material_Q_mapping" / "FINITE_T_DIAGONAL_MATERIAL_Q_MAPPING_SUMMARY.json"
RAW_OUT = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_diagonal_thermal_electrical")
OUTPUT = HERE / "results_finite_T_diagonal_thermal_electrical"
CASES = ("T11x15_linear_plus_45_Au_on", "T11x15_linear_minus_45_Au_on")


def load_source():
    spec = importlib.util.spec_from_file_location("finite_t_diagonal_thermal_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    solver = load_source()
    solver.MAPPING = MAPPING
    solver.RAW_OUT = RAW_OUT
    solver.OUTPUT = OUTPUT
    for case in CASES:
        sys.argv = [str(Path(__file__).name), "--case", case, "--cuda-device", "0"]
        code = int(solver.main())
        if code:
            return code
        summary_path = OUTPUT / case / f"{case}_THERMAL_ELECTRICAL_SUMMARY.json"
        summary = json.loads(summary_path.read_text())
        summary["status"] = "VALIDATED_FINITE_T11X15_DIAGONAL_THERMAL_ELECTRICAL_FORWARD"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        manifest_path = OUTPUT / case / "RAW_ARTIFACT_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = summary["status"]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
