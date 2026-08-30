#!/usr/bin/env python3
"""Run the validated finite 3-D thermal/electrical operator for array Q.

This is a narrow adapter around stage 81.  It changes only the mapped-Q input,
output roots, and the exact top-Au area fractions obtained from stage 86.  All
thermal materials, physical boundary conditions, TBCs, electrical equations,
terminal definitions, GPU residual gates, and plotting remain identical to the
single-object forward certificate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
MAPPING = HERE / "results_finite_T_Z_array_material_Q_mapping" / "FINITE_T_Z_ARRAY_MATERIAL_Q_MAPPING_SUMMARY.json"
RAW_OUT = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_array_thermal_electrical")
OUTPUT = HERE / "results_finite_T_Z_array_thermal_electrical"


def load_stage81():
    path = HERE / "81_solve_finite_t_z_thermal_electrical.py"
    spec = importlib.util.spec_from_file_location("finite_t_z_stage81", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_stage81()


def overlap_fraction(x_edges: np.ndarray, y_edges: np.ndarray, rectangles: list[list[float]]) -> np.ndarray:
    result = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=np.float64)
    area = np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :]
    for xmin, xmax, ymin, ymax, _zmin, _zmax in rectangles:
        ix = np.flatnonzero((x_edges[:-1] < xmax) & (x_edges[1:] > xmin))
        iy = np.flatnonzero((y_edges[:-1] < ymax) & (y_edges[1:] > ymin))
        if not (ix.size and iy.size):
            continue
        ox = np.maximum(0.0, np.minimum(x_edges[ix + 1], xmax) - np.maximum(x_edges[ix], xmin))
        oy = np.maximum(0.0, np.minimum(y_edges[iy + 1], ymax) - np.maximum(y_edges[iy], ymin))
        result[np.ix_(ix, iy)] += ox[:, None] * oy[None, :] / area[np.ix_(ix, iy)]
    if np.max(result, initial=0.0) > 1.0 + 1e-10:
        raise RuntimeError("array top-Au area rectangles overlap")
    return np.minimum(result, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    mapping = json.loads(MAPPING.read_text())
    if args.case not in mapping["cases"]:
        raise KeyError(args.case)
    rectangles = mapping["cases"][args.case]["top_Au_rectangles_m"]

    def array_top_au_fraction(x_edges, y_edges, architecture, enabled):
        del architecture
        if not enabled:
            return np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=np.float64)
        return overlap_fraction(x_edges, y_edges, rectangles)

    BASE.MAPPING = MAPPING
    BASE.RAW_OUT = RAW_OUT
    BASE.OUTPUT = OUTPUT
    BASE.top_au_fraction = array_top_au_fraction
    sys.argv = [str(Path(__file__).name), "--case", args.case, "--cuda-device", str(args.cuda_device)]
    code = int(BASE.main())
    if code == 0:
        summary_path = OUTPUT / args.case / f"{args.case}_THERMAL_ELECTRICAL_SUMMARY.json"
        summary = json.loads(summary_path.read_text())
        summary["status"] = "VALIDATED_FINITE_T_Z_ARRAY_THERMAL_ELECTRICAL_FORWARD"
        summary["array_contract"] = mapping["cases"][args.case]["array_contract"]
        summary["top_Au_geometry_source"] = "exact cut-cell rectangles from stage 86; no whole-cell assignment"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        manifest_path = OUTPUT / args.case / "RAW_ARTIFACT_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = summary["status"]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
