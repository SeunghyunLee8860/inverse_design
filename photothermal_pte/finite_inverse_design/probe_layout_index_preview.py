#!/usr/bin/env python3
"""Probe layout-mode index-preview availability without a Maxwell solve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .probe_v261_cpu_tfsf_device import PABS_INDEX
from .probe_v261_gpu_plane_wave_roi import load_lumapi
from .run_v261_large_background_tfsf_forward import sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    project = Path(args.project).expanduser().resolve()
    if sha256(project) != args.expected_sha256:
        raise RuntimeError("FSP SHA-256 mismatch")
    fdtd = None
    try:
        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        fdtd.load(str(project))
        completed_detail = fdtd.getresult(PABS_INDEX, "index_detail")
        fdtd.switchtolayout()
        available = fdtd.getresult(PABS_INDEX)
        datasets = {}
        for name in ("index preview", "index", "index_detail"):
            try:
                value = fdtd.getresult(PABS_INDEX, name)
                datasets[name] = {
                    "keys": list(value),
                    "shapes": {
                        key: list(np.asarray(item).shape)
                        for key, item in value.items()
                        if key != "Lumerical_dataset"
                    },
                    "offset_bounds_m": {
                        axis: [
                            float(np.min(value[f"{axis}_offset"])),
                            float(np.max(value[f"{axis}_offset"])),
                        ]
                        for axis in "xyz"
                        if f"{axis}_offset" in value
                    },
                }
            except Exception as error:
                datasets[name] = {"error": str(error)}
        result = {
            "status": "PROBED_LAYOUT_INDEX_PREVIEW",
            "electromagnetic_solve_run": False,
            "available_results": available,
            "datasets": datasets,
            "completed_vs_layout_index_detail": {},
        }
        layout_detail = fdtd.getresult(PABS_INDEX, "index_detail")
        for key in (
            "x",
            "x_offset",
            "y",
            "y_offset",
            "z",
            "z_offset",
            "f",
            "index_x",
            "index_y",
            "index_z",
        ):
            left = np.asarray(completed_detail[key])
            right = np.asarray(layout_detail[key])
            result["completed_vs_layout_index_detail"][key] = {
                "shape_equal": left.shape == right.shape,
                "maximum_absolute_difference": (
                    float(np.max(np.abs(left - right)))
                    if left.shape == right.shape
                    else None
                ),
            }
        print(json.dumps(result, indent=2))
        return 0
    finally:
        if fdtd is not None:
            fdtd.close()


if __name__ == "__main__":
    raise SystemExit(main())
