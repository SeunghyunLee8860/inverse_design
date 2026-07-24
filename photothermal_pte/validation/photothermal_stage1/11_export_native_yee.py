#!/usr/bin/env python3
"""Export stored, non-interpolated Yee E fields and sampled index tensors."""

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


def squeezed(value: Any) -> np.ndarray:
    return np.asarray(value).squeeze()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lumerical-version", default="v261")
    args = parser.parse_args()
    project = Path(args.project).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    installation = select_installation(args.lumerical_version)
    os.environ["VC_LUMERICAL_ROOT"] = str(installation.root)
    os.environ["LUMERICAL_ROOT"] = str(installation.root)
    for path in (config.REPOSITORY_ROOT, config.REPOSITORY_ROOT / "bundle"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import eqc_lib as runtime

    fdtd = runtime.open_control(project)
    try:
        group = "stage1_pabs_adv"
        field = f"{group}::field"
        index = f"{group}::index"
        arrays = {
            "x_m": squeezed(fdtd.getdata(field, "x", 1)).real,
            "y_m": squeezed(fdtd.getdata(field, "y", 1)).real,
            "z_m": squeezed(fdtd.getdata(field, "z", 1)).real,
            "delta_x_m": squeezed(fdtd.getdata(field, "delta_x", 1)).real,
            "delta_y_m": squeezed(fdtd.getdata(field, "delta_y", 1)).real,
            "delta_z_m": squeezed(fdtd.getdata(field, "delta_z", 1)).real,
            "frequency_Hz": squeezed(fdtd.getdata(field, "f")).real,
            "Ex": squeezed(fdtd.getdata(field, "Ex", 1)),
            "Ey": squeezed(fdtd.getdata(field, "Ey", 1)),
            "Ez": squeezed(fdtd.getdata(field, "Ez", 1)),
            "index_x": squeezed(fdtd.getdata(index, "index_x", 1)),
            "index_y": squeezed(fdtd.getdata(index, "index_y", 1)),
            "index_z": squeezed(fdtd.getdata(index, "index_z", 1)),
        }
        frequency = float(np.asarray(arrays["frequency_Hz"]).reshape(-1)[0])
        arrays["source_power_W"] = np.array(
            float(np.asarray(fdtd.sourcepower(frequency)).reshape(-1)[0])
        )
        arrays["source_intensity_W_m2"] = np.array(
            float(np.asarray(fdtd.sourceintensity(frequency)).reshape(-1)[0])
        )
        path = output / "native_yee_fields_and_index.npz"
        np.savez_compressed(path, **arrays)
        shape_summary = {
            name: list(np.asarray(value).shape)
            for name, value in arrays.items()
        }
        result = {
            "source_project": str(project),
            "output_npz": str(path),
            "raw_getdata_option": 1,
            "spatial_interpolation": "none",
            "shapes": shape_summary,
            "native_coordinate_contract": {
                "Ex_and_index_x": "(x + delta_x, y, z)",
                "Ey_and_index_y": "(x, y + delta_y, z)",
                "Ez_and_index_z": "(x, y, z + delta_z)",
            },
            "maxwell_rerun": False,
            "heat_run": False,
        }
        write_json(output / "native_yee_export_summary.json", result)
        print(json.dumps(result, indent=2))
    finally:
        fdtd.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
