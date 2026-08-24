#!/usr/bin/env python3
"""Extract official pabs_adv Pabs/index_x arrays from a completed FSP.

This runs only the saved analysis-group post-processing.  It never launches
the Maxwell engine and never overwrites the original FSP, raw NPZ, or JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.native_yee_q import integrate_xyz  # noqa: E402
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_GROUP,
    PABS_INDEX,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_maxwell_contract import (  # noqa: E402
    LUMAPI_PATH,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not str(payload.get("status", "")).startswith(
        "PASSED"
    ):
        raise RuntimeError("input must be one passed Lumerical material result")
    if payload.get("case") == "source_only":
        raise RuntimeError("source-only results do not contain pabs_adv material data")
    return payload


def _fsp_artifact(payload: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    artifacts = payload.get("raw_artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("result JSON has no raw_artifacts list")
    matches = [item for item in artifacts if str(item.get("path", "")).endswith(".fsp")]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one FSP artifact, found {len(matches)}")
    record = matches[0]
    path = Path(record["path"]).resolve()
    if not path.is_file() or sha256(path) != record.get("sha256"):
        raise RuntimeError("FSP artifact is missing or its SHA256 does not match")
    return path, record


def _cube(value: Any, shape: tuple[int, int, int], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (*shape, 1):
        array = array[..., 0]
    if array.shape != shape:
        raise RuntimeError(f"{label} shape {array.shape} != {shape}")
    return array


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    args = parser.parse_args()

    result_json = args.result_json.resolve()
    output_npz = args.output_npz.resolve()
    output_json = output_npz.with_suffix(".json")
    if output_npz.exists() or output_json.exists():
        raise FileExistsError("official Pabs companion output already exists")
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_result(result_json)
    fsp_path, fsp_record = _fsp_artifact(payload)

    os.environ["LUMERICAL_PYTHONPATH"] = str(LUMAPI_PATH.parent)
    sys.path.insert(0, str(LUMAPI_PATH.parent))
    import lumapi

    fdtd = None
    try:
        fdtd = lumapi.FDTD(
            filename=str(fsp_path),
            hide=True,
            serverArgs={"platform": "offscreen"},
        )
        fdtd.runanalysis(PABS_GROUP)
        pabs = fdtd.getresult(PABS_GROUP, "Pabs")
        index = fdtd.getresult(PABS_INDEX, "index")
        coordinates = {
            axis: np.asarray(pabs[axis], dtype=np.float64).reshape(-1)
            for axis in "xyz"
        }
        shape = tuple(coordinates[axis].size for axis in "xyz")
        coordinate_difference_m: dict[str, float] = {}
        for axis in "xyz":
            index_axis = np.asarray(index[axis], dtype=np.float64).reshape(-1)
            difference = float(np.max(np.abs(index_axis - coordinates[axis])))
            coordinate_difference_m[axis] = difference
            if difference > 1.0e-18:
                raise RuntimeError(
                    f"Pabs and index {axis} coordinates differ by {difference} m"
                )
        pabs_over_sourcepower = np.asarray(
            _cube(pabs["Pabs"], shape, "Pabs"), dtype=np.float64
        )
        index_x = np.asarray(
            _cube(index["index_x"], shape, "index_x"), dtype=np.complex128
        )
    finally:
        if fdtd is not None:
            fdtd.close()

    if not np.all(np.isfinite(pabs_over_sourcepower)):
        raise RuntimeError("official Pabs contains NaN or Inf values")
    if not np.all(np.isfinite(index_x)):
        raise RuntimeError("official index_x contains NaN or Inf")
    source_power_W = float(payload["source_power_W_raw"])
    pabs_W_m3 = pabs_over_sourcepower * source_power_W
    negative_power_W = integrate_xyz(
        np.where(pabs_W_m3 < 0.0, -pabs_W_m3, 0.0),
        coordinates["x"],
        coordinates["y"],
        coordinates["z"],
    )
    positive_power_W = integrate_xyz(
        np.where(pabs_W_m3 > 0.0, pabs_W_m3, 0.0),
        coordinates["x"],
        coordinates["y"],
        coordinates["z"],
    )
    integrated_W = integrate_xyz(
        pabs_W_m3,
        coordinates["x"],
        coordinates["y"],
        coordinates["z"],
    )
    expected_W = float(payload["P_Q_pabs_W_raw"])
    closure = abs(integrated_W - expected_W) / max(
        abs(expected_W), np.finfo(float).tiny
    )
    if closure >= 1.0e-12:
        raise RuntimeError(f"official spatial Pabs does not close Pabs_total: {closure}")
    negative_relative = negative_power_W / max(
        abs(integrated_W), np.finfo(float).tiny
    )
    if negative_relative >= 1.0e-12:
        raise RuntimeError(
            "official spatial Pabs negative interpolation artifact is too large: "
            f"{negative_relative}"
        )

    np.savez_compressed(
        output_npz,
        Pabs_W_m3=pabs_W_m3,
        Pabs_index_x=index_x,
        **{f"Pabs_{axis}_m": coordinates[axis] for axis in "xyz"},
    )
    audit = {
        "schema": "lumerical-4um-official-pabs-material-filter-input-v1",
        "status": "EXTRACTED_LUMERICAL_OFFICIAL_PABS_INDEX_X",
        "source_result_json": {
            "path": str(result_json),
            "sha256": sha256(result_json),
        },
        "source_fsp": fsp_record,
        "case": payload["case"],
        "polarization": payload["polarization"],
        "mesh_spec": payload["mesh_spec"],
        "source_power_W_raw": source_power_W,
        "integrated_Pabs_W_raw": integrated_W,
        "expected_Pabs_total_W_raw": expected_W,
        "spatial_vs_total_relative": closure,
        "negative_sample_count": int(np.count_nonzero(pabs_W_m3 < 0.0)),
        "minimum_Pabs_W_m3": float(np.min(pabs_W_m3)),
        "negative_absorption_magnitude_W": negative_power_W,
        "positive_absorption_W": positive_power_W,
        "negative_absorption_relative": negative_relative,
        "Pabs_vs_index_coordinate_max_abs_difference_m": coordinate_difference_m,
        "method": (
            "Lumerical pabs_adv common-grid Pabs plus index.index_x; "
            "Maxwell engine was not rerun"
        ),
        "output_npz": {
            "path": str(output_npz),
            "size_bytes": output_npz.stat().st_size,
            "sha256": sha256(output_npz),
        },
    }
    output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
