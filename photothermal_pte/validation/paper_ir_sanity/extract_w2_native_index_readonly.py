#!/usr/bin/env python3
"""Read saved w0=2 um planar/edge Yee field and index data without solving.

The script opens completed FSP files and calls only ``load`` and ``getdata``.
It deliberately contains no call to FDTD ``run`` or ``runanalysis``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (
    run_lumerical_device_a_ir_q as runner,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def coordinate(fdtd: Any, monitor: str, axis: str) -> np.ndarray:
    return np.asarray(fdtd.getdata(monitor, axis, 1), float).reshape(-1)


def extract_case(
    lumapi: Any,
    base: Any,
    fsp: Path,
    label: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fdtd = None
    arrays: dict[str, np.ndarray] = {}
    try:
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        fdtd.load(str(fsp))
        field_common = {
            axis: coordinate(fdtd, base.PABS_FIELD, axis) for axis in "xyz"
        }
        index_common = {
            axis: coordinate(fdtd, base.PABS_INDEX, axis) for axis in "xyz"
        }
        field_delta = {
            axis: np.asarray(
                fdtd.getdata(base.PABS_FIELD, f"delta_{axis}", 1), float
            ).reshape(-1)
            for axis in "xyz"
        }
        index_delta = {
            axis: np.asarray(
                fdtd.getdata(base.PABS_INDEX, f"delta_{axis}", 1), float
            ).reshape(-1)
            for axis in "xyz"
        }
        pairing: dict[str, Any] = {}
        for component in "xyz":
            electric = np.asarray(
                fdtd.getdata(base.PABS_FIELD, f"E{component}", 1)
            ).squeeze()
            refractive_index = np.asarray(
                fdtd.getdata(base.PABS_INDEX, f"index_{component}", 1)
            ).squeeze()
            epsilon = refractive_index**2
            field_coordinates = {
                axis: np.array(field_common[axis], copy=True)
                for axis in "xyz"
            }
            index_coordinates = {
                axis: np.array(index_common[axis], copy=True)
                for axis in "xyz"
            }
            field_coordinates[component] += field_delta[component]
            index_coordinates[component] += index_delta[component]
            if electric.shape != epsilon.shape:
                raise RuntimeError(
                    f"{label}:{component} E/index shape mismatch: "
                    f"{electric.shape} != {epsilon.shape}"
                )
            mismatch: dict[str, float] = {}
            for axis in "xyz":
                if (
                    field_coordinates[axis].shape
                    != index_coordinates[axis].shape
                ):
                    raise RuntimeError(
                        f"{label}:{component}:{axis} coordinate shape mismatch"
                    )
                mismatch[axis] = float(
                    np.max(
                        np.abs(
                            field_coordinates[axis]
                            - index_coordinates[axis]
                        )
                    )
                )
                arrays[
                    f"{label}_E{component}_{axis}_m"
                ] = field_coordinates[axis]
                arrays[
                    f"{label}_index_{component}_{axis}_m"
                ] = index_coordinates[axis]
            arrays[f"{label}_E{component}"] = electric
            arrays[f"{label}_epsilon_{component}"] = epsilon
            pairing[component] = {
                "shape": list(electric.shape),
                "coordinate_mismatch_m": mismatch,
                "maximum_coordinate_mismatch_m": max(mismatch.values()),
            }
        return arrays, {
            "fsp": {
                "path": str(fsp),
                "size_bytes": fsp.stat().st_size,
                "sha256": sha256(fsp),
            },
            "pairing": pairing,
            "maximum_field_index_coordinate_mismatch_m": max(
                value["maximum_coordinate_mismatch_m"]
                for value in pairing.values()
            ),
        }
    finally:
        if fdtd is not None:
            fdtd.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planar-fsp", type=Path, required=True)
    parser.add_argument("--edge-fsp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    planar_fsp = args.planar_fsp.expanduser().resolve()
    edge_fsp = args.edge_fsp.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not planar_fsp.is_file() or not edge_fsp.is_file():
        raise FileNotFoundError("both completed FSP files are required")
    output_dir.mkdir(parents=True, exist_ok=True)

    base = runner.load_base()
    base.TARGET_WAVELENGTH_M = runner.WAVELENGTH_M
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

    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for label, fsp in (("planar", planar_fsp), ("edge", edge_fsp)):
        extracted, audit = extract_case(lumapi, base, fsp, label)
        arrays.update(extracted)
        metadata[label] = audit

    for component in "xyz":
        for axis in "xyz":
            planar_coordinate = arrays[f"planar_E{component}_{axis}_m"]
            edge_coordinate = arrays[f"edge_E{component}_{axis}_m"]
            if planar_coordinate.shape != edge_coordinate.shape:
                raise RuntimeError(
                    f"planar/edge {component}:{axis} coordinate shape mismatch"
                )
            mismatch = float(
                np.max(np.abs(planar_coordinate - edge_coordinate))
            )
            metadata.setdefault("planar_edge_coordinate_pairing", {}).setdefault(
                component, {}
            )[axis] = mismatch

    artifact = output_dir / "w2_planar_edge_native_fields_index.npz"
    np.savez_compressed(
        artifact,
        **arrays,
        metadata_json=np.asarray([json.dumps(metadata, sort_keys=True)]),
    )
    payload = {
        "status": "EXTRACTED_READ_ONLY_NATIVE_FIELD_INDEX_DATA",
        "FDTD_run": False,
        "runanalysis_called": False,
        "thermal_run": False,
        "PTE_run": False,
        "adjoint_run": False,
        "generation_command": " ".join(sys.argv),
        "artifact": {
            "path": str(artifact),
            "size_bytes": artifact.stat().st_size,
            "sha256": sha256(artifact),
        },
        "audit": metadata,
    }
    (output_dir / "read_only_extraction_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
