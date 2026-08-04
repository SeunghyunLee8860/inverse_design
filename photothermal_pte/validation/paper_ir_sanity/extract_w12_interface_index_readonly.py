#!/usr/bin/env python3
"""Read component-specific index slices at the TaIrTe4/air interface.

This command opens already completed FSP files and calls only ``load`` and
``getdata``.  It never calls FDTD ``run`` or ``runanalysis`` and stores only
small numerical summaries, not the full monitor arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    run_lumerical_device_a_ir_q as runner,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate(fdtd: Any, monitor: str, axis: str) -> np.ndarray:
    return np.asarray(fdtd.getdata(monitor, axis, 1), float).reshape(-1)


def complex_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, complex).reshape(-1)
    return {
        "count": int(array.size),
        "real_min": float(np.min(array.real)),
        "real_median": float(np.median(array.real)),
        "real_max": float(np.max(array.real)),
        "imag_min": float(np.min(array.imag)),
        "imag_median": float(np.median(array.imag)),
        "imag_max": float(np.max(array.imag)),
    }


def local_dual_widths(coordinate: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinate, float).reshape(-1)
    edges = np.concatenate(
        (
            [values[0] - 0.5 * (values[1] - values[0])],
            0.5 * (values[:-1] + values[1:]),
            [values[-1] + 0.5 * (values[-1] - values[-2])],
        )
    )
    return np.diff(edges)


def extract_case(
    lumapi: Any,
    base: Any,
    *,
    fsp: Path,
    result_path: Path,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETED":
        raise RuntimeError(f"case is not completed: {result_path}")
    requested = result["run_result"]["material_epsilon_readback"]["axes"]
    q_bounds = result["run_result"]["native_Yee_mesh_audit"][
        "Q_quadrature_control_volume_bounds_m"
    ]
    fdtd = None
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
        components: dict[str, Any] = {}
        for component in "xyz":
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
            mismatch = {
                axis: float(
                    np.max(
                        np.abs(
                            field_coordinates[axis]
                            - index_coordinates[axis]
                        )
                    )
                )
                for axis in "xyz"
            }
            refractive_index = np.asarray(
                fdtd.getdata(
                    base.PABS_INDEX,
                    f"index_{component}",
                    1,
                )
            ).squeeze()
            epsilon = refractive_index**2
            expected_shape = tuple(
                index_coordinates[axis].size for axis in "xyz"
            )
            if epsilon.shape != expected_shape:
                raise RuntimeError(
                    f"{component} epsilon shape {epsilon.shape} "
                    f"!= coordinates {expected_shape}"
                )
            x, y, z = (
                index_coordinates[axis] for axis in "xyz"
            )
            lateral = (
                (y[None, :] <= x[:, None] + 1.0e-15)
                & (np.abs(x[:, None]) <= 10.0e-6)
                & (np.abs(y[None, :]) <= 10.0e-6)
            )
            near = np.flatnonzero(np.abs(z) <= 10.0e-9 + 1.0e-18)
            if near.size == 0:
                raise RuntimeError(f"{component} has no interface-near z")
            material_epsilon = complex(
                requested[component]["finite_dt_numerical_permittivity"][
                    "real"
                ],
                requested[component]["finite_dt_numerical_permittivity"][
                    "imag"
                ],
            )
            layers = []
            for index in near:
                selected = epsilon[:, :, index][lateral]
                loss_fraction = selected.imag / material_epsilon.imag
                layers.append(
                    {
                        "index": int(index),
                        "z_m": float(z[index]),
                        "is_exact_z0_sample": bool(
                            abs(float(z[index])) <= 1.0e-18
                        ),
                        "epsilon": complex_summary(selected),
                        "loss_fraction_vs_bulk_material": {
                            "minimum": float(np.min(loss_fraction)),
                            "median": float(np.median(loss_fraction)),
                            "maximum": float(np.max(loss_fraction)),
                            "fraction_above_0p5": float(
                                np.mean(loss_fraction > 0.5)
                            ),
                        },
                    }
                )
            weights = {
                axis: local_dual_widths(index_coordinates[axis])
                for axis in "xyz"
            }
            nearest_z = int(np.argmin(np.abs(z)))
            x_core = np.abs(x) <= 10.0e-6
            y_core = np.abs(y) <= 10.0e-6
            wx = weights["x"][x_core]
            wy = weights["y"][y_core]
            positive_volumes = (
                wx[:, None]
                * wy[None, :]
                * weights["z"][nearest_z]
            )
            if component in "xy":
                exact = min(layers, key=lambda item: abs(item["z_m"]))
                below = min(
                    (item for item in layers if item["z_m"] < -1.0e-12),
                    key=lambda item: abs(item["z_m"]),
                )
                above = min(
                    (item for item in layers if item["z_m"] > 1.0e-12),
                    key=lambda item: abs(item["z_m"]),
                )
                assignment = {
                    "kind": (
                        "exact-boundary tangential component sample with "
                        "approximately half-loss conformal epsilon"
                    ),
                    "below_loss_fraction_median": below[
                        "loss_fraction_vs_bulk_material"
                    ]["median"],
                    "boundary_loss_fraction_median": exact[
                        "loss_fraction_vs_bulk_material"
                    ]["median"],
                    "above_loss_fraction_median": above[
                        "loss_fraction_vs_bulk_material"
                    ]["median"],
                }
            else:
                below = min(
                    (item for item in layers if item["z_m"] < 0.0),
                    key=lambda item: abs(item["z_m"]),
                )
                above = min(
                    (item for item in layers if item["z_m"] > 0.0),
                    key=lambda item: abs(item["z_m"]),
                )
                assignment = {
                    "kind": (
                        "z-staggered one-sided samples: negative-z material "
                        "and positive-z air; no exact z=0 sample"
                    ),
                    "below_loss_fraction_median": below[
                        "loss_fraction_vs_bulk_material"
                    ]["median"],
                    "above_loss_fraction_median": above[
                        "loss_fraction_vs_bulk_material"
                    ]["median"],
                }
            components[component] = {
                "staggering_axis": component,
                "shape": list(epsilon.shape),
                "field_index_coordinate_mismatch_m": mismatch,
                "maximum_coordinate_mismatch_m": max(mismatch.values()),
                "z_sample_bounds_m": [float(z[0]), float(z[-1])],
                "nearest_z_to_interface_m": float(
                    z[int(np.argmin(np.abs(z)))]
                ),
                "has_exact_z0_sample": bool(np.any(np.abs(z) <= 1.0e-18)),
                "interface_near_layers": layers,
                "interface_assignment": assignment,
                "interface_sample_quadrature": {
                    "contract": (
                        "component-local dual widths on the component's own "
                        "Yee/index coordinates; not common-Q bounds"
                    ),
                    "nearest_z_index": nearest_z,
                    "nearest_z_m": float(z[nearest_z]),
                    "z_dual_weight_m": float(weights["z"][nearest_z]),
                    "positive_cell_volume_m3": {
                        "minimum": float(np.min(positive_volumes)),
                        "median": float(np.median(positive_volumes)),
                        "maximum": float(np.max(positive_volumes)),
                    },
                },
                "bulk_material_finite_dt_epsilon": {
                    "real": material_epsilon.real,
                    "imag": material_epsilon.imag,
                },
            }
            del epsilon, refractive_index
        return {
            "fsp": {
                "path": str(fsp.resolve()),
                "size_bytes": fsp.stat().st_size,
                "sha256": sha256(fsp),
            },
            "case_result": {
                "path": str(result_path.resolve()),
                "size_bytes": result_path.stat().st_size,
                "sha256": sha256(result_path),
            },
            "Q_quadrature_control_volume_bounds_m": q_bounds,
            "components": components,
            "maximum_field_index_coordinate_mismatch_m": max(
                value["maximum_coordinate_mismatch_m"]
                for value in components.values()
            ),
        }
    finally:
        if fdtd is not None:
            fdtd.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-case", type=Path, required=True)
    parser.add_argument("--fine-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {
        "50nm": args.coarse_case.expanduser().resolve(),
        "25nm": args.fine_case.expanduser().resolve(),
    }
    for directory in cases.values():
        for name in ("finite_2um_optical_q.fsp", "case_result.json"):
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)

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
    audit = {
        label: extract_case(
            lumapi,
            base,
            fsp=directory / "finite_2um_optical_q.fsp",
            result_path=directory / "case_result.json",
        )
        for label, directory in cases.items()
    }
    payload = {
        "status": "EXTRACTED_W12_INTERFACE_INDEX_READ_ONLY",
        "FDTD_run": False,
        "runanalysis_called": False,
        "thermal_run": False,
        "PTE_run": False,
        "scope": (
            "component-specific E/index coordinate pairing and saved-index "
            "material assignment near the z=0 TaIrTe4/air interface"
        ),
        "cases": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
