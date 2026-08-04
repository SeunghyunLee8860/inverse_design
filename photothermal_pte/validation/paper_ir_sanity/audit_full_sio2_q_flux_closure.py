#!/usr/bin/env python3
"""Offline Q-quadrature and six-face cancellation audit for a saved FSP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


EPS0 = 8.8541878128e-12
C0 = 299792458.0
WAVELENGTH_M = 11.0e-6
APPROVED_ROOT = Path("/home/seunghyun/lumerical_r12/opt/lumerical/v261")
APPROVED_LUMAPI = APPROVED_ROOT / "api/python/lumapi.py"
PABS_GROUP = "finite_pabs_adv"
PABS_FIELD = f"{PABS_GROUP}::field"
PABS_INDEX = f"{PABS_GROUP}::index"


def scalar(value: Any) -> float:
    array = np.asarray(value).squeeze()
    if array.size != 1:
        raise RuntimeError(f"expected scalar, got shape {array.shape}")
    return float(array)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trapezoid_weights(coordinate: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinate, float).reshape(-1)
    if values.size < 2 or np.any(np.diff(values) <= 0.0):
        raise RuntimeError("quadrature coordinate is not strictly increasing")
    weights = np.empty_like(values)
    weights[0] = 0.5 * (values[1] - values[0])
    weights[-1] = 0.5 * (values[-1] - values[-2])
    weights[1:-1] = 0.5 * (values[2:] - values[:-2])
    return weights


def bounded_dual_cell_weights(
    coordinate: np.ndarray, low: float, high: float
) -> np.ndarray:
    values = np.asarray(coordinate, float).reshape(-1)
    if values.size < 2 or np.any(np.diff(values) <= 0.0) or high <= low:
        raise RuntimeError("invalid bounded dual-cell coordinate")
    midpoints = 0.5 * (values[:-1] + values[1:])
    edges = np.concatenate(
        (
            [values[0] - 0.5 * (values[1] - values[0])],
            midpoints,
            [values[-1] + 0.5 * (values[-1] - values[-2])],
        )
    )
    lower_gap = max(0.0, float(edges[0] - low))
    upper_gap = max(0.0, float(high - edges[-1]))
    if lower_gap > values[1] - values[0] + 1e-18:
        raise RuntimeError("lower Q support misses more than one grid step")
    if upper_gap > values[-1] - values[-2] + 1e-18:
        raise RuntimeError("upper Q support misses more than one grid step")
    if lower_gap > 0.0:
        edges[0] = low
    if upper_gap > 0.0:
        edges[-1] = high
    weights = np.maximum(
        0.0,
        np.minimum(edges[1:], high) - np.maximum(edges[:-1], low),
    )
    if not np.isclose(weights.sum(), high - low, rtol=1e-13, atol=1e-18):
        raise RuntimeError("bounded weights do not close on requested bounds")
    return weights


def integrate(values: np.ndarray, weights: dict[str, np.ndarray]) -> float:
    return float(
        np.einsum(
            "i,j,k,ijk->",
            weights["x"],
            weights["y"],
            weights["z"],
            np.asarray(values, float),
            optimize=True,
        )
    )


def load_lumapi() -> Any:
    if not APPROVED_LUMAPI.is_file():
        raise FileNotFoundError(APPROVED_LUMAPI)
    os.environ["VC_LUMERICAL_ROOT"] = str(APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(APPROVED_ROOT)
    spec = importlib.util.spec_from_file_location("lumapi", APPROVED_LUMAPI)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load approved v261 lumapi")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def component_native_power(
    fdtd: Any,
    component: str,
    common_coordinates: dict[str, np.ndarray],
    bounds: dict[str, list[float]],
    intensity_scale: float,
) -> dict[str, Any]:
    field_delta = np.asarray(
        fdtd.getdata(PABS_FIELD, f"delta_{component}", 1), float
    ).reshape(-1)
    index_delta = np.asarray(
        fdtd.getdata(PABS_INDEX, f"delta_{component}", 1), float
    ).reshape(-1)
    field_coordinates = {
        axis: np.array(values, copy=True)
        for axis, values in common_coordinates.items()
    }
    index_coordinates = {
        axis: np.asarray(fdtd.getdata(PABS_INDEX, axis, 1), float).reshape(-1)
        for axis in "xyz"
    }
    field_coordinates[component] += field_delta
    index_coordinates[component] += index_delta
    coordinate_mismatch = {
        axis: float(
            np.max(np.abs(field_coordinates[axis] - index_coordinates[axis]))
        )
        for axis in "xyz"
    }
    electric = np.asarray(fdtd.getdata(PABS_FIELD, f"E{component}", 1)).squeeze()
    index = np.asarray(
        fdtd.getdata(PABS_INDEX, f"index_{component}", 1)
    ).squeeze()
    expected_shape = tuple(field_coordinates[axis].size for axis in "xyz")
    if electric.shape != expected_shape or index.shape != expected_shape:
        raise RuntimeError(
            f"{component} native shape mismatch: E={electric.shape}, "
            f"index={index.shape}, coordinates={expected_shape}"
        )
    omega = 2.0 * np.pi * C0 / WAVELENGTH_M
    q_native = (
        0.5
        * EPS0
        * omega
        * np.abs(electric) ** 2
        * np.imag(index**2)
        * intensity_scale
    )
    trapezoid = {
        axis: trapezoid_weights(values)
        for axis, values in field_coordinates.items()
    }
    bounded = {
        axis: bounded_dual_cell_weights(
            values, float(bounds[axis][0]), float(bounds[axis][1])
        )
        for axis, values in field_coordinates.items()
    }
    result = {
        "native_trapezoid_power_W": integrate(q_native, trapezoid),
        "native_bounded_power_W": integrate(q_native, bounded),
        "maximum_E_index_coordinate_mismatch_m": max(
            coordinate_mismatch.values()
        ),
        "E_index_coordinate_mismatch_m": coordinate_mismatch,
        "shape": list(expected_shape),
    }
    del electric, index, q_native
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    case_dir = args.case_dir.resolve()
    result_path = case_dir / "case_result.json"
    artifact_path = case_dir / "finite_q_on_artifact.npz"
    fsp_path = case_dir / "finite_2um_optical_q.fsp"
    result = json.loads(result_path.read_text())
    run = result["run_result"]
    native_audit = run["native_Yee_mesh_audit"]
    bounds = native_audit["realized_six_face_control_volume"]["bounds_m"]
    q_requested_bounds = native_audit["Q_quadrature_control_volume_bounds_m"]

    with np.load(artifact_path, allow_pickle=False) as stored:
        common_coordinates = {
            axis: np.asarray(stored[f"{axis}_m"], float) for axis in "xyz"
        }
        q_common = np.asarray(stored["Q_on_W_m3"], float)
    common_trapezoid_weights = {
        axis: trapezoid_weights(values)
        for axis, values in common_coordinates.items()
    }
    common_bounded_weights = {
        axis: bounded_dual_cell_weights(
            values, float(bounds[axis][0]), float(bounds[axis][1])
        )
        for axis, values in common_coordinates.items()
    }
    common_trapezoid = integrate(q_common, common_trapezoid_weights)
    common_bounded = integrate(q_common, common_bounded_weights)
    del q_common

    lumapi = load_lumapi()
    fdtd = lumapi.FDTD(filename=str(fsp_path), hide=True)
    try:
        pabs_total = fdtd.getresult(PABS_GROUP, "Pabs_total")
        pabs_fraction = scalar(pabs_total["Pabs_total"])
        normalization = run["normalization"]
        pabs_official = (
            pabs_fraction
            * float(normalization["measured_source_power_native_W"])
            * float(normalization["scale_to_1_W_m2"])
        )
        component_results = {
            component: component_native_power(
                fdtd,
                component,
                common_coordinates,
                bounds,
                float(normalization["scale_to_1_W_m2"]),
            )
            for component in "xyz"
        }
        pabs_object_bounds = {
            axis: [
                scalar(fdtd.getnamed(PABS_GROUP, axis))
                - 0.5 * scalar(fdtd.getnamed(PABS_GROUP, f"{axis} span")),
                scalar(fdtd.getnamed(PABS_GROUP, axis))
                + 0.5 * scalar(fdtd.getnamed(PABS_GROUP, f"{axis} span")),
            ]
            for axis in "xyz"
        }
    finally:
        fdtd.close()

    native_trapezoid = sum(
        item["native_trapezoid_power_W"] for item in component_results.values()
    )
    native_bounded = sum(
        item["native_bounded_power_W"] for item in component_results.values()
    )
    p_six = float(run["P_six_face_W"])
    faces = run["six_face"]["faces"]
    face_rows = []
    for face, payload in faces.items():
        outward = float(payload["outward_power_W_at_1_W_m2"])
        face_rows.append(
            {
                "face": face,
                "outward_power_W": outward,
                "absolute_power_W": abs(outward),
                "absolute_power_over_abs_P_six": abs(outward) / abs(p_six),
            }
        )
    cancellation = sum(row["absolute_power_W"] for row in face_rows) / abs(p_six)
    z_cancellation = sum(
        row["absolute_power_W"] for row in face_rows if row["face"].startswith("z")
    ) / abs(p_six)
    lateral_cancellation = sum(
        row["absolute_power_W"] for row in face_rows if not row["face"].startswith("z")
    ) / abs(p_six)
    quadratures = {
        "common_trapezoid_W": common_trapezoid,
        "common_bounded_W": common_bounded,
        "native_trapezoid_W": native_trapezoid,
        "native_bounded_W": native_bounded,
        "Lumerical_Pabs_total_W": pabs_official,
    }
    closures = {
        key: abs(value - p_six) / abs(p_six)
        for key, value in quadratures.items()
    }
    requested_vs_six = {
        axis: max(
            abs(float(q_requested_bounds[axis][0]) - float(bounds[axis][0])),
            abs(float(q_requested_bounds[axis][1]) - float(bounds[axis][1])),
        )
        for axis in "xyz"
    }
    object_vs_six = {
        axis: max(
            abs(float(pabs_object_bounds[axis][0]) - float(bounds[axis][0])),
            abs(float(pabs_object_bounds[axis][1]) - float(bounds[axis][1])),
        )
        for axis in "xyz"
    }
    summary = {
        "status": "DIAGNOSED_FULL_SIO2_Q_FLUX_CANCELLATION_UNRESOLVED",
        "new_FDTD_run": False,
        "case_status_preserved": result["status"],
        "P_six_W": p_six,
        "face_powers": face_rows,
        "cancellation_factor_sum_abs_faces_over_abs_net": cancellation,
        "z_face_cancellation_factor": z_cancellation,
        "lateral_face_cancellation_factor": lateral_cancellation,
        "Q_quadratures": quadratures,
        "closures_relative_to_P_six": closures,
        "component_native_quadrature": component_results,
        "bounds": {
            "realized_six_face_m": bounds,
            "Q_quadrature_requested_m": q_requested_bounds,
            "Pabs_object_readback_m": pabs_object_bounds,
            "requested_Q_vs_six_maximum_mismatch_m": requested_vs_six,
            "Pabs_object_vs_six_maximum_mismatch_m": object_vs_six,
            "all_bounds_match_within_1e_minus_15_m": (
                max((*requested_vs_six.values(), *object_vs_six.values()))
                < 1e-15
            ),
        },
        "interpretation": (
            "common/native and trapezoid/bounded volume-Q paths are compared "
            "without rerunning FDTD; a persistent closure in every path "
            "isolates the unresolved discrepancy to volume absorption versus "
            "the cancellation-sensitive face-flux balance"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "full_sio2_q_flux_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "full_sio2_q_flux_faces.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(face_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(face_rows)
    manifest = {
        "raw_artifacts_committed_to_git": False,
        "artifacts": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (result_path, artifact_path, fsp_path)
        ],
    }
    (args.output_dir / "FULL_SIO2_Q_FLUX_AUDIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    report = f"""# Full-SiO2 Q/flux offline audit

Status: `{summary['status']}`

No new FDTD, thermal, PTE, adjoint, or optimization run was made.  The saved
5-nm-oxide FSP was reopened read-only to compare four quadrature paths and
Lumerical's own `Pabs_total` against the same six-face balance.

## Face-flux cancellation

- `P_six = {p_six:.12e} W`.
- `sum(abs(P_face))/abs(P_six) = {cancellation:.9f}`.
- z-face contribution to that factor: `{z_cancellation:.9f}`.
- lateral-face contribution: `{lateral_cancellation:.9e}`.

| Q path | power (W) | closure versus P_six |
|---|---:|---:|
"""
    for key, value in quadratures.items():
        report += f"| {key} | {value:.12e} | {100*closures[key]:.9f}% |\n"
    report += f"""

The requested Q bounds, `finite_pabs_adv` object readback, and independently
read six-face bounds match within `1e-15 m`: `{summary['bounds']['all_bounds_match_within_1e_minus_15_m']}`.
The maximum independently read E/index component-coordinate mismatch is
`{max(item['maximum_E_index_coordinate_mismatch_m'] for item in component_results.values()):.3e} m`.

If every volume-Q path retains approximately the same 1.249% closure, changing
the Python quadrature or oxide z mesh is not a justified fix.  The next single
GPU test should keep the 5-nm geometry and use a stricter temporal/DFT stopping
condition while recording both Q and each face power; it must not rescale Q.
"""
    (args.output_dir / "FULL_SIO2_Q_FLUX_AUDIT_REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
