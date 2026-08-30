#!/usr/bin/env python3
"""Locate the source of Au vertical-edge quadrature non-convergence offline.

The Lumerical engine writes native monitor fields to companion HDF5 files.
This script reads those completed forward/adjoint artifacts without opening a
Lumerical session and examines the tangential-E part of the sharp-interface
kernel along the two moving x-normal Au faces.  Absolute adjoint normalization
is deliberately irrelevant here: the diagnostic asks where the quadrature is
concentrated and how that spatial concentration changes with sample spacing.

It does not replace the complete tangential-E/normal-D boundary derivative and
cannot promote an optical gradient by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = Path("/home/seunghyun/tairte4/raw_artifacts/au_topology_validation")
FORWARD_CASE = "sharp_width_8p0_edge25_forward"
ADJOINT_CASE = "sharp_width_8p0_edge25_external_field_adjoint_gpu0"
MONITOR_GROUP = "Monitor3"
HALF_WIDTH_M = 8.0e-6
HALF_Y_M = 10.0e-6
Z_CENTER_M = 75.0e-9
INTERIOR_HALF_Y_M = 9.5e-6
SAMPLE_COUNTS = (201, 401, 801, 1601, 3201, 6401)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_complex_field(dataset: h5py.Dataset) -> np.ndarray:
    raw = np.asarray(dataset)
    if raw.ndim != 4 or raw.shape[-1] != 2:
        raise ValueError(f"unexpected engine field shape {raw.shape}")
    # Engine HDF5 order is z,y,x,(real,imag); repository native order is x,y,z.
    return (raw[..., 0] + 1j * raw[..., 1]).transpose(2, 1, 0)


def load_monitor(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with h5py.File(path, "r") as handle:
        group = handle[MONITOR_GROUP]
        base = {
            axis: np.asarray(group[axis], float).reshape(-1) * 1.0e-6
            for axis in "xyz"
        }
        electric = {
            component: decode_complex_field(group[f"E{component}"])
            for component in "xyz"
        }
    expected = tuple(base[axis].size for axis in "xyz")
    for component, value in electric.items():
        if value.shape != expected:
            raise ValueError(f"E{component} {value.shape} != {expected}")
    return base, electric


def component_coordinates(
    q_npz: Path, base: dict[str, np.ndarray]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    coordinates: dict[str, dict[str, np.ndarray]] = {}
    audit: dict[str, object] = {}
    with np.load(q_npz, allow_pickle=False) as data:
        for component in "xyz":
            coordinates[component] = {
                axis: np.asarray(data[f"Q{component}_{axis}_m"], float)
                for axis in "xyz"
            }
            offsets = {
                axis: coordinates[component][axis] - base[axis]
                for axis in "xyz"
            }
            audit[component] = {
                "shape": [coordinates[component][axis].size for axis in "xyz"],
                "offset_range_m": {
                    axis: [float(np.min(value)), float(np.max(value))]
                    for axis, value in offsets.items()
                },
                "non_component_axis_max_abs_offset_m": float(
                    max(
                        np.max(np.abs(offsets[axis]))
                        for axis in "xyz"
                        if axis != component
                    )
                ),
            }
    return coordinates, audit


def make_interpolators(
    electric: dict[str, np.ndarray],
    coordinates: dict[str, dict[str, np.ndarray]],
) -> dict[str, RegularGridInterpolator]:
    return {
        component: RegularGridInterpolator(
            tuple(coordinates[component][axis] for axis in "xyz"),
            electric[component],
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        for component in "xyz"
    }


def face_profile(
    forward: dict[str, RegularGridInterpolator],
    adjoint: dict[str, RegularGridInterpolator],
    *,
    x_m: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.linspace(-HALF_Y_M, HALF_Y_M, n_points)
    points = np.column_stack(
        (np.full_like(y, x_m), y, np.full_like(y, Z_CENTER_M))
    )
    # For an x-normal face, Ey and Ez are tangential. The omitted global
    # material/source constants do not affect localization or convergence.
    kernel = np.real(
        forward["y"](points) * adjoint["y"](points)
        + forward["z"](points) * adjoint["z"](points)
    )
    if not np.all(np.isfinite(kernel)):
        raise RuntimeError("non-finite tangential boundary proxy")
    return y, kernel


def profile_metrics(y: np.ndarray, kernel: np.ndarray) -> dict[str, object]:
    dy = float(y[1] - y[0])
    full = float(np.trapezoid(kernel, x=y))
    endpoint = float(0.5 * dy * (kernel[0] + kernel[-1]))
    interior = np.abs(y) <= INTERIOR_HALF_Y_M + 1.0e-18
    interior_value = float(np.trapezoid(kernel[interior], x=y[interior]))
    maximum = int(np.argmax(np.abs(kernel)))
    return {
        "n_points": int(y.size),
        "dy_m": dy,
        "full_integral_raw": full,
        "endpoint_trapezoid_raw": endpoint,
        "endpoint_fraction_of_full": endpoint / full if full != 0.0 else None,
        "interior_abs_y_le_9p5um_integral_raw": interior_value,
        "maximum_abs_kernel_y_m": float(y[maximum]),
        "maximum_abs_kernel_raw": float(np.abs(kernel[maximum])),
        "kernel_y_min_raw": float(kernel[0]),
        "kernel_y_max_raw": float(kernel[-1]),
    }


def relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), np.finfo(float).tiny)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    forward_root = args.raw_root / FORWARD_CASE
    adjoint_root = args.raw_root / ADJOINT_CASE
    forward_h5 = (
        forward_root
        / "complex_material_control"
        / "complex_material_control_output.h5"
    )
    adjoint_h5 = (
        adjoint_root
        / "au_external_field_adjoint_gpu"
        / "au_external_field_adjoint_gpu_output.h5"
    )
    q_npz = forward_root / "complex_material_control_q.npz"
    raw_boundary_result = (
        adjoint_root / "au_sharp_interface_external_field_result.json"
    )
    full_result = json.loads(raw_boundary_result.read_text())

    base_forward, electric_forward = load_monitor(forward_h5)
    base_adjoint, electric_adjoint = load_monitor(adjoint_h5)
    base_mismatch = max(
        float(np.max(np.abs(base_forward[axis] - base_adjoint[axis])))
        for axis in "xyz"
    )
    coordinates, coordinate_audit = component_coordinates(q_npz, base_forward)
    forward = make_interpolators(electric_forward, coordinates)
    adjoint = make_interpolators(electric_adjoint, coordinates)

    rows: list[dict[str, object]] = []
    dense_profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for n_points in SAMPLE_COUNTS:
        for face, x_m in (("x_min", -HALF_WIDTH_M), ("x_max", HALF_WIDTH_M)):
            y, kernel = face_profile(
                forward, adjoint, x_m=x_m, n_points=n_points
            )
            row = {"face": face, **profile_metrics(y, kernel)}
            rows.append(row)
            if n_points == SAMPLE_COUNTS[-1]:
                dense_profiles[face] = (y, kernel)

    def combined(n_points: int, key: str) -> float:
        return float(
            sum(
                float(row[key])
                for row in rows
                if int(row["n_points"]) == n_points
            )
        )

    endpoint_fraction_801 = combined(801, "endpoint_trapezoid_raw") / combined(
        801, "full_integral_raw"
    )
    interior_change = relative(
        combined(SAMPLE_COUNTS[0], "interior_abs_y_le_9p5um_integral_raw"),
        combined(SAMPLE_COUNTS[-1], "interior_abs_y_le_9p5um_integral_raw"),
    )
    max_locations_801 = [
        float(row["maximum_abs_kernel_y_m"])
        for row in rows
        if int(row["n_points"]) == 801
    ]
    endpoint_values_801 = [
        {
            "face": str(row["face"]),
            "y_min_raw": float(row["kernel_y_min_raw"]),
            "y_max_raw": float(row["kernel_y_max_raw"]),
        }
        for row in rows
        if int(row["n_points"]) == 801
    ]
    diagnosed = bool(
        endpoint_fraction_801 > 0.80
        and interior_change < 5.0e-3
        and all(abs(value - HALF_Y_M) < 1.0e-15 for value in max_locations_801)
        and base_mismatch == 0.0
    )
    status = (
        "DIAGNOSED_AU_BOUNDARY_QUADRATURE_CORNER_DOMINANCE"
        if diagnosed
        else "FAILED_AU_BOUNDARY_CORNER_LOCALIZATION_DIAGNOSTIC"
    )

    summary = {
        "status": status,
        "passed_localization_gate": diagnosed,
        "scope": (
            "offline localization of the tangential-E part of the x-normal "
            "Au boundary kernel; no Maxwell solve and no gradient promotion"
        ),
        "forward_adjoint_HDF5_base_coordinate_mismatch_m": base_mismatch,
        "component_coordinate_audit": coordinate_audit,
        "sample_counts": list(SAMPLE_COUNTS),
        "face_metrics": rows,
        "combined_endpoint_fraction_at_801_points": endpoint_fraction_801,
        "combined_interior_201_to_6401_relative_change": interior_change,
        "maximum_locations_at_801_points_m": max_locations_801,
        "endpoint_values_at_801_points": endpoint_values_801,
        "complete_boundary_kernel_reference": full_result["boundary_quadrature"],
        "interpretation": {
            "established": (
                "the tangential-E boundary proxy is dominated by the two exact "
                "Au corners at y=+-10 um (with the absolute maximum at +10 um), "
                "while the interior |y|<=9.5 um integral is stable under "
                "quadrature refinement"
            ),
            "not_established": (
                "the normal-D term is not available in the engine HDF5 export; "
                "this offline diagnostic alone is not a complete AD-FD certificate"
            ),
            "next_control": (
                "move the fixed Au y-ends far outside the illuminated/adjoint "
                "support and repeat the width AD-FD with a corner-free active x-face"
            ),
        },
        "Maxwell_solves_this_analysis": 0,
        "CPU_FDTD_fallback": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "production_Au_optimization_permitted": False,
    }
    summary_path = output / "au_boundary_corner_localization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    csv_path = output / "au_boundary_corner_localization.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for face, (y, kernel) in dense_profiles.items():
        axes[0].plot(y * 1e6, kernel, label=face)
        zoom = y >= 9.4e-6
        axes[1].plot(y[zoom] * 1e6, kernel[zoom], label=face)
    axes[0].axvline(10.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].axvspan(-9.5, 9.5, color="#2878B5", alpha=0.08, label="interior audit")
    axes[0].set_xlabel("y along moving x-face (um)")
    axes[0].set_ylabel("raw tangential-E kernel proxy")
    axes[0].set_title("Full vertical Au face")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].axvline(10.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("y along moving x-face (um)")
    axes[1].set_ylabel("raw tangential-E kernel proxy")
    axes[1].set_title("Au corner at y=+10 um")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    plot_path = output / "au_boundary_corner_localization.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    report = f"""# Au boundary-quadrature corner-localization diagnostic

Status: `{status}`

This is an offline analysis of the completed GPU forward and adjoint engine
HDF5 files. It launches zero Maxwell solves and uses no Lumerical license. It
examines the tangential-E part of the official x-normal boundary kernel. The
absolute HDF5 field normalization is intentionally not used: localization and
quadrature-convergence ratios are invariant to one global factor.

At 801 points per edge, the two exact trapezoid endpoint samples on each face
(`y=-10 um` and `y=+10 um`) contribute
`{100.0*endpoint_fraction_801:.6f}%` of the complete tangential-E proxy
integral. Both endpoints are large; the absolute maximum on both moving faces
occurs at `y=+10 um`. These are exactly the two corners where the moving
vertical Au face meets the fixed horizontal Au edges. In contrast, the
combined interior integral over `|y|<=9.5 um` changes by only
`{100.0*interior_change:.6f}%` between 201 and 6401 samples.

This establishes that the observed sampling drift is not distributed over the
smooth vertical face. It is localized at the sharp metal corner. The complete
AD result also contains the normal-D term, which the engine HDF5 export does
not provide together with component epsilon; therefore this analysis does not
replace the full AD--FD gate. It does, however, identify the correct next
control: keep the Au y-ends fixed far outside the illuminated and adjoint
support, then repeat the x-width FD and boundary adjoint on the corner-free
active face.

No production gradient is promoted, and Au thermal/electrical/PTE optimization
remains blocked.
"""
    report_path = output / "AU_BOUNDARY_CORNER_LOCALIZATION_REPORT.md"
    report_path.write_text(report)

    raw_paths = [forward_h5, adjoint_h5, q_npz, raw_boundary_result]
    manifest = {
        "status": status,
        "raw_files_committed": False,
        "generation_command": (
            "python 14_analyze_au_boundary_corner_localization.py"
        ),
        "raw_files": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in raw_paths
        ],
    }
    manifest_path = output / "AU_BOUNDARY_CORNER_LOCALIZATION_RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "combined_endpoint_fraction_at_801_points": endpoint_fraction_801,
                "combined_interior_201_to_6401_relative_change": interior_change,
            },
            indent=2,
        )
    )
    return 0 if diagnosed else 2


if __name__ == "__main__":
    raise SystemExit(main())
