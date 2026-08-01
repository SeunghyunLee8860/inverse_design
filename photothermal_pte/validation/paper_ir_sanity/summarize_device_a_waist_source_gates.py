#!/usr/bin/env python3
"""Summarize explicit-waist source-only gates and Device-A illumination overlap."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=/absolute/case/directory")
    return name, Path(path).expanduser().resolve()


def dual_edges(coordinates: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    edges = np.empty(coordinates.size + 1)
    edges[1:-1] = 0.5 * (coordinates[:-1] + coordinates[1:])
    edges[0], edges[-1] = bounds
    return edges


def clip_half_plane(
    polygon: list[np.ndarray], axis: int, boundary: float, keep_greater: bool
) -> list[np.ndarray]:
    if not polygon:
        return []
    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_inside = (
        previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
    )
    for current in polygon:
        current_inside = (
            current[axis] >= boundary if keep_greater else current[axis] <= boundary
        )
        if current_inside != previous_inside:
            fraction = (boundary - previous[axis]) / (current[axis] - previous[axis])
            intersection = previous + fraction * (current - previous)
            output.append(intersection)
        if current_inside:
            output.append(current)
        previous, previous_inside = current, current_inside
    return output


def rectangle_polygon_overlap_area(
    vertices: np.ndarray, x0: float, x1: float, y0: float, y1: float
) -> float:
    polygon = [np.asarray(point, float) for point in vertices]
    for axis, boundary, keep_greater in (
        (0, x0, True),
        (0, x1, False),
        (1, y0, True),
        (1, y1, False),
    ):
        polygon = clip_half_plane(polygon, axis, boundary, keep_greater)
        if not polygon:
            return 0.0
    points = np.asarray(polygon)
    return 0.5 * abs(
        np.dot(points[:, 0], np.roll(points[:, 1], -1))
        - np.dot(points[:, 1], np.roll(points[:, 0], -1))
    )


def polygon_overlap_weights(
    x_edges: np.ndarray, y_edges: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    weights = np.zeros((x_edges.size - 1, y_edges.size - 1), float)
    x_indices = np.flatnonzero(
        (x_edges[:-1] < np.max(vertices[:, 0]))
        & (x_edges[1:] > np.min(vertices[:, 0]))
    )
    y_indices = np.flatnonzero(
        (y_edges[:-1] < np.max(vertices[:, 1]))
        & (y_edges[1:] > np.min(vertices[:, 1]))
    )
    for i in x_indices:
        for j in y_indices:
            weights[i, j] = rectangle_polygon_overlap_area(
                vertices,
                x_edges[i],
                x_edges[i + 1],
                y_edges[j],
                y_edges[j + 1],
            )
    return weights


def load_case(path: Path, polygons_m: dict[str, np.ndarray]) -> dict[str, Any]:
    result_path = path / "source_only_case_result.json"
    field_path = path / "paper_ir_source_only_fields.npz"
    result = json.loads(result_path.read_text())
    with np.load(field_path) as raw:
        x = np.asarray(raw["flake_target_plane_x_m"], float)
        y = np.asarray(raw["flake_target_plane_y_m"], float)
        flux = np.asarray(raw["flake_target_plane_downward_Poynting_W_m2"], float)
    # The saved target-plane monitor spans the 50-um source square.  Its
    # reported power uses trapezoidal nodal quadrature, which is exactly the
    # bounded-dual-cell rule with the first/last monitor coordinates as
    # bounds.  Do not extrapolate the boundary samples to the 60-um FDTD box.
    x_edges = dual_edges(x, (float(x[0]), float(x[-1])))
    y_edges = dual_edges(y, (float(y[0]), float(y[-1])))
    cell_area = np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :]
    integrated = float(np.sum(flux * cell_area))
    overlap_power = {}
    overlap_weights = {}
    for name, vertices in polygons_m.items():
        weights = polygon_overlap_weights(x_edges, y_edges, vertices)
        overlap_weights[name] = weights
        overlap_power[name] = float(np.sum(flux * weights))
    electrode_power = overlap_power["top_electrode"] + overlap_power["bottom_electrode"]
    focus = result["planes"]["flake_target_plane"]
    built = result["pre_run"]["built_contract"]["source"]
    return {
        "path": path,
        "result_path": result_path,
        "field_path": field_path,
        "status": result["status"],
        "gate_passed": bool(result.get("source_only_gate_passed", False)),
        "target_waist_um": float(built["target_realized_waist_radius_m"] * 1e6),
        "source_object_waist_um": float(
            built["Lumerical_source_object_waist_radius_m"] * 1e6
        ),
        "fitted_waist_x_um": float(focus["fitted_waist_x_m"] * 1e6),
        "fitted_waist_y_um": float(focus["fitted_waist_y_m"] * 1e6),
        "fitted_waist_effective_um": float(focus["fitted_waist_effective_m"] * 1e6),
        "waist_effective_relative_error": abs(
            float(focus["fitted_waist_effective_m"] * 1e6)
            - float(built["target_realized_waist_radius_m"] * 1e6)
        ) / float(built["target_realized_waist_radius_m"] * 1e6),
        "Gaussian_fit_NRMSE": float(focus["Gaussian_fit_NRMSE"]),
        "ellipticity": float(focus["fitted_xy_ellipticity"]),
        "boundary_max_intensity_over_peak": float(
            focus["boundary_max_intensity_over_peak"]
        ),
        "incident_plane_power_W_reintegrated": integrated,
        "incident_plane_power_W_reported": float(focus["downward_Poynting_power_W"]),
        "incident_plane_reintegration_relative_error": abs(
            integrated - float(focus["downward_Poynting_power_W"])
        ) / abs(float(focus["downward_Poynting_power_W"])),
        "top_electrode_power_fraction": overlap_power["top_electrode"] / integrated,
        "bottom_electrode_power_fraction": overlap_power["bottom_electrode"] / integrated,
        "total_electrode_power_fraction": electrode_power / integrated,
        "flake_power_fraction": overlap_power["flake"] / integrated,
        "field_component_E2_fraction": {
            "x": float(focus["x_polarization_E2_fraction"]),
            "y": float(focus["cross_polarized_Ey_E2_fraction"]),
            "z": float(focus["longitudinal_Ez_E2_fraction"]),
        },
        "failed_gates": [
            name for name, passed in result["acceptance"].items() if not passed
        ],
        "x_m": x,
        "y_m": y,
        "flux_W_m2": flux,
        "polygons_m": polygons_m,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-contract", type=Path, required=True)
    parser.add_argument("--case", action="append", type=parse_named_path, required=True)
    parser.add_argument("--primary-case", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry = json.loads(args.geometry_contract.read_text())
    beam = np.asarray(geometry["pre_registered_beam_center_code_um"], float)
    polygons_m = {
        "flake": (
            np.asarray(geometry["flake_vertices_code_um"], float) - beam
        ) * 1e-6,
        "top_electrode": (
            np.asarray(geometry["top_metal_polygon_code_um"], float) - beam
        ) * 1e-6,
        "bottom_electrode": (
            np.asarray(geometry["bottom_metal_polygon_code_um"], float) - beam
        ) * 1e-6,
    }
    cases = {name: load_case(path, polygons_m) for name, path in args.case}
    primary = {name: cases[name] for name in args.primary_case}
    rows = []
    for name, case in cases.items():
        rows.append(
            {
                key: value
                for key, value in {"case": name, **case}.items()
                if key
                not in (
                    "path",
                    "result_path",
                    "field_path",
                    "x_m",
                    "y_m",
                    "flux_W_m2",
                    "polygons_m",
                    "field_component_E2_fraction",
                    "failed_gates",
                )
            }
            | {
                "E2_x_fraction": case["field_component_E2_fraction"]["x"],
                "E2_y_fraction": case["field_component_E2_fraction"]["y"],
                "E2_z_fraction": case["field_component_E2_fraction"]["z"],
                "failed_gates": ";".join(case["failed_gates"]),
            }
        )
    with (args.output_dir / "device_a_waist_source_gate_cases.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    compact_primary = {
        name: {
            key: value
            for key, value in case.items()
            if key
            not in (
                "path",
                "result_path",
                "field_path",
                "x_m",
                "y_m",
                "flux_W_m2",
                "polygons_m",
            )
        }
        for name, case in primary.items()
    }
    reference = cases.get("w12_validated")
    compact_reference = (
        None
        if reference is None
        else {
            key: value
            for key, value in reference.items()
            if key
            not in (
                "path",
                "result_path",
                "field_path",
                "x_m",
                "y_m",
                "flux_W_m2",
                "polygons_m",
            )
        }
    )
    any_passed = any(case["gate_passed"] for case in primary.values())
    summary = {
        "status": (
            "VALIDATED_AT_LEAST_ONE_DEVICE_A_WAIST_SOURCE_GATE"
            if any_passed
            else "BLOCKED_DEVICE_A_WAIST_SWEEP_NO_SOURCE_GATE_PASSED"
        ),
        "paper_spot_definition_ambiguity": True,
        "paper_SI_w0_definition": "1/e^2 intensity radius",
        "primary_cases": compact_primary,
        "preserved_w12_large_beam_reference": compact_reference,
        "material_FDTD_authorized_waists_um": [
            case["target_waist_um"] for case in primary.values() if case["gate_passed"]
        ],
        "material_FDTD_started": False,
        "no_gate_relaxation_or_field_rescaling": True,
    }
    (args.output_dir / "device_a_waist_source_gate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for name, case in cases.items():
        marker = "o" if name in primary else "x"
        axes[0].scatter(case["target_waist_um"], case["fitted_waist_effective_um"], marker=marker)
        axes[1].scatter(case["target_waist_um"], 100 * case["Gaussian_fit_NRMSE"], marker=marker)
    axes[0].plot([4, 9], [4, 9], "k--", label="target")
    axes[0].set(xlabel="target w0 (um)", ylabel="realized fitted w0 (um)")
    axes[1].axhline(0.5, color="black", linestyle="--", label="0.5% gate")
    axes[1].set(xlabel="target w0 (um)", ylabel="Gaussian fit NRMSE (%)")
    ordered = sorted(primary.values(), key=lambda item: item["target_waist_um"])
    illumination_cases = ordered + ([] if reference is None else [reference])
    for key, marker, label in (
        ("total_electrode_power_fraction", "o-", "Au/Ti total"),
        ("top_electrode_power_fraction", "^-", "top Au/Ti"),
        ("bottom_electrode_power_fraction", "v-", "bottom Au/Ti"),
    ):
        axes[2].semilogy(
            [case["target_waist_um"] for case in illumination_cases],
            [100 * case[key] for case in illumination_cases],
            marker,
            label=label,
        )
    axes[2].set(
        xlabel="target w0 (um)",
        ylabel="exact Au/Ti target-plane power fraction (%)",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    for axis in axes:
        axis.legend(fontsize=7)
    figure.savefig(args.output_dir / "DEVICE_A_WAIST_SOURCE_GATE_METRICS.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, len(ordered), figsize=(5 * len(ordered), 4.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, case in zip(axes, ordered):
        handle = axis.pcolormesh(
            case["x_m"] * 1e6,
            case["y_m"] * 1e6,
            case["flux_W_m2"].T,
            shading="nearest",
            cmap="magma",
        )
        for name, vertices in case["polygons_m"].items():
            closed = np.vstack((vertices, vertices[0])) * 1e6
            axis.plot(closed[:, 0], closed[:, 1], label=name)
        axis.set(
            title=f"target {case['target_waist_um']:g} um; fit {case['fitted_waist_effective_um']:.3f} um",
            xlabel="x=b relative to beam (um)",
            ylabel="y=a relative to beam (um)",
            aspect="equal",
        )
        axis.legend(fontsize=7)
        figure.colorbar(handle, ax=axis, label="downward Poynting (W/m2)")
    figure.savefig(args.output_dir / "DEVICE_A_WAIST_TARGET_PLANE_ILLUMINATION.png", dpi=180)
    plt.close(figure)
    artifacts = []
    for name, case in cases.items():
        for role, path in (
            ("source result", case["result_path"]),
            ("source field", case["field_path"]),
            ("source project", case["path"] / "paper_ir_source_only.fsp"),
        ):
            artifacts.append(
                {
                    "role": f"{name} {role}",
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "committed_to_git": False,
                }
            )
    (args.output_dir / "RAW_ARTIFACT_MANIFEST_WAIST_SOURCE_ONLY.json").write_text(
        json.dumps(
            {
                "status": "RECORDED_EXTERNAL_RAW_ARTIFACTS_NOT_COMMITTED",
                "artifacts": artifacts,
            },
            indent=2,
        )
        + "\n"
    )
    table = "\n".join(
        f"| {case['target_waist_um']:g} | {case['source_object_waist_um']:.6f} | {case['fitted_waist_x_um']:.4f} | {case['fitted_waist_y_um']:.4f} | {100*case['Gaussian_fit_NRMSE']:.4f}% | {100*case['ellipticity']:.4f}% | {100*case['total_electrode_power_fraction']:.4f}% | {case['gate_passed']} |"
        for case in ordered
    )
    reference_text = (
        ""
        if reference is None
        else (
            f"\nThe preserved 12-um large-beam source passes its historical gate, "
            f"has fitted waist `{reference['fitted_waist_effective_um']:.4f} um`, "
            f"and sends `{100*reference['total_electrode_power_fraction']:.4f}%` "
            "of the stored target-plane downward power through the digitized "
            "Au/Ti polygons."
        )
    )
    report = f"""# Device-A explicit-waist source-only gates

Status: `{summary['status']}`

The paper SI defines `w0` as the 1/e^2 intensity radius, but the main-text
9--16 um diffraction-limited spot does not identify its radius/diameter
convention.  These are explicit sensitivity scenarios, not paper-certified
beam measurements.

| target w0 (um) | source-object w0 (um) | fitted wx (um) | fitted wy (um) | fit NRMSE | ellipticity | Au/Ti incident fraction | gate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
{table}

Power fractions are exact polygon--bounded-dual-cell overlaps of the stored
target-plane downward Poynting field.  They are not single-point intensity
estimates.  Failed source fields are retained as diagnostics and are not
used for material Q, thermal, or terminal-current calculations.

No fit threshold was relaxed and no field, power, Q, or current was rescaled.
The pre-existing 12-um large-beam scenario remains unchanged.
{reference_text}
"""
    (args.output_dir / "DEVICE_A_WAIST_SOURCE_GATE_REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2))
    return 0 if any_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
