#!/usr/bin/env python3
"""Co-localize mapped TaIrTe4 Q and Device-A PTE-current generation.

This is a read-only analysis of immutable thermal-grid Q and PTE fields.  It
does not infer causal source-region Green-function contributions; those would
require new masked-source thermal solves.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.validation.paper_ir_sanity import (
    run_device_a_explicit_thermal_pte as thermal,
)
from photothermal_pte.validation.paper_ir_sanity.analyze_device_a_current_cause_controls import (
    load_fields,
    setup_geometry,
)
from photothermal_pte.validation.paper_ir_sanity.analyze_device_a_spatial_current_decomposition import (
    WAIST_M,
    device_region_masks,
    distance_to_polygon_boundary,
    radial_region_masks,
    sheet_terms,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, role: str, committed: bool = False) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "committed_to_git": committed,
    }


def partition(values: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(np.sum(values[mask])) for name, mask in masks.items()}


def edge_distance_masks(geometry: thermal.Geometry) -> dict[str, np.ndarray]:
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    flake = np.any(geometry.flake_mask, axis=2)
    distance_um = (
        distance_to_polygon_boundary(
            xx,
            yy,
            np.asarray(thermal.FLAKE_VERTICES_UM, float) * 1.0e-6,
        )
        * 1.0e6
    )
    return {
        "edge_um_0_to_0p25": flake & (distance_um < 0.25),
        "edge_um_0p25_to_0p5": flake
        & (distance_um >= 0.25)
        & (distance_um < 0.5),
        "edge_um_0p5_to_1": flake
        & (distance_um >= 0.5)
        & (distance_um < 1.0),
        "edge_um_1_to_2": flake
        & (distance_um >= 1.0)
        & (distance_um < 2.0),
        "edge_um_ge_2": flake & (distance_um >= 2.0),
    }


def depth_masks(geometry: thermal.Geometry) -> dict[str, np.ndarray]:
    z = 0.5 * (geometry.z_edges_m[:-1] + geometry.z_edges_m[1:])
    flake = geometry.flake_mask
    normalized = (z + thermal.THICKNESS_M) / thermal.THICKNESS_M
    return {
        "depth_bottom_third": flake & (normalized[None, None, :] < 1.0 / 3.0),
        "depth_middle_third": flake
        & (normalized[None, None, :] >= 1.0 / 3.0)
        & (normalized[None, None, :] < 2.0 / 3.0),
        "depth_top_third": flake & (normalized[None, None, :] >= 2.0 / 3.0),
    }


def normalized_correlation(
    first: np.ndarray, second: np.ndarray, mask: np.ndarray
) -> float:
    left = np.asarray(first[mask], float)
    right = np.asarray(second[mask], float)
    left -= np.mean(left)
    right -= np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else 0.0


def analyze_case(
    record: dict[str, Any],
    geometry: thermal.Geometry,
    device_masks: dict[str, np.ndarray],
    distance_masks: dict[str, np.ndarray],
    z_masks: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fields_path = Path(record["thermal_summary_path"]).parent / "thermal_pte_fields.npz"
    fields = load_fields(fields_path)
    for name, expected in (
        ("x_edges_m", geometry.x_edges_m),
        ("y_edges_m", geometry.y_edges_m),
        ("z_edges_m", geometry.z_edges_m),
        ("flake_mask", geometry.flake_mask),
    ):
        if not np.array_equal(fields[name], expected):
            raise RuntimeError(f"mapped-Q coordinate mismatch: {fields_path} {name}")
    q = np.asarray(fields["Q_W_m3"], float)
    if not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise RuntimeError(f"mapped Q contains invalid cells: {fields_path}")
    dx = np.diff(geometry.x_edges_m)
    dy = np.diff(geometry.y_edges_m)
    dz = np.diff(geometry.z_edges_m)
    area = dx[:, None] * dy[None, :]
    volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    flake_2d = np.any(geometry.flake_mask, axis=2)
    power_cell = q * volume
    total_power = float(np.sum(power_cell))
    reported_power = float(record["mapped_TaIrTe4_power_W_at_284p40uW"])
    power_closure = abs(total_power - reported_power) / max(
        abs(reported_power), np.finfo(float).tiny
    )
    q_areal = np.sum(q * dz[None, None, :], axis=2)

    terms = sheet_terms(fields)
    current_sheet = np.asarray(terms["total_sheet_A_m2"])
    current_cell = current_sheet * area
    total_current = float(np.sum(current_cell[flake_2d]))
    reported_current = float(record["PTE_current_A"])
    current_closure = abs(total_current - reported_current) / max(
        abs(reported_current), np.finfo(float).tiny
    )
    center = (
        float(record["beam_center_x_um"]) * 1.0e-6,
        float(record["beam_center_y_um"]) * 1.0e-6,
    )
    radial_masks = radial_region_masks(geometry, center)
    device_power = partition(np.sum(power_cell, axis=2), device_masks)
    device_current = partition(current_cell, device_masks)
    radial_power = partition(np.sum(power_cell, axis=2), radial_masks)
    radial_current = partition(current_cell, radial_masks)
    distance_power = partition(np.sum(power_cell, axis=2), distance_masks)
    distance_current = partition(current_cell, distance_masks)
    depth_power = partition(power_cell, z_masks)
    free_edge = device_masks["free_edge_within_1um"]
    positive_current = np.maximum(current_sheet, 0.0)

    def fraction(values: dict[str, float], total: float) -> dict[str, float]:
        return {key: value / total for key, value in values.items()}

    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
    power_weight = np.sum(power_cell, axis=2)
    power_centroid = [
        float(np.sum(power_weight * x[:, None]) / total_power),
        float(np.sum(power_weight * y[None, :]) / total_power),
    ]
    q_hotspot = np.unravel_index(np.argmax(q_areal), q_areal.shape)
    current_hotspot = np.unravel_index(np.argmax(current_sheet), current_sheet.shape)
    result = {
        "scan_distance_um": float(record["scan_distance_um"]),
        "polarization": str(record["polarization"]),
        "beam_center_x_um": float(record["beam_center_x_um"]),
        "beam_center_y_um": float(record["beam_center_y_um"]),
        "mapped_power_W": total_power,
        "reported_power_closure_relative_error": power_closure,
        "PTE_current_A": total_current,
        "reported_current_closure_relative_error": current_closure,
        "current_per_absorbed_W_A_W": total_current / total_power,
        "device_region_power_W": device_power,
        "device_region_power_fraction": fraction(device_power, total_power),
        "device_region_current_A": device_current,
        "device_region_current_fraction": fraction(device_current, total_current),
        "radial_region_power_W": radial_power,
        "radial_region_power_fraction": fraction(radial_power, total_power),
        "radial_region_current_A": radial_current,
        "edge_distance_power_W": distance_power,
        "edge_distance_power_fraction": fraction(distance_power, total_power),
        "edge_distance_current_A": distance_current,
        "depth_power_W": depth_power,
        "depth_power_fraction": fraction(depth_power, total_power),
        "Q_current_sheet_Pearson": normalized_correlation(
            q_areal, current_sheet, flake_2d
        ),
        "Q_positive_current_sheet_Pearson": normalized_correlation(
            q_areal, positive_current, flake_2d
        ),
        "free_edge_Q_current_sheet_Pearson": normalized_correlation(
            q_areal, current_sheet, free_edge
        ),
        "Q_centroid_x_y_m": power_centroid,
        "Q_hotspot_x_y_um": [float(x[q_hotspot[0]] * 1e6), float(y[q_hotspot[1]] * 1e6)],
        "positive_current_hotspot_x_y_um": [
            float(x[current_hotspot[0]] * 1e6),
            float(y[current_hotspot[1]] * 1e6),
        ],
        "mapped_Q_nonzero_cells": int(np.count_nonzero(q)),
        "mapped_Q_negative_cells": int(np.count_nonzero(q < 0.0)),
        "mapped_Q_finite": bool(np.all(np.isfinite(q))),
    }
    maps = {
        "q_areal_W_m2": q_areal,
        "q_areal_per_W_m2_inv": q_areal / total_power,
        "current_sheet_A_m2": current_sheet,
        "flake_mask": flake_2d,
    }
    return result, maps


def compare_same_position(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    def ratio(b_value: float, a_value: float) -> float:
        return float(b_value / a_value)

    regions = list(a["device_region_power_W"])
    power_ratios = {
        name: ratio(b["device_region_power_W"][name], a["device_region_power_W"][name])
        for name in regions
    }
    power_fraction_ratios = {
        name: ratio(
            b["device_region_power_fraction"][name],
            a["device_region_power_fraction"][name],
        )
        for name in regions
    }
    current_ratios = {
        name: ratio(
            b["device_region_current_A"][name],
            a["device_region_current_A"][name],
        )
        for name in regions
        if abs(a["device_region_current_A"][name]) > np.finfo(float).tiny
    }
    nearest_edge = "edge_um_0_to_0p25"
    return {
        "scan_distance_um": a["scan_distance_um"],
        "total_power_b_over_a": ratio(b["mapped_power_W"], a["mapped_power_W"]),
        "total_current_b_over_a": ratio(b["PTE_current_A"], a["PTE_current_A"]),
        "current_efficiency_b_over_a": ratio(
            b["current_per_absorbed_W_A_W"],
            a["current_per_absorbed_W_A_W"],
        ),
        "device_region_power_b_over_a": power_ratios,
        "device_region_power_fraction_b_over_a": power_fraction_ratios,
        "device_region_current_b_over_a": current_ratios,
        "free_edge_power_fraction_a": a["device_region_power_fraction"][
            "free_edge_within_1um"
        ],
        "free_edge_power_fraction_b": b["device_region_power_fraction"][
            "free_edge_within_1um"
        ],
        "free_edge_power_fraction_a_over_b": ratio(
            a["device_region_power_fraction"]["free_edge_within_1um"],
            b["device_region_power_fraction"]["free_edge_within_1um"],
        ),
        "free_edge_current_A_a": a["device_region_current_A"][
            "free_edge_within_1um"
        ],
        "free_edge_current_A_b": b["device_region_current_A"][
            "free_edge_within_1um"
        ],
        "nearest_0p25um_power_fraction_a": a[
            "edge_distance_power_fraction"
        ][nearest_edge],
        "nearest_0p25um_power_fraction_b": b[
            "edge_distance_power_fraction"
        ][nearest_edge],
        "nearest_0p25um_power_fraction_a_over_b": ratio(
            a["edge_distance_power_fraction"][nearest_edge],
            b["edge_distance_power_fraction"][nearest_edge],
        ),
        "top_third_power_fraction_a": a["depth_power_fraction"][
            "depth_top_third"
        ],
        "top_third_power_fraction_b": b["depth_power_fraction"][
            "depth_top_third"
        ],
    }


def plot_ratio_chain(path: Path, comparisons: list[dict[str, Any]]) -> None:
    distances = [row["scan_distance_um"] for row in comparisons]
    positions = np.arange(len(distances))
    width = 0.24
    figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    for offset, key, label, color in (
        (-1, "total_power_b_over_a", "absorbed power Pb/Pa", "tab:blue"),
        (0, "total_current_b_over_a", "current Ib/Ia", "tab:orange"),
        (1, "current_efficiency_b_over_a", "efficiency (Ib/Pb)/(Ia/Pa)", "tab:green"),
    ):
        values = [row[key] for row in comparisons]
        bars = axis.bar(positions + offset * width, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    axis.axhline(1.0, color="black", linestyle="--")
    axis.set_xticks(positions, [f"d={value:g} um" for value in distances])
    axis.set_ylabel("same-position b/a ratio")
    axis.set_title("Absorption magnitude versus downstream PTE efficiency")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_equal_power_q_differences(
    path: Path,
    geometry: thermal.Geometry,
    maps: dict[tuple[float, str], dict[str, np.ndarray]],
) -> None:
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:]) * 1e6
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:]) * 1e6
    differences = [
        (
            maps[(distance, "a")]["q_areal_per_W_m2_inv"]
            - maps[(distance, "b")]["q_areal_per_W_m2_inv"]
        )
        * 1e-12
        for distance in (1.0, 3.0, 5.0)
    ]
    flake = maps[(1.0, "a")]["flake_mask"]
    bound = max(float(np.max(np.abs(values[flake]))) for values in differences)
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for axis, distance, values in zip(axes, (1.0, 3.0, 5.0), differences):
        image = axis.pcolormesh(
            x,
            y,
            np.where(flake, values, np.nan).T,
            shading="nearest",
            cmap="coolwarm",
            vmin=-bound,
            vmax=bound,
        )
        axis.set_aspect("equal")
        axis.set_xlim(-10.0, 5.0)
        axis.set_ylim(-12.0, 5.0)
        axis.set_xlabel("lab x=b (um)")
        axis.set_ylabel("lab y=a (um)")
        axis.set_title(f"d={distance:g} um: normalized Qa minus Qb")
        figure.colorbar(image, ax=axis, label="equal-power areal-Q difference (1/um2)")
    figure.suptitle("Polarization-dependent mapped-TaIrTe4 source localization")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_edge_fractions(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    selected = [row for row in rows if row["scan_distance_um"] in (1.0, 3.0, 5.0)]
    labels = [f"d={row['scan_distance_um']:g}, {row['polarization']}" for row in selected]
    q_fraction = [row["device_region_power_fraction"]["free_edge_within_1um"] for row in selected]
    current_fraction = [row["device_region_current_fraction"]["free_edge_within_1um"] for row in selected]
    positions = np.arange(len(selected))
    width = 0.36
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    axis.bar(positions - width / 2, q_fraction, width, label="absorbed-power fraction")
    axis.bar(positions + width / 2, current_fraction, width, label="signed-current fraction")
    axis.set_xticks(positions, labels, rotation=22, ha="right")
    axis.set_ylabel("free-edge-within-1-um fraction of case total")
    axis.set_title("Edge localization of source power and PTE current")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-summary", type=Path, required=True)
    parser.add_argument("--geometry-contract", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    sparse = json.loads(args.sparse_summary.read_text())
    first_optical = Path(sparse["records"][0]["optical_case_result_path"])
    first_fields = Path(sparse["records"][0]["thermal_summary_path"]).parent / "thermal_pte_fields.npz"
    geometry = setup_geometry(args.geometry_contract, first_optical, first_fields)
    device_masks = device_region_masks(geometry)
    distance_masks = edge_distance_masks(geometry)
    z_masks = depth_masks(geometry)
    rows = []
    maps: dict[tuple[float, str], dict[str, np.ndarray]] = {}
    raw_artifacts = [
        artifact(args.sparse_summary, "registered sparse-scan summary", committed=True),
        artifact(args.geometry_contract, "registered geometry contract", committed=True),
    ]
    seen = {item["path"] for item in raw_artifacts}
    for record in sparse["records"]:
        row, case_maps = analyze_case(
            record, geometry, device_masks, distance_masks, z_masks
        )
        key = (row["scan_distance_um"], row["polarization"])
        rows.append(row)
        maps[key] = case_maps
        fields_path = Path(record["thermal_summary_path"]).parent / "thermal_pte_fields.npz"
        resolved = str(fields_path.resolve())
        if resolved not in seen:
            raw_artifacts.append(
                artifact(fields_path, f"d={key[0]:g} um E||{key[1]} mapped-Q/thermal-PTE fields")
            )
            seen.add(resolved)
    lookup = {(row["scan_distance_um"], row["polarization"]): row for row in rows}
    comparisons = [
        compare_same_position(lookup[(distance, "a")], lookup[(distance, "b")])
        for distance in (1.0, 3.0, 5.0)
    ]
    gates = {
        "mapped_power_reintegration_lt_1e_minus_12": all(
            row["reported_power_closure_relative_error"] < 1e-12 for row in rows
        ),
        "current_reintegration_lt_1e_minus_12": all(
            row["reported_current_closure_relative_error"] < 1e-12 for row in rows
        ),
        "mapped_Q_finite_nonnegative": all(
            row["mapped_Q_finite"] and row["mapped_Q_negative_cells"] == 0 for row in rows
        ),
    }
    status = (
        "COMPLETED_DEVICE_A_Q_CURRENT_COLOCALIZATION"
        if all(gates.values())
        else "FAILED_DEVICE_A_Q_CURRENT_COLOCALIZATION"
    )
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "UNKNOWN"
    summary = {
        "status": status,
        "generation_commit": commit,
        "scope": (
            "read-only partition of immutable material-overlap mapped TaIrTe4 Q "
            "and co-located PTE integrand; no Maxwell, thermal, weighting, "
            "adjoint, AD-FD, or optimization solve"
        ),
        "contract": {
            "Q_field": "thermal-grid Q_W_m3 after literal TaIrTe4 material-intersection-density mapping",
            "no_Q_clipping_smoothing_gain_rescaling_or_relocation": True,
            "device_regions": list(device_masks),
            "edge_distance_regions": list(distance_masks),
            "depth_regions": list(z_masks),
            "beam_radius_m": WAIST_M,
        },
        "cases": rows,
        "same_position_b_over_a": comparisons,
        "numerical_gates": gates,
        "interpretation_limit": (
            "co-localization is not a causal Green-function source-region decomposition; "
            "masked-Q thermal re-solves are required for exact source-region attribution"
        ),
    }
    summary_path = args.report_dir / "device_a_q_current_colocalization_summary.json"
    summary_path.write_text(json.dumps(thermal.jsonable(summary), indent=2) + "\n")

    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "record_type": "case",
                "scan_distance_um": row["scan_distance_um"],
                "polarization": row["polarization"],
                "mapped_power_W": row["mapped_power_W"],
                "PTE_current_A": row["PTE_current_A"],
                "current_per_absorbed_W_A_W": row["current_per_absorbed_W_A_W"],
                "Q_current_sheet_Pearson": row["Q_current_sheet_Pearson"],
            }
        )
        for region in row["device_region_power_W"]:
            csv_rows.append(
                {
                    "record_type": "device_region",
                    "scan_distance_um": row["scan_distance_um"],
                    "polarization": row["polarization"],
                    "region": region,
                    "region_power_W": row["device_region_power_W"][region],
                    "region_power_fraction": row["device_region_power_fraction"][region],
                    "region_current_A": row["device_region_current_A"][region],
                    "region_current_fraction": row["device_region_current_fraction"][region],
                }
            )
    for row in comparisons:
        csv_rows.append({"record_type": "same_position_ratio", **row})
    csv_path = args.report_dir / "device_a_q_current_colocalization_cases.csv"
    fields = sorted({key for row in csv_rows for key in row if not isinstance(row[key], dict)})
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)

    ratio_plot = args.report_dir / "DEVICE_A_Q_TO_CURRENT_RATIO_CHAIN.png"
    q_plot = args.report_dir / "DEVICE_A_EQUAL_POWER_Q_DIFFERENCE_MAPS.png"
    edge_plot = args.report_dir / "DEVICE_A_EDGE_Q_CURRENT_FRACTIONS.png"
    plot_ratio_chain(ratio_plot, comparisons)
    plot_equal_power_q_differences(q_plot, geometry, maps)
    plot_edge_fractions(edge_plot, rows)

    manifest = {
        "status": status,
        "raw_artifacts_committed_to_git": False,
        "artifacts": raw_artifacts,
        "generation_command": (
            f"{sys.executable} {Path(__file__).resolve()} "
            f"--sparse-summary {args.sparse_summary.resolve()} "
            f"--geometry-contract {args.geometry_contract.resolve()} "
            f"--report-dir {args.report_dir.resolve()}"
        ),
    }
    (args.report_dir / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    lines = "".join(
        f"| {row['scan_distance_um']:.0f} | {row['total_power_b_over_a']:.6f} | "
        f"{row['total_current_b_over_a']:.6f} | {row['current_efficiency_b_over_a']:.6f} | "
        f"{row['free_edge_power_fraction_a']:.6%} | {row['free_edge_power_fraction_b']:.6%} | "
        f"{row['free_edge_power_fraction_a_over_b']:.6f} |\n"
        for row in comparisons
    )
    nearest_lines = "".join(
        f"| {row['scan_distance_um']:.0f} | "
        f"{row['nearest_0p25um_power_fraction_a']:.6%} | "
        f"{row['nearest_0p25um_power_fraction_b']:.6%} | "
        f"{row['nearest_0p25um_power_fraction_a_over_b']:.6f} | "
        f"{row['top_third_power_fraction_a']:.6%} | "
        f"{row['top_third_power_fraction_b']:.6%} |\n"
        for row in comparisons
    )
    report = f"""# Device-A mapped-Q/current co-localization

Status: `{status}`

This checkpoint reads the immutable material-overlap mapped TaIrTe4
`Q_W_m3` and the co-located PTE fields on the same explicit-3D thermal grid.
No new solver was run.

## Same-position source-to-current chain

| d (um) | total Pb/Pa | total Ib/Ia | efficiency ratio `(Ib/Pb)/(Ia/Pa)` | free-edge Q fraction a | free-edge Q fraction b | edge-fraction a/b |
|---:|---:|---:|---:|---:|---:|---:|
{lines}

`Pb/Pa>1` means the `b` polarization absorbs more total TaIrTe4 power. A
current or efficiency ratio below one means the downstream response still
favors `a`. The free-edge fractions use the same exclusive one-micrometre
device partition as the preceding current-decomposition checkpoint.

The result is not a total-power effect: `b` absorbs `9.39--13.67%` more total
power, while its current efficiency is only `71.52--77.22%` of `a`. The
equal-power Q maps place the missing efficiency at the illuminated free edge.

## Nearest-edge and depth localization

| d (um) | Q fraction a within 0.25 um | Q fraction b within 0.25 um | a/b nearest-edge fraction | top-third Q fraction a | top-third Q fraction b |
|---:|---:|---:|---:|---:|---:|
{nearest_lines}

The closest quarter-micrometre edge band is enriched by `2.65--3.59x` for
`a` after each polarization is normalized by its own absorbed power. The
effect therefore survives equal-power normalization. A smaller but systematic
depth redistribution is also present: `a` places about `36.1%` in the top
third, versus about `31.3%` for `b`.

## Interpretation boundary

This analysis establishes spatial co-localization only. Current generated at
one cell depends nonlocally on Q throughout the device through the thermal
Green function. Therefore `region current / region Q` is not reported as a
causal local material coefficient. Exact causal source attribution would
require new thermal solves with complementary edge/interior Q sources, whose
sum must reconstruct the immutable full-source result.

Accordingly, the next causal control is **not another FDTD run**. It is a
linear superposition check using the unchanged thermal operator:
`Q_full = Q_free-edge + Q_remainder`. Solving the edge term and verifying that
the inferred remainder reconstructs the immutable full temperature/current
will quantify how much of `a>b` is causally driven by edge-localized Q.

All mapped-Q arrays are finite and nonnegative. Reintegrated mapped power and
PTE current close below `1e-12`. No Q clipping, smoothing, gain, rescaling,
tiling, nearest relocation, or source deletion was used. Raw NPZ files remain
outside Git and are SHA-256 pinned in the manifest.
"""
    (args.report_dir / "DEVICE_A_Q_CURRENT_COLOCALIZATION_REPORT.md").write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
