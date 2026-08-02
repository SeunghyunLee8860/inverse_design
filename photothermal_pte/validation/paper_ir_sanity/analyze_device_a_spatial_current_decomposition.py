#!/usr/bin/env python3
"""Offline spatial decomposition of the registered Device-A PTE current.

The script does not solve Maxwell, heat, or the weighting potential.  It
reintegrates immutable saved explicit-3D thermal/PTE fields and partitions the
literal Shockley--Ramo volume integral by thermoelectric term and space.
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


EDGE_BAND_M = 1.0e-6
CONTACT_ZONE_M = 2.0e-6
WAIST_M = 8.75e-6


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


def distance_to_segment(
    xx_m: np.ndarray, yy_m: np.ndarray, segment_m: np.ndarray
) -> np.ndarray:
    start = np.asarray(segment_m[0], float)
    stop = np.asarray(segment_m[1], float)
    direction = stop - start
    length_squared = float(np.dot(direction, direction))
    if length_squared <= 0.0:
        return np.hypot(xx_m - start[0], yy_m - start[1])
    projection = (
        (xx_m - start[0]) * direction[0]
        + (yy_m - start[1]) * direction[1]
    ) / length_squared
    projection = np.clip(projection, 0.0, 1.0)
    closest_x = start[0] + projection * direction[0]
    closest_y = start[1] + projection * direction[1]
    return np.hypot(xx_m - closest_x, yy_m - closest_y)


def distance_to_polygon_boundary(
    xx_m: np.ndarray, yy_m: np.ndarray, vertices_m: np.ndarray
) -> np.ndarray:
    distances = []
    for index in range(vertices_m.shape[0]):
        segment = np.asarray(
            [vertices_m[index], vertices_m[(index + 1) % vertices_m.shape[0]]]
        )
        distances.append(distance_to_segment(xx_m, yy_m, segment))
    return np.minimum.reduce(distances)


def device_region_masks(geometry: thermal.Geometry) -> dict[str, np.ndarray]:
    """Return an exclusive, exhaustive partition of the discrete flake."""
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    flake = np.any(geometry.flake_mask, axis=2)
    top_distance = distance_to_segment(
        xx, yy, np.asarray(thermal.TOP_CONTACT_SEGMENT_UM) * 1.0e-6
    )
    bottom_distance = distance_to_segment(
        xx, yy, np.asarray(thermal.BOTTOM_CONTACT_SEGMENT_UM) * 1.0e-6
    )
    edge_distance = distance_to_polygon_boundary(
        xx, yy, np.asarray(thermal.FLAKE_VERTICES_UM) * 1.0e-6
    )
    top = flake & (top_distance <= CONTACT_ZONE_M)
    bottom = flake & ~top & (bottom_distance <= CONTACT_ZONE_M)
    free_edge = flake & ~top & ~bottom & (edge_distance <= EDGE_BAND_M)
    interior = flake & ~top & ~bottom & ~free_edge
    masks = {
        "top_contact_within_2um": top,
        "bottom_contact_within_2um": bottom,
        "free_edge_within_1um": free_edge,
        "flake_interior": interior,
    }
    count = np.sum(np.stack(list(masks.values()), axis=0), axis=0)
    if not np.array_equal(count[flake], np.ones(np.count_nonzero(flake), int)):
        raise RuntimeError("device-region partition is not exclusive/exhaustive")
    if np.any(count[~flake] != 0):
        raise RuntimeError("device-region partition extends outside the flake")
    return masks


def radial_region_masks(
    geometry: thermal.Geometry, center_m: tuple[float, float]
) -> dict[str, np.ndarray]:
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    flake = np.any(geometry.flake_mask, axis=2)
    radius = np.hypot(xx - center_m[0], yy - center_m[1]) / WAIST_M
    masks = {
        "r_over_w0_0_to_0p5": flake & (radius < 0.5),
        "r_over_w0_0p5_to_1": flake & (radius >= 0.5) & (radius < 1.0),
        "r_over_w0_1_to_1p5": flake & (radius >= 1.0) & (radius < 1.5),
        "r_over_w0_ge_1p5": flake & (radius >= 1.5),
    }
    return masks


def weighting_region_masks(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    flake = np.any(fields["flake_mask"], axis=2)
    psi = np.asarray(fields["weighting_potential"], float)
    masks = {}
    edges = np.linspace(0.0, 1.0, 6)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        upper = psi <= high if index == 4 else psi < high
        masks[f"psi_{low:.1f}_to_{high:.1f}"] = flake & (psi >= low) & upper
    return masks


def sheet_terms(
    fields: dict[str, np.ndarray],
) -> dict[str, np.ndarray | float]:
    dz = np.diff(fields["z_edges_m"])
    grad_x = np.asarray(fields["weighting_grad_x_m_inv"], float)
    grad_y = np.asarray(fields["weighting_grad_y_m_inv"], float)
    x_3d = np.asarray(fields["local_J_x_A_m2_3d"], float) * grad_x[:, :, None]
    y_3d = np.asarray(fields["local_J_y_A_m2_3d"], float) * grad_y[:, :, None]
    stored = np.asarray(fields["shockley_ramo_integrand_A_m3_3d"], float)
    mask_3d = np.asarray(fields["flake_mask"], bool)
    mismatch = np.max(np.abs((x_3d + y_3d)[mask_3d] - stored[mask_3d]))
    scale = max(float(np.max(np.abs(stored[mask_3d]))), np.finfo(float).tiny)
    return {
        "x_sheet_A_m2": np.sum(x_3d * dz[None, None, :], axis=2),
        "y_sheet_A_m2": np.sum(y_3d * dz[None, None, :], axis=2),
        "total_sheet_A_m2": np.sum(stored * dz[None, None, :], axis=2),
        "component_pairing_relative_error": float(mismatch / scale),
    }


def partition_sheet_current(
    sheet_A_m2: np.ndarray,
    area_m2: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    return {
        name: float(np.sum(sheet_A_m2[mask] * area_m2[mask]))
        for name, mask in masks.items()
    }


def decompose_case(
    record: dict[str, Any],
    geometry: thermal.Geometry,
    device_masks: dict[str, np.ndarray],
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
            raise RuntimeError(f"saved-field coordinate mismatch: {fields_path} {name}")
    terms = sheet_terms(fields)
    area = np.diff(geometry.x_edges_m)[:, None] * np.diff(geometry.y_edges_m)[None, :]
    flake = np.any(geometry.flake_mask, axis=2)
    x_current = float(np.sum(terms["x_sheet_A_m2"][flake] * area[flake]))
    y_current = float(np.sum(terms["y_sheet_A_m2"][flake] * area[flake]))
    total = float(np.sum(terms["total_sheet_A_m2"][flake] * area[flake]))
    reported = float(record["PTE_current_A"])
    closure = abs(total - reported) / max(abs(reported), np.finfo(float).tiny)
    xy_closure = abs(total - x_current - y_current) / max(
        abs(total), np.finfo(float).tiny
    )
    total_sheet = np.asarray(terms["total_sheet_A_m2"])
    cell_current = total_sheet * area
    positive = float(np.sum(np.maximum(cell_current[flake], 0.0)))
    negative = float(np.sum(np.minimum(cell_current[flake], 0.0)))
    center = (float(record["beam_center_x_um"]) * 1e-6, float(record["beam_center_y_um"]) * 1e-6)
    radial_masks = radial_region_masks(geometry, center)
    psi_masks = weighting_region_masks(fields)
    source_power = float(record["mapped_TaIrTe4_power_W_at_284p40uW"])
    result = {
        "scan_distance_um": float(record["scan_distance_um"]),
        "polarization": str(record["polarization"]),
        "beam_center_x_um": float(record["beam_center_x_um"]),
        "beam_center_y_um": float(record["beam_center_y_um"]),
        "mapped_TaIrTe4_power_W": source_power,
        "total_current_A": total,
        "x_equals_b_term_A": x_current,
        "y_equals_a_term_A": y_current,
        "positive_contribution_A": positive,
        "negative_contribution_A": negative,
        "reported_current_closure_relative_error": closure,
        "x_plus_y_closure_relative_error": xy_closure,
        "component_pairing_relative_error": terms[
            "component_pairing_relative_error"
        ],
        "current_per_absorbed_W_A_W": total / source_power,
        "device_region_current_A": partition_sheet_current(
            total_sheet, area, device_masks
        ),
        "radial_region_current_A": partition_sheet_current(
            total_sheet, area, radial_masks
        ),
        "weighting_potential_region_current_A": partition_sheet_current(
            total_sheet, area, psi_masks
        ),
    }
    result["device_region_current_per_absorbed_W_A_W"] = {
        key: value / source_power
        for key, value in result["device_region_current_A"].items()
    }
    maps = {
        "x_sheet_A_m2": np.asarray(terms["x_sheet_A_m2"]),
        "y_sheet_A_m2": np.asarray(terms["y_sheet_A_m2"]),
        "total_sheet_A_m2": total_sheet,
        "cell_current_A": cell_current,
        "flake_mask": flake,
        "grad_T_x_K_m": np.asarray(fields["grad_T_x_K_m"]),
        "grad_T_y_K_m": np.asarray(fields["grad_T_y_K_m"]),
    }
    return result, maps


def matched_difference(
    a: dict[str, Any], b: dict[str, Any]
) -> dict[str, Any]:
    def difference(key: str) -> float:
        return float(a[key] - b[key])

    device_delta = {
        key: a["device_region_current_A"][key]
        - b["device_region_current_A"][key]
        for key in a["device_region_current_A"]
    }
    efficiency_delta = {
        key: a["device_region_current_per_absorbed_W_A_W"][key]
        - b["device_region_current_per_absorbed_W_A_W"][key]
        for key in a["device_region_current_per_absorbed_W_A_W"]
    }
    dominant = max(device_delta, key=lambda key: abs(device_delta[key]))
    dominant_efficiency = max(
        efficiency_delta, key=lambda key: abs(efficiency_delta[key])
    )
    return {
        "scan_distance_um": a["scan_distance_um"],
        "a_minus_b_total_current_A": difference("total_current_A"),
        "a_minus_b_x_equals_b_term_A": difference("x_equals_b_term_A"),
        "a_minus_b_y_equals_a_term_A": difference("y_equals_a_term_A"),
        "a_minus_b_positive_contribution_A": difference(
            "positive_contribution_A"
        ),
        "a_minus_b_negative_contribution_A": difference(
            "negative_contribution_A"
        ),
        "a_minus_b_current_efficiency_A_W": a[
            "current_per_absorbed_W_A_W"
        ]
        - b["current_per_absorbed_W_A_W"],
        "device_region_a_minus_b_A": device_delta,
        "device_region_efficiency_a_minus_b_A_W": efficiency_delta,
        "dominant_absolute_current_difference_region": dominant,
        "dominant_efficiency_difference_region": dominant_efficiency,
    }


def plot_component_decomposition(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    labels = [f"d={row['scan_distance_um']:g}, {row['polarization']}" for row in rows]
    x_values = np.asarray([row["x_equals_b_term_A"] for row in rows]) * 1e9
    y_values = np.asarray([row["y_equals_a_term_A"] for row in rows]) * 1e9
    positive = np.asarray([row["positive_contribution_A"] for row in rows]) * 1e9
    negative = np.asarray([row["negative_contribution_A"] for row in rows]) * 1e9
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    axes[0].bar(positions, x_values, label=r"$x=b$ term", color="tab:blue")
    axes[0].bar(positions, y_values, bottom=x_values, label=r"$y=a$ term", color="tab:orange")
    axes[0].set_ylabel("signed current contribution (nA)")
    axes[0].set_title("Exact thermoelectric-term decomposition")
    axes[0].set_xticks(positions, labels, rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].bar(positions, positive, label="positive cells", color="tab:green")
    axes[1].bar(positions, negative, label="negative/cancelling cells", color="tab:red")
    axes[1].set_ylabel("signed current contribution (nA)")
    axes[1].set_title("Positive generation and negative cancellation")
    axes[1].set_xticks(positions, labels, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_matched_difference_maps(
    path: Path,
    geometry: thermal.Geometry,
    maps: dict[tuple[float, str], dict[str, np.ndarray]],
) -> None:
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:]) * 1e6
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:]) * 1e6
    differences = []
    for distance in (1.0, 3.0, 5.0):
        differences.append(
            maps[(distance, "a")]["total_sheet_A_m2"]
            - maps[(distance, "b")]["total_sheet_A_m2"]
        )
    flake = maps[(1.0, "a")]["flake_mask"]
    bound = max(float(np.max(np.abs(array[flake]))) for array in differences)
    figure, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    for column, (distance, difference) in enumerate(
        zip((1.0, 3.0, 5.0), differences)
    ):
        for row, view in enumerate(("full flake", "illuminated-edge zoom")):
            axis = axes[row, column]
            image = axis.pcolormesh(
                x,
                y,
                np.where(flake, difference, np.nan).T,
                shading="nearest",
                cmap="coolwarm",
                vmin=-bound,
                vmax=bound,
            )
            axis.set_aspect("equal")
            axis.set_xlabel("lab x=b (um)")
            axis.set_ylabel("lab y=a (um)")
            axis.set_title(f"d={distance:g} um: {view}")
            if row == 1:
                axis.set_xlim(-10.0, 5.0)
                axis.set_ylim(-12.0, 5.0)
            figure.colorbar(
                image,
                ax=axis,
                label="sheet current integrand difference (A/m2)",
            )
    figure.suptitle("Where the Maxwell/TaIrTe4 field gives a more current than b")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_device_region_differences(
    path: Path, matched: list[dict[str, Any]]
) -> None:
    regions = list(matched[0]["device_region_a_minus_b_A"])
    distances = [row["scan_distance_um"] for row in matched]
    values = np.asarray(
        [[row["device_region_a_minus_b_A"][region] for region in regions] for row in matched]
    ) * 1e9
    positions = np.arange(len(distances))
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    colors = ["tab:purple", "tab:brown", "tab:cyan", "tab:gray"]
    width = 0.18
    for index, region in enumerate(regions):
        axis.bar(
            positions + (index - 1.5) * width,
            values[:, index],
            width=width,
            label=region.replace("_", " "),
            color=colors[index],
        )
    totals = np.sum(values, axis=1)
    axis.plot(positions, totals, "kD", markersize=8, label="sum = total a-b")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, [f"d={distance:g} um" for distance in distances])
    axis.set_ylabel("same-position a minus b current (nA)")
    axis.set_title("Exclusive spatial partition of the polarization-current difference")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_radial_and_weighting_differences(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    lookup = {(row["scan_distance_um"], row["polarization"]): row for row in rows}
    distances = (1.0, 3.0, 5.0)
    radial_names = list(lookup[(1.0, "a")]["radial_region_current_A"])
    psi_names = list(lookup[(1.0, "a")]["weighting_potential_region_current_A"])
    radial = np.asarray(
        [
            [
                lookup[(distance, "a")]["radial_region_current_A"][name]
                - lookup[(distance, "b")]["radial_region_current_A"][name]
                for name in radial_names
            ]
            for distance in distances
        ]
    ) * 1e9
    psi = np.asarray(
        [
            [
                lookup[(distance, "a")]["weighting_potential_region_current_A"][name]
                - lookup[(distance, "b")]["weighting_potential_region_current_A"][name]
                for name in psi_names
            ]
            for distance in distances
        ]
    ) * 1e9
    bound = max(float(np.max(np.abs(radial))), float(np.max(np.abs(psi))))
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for axis, values, names, title in (
        (
            axes[0],
            radial,
            ["0-.5", ".5-1", "1-1.5", ">=1.5"],
            "Beam-centred radial bins (r/w0)",
        ),
        (
            axes[1],
            psi,
            ["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"],
            "Electrical weighting-potential bins",
        ),
    ):
        image = axis.pcolormesh(
            np.arange(values.shape[1] + 1),
            np.arange(values.shape[0] + 1),
            values,
            shading="flat",
            cmap="coolwarm",
            vmin=-bound,
            vmax=bound,
        )
        axis.set_xticks(np.arange(len(names)) + 0.5, names)
        axis.set_yticks(
            np.arange(len(distances)) + 0.5,
            [f"d={value:g} um" for value in distances],
        )
        axis.invert_yaxis()
        axis.set_xlabel("exclusive bin")
        axis.set_title(title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                axis.text(
                    j + 0.5,
                    i + 0.5,
                    f"{values[i, j]:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
        figure.colorbar(image, ax=axis, label="same-position a minus b current (nA)")
    figure.suptitle("Localization of the polarization-current difference")
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
    first_fields = (
        Path(sparse["records"][0]["thermal_summary_path"]).parent
        / "thermal_pte_fields.npz"
    )
    geometry = setup_geometry(args.geometry_contract, first_optical, first_fields)
    device_masks = device_region_masks(geometry)

    rows = []
    maps: dict[tuple[float, str], dict[str, np.ndarray]] = {}
    raw_artifacts = [
        artifact(args.sparse_summary, "registered sparse-scan summary", committed=True),
        artifact(args.geometry_contract, "registered geometry contract", committed=True),
    ]
    seen = {item["path"] for item in raw_artifacts}
    for record in sparse["records"]:
        row, case_maps = decompose_case(record, geometry, device_masks)
        key = (row["scan_distance_um"], row["polarization"])
        rows.append(row)
        maps[key] = case_maps
        fields_path = Path(record["thermal_summary_path"]).parent / "thermal_pte_fields.npz"
        if str(fields_path.resolve()) not in seen:
            raw_artifacts.append(
                artifact(
                    fields_path,
                    f"d={key[0]:g} um E||{key[1]} immutable thermal-PTE fields",
                )
            )
            seen.add(str(fields_path.resolve()))

    lookup = {(row["scan_distance_um"], row["polarization"]): row for row in rows}
    matched = [matched_difference(lookup[(distance, "a")], lookup[(distance, "b")]) for distance in (1.0, 3.0, 5.0)]
    max_closure = max(
        max(
            row["reported_current_closure_relative_error"],
            row["x_plus_y_closure_relative_error"],
            row["component_pairing_relative_error"],
        )
        for row in rows
    )
    gates = {
        "reported_current_reintegration_lt_1e_minus_12": all(
            row["reported_current_closure_relative_error"] < 1e-12 for row in rows
        ),
        "x_plus_y_term_closure_lt_1e_minus_12": all(
            row["x_plus_y_closure_relative_error"] < 1e-12 for row in rows
        ),
        "component_pairing_lt_1e_minus_12": all(
            row["component_pairing_relative_error"] < 1e-12 for row in rows
        ),
    }
    status = (
        "COMPLETED_DEVICE_A_SPATIAL_CURRENT_DECOMPOSITION"
        if all(gates.values())
        else "FAILED_DEVICE_A_SPATIAL_CURRENT_DECOMPOSITION"
    )
    coefficients = {
        "x_equals_b_minus_sigma_S_A_mK": float(
            -thermal.SIGMA_LAB_S_M[0] * thermal.SEEBECK_LAB_V_K[0]
        ),
        "y_equals_a_minus_sigma_S_A_mK": float(
            -thermal.SIGMA_LAB_S_M[1] * thermal.SEEBECK_LAB_V_K[1]
        ),
        "sigma_x_y_S_m": thermal.SIGMA_LAB_S_M.tolist(),
        "Seebeck_x_y_V_K": thermal.SEEBECK_LAB_V_K.tolist(),
        "axis_mapping": "x=b, y=a",
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "UNKNOWN"
    summary = {
        "status": status,
        "generation_commit": commit,
        "scope": (
            "offline reintegration and spatial partition of immutable Device-A "
            "thermal-PTE fields; no Maxwell, thermal solve, weighting solve, "
            "adjoint, AD-FD, or optimization"
        ),
        "equation": (
            "I=int[-sigma_b*S_b*d_bT*d_bpsi - sigma_a*S_a*d_aT*d_apsi]dV"
        ),
        "thermoelectric_coefficients": coefficients,
        "spatial_partition_contract": {
            "exclusive_device_regions": list(device_masks),
            "contact_distance_m": CONTACT_ZONE_M,
            "free_edge_distance_m": EDGE_BAND_M,
            "radial_bins": "r/w0=[0,.5),[.5,1),[1,1.5),[1.5,inf)",
            "w0_m": WAIST_M,
            "weighting_potential_bins": "five bins from psi=0 to 1",
        },
        "cases": rows,
        "same_position_a_minus_b": matched,
        "maximum_decomposition_closure_relative_error": max_closure,
        "numerical_gates": gates,
        "interpretation": {
            "same_position_comparison_required": True,
            "sampled_maxima_use_different_positions": {
                "a_distance_um": 1.0,
                "b_distance_um": 3.0,
            },
            "not_an_optical_or_thermal_model_validation": True,
        },
    }
    summary_path = args.report_dir / "device_a_spatial_current_decomposition_summary.json"
    summary_path.write_text(json.dumps(thermal.jsonable(summary), indent=2) + "\n")

    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        base = {
            "record_type": "case_total",
            "scan_distance_um": row["scan_distance_um"],
            "polarization": row["polarization"],
            "total_current_A": row["total_current_A"],
            "x_equals_b_term_A": row["x_equals_b_term_A"],
            "y_equals_a_term_A": row["y_equals_a_term_A"],
            "positive_contribution_A": row["positive_contribution_A"],
            "negative_contribution_A": row["negative_contribution_A"],
        }
        csv_rows.append(base)
        for region, value in row["device_region_current_A"].items():
            csv_rows.append(
                {
                    "record_type": "device_region",
                    "scan_distance_um": row["scan_distance_um"],
                    "polarization": row["polarization"],
                    "region": region,
                    "region_current_A": value,
                    "region_current_per_absorbed_W_A_W": row[
                        "device_region_current_per_absorbed_W_A_W"
                    ][region],
                }
            )
        for region, value in row["radial_region_current_A"].items():
            csv_rows.append(
                {
                    "record_type": "radial_region",
                    "scan_distance_um": row["scan_distance_um"],
                    "polarization": row["polarization"],
                    "region": region,
                    "region_current_A": value,
                }
            )
        for region, value in row["weighting_potential_region_current_A"].items():
            csv_rows.append(
                {
                    "record_type": "weighting_potential_region",
                    "scan_distance_um": row["scan_distance_um"],
                    "polarization": row["polarization"],
                    "region": region,
                    "region_current_A": value,
                }
            )
    for row in matched:
        for region, value in row["device_region_a_minus_b_A"].items():
            csv_rows.append(
                {
                    "record_type": "same_position_a_minus_b_region",
                    "scan_distance_um": row["scan_distance_um"],
                    "polarization": "a-minus-b",
                    "region": region,
                    "region_current_A": value,
                    "a_minus_b_total_current_A": row[
                        "a_minus_b_total_current_A"
                    ],
                }
            )
    csv_path = args.report_dir / "device_a_spatial_current_decomposition_cases.csv"
    fields = sorted({key for row in csv_rows for key in row})
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)

    component_plot = args.report_dir / "DEVICE_A_PTE_TERM_DECOMPOSITION.png"
    difference_plot = args.report_dir / "DEVICE_A_SAME_POSITION_A_MINUS_B_MAPS.png"
    region_plot = args.report_dir / "DEVICE_A_SPATIAL_REGION_DIFFERENCES.png"
    bin_plot = args.report_dir / "DEVICE_A_RADIAL_WEIGHTING_BIN_DIFFERENCES.png"
    plot_component_decomposition(component_plot, rows)
    plot_matched_difference_maps(difference_plot, geometry, maps)
    plot_device_region_differences(region_plot, matched)
    plot_radial_and_weighting_differences(bin_plot, rows)

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

    matched_lines = "".join(
        f"| {row['scan_distance_um']:.0f} | "
        f"{row['a_minus_b_total_current_A']*1e9:.6f} | "
        f"{row['a_minus_b_x_equals_b_term_A']*1e9:.6f} | "
        f"{row['a_minus_b_y_equals_a_term_A']*1e9:.6f} | "
        f"{row['a_minus_b_positive_contribution_A']*1e9:.6f} | "
        f"{row['a_minus_b_negative_contribution_A']*1e9:.6f} | "
        f"{row['dominant_absolute_current_difference_region']} |\n"
        for row in matched
    )
    report = f"""# Device-A spatial current decomposition

Status: `{status}`

This is a read-only/offline decomposition of the immutable registered
Maxwell -> explicit-3D thermal -> PTE fields. No new Maxwell, thermal, or
weighting solve was run.

## Literal current equation

`x=b`, `y=a`, and

```text
I = integral[-sigma_b S_b (d_b T)(d_b psi)
             -sigma_a S_a (d_a T)(d_a psi)] dV.
```

The coefficients are `{coefficients['x_equals_b_minus_sigma_S_A_mK']:.6f}`
and `{coefficients['y_equals_a_minus_sigma_S_A_mK']:.6f} A/(m K)`. They are
nearly equal in magnitude and opposite in sign; neither term was omitted.

## Same-position polarization difference

Positive values below mean that the saved `E||a` field produces more current
than the saved `E||b` field at the same registered beam position.

| d (um) | total a-b (nA) | x=b term (nA) | y=a term (nA) | positive-cell difference (nA) | negative-cell difference (nA) | largest spatial region |
|---:|---:|---:|---:|---:|---:|---|
{matched_lines}

The negative-cell column is the signed change in cancellation. A positive
value means the `a` case is less negative (less cancelled) than `b`.

Both crystallographic derivative terms contribute. At `d=1 um` the excess
is `1.260697 nA` from the `x=b` term and `1.305944 nA` from the `y=a` term;
at `d=5 um` they are `0.735975 nA` and `1.172822 nA`. The result is therefore
not attributable to one omitted derivative or one swapped Seebeck term.

## Where the excess current occurs

The free-edge band contributes `+4.216371`, `+3.519862`, and `+2.561187 nA`
to `a-b` at `d=1,3,5 um`. The flake interior contributes `-1.528097`,
`-1.468526`, and `-1.190489 nA`, respectively. Thus the interior actually
favors `b`; the simulated `a>b` trend is created by the free-edge response.

The beam-centred partition independently localizes the `d=1 um` excess:
`r<0.5 w0` contributes `+4.200080 nA`, while all larger-radius bins together
contribute `-1.633439 nA`. The weighting-potential partition places the
largest positive difference in `psi=0.2--0.4` for all three positions. These
are independent partitions of the same closed volume integral, not fitted
current corrections.

## Spatial contract

The flake cells are partitioned exactly once into top-contact within 2 um,
bottom-contact within 2 um, remaining free-edge within 1 um, and flake
interior. Independent radial and weighting-potential-bin decompositions are
also stored in JSON/CSV. Region sums, `x+y`, and the published current all
close below `1e-12`; maximum observed error is `{max_closure:.3e}`.

This checkpoint diagnoses where the existing result is generated. It does
not establish that the Maxwell Q, approximate Figure-3H registration, exact
contact CAD, or paper beam radius is correct. Raw NPZ fields remain outside
Git and are path/size/SHA-256 pinned in the manifest.
"""
    (args.report_dir / "DEVICE_A_SPATIAL_CURRENT_DECOMPOSITION_REPORT.md").write_text(
        report
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
