#!/usr/bin/env python3
"""Plot the frozen nine-position Device-A illumination plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np


WAIST_UM = 8.75
FIXED_LUMERICAL_SHIFT_UM = np.asarray([0.0, -3.0])
OUTSIDE_X_UM = -16.5625
INSIDE_X_UM = -6.0
Y_LEVELS_UM = (3.0, 0.0, -3.0)
LEVEL_LABELS = {3.0: "TOP", 0.0: "MIDDLE", -3.0: "BOTTOM"}
LEVEL_COLORS = {3.0: "#2ca02c", 0.0: "#ff7f0e", -3.0: "#1f77b4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def boundary_intersection_x(
    vertices: np.ndarray,
    y_um: float,
    *,
    x_low_um: float,
    x_high_um: float,
) -> tuple[float, int]:
    intersections: list[tuple[float, int]] = []
    for index, (start, stop) in enumerate(
        zip(vertices, np.roll(vertices, -1, axis=0))
    ):
        if np.isclose(start[1], stop[1]):
            continue
        fraction = (y_um - start[1]) / (stop[1] - start[1])
        if 0.0 <= fraction <= 1.0:
            x_value = float(start[0] + fraction * (stop[0] - start[0]))
            if x_low_um <= x_value <= x_high_um:
                intersections.append((x_value, index))
    if not intersections:
        raise RuntimeError(f"y={y_um:g} um misses the selected flake boundary")
    return min(intersections, key=lambda item: item[0])


def draw_device(
    axis: plt.Axes,
    flake: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
) -> None:
    axis.add_patch(
        Polygon(
            flake,
            closed=True,
            facecolor="#8b5fbf",
            edgecolor="#542788",
            linewidth=1.8,
            alpha=0.55,
        )
    )
    for metal in (top, bottom):
        axis.add_patch(
            Polygon(
                metal,
                closed=True,
                facecolor="#d4af37",
                edgecolor="#8c6d00",
                alpha=0.65,
            )
        )
    axis.axhline(0.0, color="0.86", linewidth=0.7, zorder=0)
    axis.axvline(0.0, color="0.86", linewidth=0.7, zorder=0)
    axis.set_aspect("equal")
    axis.set_xlabel("fixed Lumerical x=b (µm)")
    axis.set_ylabel("fixed Lumerical y=a (µm)")


def build_position_contract(payload: dict[str, object]) -> dict[str, object]:
    """Return the frozen nine source centres in the canonical solver frame."""
    flake = (
        np.asarray(payload["flake_vertices_code_um"], float)
        + FIXED_LUMERICAL_SHIFT_UM
    )
    centers: dict[str, dict[float, np.ndarray]] = {
        "OUTSIDE": {
            y: np.asarray([OUTSIDE_X_UM, y]) for y in Y_LEVELS_UM
        },
        "EDGE": {},
        "INSIDE": {y: np.asarray([INSIDE_X_UM, y]) for y in Y_LEVELS_UM},
    }
    edge_segments: dict[float, int] = {}
    for y in Y_LEVELS_UM:
        x_edge, segment = boundary_intersection_x(
            flake,
            y,
            x_low_um=OUTSIDE_X_UM,
            x_high_um=INSIDE_X_UM,
        )
        centers["EDGE"][y] = np.asarray([x_edge, y])
        edge_segments[y] = segment

    baseline = centers["OUTSIDE"][0.0]
    cases = []
    for category in ("OUTSIDE", "EDGE", "INSIDE"):
        for y in Y_LEVELS_UM:
            center = centers[category][y]
            cases.append(
                {
                    "label": (
                        f"{category.lower()}_"
                        f"{LEVEL_LABELS[y].lower()}"
                    ),
                    "category": category.lower(),
                    "vertical_level": LEVEL_LABELS[y].lower(),
                    "beam_center_lumerical_um": center.tolist(),
                    "beam_offset_from_fixed_baseline_um": (
                        center - baseline
                    ).tolist(),
                    "edge_segment_index": (
                        edge_segments[y] if category == "EDGE" else None
                    ),
                }
            )
    return {
        "status": "FROZEN_DEVICE_A_NINE_POSITION_CONTRACT",
        "coordinate_frame": {
            "name": "DEVICE_A_FIXED_LUMERICAL_X_B_Y_A",
            "axis_mapping": "x=b, y=a",
            "digitized_to_lumerical_shift_um": (
                FIXED_LUMERICAL_SHIFT_UM.tolist()
            ),
            "fixed_baseline_source_center_um": baseline.tolist(),
            "device_and_PML_are_case_invariant": True,
            "only_source_is_translated": True,
        },
        "waist_um": WAIST_UM,
        "cases": cases,
        "thermal_interface_scenarios_W_m2K": {
            "thermally_grown_SiO2": 7.37e6,
            "evaporated_SiO2": 7.37e4,
        },
    }


def main() -> int:
    args = parse_args()
    payload = json.loads(args.geometry_contract.read_text())
    contract = build_position_contract(payload)
    flake = (
        np.asarray(payload["flake_vertices_code_um"], float)
        + FIXED_LUMERICAL_SHIFT_UM
    )
    top = (
        np.asarray(payload["top_metal_polygon_code_um"], float)
        + FIXED_LUMERICAL_SHIFT_UM
    )
    bottom = (
        np.asarray(payload["bottom_metal_polygon_code_um"], float)
        + FIXED_LUMERICAL_SHIFT_UM
    )

    centers: dict[str, dict[float, np.ndarray]] = {
        "OUTSIDE": {
            y: np.asarray([OUTSIDE_X_UM, y]) for y in Y_LEVELS_UM
        },
        "EDGE": {},
        "INSIDE": {y: np.asarray([INSIDE_X_UM, y]) for y in Y_LEVELS_UM},
    }
    edge_segments: set[int] = set()
    for y in Y_LEVELS_UM:
        x_edge, segment = boundary_intersection_x(
            flake,
            y,
            x_low_um=OUTSIDE_X_UM,
            x_high_um=INSIDE_X_UM,
        )
        centers["EDGE"][y] = np.asarray([x_edge, y])
        edge_segments.add(segment)

    figure, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    overview = axes[0, 0]
    draw_device(overview, flake, top, bottom)
    markers = {"OUTSIDE": "o", "EDGE": "s", "INSIDE": "^"}
    for category, category_centers in centers.items():
        for y, center in category_centers.items():
            overview.scatter(
                center[0],
                center[1],
                marker=markers[category],
                s=110,
                facecolor=LEVEL_COLORS[y],
                edgecolor="black",
                linewidth=1.1,
                zorder=8,
            )
            overview.text(
                center[0] + 0.35,
                center[1] + 0.3,
                f"{category[0]}-{LEVEL_LABELS[y][0]}",
                fontsize=9,
                fontweight="bold",
            )
    for segment in edge_segments:
        points = flake[[segment, (segment + 1) % flake.shape[0]]]
        overview.plot(points[:, 0], points[:, 1], color="#ff7f0e", linewidth=4)
    overview.set(
        xlim=(-20, 8),
        ylim=(-10, 10),
        title="All 9 frozen beam centres (markers only)",
    )
    overview.text(
        0.02,
        0.02,
        "O=outside, E=edge, I=inside; T/M/B=top/middle/bottom",
        transform=overview.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )

    panel_settings = {
        "OUTSIDE": (axes[0, 1], (-28, -5)),
        "EDGE": (axes[1, 0], (-24, 0)),
        "INSIDE": (axes[1, 1], (-18, 7)),
    }
    for category, (axis, x_limits) in panel_settings.items():
        draw_device(axis, flake, top, bottom)
        lines = []
        for y in Y_LEVELS_UM:
            center = centers[category][y]
            color = LEVEL_COLORS[y]
            axis.add_patch(
                Circle(
                    center,
                    WAIST_UM,
                    fill=False,
                    edgecolor=color,
                    linewidth=2,
                    linestyle="--",
                    alpha=0.9,
                )
            )
            axis.scatter(
                center[0],
                center[1],
                marker="x",
                s=100,
                linewidth=3,
                color=color,
                zorder=9,
            )
            lines.append(
                f"{LEVEL_LABELS[y]:6s}: ({center[0]:.4f}, {center[1]:+.1f}) µm"
            )
        if category == "EDGE":
            for segment in edge_segments:
                points = flake[[segment, (segment + 1) % flake.shape[0]]]
                axis.plot(
                    points[:, 0], points[:, 1], color="#ff7f0e", linewidth=4
                )
        axis.set(
            xlim=x_limits,
            ylim=(-13, 13),
            title=(
                f"{category}: three centres\n"
                r"dashed circles = Gaussian $w_0=8.75$ µm"
            ),
        )
        axis.text(
            0.03,
            0.03,
            "\n".join(lines),
            transform=axis.transAxes,
            family="monospace",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.7"},
        )

    figure.suptitle(
        "Device-A final 9-position illumination plan — fixed Lumerical frame\n"
        "Each position will use E||a and E||b, then thermally-grown "
        "G=7.37e6 and evaporated G=7.37e4 W/(m²K)",
        fontsize=16,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)

    result = dict(contract)
    result["solve_started"] = False
    result["plot"] = str(args.output.resolve())
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")
        result["json_output"] = str(args.json_output.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
