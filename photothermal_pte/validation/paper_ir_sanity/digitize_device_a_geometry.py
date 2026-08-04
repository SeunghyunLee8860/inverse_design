#!/usr/bin/env python3
"""Freeze a transparent Figure-2/3 Device-A geometry approximation.

The coordinates below are manual picks on 600-dpi renders of pages 5 and 6
of the paper PDF.  They are not unpublished CAD.  The script records the
pixel calibration, the coordinate transform, and conservative uncertainty
labels, then writes overlays and a machine-readable geometry contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PDF_DEFAULT = Path(
    "/home/seunghyun/tairte4/papers/"
    "Adv Funct Materials - 2026 - Blevins - Large Transverse "
    "Thermoelectric Effect in Weyl Semimetal TaIrTe4 Engineered for-2.pdf"
)

# Pixel coordinates in the committed analysis are relative to the crops
# produced by the generation command stored in the output JSON.
FIG2D_SCALE_BAR_PX = np.asarray([[419.0, 758.0], [600.0, 758.0]])
FIG2D_SCALE_BAR_UM = 10.0

# Clockwise visible flake outline.  Portions beneath Au/Ti are inferred from
# the dashed outline/reflection contrast and receive the larger uncertainty.
FIG2D_FLAKE_PX = np.asarray(
    [
        [247.0, 386.0],
        [685.0, 386.0],
        [684.0, 812.0],
        [141.0, 810.0],
        [141.0, 628.0],
        [169.0, 584.0],
        [169.0, 526.0],
        [247.0, 484.0],
    ]
)

# Optical metal rectangles.  The electrically active overlap is represented
# separately by the top/bottom boundary segments of the flake polygon.
FIG2D_TOP_METAL_PX = np.asarray([[90.0, 274.0], [779.0, 386.0]])
FIG2D_BOTTOM_METAL_PX = np.asarray([[90.0, 810.0], [779.0, 875.0]])
FIG2D_TOP_CONTACT_PX = np.asarray([[247.0, 386.0], [685.0, 386.0]])
FIG2D_BOTTOM_CONTACT_PX = np.asarray([[141.0, 810.0], [684.0, 812.0]])

# The full digitized off-axis side (including its small staircase vertices)
# is used to define one reproducible mean tangent/normal.  It is closer to
# the paper's nominal 45-degree description than any one blurred subsegment.
OFF_AXIS_VERTEX_PAIR = (4, 7)
EDGE_PEAK_INWARD_OFFSET_UM = 3.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig2d-crop", type=Path, required=True)
    parser.add_argument("--fig3h-crop", type=Path, required=True)
    parser.add_argument("--fig3i-crop", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--paper-pdf", type=Path, default=PDF_DEFAULT)
    return parser.parse_args()


def pixel_to_code_um(points: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    px_per_um = float(
        np.linalg.norm(FIG2D_SCALE_BAR_PX[1] - FIG2D_SCALE_BAR_PX[0])
        / FIG2D_SCALE_BAR_UM
    )
    bbox_min = np.min(FIG2D_FLAKE_PX, axis=0)
    bbox_max = np.max(FIG2D_FLAKE_PX, axis=0)
    origin = 0.5 * (bbox_min + bbox_max)
    transformed = np.column_stack(
        (
            (points[:, 0] - origin[0]) / px_per_um,
            (origin[1] - points[:, 1]) / px_per_um,
        )
    )
    return transformed, {
        "pixels_per_um": px_per_um,
        "origin_pixel_x": float(origin[0]),
        "origin_pixel_y": float(origin[1]),
        "x_code_direction": "image right = crystal b",
        "y_code_direction": "image up = crystal a",
    }


def rectangle_to_code_um(rectangle: np.ndarray) -> list[list[float]]:
    x0, y0 = rectangle[0]
    x1, y1 = rectangle[1]
    corners = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    transformed, _ = pixel_to_code_um(corners)
    return transformed.tolist()


def plot_fig2_overlay(image: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 8.4), constrained_layout=True)
    ax.imshow(image)
    closed = np.vstack((FIG2D_FLAKE_PX, FIG2D_FLAKE_PX[0]))
    ax.plot(closed[:, 0], closed[:, 1], "w-", lw=2.2, label="digitized flake")
    for rectangle, color, label in (
        (FIG2D_TOP_METAL_PX, "tab:orange", "Au/Ti footprint"),
        (FIG2D_BOTTOM_METAL_PX, "tab:orange", None),
    ):
        x0, y0 = rectangle[0]
        x1, y1 = rectangle[1]
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False,
                edgecolor=color, linewidth=2.0, label=label,
            )
        )
    ax.plot(*FIG2D_SCALE_BAR_PX.T, color="black", lw=3, label="10 µm scale")
    i0, i1 = OFF_AXIS_VERTEX_PAIR
    edge = FIG2D_FLAKE_PX[[i0, i1]]
    ax.plot(edge[:, 0], edge[:, 1], color="cyan", lw=4, label="target edge")
    ax.set_xlim(60, 810)
    ax.set_ylim(920, 240)
    ax.set_xlabel("600-dpi crop x (pixel)")
    ax.set_ylabel("600-dpi crop y (pixel)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Device A Figure 2D digitization overlay\nFIGURE_DIGITIZED_APPROXIMATION")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_geometry_contract(
    flake_um: np.ndarray,
    beam_um: np.ndarray,
    inward: np.ndarray,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    closed = np.vstack((flake_um, flake_um[0]))
    ax.fill(closed[:, 0], closed[:, 1], color="#d9c27e", alpha=0.6)
    ax.plot(closed[:, 0], closed[:, 1], "k-", lw=2, label="TaIrTe4")
    for rectangle, color in (
        (FIG2D_TOP_METAL_PX, "goldenrod"),
        (FIG2D_BOTTOM_METAL_PX, "goldenrod"),
    ):
        polygon = np.asarray(rectangle_to_code_um(rectangle))
        polygon = np.vstack((polygon, polygon[0]))
        ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.55)
    i0, i1 = OFF_AXIS_VERTEX_PAIR
    edge = flake_um[[i0, i1]]
    ax.plot(edge[:, 0], edge[:, 1], color="tab:blue", lw=4, label="off-axis edge")
    ax.scatter(*beam_um, s=85, color="red", label="pre-registered peak position")
    ax.arrow(
        beam_um[0], beam_um[1], inward[0] * 2.0, inward[1] * 2.0,
        width=0.08, color="red", length_includes_head=True,
    )
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_{code}=b$ (µm)")
    ax.set_ylabel(r"$y_{code}=a$ (µm)")
    ax.set_title("Digitized Device A contract (not unpublished CAD)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    flake_um, transform = pixel_to_code_um(FIG2D_FLAKE_PX)
    top_contact_um, _ = pixel_to_code_um(FIG2D_TOP_CONTACT_PX)
    bottom_contact_um, _ = pixel_to_code_um(FIG2D_BOTTOM_CONTACT_PX)

    i0, i1 = OFF_AXIS_VERTEX_PAIR
    edge = flake_um[[i0, i1]]
    tangent = edge[1] - edge[0]
    tangent /= np.linalg.norm(tangent)
    candidate = np.asarray([tangent[1], -tangent[0]])
    polygon_centroid = np.mean(flake_um, axis=0)
    midpoint = np.mean(edge, axis=0)
    inward = candidate if np.dot(candidate, polygon_centroid - midpoint) > 0 else -candidate
    beam_um = midpoint + EDGE_PEAK_INWARD_OFFSET_UM * inward

    fig2 = plt.imread(args.fig2d_crop)
    plot_fig2_overlay(fig2, args.report_dir / "DEVICE_A_FIG2D_DIGITIZATION_OVERLAY.png")
    plot_geometry_contract(
        flake_um,
        beam_um,
        inward,
        args.report_dir / "DEVICE_A_GEOMETRY_CONTRACT.png",
    )

    summary = {
        "status": "DEVICE_A_FIGURE_DIGITIZATION_AUDITED",
        "geometry_status": "FIGURE_DIGITIZED_APPROXIMATION",
        "paper": {
            "path": str(args.paper_pdf),
            "size_bytes": args.paper_pdf.stat().st_size,
            "sha256": sha256(args.paper_pdf),
            "render_dpi": 600,
            "figure_2_page_one_indexed": 5,
            "figure_3_page_one_indexed": 6,
        },
        "coordinate_transform": transform,
        "flake_vertices_pixel": FIG2D_FLAKE_PX.tolist(),
        "flake_vertices_code_um": flake_um.tolist(),
        "top_metal_polygon_code_um": rectangle_to_code_um(FIG2D_TOP_METAL_PX),
        "bottom_metal_polygon_code_um": rectangle_to_code_um(FIG2D_BOTTOM_METAL_PX),
        "top_electrical_contact_segment_code_um": top_contact_um.tolist(),
        "bottom_electrical_contact_segment_code_um": bottom_contact_um.tolist(),
        "off_axis_edge_vertex_indices": list(OFF_AXIS_VERTEX_PAIR),
        "off_axis_edge_midpoint_code_um": midpoint.tolist(),
        "off_axis_edge_unit_tangent_code": tangent.tolist(),
        "off_axis_edge_unit_inward_normal_code": inward.tolist(),
        "pre_registered_beam_center_code_um": beam_um.tolist(),
        "beam_center_rule": (
            "target off-axis segment midpoint plus 3 um inward, frozen before "
            "any simulated current is inspected"
        ),
        "figure_3I_peak_distance_digitization_um": 3.0,
        "measured_ratio": {
            "abs_Ia_over_abs_Ib": 0.8365896980461811,
            "digitization_uncertainty": 0.00852575488454707,
        },
        "uncertainty_um": {
            "visible_flake_edge": 0.4,
            "flake_under_metal": 0.8,
            "contact_overlap_endpoint": 0.6,
            "beam_center_registration": 1.0,
            "scale_bar": 0.15,
        },
        "limitations": [
            "not unpublished CAD",
            "contact overlap beneath metal is inferred",
            "Figure 3I supplies a relative scan coordinate, not absolute stage metrology",
            "w0=12 um remains an explicit assumption rather than a paper-published value",
        ],
        "input_crops": {
            "figure_2D": str(args.fig2d_crop),
            "figure_3H": str(args.fig3h_crop),
            "figure_3I": str(args.fig3i_crop),
        },
        "generation_command": (
            "pdftoppm -f 5 -l 6 -r 600 -png PAPER page; "
            "Figure 2D crop=1050x1150+3550+150; "
            "Figure 3H crop=1900x1900+450+2300; "
            "Figure 3I crop=1200x1700+2250+2300"
        ),
    }
    with (args.report_dir / "device_a_geometry_digitization.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    with (args.report_dir / "device_a_geometry_vertices.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["vertex", "pixel_x", "pixel_y", "x_code_b_um", "y_code_a_um"])
        for index, (pixel, physical) in enumerate(zip(FIG2D_FLAKE_PX, flake_um)):
            writer.writerow([index, *pixel.tolist(), *physical.tolist()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
