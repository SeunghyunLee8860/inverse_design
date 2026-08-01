#!/usr/bin/env python3
"""Render the frozen straight-45-degree FDTD geometry from case metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle


COLORS = {
    "air": "#edf6ff",
    "flake": "#e69532",
    "sio2": "#8fd3df",
    "si": "#667c99",
    "source": "#159447",
    "mesh": "#7b4ab5",
    "pabs": "#1f77b4",
    "outer": "#555555",
    "pml": "#b34fc8",
}


def add_lateral_pml(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    """Draw a schematic PML band; its plotted thickness is not a readback."""
    width = 0.8
    common = dict(
        facecolor=COLORS["pml"], edgecolor=COLORS["pml"], alpha=0.18,
        hatch="////", linewidth=0.8, zorder=10,
    )
    ax.add_patch(Rectangle((xlim[0], ylim[0]), width, ylim[1] - ylim[0], **common))
    ax.add_patch(Rectangle((xlim[1] - width, ylim[0]), width, ylim[1] - ylim[0], **common))
    ax.add_patch(Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], width, **common))
    ax.add_patch(Rectangle((xlim[0], ylim[1] - width), xlim[1] - xlim[0], width, **common))


def add_xz_pml(ax: plt.Axes, xlim: tuple[float, float], zlim: tuple[float, float]) -> None:
    x_width = 0.8
    z_width = 0.35
    common = dict(
        facecolor=COLORS["pml"], edgecolor=COLORS["pml"], alpha=0.18,
        hatch="////", linewidth=0.8, zorder=10,
    )
    ax.add_patch(Rectangle((xlim[0], zlim[0]), x_width, zlim[1] - zlim[0], **common))
    ax.add_patch(Rectangle((xlim[1] - x_width, zlim[0]), x_width, zlim[1] - zlim[0], **common))
    ax.add_patch(Rectangle((xlim[0], zlim[0]), xlim[1] - xlim[0], z_width, **common))
    ax.add_patch(Rectangle((xlim[0], zlim[1] - z_width), xlim[1] - xlim[0], z_width, **common))


def draw_top_view(ax: plt.Axes, geometry: dict) -> None:
    bounds = geometry["domain_bounds_m"]
    xmin, xmax = [value * 1e6 for value in bounds["x"]]
    ymin, ymax = [value * 1e6 for value in bounds["y"]]
    ax.set_facecolor(COLORS["air"])
    ax.add_patch(
        Polygon(
            [(xmin, ymin), (xmax, ymin), (xmax, ymax)],
            closed=True,
            facecolor=COLORS["flake"], edgecolor="#8c570e", linewidth=1.1,
            label=r"TaIrTe$_4$: $y\leq x$",
        )
    )
    ax.plot([xmin, xmax], [xmin, xmax], color="black", linewidth=1.2, label=r"edge $y=x$")

    source = geometry["source"]
    half_source = 0.5 * source["source_span_m"] * 1e6
    ax.add_patch(
        Rectangle(
            (-half_source, -half_source), 2 * half_source, 2 * half_source,
            fill=False, edgecolor=COLORS["source"], linewidth=1.5,
            label="source aperture projection (50 µm square)",
        )
    )
    waist = source["physical_target_waist_radius_m"] * 1e6
    ax.add_patch(
        Circle(
            (0, 0), waist, fill=False, edgecolor=COLORS["source"],
            linestyle="--", linewidth=1.4, label=r"target-plane $w_0=8.75$ µm",
        )
    )

    mesh = geometry["large_domain_mesh_policy"]["fine_refinement_region_bounds_m"]
    mx0, mx1 = [value * 1e6 for value in mesh["x"]]
    my0, my1 = [value * 1e6 for value in mesh["y"]]
    ax.add_patch(
        Rectangle(
            (mx0, my0), mx1 - mx0, my1 - my0, fill=False,
            edgecolor=COLORS["mesh"], linewidth=1.1, linestyle=":",
            label="100-nm x/y local mesh",
        )
    )
    pabs = geometry["pabs_nominal_control_volume_bounds_m"]
    px0, px1 = [value * 1e6 for value in pabs["x"]]
    py0, py1 = [value * 1e6 for value in pabs["y"]]
    ax.add_patch(
        Rectangle(
            (px0, py0), px1 - px0, py1 - py0, fill=False,
            edgecolor=COLORS["pabs"], linewidth=1.1, linestyle="--",
            label="matched Q / six-face box",
        )
    )
    outer = geometry["outer_flux_box_bounds_m"]
    ox0, ox1 = [value * 1e6 for value in outer["x"]]
    oy0, oy1 = [value * 1e6 for value in outer["y"]]
    ax.add_patch(
        Rectangle(
            (ox0, oy0), ox1 - ox0, oy1 - oy0, fill=False,
            edgecolor=COLORS["outer"], linewidth=0.9, linestyle="-.",
            label="outer diagnostic flux box",
        )
    )
    add_lateral_pml(ax, (xmin, xmax), (ymin, ymax))
    ax.set(
        xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal",
        xlabel="lab x = crystal b (µm)", ylabel="lab y = crystal a (µm)",
        title="A. Top view (xy): straight 45° edge",
    )
    ax.legend(loc="upper left", fontsize=7.2, framealpha=0.94)


def draw_vertical_view(ax: plt.Axes, geometry: dict, axis_name: str) -> None:
    bounds = geometry["domain_bounds_m"]
    lateral = [value * 1e6 for value in bounds[axis_name]]
    zlim = [value * 1e6 for value in bounds["z"]]
    sio2_bottom = -0.415
    flake_bottom = -0.130
    ax.set_facecolor(COLORS["air"])
    ax.add_patch(
        Rectangle(
            (lateral[0], zlim[0]), lateral[1] - lateral[0], sio2_bottom - zlim[0],
            facecolor=COLORS["si"], edgecolor="none", label="Si (3 µm in FDTD)",
        )
    )
    ax.add_patch(
        Rectangle(
            (lateral[0], sio2_bottom), lateral[1] - lateral[0], 0.285,
            facecolor=COLORS["sio2"], edgecolor="#348f9d", linewidth=0.6,
            label="SiO₂ (285 nm)",
        )
    )
    if axis_name == "x":
        support_start, support_end = 0.0, lateral[1]
        cut_label = "y=0: TaIrTe₄ exists for x≥0"
        xlabel = "lab x = crystal b (µm)"
        panel = "B. Front view (xz)"
    else:
        support_start, support_end = lateral[0], 0.0
        cut_label = "x=0: TaIrTe₄ exists for y≤0"
        xlabel = "lab y = crystal a (µm)"
        panel = "C. Side view (yz)"
    ax.add_patch(
        Rectangle(
            (support_start, flake_bottom), support_end - support_start, 0.130,
            facecolor=COLORS["flake"], edgecolor="#8c570e", linewidth=0.7,
            label=cut_label,
        )
    )

    source_z = 5.0
    half_source = 25.0
    focus_z = -0.065
    ax.plot(
        [-half_source, half_source], [source_z, source_z],
        color=COLORS["source"], linewidth=2.0, label="Gaussian source plane z=5 µm",
    )
    ax.annotate(
        "propagation −z", xy=(0, 1.0), xytext=(0, 4.4), ha="center",
        arrowprops=dict(arrowstyle="-|>", color=COLORS["source"], linewidth=1.5),
        color=COLORS["source"], fontsize=9,
    )
    ax.scatter([0], [focus_z], marker="x", s=60, linewidths=1.8,
               color=COLORS["source"], label="waist/focus z=−65 nm")
    ax.plot([-9.0, -8.75], [source_z, focus_z], color=COLORS["source"], alpha=0.55)
    ax.plot([9.0, 8.75], [source_z, focus_z], color=COLORS["source"], alpha=0.55)

    outer = geometry["outer_flux_box_bounds_m"]
    outer_lateral = [value * 1e6 for value in outer[axis_name]]
    outer_z = [value * 1e6 for value in outer["z"]]
    ax.add_patch(
        Rectangle(
            (outer_lateral[0], outer_z[0]),
            outer_lateral[1] - outer_lateral[0], outer_z[1] - outer_z[0],
            fill=False, edgecolor=COLORS["outer"], linestyle="-.", linewidth=0.9,
            label="outer flux box",
        )
    )
    add_xz_pml(ax, tuple(lateral), tuple(zlim))
    ax.set(
        xlim=lateral, ylim=zlim, xlabel=xlabel, ylabel="z (µm)",
        title=panel,
    )
    ax.legend(loc="upper right", fontsize=7.0, framealpha=0.94)


def draw_stack_zoom(ax: plt.Axes, geometry: dict) -> None:
    x0, x1 = 0.0, 10.0
    ax.add_patch(Rectangle((x0, -0.50), x1 - x0, 0.085, facecolor=COLORS["si"], edgecolor="none"))
    ax.add_patch(Rectangle((x0, -0.415), x1 - x0, 0.285, facecolor=COLORS["sio2"], edgecolor="#348f9d"))
    ax.add_patch(Rectangle((x0, -0.130), x1 - x0, 0.130, facecolor=COLORS["flake"], edgecolor="#8c570e"))
    ax.add_patch(Rectangle((x0, 0.0), x1 - x0, 0.10, facecolor=COLORS["air"], edgecolor="none"))

    ax.text(5, -0.455, "Si", ha="center", va="center", fontsize=10, color="white")
    ax.text(5, -0.2725, "SiO₂ 285 nm", ha="center", va="center", fontsize=10)
    ax.text(5, -0.065, "TaIrTe₄ 130 nm", ha="center", va="center", fontsize=10)
    ax.text(5, 0.05, "air", ha="center", va="center", fontsize=10)

    pabs_z = [value * 1e6 for value in geometry["pabs_nominal_control_volume_bounds_m"]["z"]]
    ax.add_patch(
        Rectangle(
            (0.3, pabs_z[0]), 9.4, pabs_z[1] - pabs_z[0], fill=False,
            edgecolor=COLORS["pabs"], linestyle="--", linewidth=2.0,
            label="raw Q / six-face CV: z=−180…+50 nm",
        )
    )
    ax.annotate(
        "CV includes only the upper 50 nm of SiO₂\nplus TaIrTe₄ and 50 nm air padding",
        xy=(9.5, -0.155), xytext=(6.8, -0.37), fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color=COLORS["pabs"]),
        color=COLORS["pabs"],
    )
    ax.annotate(
        "thermal input used here:\nTaIrTe₄-supported Q only",
        xy=(2.0, -0.065), xytext=(2.4, 0.075), fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color="#8c570e"), color="#8c570e",
    )
    ax.set(
        xlim=(x0, x1), ylim=(-0.50, 0.10), xticks=[], ylabel="z (µm)",
        title="D. Layer zoom and optical-Q control volume",
    )
    ax.text(
        9.55, 0.043, "raw Q / six-face CV: z=−180…+50 nm",
        color=COLORS["pabs"], ha="right", va="center", fontsize=8,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.case_result.read_text())
    geometry = payload["pre_run_contract"]["geometry"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    draw_top_view(axes[0, 0], geometry)
    draw_vertical_view(axes[0, 1], geometry, "x")
    draw_vertical_view(axes[1, 0], geometry, "y")
    draw_stack_zoom(axes[1, 1], geometry)
    fig.suptitle(
        "Corner-free 45° FDTD geometry — exact object bounds with schematic PML bands\n"
        "λ=11 µm; scalar Gaussian; six PML boundaries, 24 layers; no periodic/Bloch boundary",
        fontsize=16,
    )
    fig.text(
        0.5, 0.002,
        "PML hatch width is schematic: the contract stores 24 layers and six-face assignment, "
        "but not an independently read-back physical PML inner boundary.",
        ha="center", fontsize=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
