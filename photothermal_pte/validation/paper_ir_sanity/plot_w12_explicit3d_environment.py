#!/usr/bin/env python3
"""Draw the exact optical/thermal contract used by the 11 um sanity check.

The cross-section layer heights are intentionally schematic so that the
130-nm TaIrTe4 and 285-nm SiO2 layers remain visible next to the micron-scale
air and Si domains.  Every physical dimension is written explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle


REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_DIR = (
    REPOSITORY
    / "photothermal_pte"
    / "reports"
    / "paper_ir_w12_50nm_maxwell_analytic_explicit3d"
)
DEFAULT_OPTICAL_CASE = Path(
    "/data/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/"
    "w12_edge45_a_L60_nested_xy50_h22_dz5_pml24_t4_gpu5_20260731/"
    "case_result.json"
)
DEFAULT_THERMAL_SUMMARY = (
    DEFAULT_REPORT_DIR / "w12_50nm_maxwell_analytic_explicit3d_summary.json"
)
DEFAULT_PAPER = Path(
    "/home/seunghyun/tairte4/papers/"
    "Adv Funct Materials - 2026 - Blevins - Large Transverse "
    "Thermoelectric Effect in Weyl Semimetal TaIrTe4 Engineered for-2.pdf"
)

COLORS = {
    "air": "#eaf6ff",
    "TaIrTe4": "#ef7b6c",
    "SiO2": "#7fd3dc",
    "Si": "#526b88",
    "source": "#19a974",
    "pml": "#8f5cc2",
    "qbox": "#d58512",
    "core": "#087e8b",
    "fixed": "#b23a48",
    "conv": "#3d7a2a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def clean_axis(axis: plt.Axes, *, equal: bool = False) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("white")
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#24313f")
    if equal:
        axis.set_aspect("equal", adjustable="box")


def pml_frame(axis: plt.Axes, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """Draw a schematic PML frame; width is deliberately not a physical scale."""
    width = 1.5
    patches = (
        Rectangle((xmin, ymin), width, ymax - ymin),
        Rectangle((xmax - width, ymin), width, ymax - ymin),
        Rectangle((xmin, ymin), xmax - xmin, width),
        Rectangle((xmin, ymax - width), xmax - xmin, width),
    )
    for patch in patches:
        patch.set(
            facecolor=COLORS["pml"],
            edgecolor=COLORS["pml"],
            alpha=0.22,
            hatch="////",
            linewidth=0.7,
        )
        axis.add_patch(patch)


def draw_optical_xy(axis: plt.Axes) -> None:
    clean_axis(axis, equal=True)
    axis.set_xlim(-31, 31)
    axis.set_ylim(-31, 31)
    axis.add_patch(Rectangle((-30, -30), 60, 60, color=COLORS["air"], zorder=0))
    pml_frame(axis, -30, 30, -30, 30)

    # TaIrTe4 is the exact straight-edge half-plane y<=x.
    axis.add_patch(
        Polygon(
            [(-30, -30), (30, -30), (30, 30)],
            closed=True,
            facecolor=COLORS["TaIrTe4"],
            edgecolor="#a3342a",
            alpha=0.78,
            linewidth=1.5,
        )
    )
    axis.plot([-30, 30], [-30, 30], color="#762a22", linewidth=2.1)
    axis.add_patch(
        Rectangle(
            (-25, -25),
            50,
            50,
            fill=False,
            edgecolor=COLORS["source"],
            linestyle="--",
            linewidth=1.5,
        )
    )
    axis.add_patch(
        Circle(
            (0, 0),
            12,
            fill=False,
            edgecolor=COLORS["source"],
            linewidth=2.0,
        )
    )
    axis.add_patch(
        Rectangle(
            (-22, -22),
            44,
            44,
            fill=False,
            edgecolor="#2878b5",
            linestyle=":",
            linewidth=1.6,
        )
    )
    axis.add_patch(
        Rectangle(
            (-27, -27),
            54,
            54,
            fill=False,
            edgecolor=COLORS["qbox"],
            linestyle="-.",
            linewidth=1.5,
        )
    )
    axis.annotate("", xy=(15, -23), xytext=(5, -23), arrowprops={"arrowstyle": "->", "lw": 2})
    axis.text(16, -23, "x = b", va="center", fontsize=9)
    axis.annotate("", xy=(-23, 15), xytext=(-23, 5), arrowprops={"arrowstyle": "->", "lw": 2})
    axis.text(-23, 17, "y = a", ha="center", fontsize=9)
    axis.text(-19, 20, "air", color="#326587", fontsize=9, weight="bold")
    axis.text(8, -15, r"TaIrTe$_4$: $y\leq x$", color="#6f1811", fontsize=9, weight="bold")
    axis.text(-10, -1, r"$w_0=12\,\mu$m", color=COLORS["source"], fontsize=9, weight="bold")
    axis.text(-24, 26, "source square ±25 µm", color=COLORS["source"], fontsize=8)
    axis.text(-21.5, -20.5, "50-nm xy mesh to ±22 µm", color="#225b8a", fontsize=7.5)
    axis.text(-26.5, -27.8, "Q analysis ±27 µm", color=COLORS["qbox"], fontsize=7.5)
    axis.text(-29, 28.1, "PML: all x/y faces (24 layers)", color="#643696", fontsize=8)
    axis.set_title("A. Optical xy (top view at flake)", fontsize=11, weight="bold")


def draw_cross_section(
    axis: plt.Axes,
    *,
    optical: bool,
    horizontal_axis: str,
) -> None:
    clean_axis(axis)
    axis.set_xlim(-31, 31)
    axis.set_ylim(0, 10)

    if optical:
        pml_frame(axis, -30, 30, 0.2, 9.8)
        si_bottom, si_top = 1.0, 2.7
        ox_top, flake_top = 3.25, 3.70
        air_top = 9.8
        axis.add_patch(Rectangle((-30, si_bottom), 60, si_top - si_bottom, color=COLORS["Si"]))
        axis.add_patch(Rectangle((-30, si_top), 60, ox_top - si_top, color=COLORS["SiO2"]))
        axis.add_patch(Rectangle((-30, ox_top), 60, air_top - ox_top, color=COLORS["air"]))
        if horizontal_axis == "x":
            flake_x0, flake_x1 = 0, 30
            material_side = f"At y=0: TaIrTe$_4$ only for x≥0"
        else:
            flake_x0, flake_x1 = -30, 0
            material_side = f"At x=0: TaIrTe$_4$ only for y≤0"
        axis.add_patch(
            Rectangle(
                (flake_x0, ox_top),
                flake_x1 - flake_x0,
                flake_top - ox_top,
                facecolor=COLORS["TaIrTe4"],
                edgecolor="#a3342a",
            )
        )
        axis.plot([-25, 25], [7.15, 7.15], color=COLORS["source"], linewidth=2.4)
        axis.text(-24.5, 7.35, "scalar Gaussian source: z=+5 µm, span=50 µm", color=COLORS["source"], fontsize=7.5)
        axis.annotate(
            "",
            xy=(0, 4.05),
            xytext=(0, 6.95),
            arrowprops={"arrowstyle": "-|>", "lw": 2.1, "color": COLORS["source"]},
        )
        axis.plot([0], [3.52], marker="x", markersize=8, mew=2, color=COLORS["source"])
        axis.plot([-12, 0, 12], [6.95, 3.52, 6.95], "--", color=COLORS["source"], alpha=0.8)
        axis.text(1, 5.35, "propagation −z", color=COLORS["source"], fontsize=8)
        axis.text(1, 3.88, "waist/focus z=−65 nm", color=COLORS["source"], fontsize=7.5)
        axis.add_patch(
            Rectangle(
                (-27.5, 2.0),
                55,
                4.55,
                fill=False,
                edgecolor=COLORS["qbox"],
                linestyle="-.",
                linewidth=1.5,
            )
        )
        axis.text(-27, 6.28, "six-face closure box: ±27.5 µm, z=[−1.2,+4.5] µm", color=COLORS["qbox"], fontsize=7.2)
        axis.text(-28, 1.55, "Si: 3 µm (optical)", color="white", fontsize=8, weight="bold")
        axis.text(-28, 2.86, "SiO₂: 285 nm", color="#124952", fontsize=8, weight="bold")
        axis.text(-28, 4.12, "air", color="#326587", fontsize=8, weight="bold")
        axis.text(3 if horizontal_axis == "x" else -28, 3.35, "TaIrTe₄: 130 nm", color="#711a13", fontsize=7.5, weight="bold")
        axis.text(-28, 0.43, "z-domain = [−3.415,+10] µm; all six faces PML", color="#643696", fontsize=7.4)
        axis.text(-28, 9.1, material_side, fontsize=7.5)
        title_prefix = "Optical"
    else:
        si_bottom, si_top = 0.5, 3.1
        ox_top, flake_top = 3.85, 4.35
        air_top = 9.5
        axis.add_patch(Rectangle((-30, si_bottom), 60, si_top - si_bottom, color=COLORS["Si"]))
        axis.add_patch(Rectangle((-30, si_top), 60, ox_top - si_top, color=COLORS["SiO2"]))
        axis.add_patch(Rectangle((-30, ox_top), 60, air_top - ox_top, color=COLORS["air"]))
        if horizontal_axis == "x":
            flake_x0, flake_x1 = 0, 30
            material_side = "At y=0: flake x≥0"
        else:
            flake_x0, flake_x1 = -30, 0
            material_side = "At x=0: flake y≤0"
        axis.add_patch(
            Rectangle(
                (flake_x0, ox_top),
                flake_x1 - flake_x0,
                flake_top - ox_top,
                facecolor=COLORS["TaIrTe4"],
                edgecolor="#a3342a",
            )
        )
        axis.axhline(si_bottom, color=COLORS["fixed"], linewidth=3)
        axis.axvline(-30, color=COLORS["fixed"], linewidth=3)
        axis.axvline(30, color=COLORS["fixed"], linewidth=3)
        axis.annotate(
            "exposed surfaces: h=10 W m⁻² K⁻¹",
            xy=(-18, 9.0),
            xytext=(-18, 8.0),
            ha="center",
            arrowprops={"arrowstyle": "-|>", "color": COLORS["conv"]},
            color=COLORS["conv"],
            fontsize=7.5,
        )
        axis.plot([flake_x0, flake_x1], [ox_top, ox_top], color="#8b4a00", linewidth=2.2)
        axis.plot([flake_x0, flake_x1], [flake_top, flake_top], color="#6b276f", linewidth=2.2)
        axis.plot([-30, 30], [si_top, si_top], color="#194f55", linewidth=2.2)
        axis.text(-28, 1.55, "Si: 20 µm (thermal)", color="white", fontsize=8, weight="bold")
        axis.text(-28, 3.35, "SiO₂: 285 nm, κ=1.38", color="#124952", fontsize=7.8, weight="bold")
        axis.text(-28, 5.25, "air: 600 nm, κ=0.026", color="#326587", fontsize=7.8, weight="bold")
        axis.text(2 if horizontal_axis == "x" else -28, 4.0, "TaIrTe₄: 130 nm", color="#711a13", fontsize=7.5, weight="bold")
        axis.text(-28, 0.68, "bottom ΔT=0", color=COLORS["fixed"], fontsize=7.6, weight="bold")
        axis.text(-29.2, 6.3, "far side\nΔT=0", color=COLORS["fixed"], fontsize=7.3, ha="center")
        axis.text(24.5, 6.3, "far side\nΔT=0", color=COLORS["fixed"], fontsize=7.3, ha="center")
        axis.text(-27, 3.03, "G(SiO₂/Si)=1.1×10⁹ W m⁻² K⁻¹", color="#194f55", fontsize=6.8)
        axis.text(flake_x0 + 1, 3.70, "G(TaIrTe₄/SiO₂)=7.37×10⁶", color="#8b4a00", fontsize=6.8)
        axis.text(flake_x0 + 1, 4.46, "G(TaIrTe₄/air)=1", color="#6b276f", fontsize=6.8)
        axis.text(-28, 9.05, material_side, fontsize=7.5)
        axis.text(-28, 0.1, "No PML in thermal FVM; red boundaries are numerical Dirichlet truncations", fontsize=7.0)
        title_prefix = "Thermal"

    axis.text(29.2, 0.65, f"{horizontal_axis} →", ha="right", fontsize=8)
    axis.text(-30.2, 9.75, "z ↑", fontsize=8)
    plane = "xz (y=0)" if horizontal_axis == "x" else "yz (x=0)"
    axis.set_title(f"{title_prefix} {plane}", fontsize=11, weight="bold")


def draw_thermal_xy(axis: plt.Axes) -> None:
    clean_axis(axis, equal=True)
    axis.set_xlim(-31, 31)
    axis.set_ylim(-31, 31)
    axis.add_patch(Rectangle((-30, -30), 60, 60, color=COLORS["SiO2"], alpha=0.70))
    axis.add_patch(
        Polygon(
            [(-30, -30), (30, -30), (30, 30)],
            closed=True,
            facecolor=COLORS["TaIrTe4"],
            edgecolor="#a3342a",
            alpha=0.78,
        )
    )
    axis.plot([-30, 30], [-30, 30], color="#762a22", linewidth=2)
    axis.add_patch(
        Rectangle(
            (-12, -12),
            24,
            24,
            fill=False,
            edgecolor=COLORS["core"],
            linestyle="--",
            linewidth=1.8,
        )
    )
    axis.add_patch(
        Circle((0, 0), 12, fill=False, edgecolor=COLORS["source"], linewidth=1.8)
    )
    for position in (-30, 30):
        axis.axvline(position, color=COLORS["fixed"], linewidth=3)
        axis.axhline(position, color=COLORS["fixed"], linewidth=3)
    axis.text(-28, 27, "far x/y: fixed ΔT=0\n(numerical truncation)", color=COLORS["fixed"], fontsize=8, weight="bold")
    axis.text(-11.5, -10.5, "100-nm core cells: |x|,|y|≤12 µm", color=COLORS["core"], fontsize=7.5)
    axis.text(-10, 1, "same conservative\n3D Q remap", color=COLORS["source"], fontsize=8, ha="center")
    axis.text(8, -15, r"TaIrTe$_4$: $y\leq x$", color="#6f1811", fontsize=8, weight="bold")
    axis.text(-22, 17, "SiO₂ below flake / air above", color="#124952", fontsize=8)
    axis.annotate("", xy=(15, -23), xytext=(5, -23), arrowprops={"arrowstyle": "->", "lw": 2})
    axis.text(16, -23, "x = b", va="center", fontsize=9)
    axis.annotate("", xy=(-23, 15), xytext=(-23, 5), arrowprops={"arrowstyle": "->", "lw": 2})
    axis.text(-23, 17, "y = a", ha="center", fontsize=9)
    axis.set_title("D. Thermal xy (explicit 3D FVM)", fontsize=11, weight="bold")


def paper_ratio_digitization(paper_path: Path) -> dict[str, Any]:
    """Return a transparent digitization of the 11-um point in paper Fig. 3J."""
    # Pixel coordinates refer to page 6 rendered at 600 dpi, then cropped at
    # +3250,+2350.  The black 11-um marker center was obtained from the local
    # maximum of the binary-mask Euclidean distance transform.
    top_y_px = 186.5
    bottom_y_px = 749.5
    marker_y_px = 357.0
    plotted_ratio = (bottom_y_px - marker_y_px) / (bottom_y_px - top_y_px) * 1.2
    inverse = 1.0 / plotted_ratio
    pixel_uncertainty = 4.0
    ratio_uncertainty = pixel_uncertainty / (bottom_y_px - top_y_px) * 1.2
    inverse_uncertainty = ratio_uncertainty / plotted_ratio**2
    return {
        "source": {
            "paper_title": (
                "Large Transverse Thermoelectric Effect in Weyl Semimetal "
                "TaIrTe4 Engineered for Photodetection"
            ),
            "doi": "10.1002/adfm.75986",
            "figure": "Figure 3J",
            "local_pdf_path": str(paper_path),
            "local_pdf_size_bytes": paper_path.stat().st_size,
            "local_pdf_sha256": sha256(paper_path),
            "data_availability": (
                "The exact tabulated points are not supplied in the paper or "
                "supporting PDF; value below is a transparent plot digitization."
            ),
        },
        "digitization": {
            "pdf_page_one_indexed": 6,
            "render_dpi": 600,
            "crop_origin_in_rendered_page_px": [3250, 2350],
            "plot_axis_top_y_px": top_y_px,
            "plot_axis_bottom_y_px": bottom_y_px,
            "plot_axis_top_value": 1.2,
            "plot_axis_bottom_value": 0.0,
            "eleven_um_marker_center_in_crop_px": [850.0, marker_y_px],
            "estimated_marker_y_uncertainty_px": pixel_uncertainty,
        },
        "paper_plotted_quantity": "measured |I_a|/|I_b|",
        "wavelength_um": 11.0,
        "measured_abs_Ia_over_abs_Ib_digitized": plotted_ratio,
        "measured_abs_Ia_over_abs_Ib_uncertainty_estimate": ratio_uncertainty,
        "requested_abs_Ib_over_abs_Ia_inverted": inverse,
        "requested_abs_Ib_over_abs_Ia_uncertainty_estimate": inverse_uncertainty,
        "recommended_rounded_reporting": {
            "|I_a|/|I_b|": "approximately 0.84 ± 0.01",
            "|I_b|/|I_a|": "approximately 1.20 ± 0.02",
        },
        "signed_ratio_available": False,
        "note": (
            "The paper establishes |I_b|>|I_a| and plots magnitudes.  It does "
            "not publish a tabulated signed I_b/I_a value.  The unrelated "
            "635-nm SI ratio I(P1)/I(P2)=-1.26 is not used here."
        ),
    }


def draw_figure(output: Path, ratio: dict[str, Any]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(18, 11.4), constrained_layout=False)
    figure.subplots_adjust(left=0.035, right=0.985, bottom=0.11, top=0.88, wspace=0.14, hspace=0.24)
    draw_optical_xy(axes[0, 0])
    draw_cross_section(axes[0, 1], optical=True, horizontal_axis="x")
    draw_cross_section(axes[0, 2], optical=True, horizontal_axis="y")
    draw_thermal_xy(axes[1, 0])
    draw_cross_section(axes[1, 1], optical=False, horizontal_axis="x")
    draw_cross_section(axes[1, 2], optical=False, horizontal_axis="y")

    figure.suptitle(
        "11 µm straight-45°-edge sanity check — actual optical and explicit-3D thermal contracts",
        fontsize=17,
        weight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.922,
        (
            "Optical: scalar Gaussian (assumed target w₀=12 µm), normal incidence, "
            "60×60 µm², 6 PML, no periodic BC.  Thermal: same 60×60 µm² support, "
            "explicit air/TaIrTe₄/SiO₂/Si, no thermal PML."
        ),
        ha="center",
        fontsize=10,
    )
    rounded = ratio["recommended_rounded_reporting"]
    figure.text(
        0.5,
        0.052,
        (
            "Paper Fig. 3J at 11 µm (digitized, magnitude ratio):  "
            f"|Iₐ|/|Iᵦ| {rounded['|I_a|/|I_b|']}  ⇒  "
            f"|Iᵦ|/|Iₐ| {rounded['|I_b|/|I_a|']}.  "
            "This is the paper measurement, not the present simulation output."
        ),
        ha="center",
        fontsize=10.5,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#fff4ce", "edgecolor": "#bb8b00"},
    )
    figure.text(
        0.5,
        0.012,
        (
            "Cross-section vertical layer heights and PML band widths are schematic "
            "(not to scale); all physical dimensions and boundary types are annotated. "
            "No electrodes or weighting field are present in this optical/thermal-only stage."
        ),
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)


def write_report(
    output: Path,
    *,
    optical_case: Path,
    thermal_summary: Path,
    paper: Path,
    ratio: dict[str, Any],
) -> None:
    report = f"""# 11 µm optical and explicit-3D thermal environment

![Optical and thermal xy/xz/yz environment](W12_OPTICAL_THERMAL_XY_XZ_YZ_ENVIRONMENT.png)

## Scope

This schematic records the geometry actually used by the current straight
45-degree-edge sanity check.  It does not introduce a new solve.  Cross-section
layer heights and the drawn PML-band widths are schematic so that thin layers
remain visible; the numerical dimensions in the labels are authoritative.

## Optical contract

- scalar Gaussian at 11 µm; target waist radius 12 µm is an explicit assumption
- Lumerical source-object waist radius 11.9168648897 µm after source calibration
- source plane `z=+5 µm`, focus/target waist at `z=-65 nm`, propagation `-z`
- source aperture 50×50 µm²
- FDTD `x,y=[-30,+30] µm`, `z=[-3.415,+10] µm`
- all six boundaries PML, 24 layers; no periodic boundary
- TaIrTe4 130 nm, straight half-plane `y<=x`; lab `x=b`, `y=a`, `z=c`
- SiO2 285 nm and optical Si depth 3 µm
- nested local mesh: 50 nm xy to ±22 µm, 100 nm xy to ±27.5 µm,
  5 nm through the TaIrTe4 layer; remote regions use automatic nonuniform mesh
- component-resolved `Qx+Qy+Qz`; no clipping, smoothing, gain, or rescaling
- no electrodes in the optical model

## Explicit-3D thermal contract

- the same 60×60 µm² straight-edge support and full volumetric Maxwell/analytic Q
- air 600 nm, TaIrTe4 130 nm, SiO2 285 nm, Si 20 µm
- 100 nm core xy cells for `|x|,|y|<=12 µm`, graded outer cells, TaIrTe4 dz 10 nm
- `k_TaIrTe4=(3.8,14.4,1.0) W/(m K)` in lab `(x=b,y=a,z=c)`
- `k_SiO2=1.38`, `k_Si=145`, `k_air=0.026 W/(m K)`
- `G_TaIrTe4/air=1`, `G_TaIrTe4/SiO2=7.37e6`,
  `G_SiO2/Si=1.1e9 W/(m² K)`
- far x/y and bottom fixed `DeltaT=0` are numerical truncation boundaries
- exposed surfaces use `h=10 W/(m² K)`
- thermal FVM has no PML
- PTE, electrodes, weighting potential, adjoint, and optimization are not part
  of this four-source thermal-only result

## Paper current ratio at 11 µm

The paper's Figure 3J plots **measured `|I_a|/|I_b|`**, not `I_b/I_a`.
The exact numerical table is not published.  A 600-dpi digitization of the
11-µm marker gives

`|I_a|/|I_b| = {ratio['measured_abs_Ia_over_abs_Ib_digitized']:.6f}`

and therefore

`|I_b|/|I_a| = {ratio['requested_abs_Ib_over_abs_Ia_inverted']:.6f}`.

Accounting for marker/line thickness, these should be reported as approximately
`|I_a|/|I_b| = 0.84 ± 0.01` and `|I_b|/|I_a| = 1.20 ± 0.02`.
This is a **magnitude ratio**.  A signed `I_b/I_a` value is not tabulated by the
paper.  The SI value `I(P1)/I(P2)=-1.26` at 635 nm compares two positions and is
not a polarization-current ratio.

## Provenance

- optical case: `{optical_case}`
- optical case SHA-256: `{sha256(optical_case)}`
- thermal summary: `{thermal_summary}`
- thermal summary SHA-256: `{sha256(thermal_summary)}`
- paper DOI: `10.1002/adfm.75986`
- local paper: `{paper}`
- local paper SHA-256: `{sha256(paper)}`
"""
    output.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optical-case", type=Path, default=DEFAULT_OPTICAL_CASE)
    parser.add_argument("--thermal-summary", type=Path, default=DEFAULT_THERMAL_SUMMARY)
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    optical = load_json(args.optical_case)
    thermal = load_json(args.thermal_summary)

    geometry = optical["pre_run_contract"]["geometry"]
    checks = {
        "optical_domain_60um": bool(np.isclose(
            geometry["domain_bounds_m"]["x"][1]
            - geometry["domain_bounds_m"]["x"][0],
            60e-6,
        )),
        "all_six_optical_boundaries_pml": geometry["all_six_boundaries"] == "PML",
        "straight_edge_y_le_x": geometry["exact_flake_mask_kind"].startswith(
            "analytic half-plane"
        ),
        "thermal_domain_60um": bool(np.isclose(
            thermal["thermal_contract"]["lateral_domain_um"], 60.0
        )),
        "thermal_boundary_contract_present": (
            thermal["thermal_contract"]["far_xy_boundary"]
            == "fixed DeltaT=0 numerical truncation"
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"input contract mismatch: {checks}")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    ratio = paper_ratio_digitization(args.paper)
    ratio["contract_checks"] = checks
    ratio_path = args.report_dir / "paper_fig3j_11um_current_ratio_digitization.json"
    ratio_path.write_text(json.dumps(ratio, indent=2) + "\n", encoding="utf-8")

    figure_path = args.report_dir / "W12_OPTICAL_THERMAL_XY_XZ_YZ_ENVIRONMENT.png"
    draw_figure(figure_path, ratio)
    write_report(
        args.report_dir / "W12_OPTICAL_THERMAL_ENVIRONMENT_AND_PAPER_RATIO.md",
        optical_case=args.optical_case,
        thermal_summary=args.thermal_summary,
        paper=args.paper,
        ratio=ratio,
    )
    print(
        json.dumps(
            {
                "figure": str(figure_path),
                "ratio_json": str(ratio_path),
                "report": str(
                    args.report_dir
                    / "W12_OPTICAL_THERMAL_ENVIRONMENT_AND_PAPER_RATIO.md"
                ),
                "|I_a|/|I_b|": ratio["measured_abs_Ia_over_abs_Ib_digitized"],
                "|I_b|/|I_a|": ratio["requested_abs_Ib_over_abs_Ia_inverted"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
