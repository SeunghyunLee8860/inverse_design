#!/usr/bin/env python3
"""Offline size audit for a finite inverse-T array under a Gaussian beam."""

from __future__ import annotations

import argparse
import json
from math import ceil, exp
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np


PERIOD_X_UM = 1.5
PERIOD_Y_UM = 1.0
T_VERTICES_UM = np.asarray(
    [
        (-0.60, -0.35),
        (0.60, -0.35),
        (0.60, -0.25),
        (0.10, -0.25),
        (0.10, 0.35),
        (-0.10, 0.35),
        (-0.10, -0.25),
        (-0.60, -0.25),
    ]
)


def odd_covering_count(span_um: float, period_um: float) -> int:
    count = int(ceil(span_um / period_um))
    return count if count % 2 else count + 1


def scenario(w0_um: float, wavelength_max_um: float) -> dict[str, float | int]:
    # r=2w0 gives I/I0=exp(-8)=3.35e-4 at the finite-array boundary.
    nx = odd_covering_count(4.0 * w0_um, PERIOD_X_UM)
    ny = odd_covering_count(4.0 * w0_um, PERIOD_Y_UM)
    array_x = nx * PERIOD_X_UM
    array_y = ny * PERIOD_Y_UM
    pml_clearance = max(2.0, 0.5 * wavelength_max_um)
    domain_x = array_x + 2.0 * pml_clearance
    domain_y = array_y + 2.0 * pml_clearance
    fine_dx = 0.05
    fine_dy = 0.05
    fine_dz = 0.005
    fine_z_span = 0.60
    fine_cells = int(ceil(array_x / fine_dx) * ceil(array_y / fine_dy) * ceil(fine_z_span / fine_dz))
    return {
        "w0_um": w0_um,
        "wavelength_max_um_for_clearance": wavelength_max_um,
        "nx": nx,
        "ny": ny,
        "resonator_count": nx * ny,
        "array_span_x_um": array_x,
        "array_span_y_um": array_y,
        "PML_clearance_each_side_um": pml_clearance,
        "domain_span_x_um": domain_x,
        "domain_span_y_um": domain_y,
        "intensity_at_r_2w0_over_peak": exp(-8.0),
        "intensity_at_nearest_PML_over_peak": exp(-2.0 * (0.5 * min(domain_x, domain_y) / w0_um) ** 2),
        "local_mesh_dx_dy_nm": 50.0,
        "local_mesh_dz_nm": 5.0,
        "local_fine_Yee_cell_proxy": fine_cells,
        "memory_status": "requires v261 runsetup readback; cell proxy is not a GPU-memory claim",
    }


def plot_scenario(ax: plt.Axes, item: dict[str, float | int]) -> None:
    nx, ny = int(item["nx"]), int(item["ny"])
    array_x, array_y = float(item["array_span_x_um"]), float(item["array_span_y_um"])
    domain_x, domain_y = float(item["domain_span_x_um"]), float(item["domain_span_y_um"])
    for ix in range(nx):
        for iy in range(ny):
            x0 = (ix - 0.5 * (nx - 1)) * PERIOD_X_UM
            y0 = (iy - 0.5 * (ny - 1)) * PERIOD_Y_UM
            ax.add_patch(Polygon(T_VERTICES_UM + (x0, y0), closed=True, facecolor="#e0a600", edgecolor="#7a5300", lw=0.25))
    ax.add_patch(Rectangle((-0.5 * array_x, -0.5 * array_y), array_x, array_y, fill=False, edgecolor="#cc0000", lw=1.4, label="finite T array"))
    ax.add_patch(Rectangle((-0.5 * domain_x, -0.5 * domain_y), domain_x, domain_y, fill=False, edgecolor="#6a1b9a", lw=1.8, ls="--", label="x/y PML domain"))
    grid_x = np.linspace(-0.5 * domain_x, 0.5 * domain_x, 300)
    grid_y = np.linspace(-0.5 * domain_y, 0.5 * domain_y, 300)
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="xy")
    intensity = np.exp(-2.0 * (xx**2 + yy**2) / float(item["w0_um"]) ** 2)
    ax.contour(xx, yy, intensity, levels=[np.exp(-8), np.exp(-2), 0.5], colors=["#2c7fb8", "#41b6c4", "white"], linewidths=[1.0, 1.2, 1.4])
    ax.set_aspect("equal")
    ax.set_xlim(-0.52 * domain_x, 0.52 * domain_x)
    ax.set_ylim(-0.52 * domain_y, 0.52 * domain_y)
    ax.set_xlabel("Lumerical x=b (µm)")
    ax.set_ylabel("Lumerical y=a (µm)")
    ax.set_title(f"w0={item['w0_um']} µm: {nx}×{ny} = {nx*ny} finite T resonators")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenarios = [scenario(4.0, 12.0), scenario(8.5, 12.0)]
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    for ax, item in zip(axes, scenarios):
        plot_scenario(ax, item)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Finite multi-T Gaussian contract audit — no periodic boundary in the device solve")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(output / "finite_T_array_gaussian_size_audit.png", dpi=220)
    plt.close(fig)
    summary = {
        "status": "AUDITED_FINITE_T_ARRAY_SIZE_CANDIDATES",
        "spectral_search_um": [4.0, 12.0],
        "spectral_search_note": "The 4-12 um wavelength range is independent of the Gaussian-waist scenarios below.",
        "boundary_contract": "six PML for Maxwell; physical thermal/electrical BCs solved separately",
        "scenarios": scenarios,
        "decision": "run w0=4 um finite-array smoke first; promote w0=8.5 um only after v261 runsetup GPU-memory audit and field/Q convergence",
        "not_executed": ["FDTD", "thermal", "PTE", "adjoint", "optimization"],
    }
    (output / "FINITE_T_ARRAY_GAUSSIAN_SIZE_AUDIT.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = f"""# Finite inverse-T array Gaussian size audit

This is an offline geometry/cost audit. It does **not** contain an FDTD, thermal,
PTE, adjoint, or optimization result. The periodic resonance search spans
`4-12 um`; that wavelength interval is independent of the beam-waist choices.

| Assumed waist | Array | T count | Maxwell x/y span | Fine Yee-cell proxy |
|---:|---:|---:|---:|---:|
| 4.0 um | {scenarios[0]['array_span_x_um']:.1f} x {scenarios[0]['array_span_y_um']:.1f} um | {scenarios[0]['resonator_count']} | {scenarios[0]['domain_span_x_um']:.1f} x {scenarios[0]['domain_span_y_um']:.1f} um | {scenarios[0]['local_fine_Yee_cell_proxy']:,} |
| 8.5 um | {scenarios[1]['array_span_x_um']:.1f} x {scenarios[1]['array_span_y_um']:.1f} um | {scenarios[1]['resonator_count']} | {scenarios[1]['domain_span_x_um']:.1f} x {scenarios[1]['domain_span_y_um']:.1f} um | {scenarios[1]['local_fine_Yee_cell_proxy']:,} |

The finite-device Maxwell model uses six PML boundaries. Thermal and electrical
models use physical boundaries and do not inherit optical PML. The array reaches
`r=2*w0`, where an ideal Gaussian intensity is `exp(-8)=3.35e-4` of its peak.

Decision: run the `w0=4 um` finite-array smoke first. The `w0=8.5 um` scenario
is promoted only after a v261 runsetup audit records the realized mesh, GPU
memory, and source-boundary field level. The fine-cell proxy is not a memory
claim.
"""
    (output / "FINITE_T_ARRAY_GAUSSIAN_SIZE_AUDIT.md").write_text(report)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
