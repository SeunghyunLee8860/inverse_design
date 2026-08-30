#!/usr/bin/env python3
"""Publish the optical-only T/Z six-polarization screening contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results_periodic_T_Z_six_polarization_contract"


def load_geometry():
    path = HERE / "05_actual_metasurface_geometry.py"
    spec = importlib.util.spec_from_file_location("periodic_suite_geometry", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def polygon_bounds(geometry) -> dict[str, float]:
    vertices = np.concatenate(
        [np.asarray(item.vertices_nm, float) for item in geometry.polygons], axis=0
    )
    return {
        "xmin_nm": float(np.min(vertices[:, 0])),
        "xmax_nm": float(np.max(vertices[:, 0])),
        "ymin_nm": float(np.min(vertices[:, 1])),
        "ymax_nm": float(np.max(vertices[:, 1])),
    }


def draw(ax, geometry, title: str) -> None:
    px = float(geometry.period_x_nm)
    py = float(geometry.period_y_nm)
    ax.add_patch(
        Rectangle((-px / 2, -py / 2), px, py, facecolor="#d9ecff", edgecolor="#2166ac", lw=2)
    )
    for item in geometry.polygons:
        ax.add_patch(
            Polygon(np.asarray(item.vertices_nm), closed=True, facecolor="#fdbf2d", edgecolor="#7f5500", lw=2)
        )
    ax.axhline(0, color="0.55", lw=0.8, ls="--")
    ax.axvline(0, color="0.55", lw=0.8, ls="--")
    ax.set_aspect("equal")
    ax.set_xlim(-px / 2 * 1.06, px / 2 * 1.06)
    ax.set_ylim(-py / 2 * 1.06, py / 2 * 1.06)
    ax.set_xlabel("Lumerical x = crystal b (nm)")
    ax.set_ylabel("Lumerical y = crystal a (nm)")
    ax.set_title(title)


def main() -> int:
    geometry_module = load_geometry()
    t = geometry_module.inverse_t_mir_4750nm()
    z_paper = geometry_module.z_m2_5300nm_figure_period_corrected_tairte4_v3("LH")
    z_expanded = geometry_module.z_m2_5300nm_centered_expanded_supercell_v4("LH")
    polarizations = {
        "x_b": [1.0, 0.0, 0.0],
        "y_a": [0.0, 1.0, 0.0],
        "linear_plus_45": [2**-0.5, 2**-0.5, 0.0],
        "linear_minus_45": [2**-0.5, -2**-0.5, 0.0],
        "CP_plus": [2**-0.5, 2**-0.5, 90.0],
        "CP_minus": [2**-0.5, 2**-0.5, -90.0],
    }
    payload = {
        "status": "READY_PERIODIC_T_Z_SIX_POLARIZATION_OPTICAL_SCREEN",
        "scope": "periodic optical R/T/A first; selected volumetric Q only after closure; no thermal/weighting/PTE",
        "wavelength_range_um": [4.0, 12.0],
        "axis_mapping": {"x": "b", "y": "a", "z": "c=b closure"},
        "polarizations": polarizations,
        "T_geometry": t.as_dict(),
        "Z_paper_v3_preserved": z_paper.as_dict(),
        "Z_centered_expanded_v4": z_expanded.as_dict(),
        "Z_bounds": {
            "paper_v3": polygon_bounds(z_paper),
            "centered_expanded_v4": polygon_bounds(z_expanded),
        },
        "important_interpretation": (
            "CP_plus/minus are explicit Ey-Ex phase offsets, not assigned LCP/RCP labels; "
            "the expanded square Z supercell is a project scenario, not the paper M2 lattice"
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "PERIODIC_T_Z_SIX_POLARIZATION_CONTRACT.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)
    draw(axes[0], t, "T 2024 periodic cell")
    draw(axes[1], z_paper, "Z paper-period v3 (preserved)\n5.1 x 2.6 um; touches y seam")
    draw(axes[2], z_expanded, "Z centered expanded v4 (new)\n5.1 x 5.1 um; isolated from x/y seams")
    fig.suptitle("Periodic optical geometries; planar TaIrTe4 fills each unit cell")
    fig.savefig(OUTPUT / "periodic_T_Z_geometry_contract.png", dpi=220)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
