#!/usr/bin/env python3
"""Publish the approved 4--12 um T/Z experiment sequence for meetings."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


HERE = Path(__file__).resolve().parent


def load_geometry():
    path = HERE / "05_actual_metasurface_geometry.py"
    spec = importlib.util.spec_from_file_location("t_z_4_12_geometry", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def top_view(axis, geometry, title: str) -> None:
    px, py = geometry.period_x_nm, geometry.period_y_nm
    axis.add_patch(Rectangle((-px / 2, -py / 2), px, py, facecolor="#eaf4f8", edgecolor="#6a1b9a", lw=2, ls="--"))
    for item in geometry.polygons:
        axis.add_patch(Polygon(item.vertices_nm, closed=True, facecolor="#f7c548", edgecolor="#815800", lw=1.6))
    axis.set_xlim(-0.55 * px, 0.55 * px)
    axis.set_ylim(-0.55 * py, 0.55 * py)
    axis.set_aspect("equal")
    axis.set_xlabel("Lumerical x = TaIrTe4 b (nm)")
    axis.set_ylabel("Lumerical y = TaIrTe4 a (nm)")
    axis.set_title(title)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    module = load_geometry()
    t = module.inverse_t_mir_4750nm()
    z_lh = module.z_m2_5300nm_corner_joined_tairte4("LH")
    z_rh = module.z_m2_5300nm_corner_joined_tairte4("RH")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    top_view(axes[0, 0], t, "Stage 1: inverse-T periodic R/T/A screen, 4-12 um")
    top_view(axes[0, 1], z_lh, "Stage 4: reconstructed Z M2, LH")
    top_view(axes[1, 0], z_rh, "Stage 4: reconstructed Z M2, RH")
    axis = axes[1, 1]
    axis.axis("off")
    steps = [
        "1  Periodic T: plane wave -> R/T/A resonance",
        "2  Finite multi-T: Gaussian + six PML -> volumetric Q",
        "3  Same Q: explicit thermal + weighting -> terminal PTE",
        "4  Periodic/finite Z: CP+/CP- selectivity (no LCP label yet)",
    ]
    for index, text in enumerate(steps):
        y = 0.87 - 0.21 * index
        axis.text(0.04, y, text, fontsize=14, va="center", bbox={"boxstyle": "round", "facecolor": "#edf4ff", "edgecolor": "#3568a8"})
        if index < len(steps) - 1:
            axis.annotate("", xy=(0.12, y - 0.15), xytext=(0.12, y - 0.07), arrowprops={"arrowstyle": "->", "lw": 2})
    axis.text(0.04, 0.02, "No Q clipping/smoothing/gain/rescaling. Periodic optical BCs are never reused as thermal/electrical BCs.", fontsize=11, color="#8b1a1a")
    plot_path = output / "T_Z_4_12_screening_contract.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    summary = {
        "status": "READY_T_Z_4_12_SCREENING_CONTRACT",
        "wavelength_range_um": [4.0, 12.0],
        "T_geometry": t.as_dict(),
        "Z_LH_geometry": z_lh.as_dict(),
        "Z_RH_geometry": z_rh.as_dict(),
        "stages": steps,
        "not_claimed": ["completed GPU spectrum", "finite Q", "thermal/PTE result", "inverse design"],
    }
    json_path = output / "T_Z_4_12_SCREENING_CONTRACT.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path = output / "T_Z_4_12_SCREENING_CONTRACT.md"
    report_path.write_text(
        "# T/Z 4-12 um screening contract\n\n"
        "The TaIrTe4 substitution is allowed to select its own resonance inside "
        "4-12 um; neither paper's original wavelength is imposed as the answer.\n\n"
        "1. Periodic inverse-T plane-wave R/T/A spectrum.\n"
        "2. Selected T resonance in a finite multi-element array with Gaussian illumination and six PML.\n"
        "3. Component-resolved finite Q is conservatively mapped to the explicit thermal/PTE model.\n"
        "4. The reconstructed 2022 M2 Z geometry is screened for explicit CP+/CP- source phases.\n\n"
        "The Z polygon uses published scalar dimensions and a figure-derived corner-join closure; it is not author CAD. "
        "CP+ and CP- are not renamed LCP/RCP until the propagation/time convention is independently audited.\n"
    )
    artifacts = []
    for path in (plot_path, json_path, report_path):
        artifacts.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps({"artifacts": artifacts}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
