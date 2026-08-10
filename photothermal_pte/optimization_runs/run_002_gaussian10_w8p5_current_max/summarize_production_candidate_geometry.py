#!/usr/bin/env python3
"""Publish the matched-volume Run-002 production-candidate runsetup audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = HERE / "plots"
MANIFEST = HERE / "manifests" / "RAW_ARTIFACT_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifacts(directory: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(directory.iterdir())
        if path.is_file()
    ]


def span(bounds: list[float]) -> tuple[float, float]:
    return bounds[0] * 1e6, bounds[1] * 1e6


def setup_plot(result: dict[str, object], output: Path) -> None:
    geometry = result["geometry_readback_m"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    colors = {
        "air": "#dff3ff",
        "design": "#f6c344",
        "flake": "#e76f51",
        "sio2": "#78c6d0",
        "si": "#577590",
        "pml": "#9b5de5",
    }

    axis = axes[0]
    axis.add_patch(Rectangle((-24, -24), 48, 48, color=colors["air"], alpha=0.55))
    axis.add_patch(Rectangle((-24, -24), 48, 48, fill=False, edgecolor=colors["pml"], linewidth=4, linestyle="--", label="six-PML domain"))
    axis.add_patch(Rectangle((-20, -20), 40, 40, fill=False, edgecolor="#00a6a6", linewidth=2, label="source aperture / Q box"))
    axis.add_patch(Rectangle((-10, -10), 20, 20, color=colors["design"], alpha=0.75, label="coarse design canvas"))
    axis.plot(0, 0, marker="x", markersize=9, color="black", label="beam center")
    axis.set(xlim=(-25, 25), ylim=(-25, 25), xlabel="x=b (µm)", ylabel="y=a (µm)", title="XY top view (z=0)", aspect="equal")
    axis.legend(fontsize=8, loc="upper right")

    def cross_section(axis, horizontal: str, title: str) -> None:
        axis.add_patch(Rectangle((-24, -8), 48, 16, color=colors["air"], alpha=0.55))
        axis.add_patch(Rectangle((-24, -8), 48, 16, fill=False, edgecolor=colors["pml"], linewidth=4, linestyle="--"))
        axis.add_patch(Rectangle((-24, -8), 48, 7.615, color=colors["si"], alpha=0.9, label="Si (Palik at 10 µm)"))
        axis.add_patch(Rectangle((-24, -0.385), 48, 0.285, color=colors["sio2"], label="285 nm SiO₂"))
        axis.add_patch(Rectangle((-24, -0.100), 48, 0.100, color=colors["flake"], label="100 nm TaIrTe₄"))
        axis.add_patch(Rectangle((-10, 0), 20, 1.0, color=colors["design"], alpha=0.85, label="1 µm design"))
        axis.add_patch(Rectangle((-20, -1.25), 40, 2.5, fill=False, edgecolor="#00a6a6", linewidth=2, label="matched Q/flux box"))
        axis.annotate("Gaussian source\nz=5 µm, propagation −z", xy=(0, 4.8), xytext=(7, 5.7), arrowprops={"arrowstyle": "->", "linewidth": 1.5}, ha="center")
        axis.axhline(5, xmin=(4 / 48), xmax=(44 / 48), color="#2274a5", linewidth=2)
        axis.set(xlim=(-25, 25), ylim=(-8.2, 8.2), xlabel=f"{horizontal} (µm)", ylabel="z (µm)", title=title)

    cross_section(axes[1], "x=b", "XZ cross-section (y=0)")
    cross_section(axes[2], "y=a", "YZ cross-section (x=0)")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[2].legend(handles, labels, fontsize=7, loc="upper right")
    fig.suptitle("Run 002 production-candidate geometry — exact coordinates, thin layers not to visual scale")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-directory", required=True, type=Path)
    parser.add_argument("--diagnostic-directory", type=Path)
    args = parser.parse_args()
    raw = args.raw_directory.expanduser().resolve()
    result = json.loads((raw / "production_candidate_geometry_audit.json").read_text())
    if not result.get("passed", False):
        raise RuntimeError("matched-volume production-candidate runsetup did not pass")
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / "production_candidate_geometry_summary.json"
    report_path = RESULTS / "PRODUCTION_CANDIDATE_GEOMETRY_REPORT.md"
    plot_path = PLOTS / "production_candidate_geometry_xyz.png"
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    setup_plot(result, plot_path)
    mesh = result["mesh_readback"]
    source = result["source_contract"]["source"]
    materials = result["materials"]
    report_path.write_text(
        f"""# Run 002 production-candidate geometry audit

Status: `{result['status']}`

This is a Lumerical runsetup/readback certificate.  No Maxwell, thermal,
adjoint, or optimization solve was run.

## Frozen candidate

- FDTD: 48×48 µm lateral, z=-8..8 µm, six PML, 24 layers.
- Source: calibrated scalar Gaussian, λ=10 µm, target-plane waist 8.5 µm,
  source aperture 40×40 µm, source z=5 µm, focus z=0.
- Long optical TaIrTe4 background: 60×60×0.1 µm, extending beyond the
  transverse PML bounds; no artificial optical flake edge.
- Bottom stack: 285 nm Kitamura-SiO2 on Palik-Si.
- Coarse design canvas: 20×20×1 µm, 201×201×21 imported nodes,
  100 nm lateral and 50 nm vertical node spacing.
- Matched Q/six-face box: x,y=-20..20 µm and z=-1.25..1.25 µm.

The exact layer order is Si → bottom SiO2 → TaIrTe4 → imported design → air.
All adjacent z interfaces were read back as contiguous.

## Material values at 10 µm

- TaIrTe4 epsilon_x=epsilon_b:
  `{materials['TaIrTe4']['epsilon_at_10um']['x']}`
- TaIrTe4 epsilon_y=epsilon_a:
  `{materials['TaIrTe4']['epsilon_at_10um']['y']}`
- TaIrTe4 epsilon_z=epsilon_b closure:
  `{materials['TaIrTe4']['epsilon_at_10um']['z']}`
- SiO2 n+ik: `{materials['SiO2']['n_at_10um']}`
- Si n+ik: `{materials['Si']['n_at_10um']}`

`epsilon_c=epsilon_b` is an explicit paper-consistent 3D closure, not an
independent c-axis measurement.

## Realized mesh

- mesh lines: `{mesh['shape_xyz']}`
- grid-point product: `{mesh['grid_points']:,}`
- minimum dx/dy/dz:
  `{mesh['minimum_step_m']['x']*1e9:.3f}` /
  `{mesh['minimum_step_m']['y']*1e9:.3f}` /
  `{mesh['minimum_step_m']['z']*1e9:.3f}` nm
- maximum dx/dy/dz:
  `{mesh['maximum_step_m']['x']*1e9:.3f}` /
  `{mesh['maximum_step_m']['y']*1e9:.3f}` /
  `{mesh['maximum_step_m']['z']*1e9:.3f}` nm

The 100 nm value is the design-node spacing, not the minimum Yee step.  v261
realized about 59.7 nm minimum lateral spacing in this runsetup.

## Corrected diagnostic

The first runsetup requested an asymmetric z control volume, while the pabs
internal monitors remained centered.  It is preserved as a failed diagnostic.
The promoted v2 uses an exactly matched symmetric control volume.  No field
solve was performed with the mismatched geometry.
"""
    )
    manifest = json.loads(MANIFEST.read_text())
    manifest["production_candidate_geometry_runsetup"] = {
        "status": result["status"],
        "raw_directory": str(raw),
        "artifacts": artifacts(raw),
        "diagnostic_mismatched_control_volume": (
            {
                "raw_directory": str(args.diagnostic_directory.expanduser().resolve()),
                "artifacts": artifacts(args.diagnostic_directory.expanduser().resolve()),
            }
            if args.diagnostic_directory is not None
            else None
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
