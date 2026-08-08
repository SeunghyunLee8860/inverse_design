#!/usr/bin/env python3
"""Publish the validated Run-002 production-window selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


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


def record(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-directory", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw_directory.expanduser().resolve()
    result_path = raw / "production_design_window_selection_result.json"
    result = json.loads(result_path.read_text())
    if result.get("status") != "VALIDATED_PRODUCTION_DESIGN_WINDOW_SELECTION" or not result.get("passed"):
        raise RuntimeError("production-window selection is not validated")
    artifact = Path(result["raw_artifact"]["path"])
    if record(artifact) != result["raw_artifact"]:
        raise RuntimeError("window-selection artifact provenance mismatch")
    data = np.load(artifact)
    x = np.asarray(data["x_um"], float)
    y = np.asarray(data["y_um"], float)
    gradient = np.asarray(data["gradient_total_A"], float)
    absolute = np.abs(gradient)

    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / "production_design_window_selection_summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    promoted = result["promoted_window"]
    rows = "\n".join(
        f"| {name} | {value['area_um2']:.2f} | {value['absolute_gradient_L1_fraction']*100:.3f}% | {value['passes_90_percent_gate']} |"
        for name, value in result["original_reviewed_candidates"].items()
    )
    report_path = RESULTS / "PRODUCTION_DESIGN_WINDOW_SELECTION_REPORT.md"
    report_path.write_text(
        f"""# Production design-window selection

Status: `{result['status']}`

The window is selected from the absolute L1 mass of the already validated
combined physical-density gradient. No Maxwell, thermal, adjoint, FD, or
optimization solve was run in this checkpoint.

| original reviewed candidate | area (µm²) | retained | passes 90% |
|---|---:|---:|---:|
{rows}

Every original 12×6 µm strip and the centered 10×10 µm control fails by a
large margin. They were not silently promoted.

The promoted centered window is
`x,y=[-9.3,9.3] µm`, or 18.6×18.6 µm. It retains
`{promoted['absolute_gradient_L1_fraction']*100:.6f}%` of the full-canvas
absolute combined gradient. The immediately smaller 18.4×18.4 µm control
retains only
`{result['immediately_smaller_centered_control']['absolute_gradient_L1_fraction']*100:.6f}%`
and fails the 90% gate. At 50 nm production spacing the promoted nodal design
has shape `373×373`.

The selected area is `{result['promoted_area_fraction']*100:.2f}%` of the
20×20 µm coarse canvas. This modest reduction is a physical consequence of
the broad 8.5 µm-waist illumination; a much smaller window would discard most
of the available sensitivity.
"""
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
    vmax = max(float(np.max(np.abs(gradient))), np.finfo(float).tiny)
    image = axes[0].pcolormesh(x, y, gradient.T, shading="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    fig.colorbar(image, ax=axes[0], label="combined physical-density gradient (A)")
    bounds = promoted["bounds_um"]
    axes[0].add_patch(Rectangle((bounds["x"][0], bounds["y"][0]), bounds["x"][1]-bounds["x"][0], bounds["y"][1]-bounds["y"][0], fill=False, lw=2.2, ec="lime", label="promoted 18.6 µm window"))
    axes[0].set(xlabel="x=b (µm)", ylabel="y=a (µm)", title="Validated combined gradient", aspect="equal")
    axes[0].legend(loc="lower right")

    candidate_names = list(result["original_reviewed_candidates"])
    candidate_values = [result["original_reviewed_candidates"][name]["absolute_gradient_L1_fraction"] for name in candidate_names]
    labels = ["+a strip", "-a strip", "+b strip", "-b strip", "10×10", "18.4×18.4", "18.6×18.6"]
    values = candidate_values + [result["immediately_smaller_centered_control"]["absolute_gradient_L1_fraction"], promoted["absolute_gradient_L1_fraction"]]
    colors = ["tab:red" if value < 0.9 else "tab:green" for value in values]
    axes[1].bar(np.arange(len(values)), np.asarray(values)*100.0, color=colors)
    axes[1].axhline(90.0, color="black", ls="--", label="selection gate")
    axes[1].set_xticks(np.arange(len(values)), labels, rotation=35, ha="right")
    axes[1].set(ylabel="retained |gradient| L1 (%)", title="Window coverage", ylim=(0, 100))
    axes[1].legend()
    fig.suptitle("Run 002 gradient-driven production-window selection")
    plot_path = PLOTS / "production_design_window_selection.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    manifest = json.loads(MANIFEST.read_text())
    manifest["production_design_window_selection"] = {
        "status": result["status"],
        "raw_artifacts_committed_to_git": False,
        "raw_directory": str(raw),
        "artifacts": [record(result_path), record(artifact)],
        "source_combined_gradient": result["source"]["combined_gradient"],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "report": str(report_path), "summary": str(summary_path), "plot": str(plot_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
