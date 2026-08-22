#!/usr/bin/env python3
"""Publish the v1-axis error and the Fig. 1b-corrected Z M2 geometry."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_Z_M2_figure_corrected_geometry"


def load_geometry_module():
    path = HERE / "05_actual_metasurface_geometry.py"
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("z_geometry_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def draw(axis, geometry, title: str) -> None:
    px = geometry.period_x_nm * 1e-3
    py = geometry.period_y_nm * 1e-3
    axis.add_patch(Rectangle((-px / 2, -py / 2), px, py, facecolor="#d8ecff", edgecolor="#285f8f", lw=2))
    for item in geometry.polygons:
        vertices = [(x * 1e-3, y * 1e-3) for x, y in item.vertices_nm]
        axis.add_patch(Polygon(vertices, closed=True, facecolor="#f5b82e", edgecolor="#805500", lw=2))
    axis.set_xlim(-px / 2 - 0.2, px / 2 + 0.2)
    axis.set_ylim(-py / 2 - 0.2, py / 2 + 0.2)
    axis.set_aspect("equal")
    axis.set_xlabel("Lumerical x=b (um)")
    axis.set_ylabel("Lumerical y=a (um)")
    axis.set_title(title)
    axis.grid(alpha=0.15)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    module = load_geometry_module()
    v1 = module.z_m2_5300nm_corner_joined_tairte4("LH")
    v2 = module.z_m2_5300nm_figure_corrected_tairte4_v2("LH")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    draw(axes[0], v1, "v1 diagnostic: WRONG axis assignment\nL horizontal, W vertical")
    draw(axes[1], v2, "corrected v2: Fig. 1b axis assignment\nW horizontal, L vertical")
    fig.suptitle("Z2022 M2 geometry audit: published scalar dimensions, corner-join closure")
    figure = OUT / "Z2022_M2_v1_vs_figure_corrected_v2.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)
    result = {
        "status": "READY_Z2022_M2_FIGURE_AXIS_CORRECTED_DIGITIZED_CLOSURE",
        "paper_evidence": {
            "main_pdf": "/home/seunghyun/tairte4/papers/s41467-022-32309-w.pdf",
            "figure": "Fig. 1b",
            "supplement": "/home/seunghyun/tairte4/papers/41467_2022_32309_MOESM1_ESM.pdf",
            "dimensions": "Supplementary Table 1, M2",
        },
        "correction": "W1/W2 -> x=b with P2; L1/L2 -> y=a with P1",
        "v1": v1.as_dict(),
        "v2": v2.as_dict(),
        "remaining_closure": "inner-corner overlap/gap is not disclosed; exact author CAD remains unavailable; this is a named figure-digitized closure",
    }
    (OUT / "Z2022_M2_FIGURE_CORRECTED_GEOMETRY.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "Z2022_M2_FIGURE_CORRECTED_GEOMETRY_REPORT.md").write_text(
        "# Z2022 M2 geometry correction\n\n"
        "Status: `READY_Z2022_M2_FIGURE_AXIS_CORRECTED_DIGITIZED_CLOSURE`\n\n"
        "The previous v1 implementation interchanged the dimensions shown in Fig. 1b. "
        "It placed `L1/L2` horizontally and `W1/W2` vertically. The corrected v2 "
        "places `W1/W2` along Lumerical `x=b` with period `P2`, and `L1/L2` "
        "along `y=a` with period `P1`, consistent with the array micrograph in "
        "Supplementary Fig. 4. The inner-corner join remains an explicit reconstruction because the exact "
        "junction overlap/gap CAD is not published. The v1 optical results are retained "
        "only as a wrong-axis diagnostic and cannot be presented as the paper geometry.\n\n"
        "![v1 versus v2](Z2022_M2_v1_vs_figure_corrected_v2.png)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
