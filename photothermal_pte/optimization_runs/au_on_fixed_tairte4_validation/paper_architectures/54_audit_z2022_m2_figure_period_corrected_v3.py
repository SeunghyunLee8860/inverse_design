#!/usr/bin/env python3
"""Audit the legacy v2 period error against the corrected v3 M2 closure."""

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
OUT = HERE / "results_Z_M2_figure_period_corrected_v3"


def load_geometry_module():
    path = HERE / "05_actual_metasurface_geometry.py"
    spec = importlib.util.spec_from_file_location("z_geometry_v3_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def draw(axis, geometry, title: str) -> None:
    px = geometry.period_x_nm * 1e-3
    py = geometry.period_y_nm * 1e-3
    axis.add_patch(Rectangle((-px/2, -py/2), px, py, facecolor="#d8ecff", edgecolor="#285f8f", lw=2))
    for item in geometry.polygons:
        vertices = [(x*1e-3, y*1e-3) for x, y in item.vertices_nm]
        axis.add_patch(Polygon(vertices, closed=True, facecolor="#f5b82e", edgecolor="#805500", lw=2))
    axis.axvline(0.0, color="0.5", lw=0.7, ls="--")
    axis.axhline(0.0, color="0.5", lw=0.7, ls="--")
    axis.set_xlim(-px/2 - 0.2, px/2 + 0.2)
    axis.set_ylim(-py/2 - 0.2, py/2 + 0.2)
    axis.set_aspect("equal")
    axis.set_xlabel("Lumerical x=b (um); P1 horizontal")
    axis.set_ylabel("Lumerical y=a (um); P2 vertical")
    axis.set_title(title)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    module = load_geometry_module()
    v2 = module.z_m2_5300nm_figure_corrected_tairte4_v2("LH")
    v3 = module.z_m2_5300nm_figure_period_corrected_tairte4_v3("LH")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    draw(axes[0], v2, "legacy v2: periods reversed\nP2 horizontal, P1 vertical (WRONG)")
    draw(axes[1], v3, "v3: Fig. 1b periods and axes\nP1 horizontal, P2 vertical")
    fig.suptitle("2022 M2 Z geometry correction (Table S1 dimensions; exact CAD unavailable)")
    figure = OUT / "Z2022_M2_v2_vs_figure_period_corrected_v3.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)
    payload = {
        "status": "READY_Z2022_M2_FIGURE_PERIOD_CORRECTED_V3",
        "paper_evidence": {
            "main_figure": "Fig. 1b: P1 horizontal, P2 vertical; W horizontal, L vertical",
            "supplement": "Supplementary Table 1, M2",
            "M2_nm": {"P1": 5100, "P2": 2600, "L1": 2300, "L2": 1700, "W1": 1360, "W2": 1100, "D_Al2O3": 200},
        },
        "legacy_v2_disposition": "WRONG_PERIOD_ASSIGNMENT_DIAGNOSTIC_ONLY",
        "v2": v2.as_dict(),
        "v3": v3.as_dict(),
        "unresolved": "exact relative bar offset and author CAD are not published; v3 is an edge-joined figure-constrained closure",
    }
    (OUT / "Z2022_M2_FIGURE_PERIOD_CORRECTED_V3.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "Z2022_M2_FIGURE_PERIOD_CORRECTED_V3_REPORT.md").write_text(
        "# Z2022 M2 figure/period correction\n\n"
        "Status: `READY_Z2022_M2_FIGURE_PERIOD_CORRECTED_V3`\n\n"
        "Fig. 1b defines `P1=5.1 um` horizontally and `P2=2.6 um` vertically. "
        "The earlier v2 calculation reversed those periods and is retained only as a failed geometry diagnostic. "
        "V3 uses the published W/L dimensions and the correct periods. Since the exact relative bar offset/CAD is not disclosed, "
        "the bars use a named edge-joined, figure-constrained closure rather than claiming exact author CAD.\n\n"
        "![v2 versus v3](Z2022_M2_v2_vs_figure_period_corrected_v3.png)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
