#!/usr/bin/env python3
"""Publish solver-free geometry audits for the actual T/Z paper controls."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_actual_metasurfaces"


def load_geometry_module():
    path = HERE / "05_actual_metasurface_geometry.py"
    spec = importlib.util.spec_from_file_location("paper_actual_geometry", path)
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


def draw_top_view(ax, geometry, *, diagnostic: bool) -> None:
    px = geometry.period_x_nm
    py = geometry.period_y_nm
    ax.add_patch(
        Rectangle(
            (-0.5 * px, -0.5 * py),
            px,
            py,
            facecolor="#deedf5",
            edgecolor="#3a6f8f",
            linewidth=1.5,
            label="unit cell",
        )
    )
    for index, obj in enumerate(geometry.polygons):
        ax.add_patch(
            Polygon(
                obj.vertices_nm,
                closed=True,
                facecolor="#f6c64e" if not diagnostic else "none",
                edgecolor="#a56a00" if not diagnostic else "#b22222",
                hatch=None if not diagnostic else "///",
                linewidth=2.0,
                label=("Au resonator" if not diagnostic else "dimension envelope only")
                if index == 0
                else None,
            )
        )
    ax.set_aspect("equal")
    ax.set_xlim(-0.55 * px, 0.55 * px)
    ax.set_ylim(-0.55 * py, 0.55 * py)
    ax.set_xlabel("x = TaIrTe$_4$ b (nm)")
    ax.set_ylabel("y = TaIrTe$_4$ a (nm)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(geometry.key.replace("_", " "), fontsize=10)


def draw_t_cross_section(ax, geometry) -> None:
    px = geometry.period_x_nm
    layers = [
        (-235.0, -35.0, "Au mirror", "#d29b00"),
        (-35.0, 0.0, "Al$_2$O$_3$ 35 nm", "#9ccfe3"),
        (0.0, 100.0, "TaIrTe$_4$ 100 nm", "#cf4d4d"),
        (100.0, 133.0, "Au T 33 nm", "#f6c64e"),
    ]
    for z0, z1, label, color in layers[:3]:
        ax.add_patch(Rectangle((-0.5 * px, z0), px, z1 - z0, color=color, label=label))
    ax.add_patch(Rectangle((-600.0, 100.0), 1200.0, 33.0, color=layers[3][3], label=layers[3][2]))
    ax.axhspan(133.0, 700.0, color="#edf7ff", label="air")
    ax.annotate("normal-incidence plane wave", xy=(0, 220), xytext=(0, 580), ha="center", arrowprops={"arrowstyle": "->"})
    ax.set_xlim(-0.55 * px, 0.55 * px)
    ax.set_ylim(-250.0, 700.0)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("z (nm)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("T control x-z layer stack (not to scale in z)", fontsize=10)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    module = load_geometry_module()
    t = module.inverse_t_mir_4750nm()
    z = module.z_m5_8um_geometry_topology_audit()
    payload = {
        "status": "READY_T2024_GEOMETRY_AUDIT_BLOCKED_Z2022_TOPOLOGY",
        "T2024": t.as_dict(),
        "Z2022": z.as_dict(),
        "polygon_areas_nm2": {
            "T2024": [
                abs(module.signed_polygon_area_nm2(item.vertices_nm))
                for item in t.polygons
            ],
            "Z2022_dimension_envelopes_only": [
                abs(module.signed_polygon_area_nm2(item.vertices_nm))
                for item in z.polygons
            ],
        },
        "decision": {
            "T2024": "allowed for GPU smoke as a figure-digitized MIR control",
            "Z2022": "fail closed before Maxwell until exact Z topology is recovered or an explicit approximation is approved",
        },
    }
    json_path = RESULTS / "actual_metasurface_geometry_contract.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    draw_top_view(axes[0], t, diagnostic=False)
    draw_t_cross_section(axes[1], t)
    draw_top_view(axes[2], z, diagnostic=True)
    axes[2].text(
        0.5,
        0.03,
        "NOT Maxwell CAD: Table S1 supplies L/W/P/D but not polygon vertices",
        transform=axes[2].transAxes,
        color="#b22222",
        ha="center",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    plot_path = RESULTS / "actual_t_z_geometry_audit.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    manifest = {
        "artifacts": [
            {
                "path": str(path.relative_to(HERE)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (json_path, plot_path)
        ]
    }
    (RESULTS / "GEOMETRY_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
