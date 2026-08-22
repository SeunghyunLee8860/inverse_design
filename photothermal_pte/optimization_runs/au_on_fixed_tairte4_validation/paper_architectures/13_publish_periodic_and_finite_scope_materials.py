#!/usr/bin/env python3
"""Publish seam-safe periodic plots and the finite-device scope distinction."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_architectures")
CASES = {
    "T_Ea": RAW_ROOT / "T2024_MIR_4750_ya_forward",
    "T_Eb": RAW_ROOT / "T2024_MIR_4750_xb_forward",
    "bare_Ea": RAW_ROOT / "T2024_MIR_4750_bare_ya_forward",
    "bare_Eb": RAW_ROOT / "T2024_MIR_4750_bare_xb_forward",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def periodic_canonical_display(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Remove duplicated endpoint samples for display only.

    The raw periodic pabs arrays retain both endpoint planes.  At a staggered
    material boundary, those duplicated planes can have half/zero material
    support even though the physical field is periodic.  We estimate the one
    seam value from the two inward one-sided samples and then retain only one
    copy of each endpoint.  This array is never used for power integration.
    """
    if values.shape != (x.size, y.size):
        raise ValueError((values.shape, x.shape, y.shape))
    raw = np.asarray(values, float).copy()

    def periodic_interp_axis(
        array: np.ndarray,
        source: np.ndarray,
        target: np.ndarray,
        axis: int,
        trim: int,
    ) -> np.ndarray:
        moved = np.moveaxis(array, axis, 0)
        flat = moved.reshape(moved.shape[0], -1)
        source_stable = source[trim:-trim]
        values_stable = flat[trim:-trim, :]
        period = float(source[-1] - source[0])
        source_extended = np.concatenate(
            (source_stable-period, source_stable, source_stable+period)
        )
        values_extended = np.concatenate(
            (values_stable, values_stable, values_stable), axis=0
        )
        output = np.empty((target.size, flat.shape[1]), float)
        for column in range(flat.shape[1]):
            output[:, column] = np.interp(
                target, source_extended, values_extended[:, column]
            )
        output = output.reshape((target.size,) + moved.shape[1:])
        return np.moveaxis(output, 0, axis)

    # Target the physical cell centers.  The raw common-grid endpoint audit
    # shows one affected x endpoint plane and two affected y planes.  Those
    # narrow seam samples are excluded only from this visualization, and the
    # stable one-sided interiors are periodically interpolated across them.
    x_center = 0.5 * (x[:-1] + x[1:])
    y_center = 0.5 * (y[:-1] + y[1:])
    work = periodic_interp_axis(raw, x, x_center, axis=0, trim=1)
    work = periodic_interp_axis(work, y, y_center, axis=1, trim=2)

    audit = {
        "raw_x_min_to_inward_mean_ratio": float(
            np.mean(raw[0, 1:-1]) / max(np.mean(raw[1, 1:-1]), 1e-300)
        ),
        "raw_x_max_to_inward_mean_ratio": float(
            np.mean(raw[-1, 1:-1]) / max(np.mean(raw[-2, 1:-1]), 1e-300)
        ),
        "raw_y_min_to_inward_mean_ratio": float(
            np.mean(raw[1:-1, 0]) / max(np.mean(raw[1:-1, 1]), 1e-300)
        ),
        "raw_y_max_to_inward_mean_ratio": float(
            np.mean(raw[1:-1, -1]) / max(np.mean(raw[1:-1, -2]), 1e-300)
        ),
        "display_only": True,
        "used_for_power_or_metrics": False,
        "excluded_raw_seam_planes_for_display": {"x_each_side": 1, "y_each_side": 2},
    }
    return x_center, y_center, work, audit


def periodic_map_figure(
    path: Path,
    key: str,
    x: np.ndarray,
    y: np.ndarray,
    raw: np.ndarray,
    canonical_x: np.ndarray,
    canonical_y: np.ndarray,
    canonical: np.ndarray,
) -> None:
    period_x = float(x[-1] - x[0])
    period_y = float(y[-1] - y[0])
    x_tiled = np.concatenate([canonical_x + offset * period_x for offset in (-1, 0, 1)])
    y_tiled = np.concatenate([canonical_y + offset * period_y for offset in (-1, 0, 1)])
    tiled = np.tile(canonical, (3, 3))
    vmax = max(float(np.max(raw)), float(np.max(canonical)), 1e-300)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)
    image = axes[0].pcolormesh(x*1e9, y*1e9, raw.T, shading="auto", cmap="inferno", vmin=0, vmax=vmax)
    axes[0].set_title("raw solver endpoints\n(diagnostic; duplicated seam retained)")
    image = axes[1].pcolormesh(canonical_x*1e9, canonical_y*1e9, canonical.T, shading="auto", cmap="inferno", vmin=0, vmax=vmax)
    axes[1].set_title("one periodic-canonical cell\n(display only; no power metric)")
    image = axes[2].pcolormesh(x_tiled*1e9, y_tiled*1e9, tiled.T, shading="auto", cmap="inferno", vmin=0, vmax=vmax)
    for boundary in (-1.5, -0.5, 0.5, 1.5):
        axes[2].axvline(boundary*period_x*1e9, color="white", lw=0.7, ls="--", alpha=0.7)
        axes[2].axhline(boundary*period_y*1e9, color="white", lw=0.7, ls="--", alpha=0.7)
    axes[2].set_title("3x3 periodic tiling\n(dashed lines are cell boundaries, not edges)")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("Lumerical x=b (nm)")
        axis.set_ylabel("Lumerical y=a (nm)")
    fig.colorbar(image, ax=axes, label="depth-integrated Qtotal (W/m$^2$)", shrink=0.82)
    fig.suptitle(f"{key}: periodic optical Q — raw seam versus physical tiling")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def periodic_structure_figure(path: Path, vertices: np.ndarray) -> None:
    period_x, period_y = 1500.0, 1000.0
    fig, ax = plt.subplots(figsize=(10.5, 7.0), constrained_layout=True)
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            ox, oy = ix*period_x, iy*period_y
            ax.add_patch(Rectangle((ox-period_x/2, oy-period_y/2), period_x, period_y, facecolor="#d04b4b", alpha=0.22, edgecolor="#555555", lw=0.8, ls="--"))
            shifted = vertices + np.array([ox, oy])
            ax.fill(shifted[:,0], shifted[:,1], color="#f4bf42", edgecolor="#8a5a00", lw=1.4)
    ax.text(0, 1050, r"$\otimes\quad k=-z$", fontsize=18, ha="center", va="center")
    ax.text(0, 930, "normal incidence (into page)", fontsize=11, ha="center")
    ax.set_xlim(-2250, 2250); ax.set_ylim(-1500, 1500); ax.set_aspect("equal")
    ax.set_xlabel("Lumerical x=b (nm)"); ax.set_ylabel("Lumerical y=a (nm)")
    ax.set_title("2024 paper-derived inverse-T architecture is an infinite periodic optical array")
    ax.text(0.5, -0.08, "Dashed rectangles are unit-cell boundaries, not physical material edges.", transform=ax.transAxes, ha="center", color="#8d2020", weight="bold")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def finite_scope_figure(path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), constrained_layout=True)

    ax = axes[0]
    ax.add_patch(Rectangle((-1.0, -0.72), 2.0, 1.44, facecolor="#d04b4b", alpha=0.28, edgecolor="#8d2020", lw=2, label="finite TaIrTe4 flake"))
    vertices = np.array([[-0.62,-0.35],[0.62,-0.35],[0.62,-0.25],[0.10,-0.25],[0.10,0.35],[-0.10,0.35],[-0.10,-0.25],[-0.62,-0.25]])
    ax.fill(vertices[:,0], vertices[:,1], color="#f4bf42", edgecolor="#8a5a00", lw=2, label="one Au inverse-T")
    circle = plt.Circle((0,0), 0.54, fill=False, color="#2867b2", lw=2, ls="--", label="finite Gaussian/TFSF footprint")
    ax.add_patch(circle)
    for coordinate in (-1.22, 1.22):
        ax.axvline(coordinate, color="#7a2c91", lw=4)
        ax.axhline(coordinate*0.75, color="#7a2c91", lw=4)
    ax.set_xlim(-1.3,1.3); ax.set_ylim(-0.98,0.98); ax.set_aspect("equal")
    ax.set_title("Required finite Maxwell certificate")
    ax.set_xlabel("x=b (finite; x-PML)"); ax.set_ylabel("y=a (finite; y-PML)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.add_patch(Rectangle((-1.0,-0.72),2.0,1.44,facecolor="#d04b4b",alpha=0.25,edgecolor="#8d2020",lw=2,label="finite conducting flake"))
    ax.add_patch(Rectangle((-1.0,0.58),2.0,0.14,facecolor="#d6a000",label=r"electrode: $\psi=1$"))
    ax.add_patch(Rectangle((-1.0,-0.72),2.0,0.14,facecolor="#7d6200",label=r"electrode: $\psi=0$"))
    ax.annotate("", xy=(0,0.45), xytext=(0,-0.45), arrowprops={"arrowstyle":"->","lw":3,"color":"#3c2d90"})
    ax.text(0.06,0,r"$\mathbf{E}_w=-\nabla\psi$",color="#3c2d90",fontsize=13)
    ax.text(0,-0.93,"Thermal/electrical domains use physical BCs, not optical PML.",ha="center",color="#8d2020",weight="bold")
    ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.02,1.02); ax.set_aspect("equal")
    ax.set_title("Required finite thermal/electrical certificate")
    ax.set_xlabel("finite device x=b"); ax.set_ylabel("finite device y=a")
    ax.legend(fontsize=8,loc="upper right")
    fig.suptitle("Periodic absorptance screening and finite-device PTE are different problems")
    fig.savefig(path,dpi=220)
    plt.close(fig)


def main() -> int:
    package_root = HERE / "results_actual_metasurfaces" / "meeting_plot_package"
    output = package_root / "periodic_canonical"
    output.mkdir(parents=True, exist_ok=True)
    comparison = load_module(HERE / "09_compare_t2024_tairte4_polarizations.py", "periodic_comparison")
    helpers = comparison.load_summary_helpers()
    meeting = load_module(HERE / "11_publish_meeting_plot_package.py", "periodic_meeting")
    remap_module = load_module(
        HERE.parents[3] / "photothermal_pte" / "validation" / "photothermal_stage1" / "13_analyze_native_yee_components.py",
        "periodic_remap",
    )
    audits = {}
    for index, (key, raw_dir) in enumerate(CASES.items(), start=1):
        case = comparison.load_case(raw_dir, helpers)
        remap = meeting.common_grid_q(case["raw"], remap_module)
        raw_areal = meeting.areal(remap["total"], remap["z"], remap_module)
        x, y, canonical, audit = periodic_canonical_display(remap["x"], remap["y"], raw_areal)
        audit.update({
            "raw_x_bounds_nm": [float(remap["x"][0]*1e9), float(remap["x"][-1]*1e9)],
            "raw_y_bounds_nm": [float(remap["y"][0]*1e9), float(remap["y"][-1]*1e9)],
            "canonical_x_sample_count": int(x.size),
            "canonical_y_sample_count": int(y.size),
            "raw_common_component_sum_power_W": float(remap["total_common_power_W"]),
        })
        audits[key] = audit
        periodic_map_figure(
            output / f"0{index}_{key}_periodic_Qtotal.png",
            key,
            remap["x"],
            remap["y"],
            raw_areal,
            x,
            y,
            canonical,
        )

    geometry = comparison.load_case(CASES["T_Ea"], helpers)["result"]["contract"]["geometry"]
    periodic_structure_figure(output / "00_periodic_structure_3x3.png", np.asarray(geometry["polygons"][0]["vertices_nm"], float))
    finite_scope_figure(output / "05_periodic_vs_finite_device_scope.png")
    (output / "PERIODIC_SEAM_AUDIT.json").write_text(json.dumps({
        "status": "PUBLISHED_PERIODIC_CANONICAL_DISPLAY_WITH_RAW_SEAM_PRESERVED",
        "warning": "canonical/tiled maps are display-only; all power metrics remain raw conservative integrals",
        "cases": audits,
    }, indent=2) + "\n")
    print(json.dumps({"status": "PUBLISHED_PERIODIC_AND_FINITE_SCOPE_MATERIALS", "cases": list(CASES)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
