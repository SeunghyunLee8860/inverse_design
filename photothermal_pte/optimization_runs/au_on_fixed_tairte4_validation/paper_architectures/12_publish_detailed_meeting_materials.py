#!/usr/bin/env python3
"""Add detailed setup, field, section, and comparison plots for meetings."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
import numpy as np


HERE = Path(__file__).resolve().parent
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_architectures")
CASES = {
    "T_Eb": RAW_ROOT / "T2024_MIR_4750_xb_forward",
    "T_Ea": RAW_ROOT / "T2024_MIR_4750_ya_forward",
    "bare_Eb": RAW_ROOT / "T2024_MIR_4750_bare_xb_forward",
    "bare_Ea": RAW_ROOT / "T2024_MIR_4750_bare_ya_forward",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def polygon_mask(x: np.ndarray, y: np.ndarray, vertices_nm: list[list[float]]) -> np.ndarray:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    vertices = np.asarray(vertices_nm, float) * 1e-9
    return MplPath(vertices, closed=True).contains_points(points, radius=1e-15).reshape(xx.shape)


def setup_figure(path: Path, result: dict) -> None:
    geometry = result["contract"]["geometry"]
    with_t = result["contract"].get("top_Au_T_present", True)
    pol = result["contract"]["source"]["polarization"]
    vertices = np.asarray(geometry["polygons"][0]["vertices_nm"], float)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.add_patch(Rectangle((-750, -500), 1500, 1000, facecolor="#d14e4e", alpha=0.3, edgecolor="#8d2020", linewidth=2, label="TaIrTe4 100 nm"))
    if with_t:
        ax.fill(vertices[:, 0], vertices[:, 1], color="#f6c34b", edgecolor="#8a5a00", linewidth=2, label="Au inverse-T 33 nm")
    if pol == "x_b":
        ax.annotate("", xy=(450, 420), xytext=(-450, 420), arrowprops={"arrowstyle": "->", "lw": 2.8, "color": "#315da7"})
        ax.text(0, 438, r"$E\parallel b$ (x)", ha="center", color="#315da7")
    else:
        ax.annotate("", xy=(-660, 300), xytext=(-660, -300), arrowprops={"arrowstyle": "->", "lw": 2.8, "color": "#b23b5b"})
        ax.text(-640, 0, r"$E\parallel a$ (y)", rotation=90, va="center", color="#b23b5b")
    ax.set_xlim(-780, 780); ax.set_ylim(-530, 530); ax.set_aspect("equal")
    ax.set_xlabel("x=b (nm)"); ax.set_ylabel("y=a (nm)"); ax.set_title("xy top view: periodic unit cell")
    ax.legend(fontsize=8, loc="lower right")

    def section(axis, lateral: str) -> None:
        span = 1500 if lateral == "x" else 1000
        lo, hi = -span / 2, span / 2
        axis.add_patch(Rectangle((lo, -1000), span, 765, color="#b98a00", label="Au mirror: numerical PML extension"))
        axis.add_patch(Rectangle((lo, -235), span, 200, color="#d39b00", label="physical 200-nm Au mirror"))
        axis.add_patch(Rectangle((lo, -35), span, 35, color="#8fc7dc", label="Al2O3 35 nm"))
        axis.add_patch(Rectangle((lo, 0), span, 100, color="#cf4e4e", label="TaIrTe4 100 nm"))
        if with_t:
            if lateral == "x":
                ranges = [(-100, 100)]  # y=0 intersects only the T stem
            else:
                ranges = [(-350, 350)]  # x=0 intersects stem plus crossbar continuously
            for left, right in ranges:
                axis.add_patch(Rectangle((left, 100), right-left, 33, color="#f6c34b", label="Au T 33 nm"))
        axis.axhline(800, color="#d63b32", lw=2, label="plane-wave source z=800 nm")
        axis.axhline(450, color="#3b8c5a", lw=1.5, ls="--", label="saved top-field monitor z=450 nm")
        axis.axhline(-1000, color="#7a2c91", lw=4, label="z-min PML boundary")
        axis.axhline(1200, color="#7a2c91", lw=4, label="z-max PML boundary")
        axis.axvline(lo, color="#333333", ls=":", lw=2)
        axis.axvline(hi, color="#333333", ls=":", lw=2, label=f"{lateral} periodic boundary")
        axis.annotate("k=-z", xy=(0, 200), xytext=(0, 650), ha="center", arrowprops={"arrowstyle": "->", "lw": 2})
        axis.set_xlim(lo, hi); axis.set_ylim(-1050, 1250)
        axis.set_xlabel(f"{lateral} ({'b' if lateral == 'x' else 'a'}) (nm)")
        axis.set_ylabel("z (nm)")
        axis.set_title(f"{lateral}z section at the other coordinate = 0")

    section(axes[1], "x")
    section(axes[2], "y")
    handles, labels = axes[2].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[2].legend(unique.values(), unique.keys(), fontsize=6.5, loc="upper right")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def top_field_figure(path: Path, raw: np.lib.npyio.NpzFile, key: str) -> None:
    x = np.asarray(raw["top_x_m"], float).reshape(-1)
    y = np.asarray(raw["top_y_m"], float).reshape(-1)
    components = {c: np.abs(np.asarray(raw[f"top_E{c}"])) ** 2 for c in "xyz"}
    total = sum(components.values())
    scale = max(float(np.max(total)), 1e-300)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.7), constrained_layout=True)
    for ax, (label, values) in zip(axes.ravel(), [*components.items(), ("total", total)]):
        image = ax.pcolormesh(x*1e9, y*1e9, (values/scale).T, shading="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_aspect("equal"); ax.set_xlabel("x=b (nm)"); ax.set_ylabel("y=a (nm)")
        ax.set_title(f"{key}: |E{label}|^2 / max(|E|^2)" if label != "total" else f"{key}: |E|^2 / max(|E|^2)")
        fig.colorbar(image, ax=ax, label="relative total-field intensity")
    fig.suptitle("z=450 nm total-field monitor (incident + reflected/scattered; not a pure incident field)")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def q_sections_figure(path: Path, remap: dict, key: str, remap_module) -> None:
    x, y, z = remap["x"], remap["y"], remap["z"]
    q = remap["total"]
    qxy = np.einsum("k,ijk->ij", remap_module.trapezoid_weights(z), q, optimize=True)
    iy = int(np.argmin(np.abs(y)))
    ix = int(np.argmin(np.abs(x)))
    qxz = q[:, iy, :]
    qyz = q[ix, :, :]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    image = axes[0].pcolormesh(x*1e9, y*1e9, qxy.T, shading="auto", cmap="inferno", vmin=0)
    axes[0].set_aspect("equal"); axes[0].set_xlabel("x=b (nm)"); axes[0].set_ylabel("y=a (nm)"); axes[0].set_title("xy depth-integrated Qtotal")
    fig.colorbar(image, ax=axes[0], label="W/m$^2$")
    vmax = max(float(np.max(qxz)), float(np.max(qyz)))
    image = axes[1].pcolormesh(x*1e9, z*1e9, qxz.T, shading="auto", cmap="inferno", vmin=0, vmax=vmax)
    axes[1].set_xlabel("x=b (nm)"); axes[1].set_ylabel("z (nm)"); axes[1].set_title(f"xz at y={y[iy]*1e9:.1f} nm")
    fig.colorbar(image, ax=axes[1], label="W/m$^3$")
    image = axes[2].pcolormesh(y*1e9, z*1e9, qyz.T, shading="auto", cmap="inferno", vmin=0, vmax=vmax)
    axes[2].set_xlabel("y=a (nm)"); axes[2].set_ylabel("z (nm)"); axes[2].set_title(f"yz at x={x[ix]*1e9:.1f} nm")
    fig.colorbar(image, ax=axes[2], label="W/m$^3$")
    fig.suptitle(f"{key}: conservative common-grid Qtotal sections")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def depth_profile_figure(path: Path, remap: dict, key: str, remap_module) -> None:
    wx = remap_module.trapezoid_weights(remap["x"])
    wy = remap_module.trapezoid_weights(remap["y"])
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    total_profile = np.zeros_like(remap["z"])
    for component, color in zip("xyz", ("#4477aa", "#cc6677", "#228833")):
        profile = np.einsum("i,j,ijk->k", wx, wy, remap["components"][component], optimize=True)
        total_profile += profile
        ax.plot(remap["z"]*1e9, profile, label=f"Q{component}", color=color)
    ax.plot(remap["z"]*1e9, total_profile, color="black", lw=2.2, label="Qtotal")
    for location, label in ((-235, "mirror bottom"), (-35, "Al2O3 bottom"), (0, "TaIrTe4 bottom"), (100, "TaIrTe4 top"), (133, "Au T top")):
        ax.axvline(location, color="#888888", ls=":", lw=0.8)
        ax.text(location, ax.get_ylim()[1]*0.98, label, rotation=90, va="top", ha="right", fontsize=7)
    ax.set_xlim(-300, 180)
    ax.set_xlabel("z (nm)"); ax.set_ylabel("dP/dz after xy integration (W/m)")
    ax.set_title(f"{key}: component-resolved absorption-depth profile")
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def material_map_figure(path: Path, remap: dict, result: dict, key: str, remap_module) -> None:
    x, y, z = remap["x"], remap["y"], remap["z"]
    q = remap["total"]
    geometry = result["contract"]["geometry"]
    txy = polygon_mask(x, y, geometry["polygons"][0]["vertices_nm"])
    masks = {
        "TaIrTe4 geometric z=[0,100) nm": np.broadcast_to(((z >= 0) & (z < 100e-9))[None,None,:], q.shape),
        "top-T geometric volume": txy[:,:,None] & ((z[None,None,:] >= 100e-9) & (z[None,None,:] <= 133e-9)),
        "Au mirror geometric z<=-35 nm": np.broadcast_to((z <= -35e-9)[None,None,:], q.shape),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)
    for ax, (label, mask) in zip(axes, masks.items()):
        values = np.einsum("k,ijk->ij", remap_module.trapezoid_weights(z), q*mask, optimize=True)
        image = ax.pcolormesh(x*1e9, y*1e9, values.T, shading="auto", cmap="inferno", vmin=0)
        ax.set_aspect("equal"); ax.set_xlabel("x=b (nm)"); ax.set_ylabel("y=a (nm)"); ax.set_title(label)
        fig.colorbar(image, ax=ax, label="depth-integrated W/m$^2$")
    fig.suptitle(f"{key}: geometric material/support Q maps (interface ambiguity retained)")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def comparison_figure(path: Path, left: dict, right: dict, left_label: str, right_label: str, title: str, remap_module) -> None:
    for axis in "xyz":
        if not np.array_equal(left[axis], right[axis]):
            raise RuntimeError(f"comparison grid differs on {axis}")
    ql = np.einsum("k,ijk->ij", remap_module.trapezoid_weights(left["z"]), left["total"], optimize=True)
    qr = np.einsum("k,ijk->ij", remap_module.trapezoid_weights(right["z"]), right["total"], optimize=True)
    diff = ql - qr
    vmax = max(float(np.max(ql)), float(np.max(qr)))
    dmax = max(float(np.max(np.abs(diff))), 1e-300)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)
    for ax, values, label in ((axes[0], ql, left_label), (axes[1], qr, right_label)):
        image = ax.pcolormesh(left["x"]*1e9, left["y"]*1e9, values.T, shading="auto", cmap="inferno", vmin=0, vmax=vmax)
        ax.set_aspect("equal"); ax.set_xlabel("x=b (nm)"); ax.set_ylabel("y=a (nm)"); ax.set_title(label)
        fig.colorbar(image, ax=ax, label="Qtotal (W/m$^2$)")
    image = axes[2].pcolormesh(left["x"]*1e9, left["y"]*1e9, diff.T, shading="auto", cmap="coolwarm", norm=TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax))
    axes[2].set_aspect("equal"); axes[2].set_xlabel("x=b (nm)"); axes[2].set_ylabel("y=a (nm)"); axes[2].set_title(f"{left_label} - {right_label}")
    fig.colorbar(image, ax=axes[2], label="Qtotal difference (W/m$^2$)")
    fig.suptitle(title)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def linecut_figure(path: Path, cases: list[tuple[str, dict]], title: str, remap_module) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for label, remap in cases:
        qxy = np.einsum("k,ijk->ij", remap_module.trapezoid_weights(remap["z"]), remap["total"], optimize=True)
        ix = int(np.argmin(np.abs(remap["x"])))
        iy = int(np.argmin(np.abs(remap["y"])))
        axes[0].plot(remap["x"]*1e9, qxy[:,iy], label=label)
        axes[1].plot(remap["y"]*1e9, qxy[ix,:], label=label)
    axes[0].set_xlabel("x=b at y=0 (nm)"); axes[1].set_xlabel("y=a at x=0 (nm)")
    for ax in axes:
        ax.set_ylabel("depth-integrated Qtotal (W/m$^2$)"); ax.grid(alpha=0.25); ax.legend()
    fig.suptitle(title)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def scalar_summary_figure(path: Path, rows: list[dict]) -> None:
    """Publish a slide-ready four-case scalar and component comparison."""
    labels = [row["case"] for row in rows]
    x = np.arange(len(rows))
    total = np.asarray([row["P_Q_periodic_W"] for row in rows]) * 1e15
    tairte4 = np.asarray([row["TaIrTe4_geometric_W"] for row in rows]) * 1e15
    components = np.asarray(
        [[row["Qx_W"], row["Qy_W"], row["Qz_W"]] for row in rows], float
    )
    components /= np.maximum(components.sum(axis=1, keepdims=True), 1e-300)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    width = 0.38
    axes[0].bar(x-width/2, total, width, label="all-material total", color="#4c78a8")
    axes[0].bar(x+width/2, tairte4, width, label="geometric TaIrTe4", color="#d45555")
    axes[0].set_xticks(x, labels, rotation=20)
    axes[0].set_ylabel("absorbed power (fW/cell)")
    axes[0].set_title("Raw absorbed power; equal source power")
    axes[0].legend(fontsize=8)

    bottom = np.zeros(len(rows))
    for index, (name, color) in enumerate(zip(("Qx", "Qy", "Qz"), ("#4477aa", "#cc6677", "#228833"))):
        axes[1].bar(x, components[:, index], bottom=bottom, label=name, color=color)
        bottom += components[:, index]
    axes[1].set_xticks(x, labels, rotation=20)
    axes[1].set_ylabel("fraction of native component power")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Component-resolved loss participation")
    axes[1].legend(fontsize=8)

    axes[2].axis("off")
    axes[2].text(
        0.02,
        0.96,
        "Inverse-T effect at 4.75 um",
        fontsize=15,
        weight="bold",
        va="top",
    )
    axes[2].text(
        0.02,
        0.82,
        "E||b:\n  total Q:  +10.266%\n  TaIrTe4 Q: +6.085%\n\n"
        "E||a:\n  total Q:   -5.415%\n  TaIrTe4 Q: -8.342%\n\n"
        "bare total Eb/Ea = 0.96831\n"
        "with-T total Eb/Ea = 1.12885\n"
        "with-T TaIrTe4 Eb/Ea = 1.09449",
        fontsize=12,
        family="monospace",
        va="top",
    )
    axes[2].text(
        0.02,
        0.12,
        "Optical-forward conclusion only.\nNo thermal/PTE/adjoint claim.",
        fontsize=11,
        color="#9d2020",
        weight="bold",
    )
    fig.suptitle("Paper-derived inverse-T with fixed TaIrTe4 substitution")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def publish_manifest(package_root: Path) -> None:
    """Hash every published meeting asset, excluding the manifest itself."""
    manifest_path = package_root / "MEETING_PLOT_MANIFEST.json"
    artifacts = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        artifacts.append({
            "path": str(path.relative_to(HERE)),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    manifest_path.write_text(json.dumps({
        "generation_commands": [
            "/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/paper_architectures/11_publish_meeting_plot_package.py",
            "/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/paper_architectures/12_publish_detailed_meeting_materials.py",
        ],
        "published_artifacts": artifacts,
        "raw_artifacts_committed": False,
    }, indent=2) + "\n")


def main() -> int:
    package_root = HERE / "results_actual_metasurfaces" / "meeting_plot_package"
    comparison_root = package_root / "comparisons"
    comparison_root.mkdir(parents=True, exist_ok=True)
    comparison = load_module(HERE / "09_compare_t2024_tairte4_polarizations.py", "detailed_comparison_helpers")
    helpers = comparison.load_summary_helpers()
    meeting = load_module(HERE / "11_publish_meeting_plot_package.py", "meeting_package_helpers")
    remap_module = load_module(
        HERE.parents[3] / "photothermal_pte" / "validation" / "photothermal_stage1" / "13_analyze_native_yee_components.py",
        "detailed_conservative_remap",
    )
    loaded, remaps = {}, {}
    rows = []
    for key, raw_dir in CASES.items():
        case = comparison.load_case(raw_dir, helpers)
        remap = meeting.common_grid_q(case["raw"], remap_module)
        loaded[key], remaps[key] = case, remap
        folder = package_root / key
        setup_figure(folder / "07_setup_xy_xz_yz.png", case["result"])
        top_field_figure(folder / "08_top_monitor_total_field_components.png", case["raw"], key)
        q_sections_figure(folder / "09_Qtotal_xy_xz_yz_sections.png", remap, key, remap_module)
        depth_profile_figure(folder / "10_Q_component_depth_profiles.png", remap, key, remap_module)
        material_map_figure(folder / "11_geometric_material_Q_maps.png", remap, case["result"], key, remap_module)
        result = case["result"]
        material = comparison.material_totals(case)
        rows.append({
            "case": key,
            "polarization": result["contract"]["source"]["polarization"],
            "top_T": result["contract"].get("top_Au_T_present", True),
            "source_power_W": result["source_power_W"],
            "P_Q_periodic_W": result["P_Q_pabs_periodic_W"],
            "P_flux_W": result["P_flux_absorbed_W"],
            "closure_percent": 100*result["closure_relative"],
            "auto_shutoff": result["log_audit"]["final_auto_shutoff"],
            "Qx_W": result["Q_component_power_native_W"]["x"],
            "Qy_W": result["Q_component_power_native_W"]["y"],
            "Qz_W": result["Q_component_power_native_W"]["z"],
            "TaIrTe4_geometric_W": material["TaIrTe4_geometric"],
            "top_T_envelope_W": material["top_Au_T_geometric"],
            "Au_mirror_W": material["Au_backplane_geometric"],
            "interface_other_W": material["unassigned_interface_or_other"],
            "common_grid_power_error": remap["relative_power_error"],
            "GPU_wall_time_s": result["solver_wall_time_s"],
        })

    comparison_figure(comparison_root / "01_Ea_T_vs_bare_Qtotal.png", remaps["T_Ea"], remaps["bare_Ea"], "T, E||a", "bare, E||a", "E||a: inverse-T effect on total volumetric heat-source shape", remap_module)
    comparison_figure(comparison_root / "02_Eb_T_vs_bare_Qtotal.png", remaps["T_Eb"], remaps["bare_Eb"], "T, E||b", "bare, E||b", "E||b: inverse-T effect on total volumetric heat-source shape", remap_module)
    comparison_figure(comparison_root / "03_T_Eb_vs_Ea_Qtotal.png", remaps["T_Eb"], remaps["T_Ea"], "T, E||b", "T, E||a", "inverse-T polarization comparison at equal source power", remap_module)
    linecut_figure(comparison_root / "04_Ea_T_vs_bare_center_linecuts.png", [("T, E||a", remaps["T_Ea"]), ("bare, E||a", remaps["bare_Ea"])], "E||a center linecuts", remap_module)
    linecut_figure(comparison_root / "05_Eb_T_vs_bare_center_linecuts.png", [("T, E||b", remaps["T_Eb"]), ("bare, E||b", remaps["bare_Eb"])], "E||b center linecuts", remap_module)

    csv_path = package_root / "all_case_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    scalar_summary_figure(package_root / "00_four_case_scalar_summary.png", rows)

    package_index = {
        "status": "PUBLISHED_DETAILED_T2024_MEETING_PACKAGE",
        "scope": "single-wavelength optical forward only; no thermal/PTE/adjoint claim",
        "coordinate_contract": "Lumerical x=TaIrTe4 b, y=TaIrTe4 a, z=c with epsilon_c=epsilon_b closure",
        "cases": {},
        "comparisons": [
            "comparisons/01_Ea_T_vs_bare_Qtotal.png",
            "comparisons/02_Eb_T_vs_bare_Qtotal.png",
            "comparisons/03_T_Eb_vs_Ea_Qtotal.png",
            "comparisons/04_Ea_T_vs_bare_center_linecuts.png",
            "comparisons/05_Eb_T_vs_bare_center_linecuts.png",
        ],
    }
    for row in rows:
        key = row["case"]
        package_index["cases"][key] = {
            "folder": f"results_actual_metasurfaces/meeting_plot_package/{key}",
            "common_grid_power_error": row["common_grid_power_error"],
            "figures": [
                "00_case_overview.png",
                "01_structure_and_source.png",
                "02_Qx_depth_integrated.png",
                "03_Qy_depth_integrated.png",
                "04_Qz_depth_integrated.png",
                "05_Qtotal_conservative_common_grid.png",
                "06_power_and_material_breakdown.png",
                "07_setup_xy_xz_yz.png",
                "08_top_monitor_total_field_components.png",
                "09_Qtotal_xy_xz_yz_sections.png",
                "10_Q_component_depth_profiles.png",
                "11_geometric_material_Q_maps.png",
            ],
        }
    (package_root / "meeting_plot_package.json").write_text(json.dumps(package_index, indent=2) + "\n")

    guide = package_root / "MEETING_GUIDE.md"
    guide.write_text(
        """# Meeting guide — paper-derived inverse-T with TaIrTe4 substitution

## What was actually solved

- 2024 Supplementary MIR inverse-T scenario at 4.75 um.
- Period 1500 nm x 1000 nm; normal-incidence periodic plane wave.
- Lumerical x = TaIrTe4 b, y = TaIrTe4 a, z = c with epsilon_c=epsilon_b closure.
- 100-nm TaIrTe4 / 35-nm Al2O3 / opaque Au mirror / optional 33-nm Au T.
- x/y Periodic and z PML; conformal variant 1; 10-nm x/y and 5-nm z local mesh.
- Four independent GPU forwards: T_Eb, T_Ea, bare_Eb, bare_Ea.

The T outline is digitized from Supplementary Fig. 14 axes because author CAD
vertices are not published. This is a paper-derived TaIrTe4 substitution
scenario, not a graphene-experiment reproduction.

## How to read each case folder

1. `01_structure_and_source`: top view and polarization.
2. `02--04_Qx/Qy/Qz`: native staggered-Yee component absorption. These are
   separate physical component grids.
3. `05_Qtotal`: Qx/Qy/Qz are conservatively moved to the common monitor grid
   before summation. It is not a same-index native sum.
4. `06_power_and_material_breakdown`: pabs/flux/native/common power and
   geometric material-support partition.
5. `07_setup_xy_xz_yz`: source, monitors, layers, periodic boundaries and PML.
6. `08_top_monitor_total_field_components`: total field at z=450 nm. It
   contains incident plus reflected/scattered fields and is not called the
   pure incident field.
7. `09_Qtotal_xy_xz_yz_sections`: volumetric heat-source sections.
8. `10_Q_component_depth_profiles`: where each component is absorbed in z.
9. `11_geometric_material_Q_maps`: TaIrTe4/T-envelope/mirror masks. Conformal
   interface residual is retained and never deleted or reassigned silently.

## Main numerical result

- E||b: adding the T changes total P_Q by +10.266% and geometric TaIrTe4 Q by +6.085%.
- E||a: adding the T changes total P_Q by -5.415% and geometric TaIrTe4 Q by -8.342%.
- bare total Eb/Ea = 0.96831; with T total Eb/Ea = 1.12885.
- with-T geometric TaIrTe4 Eb/Ea = 1.09449.

Thus the T creates polarization-selective active-layer absorption. This does
not yet prove a temperature-gradient or PTE-current improvement: no thermal,
electrical, adjoint, or optimization calculation is included here.

## Likely questions and precise answers

**Was only Qx plotted for E||b?**  No in this package. Qx/Qy/Qz are separate,
and Qtotal is a conservative common-grid sum. An earlier T-vs-bare diagnostic
showed only the incident-dominant component in its lower panel.

**Why is Qx dominant for E||b and Qy dominant for E||a?**  The normal incident
field is aligned with x=b or y=a, and the TaIrTe4 permittivity tensor is
diagonal in that coordinate system. The patterned T creates smaller cross and
out-of-plane components, which are retained.

**Is all absorption inside TaIrTe4?**  No. The report separates geometric
TaIrTe4, the T-envelope/interface, Au mirror and unresolved conformal-interface
power. The primary total certificate is pabs versus flux closure.

**Can Qx/Qy/Qz be added directly by array index?**  No. Their longitudinal
coordinates are staggered. The published Qtotal uses conservative deposition;
the measured power error is approximately machine precision.

**Is the 2022 Z result included?**  No Maxwell result yet. The paper supplies
L/W/P/D but not a unique arm offset/junction CAD, so it remains fail-closed.

**Can these results be called PTE enhancement?**  Not yet. They establish an
optical heat-source effect. Explicit thermal and electrical solves are the
next separate gates.

## Suggested slide order

1. `00_four_case_scalar_summary.png`: state the conclusion and the strict
   optical-only scope.
2. `T_Ea/07_setup_xy_xz_yz.png`: establish the complete E||a simulation
   contract before showing a result.
3. `T_Ea/00_case_overview.png`: show that all three loss components were
   retained and that Qtotal is conservative.
4. `comparisons/01_Ea_T_vs_bare_Qtotal.png`: explain that the T suppresses,
   rather than enhances, E||a absorption at this wavelength.
5. `T_Ea/09_Qtotal_xy_xz_yz_sections.png` and
   `T_Ea/10_Q_component_depth_profiles.png`: show the volumetric/depth
   evidence behind the scalar comparison.
6. `comparisons/03_T_Eb_vs_Ea_Qtotal.png`: close with polarization selectivity.

Do not call the top-monitor field a pure incident beam, do not call geometric
material masks an exact conformal material decomposition, and do not claim a
PTE-current enhancement from these optical figures.
"""
    )
    publish_manifest(package_root)
    print(json.dumps({"status": "PUBLISHED_DETAILED_MEETING_MATERIALS", "case_count": len(rows), "comparison_figures": 5}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
