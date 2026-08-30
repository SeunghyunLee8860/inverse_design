#!/usr/bin/env python3
"""Publish structure, near-field, and absorption plots for one T2024 case."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def positive_norm(values: np.ndarray) -> LogNorm:
    maximum = float(np.nanmax(values))
    return LogNorm(vmin=max(maximum * 1.0e-5, 1.0e-30), vmax=max(maximum, 1.0e-29))


def draw_setup(path: Path, result: dict) -> None:
    contract = result["contract"]
    geometry = contract["geometry"]
    substrate = contract["substrate"]
    vertices = np.asarray(geometry["polygons"][0]["vertices_nm"], float)
    px = float(geometry["period_x_nm"])
    py = float(geometry["period_y_nm"])
    zmin = float(substrate["domain_z_min_m"]) * 1e9
    zmax = 1200.0
    sio2 = np.asarray(substrate["SiO2_bounds_m"], float) * 1e9
    si = np.asarray(substrate["Si_bounds_m"], float) * 1e9
    au = np.asarray(substrate["Au_mirror_bounds_m"], float) * 1e9

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), constrained_layout=True)
    ax = axes[0]
    ax.add_patch(Rectangle((-px / 2, -py / 2), px, py, facecolor="#c44e52", alpha=0.30, edgecolor="#7f1d1d", lw=2, label="TaIrTe4 100 nm"))
    if contract["top_Au_T_present"]:
        ax.fill(vertices[:, 0], vertices[:, 1], color="#f2b632", edgecolor="#7f5200", lw=2, label="Au inverse-T 33 nm")
    ax.set(xlim=(-px / 2, px / 2), ylim=(-py / 2, py / 2), xlabel="x=b (nm)", ylabel="y=a (nm)", title="top view: one periodic cell")
    ax.set_aspect("equal")
    ax.text(0.03, 0.97, "x/y: periodic", transform=ax.transAxes, va="top", color="#7a2d8f", weight="bold")
    ax.legend(fontsize=8, loc="lower right")

    def cross_section(ax, horizontal: str) -> None:
        span = px if horizontal == "x" else py
        ax.add_patch(Rectangle((-span / 2, si[0]), span, si[1] - si[0], color="#8ebad9", label="Si"))
        ax.add_patch(Rectangle((-span / 2, sio2[0]), span, sio2[1] - sio2[0], color="#b7e1ed", label="SiO2 285 nm"))
        ax.add_patch(Rectangle((-span / 2, au[0]), span, au[1] - au[0], color="#bd8b00", label="Au mirror 200 nm"))
        ax.add_patch(Rectangle((-span / 2, -35), span, 35, color="#d7e3ec", label="Al2O3 35 nm"))
        ax.add_patch(Rectangle((-span / 2, 0), span, 100, color="#c44e52", alpha=0.75, label="TaIrTe4 100 nm"))
        if contract["top_Au_T_present"]:
            width = 200 if horizontal == "x" else 700
            ax.add_patch(Rectangle((-width / 2, 100), width, 33, color="#f2b632", label="Au T 33 nm"))
        ax.axhline(zmin, color="#7a2d8f", lw=5)
        ax.axhline(zmax, color="#7a2d8f", lw=5)
        ax.annotate("normal incidence\nsource z=800 nm", xy=(0, 300), xytext=(0, 850), ha="center", arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#2962a3"}, color="#2962a3")
        ax.set(xlim=(-span / 2, span / 2), ylim=(zmin, zmax), xlabel=f"{horizontal} (nm)", ylabel="z (nm)", title=f"{horizontal}z cross section; z-PML")
    cross_section(axes[1], "x")
    cross_section(axes[2], "y")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[2].legend(handles[:6], labels[:6], fontsize=7, loc="upper right")
    fig.suptitle("Reduced optical closure (physical 1.5-um oxide retained only as a control)")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_field_plane(path: Path, raw, plane: str, title: str) -> None:
    axes_for_plane = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}
    first_axis, second_axis = axes_for_plane[plane]
    first = np.asarray(raw[f"field_{plane}_{first_axis}_m"], float) * 1e9
    second = np.asarray(raw[f"field_{plane}_{second_axis}_m"], float) * 1e9
    values = [np.abs(np.asarray(raw[f"field_{plane}_E{component}"])) ** 2 for component in "xyz"]
    values.append(np.asarray(raw[f"field_{plane}_E2_V2_m2"], float))
    labels = [r"$|E_x|^2$", r"$|E_y|^2$", r"$|E_z|^2$", r"$|E|^2$"]
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.3), constrained_layout=True)
    for ax, value, label in zip(axes, values, labels):
        image = ax.pcolormesh(first, second, value.T, shading="auto", cmap="magma", norm=positive_norm(value))
        ax.set_xlabel(f"{first_axis} (nm)")
        ax.set_ylabel(f"{second_axis} (nm)")
        ax.set_title(label)
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, label=r"V$^2$/m$^2$")
    fig.suptitle(title)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_q(path: Path, raw, remap: dict, remap_module, title: str) -> None:
    x = remap["x"] * 1e9
    y = remap["y"] * 1e9
    z = remap["z"]
    component_areal = [
        np.einsum("k,ijk->ij", remap_module.trapezoid_weights(z), remap["components"][component], optimize=True)
        for component in "xyz"
    ]
    total_areal = sum(component_areal)
    values = component_areal + [total_areal]
    labels = [r"$Q_x$", r"$Q_y$", r"$Q_z$", r"$Q_{total}$"]
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.3), constrained_layout=True)
    for ax, value, label in zip(axes, values, labels):
        image = ax.pcolormesh(x, y, value.T, shading="auto", cmap="inferno", vmin=0.0)
        ax.set(xlabel="x=b (nm)", ylabel="y=a (nm)", title=label)
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, label=r"W/m$^2$")
    fig.suptitle(title)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_q_cross_sections(path: Path, remap: dict, title: str) -> None:
    total = np.asarray(remap["total"], float)
    x = np.asarray(remap["x"], float) * 1e9
    y = np.asarray(remap["y"], float) * 1e9
    z = np.asarray(remap["z"], float) * 1e9
    iy = int(np.argmin(np.abs(y)))
    ix = int(np.argmin(np.abs(x)))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for ax, first, value, label in (
        (axes[0], x, total[:, iy, :], "xz at y=0"),
        (axes[1], y, total[ix, :, :], "yz at x=0"),
    ):
        image = ax.pcolormesh(first, z, value.T, shading="auto", cmap="inferno", vmin=0.0)
        ax.set(xlabel=f"{label[0]} (nm)", ylabel="z (nm)", title=label)
        ax.set_ylim(-300, 350)
        fig.colorbar(image, ax=ax, label=r"W/m$^3$")
    fig.suptitle(title)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-case", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    raw_case = args.raw_case.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = raw_case / "T2024_TaIrTe4_optical_smoke.json"
    npz_path = raw_case / "T2024_TaIrTe4_native_q.npz"
    result = json.loads(result_path.read_text())
    if result["status"] != "COMPLETED_T2024_TAIRTE4_OPTICAL_SMOKE":
        raise RuntimeError(result["status"])
    meeting = load_module(HERE / "11_publish_meeting_plot_package.py", "reduced_field_meeting_helpers")
    remap_module = load_module(
        REPOSITORY / "photothermal_pte" / "validation" / "photothermal_stage1" / "13_analyze_native_yee_components.py",
        "reduced_field_remap",
    )
    with np.load(npz_path) as raw:
        required = {f"field_{plane}_E2_V2_m2" for plane in ("xy", "xz", "yz")}
        if not required.issubset(raw.files):
            raise RuntimeError(f"missing field slices: {sorted(required - set(raw.files))}")
        remap = meeting.common_grid_q(raw, remap_module)
        draw_setup(output / "00_reduced_substrate_setup_xy_xz_yz.png", result)
        plot_field_plane(output / "01_field_xy_tairte4_midplane.png", raw, "xy", f"{args.label}: TaIrTe4 midplane z=50 nm")
        plot_field_plane(output / "02_field_xz_y0.png", raw, "xz", f"{args.label}: near-field xz cross section at y=0")
        plot_field_plane(output / "03_field_yz_x0.png", raw, "yz", f"{args.label}: near-field yz cross section at x=0")
        plot_q(output / "04_Q_components_depth_integrated.png", raw, remap, remap_module, f"{args.label}: component-resolved absorption")
        plot_q_cross_sections(output / "05_Qtotal_xz_yz.png", remap, f"{args.label}: conservative common-grid volumetric Q")

    summary = {
        "status": "PUBLISHED_REDUCED_SUBSTRATE_NEAR_FIELD_PACKAGE",
        "case": args.label,
        "substrate": result["contract"]["substrate"],
        "mesh": result["mesh_runsetup"],
        "runtime_s": result["solver_wall_time_s"],
        "P_Q_W": result["P_Q_pabs_periodic_W"],
        "P_flux_W": result["P_flux_absorbed_W"],
        "closure_relative": result["closure_relative"],
        "auto_shutoff": result["log_audit"]["final_auto_shutoff"],
        "bottom_transmission": result["transmission_bottom_monitor"],
        "common_grid_power_error": remap["relative_power_error"],
        "field_collocation": "component-specific native Yee coordinates -> common physical-plane interpolation; no same-index pairing",
        "figures": sorted(path.name for path in output.glob("*.png")),
        "raw_npz": {"path": str(npz_path), "size_bytes": npz_path.stat().st_size, "sha256": sha256(npz_path)},
    }
    (output / "REDUCED_SUBSTRATE_FIELD_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "REDUCED_SUBSTRATE_FIELD_REPORT.md").write_text(
        f"""# Reduced optical SiO2/Si closure and near fields — {args.label}

The 1.5-um physical thermal oxide is **not** retained in this optical solve.
The optical closure uses 200-nm Au / 285-nm SiO2 / Si because the opaque Au
mirror reduces the measured bottom transmission to `{summary['bottom_transmission']:.3e}`.
The already preserved 1.5-um control established the same optical observables.

- P_Q: `{summary['P_Q_W']:.9e} W/cell`
- matched-volume closure: `{100*summary['closure_relative']:.6f}%`
- auto-shutoff: `{summary['auto_shutoff']:.3e}`
- runtime: `{summary['runtime_s']:.2f} s`
- mesh: `{summary['mesh']['shape']}`; structure dx=dy=10 nm, dz=5 nm

The plots show the actual collocated Lumerical electric fields at the TaIrTe4
midplane and through xz/yz cross sections, plus component-resolved absorption.
No clipping, smoothing, gain, or source rescaling was used.
"""
    )
    files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "RAW_ARTIFACT_MANIFEST.json")
    manifest = {
        "generation_command": f"15_publish_reduced_substrate_fields.py --raw-case {raw_case} --output-dir {output} --label {args.label}",
        "published": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
        "raw_artifacts_committed": False,
        "raw_npz": summary["raw_npz"],
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
