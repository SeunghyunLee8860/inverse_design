#!/usr/bin/env python3
"""Publish one unambiguous meeting-plot folder per optical case."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_architectures")
CASES = {
    "T_Eb": RAW_ROOT / "T2024_MIR_4750_xb_forward",
    "T_Ea": RAW_ROOT / "T2024_MIR_4750_ya_forward",
    "bare_Eb": RAW_ROOT / "T2024_MIR_4750_bare_xb_forward",
    "bare_Ea": RAW_ROOT / "T2024_MIR_4750_bare_ya_forward",
}


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


def common_grid_q(raw: np.lib.npyio.NpzFile, remap_module) -> dict:
    # On a Yee grid only the coordinate parallel to the field component is
    # staggered.  The two transverse coordinates are the base monitor grid.
    x = np.asarray(raw["Qy_x_m"], float)
    y = np.asarray(raw["Qx_y_m"], float)
    z = np.asarray(raw["Qx_z_m"], float)
    checks = {
        "Qz_x_equals_common_x": bool(np.array_equal(raw["Qz_x_m"], x)),
        "Qz_y_equals_common_y": bool(np.array_equal(raw["Qz_y_m"], y)),
        "Qy_z_equals_common_z": bool(np.array_equal(raw["Qy_z_m"], z)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"transverse Yee coordinate contract changed: {checks}")

    common = {}
    native_power = {}
    common_power = {}
    for axis_index, component in enumerate("xyz"):
        q = np.asarray(raw[f"Q{component}_W_m3"], float)
        source_coordinate = np.asarray(raw[f"Q{component}_{component}_m"], float)
        target_coordinate = (x, y, z)[axis_index]
        q_common = remap_module.conservative_remap_axis(
            q,
            source_coordinate,
            target_coordinate,
            axis=axis_index,
            periodic=component in "xy",
        )
        native_coordinates = tuple(
            np.asarray(raw[f"Q{component}_{axis}_m"], float) for axis in "xyz"
        )
        native_power[component] = remap_module.integrate_xyz(q, *native_coordinates)
        common_power[component] = remap_module.integrate_xyz(q_common, x, y, z)
        common[component] = q_common
    total = common["x"] + common["y"] + common["z"]
    total_native = float(sum(native_power.values()))
    total_common = float(sum(common_power.values()))
    return {
        "x": x,
        "y": y,
        "z": z,
        "components": common,
        "total": total,
        "native_power_W": native_power,
        "common_power_W": common_power,
        "total_native_power_W": total_native,
        "total_common_power_W": total_common,
        "relative_power_error": abs(total_common - total_native) / max(abs(total_native), 1e-300),
        "coordinate_checks": checks,
    }


def areal(values: np.ndarray, z: np.ndarray, remap_module) -> np.ndarray:
    return np.einsum(
        "k,ijk->ij", remap_module.trapezoid_weights(z), values, optimize=True
    )


def save_map(path: Path, x: np.ndarray, y: np.ndarray, values: np.ndarray, title: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
    image = ax.pcolormesh(x * 1e9, y * 1e9, values.T, shading="auto", cmap="inferno", vmin=0.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Lumerical x = TaIrTe$_4$ b (nm)")
    ax.set_ylabel("Lumerical y = TaIrTe$_4$ a (nm)")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=label)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_structure(path: Path, result: dict) -> None:
    geometry = result["contract"]["geometry"]
    with_t = result["contract"].get("top_Au_T_present", True)
    polarization = result["contract"]["source"]["polarization"]
    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    px = geometry["period_x_nm"]
    py = geometry["period_y_nm"]
    ax.add_patch(Rectangle((-px / 2, -py / 2), px, py, facecolor="#d04b4b", alpha=0.28, edgecolor="#8d2020", linewidth=2, label="100-nm TaIrTe$_4$"))
    if with_t:
        vertices = np.asarray(geometry["polygons"][0]["vertices_nm"], float)
        ax.fill(vertices[:, 0], vertices[:, 1], color="#f4bf42", edgecolor="#8a5a00", linewidth=2.5, label="33-nm top Au inverse-T")
    else:
        ax.text(0.5, 0.5, "top Au T removed", transform=ax.transAxes, ha="center", va="center", fontsize=13, color="#8a5a00")
    if polarization == "x_b":
        ax.annotate("", xy=(450, 400), xytext=(-450, 400), arrowprops={"arrowstyle": "->", "lw": 3, "color": "#325ea8"})
        ax.text(0, 425, r"$E\parallel b$ (x)", ha="center", color="#325ea8", fontsize=12)
    else:
        ax.annotate("", xy=(-650, 300), xytext=(-650, -300), arrowprops={"arrowstyle": "->", "lw": 3, "color": "#b13c5a"})
        ax.text(-625, 0, r"$E\parallel a$ (y)", rotation=90, va="center", color="#b13c5a", fontsize=12)
    ax.text(0.98, 0.03, r"normal incidence, $k=-z$", transform=ax.transAxes, ha="right", fontsize=10)
    ax.set_xlim(-px / 2 - 50, px / 2 + 50)
    ax.set_ylim(-py / 2 - 50, py / 2 + 50)
    ax.set_aspect("equal")
    ax.set_xlabel("Lumerical x = TaIrTe$_4$ b (nm)")
    ax.set_ylabel("Lumerical y = TaIrTe$_4$ a (nm)")
    ax.set_title("optical unit cell and incident polarization")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_power(path: Path, result: dict, material: dict, remap: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    power_values = np.array([
        result["P_Q_pabs_periodic_W"],
        result["P_flux_absorbed_W"],
        result["P_Q_native_uncorrected_W"],
        remap["total_common_power_W"],
    ]) * 1e15
    axes[0].bar(np.arange(4), power_values, color=["#4477aa", "#66ccee", "#228833", "#aa3377"])
    axes[0].set_xticks(np.arange(4), ["pabs", "flux", "native Yee", "common grid"], rotation=18)
    axes[0].set_ylabel("absorbed power (fW/cell)")
    axes[0].set_title("power certificates")
    keys = ["TaIrTe4_geometric", "top_Au_T_geometric", "Au_backplane_geometric", "unassigned_interface_or_other"]
    labels = ["TaIrTe4", "T-envelope/interface", "Au mirror", "interface/other"]
    colors = ["#c74c4c", "#f6c64e", "#be8f00", "#777777"]
    axes[1].bar(np.arange(4), [material[key] * 1e15 for key in keys], color=colors)
    axes[1].set_xticks(np.arange(4), labels, rotation=18)
    axes[1].set_ylabel("native geometric Q (fW/cell)")
    axes[1].set_title("material/support partition")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    output_root = HERE / "results_actual_metasurfaces" / "meeting_plot_package"
    output_root.mkdir(parents=True, exist_ok=True)
    comparison = load_module(HERE / "09_compare_t2024_tairte4_polarizations.py", "meeting_comparison_helpers")
    summary_helpers = comparison.load_summary_helpers()
    remap_module = load_module(
        HERE.parents[3] / "photothermal_pte" / "validation" / "photothermal_stage1" / "13_analyze_native_yee_components.py",
        "meeting_conservative_remap",
    )

    package = {
        "status": "PUBLISHED_T2024_MEETING_PLOT_PACKAGE",
        "coordinate_contract": "Lumerical x=TaIrTe4 b, y=TaIrTe4 a; Q components first remapped from their own Yee coordinate before Qtotal",
        "cases": {},
    }
    for key, raw_dir in CASES.items():
        case = comparison.load_case(raw_dir, summary_helpers)
        result = case["result"]
        remap = common_grid_q(case["raw"], remap_module)
        if remap["relative_power_error"] >= 1e-12:
            raise RuntimeError(f"{key} common-grid power error {remap['relative_power_error']}")
        material = comparison.material_totals(case)
        case_dir = output_root / key
        case_dir.mkdir(parents=True, exist_ok=True)
        save_structure(case_dir / "01_structure_and_source.png", result)
        component_areal = {}
        for index, component in enumerate("xyz", start=2):
            item = case["areal"][component]
            component_areal[component] = item["Q_W_m2"]
            save_map(
                case_dir / f"0{index}_Q{component}_depth_integrated.png",
                item["x"],
                item["y"],
                item["Q_W_m2"],
                f"{key}: native Yee Q{component} depth integral",
                "W/m$^2$",
            )
        q_total_areal = areal(remap["total"], remap["z"], remap_module)
        save_map(
            case_dir / "05_Qtotal_conservative_common_grid.png",
            remap["x"],
            remap["y"],
            q_total_areal,
            f"{key}: Qtotal on conservative common grid",
            "W/m$^2$",
        )
        save_power(case_dir / "06_power_and_material_breakdown.png", result, material, remap)

        # A compact overview assembled from the already certified arrays.
        fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), constrained_layout=True)
        geometry = result["contract"]["geometry"]
        vertices = np.asarray(geometry["polygons"][0]["vertices_nm"], float)
        axes[0, 0].add_patch(Rectangle((-750, -500), 1500, 1000, facecolor="#d04b4b", alpha=0.25, edgecolor="#8d2020"))
        if result["contract"].get("top_Au_T_present", True):
            axes[0, 0].fill(vertices[:, 0], vertices[:, 1], color="#f4bf42", edgecolor="#8a5a00")
        axes[0, 0].set_xlim(-750, 750); axes[0, 0].set_ylim(-500, 500); axes[0, 0].set_aspect("equal")
        axes[0, 0].set_title(f"{key}: structure")
        for axis, component in zip((axes[0, 1], axes[0, 2], axes[1, 0]), "xyz"):
            item = case["areal"][component]
            image = axis.pcolormesh(item["x"] * 1e9, item["y"] * 1e9, item["Q_W_m2"].T, shading="auto", cmap="inferno", vmin=0.0)
            axis.set_aspect("equal"); axis.set_title(f"native Q{component}")
            fig.colorbar(image, ax=axis, label="W/m$^2$")
        image = axes[1, 1].pcolormesh(remap["x"] * 1e9, remap["y"] * 1e9, q_total_areal.T, shading="auto", cmap="inferno", vmin=0.0)
        axes[1, 1].set_aspect("equal"); axes[1, 1].set_title("conservative common-grid Qtotal")
        fig.colorbar(image, ax=axes[1, 1], label="W/m$^2$")
        axes[1, 2].axis("off")
        axes[1, 2].text(0, 1, "\n".join([
            f"P_Q = {result['P_Q_pabs_periodic_W']:.6e} W/cell",
            f"closure = {100*result['closure_relative']:.5f}%",
            f"auto-shutoff = {result['log_audit']['final_auto_shutoff']:.3e}",
            f"common remap error = {remap['relative_power_error']:.3e}",
            f"TaIrTe4 Q = {material['TaIrTe4_geometric']:.6e} W/cell",
            "No clipping/smoothing/gain/rescaling",
        ]), va="top", family="monospace", fontsize=10)
        for axis in axes.ravel()[:5]:
            axis.set_xlabel("x=b (nm)"); axis.set_ylabel("y=a (nm)")
        fig.savefig(case_dir / "00_case_overview.png", dpi=220)
        plt.close(fig)

        metrics = {
            "case": key,
            "raw_result": str(case["result_path"]),
            "polarization": result["contract"]["source"]["polarization"],
            "top_Au_T_present": result["contract"].get("top_Au_T_present", True),
            "P_Q_periodic_W": result["P_Q_pabs_periodic_W"],
            "P_flux_absorbed_W": result["P_flux_absorbed_W"],
            "P_Q_native_W": result["P_Q_native_uncorrected_W"],
            "Q_component_power_native_W": result["Q_component_power_native_W"],
            "material_power_native_W": material,
            "common_grid": {
                "power_W": remap["total_common_power_W"],
                "relative_power_error": remap["relative_power_error"],
                "coordinate_checks": remap["coordinate_checks"],
            },
            "closure_relative": result["closure_relative"],
            "auto_shutoff": result["log_audit"]["final_auto_shutoff"],
        }
        (case_dir / "case_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        package["cases"][key] = {
            "folder": str(case_dir.relative_to(HERE)),
            "common_grid_power_error": remap["relative_power_error"],
            "figures": sorted(path.name for path in case_dir.glob("*.png")),
        }

    readme = output_root / "README.md"
    readme.write_text(
        """# Meeting plot package — 2024 inverse-T / TaIrTe4 optical controls

Each subfolder is one independently solved GPU case. `Qx`, `Qy`, and `Qz` are
native component-grid depth integrals. `Qtotal` is **not** a same-index sum:
each component is conservatively deposited from its staggered coordinate onto
the common monitor grid first. The JSON records the resulting power error.

- `T_Eb`: inverse-T present, E parallel TaIrTe4 b (Lumerical x)
- `T_Ea`: inverse-T present, E parallel TaIrTe4 a (Lumerical y)
- `bare_Eb`: matched stack without top T, E parallel b
- `bare_Ea`: matched stack without top T, E parallel a

The package is a single-wavelength optical-forward result. It is not a thermal,
PTE, adjoint, or optimized-metal result.
"""
    )
    package_path = output_root / "meeting_plot_package.json"
    package_path.write_text(json.dumps(package, indent=2) + "\n")
    published = sorted(
        path for path in output_root.rglob("*")
        if path.is_file() and path.name != "MEETING_PLOT_MANIFEST.json"
    )
    manifest = {
        "generation_command": (
            "/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python "
            "photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/"
            "paper_architectures/11_publish_meeting_plot_package.py"
        ),
        "published_artifacts": [
            {
                "path": str(path.relative_to(HERE)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in published
        ],
        "raw_artifacts_committed": False,
    }
    (output_root / "MEETING_PLOT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(package, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
