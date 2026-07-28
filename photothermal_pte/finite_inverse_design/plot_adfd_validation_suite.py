#!/usr/bin/env python3
"""Generate a comprehensive AD--FD validation figure suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from .contract import CERTIFICATE_BETA
from .finite_mapping import FiniteDensityMapping, _projection
from .run_combined_physical_rho_pte_adfd import physical_state
from .run_corrected_combined_physical_rho_pte_adfd import fixed_directions


DIRECTION_ORDER = [
    "adjoint_aligned",
    "central_localized",
    "design_edge_localized",
    "smooth_asymmetric",
    "fixed_seed_random",
]
SHORT = {
    "adjoint_aligned": "aligned",
    "central_localized": "central",
    "design_edge_localized": "edge",
    "smooth_asymmetric": "smooth",
    "fixed_seed_random": "random",
    "seeded_random": "random",
    "asymmetric_smooth": "smooth",
}
COLORS = {
    "adjoint_aligned": "#0072B2",
    "central_localized": "#E69F00",
    "design_edge_localized": "#009E73",
    "smooth_asymmetric": "#CC79A7",
    "fixed_seed_random": "#D55E00",
    "seeded_random": "#D55E00",
    "asymmetric_smooth": "#CC79A7",
}
MARKERS = {0.01: "o", 0.005: "s", 0.0025: "^", 0.02: "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-result", required=True)
    parser.add_argument("--gradient-dir", required=True)
    parser.add_argument("--thermal-result", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--near-null-result")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.titlesize": 15,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def through_origin_metrics(fd: np.ndarray, ad: np.ndarray) -> dict:
    slope = float(np.dot(fd, ad) / np.dot(fd, fd))
    predicted = slope * fd
    residual = ad - predicted
    total = ad - np.mean(ad)
    r2 = 1.0 - float(np.dot(residual, residual)) / max(
        float(np.dot(total, total)), np.finfo(float).tiny
    )
    nrmse = float(np.linalg.norm(ad - fd)) / max(
        float(np.linalg.norm(fd)), np.finfo(float).tiny
    )
    return {"slope": slope, "r2": r2, "nrmse": nrmse}


def selected_rows(combined: dict, scenario: str, step: float) -> list[dict]:
    rows = []
    for name in DIRECTION_ORDER:
        data = combined["scenarios"][scenario]["directions"][name]
        matches = [
            row
            for row in data["steps"]
            if np.isclose(row["step"], step, rtol=0.0, atol=1e-15)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"missing {scenario}/{name}/h={step}")
        rows.append(
            {
                "direction": name,
                "ad": float(data["analytic_directional_A"]),
                "fd": float(matches[0]["finite_difference_directional_A"]),
                "error": float(matches[0]["relative_error"]),
            }
        )
    return rows


def plot_combined_parity(combined: dict, output: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    metrics = {}
    for ax, scenario in zip(axes, ("4um", "6um")):
        rows = selected_rows(combined, scenario, 0.005)
        fd = np.asarray([row["fd"] for row in rows])
        ad = np.asarray([row["ad"] for row in rows])
        stats = through_origin_metrics(fd, ad)
        metrics[scenario] = stats
        lower = min(float(np.min(fd)), float(np.min(ad)))
        upper = max(float(np.max(fd)), float(np.max(ad)))
        pad = 0.08 * max(upper - lower, abs(upper), abs(lower))
        line = np.linspace(lower - pad, upper + pad, 200)
        ax.plot(line, line, "k--", lw=1.6, label="ideal AD = FD")
        for index, row in enumerate(rows):
            name = row["direction"]
            ax.scatter(
                row["fd"],
                row["ad"],
                s=75,
                marker=MARKERS[0.005],
                color=COLORS[name],
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
            ax.annotate(
                str(index + 1),
                (row["fd"], row["ad"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=9,
            )
        ax.set_title(
            rf"Combined physical-$\rho$: {scenario.replace('um', ' µm')}"
        )
        ax.set_xlabel("Finite-difference directional derivative [A]")
        ax.set_ylabel("Adjoint directional derivative [A]")
        ax.text(
            0.03,
            0.96,
            "h = 0.005\n"
            f"slope(through 0) = {stats['slope']:.7f}\n"
            f"$R^2$ = {stats['r2']:.8f}\n"
            f"NRMSE = {100*stats['nrmse']:.5f}%",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#0072B2", "alpha": 0.9},
        )
        ax.legend(loc="lower right")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    fig.suptitle(
        "Corrected Maxwell–thermal–PTE AD–FD parity\n"
        "1 aligned, 2 central, 3 edge, 4 smooth, 5 fixed-seed random"
    )
    fig.tight_layout()
    save(fig, output / "01_combined_adfd_parity.png")
    return metrics


def extension_rows(
    near_null: dict | None, scenario: str, direction: str
) -> list[dict]:
    if near_null is None:
        return []
    matches = [
        case
        for case in near_null.get("cases", [])
        if case["scenario"] == scenario and case["direction"] == direction
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"missing unique near-null case for {scenario}/{direction}"
        )
    return matches[0]["rows"]


def plot_combined_step_error(
    combined: dict, output: Path, near_null: dict | None
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, scenario in zip(axes, ("4um", "6um")):
        for name in DIRECTION_ORDER:
            rows = sorted(
                combined["scenarios"][scenario]["directions"][name]["steps"],
                key=lambda row: row["step"],
                reverse=True,
            )
            if name in ("central_localized", "fixed_seed_random") and near_null:
                rows = sorted(
                    extension_rows(near_null, scenario, name),
                    key=lambda row: row["step"],
                    reverse=True,
                )
            ax.plot(
                [row["step"] for row in rows],
                [100 * row["relative_error"] for row in rows],
                marker="o",
                color=COLORS[name],
                label=SHORT[name],
            )
        ax.axhline(1.0, color="black", ls="--", lw=1, label="1% gate")
        ax.axhline(0.1, color="gray", ls=":", lw=1, label="0.1% guide")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.invert_xaxis()
        ticks = [0.02, 0.01, 0.005, 0.0025] if near_null else [
            0.01,
            0.005,
            0.0025,
        ]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{value:g}" for value in ticks])
        ax.set_title(scenario.replace("um", " µm"))
        ax.set_xlabel("Centered-FD step h")
    axes[0].set_ylabel("|AD − FD| / max(|AD|, |FD|) [%]")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center")
    fig.suptitle("Combined AD–FD relative error versus step")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, output / "02_combined_relative_error_vs_step.png")


def plot_fd_ratio(
    combined: dict, output: Path, near_null: dict | None
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, scenario in zip(axes, ("4um", "6um")):
        for name in DIRECTION_ORDER:
            data = combined["scenarios"][scenario]["directions"][name]
            ad = float(data["analytic_directional_A"])
            rows = sorted(data["steps"], key=lambda row: row["step"], reverse=True)
            if name in ("central_localized", "fixed_seed_random") and near_null:
                rows = sorted(
                    extension_rows(near_null, scenario, name),
                    key=lambda row: row["step"],
                    reverse=True,
                )
            ax.plot(
                [row["step"] for row in rows],
                [
                    100
                    * (
                        float(row["finite_difference_directional_A"]) / ad
                        - 1.0
                    )
                    for row in rows
                ],
                marker="o",
                color=COLORS[name],
                label=SHORT[name],
            )
        ax.axhline(0.0, color="black", ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.invert_xaxis()
        ticks = [0.02, 0.01, 0.005, 0.0025] if near_null else [
            0.01,
            0.005,
            0.0025,
        ]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{value:g}" for value in ticks])
        ax.set_title(scenario.replace("um", " µm"))
        ax.set_xlabel("Centered-FD step h")
    axes[0].set_ylabel("(FD / AD − 1) [%]")
    fig.legend(*axes[1].get_legend_handles_labels(), ncol=5, loc="lower center")
    fig.suptitle("Signed finite-difference deviation from the adjoint")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, output / "03_combined_fd_over_ad_vs_step.png")


def plot_error_heatmap(combined: dict, output: Path) -> None:
    steps = [0.01, 0.005, 0.0025]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), constrained_layout=True)
    for ax, scenario in zip(axes, ("4um", "6um")):
        matrix = np.empty((len(DIRECTION_ORDER), len(steps)))
        for i, name in enumerate(DIRECTION_ORDER):
            data = combined["scenarios"][scenario]["directions"][name]
            by_step = {float(row["step"]): row for row in data["steps"]}
            for j, step in enumerate(steps):
                matrix[i, j] = 100 * float(by_step[step]["relative_error"])
        image = ax.imshow(
            np.maximum(matrix, 1e-8),
            cmap="magma",
            norm=LogNorm(vmin=1e-5, vmax=1.0),
            aspect="auto",
        )
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i,j]:.4f}%",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > 0.003 else "black",
                    fontsize=8,
                )
        ax.set_xticks(range(len(steps)), [str(step) for step in steps])
        ax.set_yticks(range(len(DIRECTION_ORDER)), [SHORT[n] for n in DIRECTION_ORDER])
        ax.set_xlabel("FD step h")
        ax.set_title(scenario.replace("um", " µm"))
    fig.colorbar(image, ax=axes, label="relative error [%]")
    fig.suptitle("Combined AD–FD directional-error matrix")
    save(fig, output / "04_combined_error_heatmap.png")


def load_gradients(directory: Path) -> tuple[dict, dict]:
    gradients = {}
    arrays = {}
    for flake in (4, 6):
        path = directory / f"dz_2.5nm_{flake}um_nodal_gradients.npz"
        data = np.load(path)
        arrays[float(flake)] = {
            key: np.asarray(data[key], float)
            for key in (
                "optical_gradient_A",
                "thermal_gradient_A",
                "combined_gradient_A",
            )
        }
        gradients[float(flake)] = arrays[float(flake)][
            "combined_gradient_A"
        ]
    return gradients, arrays


def plot_direction_maps(gradients: dict, output: Path) -> None:
    directions = fixed_directions(gradients)
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), constrained_layout=True)
    extent = (-1, 1, -1, 1)
    for ax, name in zip(axes, DIRECTION_ORDER):
        image = ax.imshow(
            directions[name].T,
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
        ax.set_title(SHORT[name])
        ax.set_xlabel("x [µm]")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y [µm]")
    fig.colorbar(image, ax=axes, label="normalized direction")
    fig.suptitle("Physical-density directions used in the combined AD–FD gate")
    save(fig, output / "05_combined_direction_maps.png")


def plot_gradient_maps(arrays: dict, output: Path) -> None:
    rho, _ = physical_state()
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), constrained_layout=True)
    extent = (-1, 1, -1, 1)
    keys = [
        ("rho", r"baseline $\rho$"),
        ("optical_gradient_A", "optical-Q gradient"),
        ("thermal_gradient_A", "thermal-material gradient"),
        ("combined_gradient_A", "combined gradient"),
    ]
    for row, flake in enumerate((4.0, 6.0)):
        for col, (key, title) in enumerate(keys):
            ax = axes[row, col]
            if key == "rho":
                image = ax.imshow(
                    rho.T,
                    origin="lower",
                    extent=extent,
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                )
                label = r"$\rho$"
            else:
                values = arrays[flake][key]
                scale = float(np.max(np.abs(values)))
                image = ax.imshow(
                    values.T,
                    origin="lower",
                    extent=extent,
                    cmap="coolwarm",
                    vmin=-scale,
                    vmax=scale,
                )
                label = "A per nodal density"
            ax.set_title(f"{flake:g} µm: {title}")
            ax.set_xlabel("x [µm]")
            ax.set_ylabel("y [µm]")
            fig.colorbar(image, ax=ax, shrink=0.78, label=label)
    fig.suptitle("81×81 physical-density and corrected gradient fields")
    save(fig, output / "06_combined_gradient_maps.png")


def plot_gradient_norms(arrays: dict, output: Path) -> None:
    labels = ["optical", "thermal", "combined"]
    keys = [
        "optical_gradient_A",
        "thermal_gradient_A",
        "combined_gradient_A",
    ]
    x = np.arange(2)
    width = 0.23
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for index, (label, key) in enumerate(zip(labels, keys)):
        values = [
            float(np.linalg.norm(arrays[flake][key])) for flake in (4.0, 6.0)
        ]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xticks(x, ["4 µm thermal flake", "6 µm thermal flake"])
    ax.set_ylabel(r"$L_2$ norm [A per nodal density]")
    ax.set_title("Optical and thermal contributions to the combined gradient")
    ax.legend()
    fig.tight_layout()
    save(fig, output / "07_combined_gradient_norm_decomposition.png")


def thermal_selected(thermal: dict, flake_um: float, step: float) -> list[dict]:
    scenario = next(
        item
        for item in thermal["scenarios"]
        if float(item["flake_span_um"]) == flake_um
    )
    rows = []
    for direction in scenario["directions"]:
        match = next(
            row
            for row in direction["steps"]
            if np.isclose(row["step"], step, rtol=0.0, atol=1e-15)
        )
        rows.append(match)
    return rows


def plot_thermal_parity(thermal: dict, output: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    metrics = {}
    for ax, flake in zip(axes, (4.0, 6.0)):
        rows = thermal_selected(thermal, flake, 0.0025)
        fd = np.asarray([row["finite_difference_directional_A"] for row in rows])
        ad = np.asarray([row["adjoint_directional_A"] for row in rows])
        stats = through_origin_metrics(fd, ad)
        metrics[f"{flake:g}um"] = stats
        lower = min(float(np.min(fd)), float(np.min(ad)))
        upper = max(float(np.max(fd)), float(np.max(ad)))
        pad = 0.08 * max(upper - lower, abs(upper), abs(lower))
        line = np.linspace(lower - pad, upper + pad, 200)
        ax.plot(line, line, "k--", lw=1.5, label="ideal AD = FD")
        for index, row in enumerate(rows):
            name = row["direction"]
            color = COLORS.get(name, "#0072B2")
            ax.scatter(
                row["finite_difference_directional_A"],
                row["adjoint_directional_A"],
                s=70,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.annotate(
                SHORT.get(name, name),
                (
                    row["finite_difference_directional_A"],
                    row["adjoint_directional_A"],
                ),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
            )
        ax.text(
            0.03,
            0.96,
            "h = 0.0025\n"
            f"slope = {stats['slope']:.8f}\n"
            f"$R^2$ = {stats['r2']:.9f}\n"
            f"NRMSE = {100*stats['nrmse']:.6f}%",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#009E73", "alpha": 0.9},
        )
        ax.set_title(f"{flake:g} µm")
        ax.set_xlabel("Finite-difference directional derivative [A]")
        ax.set_ylabel("Thermal adjoint directional derivative [A]")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        ax.legend(loc="lower right")
    fig.suptitle("Fixed-local-Q thermal-only AD–FD parity")
    fig.tight_layout()
    save(fig, output / "08_thermal_only_adfd_parity.png")
    return metrics


def plot_thermal_step_error(thermal: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, flake in zip(axes, (4.0, 6.0)):
        scenario = next(
            item
            for item in thermal["scenarios"]
            if float(item["flake_span_um"]) == flake
        )
        for direction in scenario["directions"]:
            rows = sorted(
                direction["steps"], key=lambda row: row["step"], reverse=True
            )
            name = direction["name"]
            ax.plot(
                [row["step"] for row in rows],
                [100 * row["relative_error"] for row in rows],
                marker="o",
                label=SHORT.get(name, name),
                color=COLORS.get(name),
            )
        ax.axhline(1.0, color="black", ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xticks([0.01, 0.005, 0.0025])
        ax.set_xticklabels(["0.01", "0.005", "0.0025"])
        ax.set_title(f"{flake:g} µm")
        ax.set_xlabel("Centered-FD step h")
    axes[0].set_ylabel("thermal-only relative error [%]")
    fig.legend(*axes[1].get_legend_handles_labels(), ncol=5, loc="lower center")
    fig.suptitle("Thermal-only AD–FD convergence")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, output / "09_thermal_only_relative_error_vs_step.png")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot_optical_dz(report_dir: Path, output: Path) -> None:
    rows = read_csv(
        report_dir / "optical_dz_downstream_pte_gradient_convergence_cases.csv"
    )
    metrics = [
        ("remapped_Q_field_NRMSE", "remapped Q NRMSE"),
        ("TaIrTe4_temperature_field_NRMSE", "temperature NRMSE"),
        ("PTE_objective_relative_difference", "PTE objective"),
        (
            "optical_directional_gradient_relative_difference",
            "optical gradient",
        ),
        (
            "combined_directional_gradient_relative_difference",
            "combined gradient",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, scenario in zip(axes, ("4um", "6um")):
        selected = [row for row in rows if row["scenario"] == scenario]
        labels = [
            f"{row['coarse_dz_nm']}→{row['fine_dz_nm']}" for row in selected
        ]
        for key, label in metrics:
            ax.plot(
                labels,
                [100 * float(row[key]) for row in selected],
                marker="o",
                label=label,
            )
        ax.axhline(0.5, color="black", ls="--", lw=1, label="0.5% gate")
        ax.set_yscale("log")
        ax.set_title(scenario.replace("um", " µm"))
        ax.set_xlabel("optical flake dz [nm]")
    axes[0].set_ylabel("relative change [%]")
    fig.legend(*axes[1].get_legend_handles_labels(), ncol=3, loc="lower center")
    fig.suptitle("Downstream optical-mesh convergence")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, output / "10_optical_dz_downstream_convergence.png")


def plot_gate_margins(combined: dict, report_dir: Path, output: Path) -> None:
    gates = combined["gates"]
    entries = [
        (
            "strong AD–FD",
            gates["worst_strong_direction_relative_error"],
            gates["strong_direction_limit"],
        ),
        (
            "multidirection",
            gates["worst_multidirection_normalized_error"],
            gates["multidirection_normalized_limit"],
        ),
        (
            "gradient angle",
            gates["worst_directional_subspace_gradient_angle_deg"],
            gates["gradient_angle_limit_deg"],
        ),
        (
            "mapping transpose",
            gates["mapping_transpose_relative_error"],
            gates["mapping_transpose_limit"],
        ),
        (
            "optical closure",
            gates["worst_optical_closure_relative_error"],
            gates["optical_closure_limit"],
        ),
        (
            "Q mapping",
            gates["worst_Q_mapping_relative_error"],
            gates["Q_mapping_limit"],
        ),
        (
            "thermal energy",
            gates["worst_thermal_energy_balance_relative_error"],
            gates["thermal_energy_balance_limit"],
        ),
        (
            "linear residual",
            gates["worst_linear_residual_relative"],
            gates["linear_residual_limit"],
        ),
    ]
    jacobian = load_json(report_dir / "component_yee_material_jacobian_summary.json")
    entries.extend(
        [
            (
                "mapping-only FD",
                jacobian["gates"]["worst_mapping_only_FD_relative_error"],
                jacobian["gates"]["mapping_only_FD_limit"],
            ),
            (
                "JVP/VJP dot",
                jacobian["gates"]["worst_JVP_VJP_dot_relative_error"],
                jacobian["gates"]["dot_limit"],
            ),
        ]
    )
    ratios = [value / limit for _, value, limit in entries]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#009E73" if ratio < 1 else "#D55E00" for ratio in ratios]
    bars = ax.bar(range(len(entries)), ratios, color=colors)
    ax.axhline(1.0, color="black", ls="--", label="gate")
    ax.set_yscale("log")
    ax.set_xticks(
        range(len(entries)),
        [entry[0] for entry in entries],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("measured value / gate limit")
    ax.set_title("AD–FD supporting-gate margins")
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ratio,
            f"{ratio:.1e}×",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )
    ax.legend()
    fig.tight_layout()
    save(fig, output / "11_supporting_gate_margins.png")


def plot_dashboard(
    combined: dict,
    arrays: dict,
    output: Path,
    near_null: dict | None,
) -> None:
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    for col, scenario in enumerate(("4um", "6um")):
        ax = fig.add_subplot(grid[0, col])
        rows = selected_rows(combined, scenario, 0.005)
        fd = np.asarray([row["fd"] for row in rows])
        ad = np.asarray([row["ad"] for row in rows])
        lower = min(float(np.min(fd)), float(np.min(ad)))
        upper = max(float(np.max(fd)), float(np.max(ad)))
        line = np.linspace(lower, upper, 100)
        ax.plot(line, line, "k--")
        for row in rows:
            ax.scatter(
                row["fd"],
                row["ad"],
                color=COLORS[row["direction"]],
                s=55,
                edgecolor="black",
                linewidth=0.4,
            )
        stats = through_origin_metrics(fd, ad)
        ax.set_title(
            f"{scenario.replace('um',' µm')} parity, "
            f"NRMSE={100*stats['nrmse']:.4f}%"
        )
        ax.set_xlabel("FD [A]")
        ax.set_ylabel("AD [A]")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax = fig.add_subplot(grid[0, 2])
    for scenario, linestyle in (("4um", "-"), ("6um", "--")):
        data = combined["scenarios"][scenario]["directions"][
            "fixed_seed_random"
        ]
        rows = sorted(data["steps"], key=lambda row: row["step"], reverse=True)
        if near_null:
            rows = sorted(
                extension_rows(
                    near_null, scenario, "fixed_seed_random"
                ),
                key=lambda row: row["step"],
                reverse=True,
            )
        ax.plot(
            [row["step"] for row in rows],
            [100 * row["relative_error"] for row in rows],
            marker="o",
            ls=linestyle,
            label=f"{scenario} random",
        )
    ax.axhline(1.0, color="black", ls="--")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_title("near-null FD resolution")
    ax.set_xlabel("h")
    ax.set_ylabel("relative error [%]")
    ax.legend()
    rho, _ = physical_state()
    for col, (values, title) in enumerate(
        [
            (rho, r"baseline $\rho$"),
            (arrays[4.0]["combined_gradient_A"], "4 µm combined gradient"),
            (arrays[6.0]["combined_gradient_A"], "6 µm combined gradient"),
        ]
    ):
        ax = fig.add_subplot(grid[1, col])
        if col == 0:
            image = ax.imshow(
                values.T,
                origin="lower",
                extent=(-1, 1, -1, 1),
                cmap="viridis",
                vmin=0,
                vmax=1,
            )
        else:
            scale = float(np.max(np.abs(values)))
            image = ax.imshow(
                values.T,
                origin="lower",
                extent=(-1, 1, -1, 1),
                cmap="coolwarm",
                vmin=-scale,
                vmax=scale,
            )
        ax.set_title(title)
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(
        "TaIrTe₄ PTE inverse-design AD–FD validation dashboard\n"
        "Corrected full-Yee measure; official five-direction status remains "
        "fail-closed only on the strict near-null step plateau"
    )
    save(fig, output / "12_adfd_validation_dashboard.png")


def plot_near_null_plateau(near_null: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, scenario in zip(axes, ("4um", "6um")):
        for direction in ("central_localized", "fixed_seed_random"):
            case = next(
                item
                for item in near_null["cases"]
                if item["scenario"] == scenario
                and item["direction"] == direction
            )
            rows = sorted(case["rows"], key=lambda row: row["step"], reverse=True)
            ax.plot(
                [row["step"] for row in rows],
                [100.0 * row["relative_error"] for row in rows],
                marker="o",
                color=COLORS[direction],
                label=SHORT[direction],
            )
            ax.text(
                rows[-1]["step"],
                100.0 * rows[-1]["relative_error"],
                " pass" if case["passed"] else " fail",
                color=COLORS[direction],
                va="bottom",
                fontsize=9,
            )
        ax.axhline(1.0, color="black", ls="--", lw=1, label="1% AD–FD gate")
        ax.axhline(
            0.1,
            color="gray",
            ls=":",
            lw=1,
            label="0.1% plateau scale",
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xticks([0.02, 0.01, 0.005])
        ax.set_xticklabels(["0.02", "0.01", "0.005"])
        ax.set_title(scenario.replace("um", " µm"))
        ax.set_xlabel("scale-adaptive centered-FD step h")
    axes[0].set_ylabel("|AD − FD| / max(|AD|, |FD|) [%]")
    fig.legend(*axes[1].get_legend_handles_labels(), ncol=4, loc="lower center")
    fig.suptitle(
        "Near-null combined AD–FD: coarser halving sequence\n"
        "No clipping, normalization, or gradient rescaling"
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, output / "13_near_null_scale_adaptive_plateau.png")


def plot_filter_projection_contract(output: Path) -> None:
    mapping = FiniteDensityMapping()
    center = np.zeros(mapping.latent_shape)
    center[40, 40] = 1.0
    corner = np.zeros(mapping.latent_shape)
    corner[0, 0] = 1.0
    center_filtered = mapping.filtered(center)
    corner_filtered = mapping.filtered(corner)
    rho = np.linspace(0.0, 1.0, 500)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)
    for ax, values, title in (
        (axes[0], center_filtered, "center impulse"),
        (axes[1], corner_filtered, "corner impulse (no wrap)"),
    ):
        image = ax.imshow(
            values.T,
            origin="lower",
            extent=(-1, 1, -1, 1),
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
        fig.colorbar(image, ax=ax, shrink=0.82, label="filter weight")
    for beta in (2.0, 4.0, CERTIFICATE_BETA, 16.0):
        axes[2].plot(
            rho,
            _projection(rho, beta, mapping.eta),
            label=rf"$\beta={beta:g}$",
        )
    axes[2].plot(rho, rho, "k--", lw=1, label="identity")
    axes[2].set_title("tanh projection")
    axes[2].set_xlabel("filtered latent density")
    axes[2].set_ylabel("physical density")
    axes[2].legend()
    fig.suptitle(
        "Finite 81×81 latent → filter → projection contract\n"
        "25 nm nodes, 500 nm conic radius, no periodic wrapping"
    )
    save(fig, output / "14_filter_projection_contract.png")


def plot_gradient_component_scatter(arrays: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1))
    for ax, flake in zip(axes, (4.0, 6.0)):
        optical = arrays[flake]["optical_gradient_A"].reshape(-1)
        thermal = arrays[flake]["thermal_gradient_A"].reshape(-1)
        correlation = float(np.corrcoef(optical, thermal)[0, 1])
        ax.scatter(optical, thermal, s=5, alpha=0.32, color="#0072B2")
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(f"{flake:g} µm; Pearson r={correlation:.4f}")
        ax.set_xlabel("optical-Q contribution [A/node]")
        ax.set_ylabel("thermal-material contribution [A/node]")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    fig.suptitle("Pixelwise optical versus thermal gradient contribution")
    fig.tight_layout()
    save(fig, output / "15_optical_thermal_gradient_scatter.png")


def plot_gradient_linecuts(arrays: dict, output: Path) -> None:
    coordinate = np.linspace(-1.0, 1.0, 81)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=True)
    for row, flake in enumerate((4.0, 6.0)):
        for ax, axis, indexer in (
            (axes[row, 0], "x", lambda value: value[:, 40]),
            (axes[row, 1], "y", lambda value: value[40, :]),
        ):
            for key, label in (
                ("optical_gradient_A", "optical"),
                ("thermal_gradient_A", "thermal"),
                ("combined_gradient_A", "combined"),
            ):
                ax.plot(coordinate, indexer(arrays[flake][key]), label=label)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_title(f"{flake:g} µm, central {axis}-line")
            ax.set_xlabel(f"{axis} [µm]")
            ax.set_ylabel("gradient [A/node]")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0, 1].legend()
    fig.suptitle("Central line cuts through the physical-density gradients")
    fig.tight_layout()
    save(fig, output / "16_gradient_central_linecuts.png")


def plot_directional_magnitude(combined: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    for ax, scenario in zip(axes, ("4um", "6um")):
        ad = [
            abs(
                float(
                    combined["scenarios"][scenario]["directions"][name][
                        "analytic_directional_A"
                    ]
                )
            )
            for name in DIRECTION_ORDER
        ]
        errors = [
            100.0 * selected_rows(combined, scenario, 0.005)[index]["error"]
            for index in range(len(DIRECTION_ORDER))
        ]
        bars = ax.bar(
            range(len(ad)),
            ad,
            color=[COLORS[name] for name in DIRECTION_ORDER],
        )
        ax.set_yscale("log")
        ax.set_xticks(
            range(len(ad)),
            [SHORT[name] for name in DIRECTION_ORDER],
            rotation=25,
            ha="right",
        )
        ax.set_title(scenario.replace("um", " µm"))
        ax.set_xlabel("physical-density direction")
        for bar, error in zip(bars, errors):
            ax.text(
                bar.get_x() + 0.5 * bar.get_width(),
                bar.get_height(),
                f"{error:.4f}% err",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7,
            )
    axes[0].set_ylabel("|adjoint directional derivative| [A]")
    fig.suptitle(
        "Directional-derivative dynamic range and h=0.005 AD–FD error"
    )
    fig.tight_layout()
    save(fig, output / "17_directional_derivative_dynamic_range.png")


def main() -> int:
    args = parse_args()
    combined_path = Path(args.combined_result).expanduser().resolve()
    gradient_dir = Path(args.gradient_dir).expanduser().resolve()
    thermal_path = Path(args.thermal_result).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    style()
    combined = load_json(combined_path)
    thermal = load_json(thermal_path)
    near_null_path = (
        Path(args.near_null_result).expanduser().resolve()
        if args.near_null_result
        else None
    )
    near_null = (
        load_json(near_null_path)
        if near_null_path is not None and near_null_path.is_file()
        else None
    )
    gradients, arrays = load_gradients(gradient_dir)
    for flake in (4.0, 6.0):
        expected = float(combined["scenarios"][f"{flake:g}um"]["gradient_L2_A"])
        actual = float(np.linalg.norm(arrays[flake]["combined_gradient_A"]))
        if not np.isclose(actual, expected, rtol=2e-12, atol=0.0):
            raise RuntimeError(
                f"{flake:g}um gradient artifact mismatch: {actual} != {expected}"
            )

    combined_metrics = plot_combined_parity(combined, output)
    plot_combined_step_error(combined, output, near_null)
    plot_fd_ratio(combined, output, near_null)
    plot_error_heatmap(combined, output)
    plot_direction_maps(gradients, output)
    plot_gradient_maps(arrays, output)
    plot_gradient_norms(arrays, output)
    thermal_metrics = plot_thermal_parity(thermal, output)
    plot_thermal_step_error(thermal, output)
    plot_optical_dz(report_dir, output)
    plot_gate_margins(combined, report_dir, output)
    plot_dashboard(combined, arrays, output, near_null)
    if near_null is not None:
        plot_near_null_plateau(near_null, output)
    plot_filter_projection_contract(output)
    plot_gradient_component_scatter(arrays, output)
    plot_gradient_linecuts(arrays, output)
    plot_directional_magnitude(combined, output)

    inputs = [combined_path, thermal_path]
    inputs.extend(
        gradient_dir / f"dz_2.5nm_{flake}um_nodal_gradients.npz"
        for flake in (4, 6)
    )
    if near_null_path is not None and near_null is not None:
        inputs.append(near_null_path)
    figures = sorted(output.glob("*.png"))
    descriptions = {
        "01_combined_adfd_parity.png": (
            "Five-direction corrected Maxwell–thermal–PTE AD versus centered FD."
        ),
        "02_combined_relative_error_vs_step.png": (
            "Relative directional error versus FD step; near-null h=0.02 is "
            "included when supplied."
        ),
        "03_combined_fd_over_ad_vs_step.png": (
            "Signed FD/AD deviation, which exposes bias separately from magnitude."
        ),
        "04_combined_error_heatmap.png": (
            "Direction-by-step matrix for the immutable five-direction sweep."
        ),
        "05_combined_direction_maps.png": (
            "The exact 81×81 physical-density perturbation fields."
        ),
        "06_combined_gradient_maps.png": (
            "Baseline density and optical, thermal-material, and total gradients."
        ),
        "07_combined_gradient_norm_decomposition.png": (
            "L2 norms of optical and thermal contributions."
        ),
        "08_thermal_only_adfd_parity.png": (
            "Independent fixed-local-Q thermal-material adjoint certificate."
        ),
        "09_thermal_only_relative_error_vs_step.png": (
            "Thermal-only centered-FD step convergence."
        ),
        "10_optical_dz_downstream_convergence.png": (
            "Q, temperature, PTE, and gradient dependence on optical flake dz."
        ),
        "11_supporting_gate_margins.png": (
            "Measured-to-limit ratios for closure, mapping, residual, and Jacobian gates."
        ),
        "12_adfd_validation_dashboard.png": (
            "Compact status dashboard with parity, near-null behavior, and gradients."
        ),
        "13_near_null_scale_adaptive_plateau.png": (
            "Dedicated 0.02→0.01→0.005 near-null plateau test."
        ),
        "14_filter_projection_contract.png": (
            "Finite nonperiodic filter support and tanh projection law."
        ),
        "15_optical_thermal_gradient_scatter.png": (
            "Pixelwise relation between optical-Q and thermal-material sensitivity."
        ),
        "16_gradient_central_linecuts.png": (
            "Central x/y line cuts through each gradient contribution."
        ),
        "17_directional_derivative_dynamic_range.png": (
            "Derivative dynamic range and h=0.005 error annotations."
        ),
    }
    report_lines = [
        "# AD–FD validation figure suite",
        "",
        f"- Immutable combined status: `{combined['status']}`",
        (
            f"- Near-null extension status: `{near_null['status']}`"
            if near_null is not None
            else "- Near-null extension: not included"
        ),
        "- Figures are derived from SHA-pinned raw JSON/NPZ artifacts.",
        "- No normalization or gradient rescaling is applied to AD or FD values.",
        "",
    ]
    for figure in figures:
        report_lines.extend(
            [
                f"## {figure.stem.replace('_', ' ')}",
                "",
                descriptions.get(figure.name, ""),
                "",
                f"![{figure.stem}]({figure.name})",
                "",
            ]
        )
    figure_report = output / "ADFD_VALIDATION_FIGURE_REPORT.md"
    figure_report.write_text("\n".join(report_lines))
    summary = {
        "status": "GENERATED_ADFD_VALIDATION_FIGURE_SUITE",
        "official_combined_status": combined["status"],
        "combined_h0p005_parity_metrics": combined_metrics,
        "thermal_h0p0025_parity_metrics": thermal_metrics,
        "near_null_extension_included": near_null is not None,
        "inputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in inputs
        ],
        "figures": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in figures
        ],
    }
    summary_path = output / "adfd_validation_figure_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "generation_command": " ".join(
            [
                "python -m",
                "photothermal_pte.finite_inverse_design."
                "plot_adfd_validation_suite",
                "--combined-result",
                str(combined_path),
                "--gradient-dir",
                str(gradient_dir),
                "--thermal-result",
                str(thermal_path),
                "--report-dir",
                str(report_dir),
                "--output-dir",
                str(output),
                *(
                    ["--near-null-result", str(near_null_path)]
                    if near_null_path is not None
                    else []
                ),
            ]
        ),
        "raw_FSP_committed_to_git": False,
        "figure_report": {
            "path": str(figure_report),
            "bytes": figure_report.stat().st_size,
            "sha256": sha256(figure_report),
        },
        "summary": {
            "path": str(summary_path),
            "bytes": summary_path.stat().st_size,
            "sha256": sha256(summary_path),
        },
        "figures": summary["figures"],
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"status": summary["status"], "figures": len(figures)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
