#!/usr/bin/env python3
"""Publish separate 4/6 um parity plots with ten latent perturbations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ORIGINAL = [
    "adjoint_aligned",
    "central_localized",
    "design_edge_localized",
    "smooth_asymmetric",
    "fixed_seed_random",
]
EXTRA = [
    "uniform",
    "x_antisymmetric",
    "y_antisymmetric",
    "diagonal_quadrupole",
    "radial_ring",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-result", required=True)
    parser.add_argument("--extended-result", required=True)
    parser.add_argument("--report-dir", required=True)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "byte_size": path.stat().st_size,
        "sha256": digest(path),
    }


def original_row(data: dict, scenario: str, name: str) -> dict:
    direction = data["scenarios"][scenario]["directions"][name]
    selected = next(
        row for row in direction["steps"] if np.isclose(row["step"], 0.005)
    )
    return {
        "scenario": scenario,
        "direction_index": ORIGINAL.index(name) + 1,
        "direction": name,
        "set": "original",
        "step": 0.005,
        "AD_A": direction["analytic_directional_A"],
        "FD_A": selected["finite_difference_directional_A"],
        "relative_error": selected["relative_error"],
    }


def extra_row(data: dict, scenario: str, name: str) -> dict:
    direction = data["scenarios"][scenario]["directions"][name]
    return {
        "scenario": scenario,
        "direction_index": len(ORIGINAL) + EXTRA.index(name) + 1,
        "direction": name,
        "set": "additional",
        "step": data["step"],
        "AD_A": direction["analytic_directional_A"],
        "FD_A": direction["finite_difference_directional_A"],
        "relative_error": direction["relative_error"],
    }


def parity_metrics(rows: list[dict]) -> dict[str, float]:
    ad = np.asarray([row["AD_A"] for row in rows])
    fd = np.asarray([row["FD_A"] for row in rows])
    denominator = float(np.dot(fd, fd))
    slope = float(np.dot(fd, ad) / denominator)
    residual = ad - slope * fd
    centered = ad - np.mean(ad)
    r2 = 1.0 - float(np.dot(residual, residual)) / max(
        float(np.dot(centered, centered)), np.finfo(float).tiny
    )
    nrmse = float(np.linalg.norm(ad - fd) / np.linalg.norm(fd))
    cosine = float(
        np.dot(ad, fd)
        / max(
            np.linalg.norm(ad) * np.linalg.norm(fd),
            np.finfo(float).tiny,
        )
    )
    return {
        "slope_through_origin": slope,
        "R2": r2,
        "NRMSE": nrmse,
        "angle_deg": float(
            np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        ),
        "worst_individual_relative_error": max(
            row["relative_error"] for row in rows
        ),
    }


def plot_scenario(
    *,
    rows: list[dict],
    scenario: str,
    output: Path,
    color: str,
) -> dict[str, float]:
    metrics = parity_metrics(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
    for group, marker, label in (
        ("original", "o", "original five"),
        ("additional", "s", "additional five"),
    ):
        selected = [row for row in rows if row["set"] == group]
        axes[0].scatter(
            [row["FD_A"] for row in selected],
            [row["AD_A"] for row in selected],
            s=72,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.55,
            label=label,
        )
    for row in rows:
        axes[0].annotate(
            str(row["direction_index"]),
            (row["FD_A"], row["AD_A"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
        )
    values = [row[key] for row in rows for key in ("AD_A", "FD_A")]
    lower, upper = min(values), max(values)
    padding = max(0.06 * (upper - lower), np.finfo(float).eps)
    axes[0].plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        "k--",
        label="ideal AD = FD",
    )
    axes[0].set_xlim(lower - padding, upper + padding)
    axes[0].set_ylim(lower - padding, upper + padding)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("Finite-difference directional derivative [A]")
    axes[0].set_ylabel("Adjoint directional derivative [A]")
    axes[0].set_title(f"{scenario} full-latent AD–FD, 10 directions")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    text = (
        f"slope (through 0) = {metrics['slope_through_origin']:.7f}\n"
        f"$R^2$ = {metrics['R2']:.8f}\n"
        f"NRMSE = {100*metrics['NRMSE']:.4f}%\n"
        f"angle = {metrics['angle_deg']:.4f}°"
    )
    axes[0].text(
        0.03,
        0.97,
        text,
        transform=axes[0].transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    positions = np.arange(len(rows))
    errors = 100 * np.asarray([row["relative_error"] for row in rows])
    bar_colors = [
        color if row["set"] == "original" else "#8c564b" for row in rows
    ]
    axes[1].bar(positions, errors, color=bar_colors)
    axes[1].axhline(1.0, color="black", linestyle="--", label="1% gate")
    axes[1].set_yscale("log")
    axes[1].set_xticks(
        positions,
        [str(row["direction_index"]) for row in rows],
    )
    axes[1].set_xlabel("Perturbation index")
    axes[1].set_ylabel("AD–FD relative error [%]")
    axes[1].set_title(f"{scenario} directional errors")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return metrics


def main() -> int:
    args = parse_args()
    full_path = Path(args.full_result).expanduser().resolve()
    extended_path = Path(args.extended_result).expanduser().resolve()
    full = json.loads(full_path.read_text())
    extended = json.loads(extended_path.read_text())
    if not full.get("passed") or not extended.get("passed"):
        raise RuntimeError("input AD-FD result is not passed")
    report_dir = Path(args.report_dir).expanduser().resolve()
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    metrics = {}
    outputs = {
        "4um": figure_dir / "21_full_latent_adfd_4um_10directions.png",
        "6um": figure_dir / "22_full_latent_adfd_6um_10directions.png",
    }
    for scenario, color in (("4um", "#1f77b4"), ("6um", "#ff7f0e")):
        scenario_rows = [
            *(original_row(full, scenario, name) for name in ORIGINAL),
            *(extra_row(extended, scenario, name) for name in EXTRA),
        ]
        rows.extend(scenario_rows)
        metrics[scenario] = plot_scenario(
            rows=scenario_rows,
            scenario=scenario,
            output=outputs[scenario],
            color=color,
        )

    full_arrays = np.load(Path(full["arrays"]["path"]))
    extra_arrays = np.load(Path(extended["direction_arrays"]["path"]))
    fig, axes = plt.subplots(2, 5, figsize=(15.5, 6.3))
    names = ORIGINAL + EXTRA
    for index, (axis, name) in enumerate(zip(axes.flat, names), start=1):
        values = (
            full_arrays[f"direction_{name}"]
            if name in ORIGINAL
            else extra_arrays[name]
        )
        maximum = float(np.max(np.abs(values)))
        image = axis.imshow(
            values.T,
            origin="lower",
            extent=(-1, 1, -1, 1),
            vmin=-maximum,
            vmax=maximum,
            cmap="coolwarm",
            aspect="equal",
        )
        axis.set_title(f"{index}. {name.replace('_', ' ')}", fontsize=10)
        axis.set_xlabel("x [µm]")
        axis.set_ylabel("y [µm]")
        fig.colorbar(image, ax=axis, shrink=0.72)
    fig.suptitle(
        "Ten finite nonperiodic 81×81 latent perturbation directions",
        fontsize=15,
    )
    fig.tight_layout()
    map_path = figure_dir / "23_ten_latent_perturbation_maps.png"
    fig.savefig(map_path, dpi=200)
    plt.close(fig)

    csv_path = report_dir / "extended_full_latent_adfd_10directions.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": extended["status"],
        "total_direction_count_per_scenario": 10,
        "step": 0.005,
        "direction_index": {
            str(index): name for index, name in enumerate(names, start=1)
        },
        "metrics": metrics,
        "extended_gates": extended["gates"],
        "figures": {**{key: str(value) for key, value in outputs.items()}, "maps": str(map_path)},
        "optimization_run": False,
    }
    summary_path = (
        report_dir / "extended_full_latent_adfd_10directions_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    raw_arrays = Path(extended["direction_arrays"]["path"]).resolve()
    manifest = {
        "status": extended["status"],
        "generation_command": (
            "python -m photothermal_pte.finite_inverse_design."
            "publish_extended_latent_perturbation_figures "
            "--full-result <external> --extended-result <external> "
            "--report-dir photothermal_pte/reports/inverse_design_pte_adfd"
        ),
        "raw_artifacts_not_committed": [
            artifact(full_path),
            artifact(extended_path),
            artifact(raw_arrays),
        ],
        "published_artifacts": [
            str(outputs["4um"]),
            str(outputs["6um"]),
            str(map_path),
            str(csv_path),
            str(summary_path),
        ],
    }
    manifest_path = (
        report_dir
        / "EXTENDED_FULL_LATENT_PERTURBATION_ADFD_RAW_MANIFEST.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report = f"""# Extended full-latent AD–FD: ten perturbations

**Status: `{extended["status"]}`**

The final five-direction certificate was preserved and five new latent
directions were evaluated with fresh centered Maxwell FD pairs at `h=0.005`:
uniform, x-antisymmetric, y-antisymmetric, diagonal-quadrupole, and
radial-ring. No clipping, gradient rescaling, or empirical normalization was
used.

| scenario | slope through zero | R2 | NRMSE | angle | worst individual error |
|---|---:|---:|---:|---:|---:|
| 4 µm | {metrics["4um"]["slope_through_origin"]:.8f} | {metrics["4um"]["R2"]:.9f} | {100*metrics["4um"]["NRMSE"]:.5f}% | {metrics["4um"]["angle_deg"]:.5f}° | {100*metrics["4um"]["worst_individual_relative_error"]:.5f}% |
| 6 µm | {metrics["6um"]["slope_through_origin"]:.8f} | {metrics["6um"]["R2"]:.9f} | {100*metrics["6um"]["NRMSE"]:.5f}% | {metrics["6um"]["angle_deg"]:.5f}° | {100*metrics["6um"]["worst_individual_relative_error"]:.5f}% |

The 4 and 6 µm plots are intentionally separate. Point labels 1–5 are the
original certificate directions and 6–10 are the new directions. The raw
FSP/NPZ files remain outside Git and are SHA-pinned in the manifest.

- [4 µm ten-direction AD–FD](figures/21_full_latent_adfd_4um_10directions.png)
- [6 µm ten-direction AD–FD](figures/22_full_latent_adfd_6um_10directions.png)
- [ten perturbation maps](figures/23_ten_latent_perturbation_maps.png)
"""
    report_path = (
        report_dir / "EXTENDED_FULL_LATENT_PERTURBATION_ADFD_REPORT.md"
    )
    report_path.write_text(report)
    print(json.dumps({"status": extended["status"], "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
