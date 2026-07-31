#!/usr/bin/env python3
"""Regenerate saved interface/downstream maps without a thermal solve."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity.coordinate_plot import (  # noqa: E402
    cell_field,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.input_npz, allow_pickle=False) as raw:
        x_edges = np.asarray(raw["x_edges_m"], float)
        y_edges = np.asarray(raw["y_edges_m"], float)
        z_edges = np.asarray(raw["z_edges_m"], float)
        q50 = np.asarray(raw["Q_T_50_W_m3"], float)
        q25 = np.asarray(raw["Q_T_25_W_m3"], float)
        t50 = np.asarray(raw["flake_average_temperature_50_K"], float)
        t25 = np.asarray(raw["flake_average_temperature_25_K"], float)
        gx50 = np.asarray(raw["grad_T_x_50_K_m"], float)
        gx25 = np.asarray(raw["grad_T_x_25_K_m"], float)
        gy50 = np.asarray(raw["grad_T_y_50_K_m"], float)
        gy25 = np.asarray(raw["grad_T_y_25_K_m"], float)

    dz = np.diff(z_edges)
    qxy50 = np.sum(q50 * dz[None, None, :], axis=2)
    qxy25 = np.sum(q25 * dz[None, None, :], axis=2)
    arrays = (
        (qxy50, "mapped Q 50 nm"),
        (qxy25, "mapped Q 25 nm"),
        (qxy50 - qxy25, "mapped Q difference"),
        (t50, "flake-average T 50 nm"),
        (t25, "flake-average T 25 nm"),
        (t50 - t25, "flake-average T difference"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axis, (values, title) in zip(axes.flat, arrays):
        image = cell_field(
            axis,
            x_edges,
            y_edges,
            values,
            coordinate_scale=1e6,
            cmap="coolwarm" if "difference" in title else "inferno",
        )
        axis.set(title=title, xlabel="x=b (µm)", ylabel="y=a (µm)")
        figure.colorbar(image, ax=axis)
    figure.savefig(args.report_dir / "W12_INTERFACE_DOWNSTREAM_COMPARISON.png", dpi=190)
    plt.close(figure)

    magnitude50 = np.hypot(gx50, gy50)
    magnitude25 = np.hypot(gx25, gy25)
    difference = magnitude50 - magnitude25
    vmax = max(float(np.max(magnitude50)), float(np.max(magnitude25)))
    limit = float(np.max(np.abs(difference)))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for axis, values, title in (
        (axes[0], magnitude50, "|in-plane grad T| 50 nm"),
        (axes[1], magnitude25, "|in-plane grad T| 25 nm"),
    ):
        image = cell_field(
            axis, x_edges, y_edges, values, coordinate_scale=1e6,
            cmap="magma", vmin=0.0, vmax=vmax,
        )
        axis.set(title=title, xlabel="x=b (µm)", ylabel="y=a (µm)")
        figure.colorbar(image, ax=axis)
    image = cell_field(
        axes[2], x_edges, y_edges, difference, coordinate_scale=1e6,
        cmap="coolwarm", vmin=-limit, vmax=limit,
    )
    axes[2].set(
        title="gradient-magnitude difference",
        xlabel="x=b (µm)", ylabel="y=a (µm)",
    )
    figure.colorbar(image, ax=axes[2])
    figure.savefig(args.report_dir / "W12_INTERFACE_GRADIENT_COMPARISON.png", dpi=190)
    plt.close(figure)
    print(f"regenerated 2 coordinate-faithful figures in {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
