#!/usr/bin/env python3
"""Regenerate one saved thermal/PTE panel using exact FVM cell edges."""

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
    strict_centered_xy_mask,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input_npz, allow_pickle=False) as raw:
        x_edges = np.asarray(raw["x_edges_m"], float)
        y_edges = np.asarray(raw["y_edges_m"], float)
        dz = np.diff(np.asarray(raw["z_edges_m"], float))
        areal_q = np.sum(np.asarray(raw["Q_W_m3"], float) * dz[None, None, :], axis=2)
        temperature = np.asarray(raw["temperature_flake_average_K"], float)
        flake_xy = np.any(np.asarray(raw["flake_mask"], bool), axis=2)
        if "grad_T_normal_K_m" in raw.files:
            straight = True
            images = (
                (areal_q, "absorbed areal power (W/m²)", "inferno"),
                (temperature, "TaIrTe4 ΔT (K)", "inferno"),
                (np.abs(raw["grad_T_normal_K_m"]), "|edge-normal ∂T/∂n| (K/m)", "magma"),
                (raw["grad_T_magnitude_K_m"], "|in-plane ∇T| (K/m)", "magma"),
            )
        else:
            straight = False
            images = (
                (areal_q, "absorbed areal power (W/m²)", "inferno"),
                (temperature, "TaIrTe4 ΔT (K)", "inferno"),
                (raw["weighting_potential"], "weighting potential ψ", "viridis"),
                (raw["shockley_ramo_integrand_A_m2"], "PTE collection integrand", "RdBu_r"),
            )
        images = tuple((np.asarray(a, float), b, c) for a, b, c in images)
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for axis, (values, title, cmap) in zip(axes.flat, images):
        if "∂T/∂n" in title or "∇T" in title:
            values = np.where(strict_centered_xy_mask(flake_xy), values, np.nan)
        image = cell_field(
            axis, x_edges, y_edges, values, coordinate_scale=1e6, cmap=cmap
        )
        if straight:
            bounds = [x_edges[0] * 1e6, x_edges[-1] * 1e6]
            axis.plot(bounds, bounds, "c--", lw=0.8)
        axis.set(title=title, xlabel="lab x = b (µm)", ylabel="lab y = a (µm)")
        figure.colorbar(image, ax=axis)
    figure.suptitle(
        "Straight 45° edge thermal control" if straight
        else "Device-A IR thermal/PTE diagnostic"
    )
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_png, dpi=200)
    plt.close(figure)
    print(args.output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
