#!/usr/bin/env python3
"""Plot the immutable +45-degree device and fixed crystal-axis contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.rotated_device import (
    device_to_crystal_coordinates,
)


def vertices(u0: float, u1: float, v0: float, v1: float) -> np.ndarray:
    u = np.asarray((u0, u1, u1, u0), dtype=float)
    v = np.asarray((v0, v0, v1, v1), dtype=float)
    x, y = device_to_crystal_coordinates(u, v)
    return np.column_stack((x, y))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if CONTRACT.geometry_mode != "diagonal_45_contact_anchored":
        raise RuntimeError("select diagonal_45_contact_anchored")

    half = 12.0
    inner = 10.0
    flake = vertices(-half, half, -half, half)
    low = vertices(-half, -inner, -half, half)
    high = vertices(inner, half, -half, half)

    figure, axis = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    axis.add_patch(
        Polygon(flake, closed=True, facecolor="#173f4f", edgecolor="#082f3c", lw=2)
    )
    for footprint in (low, high):
        axis.add_patch(
            Polygon(
                footprint,
                closed=True,
                facecolor="#f47b32",
                edgecolor="#d85f1f",
                lw=1.5,
            )
        )

    arrow = 7.0
    axis.annotate(
        "",
        xy=(0.0, arrow),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": "#d62728"},
    )
    axis.annotate(
        "",
        xy=(arrow, 0.0),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": "#2ca02c"},
    )
    axis.text(0.5, arrow + 0.4, "a (fixed)", color="#d62728", fontsize=12)
    axis.text(arrow + 0.4, 0.5, "b (fixed)", color="#2ca02c", fontsize=12)
    axis.text(0.0, 18.5, "24 x 24 um TaIrTe4, rotated +45 deg", ha="center", fontsize=13)
    axis.text(0.0, -19.2, "50 nm Au on opposite full-edge 2 um strips", ha="center", fontsize=11)
    axis.set_aspect("equal")
    axis.set_xlim(-21.0, 21.0)
    axis.set_ylim(-21.0, 21.0)
    axis.set_xlabel("global x = crystal b (um)")
    axis.set_ylabel("global y = crystal a (um)")
    axis.set_title("Run 059/060 geometry contract")
    axis.grid(alpha=0.16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
