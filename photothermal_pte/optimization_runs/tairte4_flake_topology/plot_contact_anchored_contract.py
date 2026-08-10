#!/usr/bin/env python3
"""Draw the approved contact-anchored TaIrTe4 topology contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    CONTRACT.validate()
    if CONTRACT.geometry_mode != "contact_anchored":
        raise RuntimeError("set TAIRTE4_TOPOLOGY_GEOMETRY=contact_anchored")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    ax = axes[0]
    ax.add_patch(Rectangle((-20, -20), 40, 40, facecolor="#d9ecff", edgecolor="#7b4ab5", linewidth=3, linestyle="--"))
    ax.add_patch(Rectangle((-12, -12), 24, 2, facecolor="black", edgecolor="black"))
    ax.add_patch(Rectangle((-12, 10), 24, 2, facecolor="black", edgecolor="black"))
    ax.add_patch(Rectangle((-12, -10), 24, 20, facecolor="0.5", edgecolor="tab:red", linewidth=2))
    ax.text(0, 0, "24 x 20 um design\ninitial rho=0.5", ha="center", va="center", color="white", fontsize=13)
    ax.text(0, 11, "fixed TaIrTe4 top contact strip; psi=1", ha="center", va="center", color="white")
    ax.text(0, -11, "fixed TaIrTe4 bottom contact strip; psi=0", ha="center", va="center", color="white")
    ax.annotate("finite Gaussian\nw0=8.5 um", xy=(0, 0), xytext=(14, 14), arrowprops={"arrowstyle": "->", "color": "green"}, color="green", ha="center")
    ax.axvline(-12, color="tab:cyan", linewidth=1)
    ax.axvline(12, color="tab:cyan", linewidth=1)
    ax.set(xlim=(-21, 21), ylim=(-21, 21), aspect="equal", xlabel="Lumerical x = crystal b (um)", ylabel="Lumerical y = crystal a (um)", title="Top view: no fixed left/right frame")

    ax = axes[1]
    ax.add_patch(Rectangle((-20, -4), 40, 8, facecolor="#d9ecff", edgecolor="#7b4ab5", linewidth=3, linestyle="--"))
    ax.add_patch(Rectangle((-20, -0.385), 40, 0.285, facecolor="#8bd3dd", edgecolor="black", label="285 nm SiO2"))
    ax.add_patch(Rectangle((-20, -4), 40, 3.615, facecolor="#406080", edgecolor="black", label="Si"))
    ax.add_patch(Rectangle((-12, -0.1), 24, 0.1, facecolor="0.45", edgecolor="black", label="100 nm TaIrTe4 topology sheet"))
    ax.annotate("Gaussian source z=2 um\npropagation -z", xy=(0, 0.2), xytext=(0, 2.2), arrowprops={"arrowstyle": "->", "color": "green"}, color="green", ha="center")
    ax.text(-19, 3.5, "six PML; 40 x 40 um optical domain", color="#7b4ab5", va="top")
    ax.set(xlim=(-21, 21), ylim=(-4.2, 4.2), xlabel="x or y (um)", ylabel="z (um)", title="Cross section; explicit SiO2/Si thermal stack")
    ax.legend(loc="lower right")
    fig.suptitle("Approved contact-anchored TaIrTe4 inverse-design geometry\nblack=rho=1 TaIrTe4, white=rho=0 void in all optimization plots")
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
