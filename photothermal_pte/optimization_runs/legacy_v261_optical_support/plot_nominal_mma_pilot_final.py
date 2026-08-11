#!/usr/bin/env python3
"""Create final comparison plots for the accepted nominal beta=2 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_directory.expanduser().resolve()
    summary = json.loads((run / "results" / "nominal_mma_pilot_summary.json").read_text())
    if int(summary["accepted_iterations"]) != 5:
        raise RuntimeError("final pilot plot requires five accepted iterations")
    initial_path = Path(summary["raw_artifacts"]["baseline_NPZ"]["path"])
    final_path = Path(summary["raw_artifacts"]["iteration_005"]["evaluation_NPZ"]["path"])
    initial = np.load(initial_path)
    final = np.load(final_path)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.1), constrained_layout=True)
    fields = (
        (initial["latent"], "initial latent", "viridis", None),
        (initial["filtered"], "initial filtered", "viridis", None),
        (initial["rho"], "initial physical density", "viridis", (0.0, 1.0)),
        (final["latent"], "iteration 5 latent", "viridis", None),
        (final["filtered"], "iteration 5 filtered", "viridis", None),
        (final["rho"], "iteration 5 physical density", "viridis", (0.0, 1.0)),
    )
    for ax, (value, title, cmap, limits) in zip(axes.ravel(), fields):
        kwargs = {} if limits is None else {"vmin": limits[0], "vmax": limits[1]}
        image = ax.imshow(np.asarray(value, float).T, origin="lower", cmap=cmap, aspect="equal", **kwargs)
        ax.set_title(title)
        ax.set_xlabel("node x (50 nm spacing)")
        ax.set_ylabel("node y (50 nm spacing)")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle("Nominal beta=2 pilot: initial versus accepted iteration 5")
    fig.savefig(run / "plots" / "nominal_pilot_initial_vs_final.png", dpi=180)
    plt.close(fig)

    rho_initial = np.asarray(initial["rho"], float)
    rho_final = np.asarray(final["rho"], float)
    delta_rho = rho_final - rho_initial
    delta_limit = float(np.max(np.abs(delta_rho)))
    mid_x = rho_initial.shape[0] // 2
    mid_y = rho_initial.shape[1] // 2
    coordinate_um = (np.arange(rho_initial.shape[0]) - mid_x) * 0.05
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    image = axes[0].imshow(delta_rho.T, origin="lower", cmap="coolwarm", aspect="equal", vmin=-delta_limit, vmax=delta_limit)
    axes[0].set_title(r"physical-density change $\rho_5-\rho_0$")
    axes[0].set_xlabel("node x (50 nm spacing)")
    axes[0].set_ylabel("node y (50 nm spacing)")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    axes[1].plot(coordinate_um, rho_initial[:, mid_y], label="initial")
    axes[1].plot(coordinate_um, rho_final[:, mid_y], label="iteration 5")
    axes[1].set_xlabel("x at y=0 (um)")
    axes[1].set_ylabel("physical density")
    axes[1].set_title("Horizontal center line")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].plot(coordinate_um, rho_initial[mid_x, :], label="initial")
    axes[2].plot(coordinate_um, rho_final[mid_x, :], label="iteration 5")
    axes[2].set_xlabel("y at x=0 (um)")
    axes[2].set_ylabel("physical density")
    axes[2].set_title("Vertical center line")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig.savefig(run / "plots" / "nominal_pilot_density_change.png", dpi=180)
    plt.close(fig)

    rows = summary["history"]
    iterations = np.asarray([row["iteration"] for row in rows], int)
    objective = np.asarray([row["objective_A_per_incident_W"] for row in rows], float)
    relative = np.asarray([row["relative_improvement_from_iteration0"] for row in rows], float)
    optical_gradient = np.asarray([row["gradient_optical_physical_L2_A"] for row in rows], float)
    thermal_gradient = np.asarray([row["gradient_thermal_physical_L2_A"] for row in rows], float)
    total_gradient = np.asarray([row["gradient_physical_L2_A"] for row in rows], float)
    latent_gradient = np.asarray([row["gradient_latent_L2_A"] for row in rows], float)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    axes[0, 0].plot(iterations, objective, "o-", linewidth=2)
    axes[0, 0].set_xlabel("accepted iteration")
    axes[0, 0].set_ylabel(r"$I_{PTE}/P_{inc}$ (A/W)")
    axes[0, 0].set_title(f"Actual nonlinear FOM (+{100.0 * relative[-1]:.2f}% total)")
    axes[0, 0].grid(alpha=0.3)
    per_step = np.zeros_like(relative)
    per_step[1:] = objective[1:] / objective[:-1] - 1.0
    axes[0, 1].bar(iterations[1:], 100.0 * per_step[1:])
    axes[0, 1].set_xlabel("accepted iteration")
    axes[0, 1].set_ylabel("improvement from previous (%)")
    axes[0, 1].set_title("Per-iteration actual improvement")
    axes[0, 1].grid(axis="y", alpha=0.3)
    axes[1, 0].plot(iterations, optical_gradient, "o-", label="optical physical")
    axes[1, 0].plot(iterations, thermal_gradient, "o-", label="thermal physical")
    axes[1, 0].plot(iterations, total_gradient, "o-", label="combined physical")
    axes[1, 0].plot(iterations, latent_gradient, "o-", label="full latent")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("accepted iteration")
    axes[1, 0].set_ylabel("gradient L2 norm (A)")
    axes[1, 0].set_title("Gradient decomposition")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    closure = np.asarray([row["optical_closure"] / 0.005 for row in rows], float)
    residual = np.asarray([row["thermal_residual"] / 1.0e-8 for row in rows], float)
    energy = np.asarray([row["thermal_energy_balance"] / 0.01 for row in rows], float)
    axes[1, 1].plot(iterations, closure, "o-", label="optical closure / gate")
    axes[1, 1].plot(iterations, residual, "o-", label="thermal residual / gate")
    axes[1, 1].plot(iterations, energy, "o-", label="energy balance / gate")
    axes[1, 1].axhline(1.0, color="black", linestyle="--", label="gate")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("accepted iteration")
    axes[1, 1].set_ylabel("value / gate")
    axes[1, 1].set_title("Physics gate margins")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    fig.savefig(run / "plots" / "nominal_pilot_final_summary.png", dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
