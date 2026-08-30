#!/usr/bin/env python3
"""Reclassify and summarize the production-width FDTDX Au AD-FD smoke.

The raw AD and FD values are never changed.  A directional derivative is
classified as strong only when its magnitude is at least 5% of the full
gradient L2 norm.  Near-null directions are judged with
``abs(AD-FD) / ||gradient||_2`` because their local relative error is
ill-conditioned.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STRONG_THRESHOLD = 0.05


def summarize(result_dir: Path) -> dict[str, object]:
    json_path = result_dir / "fdtdx_production_width_nonuniform_au_gradient_smoke.json"
    csv_path = result_dir / "fdtdx_production_width_nonuniform_au_gradient_smoke.csv"
    result = json.loads(json_path.read_text(encoding="utf-8"))
    grad_l2 = float(result["baseline"]["gradient_l2_W"])

    for row in result["directions"]:
        scale = max(abs(row["ad_W_per_unit_direction"]), abs(row["fd_W_per_unit_direction"]))
        row["strong_direction"] = bool(scale >= STRONG_THRESHOLD * grad_l2)

    finest = [row for row in result["directions"] if row["h"] == 0.005]
    strong_errors = [row["strong_relative_error"] for row in finest if row["strong_direction"]]
    max_strong_error = max(strong_errors, default=0.0)
    max_normalized_error = max(row["gradient_l2_normalized_error"] for row in finest)
    result["direction_classification"] = {
        "strong_threshold_fraction_of_gradient_l2": STRONG_THRESHOLD,
        "near_null_metric": "abs(AD-FD)/||gradient||_2",
        "no_empirical_gradient_rescaling": True,
    }
    result["gates"]["finest_strong_direction_error_lt_1pct"] = max_strong_error < 0.01
    result["gates"]["finest_gradient_l2_normalized_error_lt_1pct"] = max_normalized_error < 0.01
    passed = all(result["gates"].values())
    result["status"] = (
        "VALIDATED_FDTDX_PRODUCTION_WIDTH_NONUNIFORM_AU_GRADIENT_SMOKE"
        if passed
        else "FAILED_FDTDX_PRODUCTION_WIDTH_NONUNIFORM_AU_GRADIENT_SMOKE"
    )
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    fieldnames = list(result["directions"][0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["directions"])

    x = np.linspace(-1.0, 1.0, 20)[:, None]
    y = np.linspace(-1.0, 1.0, 20)[None, :]
    rho = 0.52 + 0.07 * np.cos(0.8 * np.pi * x) * np.cos(0.65 * np.pi * y) + 0.02 * x
    rows = result["directions"]
    ad = np.asarray([row["ad_W_per_unit_direction"] for row in rows])
    fd = np.asarray([row["fd_W_per_unit_direction"] for row in rows])
    labels = [f"{row['direction']}\nh={row['h']:g}" for row in rows]
    normalized_pct = 100.0 * np.asarray([row["gradient_l2_normalized_error"] for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), constrained_layout=True)
    im = axes[0].imshow(rho.T, origin="lower", extent=(-5, 5, -5, 5), cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("20×20 nonuniform Au density")
    axes[0].set_xlabel("x=b (µm)")
    axes[0].set_ylabel("y=a (µm)")
    fig.colorbar(im, ax=axes[0], label="ρ")

    lim = 1.12 * max(np.max(np.abs(ad)), np.max(np.abs(fd)), 1e-30)
    axes[1].plot([-lim, lim], [-lim, lim], "k--", label="ideal AD=FD")
    for idx, row in enumerate(rows):
        marker = "o" if row["strong_direction"] else "s"
        axes[1].scatter(fd[idx], ad[idx], marker=marker, s=65, label=labels[idx])
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("FD directional derivative (W)")
    axes[1].set_ylabel("AD directional derivative (W)")
    axes[1].set_title("Production-width optical AD–FD")
    axes[1].legend(fontsize=7, loc="best")

    colors = ["tab:blue" if row["strong_direction"] else "tab:orange" for row in rows]
    axes[2].bar(np.arange(len(rows)), normalized_pct, color=colors)
    axes[2].axhline(1.0, color="k", linestyle="--", label="1% gate")
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-4, 2.0)
    axes[2].set_xticks(np.arange(len(rows)), labels, rotation=25, ha="right")
    axes[2].set_ylabel("|AD−FD| / ||gradient||₂ (%)")
    axes[2].set_title("Well-conditioned error metric")
    axes[2].legend()
    fig.suptitle(result["status"].replace("_", " "), fontsize=13)
    plot_path = result_dir / "fdtdx_production_width_nonuniform_au_gradient_smoke.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    baseline = result["baseline"]
    runtime = result["runtime"]
    report = f"""# FDTDX production-width nonuniform-Au gradient smoke

Status: **{result['status']}**

This checkpoint validates the optical total-absorbed-power gradient for a
nonuniform 20×20 Au-density field on the production-width 48 µm × 48 µm
FDTDX domain. It does not validate thermal, electrical, PTE, or optimization
gradients.

## Contract

- wavelength: 10 µm; scalar Gaussian waist: 8.5 µm
- fixed TaIrTe4: 20 µm × 20 µm × 100 nm
- design Au: 10 µm × 10 µm × 50 nm
- latent density: 20×20 at 500 nm; Yee sampling: 100×100 at 100 nm
- passive material relaxation: Drude coupling strength `s(rho)=rho^3`
- FDTDX source commit: `{result['audit']['software']['fdtdx_source_commit']}`
- checkpointed AD: {result['audit']['numerics']['gradient_checkpoints']} checkpoints
- no clipping, smoothing, gain, or result/gradient rescaling

## Optical checks

- P_Q: {baseline['P_Q_W']:.9e} W
- empty-subtracted six-face closure: {100.0 * baseline['Q_flux_closure_relative']:.6f}%
- late-window change: {100.0 * baseline['late_window_relative_change']:.6f}%
- gradient L2 norm: {baseline['gradient_l2_W']:.9e} W

## AD–FD interpretation

A direction is called strong only if `max(|AD|,|FD|) >= 0.05 ||gradient||_2`.
Near-null directions retain their raw local relative error, but are gated with
`|AD-FD|/||gradient||_2`; this avoids division by a small directional
derivative. No empirical gradient rescaling is used.

- finest strong-direction relative error: {100.0 * max_strong_error:.6f}%
- finest all-direction gradient-L2-normalized error: {100.0 * max_normalized_error:.6f}%

## Runtime

- XLA compile: {runtime['compile_seconds']:.3f} s
- one value+gradient: {runtime['ad_seconds']:.3f} s
- four FD forwards: {runtime['four_fd_forward_seconds']:.3f} s

The aborted four-checkpoint attempt was a performance-contract failure, not a
physics or gradient failure. Sixteen checkpoints completed the same production
AD in {runtime['ad_seconds'] / 60.0:.2f} minutes while staying within GPU memory.
"""
    (result_dir / "FDTDX_PRODUCTION_WIDTH_NONUNIFORM_AU_GRADIENT_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    performance = {
        "scope": "checkpoint time-memory diagnostic; identical physics/grid/objective",
        "four_checkpoint_attempt": {
            "status": "ABORTED_PERFORMANCE_ONLY_AFTER_60_MINUTES",
            "elapsed_seconds_before_termination": 3749,
            "observed_gpu_memory_MiB": 11244,
            "result_generated": False,
            "physics_failure": False,
        },
        "sixteen_checkpoint_run": {
            "status": "COMPLETED",
            "compile_seconds": runtime["compile_seconds"],
            "value_and_gradient_seconds": runtime["ad_seconds"],
            "observed_gpu_memory_MiB": 35820,
            "result_generated": True,
        },
        "decision": "Use 16 checkpoints as the current production-width baseline on a 49-GB GPU.",
    }
    (result_dir / "fdtdx_checkpoint_performance_diagnostic.json").write_text(
        json.dumps(performance, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.result_dir)
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, indent=2))
    return 0 if result["status"].startswith("VALIDATED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
