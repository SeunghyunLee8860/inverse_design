#!/usr/bin/env python3
"""Offline validation for the Run 003 exact-disk constraint recovery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from optimization_support import (
    ProductionDensityMapping,
    constraint_contract,
    constraint_values_and_gradients,
    design_metrics,
    exact_binary_audit,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--constraint-device", default="cpu")
    args = parser.parse_args()
    os.environ["RUN003_CONSTRAINT_TORCH_DEVICE"] = args.constraint_device
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.checkpoint)
    latent = np.asarray(data["latent"], float)
    beta = float(data["beta"])
    mapping = ProductionDensityMapping()
    rho = mapping.physical(latent, beta)
    values, gradients, local = constraint_values_and_gradients(
        latent, beta, mapping
    )
    baseline_exact = exact_binary_audit(rho, mapping.spacing_m)

    rng = np.random.default_rng(320085)
    direction = rng.standard_normal(latent.shape)
    direction /= np.linalg.norm(direction)
    ad = gradients.reshape(2, -1) @ direction.ravel()
    dot_tests = []
    for step in (1.0e-3, 5.0e-4, 2.5e-4):
        plus = constraint_values_and_gradients(
            latent + step * direction, beta, mapping
        )[0]
        minus = constraint_values_and_gradients(
            latent - step * direction, beta, mapping
        )[0]
        fd = (plus - minus) / (2.0 * step)
        dot_tests.append({
            "step": step,
            "AD": ad.tolist(),
            "FD": fd.tolist(),
            "relative_error": (
                np.abs(ad - fd) / np.maximum(np.abs(fd), 1.0e-30)
            ).tolist(),
        })

    combined = np.sum(gradients, axis=0)
    descent = []
    candidate_rhos = []
    for move in (0.0025, 0.005, 0.01, 0.02):
        candidate = np.clip(
            latent - move * combined / np.max(np.abs(combined)), 0.0, 1.0
        )
        candidate_metrics, candidate_arrays = design_metrics(
            candidate, beta, mapping
        )
        candidate_exact = candidate_metrics["exact_binary_audit"]
        candidate_rhos.append(candidate_arrays["rho"])
        descent.append({
            "maximum_normalized_latent_move": move,
            "constraint_values": [
                candidate_metrics["solid_constraint"],
                candidate_metrics["void_constraint"],
            ],
            "solid_bad_cells": candidate_exact["solid_bad_cell_count"],
            "void_bad_cells": candidate_exact["void_bad_cell_count"],
            "rho_rms_change": float(np.sqrt(np.mean((candidate_arrays["rho"] - rho) ** 2))),
            "rho_max_change": float(np.max(np.abs(candidate_arrays["rho"] - rho))),
        })

    payload = {
        "status": "VALIDATED_OFFLINE_EXACT_DISK_CONSTRAINT_RECOVERY",
        "passed": bool(
            max(max(row["relative_error"]) for row in dot_tests) < 1.0e-5
            and descent[-1]["solid_bad_cells"] < baseline_exact["solid_bad_cell_count"]
            and descent[-1]["void_bad_cells"] < baseline_exact["void_bad_cell_count"]
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "size_bytes": args.checkpoint.stat().st_size,
            "sha256": sha256(args.checkpoint),
        },
        "beta": beta,
        "constraint_contract": constraint_contract(beta),
        "constraint_device": args.constraint_device,
        "baseline": {
            "constraint_values": values.tolist(),
            "solid_bad_cells": baseline_exact["solid_bad_cell_count"],
            "void_bad_cells": baseline_exact["void_bad_cell_count"],
        },
        "dot_tests": dot_tests,
        "offline_descent_is_not_an_accepted_design": True,
        "offline_descent": descent,
    }
    output_json = args.output_dir / "disk_constraint_recovery_validation.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n")

    extent = [-9.3, 9.3, -9.3, 9.3]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    panels = (
        (rho, "preserved beta=32 density", "viridis"),
        (baseline_exact["bad_solid"], "exact solid bad cells", "Reds"),
        (baseline_exact["bad_void"], "exact void bad cells", "Blues"),
        (local["solid_penalty_field"], "disk-aligned solid residual", "magma"),
        (local["void_penalty_field"], "disk-aligned void residual", "magma"),
        (candidate_rhos[-1] - rho, "offline move=0.02 density change", "coolwarm"),
    )
    for ax, (field, title, cmap) in zip(axes.ravel(), panels):
        image = ax.imshow(np.asarray(field).T, origin="lower", extent=extent, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.colorbar(image, ax=ax, fraction=0.046)
    figure = args.output_dir / "disk_constraint_recovery_fields.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    moves = [0.0, *[row["maximum_normalized_latent_move"] for row in descent]]
    solid = [baseline_exact["solid_bad_cell_count"], *[row["solid_bad_cells"] for row in descent]]
    void = [baseline_exact["void_bad_cell_count"], *[row["void_bad_cells"] for row in descent]]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(moves, solid, "o-", label="solid")
    ax.plot(moves, void, "s-", label="void")
    ax.set_xlabel("offline maximum normalized latent move")
    ax.set_ylabel("exact 500 nm bad-cell count")
    ax.set_title("Disk-aligned constraint descent (diagnostic only)")
    ax.grid(alpha=0.3)
    ax.legend()
    descent_figure = args.output_dir / "disk_constraint_recovery_descent.png"
    fig.savefig(descent_figure, dpi=180)
    plt.close(fig)

    report = args.output_dir / "DISK_CONSTRAINT_RECOVERY_REPORT.md"
    report.write_text(
        "# Run 003 exact-disk constraint recovery\n\n"
        f"Status: `{payload['status']}`; passed: `{payload['passed']}`.\n\n"
        f"Preserved checkpoint: `{payload['checkpoint']['sha256']}`.\n\n"
        f"Baseline exact bad cells were `{baseline_exact['solid_bad_cell_count']}` solid and "
        f"`{baseline_exact['void_bad_cell_count']}` void. The largest offline diagnostic "
        f"step reduced them to `{descent[-1]['solid_bad_cells']}` / "
        f"`{descent[-1]['void_bad_cells']}`. This diagnostic density was not accepted, "
        "was not sent to Maxwell or thermal solvers, and did not replace the checkpoint.\n\n"
        f"Maximum centered-FD relative error: "
        f"`{max(max(row['relative_error']) for row in dot_tests):.6e}`.\n"
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
