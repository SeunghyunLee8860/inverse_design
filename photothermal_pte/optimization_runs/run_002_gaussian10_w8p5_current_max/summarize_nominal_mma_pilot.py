#!/usr/bin/env python3
"""Publish accepted nominal MMA iterations and append-only run history."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


STATUS = "RUNNING_NOMINAL_MMA_PILOT"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--baseline-raw", type=Path, required=True)
    parser.add_argument("--proposal-result", type=Path, required=True)
    parser.add_argument("--proposal-raw", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--candidate-raw", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_directory.expanduser().resolve()
    results = run / "results"
    plots = run / "plots"
    checkpoints = run / "checkpoints"
    manifest_path = run / "manifests" / "RAW_ARTIFACT_MANIFEST.json"
    status_path = run / "STATUS.json"
    for path in (results, plots, checkpoints):
        path.mkdir(parents=True, exist_ok=True)

    baseline_result_path = args.baseline_result.expanduser().resolve()
    candidate_result_path = args.candidate_result.expanduser().resolve()
    proposal_result_path = args.proposal_result.expanduser().resolve()
    baseline = json.loads(baseline_result_path.read_text())
    candidate = json.loads(candidate_result_path.read_text())
    proposal = json.loads(proposal_result_path.read_text())
    if not baseline.get("passed") or not candidate.get("passed") or not proposal.get("passed"):
        raise RuntimeError("baseline, proposal, or candidate did not pass")
    base_raw_path = args.baseline_raw.expanduser().resolve()
    candidate_raw_path = args.candidate_raw.expanduser().resolve()
    proposal_raw_path = args.proposal_raw.expanduser().resolve()
    for path, expected in (
        (base_raw_path, baseline["raw_artifact"]["sha256"]),
        (candidate_raw_path, candidate["raw_artifact"]["sha256"]),
        (proposal_raw_path, proposal["raw_artifact"]["sha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"raw SHA mismatch: {path}")
    base_npz = np.load(base_raw_path)
    candidate_npz = np.load(candidate_raw_path)
    proposal_npz = np.load(proposal_raw_path)
    incident = float(baseline["incident_power_W"])
    scale = float(proposal["fixed_nondimensionalization"]["scale_W_per_A"])
    f0 = float(baseline["objective_A"])
    f1 = float(candidate["objective_A"])
    improvement = (f1 - f0) / abs(f0)
    if improvement <= 0.0:
        raise RuntimeError("candidate is not an accepted improving iteration")

    rows = []
    for iteration, record, data in ((0, baseline, base_npz), (1, candidate, candidate_npz)):
        rho = np.asarray(data["rho"], float)
        latent = np.asarray(data["latent"], float)
        objective = float(record["objective_A"])
        rows.append({
            "iteration": iteration,
            "accepted": True,
            "beta": record["beta"],
            "objective_A": objective,
            "objective_A_per_incident_W": objective / incident,
            "scaled_FOM": scale * objective / incident,
            "relative_improvement_from_iteration0": (objective - f0) / abs(f0),
            "P_Q_W": record["base_forward"]["P_Q_W"],
            "P_six_W": record["base_forward"]["P_six_W"],
            "optical_closure": record["gates"]["optical_closure"],
            "thermal_residual": record["gates"]["thermal_residual"],
            "thermal_energy_balance": record["gates"]["thermal_energy_balance"],
            "latent_min": float(np.min(latent)),
            "latent_max": float(np.max(latent)),
            "physical_density_min": float(np.min(rho)),
            "physical_density_max": float(np.max(rho)),
            "physical_density_mean": float(np.mean(rho)),
            "gray_fraction_0p05_0p95": float(np.mean((rho > 0.05) & (rho < 0.95))),
            "gradient_latent_L2_A": record["gradient_norms_A"]["latent"],
            "gradient_physical_L2_A": record["gradient_norms_A"]["physical"],
            "gradient_optical_physical_L2_A": record["gradient_norms_A"]["optical_physical"],
            "gradient_thermal_physical_L2_A": record["gradient_norms_A"]["thermal_physical"],
            "max_abs_latent_step": 0.0 if iteration == 0 else proposal["latent_max_abs_change"],
        })
    history_csv = results / "nominal_mma_iteration_history.csv"
    with history_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # Compact, resumable optimizer state is intentionally committed; large
    # Maxwell/thermal arrays remain external and SHA-pinned in the manifest.
    checkpoint_path = checkpoints / "iteration_001_accepted_state.npz"
    np.savez_compressed(
        checkpoint_path,
        latent=np.asarray(candidate_npz["latent"], np.float32),
        beta=np.asarray(candidate["beta"]),
        objective_A=np.asarray(f1),
        incident_power_W=np.asarray(incident),
        fixed_objective_scale_W_per_A=np.asarray(scale),
        iteration=np.asarray(1),
    )

    iterations = [row["iteration"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    axes[0].plot(iterations, [row["objective_A_per_incident_W"] for row in rows], "o-", linewidth=2)
    axes[0].set_xlabel("accepted iteration")
    axes[0].set_ylabel(r"$I_{PTE}/P_{inc}$ (A/W)")
    axes[0].set_title(f"FOM history (+{100.0*improvement:.2f}% at iteration 1)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(iterations, [row["scaled_FOM"] for row in rows], "o-")
    axes[1].set_xlabel("accepted iteration")
    axes[1].set_ylabel("fixed scaled FOM")
    axes[1].set_title(r"$10^{12}$ W/A × $I/P_{inc}$")
    axes[1].grid(alpha=0.3)
    axes[2].bar(["optical", "thermal"], [candidate["gradient_norms_A"]["optical_physical"], candidate["gradient_norms_A"]["thermal_physical"]])
    axes[2].set_yscale("log")
    axes[2].set_ylabel("physical-density gradient L2 (A)")
    axes[2].set_title("Iteration-1 gradient components")
    axes[2].grid(axis="y", alpha=0.3)
    fom_plot = plots / "iteration_vs_fom.png"
    fig.savefig(fom_plot, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    fields = (
        (base_npz["latent"], "iteration 0 latent", "viridis"),
        (base_npz["rho"], "iteration 0 physical density", "viridis"),
        (candidate_npz["latent"], "iteration 1 latent", "viridis"),
        (candidate_npz["rho"], "iteration 1 physical density", "viridis"),
        (candidate_npz["gradient_optical_A"], "iteration 1 optical gradient", "coolwarm"),
        (candidate_npz["gradient_thermal_A"], "iteration 1 thermal gradient", "coolwarm"),
        (candidate_npz["gradient_physical_A"], "iteration 1 physical gradient", "coolwarm"),
        (candidate_npz["gradient_latent_A"], "iteration 1 latent gradient", "coolwarm"),
    )
    for ax, (value, title, cmap) in zip(axes.ravel(), fields):
        image = ax.imshow(np.asarray(value, float).T, origin="lower", cmap=cmap, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("node x")
        ax.set_ylabel("node y")
        fig.colorbar(image, ax=ax, fraction=0.046)
    design_plot = plots / "iteration_001_design_and_gradients.png"
    fig.savefig(design_plot, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    labels = ["latent\nstep", "mean rho\nchange", "gray fraction\nchange"]
    values = [
        float(proposal["latent_max_abs_change"]) / 0.02,
        abs(rows[1]["physical_density_mean"] - rows[0]["physical_density_mean"]) / 0.01,
        abs(rows[1]["gray_fraction_0p05_0p95"] - rows[0]["gray_fraction_0p05_0p95"]),
    ]
    axes[0].bar(labels, values)
    axes[0].axhline(1.0, color="black", linestyle="--", label="declared scale")
    axes[0].set_title("Design/constraint diagnostics")
    axes[0].legend()
    axes[1].bar(["closure", "residual", "energy"], [candidate["gates"]["optical_closure"] / 0.005, candidate["gates"]["thermal_residual"] / 1.0e-8, candidate["gates"]["thermal_energy_balance"] / 0.01])
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("value / gate")
    axes[1].set_title("Physics gate margins")
    axes[2].bar(["P_Q", "P_six"], [candidate["base_forward"]["P_Q_W"], candidate["base_forward"]["P_six_W"]])
    axes[2].set_ylabel("power (W)")
    axes[2].set_title("Iteration-1 optical closure")
    constraint_plot = plots / "iteration_001_constraints_and_physics.png"
    fig.savefig(constraint_plot, dpi=180)
    plt.close(fig)

    generated = datetime.now(timezone.utc).isoformat()
    summary = {
        "status": STATUS,
        "generated_at_utc": generated,
        "optimization_started": True,
        "accepted_iterations": 1,
        "current_beta": candidate["beta"],
        "scenario": "grown_grown nominal",
        "signed_objective": "+I_PTE/P_incident",
        "baseline_objective_A": f0,
        "iteration_001_objective_A": f1,
        "relative_improvement": improvement,
        "fixed_nondimensionalization": proposal["fixed_nondimensionalization"],
        "constraints": proposal["constraints"],
        "history": rows,
        "checkpoint": artifact(checkpoint_path),
        "raw_artifacts": {
            "baseline_result": artifact(baseline_result_path),
            "baseline_NPZ": artifact(base_raw_path),
            "proposal_result": artifact(proposal_result_path),
            "proposal_NPZ": artifact(proposal_raw_path),
            "iteration_001_result": artifact(candidate_result_path),
            "iteration_001_NPZ": artifact(candidate_raw_path),
        },
        "warnings": [
            "continuous beta=2 nominal pilot; not a final binary design",
            "uniform-45-degree weighting surrogate; not a full electrode simulation",
            "grown/grown is one named interface scenario, not a final experiment prediction",
            "exact binary DRC and robust four-interface evaluation remain",
        ],
    }
    summary_path = results / "nominal_mma_pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path = results / "NOMINAL_MMA_PILOT_REPORT.md"
    report_path.write_text(
        "# Run 002 nominal MMA pilot\n\n"
        f"Status: `{STATUS}`\n\n"
        f"The first actual GPU-Maxwell/CUDA-thermal MMA candidate is accepted. "
        f"The signed PTE objective increased from `{f0:.12e} A` to `{f1:.12e} A`, "
        f"a `{100.0*improvement:.6f}%` improvement.\n\n"
        "The optimizer uses 0≤latent≤1, a finite nonperiodic 500 nm conic filter, "
        "beta=2 tanh projection, and a fixed objective nondimensionalization of "
        f"`{scale:.3e} W/A × I/P_incident`. It does not dynamically rescale gradients. "
        "No volume or symmetry constraint is imposed in this nominal pilot. Exact "
        "binary 500 nm solid/void DRC is still required before fabrication promotion.\n\n"
        "This is not a final optimized structure or an experimental-current prediction. "
        "It is iteration 1 of the grown/grown, +I, uniform-45° weighting-surrogate stage.\n"
    )

    manifest = json.loads(manifest_path.read_text())
    manifest["nominal_mma_pilot"] = {"status": STATUS, "raw_artifacts_committed_to_git": False, **summary["raw_artifacts"], "checkpoint": summary["checkpoint"]}
    manifest["current_promoted_status"] = STATUS
    manifest["current_promoted_at_utc"] = generated
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    status = json.loads(status_path.read_text())
    status.update({
        "status": STATUS,
        "last_updated_utc": generated,
        "optimization_started": True,
        "message": f"Nominal beta=2 MMA iteration 1 accepted; +I PTE objective improved by {100.0*improvement:.3f}%. Exact binary DRC and robust interface scenarios remain.",
    })
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
