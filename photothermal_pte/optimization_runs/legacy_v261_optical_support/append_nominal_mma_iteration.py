#!/usr/bin/env python3
"""Append one improving nominal MMA candidate to the committed run history."""

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
    parser.add_argument("--iteration", type=int, required=True)
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
    summary_path = results / "nominal_mma_pilot_summary.json"
    history_csv = results / "nominal_mma_iteration_history.csv"
    report_path = results / "NOMINAL_MMA_PILOT_REPORT.md"
    manifest_path = run / "manifests" / "RAW_ARTIFACT_MANIFEST.json"
    status_path = run / "STATUS.json"
    previous = json.loads(summary_path.read_text())
    if args.iteration != int(previous["accepted_iterations"]) + 1:
        raise RuntimeError("iteration is not the next append-only index")
    proposal_result_path = args.proposal_result.expanduser().resolve()
    proposal_raw_path = args.proposal_raw.expanduser().resolve()
    candidate_result_path = args.candidate_result.expanduser().resolve()
    candidate_raw_path = args.candidate_raw.expanduser().resolve()
    proposal = json.loads(proposal_result_path.read_text())
    candidate = json.loads(candidate_result_path.read_text())
    if not proposal.get("passed") or not candidate.get("passed"):
        raise RuntimeError("proposal or candidate did not pass")
    if int(proposal["optimizer_iteration_proposed"]) != args.iteration:
        raise RuntimeError("proposal iteration does not match requested append index")
    if sha256(proposal_raw_path) != proposal["raw_artifact"]["sha256"]:
        raise RuntimeError("proposal raw SHA mismatch")
    if sha256(candidate_raw_path) != candidate["raw_artifact"]["sha256"]:
        raise RuntimeError("candidate raw SHA mismatch")
    proposal_data = np.load(proposal_raw_path)
    data = np.load(candidate_raw_path)
    if not np.array_equal(np.asarray(proposal_data["latent"], float), np.asarray(data["latent"], float)):
        raise RuntimeError("evaluated latent design differs from the MMA proposal")
    if not np.array_equal(np.asarray(proposal_data["rho"], float), np.asarray(data["rho"], float)):
        raise RuntimeError("evaluated physical density differs from the MMA proposal")
    incident = float(candidate["incident_power_W"])
    scale = float(proposal["fixed_nondimensionalization"]["scale_W_per_A"])
    rows = list(previous["history"])
    prior_objective = float(rows[-1]["objective_A"])
    baseline_objective = float(rows[0]["objective_A"])
    objective = float(candidate["objective_A"])
    improvement_prior = (objective - prior_objective) / abs(prior_objective)
    improvement_total = (objective - baseline_objective) / abs(baseline_objective)
    if improvement_prior <= 0.0:
        raise RuntimeError("candidate did not improve the previous accepted objective")
    latent = np.asarray(data["latent"], float)
    rho = np.asarray(data["rho"], float)
    row = {
        "iteration": args.iteration,
        "accepted": True,
        "beta": candidate["beta"],
        "objective_A": objective,
        "objective_A_per_incident_W": objective / incident,
        "scaled_FOM": scale * objective / incident,
        "relative_improvement_from_iteration0": improvement_total,
        "P_Q_W": candidate["base_forward"]["P_Q_W"],
        "P_six_W": candidate["base_forward"]["P_six_W"],
        "optical_closure": candidate["gates"]["optical_closure"],
        "thermal_residual": candidate["gates"]["thermal_residual"],
        "thermal_energy_balance": candidate["gates"]["thermal_energy_balance"],
        "latent_min": float(np.min(latent)),
        "latent_max": float(np.max(latent)),
        "physical_density_min": float(np.min(rho)),
        "physical_density_max": float(np.max(rho)),
        "physical_density_mean": float(np.mean(rho)),
        "gray_fraction_0p05_0p95": float(np.mean((rho > 0.05) & (rho < 0.95))),
        "gradient_latent_L2_A": candidate["gradient_norms_A"]["latent"],
        "gradient_physical_L2_A": candidate["gradient_norms_A"]["physical"],
        "gradient_optical_physical_L2_A": candidate["gradient_norms_A"]["optical_physical"],
        "gradient_thermal_physical_L2_A": candidate["gradient_norms_A"]["thermal_physical"],
        "max_abs_latent_step": proposal["latent_max_abs_change"],
    }
    rows.append(row)
    with history_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    checkpoint_path = checkpoints / f"iteration_{args.iteration:03d}_accepted_state.npz"
    np.savez_compressed(
        checkpoint_path,
        latent=np.asarray(latent, np.float32),
        beta=np.asarray(candidate["beta"]),
        objective_A=np.asarray(objective),
        incident_power_W=np.asarray(incident),
        fixed_objective_scale_W_per_A=np.asarray(scale),
        iteration=np.asarray(args.iteration),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    x = [int(value["iteration"]) for value in rows]
    axes[0].plot(x, [float(value["objective_A_per_incident_W"]) for value in rows], "o-", linewidth=2)
    axes[0].set_xlabel("accepted iteration")
    axes[0].set_ylabel(r"$I_{PTE}/P_{inc}$ (A/W)")
    axes[0].set_title(f"FOM history (+{100.0*improvement_total:.2f}% total)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, [float(value["scaled_FOM"]) for value in rows], "o-")
    axes[1].set_xlabel("accepted iteration")
    axes[1].set_ylabel("fixed scaled FOM")
    axes[1].set_title(r"$10^{12}$ W/A × $I/P_{inc}$")
    axes[1].grid(alpha=0.3)
    axes[2].plot(x, [float(value["physical_density_mean"]) for value in rows], "o-", label="mean rho")
    axes[2].plot(x, [float(value["gray_fraction_0p05_0p95"]) for value in rows], "s-", label="gray fraction")
    axes[2].set_xlabel("accepted iteration")
    axes[2].set_title("Design metrics")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig.savefig(plots / "iteration_vs_fom.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    fields = (
        (data["latent"], f"iteration {args.iteration} latent", "viridis"),
        (data["filtered"], f"iteration {args.iteration} filtered", "viridis"),
        (data["rho"], f"iteration {args.iteration} physical density", "viridis"),
        (np.asarray(data["rho"], float) - np.asarray(proposal_data["rho_previous"], float), "physical-density step", "coolwarm"),
        (data["gradient_optical_A"], "optical gradient", "coolwarm"),
        (data["gradient_thermal_A"], "thermal gradient", "coolwarm"),
        (data["gradient_physical_A"], "physical gradient", "coolwarm"),
        (data["gradient_latent_A"], "latent gradient", "coolwarm"),
    )
    for ax, (value, title, cmap) in zip(axes.ravel(), fields):
        image = ax.imshow(np.asarray(value, float).T, origin="lower", cmap=cmap, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("node x")
        ax.set_ylabel("node y")
        fig.colorbar(image, ax=ax, fraction=0.046)
    design_plot = plots / f"iteration_{args.iteration:03d}_design_and_gradients.png"
    fig.savefig(design_plot, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    axes[0].bar(["move / 0.02", "mean-rho delta / 0.01", "gray-fraction delta"], [
        float(proposal["latent_max_abs_change"]) / 0.02,
        abs(float(row["physical_density_mean"]) - float(rows[-2]["physical_density_mean"])) / 0.01,
        abs(float(row["gray_fraction_0p05_0p95"]) - float(rows[-2]["gray_fraction_0p05_0p95"])),
    ])
    axes[0].axhline(1.0, color="black", linestyle="--")
    axes[0].set_title("Design diagnostics")
    axes[1].bar(["closure", "residual", "energy"], [candidate["gates"]["optical_closure"] / 0.005, candidate["gates"]["thermal_residual"] / 1.0e-8, candidate["gates"]["thermal_energy_balance"] / 0.01])
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("value / gate")
    axes[1].set_title("Physics gate margins")
    axes[2].bar(["P_Q", "P_six"], [candidate["base_forward"]["P_Q_W"], candidate["base_forward"]["P_six_W"]])
    axes[2].set_ylabel("power (W)")
    axes[2].set_title("Optical closure")
    constraint_plot = plots / f"iteration_{args.iteration:03d}_constraints_and_physics.png"
    fig.savefig(constraint_plot, dpi=180)
    plt.close(fig)

    generated = datetime.now(timezone.utc).isoformat()
    raw_entry = {
        "proposal_result": artifact(proposal_result_path),
        "proposal_NPZ": artifact(proposal_raw_path),
        "evaluation_result": artifact(candidate_result_path),
        "evaluation_NPZ": artifact(candidate_raw_path),
    }
    raw_artifacts = dict(previous["raw_artifacts"])
    raw_artifacts[f"iteration_{args.iteration:03d}"] = raw_entry
    previous.update({
        "status": STATUS,
        "generated_at_utc": generated,
        "optimization_started": True,
        "accepted_iterations": args.iteration,
        "current_beta": candidate["beta"],
        "current_objective_A": objective,
        "last_iteration_relative_improvement": improvement_prior,
        "relative_improvement": improvement_total,
        "history": rows,
        "checkpoint": artifact(checkpoint_path),
        "raw_artifacts": raw_artifacts,
    })
    summary_path.write_text(json.dumps(previous, indent=2) + "\n")
    table = "\n".join(
        f"| {int(value['iteration'])} | {float(value['objective_A']):.12e} | {float(value['objective_A_per_incident_W']):.12e} | {100.0*float(value['relative_improvement_from_iteration0']):.6f}% |"
        for value in rows
    )
    report_path.write_text(
        "# Run 002 nominal MMA pilot\n\n"
        f"Status: `{STATUS}`\n\n"
        "| accepted iteration | objective (A) | objective / incident power (A/W) | improvement from iteration 0 |\n"
        "|---:|---:|---:|---:|\n"
        f"{table}\n\n"
        f"Iteration {args.iteration} improves the previous accepted objective by `{100.0*improvement_prior:.6f}%` "
        f"and the baseline by `{100.0*improvement_total:.6f}%`. The fixed objective scale remains "
        f"`{scale:.3e} W/A`; it is not changed between iterations.\n\n"
        "This remains a continuous beta=2 grown/grown +I pilot using the uniform-45° weighting surrogate. "
        "Each accepted design is generated by a restartable one-step NLopt LD_MMA proposal; internal MMA "
        "asymptote history is not persisted across server-safe checkpoints. Exact binary DRC, opposite sign, "
        "named interface robustness, and full electrodes remain pending.\n"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["nominal_mma_pilot"].update({
        "status": STATUS,
        f"iteration_{args.iteration:03d}": raw_entry,
        "checkpoint": previous["checkpoint"],
    })
    manifest["current_promoted_status"] = STATUS
    manifest["current_promoted_at_utc"] = generated
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    status = json.loads(status_path.read_text())
    status.update({
        "status": STATUS,
        "last_updated_utc": generated,
        "optimization_started": True,
        "message": f"Nominal beta=2 MMA iteration {args.iteration} accepted; last improvement={100.0*improvement_prior:.3f}%, total={100.0*improvement_total:.3f}%. Exact binary DRC and robust interface scenarios remain.",
    })
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"status": STATUS, "iteration": args.iteration, "objective_A": objective, "improvement_previous": improvement_prior, "improvement_total": improvement_total, "checkpoint": previous["checkpoint"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
