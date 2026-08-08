#!/usr/bin/env python3
"""Append one Run 003 state and regenerate all progress artifacts."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from optimization_support import design_metrics, stage_convergence


HERE = Path(__file__).resolve().parent
STATUS = "RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def load_summary() -> dict[str, object]:
    path = HERE / "results/optimization_summary.json"
    if path.exists():
        return json.loads(path.read_text())
    return {
        "status": STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "fixed stage inequalities plus convergence-based beta continuation",
        "history": [],
        "raw_artifacts": {},
    }


def save_design_plots(tag: str, metrics: dict, arrays: dict, data) -> list[Path]:
    plots = HERE / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    extent = [-9.3, 9.3, -9.3, 9.3]
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), constrained_layout=True)
    entries = (
        (arrays["latent"], "latent", "viridis", 0, 1),
        (arrays["filtered"], "500 nm conic filtered", "viridis", 0, 1),
        (arrays["rho"], f"physical rho, beta={metrics['beta']:g}", "viridis", 0, 1),
        (arrays["binary"], "thresholded audit", "gray", 0, 1),
        (arrays["bad_solid"], "exact solid violations", "Reds", 0, 1),
        (arrays["bad_void"], "exact void violations", "Blues", 0, 1),
        (arrays["solid_penalty_field"], f"solid penalty: {metrics['constraint_contract']}", "magma", None, None),
        (arrays["void_penalty_field"], f"void penalty: {metrics['constraint_contract']}", "magma", None, None),
        (np.asarray(data["gradient_optical_A"]), "optical physical gradient", "coolwarm", None, None),
        (np.asarray(data["gradient_thermal_A"]), "thermal physical gradient", "coolwarm", None, None),
    )
    for ax, (field, title, cmap, vmin, vmax) in zip(axes.ravel(), entries):
        image = ax.imshow(np.asarray(field).T, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(
        f"{tag}: FOM={metrics['objective_A_per_W']:.6e} A/W; "
        f"C_s={metrics['solid_constraint']:.3e}, C_v={metrics['void_constraint']:.3e}"
    )
    design = plots / f"{tag}_design_constraints_gradients.png"
    fig.savefig(design, dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), constrained_layout=True)
    axes[0].hist(np.asarray(arrays["rho"]).ravel(), bins=64, range=(0, 1))
    axes[0].set_title("Physical-density histogram")
    axes[0].set_xlabel("rho")
    axes[1].bar(
        ["solid", "void"],
        [metrics["solid_constraint"], metrics["void_constraint"]],
        color=["#c33", "#36c"],
    )
    axes[1].axhline(metrics["solid_constraint_cap"], color="black", ls="--", label="fixed cap")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].set_title("Fixed stage inequalities")
    axes[2].bar(
        ["gray 1-99%", "4rho(1-rho)"],
        [metrics["gray_fraction_0p01_0p99"], metrics["binarization_metric_mean_4rho1mrho"]],
    )
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Binarization")
    exact = metrics["exact_binary_audit"]
    axes[3].bar(["solid", "void"], [exact["solid_bad_cell_count"], exact["void_bad_cell_count"]])
    axes[3].set_title("Exact 500 nm bad cells")
    summary_plot = plots / f"{tag}_density_constraints_drc.png"
    fig.savefig(summary_plot, dpi=170)
    plt.close(fig)
    return [design, summary_plot]


def save_history_outputs(summary: dict[str, object]) -> None:
    results = HERE / "results"
    plots = HERE / "plots"
    history = summary["history"]
    results.mkdir(parents=True, exist_ok=True)
    with (results / "optimization_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    x = [row["evaluation_index"] for row in history]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    axes[0, 0].plot(x, [row["objective_A_per_W"] for row in history], "o-")
    axes[0, 0].set_title("Actual PTE FOM")
    axes[0, 0].set_ylabel("A/W")
    axes[0, 1].step(x, [row["beta"] for row in history], where="post")
    axes[0, 1].set_title("Convergence-based beta")
    axes[0, 2].semilogy(x, [row["solid_constraint"] for row in history], "o-", label="solid")
    axes[0, 2].semilogy(x, [row["void_constraint"] for row in history], "s-", label="void")
    axes[0, 2].semilogy(x, [row["constraint_cap"] for row in history], "k--", label="fixed cap")
    axes[0, 2].legend()
    axes[0, 2].set_title("Manufacturing inequalities")
    axes[1, 0].plot(x, [row["gray_fraction_0p01_0p99"] for row in history], "o-", label="1-99%")
    axes[1, 0].plot(x, [row["binarization_metric"] for row in history], "s-", label="4rho(1-rho)")
    axes[1, 0].legend()
    axes[1, 0].set_title("Binarization")
    axes[1, 1].semilogy(x, np.maximum([row["solid_bad_cells"] for row in history], 0.5), "o-", label="solid")
    axes[1, 1].semilogy(x, np.maximum([row["void_bad_cells"] for row in history], 0.5), "s-", label="void")
    axes[1, 1].legend()
    axes[1, 1].set_title("Exact 500 nm audit")
    axes[1, 2].semilogy(x, np.maximum([row["rho_rms_change"] for row in history], 1e-12), "o-", label="RMS")
    axes[1, 2].semilogy(x, np.maximum([row["rho_max_change"] for row in history], 1e-12), "s-", label="max")
    axes[1, 2].legend()
    axes[1, 2].set_title("Physical design change")
    for ax in axes.ravel():
        ax.set_xlabel("recorded evaluation")
        ax.grid(alpha=0.25)
    fig.savefig(plots / "iteration_fom_beta_constraints_drc.png", dpi=180)
    plt.close(fig)


def record(
    evaluation_result: Path,
    evaluation_raw: Path,
    global_iteration: int,
    stage_iteration: int,
    role: str,
    proposal_result: Path | None = None,
    proposal_raw: Path | None = None,
    mma_state: Path | None = None,
) -> dict[str, object]:
    result = json.loads(evaluation_result.read_text())
    if not result.get("passed"):
        raise RuntimeError("refusing to record a failed physics evaluation")
    if sha256(evaluation_raw) != result["raw_artifact"]["sha256"]:
        raise RuntimeError("evaluation raw SHA mismatch")
    data = np.load(evaluation_raw)
    latent = np.asarray(data["latent"], float)
    beta = float(result["beta"])
    metrics, arrays = design_metrics(latent, beta)
    incident = float(result["incident_power_W"])
    metrics.update({
        "objective_A": float(result["objective_A"]),
        "objective_A_per_W": float(result["objective_A"]) / incident,
        "P_Q_W": float(result["base_forward"]["P_Q_W"]),
        "P_six_W": float(result["base_forward"]["P_six_W"]),
        "optical_closure": float(result["gates"]["optical_closure"]),
        "thermal_residual": float(result["gates"]["thermal_residual"]),
        "thermal_energy_balance": float(result["gates"]["thermal_energy_balance"]),
    })
    summary = load_summary()
    history = summary["history"]
    for previous_row in history:
        previous_row.setdefault("constraint_contract", "legacy_zhou_p8_v1")
    previous = history[-1] if history else None
    if previous is None or float(previous["beta"]) != beta:
        relative_fom_change = 0.0
        rho_rms_change = 0.0
        rho_max_change = 0.0
    else:
        previous_raw = Path(summary["raw_artifacts"][previous["tag"]]["evaluation_NPZ"]["path"])
        previous_rho = design_metrics(np.asarray(np.load(previous_raw)["latent"], float), beta)[1]["rho"]
        delta = arrays["rho"] - previous_rho
        relative_fom_change = (
            metrics["objective_A_per_W"] - float(previous["objective_A_per_W"])
        ) / float(previous["objective_A_per_W"])
        rho_rms_change = float(np.sqrt(np.mean(delta * delta)))
        rho_max_change = float(np.max(np.abs(delta)))
    tag = f"run003_b{int(beta):03d}_s{stage_iteration:03d}_g{global_iteration:03d}_{role}"
    generated_plots = save_design_plots(tag, metrics, arrays, data)
    checkpoint_dir = HERE / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{tag}.npz"
    np.savez_compressed(
        checkpoint,
        latent=np.asarray(latent, np.float32), beta=np.asarray(beta),
        objective_A=np.asarray(metrics["objective_A"]),
        global_iteration=np.asarray(global_iteration), stage_iteration=np.asarray(stage_iteration),
    )
    exact = metrics["exact_binary_audit"]
    row = {
        "evaluation_index": len(history), "tag": tag,
        "global_iteration": global_iteration, "stage_iteration": stage_iteration,
        "role": role, "beta": beta,
        "constraint_contract": metrics["constraint_contract"],
        "objective_A": metrics["objective_A"], "objective_A_per_W": metrics["objective_A_per_W"],
        "relative_fom_change": relative_fom_change,
        "rho_mean": metrics["rho_mean"], "rho_rms_change": rho_rms_change,
        "rho_max_change": rho_max_change,
        "gray_fraction_0p01_0p99": metrics["gray_fraction_0p01_0p99"],
        "binarization_metric": metrics["binarization_metric_mean_4rho1mrho"],
        "solid_constraint": metrics["solid_constraint"], "void_constraint": metrics["void_constraint"],
        "constraint_cap": metrics["solid_constraint_cap"],
        "constraints_feasible": metrics["constraints_feasible"],
        "solid_bad_cells": exact["solid_bad_cell_count"], "void_bad_cells": exact["void_bad_cell_count"],
        "optical_closure": metrics["optical_closure"],
        "thermal_residual": metrics["thermal_residual"],
        "thermal_energy_balance": metrics["thermal_energy_balance"],
    }
    history.append(row)
    raw_entry = {
        "evaluation_result": artifact(evaluation_result),
        "evaluation_NPZ": artifact(evaluation_raw),
        "checkpoint": artifact(checkpoint),
    }
    if proposal_result is not None:
        raw_entry["proposal_result"] = artifact(proposal_result)
    if proposal_raw is not None:
        raw_entry["proposal_NPZ"] = artifact(proposal_raw)
    if mma_state is not None:
        raw_entry["mma_state"] = artifact(mma_state)
    summary["raw_artifacts"][tag] = raw_entry
    summary.update({
        "status": STATUS, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_beta": beta, "current_global_iteration": global_iteration,
        "current_stage_iteration": stage_iteration, "current_metrics": metrics,
        "current_checkpoint": artifact(checkpoint), "history": history,
    })
    convergence = stage_convergence(history, beta)
    summary["current_stage_convergence"] = convergence.__dict__
    results = HERE / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "optimization_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_history_outputs(summary)
    (HERE / "STATUS.json").write_text(json.dumps({
        "run_id": HERE.name, "status": STATUS,
        "last_updated_utc": summary["generated_at_utc"],
        "beta": beta, "global_iteration": global_iteration,
        "stage_iteration": stage_iteration, "stage_convergence": convergence.__dict__,
        "FOM_A_per_W": metrics["objective_A_per_W"],
        "constraints_feasible": metrics["constraints_feasible"],
        "exact_500nm_solid_bad_cells": exact["solid_bad_cell_count"],
        "exact_500nm_void_bad_cells": exact["void_bad_cell_count"],
    }, indent=2) + "\n")
    manifest = {
        "status": STATUS, "generated_at_utc": summary["generated_at_utc"],
        "raw_artifacts_are_gitignored": True,
        "base_FSP": summary.get("base_FSP"),
        "material_Jacobian": summary.get("material_Jacobian"),
        "constraint_recovery_validation": summary.get("constraint_recovery_validation"),
        "records": summary["raw_artifacts"],
    }
    manifest_dir = HERE / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = (
        "# Run 003 convergence-based constrained PTE optimization\n\n"
        f"Status: `{STATUS}`\n\n"
        f"Current stage: beta={beta:g}, accepted stage iteration={stage_iteration}, "
        f"global iteration={global_iteration}.\n\n"
        f"Constraint contract: `{metrics['constraint_contract']}`.\n\n"
        f"Actual FOM: `{metrics['objective_A_per_W']:.12e} A/W`. Fixed-cap solid/void "
        f"constraints: `{metrics['solid_constraint']:.6e}` / `{metrics['void_constraint']:.6e}` "
        f"with cap `{metrics['solid_constraint_cap']:.6e}`.\n\n"
        f"Exact 500 nm bad cells: solid `{exact['solid_bad_cell_count']}`, void "
        f"`{exact['void_bad_cell_count']}`. Stage convergence: `{convergence.converged}` "
        f"({convergence.reason}).\n\n"
        "Beta is promoted only after fixed-inequality feasibility and a four-update "
        "FOM/design plateau; it is never promoted after one nominal update.\n"
    )
    (results / "OPTIMIZATION_REPORT.md").write_text(report)
    return {"tag": tag, "metrics": metrics, "convergence": convergence.__dict__, "plots": [str(p) for p in generated_plots]}
