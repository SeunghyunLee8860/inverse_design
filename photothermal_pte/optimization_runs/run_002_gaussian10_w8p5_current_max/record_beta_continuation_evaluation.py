#!/usr/bin/env python3
"""Record a beta-continuation evaluation and regenerate all progress figures."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from beta_continuation_support import design_metrics
from production_density_mapping import ProductionDensityMapping


STATUS = "RUNNING_BETA_CONTINUATION_WITH_500NM_SOLID_VOID_CONSTRAINTS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def plot_design(plots: Path, tag: str, metrics: dict, arrays: dict, data) -> list[str]:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    entries = (
        (arrays["latent"], "latent design", "viridis", 0.0, 1.0),
        (arrays["filtered"], "500 nm finite conic filter", "viridis", 0.0, 1.0),
        (arrays["rho"], f"physical density (beta={metrics['beta']:g})", "viridis", 0.0, 1.0),
        (arrays["binary"], "0.5-threshold binary audit", "gray", 0.0, 1.0),
        (arrays["bad_solid"], "solid <500 nm violations", "Reds", 0.0, 1.0),
        (arrays["bad_void"], "void <500 nm violations", "Blues", 0.0, 1.0),
        (np.asarray(data["gradient_optical_A"], float), "optical physical gradient", "coolwarm", None, None),
        (np.asarray(data["gradient_thermal_A"], float), "thermal physical gradient", "coolwarm", None, None),
    )
    extent = [-9.3, 9.3, -9.3, 9.3]
    for ax, (field, title, cmap, vmin, vmax) in zip(axes.ravel(), entries):
        image = ax.imshow(np.asarray(field, float).T, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(
        f"{tag}: FOM={metrics['objective_A_per_W']:.6e} A/W, "
        f"gray={metrics['binarization_metric_mean_4rho1mrho']:.4f}",
        fontsize=14,
    )
    design_path = plots / f"{tag}_design_constraints_gradients.png"
    fig.savefig(design_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    axes[0].hist(np.asarray(arrays["rho"]).ravel(), bins=60, range=(0, 1), color="#3465a4")
    axes[0].set_xlabel("physical density")
    axes[0].set_ylabel("node count")
    axes[0].set_title("Density histogram")
    axes[1].bar(
        ["gray 1-99%", "gray 5-95%", "4rho(1-rho)"],
        [metrics["gray_fraction_0p01_0p99"], metrics["gray_fraction_0p05_0p95"], metrics["binarization_metric_mean_4rho1mrho"]],
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"Binarization at beta={metrics['beta']:g}")
    exact = metrics["exact_binary_audit"]
    axes[2].bar(
        ["solid bad", "void bad"],
        [exact["solid_bad_fraction_all_cells"], exact["void_bad_fraction_all_cells"]],
        color=["#cc0000", "#204a87"],
    )
    axes[2].set_ylabel("fraction of all design nodes")
    axes[2].set_title("Exact 500 nm binary opening audit")
    metric_path = plots / f"{tag}_binarization_and_drc.png"
    fig.savefig(metric_path, dpi=180)
    plt.close(fig)
    return [str(design_path), str(metric_path)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-result", type=Path, required=True)
    parser.add_argument("--evaluation-raw", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--global-iteration", type=int, required=True)
    parser.add_argument("--stage-iteration", type=int, required=True)
    parser.add_argument("--role", choices=("stage_baseline", "accepted_mma"), required=True)
    parser.add_argument("--proposal-result", type=Path)
    parser.add_argument("--proposal-raw", type=Path)
    parser.add_argument("--mma-state", type=Path)
    args = parser.parse_args()
    run = args.run_directory.expanduser().resolve()
    results = run / "results"
    plots = run / "plots"
    checkpoints = run / "checkpoints"
    result_path = args.evaluation_result.expanduser().resolve()
    raw_path = args.evaluation_raw.expanduser().resolve()
    result = json.loads(result_path.read_text())
    if not result.get("passed"):
        raise RuntimeError("evaluation failed physics gates")
    if sha256(raw_path) != result["raw_artifact"]["sha256"]:
        raise RuntimeError("evaluation raw SHA mismatch")
    data = np.load(raw_path)
    latent = np.asarray(data["latent"], float)
    beta = float(result["beta"])
    metrics, arrays = design_metrics(latent, beta, ProductionDensityMapping())
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
    tag = f"continuation_b{int(round(beta)):03d}_i{args.stage_iteration:03d}_g{args.global_iteration:03d}"
    generated_plots = plot_design(plots, tag, metrics, arrays, data)
    checkpoint = checkpoints / f"{tag}_accepted_state.npz"
    np.savez_compressed(
        checkpoint,
        latent=np.asarray(latent, np.float32),
        beta=np.asarray(beta),
        objective_A=np.asarray(result["objective_A"]),
        incident_power_W=np.asarray(incident),
        global_iteration=np.asarray(args.global_iteration),
        stage_iteration=np.asarray(args.stage_iteration),
    )
    row = {
        "evaluation_index": 0,
        "global_iteration": args.global_iteration,
        "stage_iteration": args.stage_iteration,
        "role": args.role,
        "beta": beta,
        "objective_A": metrics["objective_A"],
        "objective_A_per_W": metrics["objective_A_per_W"],
        "rho_mean": metrics["rho_mean"],
        "gray_fraction_0p01_0p99": metrics["gray_fraction_0p01_0p99"],
        "gray_fraction_0p05_0p95": metrics["gray_fraction_0p05_0p95"],
        "binarization_metric": metrics["binarization_metric_mean_4rho1mrho"],
        "smooth_solid_constraint": metrics["smooth_solid_constraint"],
        "smooth_void_constraint": metrics["smooth_void_constraint"],
        "solid_bad_fraction": metrics["exact_binary_audit"]["solid_bad_fraction_all_cells"],
        "void_bad_fraction": metrics["exact_binary_audit"]["void_bad_fraction_all_cells"],
        "solid_exact_pass": metrics["exact_binary_audit"]["solid_pass"],
        "void_exact_pass": metrics["exact_binary_audit"]["void_pass"],
        "optical_closure": metrics["optical_closure"],
        "thermal_residual": metrics["thermal_residual"],
        "thermal_energy_balance": metrics["thermal_energy_balance"],
    }
    summary_path = results / "beta_continuation_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        history = list(summary["history"])
    else:
        summary = {
            "status": STATUS,
            "contract": {
                "beta_schedule": [4, 8, 16, 32, 64, 128],
                "minimum_solid_feature_nm": 500.0,
                "minimum_void_feature_nm": 500.0,
                "finite_nonperiodic_filter_radius_nm": 500.0,
                "no_periodic_wrap": True,
                "exact_binary_audit_is_not_repair": True,
                "final_gates": {
                    "gray_fraction_0p01_0p99_max": 0.001,
                    "binarization_metric_max": 0.001,
                    "solid_bad_cell_count": 0,
                    "void_bad_cell_count": 0,
                },
                "promotion": "requires final thresholded-binary Maxwell/CUDA-thermal reevaluation",
            },
            "history": [],
            "raw_artifacts": {},
        }
        history = []
    if history:
        last = history[-1]
        if (
            float(last["beta"]) == beta
            and int(last["stage_iteration"]) == args.stage_iteration
            and int(last["global_iteration"]) == args.global_iteration
        ):
            raise RuntimeError("refusing duplicate continuation record")
    row["evaluation_index"] = len(history)
    history.append(row)
    raw_entry = {
        "evaluation_result": artifact(result_path),
        "evaluation_NPZ": artifact(raw_path),
        "checkpoint": artifact(checkpoint),
    }
    if args.proposal_result and args.proposal_raw:
        raw_entry["proposal_result"] = artifact(args.proposal_result.expanduser().resolve())
        raw_entry["proposal_NPZ"] = artifact(args.proposal_raw.expanduser().resolve())
    if args.mma_state:
        raw_entry["mma_state"] = artifact(args.mma_state.expanduser().resolve())
    summary.update({
        "status": STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_beta": beta,
        "current_global_iteration": args.global_iteration,
        "current_stage_iteration": args.stage_iteration,
        "current_metrics": metrics,
        "history": history,
        "current_checkpoint": artifact(checkpoint),
    })
    summary["raw_artifacts"][tag] = raw_entry
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    history_path = results / "beta_continuation_history.csv"
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    indices = [item["evaluation_index"] for item in history]
    axes[0, 0].plot(indices, [item["objective_A_per_W"] for item in history], "o-")
    axes[0, 0].set_ylabel("PTE FOM (A/W)")
    axes[0, 0].set_title("FOM per evaluated/accepted continuation state")
    axes[0, 1].step(indices, [item["beta"] for item in history], where="mid")
    axes[0, 1].set_ylabel("projection beta")
    axes[0, 1].set_title("Beta continuation")
    axes[1, 0].plot(indices, [item["gray_fraction_0p01_0p99"] for item in history], "o-", label="0.01<rho<0.99")
    axes[1, 0].plot(indices, [item["binarization_metric"] for item in history], "s-", label="mean 4rho(1-rho)")
    axes[1, 0].legend()
    axes[1, 0].set_ylabel("gray/binarization metric")
    axes[1, 1].semilogy(indices, np.maximum([item["solid_bad_fraction"] for item in history], 1e-8), "o-", label="solid")
    axes[1, 1].semilogy(indices, np.maximum([item["void_bad_fraction"] for item in history], 1e-8), "s-", label="void")
    axes[1, 1].legend()
    axes[1, 1].set_ylabel("exact 500 nm violation fraction")
    for ax in axes.ravel():
        ax.set_xlabel("continuation evaluation index")
        ax.grid(alpha=0.3)
    progress_plot = plots / "beta_continuation_fom_binarization_drc.png"
    fig.savefig(progress_plot, dpi=180)
    plt.close(fig)
    report = results / "BETA_CONTINUATION_REPORT.md"
    report.write_text(
        "# Run 002 beta continuation with 500 nm solid/void constraints\n\n"
        f"Status: `{STATUS}`\n\n"
        "This is the continuation of the validated nominal run. It uses a stateful MMA "
        "subproblem, the existing finite nonperiodic 500 nm conic filter, explicit smooth "
        "solid/void constraints, and an independent exact 500 nm binary morphology audit. "
        "The exact audit never modifies or repairs the design.\n\n"
        f"Current beta: `{beta:g}`; current global iteration: `{args.global_iteration}`; "
        f"FOM: `{metrics['objective_A_per_W']:.12e} A/W`; mean `4 rho (1-rho)`: "
        f"`{metrics['binarization_metric_mean_4rho1mrho']:.8g}`.\n\n"
        f"Exact solid violation fraction: `{metrics['exact_binary_audit']['solid_bad_fraction_all_cells']:.8g}`; "
        f"exact void violation fraction: `{metrics['exact_binary_audit']['void_bad_fraction_all_cells']:.8g}`.\n\n"
        "`RUNNING` is intentional: fully binary promotion requires the final grayness gates, "
        "zero exact solid/void violations, and a fresh thresholded-binary Maxwell/CUDA-thermal reevaluation.\n"
    )
    status_path = run / "STATUS.json"
    status = json.loads(status_path.read_text())
    status.update({
        "status": STATUS,
        "last_updated_utc": summary["generated_at_utc"],
        "optimization_started": True,
        "message": (
            f"Beta={beta:g} continuation state g={args.global_iteration}, "
            f"stage={args.stage_iteration} recorded; FOM={metrics['objective_A_per_W']:.6e} A/W, "
            f"grayness={metrics['binarization_metric_mean_4rho1mrho']:.6g}. "
            "Fully binary and exact 500 nm DRC promotion remain pending."
        ),
    })
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    config_path = run / "run_config.json"
    config_text = config_path.read_text()
    updated_config, replacements = re.subn(
        r'("driver_status"\s*:\s*")[^"]+("\s*)',
        rf'\g<1>{STATUS}\g<2>',
        config_text,
        count=1,
    )
    if replacements != 1:
        raise RuntimeError("run_config driver_status was not uniquely replaceable")
    config_path.write_text(updated_config)
    manifest_path = run / "manifests" / "RAW_ARTIFACT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    continuation = manifest.setdefault("beta_continuation", {
        "status": STATUS,
        "raw_artifacts_are_gitignored": True,
        "minimum_solid_feature_nm": 500.0,
        "minimum_void_feature_nm": 500.0,
        "records": {},
    })
    continuation["status"] = STATUS
    continuation["records"][tag] = raw_entry
    # A running continuation is active work, not a promoted physical result.
    # Preserve the immutable beta=2 promotion until every final binary gate
    # and the thresholded-binary solver reevaluation have passed.
    manifest["active_work_status"] = STATUS
    manifest["active_work_updated_at_utc"] = summary["generated_at_utc"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": STATUS,
        "tag": tag,
        "metrics": metrics,
        "plots": generated_plots + [str(progress_plot)],
        "checkpoint": artifact(checkpoint),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
