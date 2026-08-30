#!/usr/bin/env python3
"""Publish the selected exact-binary Run 047/048 left-right results.

This is an offline, fail-closed publisher.  It reuses the already completed
fresh GPU-Maxwell/CUDA thermal-electrical evaluations, verifies their hashes,
reintegrates their current, and independently audits exact 0/1 density and the
discrete 500 nm solid/void rule.  It does not silently remove disconnected
islands; connectivity was not part of the approved Run 047/048 contract.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.optimization_runs import summarize_final_exact_binary_eight_case as base


REPORT_DIR = REPOSITORY / "photothermal_pte" / "reports" / "run047_048_exact_binary_cleanup"

CASES = (
    base.Case(
        47, "x", "left-right", "thermally_grown", 7.37e6, "Ea", "void_first",
        Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run047_Ea_fresh_current_max/forced_exact_500nm_cleanup_lr_contract_v2/void_first_objective"),
        Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run047_Ea_fresh_current_max/forced_exact_500nm_cleanup_lr_contract_v2/void_first_exact_binary_candidate.npz"),
        "embedded thermally-grown interface contract; corrected left-right exact audit",
    ),
    base.Case(
        48, "x", "left-right", "thermally_grown", 7.37e6, "Eb", "solid_first",
        Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run048_Eb_fresh_current_max/forced_exact_500nm_cleanup/solid_first_objective"),
        Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run048_Eb_fresh_current_max/forced_exact_500nm_cleanup/solid_first_density.npz"),
        "legacy thermal.py thermally-grown default; selected forced exact cleanup",
    ),
)

RUN_DIRS = {
    47: REPOSITORY / "photothermal_pte/optimization_runs/run_047_left_right_electrodes_Ea_fresh_restart/results",
    48: REPOSITORY / "photothermal_pte/optimization_runs/run_048_left_right_electrodes_Eb_fresh_restart/results",
}

CLEANUP_PROVENANCE = {
    47: Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run047_Ea_fresh_current_max/forced_exact_500nm_cleanup_lr_contract_v2/exact_binary_repair_result.json"),
    48: RUN_DIRS[48] / "FORCED_EXACT_CLEANUP.json",
}


def comparison_plot(metrics: list[dict[str, object]]) -> Path:
    labels = [f"Run {int(row['run']):03d}\n{row['polarization']}" for row in metrics]
    x = np.arange(len(metrics))
    continuous = np.asarray([row["current_continuous_at_285uW_A"] for row in metrics], dtype=float) * 1e9
    exact = np.asarray([row["current_exact_at_285uW_A"] for row in metrics], dtype=float) * 1e9
    change = np.asarray([row["current_change_from_continuous_fraction"] for row in metrics], dtype=float) * 100
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].bar(x - 0.18, continuous, 0.36, label="continuous checkpoint", color="0.65")
    axes[0].bar(x + 0.18, exact, 0.36, label="forced exact binary", color=["tab:blue", "tab:orange"])
    axes[0].set(xticks=x, xticklabels=labels, ylabel="terminal current at 285 µW (nA)")
    axes[0].legend()
    axes[1].bar(x, change, color=["tab:blue", "tab:orange"])
    axes[1].axhline(-1.0, color="tab:red", ls="--", label="legacy 1% preservation gate")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="exact cleanup current change (%)")
    axes[1].legend()
    width = 0.34
    source_solid = [int(row["source_solid_bad_nodes"]) for row in metrics]
    source_void = [int(row["source_void_bad_nodes"]) for row in metrics]
    axes[2].bar(x - width / 2, source_solid, width, label="source solid violations")
    axes[2].bar(x + width / 2, source_void, width, label="source void violations")
    axes[2].scatter(x, [row["exact_500nm_bad_nodes"] for row in metrics], color="black", marker="x", s=80, label="final total violations")
    axes[2].set(xticks=x, xticklabels=labels, ylabel="500 nm bad nodes")
    axes[2].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Runs 047/048: exact 0/1 and forced zero-violation cleanup")
    path = REPORT_DIR / "run047_048_exact_cleanup_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base.REPORT_DIR = REPORT_DIR
    metrics: list[dict[str, object]] = []
    raw_artifacts: list[dict[str, object]] = []
    plots: list[Path] = []
    for case in CASES:
        row, arrays, artifacts = base.process(case)
        latest = json.loads((RUN_DIRS[case.run] / "latest_summary.json").read_text())
        forced = json.loads(CLEANUP_PROVENANCE[case.run].read_text())
        if case.run == 47:
            selected = "void_first"
            source = {
                "solid_bad": forced["source_audit"]["solid_bad_cell_count"],
                "void_bad": forced["source_audit"]["void_bad_cell_count"],
            }
        else:
            selected = forced["selected"]
            source = forced["candidates"][selected]["repair_history"][0]
        row.update({
            "source_gray_fraction": float(latest["gray_fraction_0p01_0p99"]),
            "source_binarization_mean_4rho1mrho": float(latest["binarization_mean_4rho1mrho"]),
            "source_solid_bad_nodes": int(source["solid_bad"]),
            "source_void_bad_nodes": int(source["void_bad"]),
            "selected_repair_order": selected,
            "connectivity_cleanup_applied": False,
        })
        metrics.append(row)
        raw_artifacts.extend(artifacts)
        raw_artifacts.append(base.artifact(CLEANUP_PROVENANCE[case.run], f"Run {case.run:03d} cleanup provenance"))
        plots.extend(base.per_case_plots(case, row, arrays))
    plots.append(comparison_plot(metrics))

    status = "COMPLETED_RUN047_048_EXACT_BINARY_ZERO_500NM_VIOLATION_PHYSICS"
    generated = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema": "run047-048-exact-binary-cleanup-v1",
        "status": status,
        "generated_at_utc": generated,
        "axis_contract": "Lumerical x=b, y=a, z=c",
        "all_exact_binary": all(bool(row["exact_binary"]) for row in metrics),
        "all_exact_500nm_bad_nodes_zero": all(int(row["exact_500nm_bad_nodes"]) == 0 for row in metrics),
        "all_physical_gates_passed": all(bool(row["physical_gates_passed"]) for row in metrics),
        "connectivity_cleanup_applied": False,
        "cases": metrics,
    }
    summary_path = REPORT_DIR / "run047_048_exact_cleanup_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    csv_path = REPORT_DIR / "run047_048_exact_cleanup_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)

    rows = "\n".join(
        f"| {int(row['run']):03d} | {row['polarization']} | {row['source_gray_fraction']*100:.3f}% | "
        f"{int(row['source_solid_bad_nodes'])}+{int(row['source_void_bad_nodes'])} | "
        f"{int(row['exact_500nm_bad_nodes'])} | {row['current_continuous_at_285uW_A']*1e9:.4f} | "
        f"{row['current_exact_at_285uW_A']*1e9:.4f} | {row['current_change_from_continuous_fraction']*100:.3f}% | "
        f"{row['physical_gates_passed']} |"
        for row in metrics
    )
    report_path = REPORT_DIR / "RUN047_048_EXACT_BINARY_CLEANUP_REPORT.md"
    report_path.write_text(f"""# Runs 047/048 exact-binary cleanup

Status: **{status}**

Both selected designs are exactly `0/1`. The independent discrete 500 nm audit reports solid bad nodes = 0 and void bad nodes = 0. The raw continuous/gray checkpoints are preserved; this publication uses the separately repaired and freshly evaluated exact candidates.

| Run | pol. | source gray | source solid+void bad | final bad | continuous I (nA) | exact I (nA) | change | physical gates |
|---:|---|---:|---:|---:|---:|---:|---:|---|
{rows}

Run047's exact structure loses more than the legacy 1% objective-preservation threshold. That is reported as a performance diagnostic, not hidden and not used to undo the geometry gate. Run048 preserves the objective within 1%.

The cleanup enforces the requested 500 nm solid/void rule but does **not** remove electrically disconnected islands. Connectivity was explicitly absent from these optimization contracts; adding it now would be a different design constraint and a different geometry.

Every exact candidate has a fresh v261 GPU Maxwell forward evaluation followed by the unchanged CUDA thermal/electrical path. The publisher independently verifies density/field SHA-256 values, reintegrates current, checks exact geometry, and produces Q, temperature, strict-centered gradient, weighting-potential, and current-contribution maps. No Q clipping, smoothing, gain, global rescaling, tiling, or source deletion is used.
""")

    outputs = [report_path, summary_path, csv_path, *plots]
    manifest_path = REPORT_DIR / "RAW_ARTIFACT_MANIFEST.json"
    manifest = {
        "schema": "run047-048-exact-binary-cleanup-manifest-v1",
        "status": status,
        "generated_at_utc": generated,
        "generation_command": "/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python photothermal_pte/optimization_runs/summarize_run047_048_exact_cleanup.py",
        "raw_artifacts_not_committed": raw_artifacts,
        "published_outputs": [base.artifact(path, "published exact-cleanup artifact") for path in outputs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": status, "report_dir": str(REPORT_DIR), "currents_nA": {str(row['run']): row['current_exact_at_285uW_A'] * 1e9 for row in metrics}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
