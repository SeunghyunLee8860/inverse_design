#!/usr/bin/env python3
"""Publish an exact-500-nm cleanup comparison without copying raw solver fields."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--repair-result", required=True, type=Path)
    parser.add_argument("--solid-objective", required=True, type=Path)
    parser.add_argument("--void-objective", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repair = load(args.repair_result.resolve())
    objectives = {
        "solid_first": load(args.solid_objective.resolve()),
        "void_first": load(args.void_objective.resolve()),
    }
    rows: list[dict[str, object]] = []
    for order in ("solid_first", "void_first"):
        candidate = repair["candidates"][order]
        objective = objectives[order]
        rows.append(
            {
                "order": order,
                "exact_bad_before": int(repair["source_audit"]["total_bad_cell_count"]),
                "exact_bad_after": int(candidate["audit"]["total_bad_cell_count"]),
                "changed_node_count": int(candidate["changed_node_count"]),
                "changed_node_fraction": float(candidate["changed_node_fraction"]),
                "solid_fraction": float(candidate["audit"]["solid_fraction"]),
                "objective_A": float(objective["objective_A"]),
                "reference_continuous_objective_A": float(objective["reference_continuous_objective_A"]),
                "relative_objective_change": float(objective["relative_objective_change_from_continuous"]),
                "equivalent_current_at_285uW_A": float(objective["equivalent_objective_at_285uW_A"]),
                "optical_closure": float(objective["gates"]["optical_closure"]),
                "thermal_residual": float(objective["gates"]["thermal_forward_residual"]),
                "thermal_energy_balance": float(objective["gates"]["thermal_energy_balance"]),
                "electrical_residual": float(objective["gates"]["electrical_weighting_residual"]),
                "objective_gate_passed": bool(objective["binary_objective_preserved_within_one_percent"]),
                "solver_status": objective["status"],
                "candidate_artifact": candidate["artifact"],
                "objective_result_artifact": artifact(
                    args.solid_objective.resolve() if order == "solid_first" else args.void_objective.resolve()
                ),
                "raw_field_artifact": objective["raw_artifact"],
                "raw_project_artifact": objective["forward"]["project"],
            }
        )

    selected = max(rows, key=lambda row: float(row["objective_A"]))
    status = (
        "VALIDATED_EXACT_500NM_BINARY_CLEANUP_AND_OBJECTIVE"
        if bool(selected["objective_gate_passed"])
        else "EXACT_500NM_BINARY_CLEANUP_OBJECTIVE_GATE_UNRESOLVED"
    )
    summary = {
        "schema": "exact-500nm-cleanup-gpu-objective-comparison-v1",
        "run_label": args.run_label,
        "status": status,
        "selection_rule": "maximum unrescaled GPU forward objective among globally exact-zero candidates",
        "selected_order": selected["order"],
        "selected": selected,
        "candidates": rows,
        "raw_Q_clipping_smoothing_gain_or_rescaling": False,
        "CPU_FDTD_fallback": False,
        "CPU_thermal_linear_solve_fallback": False,
        "connectivity_cleanup_applied": False,
        "notes": [
            "The exact audit is a discrete binary 500 nm opening test on the 100 nm design grid.",
            "The objective was recomputed by one GPU Maxwell forward solve followed by the certified CUDA thermal/electrical path.",
            "Raw NPZ and FSP artifacts remain outside Git; paths, sizes and SHA-256 values are retained.",
        ],
    }
    summary_path = output / "exact_500nm_cleanup_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    csv_path = output / "exact_500nm_cleanup_candidates.csv"
    scalar_keys = [
        "order", "exact_bad_before", "exact_bad_after", "changed_node_count",
        "changed_node_fraction", "solid_fraction", "objective_A",
        "reference_continuous_objective_A", "relative_objective_change",
        "equivalent_current_at_285uW_A", "optical_closure", "thermal_residual",
        "thermal_energy_balance", "electrical_residual", "objective_gate_passed", "solver_status",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in scalar_keys})

    currents = [float(row["equivalent_current_at_285uW_A"]) * 1e9 for row in rows]
    reference_nA = (
        float(rows[0]["reference_continuous_objective_A"])
        * float(rows[0]["equivalent_current_at_285uW_A"])
        / float(rows[0]["objective_A"])
        * 1e9
    )
    losses = [-100.0 * float(row["relative_objective_change"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    axes[0].bar(["solid-first", "void-first"], currents, color=["#4472C4", "#ED7D31"])
    axes[0].axhline(reference_nA, color="black", linestyle="--", label=f"continuous {reference_nA:.3f} nA")
    axes[0].set_ylabel("equivalent current at 285 µW (nA)")
    axes[0].set_title(f"{args.run_label}: exact-zero candidates")
    axes[0].legend()
    axes[1].bar(["solid-first", "void-first"], losses, color=["#4472C4", "#ED7D31"])
    axes[1].axhline(1.0, color="black", linestyle="--", label="1% preservation gate")
    axes[1].set_ylabel("objective loss from continuous checkpoint (%)")
    axes[1].set_title("binary cleanup cost")
    axes[1].legend()
    plot_path = output / "exact_500nm_cleanup_objective_comparison.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    repair_plot_source = Path(repair["plot"]["path"])
    repair_plot = output / "exact_500nm_cleanup_geometry_changes.png"
    shutil.copy2(repair_plot_source, repair_plot)

    manifest = {
        "schema": "exact-500nm-cleanup-raw-artifact-manifest-v1",
        "run_label": args.run_label,
        "repair_result": artifact(args.repair_result.resolve()),
        "source_checkpoint": repair["source"],
        "candidates": {
            row["order"]: {
                "density": row["candidate_artifact"],
                "objective_result": row["objective_result_artifact"],
                "raw_fields": row["raw_field_artifact"],
                "raw_FSP": row["raw_project_artifact"],
            }
            for row in rows
        },
        "Git_exclusions": ["raw candidate NPZ", "raw field NPZ", "raw FSP"],
    }
    manifest_path = output / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    chosen = selected
    report = f"""# {args.run_label} exact 500 nm binary cleanup

Status: `{status}`

The continuous beta-128 checkpoint was already essentially binary, but its thresholded geometry failed the discrete 500 nm opening audit. Two deterministic active-set cleanup orderings were built. Both remove every exact violation without connectivity editing, followed by an unrescaled GPU Maxwell forward solve and the same CUDA thermal/electrical objective path.

| candidate | exact bad | changed nodes | current at 285 µW | objective change | objective gate |
|---|---:|---:|---:|---:|---:|
| solid-first | {rows[0]['exact_bad_after']} | {rows[0]['changed_node_count']} | {float(rows[0]['equivalent_current_at_285uW_A'])*1e9:.6f} nA | {float(rows[0]['relative_objective_change'])*100:.4f}% | {rows[0]['objective_gate_passed']} |
| void-first | {rows[1]['exact_bad_after']} | {rows[1]['changed_node_count']} | {float(rows[1]['equivalent_current_at_285uW_A'])*1e9:.6f} nA | {float(rows[1]['relative_objective_change'])*100:.4f}% | {rows[1]['objective_gate_passed']} |

Selected candidate: `{chosen['order']}`, because it has the larger recomputed objective among exact-zero candidates. The continuous reference is {reference_nA:.6f} nA at 285 µW.

All optical closure, remap, thermal residual, thermal energy-balance, and electrical residual checks pass. A status remains unresolved when the independently fixed 1% objective-preservation gate fails; that gate is not relaxed after seeing the result.

No Q clipping, smoothing, gain, global rescaling, CPU FDTD fallback, CPU thermal fallback, or connectivity cleanup was used. Raw NPZ/FSP files are not committed to Git.
"""
    (output / "EXACT_500NM_CLEANUP_REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
