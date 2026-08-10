#!/usr/bin/env python3
"""Report what a bounded beta continuation did and did not certify."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--published-dir", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    raw = args.raw_root.resolve()
    published = args.published_dir.resolve()
    history = json.loads((raw / "history.json").read_text())
    events = []
    for line in (raw / "events.jsonl").read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows = [row for row in history if row.get("accepted")]
    stages = []
    for beta in sorted({float(row["beta"]) for row in rows}):
        selected = [row for row in rows if float(row["beta"]) == beta]
        advances = [
            row for row in events
            if row.get("event") == "beta_advance" and float(row.get("beta", -1)) == beta
        ]
        stages.append({
            "beta": beta,
            "accepted_records_including_reprojection_or_initial": len(selected),
            "accepted_design_updates": sum(
                not row.get("stage_reprojection", False)
                and not row.get("initial_uniform", False)
                for row in selected
            ),
            "objective_A_first": float(selected[0]["objective_A"]),
            "objective_A_last": float(selected[-1]["objective_A"]),
            "gray_fraction_first": float(selected[0]["gray_fraction"]),
            "gray_fraction_last": float(selected[-1]["gray_fraction"]),
            "binarization_first": float(selected[0].get("binarization", float("nan"))),
            "binarization_last": float(selected[-1].get("binarization", float("nan"))),
            "bad_discrete_opening_nodes_first": int(selected[0]["exact_bad_cells"]),
            "bad_discrete_opening_nodes_last": int(selected[-1]["exact_bad_cells"]),
            "termination_reason": advances[-1]["reason"] if advances else "final_schedule_end",
        })
    complete = any(row.get("event") == "continuous_continuation_complete" for row in events)
    last = rows[-1]
    payload = {
        "schema": "bounded-beta-continuation-convergence-assessment-v1",
        "status": (
            "COMPLETED_BOUNDED_CONTINUATION_NOT_CERTIFIED_OPTIMUM"
            if complete else "INCOMPLETE_BOUNDED_CONTINUATION"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": args.run_label,
        "schedule_completed": complete,
        "certified_local_or_global_optimum": False,
        "reason_not_certified": (
            "beta stages have small hard maximum update counts; the final beta=64 stage "
            "ended at its bounded stage budget rather than a strict optimizer convergence gate"
        ),
        "binarization_interpretation": (
            "the rapid gray-density collapse is primarily induced by tanh projection as beta "
            "increases; it is not evidence that the underlying latent topology independently converged"
        ),
        "feature_audit_interpretation": (
            "500 nm was requested, but ceil(250 nm / 100 nm)=3 nodal offsets realizes a "
            "conservative ~600 nm nominal discrete opening; bad counts are design nodes"
        ),
        "final_continuous": {
            "objective_A": float(last["objective_A"]),
            "objective_at_285uW_A": float(last.get("objective_at_reference_power_A", float("nan"))),
            "gray_fraction": float(last["gray_fraction"]),
            "binarization_4rho1mrho": float(last.get("binarization", float("nan"))),
            "bad_discrete_opening_nodes": int(last["exact_bad_cells"]),
        },
        "stages": stages,
    }
    published.mkdir(parents=True, exist_ok=True)
    (published / "CONTINUATION_CONVERGENCE_ASSESSMENT.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    lines = [
        f"# {args.run_label} continuation convergence assessment",
        "",
        f"Status: `{payload['status']}`.",
        "",
        "The beta schedule completed, but this is **not** a certified local or global optimum. "
        "The rapid binarization is mainly the imposed tanh projection. The final beta=64 stage "
        "ended because its three-update budget was exhausted, not because a strict gradient or "
        "KKT convergence test passed.",
        "",
        "The feature audit also needs a discretization qualification: 500 nm was requested, "
        "while the 100 nm nodal grid rounds the 250 nm opening radius to three offsets, giving "
        "a conservative ~600 nm nominal discrete opening. Reported bad counts are design nodes.",
        "",
        "| beta | accepted updates | gray first→last | binarization first→last | bad nodes first→last | termination |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for stage in stages:
        lines.append(
            f"| {stage['beta']:g} | {stage['accepted_design_updates']} | "
            f"{stage['gray_fraction_first']:.4f}→{stage['gray_fraction_last']:.4f} | "
            f"{stage['binarization_first']:.4f}→{stage['binarization_last']:.4f} | "
            f"{stage['bad_discrete_opening_nodes_first']}→{stage['bad_discrete_opening_nodes_last']} | "
            f"{stage['termination_reason']} |"
        )
    (published / "CONTINUATION_CONVERGENCE_ASSESSMENT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
