#!/usr/bin/env python3
"""Create an offline, immutable audit of the halted Run 003 continuation.

This script reads accepted checkpoints and already-completed adjoint metadata only.
It never launches Maxwell, thermal, PTE, adjoint, or optimization solves.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = HERE / "plots"
CHECKPOINTS = HERE / "checkpoints"
RAW_ROOT = Path("/data/seunghyun/tairte4/raw_artifacts/run003_converged_constrained_20260807")
BETAS = (2, 4, 8, 16, 32)
HALT_STATUS = "HALTED_PREMATURE_BETA2_SATURATION_AND_LATE_CONSTRAINT_REPAIR"


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def final_gradient_norm(beta: int, stage: int, global_iteration: int) -> float:
    pattern = (
        f"b{beta:04d}_s{stage:03d}_g{global_iteration:03d}_retry*_evaluation/"
        "selected_full_latent_adjoint_preparation_result.json"
    )
    candidates = sorted(RAW_ROOT.glob(pattern))
    if not candidates:
        return float("nan")
    return float(load_json(candidates[-1])["gradient_norms_A"]["latent"])


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    summary = load_json(RESULTS / "optimization_summary.json")
    history = summary["history"]
    rows = []
    for beta in BETAS:
        baselines = [r for r in history if r["beta"] == beta and r["role"] == "stage_baseline"]
        accepted = [r for r in history if r["beta"] == beta and r["role"] == "accepted_mma"]
        if not baselines or not accepted:
            raise RuntimeError(f"missing baseline/accepted history for beta={beta}")
        baseline, first, last = baselines[-1], accepted[0], accepted[-1]
        checkpoint = sorted(CHECKPOINTS.glob(f"run003_b{beta:03d}_*_accepted_mma.npz"))[-1]
        latent = np.load(checkpoint)["latent"].astype(float)
        at_zero = float(np.mean(latent <= 1e-12))
        at_one = float(np.mean(latent >= 1.0 - 1e-12))
        interior = 1.0 - at_zero - at_one
        grad = final_gradient_norm(beta, int(last["stage_iteration"]), int(last["global_iteration"]))
        rows.append(
            {
                "beta": beta,
                "accepted_updates": len(accepted),
                "baseline_fom_A_per_W": float(baseline["objective_A_per_W"]),
                "first_accepted_fom_A_per_W": float(first["objective_A_per_W"]),
                "final_fom_A_per_W": float(last["objective_A_per_W"]),
                "stage_fom_gain_percent": 100.0 * (last["objective_A_per_W"] / baseline["objective_A_per_W"] - 1.0),
                "accepted_span_fom_gain_percent": 100.0 * (last["objective_A_per_W"] / first["objective_A_per_W"] - 1.0),
                "solid_constraint": float(last["solid_constraint"]),
                "void_constraint": float(last["void_constraint"]),
                "constraint_cap": float(last["constraint_cap"]),
                "solid_constraint_over_cap": float(last["solid_constraint"] / last["constraint_cap"]),
                "void_constraint_over_cap": float(last["void_constraint"] / last["constraint_cap"]),
                "exact_solid_bad_cells": int(last["solid_bad_cells"]),
                "exact_void_bad_cells": int(last["void_bad_cells"]),
                "latent_at_zero_fraction": at_zero,
                "latent_at_one_fraction": at_one,
                "latent_box_saturated_fraction": at_zero + at_one,
                "latent_interior_fraction": interior,
                "latent_objective_gradient_l2_A": grad,
                "final_checkpoint": str(checkpoint),
            }
        )

    csv_path = RESULTS / "run003_beta_stage_audit.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    beta2 = rows[0]
    beta16 = rows[3]
    audit = {
        "status": HALT_STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "offline audit of accepted Run 003 history; no new solver execution",
        "last_accepted_checkpoint": "run003_b032_s012_g095_accepted_mma",
        "interrupted_unaccepted_work": "g096; interrupted during adjoint and excluded from all accepted histories",
        "solver_processes_running_after_halt": False,
        "findings": {
            "beta2_stage_fom_gain_percent": beta2["stage_fom_gain_percent"],
            "beta2_accepted_span_fom_gain_percent": beta2["accepted_span_fom_gain_percent"],
            "beta2_latent_box_saturated_percent": 100.0 * beta2["latent_box_saturated_fraction"],
            "beta2_constraint_activation": {
                "solid_over_cap": beta2["solid_constraint_over_cap"],
                "void_over_cap": beta2["void_constraint_over_cap"],
            },
            "beta16_constraint_activation": {
                "solid_over_cap": beta16["solid_constraint_over_cap"],
                "void_over_cap": beta16["void_constraint_over_cap"],
            },
            "latent_gradient_collapse_beta2_to_beta16_factor": (
                beta2["latent_objective_gradient_l2_A"] / beta16["latent_objective_gradient_l2_A"]
            ),
            "exact_bad_cells_beta2_final": {
                "solid": beta2["exact_solid_bad_cells"],
                "void": beta2["exact_void_bad_cells"],
            },
            "exact_bad_cells_beta16_final": {
                "solid": beta16["exact_solid_bad_cells"],
                "void": beta16["exact_void_bad_cells"],
            },
        },
        "diagnosis": [
            "The beta=2 stage performed nearly all useful objective optimization while the manufacturing inequalities had large slack.",
            "By the end of beta=2 most latent variables were already at box bounds, leaving little design freedom for later stages.",
            "The latent objective gradient collapsed as beta increased, while the exact 500 nm morphology became substantially worse at beta=4--16.",
            "The beta=32 work was therefore primarily late constraint repair, not productive joint objective/manufacturability optimization.",
        ],
        "not_claimed": [
            "Run 003 is not converged.",
            "The exact 500 nm solid/void gate has not passed.",
            "g096 is not an accepted checkpoint.",
        ],
        "beta_rows": rows,
    }
    json_path = RESULTS / "run003_continuation_pathology_summary.json"
    json_path.write_text(json.dumps(audit, indent=2) + "\n")

    betas = np.array([r["beta"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(betas, [r["final_fom_A_per_W"] for r in rows], "o-", lw=2)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(betas, [str(v) for v in betas])
    axes[0].set_xlabel(r"projection $\beta$")
    axes[0].set_ylabel("final accepted FOM (A/W)")
    axes[0].set_title("FOM plateau after beta=2")
    axes[0].grid(alpha=0.3)
    axes[1].bar([str(v) for v in betas], [r["stage_fom_gain_percent"] for r in rows])
    axes[1].set_xlabel(r"projection $\beta$")
    axes[1].set_ylabel("stage FOM gain (%)")
    axes[1].set_title("Nearly all gain occurred before constraints activated")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "run003_fom_gain_by_beta.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(betas, [r["solid_constraint_over_cap"] for r in rows], "o-", label="solid / cap")
    axes[0].plot(betas, [r["void_constraint_over_cap"] for r in rows], "s-", label="void / cap")
    axes[0].axhline(1.0, color="k", ls="--", label="active limit")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(betas, [str(v) for v in betas])
    axes[0].set_xlabel(r"projection $\beta$")
    axes[0].set_ylabel("smooth constraint / stage cap")
    axes[0].set_title("Manufacturing constraints activated too late")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(betas, [r["exact_solid_bad_cells"] for r in rows], "o-", label="exact solid bad cells")
    axes[1].plot(betas, [r["exact_void_bad_cells"] for r in rows], "s-", label="exact void bad cells")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(betas, [str(v) for v in betas])
    axes[1].set_xlabel(r"projection $\beta$")
    axes[1].set_ylabel("exact 500 nm bad cells")
    axes[1].set_title("Exact DRC degraded during beta=4--16")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "run003_constraint_activation_and_exact_drc.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(betas, [100 * r["latent_box_saturated_fraction"] for r in rows], "o-", label="at 0 or 1")
    axes[0].plot(betas, [100 * r["latent_interior_fraction"] for r in rows], "s-", label="strict interior")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(betas, [str(v) for v in betas])
    axes[0].set_xlabel(r"projection $\beta$")
    axes[0].set_ylabel("latent-variable fraction (%)")
    axes[0].set_title("Premature box saturation")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].semilogy(betas, [r["latent_objective_gradient_l2_A"] for r in rows], "o-", lw=2)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(betas, [str(v) for v in betas])
    axes[1].set_xlabel(r"projection $\beta$")
    axes[1].set_ylabel(r"latent objective-gradient $L_2$ norm (A)")
    axes[1].set_title("Objective-gradient collapse")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "run003_latent_saturation_gradient_collapse.png", dpi=180)
    plt.close(fig)

    md = f"""# Run 003 continuation pathology audit

Status: `{HALT_STATUS}`

This is an offline audit of already accepted checkpoints. No Maxwell, thermal,
PTE, adjoint, or optimization solve was launched. The last accepted checkpoint
is `g095`; interrupted `g096` is excluded.

## Conclusion

The user's concern is correct. Run 003 did not perform a healthy joint
objective/manufacturability continuation. Beta=2 captured nearly all useful FOM
gain while its manufacturing inequalities were inactive and {100*beta2['latent_box_saturated_fraction']:.1f}%
of latent variables reached a box bound. Later stages inherited a largely frozen
topology, the objective gradient collapsed by a factor of
{audit['findings']['latent_gradient_collapse_beta2_to_beta16_factor']:.1f} by beta=16,
and exact 500 nm defects grew from {beta2['exact_solid_bad_cells']}/{beta2['exact_void_bad_cells']}
(solid/void) at beta=2 to {beta16['exact_solid_bad_cells']}/{beta16['exact_void_bad_cells']}
at beta=16. Beta=32 therefore became late repair work.

## Stage evidence

| beta | accepted updates | stage FOM gain | final FOM (A/W) | solid/cap | void/cap | exact bad solid/void | latent at bounds | objective grad L2 (A) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in rows:
        md += (
            f"| {row['beta']} | {row['accepted_updates']} | {row['stage_fom_gain_percent']:.4f}% | "
            f"{row['final_fom_A_per_W']:.9e} | {row['solid_constraint_over_cap']:.3f} | "
            f"{row['void_constraint_over_cap']:.3f} | {row['exact_solid_bad_cells']}/"
            f"{row['exact_void_bad_cells']} | {100*row['latent_box_saturated_fraction']:.1f}% | "
            f"{row['latent_objective_gradient_l2_A']:.3e} |\n"
        )
    md += """

## Why continuing was wasteful

1. The loose beta=2 caps (`0.04`) left the constraints far from active at the
   final beta=2 design (solid/cap 0.335, void/cap 0.252).
2. Forty-four accepted beta=2 updates drove the objective and 88.3% of latent
   variables to their box limits before exact manufacturability was controlled.
3. The legacy smooth surrogate did not track exact disk-opening defects: exact
   defects worsened sharply at beta=4--16 despite nominal smooth feasibility.
4. Introducing the disk-opening contract only at beta=32 could repair defects,
   but could not recover the design freedom already lost. The tiny late moves
   were therefore expensive morphology repair with negligible FOM benefit.

## Required restart principle (proposal only)

Do not resume g096. A replacement run should activate the same 500 nm disk-based
solid and void constraints from the first iteration, prevent prolonged objective-
only saturation at low beta, and advance beta based on joint FOM and morphology
progress. The exact schedule and acceptance rule require user review before any
new GPU execution.

Run 003 is not converged and has not passed exact 500 nm solid/void constraints.
"""
    (RESULTS / "RUN003_CONTINUATION_PATHOLOGY_AUDIT.md").write_text(md)

    status_path = HERE / "STATUS.json"
    status = load_json(status_path)
    status.update(
        {
            "status": HALT_STATUS,
            "last_updated_utc": audit["generated_at_utc"],
            "halt_reason": "beta=2 premature latent saturation followed by late exact-constraint repair",
            "last_accepted_checkpoint": "run003_b032_s012_g095_accepted_mma",
            "interrupted_unaccepted_checkpoint": "g096",
            "optimization_process_running": False,
        }
    )
    status_path.write_text(json.dumps(status, indent=2) + "\n")

    report_path = RESULTS / "OPTIMIZATION_REPORT.md"
    report_path.write_text(
        f"""# Run 003 convergence-based constrained PTE optimization

Status: `{HALT_STATUS}`

The run was deliberately halted after accepted checkpoint `g095`. The partial
`g096` evaluation was interrupted during the adjoint and is not accepted.

Actual FOM at g095: `8.852107939055e-07 A/W`. Fixed-cap solid/void constraints:
`9.230433e-04` / `2.014664e-03` with cap `2.000000e-03`.

Exact 500 nm bad cells: solid `95`, void `260`. The run is neither converged nor
manufacturing-feasible. See `RUN003_CONTINUATION_PATHOLOGY_AUDIT.md` for the
beta-stage evidence and restart rationale.
"""
    )


if __name__ == "__main__":
    main()
