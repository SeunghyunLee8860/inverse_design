#!/usr/bin/env python3
"""Fail-closed acceptance check for one evaluated continuation proposal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from beta_continuation_support import design_metrics


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
    parser.add_argument("--proposal-result", type=Path, required=True)
    parser.add_argument("--proposal-raw", type=Path, required=True)
    parser.add_argument("--evaluation-result", type=Path, required=True)
    parser.add_argument("--evaluation-raw", type=Path, required=True)
    parser.add_argument("--accepted-mma-state", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--per-constraint-worsening-tolerance", type=float, default=0.005)
    parser.add_argument("--minimum-aggregate-improvement", type=float, default=0.005)
    parser.add_argument("--maximum-objective-drop", type=float, default=0.05)
    args = parser.parse_args()
    proposal_result_path = args.proposal_result.expanduser().resolve()
    proposal_raw_path = args.proposal_raw.expanduser().resolve()
    evaluation_result_path = args.evaluation_result.expanduser().resolve()
    evaluation_raw_path = args.evaluation_raw.expanduser().resolve()
    proposal = json.loads(proposal_result_path.read_text())
    evaluation = json.loads(evaluation_result_path.read_text())
    if sha256(proposal_raw_path) != proposal["raw_artifact"]["sha256"]:
        raise RuntimeError("proposal NPZ SHA mismatch")
    if sha256(evaluation_raw_path) != evaluation["raw_artifact"]["sha256"]:
        raise RuntimeError("evaluation NPZ SHA mismatch")
    proposal_data = np.load(proposal_raw_path)
    evaluation_data = np.load(evaluation_raw_path)
    proposed_latent = np.asarray(proposal_data["latent"], float)
    evaluated_latent = np.asarray(evaluation_data["latent"], float)
    latent_exact = bool(np.array_equal(proposed_latent, evaluated_latent))
    beta_exact = bool(float(proposal["beta"]) == float(evaluation["beta"]))
    candidate_metrics, _ = design_metrics(evaluated_latent, float(evaluation["beta"]))
    current = np.asarray(proposal["constraint_current"], float)
    candidate = np.asarray([
        candidate_metrics["smooth_solid_constraint"],
        candidate_metrics["smooth_void_constraint"],
        candidate_metrics["binarization_metric_mean_4rho1mrho"],
    ])
    ratios = candidate / np.maximum(current, 1.0e-300)
    per_constraint_pass = bool(
        np.all(ratios <= 1.0 + float(args.per_constraint_worsening_tolerance))
    )
    aggregate_improvement = float(3.0 - np.sum(ratios))
    aggregate_pass = bool(aggregate_improvement >= float(args.minimum_aggregate_improvement))
    previous_result = json.loads(
        Path(proposal["inputs"]["evaluation_result"]["path"]).read_text()
    )
    previous_objective = float(previous_result["objective_A"])
    candidate_objective = float(evaluation.get("objective_A", np.nan))
    objective_ratio = candidate_objective / previous_objective
    objective_pass = bool(
        np.isfinite(objective_ratio)
        and objective_ratio >= 1.0 - float(args.maximum_objective_drop)
    )
    physics_pass = bool(evaluation.get("passed"))
    accepted = bool(
        latent_exact
        and beta_exact
        and physics_pass
        and per_constraint_pass
        and aggregate_pass
        and objective_pass
    )
    accepted_state = args.accepted_mma_state.expanduser().resolve()
    accepted_state.parent.mkdir(parents=True, exist_ok=True)
    if accepted:
        candidate_state = Path(proposal["candidate_state"]["path"])
        if sha256(candidate_state) != proposal["candidate_state"]["sha256"]:
            raise RuntimeError("candidate MMA-state SHA mismatch")
        shutil.copy2(candidate_state, accepted_state)
    result = {
        "status": "ACCEPTED_BETA_CONTINUATION_STEP" if accepted else "REJECTED_BETA_CONTINUATION_STEP",
        "accepted": accepted,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "global_iteration": proposal["global_iteration_proposed"],
        "beta": evaluation["beta"],
        "checks": {
            "latent_exact": latent_exact,
            "beta_exact": beta_exact,
            "physics_pass": physics_pass,
            "per_constraint_pass": per_constraint_pass,
            "aggregate_pass": aggregate_pass,
            "objective_pass": objective_pass,
        },
        "constraint_names": proposal["constraint_names"],
        "constraint_current": current.tolist(),
        "constraint_candidate": candidate.tolist(),
        "constraint_ratios": ratios.tolist(),
        "aggregate_improvement": aggregate_improvement,
        "objective_previous_A": previous_objective,
        "objective_candidate_A": candidate_objective,
        "objective_ratio": objective_ratio,
        "accepted_mma_state": artifact(accepted_state) if accepted else None,
    }
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
