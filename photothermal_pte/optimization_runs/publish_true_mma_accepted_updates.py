#!/usr/bin/env python3
"""Commit/push only small MMA reports and evaluated-iteration plots."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


REPOSITORY = Path(__file__).resolve().parents[2]
BRANCH = "agent/restart-true-mma-pte-optimization"
POLL_SECONDS = 20.0
DIRECTORIES = (
    REPOSITORY / "photothermal_pte/optimization_runs/run_016_true_mma_contact_anchored_Ea_current_max",
    REPOSITORY / "photothermal_pte/optimization_runs/run_017_true_mma_contact_anchored_Eb_current_max",
    REPOSITORY / "photothermal_pte/optimization_runs/true_mma_preflight",
    REPOSITORY / "photothermal_pte/optimization_runs/run_018_nlopt_mma_contact_anchored_Ea_current_max",
    REPOSITORY / "photothermal_pte/optimization_runs/run_019_nlopt_mma_contact_anchored_Eb_current_max",
)
STATUSES = (
    REPOSITORY / "photothermal_pte/optimization_runs/TRUE_MMA_DUAL_RUN_STATUS.json",
    REPOSITORY / "photothermal_pte/optimization_runs/NLOPT_MMA_DUAL_RUN_STATUS.json",
)


def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=check,
    )


def publish() -> None:
    paths = [str(path.relative_to(REPOSITORY)) for path in DIRECTORIES if path.exists()]
    for status in STATUSES:
        if status.exists():
            paths.append(str(status.relative_to(REPOSITORY)))
    if not paths:
        return
    git("add", "--", *paths)
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        return
    # No raw FSP/NPZ is inside the tracked publication directories.
    names = git("diff", "--cached", "--name-only").stdout.splitlines()
    forbidden = [name for name in names if name.endswith((".fsp", ".npz"))]
    if forbidden:
        git("restore", "--staged", "--", *forbidden)
        raise RuntimeError(f"refusing raw artifact publication: {forbidden}")
    latest = 0
    for directory in DIRECTORIES:
        for path in list(directory.glob("iteration_*.json")) + list(directory.glob("evaluation_*.json")):
            latest = max(latest, int(path.stem.split("_")[-1]))
    git("commit", "-m", f"Update MMA evaluated iteration {latest}")
    git("push", "origin", BRANCH)
    print(json.dumps({"published_accepted_iteration": latest}), flush=True)


def main() -> int:
    branch = git("branch", "--show-current").stdout.strip()
    if branch != BRANCH:
        raise RuntimeError(f"refusing branch {branch!r}; expected {BRANCH!r}")
    while True:
        try:
            publish()
        except Exception as error:
            print(json.dumps({"status": "PUBLISH_RETRY", "error": str(error)}), flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
