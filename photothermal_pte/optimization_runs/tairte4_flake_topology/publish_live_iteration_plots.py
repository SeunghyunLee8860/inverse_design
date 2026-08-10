#!/usr/bin/env python3
"""Regenerate, commit, and push each completed optimization evaluation."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time

from photothermal_pte.optimization_runs.tairte4_flake_topology.regenerate_iteration_plots import (
    regenerate,
)


def run(command: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, check=check, text=True)


def publish(
    repo: Path,
    relative_published: Path,
    branch: str,
    evaluation_id: int,
    run_label: str,
) -> bool:
    run(["git", "add", "--", str(relative_published)], cwd=repo)
    changed = run(
        ["git", "diff", "--cached", "--quiet", "--", str(relative_published)],
        cwd=repo,
        check=False,
    ).returncode != 0
    if not changed:
        return False
    run(
        ["git", "commit", "-m", f"Update {run_label} plots through evaluation {evaluation_id}"],
        cwd=repo,
    )
    run(["git", "push", "origin", branch], cwd=repo)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--published-dir", required=True, type=Path)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--run-label")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    raw_root = args.raw_root.resolve()
    published = args.published_dir.resolve()
    relative_published = published.relative_to(repo)
    run_label = args.run_label or published.name
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_label)
    lock_path = Path(f"/tmp/tairte4_{safe_label}_plot_publisher.lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        last_seen = -1
        while True:
            try:
                history = json.loads((raw_root / "history.json").read_text())
                complete = len(history)
                if complete != last_seen:
                    time.sleep(2.0)
                    payload = regenerate(raw_root, published)
                    latest = int(payload["latest_evaluation_id"] or 0)
                    publish(repo, relative_published, args.branch, latest, run_label)
                    last_seen = complete
            except (json.JSONDecodeError, FileNotFoundError):
                pass
            if args.once:
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
