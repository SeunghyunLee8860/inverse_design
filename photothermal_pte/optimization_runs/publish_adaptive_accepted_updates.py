#!/usr/bin/env python3
"""Commit and push only accepted Run014/015 iteration figures."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


REPOSITORY = Path("/home/seunghyun/tairte4/worktrees/pte_optimization_runs")
RAW_PARENT = Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored")
BRANCH = "agent/organize-pte-optimization-runs"
POLL_SECONDS = 20.0
RUNS = (
    (
        "Run014-Ea",
        RAW_PARENT / "run014_adaptive_Ea_20260810",
        REPOSITORY / "photothermal_pte/optimization_runs/run_014_adaptive_contact_anchored_Ea_current_max",
    ),
    (
        "Run015-Eb",
        RAW_PARENT / "run015_adaptive_Eb_20260810",
        REPOSITORY / "photothermal_pte/optimization_runs/run_015_adaptive_contact_anchored_Eb_current_max",
    ),
)
STATE = RAW_PARENT / "adaptive_github_publish_state.json"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=REPOSITORY, text=True, capture_output=True, check=check
    )


def load_state() -> dict[str, int]:
    if not STATE.is_file():
        return {}
    return {key: int(value) for key, value in json.loads(STATE.read_text()).items()}


def save_state(state: dict[str, int]) -> None:
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def accepted_rows(raw: Path) -> list[dict[str, object]]:
    history = raw / "history.json"
    if not history.is_file():
        return []
    return [row for row in json.loads(history.read_text()) if row.get("accepted")]


def figure_for(published: Path, evaluation_id: int) -> Path | None:
    matches = sorted(published.glob(f"evaluation_{evaluation_id:04d}_*.png"))
    return matches[0] if len(matches) == 1 else None


def publish_one(label: str, raw: Path, published: Path, state: dict[str, int]) -> bool:
    rows = accepted_rows(raw)
    if not rows:
        return False
    last_published = state.get(label, 0)
    pending = [row for row in rows if int(row["evaluation_id"]) > last_published]
    paths: list[Path] = []
    complete_ids: list[int] = []
    for row in pending:
        evaluation_id = int(row["evaluation_id"])
        figure = figure_for(published, evaluation_id)
        if figure is None:
            continue
        paths.append(figure)
        complete_ids.append(evaluation_id)
    if not complete_ids:
        return False
    for name in ("latest_iteration.png", "latest_summary.json"):
        path = published / name
        if path.is_file():
            paths.append(path)
    relative = [str(path.relative_to(REPOSITORY)) for path in paths]
    run(["git", "add", "--", *relative])
    staged = run(["git", "diff", "--cached", "--quiet"], check=False)
    if staged.returncode == 0:
        state[label] = max(complete_ids)
        save_state(state)
        return False
    highest = max(complete_ids)
    run(["git", "commit", "-m", f"Update {label} accepted evaluation {highest}"])
    run(["git", "push", "origin", BRANCH])
    state[label] = highest
    save_state(state)
    print(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "published": label,
                "accepted_evaluation": highest,
            }
        ),
        flush=True,
    )
    return True


def main() -> int:
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    if branch != BRANCH:
        raise RuntimeError(f"refusing to publish branch {branch!r}; expected {BRANCH!r}")
    state = load_state()
    while True:
        for label, raw, published in RUNS:
            try:
                publish_one(label, raw, published, state)
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                            "status": "PUBLISH_RETRY",
                            "run": label,
                            "error": str(exc),
                        }
                    ),
                    flush=True,
                )
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
