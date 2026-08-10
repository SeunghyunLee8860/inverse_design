#!/usr/bin/env python3
"""Fail-closed watchdog for the long independent E||b continuation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPOSITORY = Path("/home/seunghyun/tairte4/worktrees/pte_optimization_runs")
RAW_ROOT = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "run013_Eb_optimization_20260810"
)
PUBLISHED = REPOSITORY / (
    "photothermal_pte/optimization_runs/"
    "run_013_contact_anchored_tairte4_Eb_current_max"
)
BASE_FSP = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "run012_uniform_rho0p5_Ea_forward_retry_20260810/tairte4_flake_forward_Ea.fsp"
)
BASE_SHA256 = "454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83"
JACOBIAN = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "run012_component_yee_jacobian_retry_20260810"
)
STATUS = PUBLISHED / "SUPERVISOR_STATUS.json"
GPU = 5
MAX_RESTARTS = 3


def read_events() -> list[dict[str, object]]:
    path = RAW_ROOT / "events.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def complete() -> bool:
    return any(row.get("event") == "continuous_continuation_complete" for row in read_events())


def optimizer_is_running() -> bool:
    needle = str(RAW_ROOT).encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if needle in command and b"tairte4_flake_topology.run_optimization" in command:
            return True
    return False


def write_status(status: str, **extra: object) -> None:
    payload = {
        "schema": "run013-fail-closed-watchdog-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "GPU_index": GPU,
        "raw_root": str(RAW_ROOT),
        "maximum_automatic_restarts": MAX_RESTARTS,
        **extra,
    }
    STATUS.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


def archive_incomplete_evaluation(restart: int) -> str | None:
    events = read_events()
    starts = [row for row in events if row.get("event") == "evaluation_start"]
    if not starts:
        return None
    latest = starts[-1]
    output = Path(str(latest["output"]))
    matching_end = [
        row for row in events
        if row.get("event") == "evaluation_end" and row.get("output") == str(output)
    ]
    valid = bool(
        matching_end
        and matching_end[-1].get("returncode") == 0
        and matching_end[-1].get("status") == "VALIDATED_TAIRTE4_FLAKE_OBJECTIVE_GRADIENT"
    )
    if valid or not output.is_dir():
        return None
    destination = output.with_name(f"{output.name}_interrupted_retry_{restart:02d}")
    suffix = 1
    while destination.exists():
        destination = output.with_name(
            f"{output.name}_interrupted_retry_{restart:02d}_{suffix:02d}"
        )
        suffix += 1
    shutil.move(str(output), str(destination))
    return str(destination)


def command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology.run_optimization",
        "--polarization", "Eb",
        "--raw-root", str(RAW_ROOT),
        "--published-dir", str(PUBLISHED),
        "--gpu", str(GPU),
        "--base-fsp", str(BASE_FSP),
        "--base-sha256", BASE_SHA256,
        "--jacobian-dir", str(JACOBIAN),
    ]


def main() -> int:
    restarts = 0
    while True:
        if complete():
            write_status("RUN013_COMPLETE", automatic_restarts=restarts)
            return 0
        if optimizer_is_running():
            write_status("RUN013_RUNNING", automatic_restarts=restarts)
            time.sleep(20.0)
            continue
        if restarts >= MAX_RESTARTS:
            write_status("RUN013_BLOCKED_AFTER_RESTART_LIMIT", automatic_restarts=restarts)
            return 2
        restarts += 1
        archived = archive_incomplete_evaluation(restarts)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(GPU)
        environment["TAIRTE4_TOPOLOGY_GEOMETRY"] = "contact_anchored"
        write_status(
            "RUN013_RESTARTING_FROM_LAST_ACCEPTED_CHECKPOINT",
            automatic_restarts=restarts,
            archived_incomplete_evaluation=archived,
        )
        completed = subprocess.run(command(), cwd=REPOSITORY, env=environment)
        if completed.returncode == 0 and complete():
            write_status("RUN013_COMPLETE", automatic_restarts=restarts)
            return 0
        write_status(
            "RUN013_EXITED_WITHOUT_COMPLETION",
            automatic_restarts=restarts,
            returncode=completed.returncode,
        )
        time.sleep(10.0)


if __name__ == "__main__":
    raise SystemExit(main())
