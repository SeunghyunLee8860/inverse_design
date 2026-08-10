#!/usr/bin/env python3
"""Restartable sequential supervisor for adaptive Run014 Ea then Run015 Eb."""

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
RAW_PARENT = Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored")
BASE_FSP = RAW_PARENT / (
    "run012_uniform_rho0p5_Ea_forward_retry_20260810/"
    "tairte4_flake_forward_Ea.fsp"
)
BASE_SHA256 = "454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83"
JACOBIAN = RAW_PARENT / "run012_component_yee_jacobian_retry_20260810"
GPU = 1
MAX_RESTARTS_PER_POLARIZATION = 8
RUNS = (
    {
        "name": "run014",
        "polarization": "Ea",
        "raw": RAW_PARENT / "run014_adaptive_Ea_20260810",
        "published": REPOSITORY / (
            "photothermal_pte/optimization_runs/"
            "run_014_adaptive_contact_anchored_Ea_current_max"
        ),
    },
    {
        "name": "run015",
        "polarization": "Eb",
        "raw": RAW_PARENT / "run015_adaptive_Eb_20260810",
        "published": REPOSITORY / (
            "photothermal_pte/optimization_runs/"
            "run_015_adaptive_contact_anchored_Eb_current_max"
        ),
    },
)
STATUS = REPOSITORY / "photothermal_pte/optimization_runs/ADAPTIVE_DUAL_RUN_STATUS.json"


def events(raw: Path) -> list[dict[str, object]]:
    path = raw / "events.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def complete(raw: Path) -> bool:
    return any(row.get("event") == "continuous_continuation_complete" for row in events(raw))


def archive_incomplete(raw: Path, restart: int) -> str | None:
    rows = events(raw)
    starts = [row for row in rows if row.get("event") == "evaluation_start"]
    if not starts:
        return None
    output = Path(str(starts[-1]["output"]))
    ends = [
        row for row in rows
        if row.get("event") == "evaluation_end" and row.get("output") == str(output)
    ]
    valid = bool(ends and ends[-1].get("returncode") == 0)
    if valid or not output.is_dir():
        return None
    destination = output.with_name(f"{output.name}_interrupted_{restart:02d}")
    suffix = 1
    while destination.exists():
        destination = output.with_name(
            f"{output.name}_interrupted_{restart:02d}_{suffix:02d}"
        )
        suffix += 1
    shutil.move(str(output), str(destination))
    return str(destination)


def write_status(status: str, **extra: object) -> None:
    payload = {
        "schema": "adaptive-dual-polarization-supervisor-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "gpu": GPU,
        **extra,
    }
    STATUS.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


def command(run: dict[str, object]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology.run_optimization",
        "--polarization", str(run["polarization"]),
        "--raw-root", str(run["raw"]),
        "--published-dir", str(run["published"]),
        "--gpu", str(GPU),
        "--base-fsp", str(BASE_FSP),
        "--base-sha256", BASE_SHA256,
        "--jacobian-dir", str(JACOBIAN),
        "--connectivity-fraction", "0.10",
    ]


def main() -> int:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(GPU)
    environment["TAIRTE4_TOPOLOGY_GEOMETRY"] = "contact_anchored"
    for run in RUNS:
        raw = Path(run["raw"])
        published = Path(run["published"])
        raw.mkdir(parents=True, exist_ok=True)
        published.mkdir(parents=True, exist_ok=True)
        restarts = 0
        while not complete(raw):
            if restarts >= MAX_RESTARTS_PER_POLARIZATION:
                write_status(
                    "BLOCKED_RESTART_LIMIT", run=run["name"],
                    polarization=run["polarization"], restarts=restarts,
                )
                return 2
            restarts += 1
            archived = archive_incomplete(raw, restarts)
            write_status(
                "RUNNING", run=run["name"], polarization=run["polarization"],
                restart=restarts, archived_incomplete=archived,
            )
            completed = subprocess.run(command(run), cwd=REPOSITORY, env=environment)
            if completed.returncode == 0 and complete(raw):
                break
            write_status(
                "RESTARTING_FROM_CHECKPOINT", run=run["name"],
                polarization=run["polarization"], restart=restarts,
                returncode=completed.returncode,
            )
            time.sleep(10.0)
        write_status(
            "POLARIZATION_COMPLETE", run=run["name"],
            polarization=run["polarization"], next=("Eb" if run["polarization"] == "Ea" else None),
        )
    write_status("ADAPTIVE_EA_EB_CONTINUATION_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
