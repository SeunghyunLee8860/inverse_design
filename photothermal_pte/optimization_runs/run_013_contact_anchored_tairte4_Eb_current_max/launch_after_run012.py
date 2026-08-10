#!/usr/bin/env python3
"""Fail-closed Run012(Ea) -> independent Run013(Eb) optimization supervisor."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


REPOSITORY = Path("/home/seunghyun/tairte4/worktrees/pte_optimization_runs")
RUN012_RAW = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "run012_Ea_pilot_20260810"
)
RUN013_RAW = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "run013_Eb_optimization_20260810"
)
RUN013_PUBLISHED = REPOSITORY / (
    "photothermal_pte/optimization_runs/"
    "run_013_contact_anchored_tairte4_Eb_current_max"
)
BASE_FSP = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "run012_uniform_rho0p5_Ea_forward_retry_20260810/"
    "tairte4_flake_forward_Ea.fsp"
)
BASE_SHA256 = "454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83"
JACOBIAN = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "run012_component_yee_jacobian_retry_20260810"
)
STATUS_PATH = RUN013_PUBLISHED / "AUTO_CHAIN_STATUS.json"
GPU_INDEX = 5


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(status: str, **extra: object) -> None:
    payload = {
        "schema": "tairte4-ea-to-eb-auto-chain-v1",
        "generated_at_utc": now(),
        "status": status,
        "run012_raw": str(RUN012_RAW),
        "run013_raw": str(RUN013_RAW),
        "run013_published": str(RUN013_PUBLISHED),
        "GPU_index": GPU_INDEX,
        "Eb_start_density": "independent uniform rho=0.5; no Ea warm start",
        "source_axis_contract": "Lumerical x=b, y=a; Eb polarization angle=0 deg",
        **extra,
    }
    RUN013_PUBLISHED.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def has_event(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text().splitlines():
        try:
            if json.loads(line).get("event") == expected:
                return True
        except json.JSONDecodeError:
            continue
    return False


def main() -> int:
    run012_events = RUN012_RAW / "events.jsonl"
    run013_events = RUN013_RAW / "events.jsonl"
    if has_event(run013_events, "continuous_continuation_complete"):
        write_status("RUN013_ALREADY_COMPLETE")
        return 0

    write_status("WAITING_FOR_RUN012_NORMAL_COMPLETION")
    while not has_event(run012_events, "continuous_continuation_complete"):
        time.sleep(20.0)

    if not BASE_FSP.is_file():
        write_status("BLOCKED_MISSING_BASE_FSP", base_fsp=str(BASE_FSP))
        return 2
    actual_sha = sha256(BASE_FSP)
    if actual_sha != BASE_SHA256:
        write_status(
            "BLOCKED_BASE_FSP_SHA_MISMATCH",
            base_fsp=str(BASE_FSP),
            expected_sha256=BASE_SHA256,
            actual_sha256=actual_sha,
        )
        return 2
    jacobian_result = JACOBIAN / "component_yee_jacobian_result.json"
    if not jacobian_result.is_file():
        write_status("BLOCKED_MISSING_COMPONENT_YEE_JACOBIAN", jacobian=str(JACOBIAN))
        return 2

    command = [
        sys.executable,
        "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology.run_optimization",
        "--polarization", "Eb",
        "--raw-root", str(RUN013_RAW),
        "--published-dir", str(RUN013_PUBLISHED),
        "--gpu", str(GPU_INDEX),
        "--base-fsp", str(BASE_FSP),
        "--base-sha256", BASE_SHA256,
        "--jacobian-dir", str(JACOBIAN),
    ]
    write_status(
        "RUN013_LAUNCHING",
        run012_completion_event="continuous_continuation_complete",
        base_fsp=str(BASE_FSP),
        base_fsp_sha256=actual_sha,
        component_yee_jacobian=str(jacobian_result),
        command=command,
    )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
    completed = subprocess.run(command, cwd=REPOSITORY, env=environment)
    if completed.returncode != 0:
        write_status("RUN013_FAILED", returncode=completed.returncode)
        return completed.returncode
    if not has_event(run013_events, "continuous_continuation_complete"):
        write_status("RUN013_EXITED_WITHOUT_COMPLETION_EVENT")
        return 3
    write_status("RUN013_COMPLETE", returncode=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
