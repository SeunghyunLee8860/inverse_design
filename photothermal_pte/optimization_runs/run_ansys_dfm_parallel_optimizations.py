#!/usr/bin/env python3
"""Launch and supervise independent Ea/Eb Ansys-style DFM optimizations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[2]
MODULE = (
    "photothermal_pte.optimization_runs.tairte4_flake_topology."
    "run_ansys_dfm_ld_mma_optimization"
)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def command(
    *,
    polarization: str,
    gpu: int,
    raw: Path,
    published: Path,
    base_fsp: Path,
    base_sha256: str,
    jacobian_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        MODULE,
        "--polarization", polarization,
        "--gpu", str(gpu),
        "--raw-root", str(raw),
        "--published-dir", str(published),
        "--base-fsp", str(base_fsp),
        "--base-sha256", base_sha256,
        "--jacobian-dir", str(jacobian_dir),
        "--constraint-device", "cuda:0",
        "--constraint-aware-continuation",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ea", type=int, required=True)
    parser.add_argument("--gpu-eb", type=int, required=True)
    parser.add_argument("--raw-ea", required=True, type=Path)
    parser.add_argument("--raw-eb", required=True, type=Path)
    parser.add_argument("--published-ea", required=True, type=Path)
    parser.add_argument("--published-eb", required=True, type=Path)
    parser.add_argument("--base-fsp", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--jacobian-dir", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.gpu_ea == args.gpu_eb:
        raise RuntimeError("Ea and Eb must use different physical GPUs")
    if args.poll_seconds <= 0.0:
        raise ValueError("poll interval must be positive")

    status_path = args.status.expanduser().resolve()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    runs = {}
    for polarization, gpu, raw_arg, published_arg in (
        ("Ea", args.gpu_ea, args.raw_ea, args.published_ea),
        ("Eb", args.gpu_eb, args.raw_eb, args.published_eb),
    ):
        raw = raw_arg.expanduser().resolve()
        published = published_arg.expanduser().resolve()
        raw.parent.mkdir(parents=True, exist_ok=True)
        published.parent.mkdir(parents=True, exist_ok=True)
        log = raw.with_name(raw.name + ".log")
        stream = log.open("ab", buffering=0)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["XDG_CONFIG_HOME"] = f"/tmp/seunghyun_lumerical_constraint_aware_{polarization}"
        env["MPLCONFIGDIR"] = f"/tmp/seunghyun_matplotlib_constraint_aware_{polarization}"
        # The contract module is imported when the child interpreter starts.
        # Pin the intended top/bottom-contact geometry explicitly instead of
        # inheriting its fixed-frame default from an interactive shell.
        env["TAIRTE4_TOPOLOGY_GEOMETRY"] = "contact_anchored"
        cmd = command(
            polarization=polarization,
            gpu=gpu,
            raw=raw,
            published=published,
            base_fsp=args.base_fsp.expanduser().resolve(),
            base_sha256=args.base_sha256,
            jacobian_dir=args.jacobian_dir.expanduser().resolve(),
        )
        process = subprocess.Popen(
            cmd,
            cwd=REPOSITORY,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        runs[polarization] = {
            "gpu": gpu,
            "raw": str(raw),
            "published": str(published),
            "log": str(log),
            "command": cmd,
            "process": process,
            "stream": stream,
        }

    while True:
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema": "parallel-ansys-dfm-ld-mma-supervisor-v1",
            "generated_at_utc": now,
            "runs": {},
        }
        all_finished = True
        all_passed = True
        for polarization, record in runs.items():
            process = record["process"]
            returncode = process.poll()
            all_finished &= returncode is not None
            final_path = Path(record["published"]) / "FINAL_RESULT.json"
            final_passed = False
            if final_path.is_file():
                try:
                    final_passed = bool(json.loads(final_path.read_text()).get("passed"))
                except (OSError, json.JSONDecodeError):
                    final_passed = False
            if returncode is not None:
                all_passed &= returncode == 0 and final_passed
            payload["runs"][polarization] = {
                "gpu": record["gpu"],
                "pid": process.pid,
                "returncode": returncode,
                "running": returncode is None,
                "final_result_passed": final_passed,
                "raw": record["raw"],
                "published": record["published"],
                "log": record["log"],
            }
        payload["all_finished"] = all_finished
        payload["all_passed"] = bool(all_finished and all_passed)
        write_json(status_path, payload)
        if all_finished:
            for record in runs.values():
                record["stream"].close()
            return 0 if all_passed else 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
