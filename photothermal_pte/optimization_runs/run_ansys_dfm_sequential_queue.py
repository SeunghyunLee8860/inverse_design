#!/usr/bin/env python3
"""Wait for a validated optimization, then launch the queued polarization.

Lumerical v261 did not reliably support two concurrent FDTD API sessions for
this account even when their engine calls were serialized and different GPUs
were selected.  This supervisor therefore keeps the expensive optimizations
fully sequential: the queued child starts only after the preceding
``FINAL_RESULT.json`` exists and reports ``passed=true``.
"""

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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_passed(path: Path) -> bool:
    try:
        return bool(json.loads(path.read_text()).get("passed"))
    except (OSError, json.JSONDecodeError):
        return False


def gpu_compute_processes(gpu: int) -> list[dict[str, object]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i", str(gpu),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=2)]
        if len(fields) != 3 or not fields[0]:
            continue
        processes.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_memory_MiB": int(fields[2]),
            }
        )
    return processes


def status_payload(args: argparse.Namespace, state: str, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "ansys-dfm-sequential-polarization-queue-v1",
        "generated_at_utc": utc_now(),
        "state": state,
        "predecessor": {
            "pid": args.predecessor_pid,
            "final_result": str(args.predecessor_final),
        },
        "queued": {
            "polarization": args.polarization,
            "gpu": args.gpu,
            "raw_root": str(args.raw_root),
            "published_dir": str(args.published_dir),
            "log": str(args.log),
        },
    }
    payload.update(extra)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor-pid", type=int, required=True)
    parser.add_argument("--predecessor-final", type=Path, required=True)
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--published-dir", type=Path, required=True)
    parser.add_argument("--base-fsp", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--jacobian-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0.0:
        raise ValueError("poll interval must be positive")

    for attribute in (
        "predecessor_final",
        "raw_root",
        "published_dir",
        "base_fsp",
        "jacobian_dir",
        "status",
        "log",
    ):
        setattr(args, attribute, getattr(args, attribute).expanduser().resolve())

    write_json(args.status, status_payload(args, "WAITING_FOR_PREDECESSOR"))
    while not read_passed(args.predecessor_final):
        if not process_exists(args.predecessor_pid):
            write_json(
                args.status,
                status_payload(
                    args,
                    "BLOCKED_PREDECESSOR_STOPPED_WITHOUT_VALIDATED_FINAL",
                ),
            )
            return 2
        time.sleep(args.poll_seconds)
        write_json(args.status, status_payload(args, "WAITING_FOR_PREDECESSOR"))

    while True:
        try:
            occupants = gpu_compute_processes(args.gpu)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            write_json(
                args.status,
                status_payload(
                    args,
                    "WAITING_FOR_GPU_AUDIT",
                    gpu_audit_error=f"{type(error).__name__}: {error}",
                ),
            )
            time.sleep(args.poll_seconds)
            continue
        if not occupants:
            break
        write_json(
            args.status,
            status_payload(
                args,
                "WAITING_FOR_GPU_TO_BECOME_IDLE",
                gpu_compute_processes=occupants,
            ),
        )
        time.sleep(args.poll_seconds)

    if args.raw_root.exists() and any(args.raw_root.iterdir()):
        raise RuntimeError(f"refusing non-empty queued raw root: {args.raw_root}")
    if args.published_dir.exists() and any(args.published_dir.iterdir()):
        raise RuntimeError(
            f"refusing non-empty queued published directory: {args.published_dir}"
        )
    args.raw_root.parent.mkdir(parents=True, exist_ok=True)
    args.published_dir.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-u",
        "-m",
        MODULE,
        "--polarization", args.polarization,
        "--gpu", str(args.gpu),
        "--raw-root", str(args.raw_root),
        "--published-dir", str(args.published_dir),
        "--base-fsp", str(args.base_fsp),
        "--base-sha256", args.base_sha256,
        "--jacobian-dir", str(args.jacobian_dir),
        "--constraint-device", "cuda:0",
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["TAIRTE4_TOPOLOGY_GEOMETRY"] = "contact_anchored"
    with args.log.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_json(
            args.status,
            status_payload(
                args,
                "QUEUED_OPTIMIZATION_RUNNING",
                child_pid=process.pid,
                command=command,
            ),
        )
        while process.poll() is None:
            time.sleep(args.poll_seconds)
            write_json(
                args.status,
                status_payload(
                    args,
                    "QUEUED_OPTIMIZATION_RUNNING",
                    child_pid=process.pid,
                ),
            )
        returncode = int(process.returncode)

    passed = read_passed(args.published_dir / "FINAL_RESULT.json")
    final_state = (
        "COMPLETED_VALIDATED"
        if returncode == 0 and passed
        else "FAILED_QUEUED_OPTIMIZATION"
    )
    write_json(
        args.status,
        status_payload(
            args,
            final_state,
            child_pid=process.pid,
            returncode=returncode,
            final_result_passed=passed,
        ),
    )
    return 0 if final_state == "COMPLETED_VALIDATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
