#!/usr/bin/env python3
"""Schedule the eight fixed-device beam-response runs across selected GPUs."""

from __future__ import annotations

import argparse
import atexit
import getpass
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from photothermal_pte.optimization_runs.tairte4_flake_topology.beam_response_contract import CASES


ANSYSCL = Path(
    "/home/seunghyun/lumerical_r12/opt/lumerical/v261/"
    "licensingclient/linx64/ansyscl"
)


def start_ansysli_broker(port: int) -> tuple[subprocess.Popen[bytes], dict[str, object]]:
    host = socket.gethostname().split(".", 1)[0]
    acl = f"{host}_{host}_{getpass.getuser()}_261"
    ansys_dir = Path.home() / ".ansys"
    ansys_dir.mkdir(parents=True, exist_ok=True)
    port_file = ansys_dir / f".ansyscl.{host}.{acl}"
    log_path = ansys_dir / f"ansyscl.{host}.{acl}.log"
    if port_file.exists() or port_file.is_symlink():
        port_file.unlink()
    invocation = [
        str(ANSYSCL),
        "-acl",
        acl,
        "-nodaemon",
        "-aclport",
        str(port),
        "-log",
        str(log_path),
    ]
    process = subprocess.Popen(
        invocation,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    for _ in range(200):
        if process.poll() is not None:
            raise RuntimeError(
                f"ANSYSLI broker exited with code {process.returncode}; see {log_path}"
            )
        if port_file.is_file():
            value = port_file.read_text().strip()
            if value.startswith(f"{port}:"):
                return process, {
                    "port": port,
                    "pid": process.pid,
                    "acl": acl,
                    "port_file": str(port_file),
                    "log": str(log_path),
                }
        time.sleep(0.1)
    process.terminate()
    process.wait(timeout=10.0)
    raise RuntimeError(f"ANSYSLI broker did not publish {port_file}")


def stop_ansysli_broker(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)


def completed(path: Path) -> bool:
    result = path / "beam_response_result.json"
    if not result.is_file():
        return False
    try:
        payload = json.loads(result.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("passed") and payload.get("status") == "COMPLETED")


def command(
    run: int,
    gpu: int,
    output: Path,
    ansysli_local_port: int,
) -> tuple[list[str], dict[str, str]]:
    case = CASES[run]
    args = [
        sys.executable,
        "-m",
        "photothermal_pte.optimization_runs.run_exact_binary_beam_response",
        "--run",
        str(run),
        "--output-dir",
        str(output),
        "--gpu-device",
        f"GPU {gpu}",
        "--cuda-device",
        str(gpu),
        "--resume",
    ]
    environment = os.environ.copy()
    environment.update(
        TAIRTE4_TOPOLOGY_GEOMETRY=case.geometry_mode,
        TAIRTE4_SIO2_INTERFACE_SCENARIO=case.interface_scenario,
        LUMERICAL_GPU_ENGINE_LOCK=f"/tmp/seunghyun_exact_binary_beam_response_gpu{gpu}.lock",
        ANSYSLI_PORT=str(ansysli_local_port),
    )
    return args, environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--runs", type=int, nargs="+", default=sorted(CASES))
    parser.add_argument("--max-run-attempts", type=int, default=3)
    parser.add_argument("--ansysli-local-port", type=int, default=45127)
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU indices must be unique")
    unknown = sorted(set(args.runs) - set(CASES))
    if unknown:
        parser.error(f"unknown runs: {unknown}")
    if args.max_run_attempts < 1:
        parser.error("--max-run-attempts must be at least 1")
    if not 1024 <= args.ansysli_local_port <= 65535:
        parser.error("--ansysli-local-port must be between 1024 and 65535")

    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    pending = [run for run in args.runs if not completed(root / f"run{run:03d}")]
    active: dict[int, tuple[int, subprocess.Popen[bytes], object]] = {}
    failures: list[int] = []
    attempts = {run: 0 for run in args.runs}
    broker = None
    broker_metadata = None
    if pending:
        broker, broker_metadata = start_ansysli_broker(args.ansysli_local_port)
        atexit.register(stop_ansysli_broker, broker)
        print(json.dumps({
            "event": "ansysli_broker_started",
            **broker_metadata,
        }), flush=True)

    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            run = pending.pop(0)
            attempts[run] += 1
            output = root / f"run{run:03d}"
            output.mkdir(parents=True, exist_ok=True)
            invocation, environment = command(
                run, gpu, output, args.ansysli_local_port
            )
            log_path = output / "launcher.log"
            log = log_path.open("ab")
            process = subprocess.Popen(
                invocation,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = (run, process, log)
            print(json.dumps({
                "event": "started",
                "run": run,
                "attempt": attempts[run],
                "gpu": gpu,
                "pid": process.pid,
            }), flush=True)

        time.sleep(5.0)
        for gpu, (run, process, log) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            log.close()
            del active[gpu]
            succeeded = status == 0 and completed(root / f"run{run:03d}")
            if not succeeded:
                if attempts[run] < args.max_run_attempts:
                    pending.append(run)
                else:
                    failures.append(run)
            print(json.dumps({
                "event": "finished",
                "run": run,
                "attempt": attempts[run],
                "gpu": gpu,
                "exit_code": status,
                "requeued": not succeeded and attempts[run] < args.max_run_attempts,
            }), flush=True)

    summary = {
        "status": "COMPLETED" if not failures else "FAILED",
        "runs": args.runs,
        "failures": failures,
        "attempts": attempts,
        "output_root": str(root),
        "ansysli_local_port": args.ansysli_local_port,
        "ansysli_broker": broker_metadata,
    }
    (root / "launcher_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if broker is not None:
        stop_ansysli_broker(broker)
        atexit.unregister(stop_ansysli_broker)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
