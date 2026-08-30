#!/usr/bin/env python3
"""Schedule eight exact-binary 25-position spatial-field scans."""

from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from photothermal_pte.optimization_runs.launch_exact_binary_beam_response import (
    start_ansysli_broker,
    stop_ansysli_broker,
)
from photothermal_pte.optimization_runs.run_exact_binary_beam_position_fields import (
    RESULT_SCHEMA,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.beam_response_contract import (
    CASES,
)


def completed(path: Path) -> bool:
    result = path / "position_fields_result.json"
    if not result.is_file():
        return False
    try:
        payload = json.loads(result.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("schema") == RESULT_SCHEMA
        and payload.get("passed")
        and payload.get("status") == "COMPLETED"
        and len(payload.get("responses", [])) == 25
    )


def command(
    run: int,
    gpu: int,
    output: Path,
    scalar_reference_root: Path,
    ansysli_local_port: int,
) -> tuple[list[str], dict[str, str]]:
    case = CASES[run]
    invocation = [
        sys.executable,
        "-m",
        "photothermal_pte.optimization_runs.run_exact_binary_beam_position_fields",
        "--run",
        str(run),
        "--output-dir",
        str(output),
        "--scalar-reference-root",
        str(scalar_reference_root),
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
        LUMERICAL_GPU_ENGINE_LOCK=(
            f"/tmp/seunghyun_exact_binary_position_fields_gpu{gpu}.lock"
        ),
        ANSYSLI_PORT=str(ansysli_local_port),
    )
    return invocation, environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scalar-reference-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--runs", type=int, nargs="+", default=sorted(CASES))
    parser.add_argument("--max-run-attempts", type=int, default=8)
    parser.add_argument("--ansysli-local-port", type=int, default=45137)
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU indices must be unique")
    unknown = sorted(set(args.runs) - set(CASES))
    if unknown:
        parser.error(f"unknown runs: {unknown}")

    root = args.output_root.expanduser().resolve()
    reference_root = args.scalar_reference_root.expanduser().resolve()
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
        print(json.dumps({"event": "ansysli_broker_started", **broker_metadata}), flush=True)

    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            run = pending.pop(0)
            attempts[run] += 1
            output = root / f"run{run:03d}"
            output.mkdir(parents=True, exist_ok=True)
            invocation, environment = command(
                run, gpu, output, reference_root, args.ansysli_local_port
            )
            log = (output / "launcher.log").open("ab")
            process = subprocess.Popen(
                invocation, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            active[gpu] = (run, process, log)
            print(json.dumps({
                "event": "started", "run": run, "attempt": attempts[run],
                "gpu": gpu, "pid": process.pid,
            }), flush=True)

        time.sleep(5.0)
        for gpu, (run, process, log) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            log.close()
            del active[gpu]
            succeeded = status == 0 and completed(root / f"run{run:03d}")
            requeued = not succeeded and attempts[run] < args.max_run_attempts
            if requeued:
                pending.append(run)
            elif not succeeded:
                failures.append(run)
            print(json.dumps({
                "event": "finished", "run": run, "attempt": attempts[run],
                "gpu": gpu, "exit_code": status, "requeued": requeued,
            }), flush=True)

    summary = {
        "status": "COMPLETED" if not failures else "FAILED",
        "runs": args.runs,
        "failures": failures,
        "attempts": attempts,
        "output_root": str(root),
        "scalar_reference_root": str(reference_root),
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
