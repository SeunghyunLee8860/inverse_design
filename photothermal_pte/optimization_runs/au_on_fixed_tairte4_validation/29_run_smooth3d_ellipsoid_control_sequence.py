#!/usr/bin/env python3
"""Run the remaining smooth-3D Au FD/adjoint controls sequentially.

The sequence is intentionally serial on one requested GPU.  Every child is
GPU-only and fail-closed; a failed forward stops the sequence before the
adjoint.  Completed passing cases are reused after SHA verification by stage
26, while raw FSP/NPZ files remain outside Git.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
STAGE25 = HERE / "25_run_au_smooth_3d_ellipsoid_width_control.py"
STAGE26 = HERE / "26_validate_au_smooth_3d_ellipsoid_boundary_adjoint.py"
STAGE28 = HERE / "28_summarize_au_pva_rim_resolution.py"
DEFAULT_RAW = Path("/data/seunghyun/tairte4/raw_artifacts/au_topology_validation")
CASES = (
    (8.10, "pva5_smooth3d_ellipsoid_a8p1_b18_c1_forward"),
    (7.95, "pva5_smooth3d_ellipsoid_a7p95_b18_c1_forward"),
    (8.05, "pva5_smooth3d_ellipsoid_a8p05_b18_c1_forward"),
)


def passed_case(case_dir: Path) -> bool:
    for name in ("case_result_recovered.json", "case_result.json"):
        path = case_dir / name
        if path.is_file() and bool(json.loads(path.read_text()).get("passed", False)):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument("--state-path", type=Path)
    args = parser.parse_args()
    raw = args.raw_root.expanduser().resolve()
    raw.mkdir(parents=True, exist_ok=True)
    state_path = (
        args.state_path.expanduser().resolve()
        if args.state_path is not None
        else raw / "pva5_smooth3d_ellipsoid_sequence_state.json"
    )
    state: dict[str, object] = {
        "status": "RUNNING_AU_SMOOTH3D_ELLIPSOID_SEQUENCE",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_device": args.gpu_device,
        "CPU_FDTD_fallback": False,
        "cases": [],
    }

    try:
        for width_um, name in CASES:
            case_dir = raw / name
            if passed_case(case_dir):
                state["cases"].append({"name": name, "status": "reused_passed"})
                continue
            if case_dir.exists() and any(case_dir.iterdir()):
                raise RuntimeError(f"non-empty failed/incomplete case requires audit: {case_dir}")
            command = [
                sys.executable,
                str(STAGE25),
                "--au-half-x-um",
                str(width_um),
                "--au-half-y-um",
                "18",
                "--au-half-z-um",
                "1",
                "--ellipsoid-dxy-nm",
                "50",
                "--ellipsoid-dz-nm",
                "25",
                "--gpu-device",
                args.gpu_device,
                "--output-dir",
                str(case_dir),
            ]
            state["cases"].append({"name": name, "status": "running", "command": command})
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            subprocess.run(command, check=True)
            if not passed_case(case_dir):
                raise RuntimeError(f"forward did not produce a passing result: {case_dir}")
            state["cases"][-1]["status"] = "completed_passed"
            state_path.write_text(json.dumps(state, indent=2) + "\n")

        adjoint_dir = raw / "pva5_smooth3d_ellipsoid_boundary_adjoint_gpu0"
        command = [
            sys.executable,
            str(STAGE26),
            "--raw-root",
            str(raw),
            "--output-dir",
            str(adjoint_dir),
            "--gpu-device",
            args.gpu_device,
        ]
        state["adjoint"] = {"status": "running", "command": command}
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        completed = subprocess.run(command, check=False)
        state["adjoint"]["return_code"] = int(completed.returncode)
        result_path = adjoint_dir / "au_smooth3d_ellipsoid_boundary_adjoint_result.json"
        if not result_path.is_file():
            raise RuntimeError("stage 26 did not write its fail-closed result")
        state["adjoint"]["result"] = json.loads(result_path.read_text())
        # Stage 28 publishes both a pass and a fail-closed physical conclusion.
        subprocess.run([sys.executable, str(STAGE28)], check=False)
        state["status"] = "COMPLETED_AU_SMOOTH3D_ELLIPSOID_SEQUENCE"
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(state, indent=2, default=str) + "\n")
        return 0
    except Exception as exc:
        state["status"] = "BLOCKED_AU_SMOOTH3D_ELLIPSOID_SEQUENCE"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(state, indent=2, default=str) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
