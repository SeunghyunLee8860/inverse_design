#!/usr/bin/env python3
"""Fail-closed sequential supervisor for corrected Ea then Eb optimizations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


REPO = Path(__file__).resolve().parents[2]
PYTHON = "/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python"
RUN002 = REPO / "photothermal_pte/optimization_runs/run_002_gaussian10_w8p5_current_max"
ROOT = Path("/data/seunghyun/tairte4/raw_artifacts/corrected_dual_polarization_supervisor_20260809")
EVENTS = ROOT / "events.jsonl"
EA_ADFD = Path("/data/seunghyun/tairte4/raw_artifacts/run006_corrected_Ea_magnitude_adfd_h005_20260809")
EB_ADFD = Path("/data/seunghyun/tairte4/raw_artifacts/run007_corrected_Eb_magnitude_adfd_h005_20260809")
BASE = Path("/home/seunghyun/tairte4/raw_artifacts/run002_selected_production_geometry_runsetup_v2_20260806/production_candidate_runsetup.fsp")
BASE_SHA = "a86644647b8bf03ec1b83c34d0cf18b6b1c5316342d845ccbdccf60df3d8f904"


def emit(event: str, **payload) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    row = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
    with EVENTS.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def wait_for_passed(path: Path, timeout_s: float = 6 * 3600) -> dict:
    started = time.monotonic()
    while not path.exists():
        if time.monotonic() - started > timeout_s:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(30)
    result = json.loads(path.read_text())
    if not result.get("passed"):
        raise RuntimeError(f"gate failed: {path}: {result.get('error')}")
    if float(result["relative_error"]) >= 0.01:
        raise RuntimeError(f"AD-FD error gate failed: {result['relative_error']}")
    return result


def run(command: list[str], label: str) -> None:
    emit("command_start", label=label, command=command)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    completed = subprocess.run(command, cwd=REPO, env=env)
    emit("command_end", label=label, returncode=completed.returncode)
    if completed.returncode:
        raise RuntimeError(f"{label} failed with {completed.returncode}")


def main() -> int:
    emit("supervisor_started", gpu=0, sequence=["Ea_ADFD", "Eb_ADFD", "Run006_Ea", "Run007_Eb"])
    ea = wait_for_passed(EA_ADFD / "selected_full_latent_direction_adfd_result.json")
    emit("Ea_ADFD_passed", relative_error=ea["relative_error"])
    eb_result = EB_ADFD / "selected_full_latent_direction_adfd_result.json"
    if EB_ADFD.exists() and any(EB_ADFD.iterdir()) and not eb_result.exists():
        emit("Eb_ADFD_external_attempt_detected", directory=str(EB_ADFD))
    elif not eb_result.exists():
        run([
            PYTHON, str(RUN002 / "validate_selected_full_latent_direction.py"),
            "--preparation-result", "/data/seunghyun/tairte4/raw_artifacts/run007_corrected_Eb_magnitude_initial_20260809/selected_full_latent_adjoint_preparation_result.json",
            "--preparation-raw", "/data/seunghyun/tairte4/raw_artifacts/run007_corrected_Eb_magnitude_initial_20260809/selected_full_latent_adjoint_preparation.npz",
            "--preparation-raw-sha256", "868f7aacb735e028f53f3c0df30ca498f05b6a12cc4cfdaa5e0fd9237a4d4dc4",
            "--base-fsp", str(BASE), "--base-fsp-sha256", BASE_SHA,
            "--direction", "adjoint_aligned", "--step", "0.005",
            "--gpu-device", "GPU 0", "--cuda-device", "0",
            "--output-dir", str(EB_ADFD),
        ], "Eb_combined_ADFD")
    eb = wait_for_passed(eb_result)
    emit("Eb_ADFD_passed", relative_error=eb["relative_error"])
    run([PYTHON, "photothermal_pte/optimization_runs/run_006_corrected_Ea_pte_magnitude_max/run_optimization.py", "--gpu", "0", "--constraint-device", "cuda:0"], "Run006_Ea_full_binary")
    run([PYTHON, "photothermal_pte/optimization_runs/run_007_corrected_Eb_pte_magnitude_max/run_optimization.py", "--gpu", "0", "--constraint-device", "cuda:0"], "Run007_Eb_full_binary")
    emit("supervisor_completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit("supervisor_failed", error=f"{type(exc).__name__}: {exc}")
        raise
