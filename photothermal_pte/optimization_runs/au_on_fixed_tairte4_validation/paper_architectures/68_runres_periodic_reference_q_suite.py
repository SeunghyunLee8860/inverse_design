#!/usr/bin/env python3
"""Run six-polarization periodic volumetric-Q gates at reference wavelengths."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts/periodic_T_Z_six_polarization_20260822/selected_Q")
POLS = (
    "x_b", "y_a", "linear_plus_45", "linear_minus_45", "CP_plus", "CP_minus"
)


def gpu_device() -> str:
    value = os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", "").strip()
    if not value.startswith("GPU "):
        raise RuntimeError("driver must be launched through runres")
    return value


def archive_incomplete(directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        directory.rename(directory.with_name(directory.name + f"_incomplete_{stamp}"))


def main() -> int:
    architecture = os.environ.get("PERIODIC_ARCHITECTURE", "").strip().upper()
    if architecture not in ("T", "Z"):
        raise RuntimeError("set PERIODIC_ARCHITECTURE=T or Z")
    root = RAW_ROOT / architecture
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "RUNRES_Q_SUITE_STATUS.json"
    if architecture == "T":
        runner = HERE / "07_run_v261_t2024_tairte4_optical_smoke.py"
        wavelength_um = 4.75
        json_name = "T2024_TaIrTe4_optical_smoke.json"
        expected = "COMPLETED_T2024_TAIRTE4_OPTICAL_SMOKE"
    else:
        runner = HERE / "41_run_v261_z2022_m2_selected_q.py"
        wavelength_um = 5.30
        json_name = "Z2022_M2_selected_Q.json"
        expected = "COMPLETED_Z2022_M2_CENTERED_EXPANDED_SELECTED_Q"
    records: list[dict[str, object]] = []

    def publish(status: str, command: list[str] | None = None) -> None:
        status_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "architecture": architecture,
                    "reference_wavelength_um": wavelength_um,
                    "gpu_device": gpu_device(),
                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "scope": "periodic selected-wavelength volumetric Q only; no thermal/weighting/PTE",
                    "current_command": command,
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )

    publish("STARTING_PERIODIC_REFERENCE_Q_SUITE")
    for polarization in POLS:
        output = root / polarization
        metadata = output / json_name
        if metadata.is_file() and json.loads(metadata.read_text()).get("status") == expected:
            records.append({"polarization": polarization, "status": "reused", "output": str(output)})
            continue
        archive_incomplete(output)
        command = [
            sys.executable, str(runner), "--polarization", polarization,
            "--wavelength-um", f"{wavelength_um:.6f}",
            "--output-dir", str(output), "--gpu-device", gpu_device(),
        ]
        if architecture == "T":
            command.extend(["--substrate-mode", "sio2_si_reduced_285nm", "--duration-ps", "1.0"])
        else:
            command.extend(
                [
                    "--handedness", "LH",
                    "--geometry-variant", "centered_expanded_supercell_v4",
                    "--duration-ps", "1.5",
                ]
            )
        publish("RUNNING_PERIODIC_REFERENCE_Q_SUITE", command)
        started = time.monotonic()
        result = subprocess.run(command, check=False)
        observed = None
        if metadata.is_file():
            observed = json.loads(metadata.read_text()).get("status")
        records.append(
            {
                "polarization": polarization,
                "output": str(output),
                "returncode": int(result.returncode),
                "observed_status": observed,
                "wall_time_s": time.monotonic() - started,
            }
        )
        if result.returncode != 0 or observed != expected:
            publish("FAILED_PERIODIC_REFERENCE_Q_SUITE", command)
            raise RuntimeError(
                f"fail-closed Q gate {architecture}/{polarization}: {observed}"
            )
    publish("COMPLETED_PERIODIC_REFERENCE_Q_SUITE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
