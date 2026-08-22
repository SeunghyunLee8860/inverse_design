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
DEFAULT_RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts/periodic_T_Z_six_polarization_20260822/selected_Q")
DEFAULT_POLS = (
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
    raw_root = Path(os.environ.get("PERIODIC_Q_RAW_ROOT", str(DEFAULT_RAW_ROOT)))
    root = raw_root / architecture
    root.mkdir(parents=True, exist_ok=True)
    requested_pols = tuple(
        item.strip()
        for item in os.environ.get("PERIODIC_Q_POLARIZATIONS", ",".join(DEFAULT_POLS)).split(",")
        if item.strip()
    )
    invalid_pols = sorted(set(requested_pols) - set(DEFAULT_POLS))
    if not requested_pols or invalid_pols:
        raise RuntimeError(f"invalid PERIODIC_Q_POLARIZATIONS: {invalid_pols}")
    z_mesh_refinement = os.environ.get(
        "PERIODIC_Z_MESH_REFINEMENT", "conformal variant 1"
    ).strip()
    if z_mesh_refinement not in (
        "conformal variant 0", "conformal variant 1", "staircase"
    ):
        raise RuntimeError(f"invalid PERIODIC_Z_MESH_REFINEMENT: {z_mesh_refinement}")
    z_edge_mesh_nm = os.environ.get("PERIODIC_Z_EDGE_MESH_NM", "").strip()
    z_omit_top_au = os.environ.get("PERIODIC_Z_OMIT_TOP_AU", "0").strip() == "1"
    z_duration_ps = float(os.environ.get("PERIODIC_Z_DURATION_PS", "4.0"))
    if not 1.0 <= z_duration_ps <= 20.0:
        raise RuntimeError("PERIODIC_Z_DURATION_PS must be in [1, 20] ps")
    status_name = os.environ.get(
        "PERIODIC_Q_STATUS_NAME", "RUNRES_Q_SUITE_STATUS.json"
    ).strip()
    if not status_name.endswith(".json") or Path(status_name).name != status_name:
        raise RuntimeError("PERIODIC_Q_STATUS_NAME must be a JSON filename")
    status_path = root / status_name
    if architecture == "T":
        runner = HERE / "07_run_v261_t2024_tairte4_optical_smoke.py"
        wavelength_um = 4.75
        json_name = "T2024_TaIrTe4_optical_smoke.json"
        expected = "COMPLETED_T2024_TAIRTE4_OPTICAL_SMOKE"
    else:
        runner = HERE / "41_run_v261_z2022_m2_selected_q.py"
        wavelength_um = float(
            os.environ.get("PERIODIC_Z_REFERENCE_WAVELENGTH_UM", "5.30")
        )
        if not 4.0 <= wavelength_um <= 12.0:
            raise RuntimeError(
                "PERIODIC_Z_REFERENCE_WAVELENGTH_UM must be within 4--12 um"
            )
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
                    "requested_polarizations": requested_pols,
                    "z_mesh_refinement": z_mesh_refinement if architecture == "Z" else None,
                    "z_edge_mesh_nm": float(z_edge_mesh_nm) if z_edge_mesh_nm else None,
                    "z_omit_top_au": z_omit_top_au if architecture == "Z" else None,
                    "z_duration_ps": z_duration_ps if architecture == "Z" else None,
                    "current_command": command,
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )

    publish("STARTING_PERIODIC_REFERENCE_Q_SUITE")
    for polarization in requested_pols:
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
            command.extend(["--substrate-mode", "sio2_si_reduced_285nm", "--duration-ps", "1.2"])
        else:
            command.extend(
                [
                    "--handedness", "LH",
                    "--geometry-variant", "centered_expanded_supercell_v4",
                    # Reuse the broadband convergence bound: several Z
                    # polarizations did not reach 1e-5 by 2 ps.
                    "--duration-ps", f"{z_duration_ps:g}",
                    "--mesh-refinement", z_mesh_refinement,
                ]
            )
            if z_edge_mesh_nm:
                command.extend(["--top-au-edge-mesh-nm", z_edge_mesh_nm])
            if z_omit_top_au:
                command.append("--omit-top-au-control")
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
