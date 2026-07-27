#!/usr/bin/env python3
"""Extract native Yee E/index/Q arrays from a completed, SHA-pinned FSP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .native_yee_q import extract_native_yee_q
from .probe_v261_cpu_tfsf_device import PABS_FIELD, PABS_INDEX
from .probe_v261_gpu_plane_wave_roi import load_lumapi
from .run_v261_large_background_tfsf_forward import (
    save_native_npz,
    sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(args.project).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not project.is_file():
        raise FileNotFoundError(project)
    actual_sha = sha256(project)
    if actual_sha != args.expected_sha256:
        raise RuntimeError("completed FSP SHA-256 mismatch")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fdtd = None
    try:
        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        fdtd.load(str(project))
        native = extract_native_yee_q(
            fdtd,
            field_monitor=PABS_FIELD,
            index_monitor=PABS_INDEX,
            wavelength_m=4.0e-6,
        )
        save_native_npz(output, native, fdtd)
        result = {
            "status": "EXTRACTED_COMPLETED_NATIVE_YEE_FIELDS",
            "electromagnetic_solve_run": False,
            "source_project": {
                "path": str(project),
                "byte_size": project.stat().st_size,
                "sha256": actual_sha,
            },
            "output_npz": {
                "path": str(output),
                "byte_size": output.stat().st_size,
                "sha256": sha256(output),
            },
            "P_Q_W": native["P_Q_W"],
        }
        print(json.dumps(result, indent=2))
        return 0
    finally:
        if fdtd is not None:
            fdtd.close()


if __name__ == "__main__":
    raise SystemExit(main())
