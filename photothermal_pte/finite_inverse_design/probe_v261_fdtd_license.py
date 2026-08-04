#!/usr/bin/env python3
"""Fail-closed v261 FDTD startup and project save/load license probe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

from .probe_v261_gpu_plane_wave_roi import (
    APPROVED_API,
    APPROVED_ROOT,
    load_lumapi,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    project = output / "v261_fdtd_license_probe.fsp"
    summary = output / "v261_fdtd_license_probe.json"
    result: dict[str, object] = {
        "status": "BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE",
        "passed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "solver_root": str(APPROVED_ROOT),
        "lumapi_path": str(APPROVED_API),
        "session_startup": False,
        "script_round_trip": False,
        "project_save": False,
        "project_reload": False,
        "solver_run": False,
        "optimization_run": False,
    }
    fdtd = None
    started = time.monotonic()
    try:
        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        result["session_startup"] = True
        fdtd.eval("codex_probe_value=261;")
        result["script_round_trip"] = int(
            fdtd.getv("codex_probe_value")
        ) == 261
        try:
            fdtd.eval("codex_probe_version=version;")
            result["lumerical_version"] = str(
                fdtd.getv("codex_probe_version")
            )
        except Exception as exc:
            result["version_readback_warning"] = (
                f"{type(exc).__name__}: {exc}"
            )
        fdtd.switchtolayout()
        fdtd.eval("selectall; delete;")
        region = fdtd.addfdtd()
        region["dimension"] = "3D"
        region["x span"] = 1.0e-6
        region["y span"] = 1.0e-6
        region["z span"] = 1.0e-6
        fdtd.save(str(project))
        result["project_save"] = project.is_file()
        fdtd.load(str(project))
        result["project_reload"] = True
        passed = bool(
            result["session_startup"]
            and result["script_round_trip"]
            and result["project_save"]
            and result["project_reload"]
        )
        result["passed"] = passed
        result["status"] = (
            "PASSED_V261_FDTD_LICENSE_API_PROBE"
            if passed
            else "BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE"
        )
    except Exception as exc:
        if result["session_startup"]:
            result["status"] = "FAILED_V261_FDTD_API_PROBE"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if fdtd is not None:
            fdtd.close()
        result["wall_s"] = time.monotonic() - started
        summary.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
