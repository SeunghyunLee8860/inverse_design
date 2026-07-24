#!/usr/bin/env python3
"""Export the complete stored pabs_adv scripts without rerunning Maxwell."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import config_stage1 as config
from lumerical_api import select_installation, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lumerical-version", default="v261")
    args = parser.parse_args()
    project = Path(args.project).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    installation = select_installation(args.lumerical_version)
    os.environ["VC_LUMERICAL_ROOT"] = str(installation.root)
    os.environ["LUMERICAL_ROOT"] = str(installation.root)
    for path in (config.REPOSITORY_ROOT, config.REPOSITORY_ROOT / "bundle"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import eqc_lib as runtime

    fdtd = runtime.open_control(project)
    try:
        group = "stage1_pabs_adv"
        analysis = str(fdtd.getnamed(group, "analysis script"))
        try:
            setup = str(fdtd.getnamed(group, "setup script"))
        except Exception:
            setup = ""
        analysis_path = output / "pabs_adv_analysis_script.lsf"
        setup_path = output / "pabs_adv_setup_script.lsf"
        analysis_path.write_text(analysis)
        setup_path.write_text(setup)
        child_names = [
            "field",
            "index",
            "Pabs",
            "Pabs_total",
        ]
        result = {
            "source_project": str(project),
            "group": group,
            "analysis_script_path": str(analysis_path),
            "analysis_script_characters": len(analysis),
            "setup_script_path": str(setup_path),
            "setup_script_characters": len(setup),
            "periodic_comment_present": (
                "Periodic boundary condition correction" in analysis
            ),
            "periodic_correction_enabled": (
                "Periodic boundary condition correction" in analysis
                and "if (1)" in analysis[
                    analysis.find("Periodic boundary condition correction"):
                ]
            ),
            "known_children": {
                name: int(fdtd.getnamednumber(f"{group}::{name}"))
                for name in child_names
            },
            "maxwell_rerun": False,
            "heat_run": False,
        }
        write_json(output / "pabs_adv_script_export.json", result)
        print(json.dumps(result, indent=2))
    finally:
        fdtd.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
