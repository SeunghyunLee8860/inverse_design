#!/usr/bin/env python3
"""Probe the actual DEVICE/HEAT object API before setting any properties."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys

import numpy as np
from pathlib import Path

from lumerical_api import (
    add_and_probe,
    environment_record,
    open_device,
    select_installation,
    write_json,
)


OBJECT_COMMANDS = (
    ("simulation_region", "addsimulationregion", ()),
    ("heat_solver", "addheatsolver", ()),
    ("uniform_heat", "adduniformheat", ()),
    ("import_heat", "addimportheat", ("HEAT",)),
    ("temperature_monitor", "addtemperaturemonitor", ("HEAT",)),
    ("heat_flux_monitor", "addheatfluxmonitor", ("HEAT",)),
    ("temperature_bc", "addtemperaturebc", ("HEAT",)),
    ("heat_mesh", "addheatmesh", ()),
)


def probe_thermal_material(installation, hide: bool):
    """Probe scalar and diagonal conductivity round trips without assuming support."""
    with open_device(installation, hide=hide) as device:
        device.addmodelmaterial()
        device.set("name", "stage1 thermal capability probe")
        device.addhtmaterialproperty("Solid")
        device.set("name", "stage1 thermal capability probe")
        props = str(device.get()).splitlines()
        device.set("thermal conductivity.active model", "constant")
        device.set("thermal conductivity.constant", 10.0)
        scalar_return = np.asarray(device.get("thermal conductivity.constant"), float).reshape(-1)
        diagonal_request = np.array([14.4, 3.8, 1.0])
        device.set("thermal conductivity.constant", diagonal_request)
        diagonal_return = np.asarray(device.get("thermal conductivity.constant"), float).reshape(-1)
    return {
        "success": True,
        "properties": props,
        "active_model": "constant",
        "scalar_request_W_mK": 10.0,
        "scalar_return": scalar_return.tolist(),
        "scalar_round_trip": bool(scalar_return.size == 1 and np.isclose(scalar_return[0], 10.0)),
        "diagonal_request_W_mK": diagonal_request.tolist(),
        "diagonal_return": diagonal_return.tolist(),
        "diagonal_round_trip": bool(diagonal_return.size == 3 and np.allclose(diagonal_return, diagonal_request)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumerical-version", choices=("auto", "v261", "v251"), default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hide-gui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--material-only",
        action="store_true",
        help="Probe only scalar/diagonal thermal-conductivity round trips.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    probe_dir = output / "api_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    log_path = probe_dir / "probe.log"
    installation = select_installation(args.lumerical_version)
    environment = environment_record(installation, working_directory=Path.cwd())

    with log_path.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print(json.dumps(environment, indent=2, default=str), flush=True)
        with open_device(installation, hide=args.hide_gui) as device:
            try:
                environment["lumerical_version"] = str(device.version())
            except Exception as exc:
                environment["lumerical_version_error"] = f"{type(exc).__name__}: {exc}"
        results = {}
        for label, command, command_args in (() if args.material_only else OBJECT_COMMANDS):
            print(f"[probe] {label}: {command}", flush=True)
            with open_device(installation, hide=args.hide_gui) as device:
                if command not in ("addsimulationregion", "addheatsolver"):
                    device.addsimulationregion()
                    device.addheatsolver()
                result = add_and_probe(
                    device,
                    command,
                    prerequisite_region=(command == "addheatsolver"),
                    command_args=command_args,
                )
            result["lumerical_version"] = environment.get("lumerical_version")
            results[label] = result
            write_json(probe_dir / f"{label}.json", result)
            print(json.dumps(result, indent=2, default=str), flush=True)

    thermal_material = probe_thermal_material(installation, args.hide_gui)
    results["thermal_material"] = thermal_material
    write_json(probe_dir / "thermal_material.json", thermal_material)

    all_commands_succeeded = all(v["success"] for v in results.values())
    required_capabilities_succeeded = bool(
        all_commands_succeeded
        and thermal_material["scalar_round_trip"]
        and thermal_material["diagonal_round_trip"]
    )
    summary = {
        "environment": environment,
        "objects": results,
        "all_commands_succeeded": all_commands_succeeded,
        "required_capabilities_succeeded": required_capabilities_succeeded,
    }
    write_json(probe_dir / "probe_summary.json", summary)
    write_json(output / "environment.json", environment)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["required_capabilities_succeeded"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
