#!/usr/bin/env python3
"""Sharp-interface scalar-Au width control at 10 um.

The test moves only the two x-normal faces of a finite binary Au film.  It
does not create a gray Au/air material and therefore exercises the approved
fallback geometry representation independently of any adjoint formula.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
BINARY_CONTROL = HERE / "04_run_au_binary_representation_control.py"


def load_binary_control():
    spec = importlib.util.spec_from_file_location("au_binary_width_base", BINARY_CONTROL)
    if spec is None or spec.loader is None:
        raise ImportError(BINARY_CONTROL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option_present(arguments: list[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in arguments)


def option_value(arguments: list[str], option: str) -> str | None:
    for index, value in enumerate(arguments):
        if value.startswith(f"{option}="):
            return value.split("=", 1)[1]
        if value == option and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--au-half-x-um", type=float, required=True)
    parsed, remaining = parser.parse_known_args()
    if not 4.0 <= parsed.au_half_x_um <= 10.4:
        raise ValueError("--au-half-x-um must remain within [4.0, 10.4] um")

    base = load_binary_control()
    half_x_m = float(parsed.au_half_x_um) * 1e-6
    base.AU_BOUNDS["x"] = (-half_x_m, half_x_m)

    if not option_present(remaining, "--rho"):
        remaining.extend(("--rho", "1"))
    if not option_present(remaining, "--representation"):
        remaining.extend(("--representation", "scalar"))
    if option_value(remaining, "--rho") != "1":
        raise ValueError("sharp-interface width control requires rho=1")
    if option_value(remaining, "--representation") != "scalar":
        raise ValueError("sharp-interface width control requires scalar Au")

    output_value = option_value(remaining, "--output-dir")
    if output_value is None:
        raise ValueError("--output-dir is required")
    result_path = Path(output_value).expanduser().resolve() / "case_result.json"

    sys.argv = [sys.argv[0], *remaining]
    return_code = int(base.main())
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        result["geometry_representation"] = "sharp_interface_binary_scalar_Au"
        result["shape_parameter"] = {
            "name": "Au_half_x",
            "value_m": half_x_m,
            "value_um": float(parsed.au_half_x_um),
            "moved_boundaries": ["x_min", "x_max"],
            "fixed_boundaries": ["y_min", "y_max", "z_min", "z_max"],
        }
        result["gray_Au_air_material_used"] = False
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
