#!/usr/bin/env python3
"""PVA smooth-Au boundary AD--FD control on the 50 nm local mesh.

This is a configuration wrapper around stage 17.  It deliberately keeps the
exact scalar Au endpoint and changes only the conformal-mesh refinement from
CV1 to Precise Volume Average (PVA).  The finite-difference projects are
independent Maxwell solves; the baseline adjoint is a native-Yee FieldRegion
solve.  No thermal, electrical, PTE, or optimization calculation is run.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
STAGE17 = HERE / "17_run_au_smooth_ellipse_external_field_adjoint.py"
ELLIPSE_HALF_Y_M = 18.0e-6
BASELINE_CASE = "pva5_smooth_ellipse_a8p0_b18_edge50_forward_retry"
PVA_FD_CASES = {
    0.10: (
        "pva5_smooth_ellipse_a7p9_b18_edge50_forward",
        "pva5_smooth_ellipse_a8p1_b18_edge50_forward",
    ),
    0.05: (
        "pva5_smooth_ellipse_a7p95_b18_edge50_forward",
        "pva5_smooth_ellipse_a8p05_b18_edge50_forward",
    ),
}


def load_stage17():
    spec = importlib.util.spec_from_file_location("au_pva_smooth_stage17", STAGE17)
    if spec is None or spec.loader is None:
        raise ImportError(STAGE17)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    stage17 = load_stage17()
    arguments = list(sys.argv[1:])
    output_value = stage17.option_value(arguments, "--output-dir")
    if output_value is None:
        raise ValueError("--output-dir is required")
    if not stage17.option_present(arguments, "--baseline-case"):
        arguments.extend(("--baseline-case", BASELINE_CASE))

    stage17.ELLIPSE_HALF_Y_M = ELLIPSE_HALF_Y_M
    stage17.BASELINE_CASE = BASELINE_CASE
    stage17.SMOOTH_FD_CASES = PVA_FD_CASES
    sys.argv = [sys.argv[0], *arguments]
    return_code = int(stage17.main())

    result_path = Path(output_value).expanduser().resolve() / (
        "au_sharp_interface_external_field_result.json"
    )
    result = json.loads(result_path.read_text())
    result["mesh_control"] = {
        "mesh_refinement": "precise volume average",
        "meshing_refinement": 5,
        "local_Au_dx_dy_m": 50.0e-9,
        "local_Au_dz_m": 5.0e-9,
        "mesh_wavelength_m": 10.0e-6,
    }
    result["comparison_to_CV1"] = (
        "same scalar Au, smooth ellipse, source, objective, and shape parameter; "
        "PVA replaces conformal variant 1 and the local lateral mesh is 50 nm"
    )
    result["production_Au_optimization_permitted"] = False
    result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
