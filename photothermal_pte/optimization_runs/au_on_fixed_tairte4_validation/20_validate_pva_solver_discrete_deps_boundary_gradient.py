#!/usr/bin/env python3
"""PVA solver-discrete d-epsilon control for the smooth Au ellipse.

The v261 mesh is rebuilt for plus/minus x-semi-axis perturbations without a
Maxwell solve.  The resulting component-specific permittivity derivative is
contracted with the already-computed collocated forward and adjoint fields.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
STAGE15 = HERE / "15_validate_solver_discrete_deps_boundary_gradient.py"
STAGE16 = HERE / "16_run_au_smooth_ellipse_width_control.py"
BASELINE_CASE = "pva5_fixedgrid_smooth_ellipse_a8p0_b18_edge50_forward"
AU_OBJECT = "rho1_scalar_complex_block"
ELLIPSE_HALF_Y_M = 18.0e-6
ELLIPSE_VERTICES = 512


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    stage15 = load("au_pva_stage15", STAGE15)
    stage16 = load("au_pva_stage16", STAGE16)

    def remeshed_ellipse_epsilon(
        fdtd: object,
        *,
        project: Path,
        half_width_m: float,
        field_monitor: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        fdtd.load(str(project))
        fdtd.switchtolayout()
        if int(fdtd.getnamednumber(AU_OBJECT)) != 1:
            raise RuntimeError(f"expected exactly one {AU_OBJECT!r}")
        vertices = stage16.ellipse_vertices(
            float(half_width_m), ELLIPSE_HALF_Y_M, ELLIPSE_VERTICES
        )
        fdtd.setnamed(AU_OBJECT, "vertices", vertices)
        stage15.add_matching_index_monitor(fdtd, field_monitor)
        fdtd.runsetup()
        dataset = fdtd.getresult(stage15.INDEX_MONITOR, "index")
        coordinates = {
            axis: np.asarray(dataset[axis], float).reshape(-1) for axis in "xyz"
        }
        shape = tuple(coordinates[axis].size for axis in "xyz")
        epsilon: dict[str, np.ndarray] = {}
        for component in "xyz":
            index = np.asarray(dataset[f"index_{component}"], np.complex128)
            if index.shape != (*shape, 1):
                raise RuntimeError(
                    f"index_{component} shape {index.shape} != {(*shape, 1)}"
                )
            epsilon[component] = np.asarray(index[..., 0] ** 2, np.complex128)
        return epsilon, coordinates

    stage15.BASELINE_CASE = BASELINE_CASE
    stage15.AU_OBJECT = AU_OBJECT
    stage15.BASELINE_HALF_WIDTH_M = 8.0e-6
    stage15.remeshed_epsilon = remeshed_ellipse_epsilon
    return_code = int(stage15.main())

    output_value = None
    for index, argument in enumerate(sys.argv[1:]):
        if argument == "--output-dir" and index + 2 <= len(sys.argv[1:]):
            output_value = sys.argv[1:][index + 1]
            break
        if argument.startswith("--output-dir="):
            output_value = argument.split("=", 1)[1]
            break
    if output_value is not None:
        result_path = Path(output_value).expanduser().resolve() / (
            "au_solver_discrete_deps_result.json"
        )
        result = json.loads(result_path.read_text())
        result["geometry_control"] = {
            "representation": "smooth_closed_binary_scalar_Au_ellipse",
            "ellipse_x_semi_axis_m": 8.0e-6,
            "ellipse_y_semi_axis_m": ELLIPSE_HALF_Y_M,
            "ellipse_vertex_count": ELLIPSE_VERTICES,
            "shape_parameter": "x semi-axis",
        }
        result["mesh_control"] = {
            "mesh_refinement": "precise volume average",
            "meshing_refinement": 5,
            "local_Au_dx_dy_m": 50.0e-9,
            "local_Au_dz_m": 5.0e-9,
            "fixed_mesh_and_monitor_bounds_m": {
                "x": [-8.6e-6, 8.6e-6],
                "y": [-18.5e-6, 18.5e-6],
            },
        }
        result["production_Au_optimization_permitted"] = False
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
