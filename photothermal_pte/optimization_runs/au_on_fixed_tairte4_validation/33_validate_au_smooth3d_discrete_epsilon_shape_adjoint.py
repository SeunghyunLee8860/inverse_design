#!/usr/bin/env python3
"""Validate a solver-discrete epsilon shape Jacobian on the smooth Au control.

Completed central-FD geometry projects provide the component-wise conformal
Yee permittivity derivative, while the existing baseline forward and adjoint
fields provide the standard volume contraction.  No Maxwell solve is run.
This is a one-parameter diagnostic: constructing this Jacobian by one solve per
topology variable is explicitly forbidden as a production optimization path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
STAGE12 = HERE / "12_run_au_sharp_interface_external_field_adjoint.py"
STAGE21 = HERE / "21_validate_fixed_geometry_au_material_adjoint.py"
STAGE26 = HERE / "26_validate_au_smooth_3d_ellipsoid_boundary_adjoint.py"
DEFAULT_RAW = Path("/data/seunghyun/tairte4/raw_artifacts/au_topology_validation")
DEFAULT_ADJOINT = DEFAULT_RAW / "pva5_smooth3d_ellipsoid_boundary_adjoint_gpu0"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage12 = load("au_discrete_shape_stage12", STAGE12)
stage21 = load("au_discrete_shape_stage21", STAGE21)
stage26 = load("au_discrete_shape_stage26", STAGE26)
from photothermal_pte.finite_inverse_design.native_yee_q import frequency_slice

PABS_INDEX = f"{stage12.PABS_GROUP}::index"


def relative(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(
        abs(float(left)), abs(float(right)), np.finfo(float).tiny
    )


def read_component_epsilon(fdtd, component: str, grid: dict) -> np.ndarray:
    spatial_shape = tuple(np.asarray(grid[axis]).size for axis in "xyz")
    index_grid = {
        axis: np.asarray(fdtd.getdata(PABS_INDEX, axis, 1), float).reshape(-1)
        for axis in "xyzf"
    }
    index_grid.update(
        {
            f"delta_{axis}": np.asarray(
                fdtd.getdata(PABS_INDEX, f"delta_{axis}", 1), float
            ).reshape(-1)
            for axis in "xyz"
        }
    )
    mismatch = stage21.maximum_grid_mismatch(grid, index_grid)
    if mismatch >= 2.0e-18:
        raise RuntimeError(f"field/index grid mismatch {mismatch:.3e} m")
    frequency_index = int(
        np.argmin(np.abs(np.asarray(index_grid["f"]) - stage12.FREQUENCY_HZ))
    )
    index = frequency_slice(
        np.asarray(fdtd.getdata(PABS_INDEX, f"index_{component}", 1)),
        spatial_shape,
        frequency_index,
        np.asarray(index_grid["f"]).size,
        f"{PABS_INDEX}.index_{component}",
    )
    return np.asarray(index, complex) ** 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--adjoint-dir", type=Path, default=DEFAULT_ADJOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 0")
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "au_smooth3d_discrete_epsilon_shape_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_AU_SMOOTH3D_DISCRETE_EPSILON_SHAPE_ADJOINT",
        "passed": False,
        "Maxwell_forward_solves_this_invocation": 0,
        "Maxwell_adjoint_solves_this_invocation": 0,
        "thermal_solves": 0,
        "optimizer_started": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "finite_difference_used_to_fit_AD": False,
        "production_Au_optimization_permitted": False,
    }
    fdtd = None
    try:
        case_names = {stage26.BASELINE_CASE}
        for pair in stage26.FD_CASES.values():
            case_names.update(pair)
        projects = {}
        case_results = {}
        for name in sorted(case_names):
            projects[name], case_results[name] = stage26.checked_ellipsoid_project(
                args.raw_root / name
            )

        source_result = json.loads(
            (
                args.adjoint_dir
                / "au_smooth3d_ellipsoid_boundary_adjoint_result.json"
            ).read_text()
        )
        adjoint_project = Path(source_result["adjoint"]["project"]["path"])
        profile_scale = float(source_result["source"]["profile_scale"])
        base_amplitude = float(source_result["source"]["fieldregion_base_amplitude"])

        fdtd, _audit, _runtime = stage12.open_fdtd(args.gpu_device)
        fdtd.load(str(projects[stage26.BASELINE_CASE]))
        fdtd.runanalysis(stage12.PABS_GROUP)
        forward, grid = stage12.monitor_electric(fdtd, stage12.PABS_FIELD)

        fdtd.load(str(adjoint_project))
        fdtd.cwnorm(1)
        adjoint_first, adjoint_grid = stage12.monitor_electric(
            fdtd, stage12.PABS_FIELD
        )
        fdtd.cwnorm(2)
        adjoint_average, average_grid = stage12.monitor_electric(
            fdtd, stage12.PABS_FIELD
        )
        adjoint, normalization = stage12.reconstruct_fieldregion_only_cw(
            adjoint_first, adjoint_average
        )
        adjoint *= profile_scale / base_amplitude
        field_mismatch = max(
            stage21.maximum_grid_mismatch(grid, adjoint_grid),
            stage21.maximum_grid_mismatch(adjoint_grid, average_grid),
        )
        if field_mismatch >= 2.0e-18:
            raise RuntimeError(f"forward/adjoint grid mismatch {field_mismatch:.3e} m")

        volumes = stage12.component_volumes(grid)
        comparisons = {}
        for h_um, (minus_name, plus_name) in stage26.FD_CASES.items():
            component_terms = {}
            total = 0.0
            complex_total = 0.0j
            for component_index, component in enumerate("xyz"):
                fdtd.load(str(projects[minus_name]))
                epsilon_minus = read_component_epsilon(fdtd, component, grid)
                fdtd.load(str(projects[plus_name]))
                epsilon_plus = read_component_epsilon(fdtd, component, grid)
                d_epsilon_per_um = (epsilon_plus - epsilon_minus) / (2.0 * h_um)
                derivative = complex(
                    np.sum(
                        2.0
                        * stage12.EPS0
                        * forward[..., 0, component_index]
                        * adjoint[..., 0, component_index]
                        * d_epsilon_per_um
                        * volumes[component_index]
                    )
                )
                nonzero = np.abs(d_epsilon_per_um) > 0.0
                component_terms[component] = {
                    "real_J_proxy_per_um": float(derivative.real),
                    "imag_J_proxy_per_um": float(derivative.imag),
                    "d_epsilon_nonzero_cell_count": int(np.count_nonzero(nonzero)),
                    "d_epsilon_l2_per_um": float(np.linalg.norm(d_epsilon_per_um)),
                    "d_epsilon_max_abs_per_um": float(
                        np.max(np.abs(d_epsilon_per_um))
                    ),
                    "all_finite": bool(np.all(np.isfinite(d_epsilon_per_um))),
                }
                total += float(derivative.real)
                complex_total += derivative
                del epsilon_minus, epsilon_plus, d_epsilon_per_um

            fd_value = float(
                source_result["finite_difference"][f"h_{h_um:g}_um"][
                    "derivative_J_proxy_per_um"
                ]
            )
            comparisons[f"h_{h_um:g}_um"] = {
                "h_um": float(h_um),
                "minus_case": minus_name,
                "plus_case": plus_name,
                "FD_J_proxy_per_um": fd_value,
                "discrete_epsilon_AD_J_proxy_per_um": total,
                "complex_contraction_J_proxy_per_um": [
                    float(complex_total.real),
                    float(complex_total.imag),
                ],
                "relative_error": relative(total, fd_value),
                "sign_agrees": bool(total * fd_value > 0.0),
                "component_terms": component_terms,
            }

        strong = comparisons["h_0.05_um"]
        derivative_step_change = relative(
            comparisons["h_0.05_um"]["discrete_epsilon_AD_J_proxy_per_um"],
            comparisons["h_0.1_um"]["discrete_epsilon_AD_J_proxy_per_um"],
        )
        passed = bool(
            strong["relative_error"] < 0.01
            and strong["sign_agrees"]
            and derivative_step_change < 0.01
            and field_mismatch < 2.0e-18
            and all(
                term["all_finite"]
                for row in comparisons.values()
                for term in row["component_terms"].values()
            )
        )
        result.update(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": (
                    "VALIDATED_AU_SMOOTH3D_DISCRETE_EPSILON_SHAPE_ADJOINT"
                    if passed
                    else "FAILED_AU_SMOOTH3D_DISCRETE_EPSILON_SHAPE_ADJOINT"
                ),
                "passed": passed,
                "scope": (
                    "one-parameter, solver-discrete conformal-Yee epsilon "
                    "shape Jacobian diagnostic using completed FSPs"
                ),
                "formula": (
                    "2*eps0*Re(sum_c integral E_fwd,c E_adj,c "
                    "d(epsilon_Yee,c)/d(a_um) dV_c)"
                ),
                "normalization": normalization,
                "maximum_forward_adjoint_grid_mismatch_m": field_mismatch,
                "comparisons": comparisons,
                "discrete_derivative_step_relative_change": derivative_step_change,
                "gates": {
                    "strong_h0p05_relative_error_lt_1pct": strong["relative_error"]
                    < 0.01,
                    "strong_sign_agrees": strong["sign_agrees"],
                    "discrete_derivative_step_change_lt_1pct": derivative_step_change
                    < 0.01,
                    "coordinate_mismatch_lt_2e_18_m": field_mismatch < 2.0e-18,
                },
                "production_contract": {
                    "one_Maxwell_solve_per_design_variable_allowed": False,
                    "this_one_parameter_FD_epsilon_Jacobian_is_production_ready": False,
                    "purpose": (
                        "separate continuum boundary-kernel error from the "
                        "solver-discrete conformal material derivative"
                    ),
                },
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")

    print(json.dumps(result, indent=2, default=str))
    return 0 if bool(result.get("passed", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
