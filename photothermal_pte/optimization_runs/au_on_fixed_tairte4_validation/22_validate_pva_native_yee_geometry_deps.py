#!/usr/bin/env python3
"""Re-test the PVA Au geometry Jacobian on native component Yee grids.

Stage 20 paired a common-grid ``getresult(..., 'E')`` array with conformal
``index_x/y/z`` values.  Equal array bounds do not prove component-wise Yee
collocation.  This control instead reads the solver's ``index_detail`` offsets
and contracts each d-epsilon component with the matching native Ex/Ey/Ez grid.
No Maxwell solve is performed; only runsetup remeshes are used.
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
STAGE16 = HERE / "16_run_au_smooth_ellipse_width_control.py"
STAGE21 = HERE / "21_validate_fixed_geometry_au_material_adjoint.py"
DEFAULT_RAW = Path("/home/seunghyun/tairte4/raw_artifacts/au_topology_validation")
BASELINE_CASE = "pva5_fixedgrid_smooth_ellipse_a8p0_b18_edge50_forward"
AU_OBJECT = "rho1_scalar_complex_block"
BASELINE_HALF_X_M = 8.0e-6
HALF_Y_M = 18.0e-6
VERTEX_COUNT = 512
FD_CASES = {
    0.10: (
        "pva5_fixedgrid_smooth_ellipse_a7p9_b18_edge50_forward",
        "pva5_fixedgrid_smooth_ellipse_a8p1_b18_edge50_forward",
    ),
    0.05: (
        "pva5_fixedgrid_smooth_ellipse_a7p95_b18_edge50_forward",
        "pva5_fixedgrid_smooth_ellipse_a8p05_b18_edge50_forward",
    ),
}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage12 = load("au_native_geometry_source", STAGE12)
stage16 = load("au_native_geometry_ellipse", STAGE16)
stage21 = load("au_native_geometry_contract", STAGE21)


def detail_coordinate(detail: dict, component: str, axis: str) -> np.ndarray:
    return np.asarray(
        detail[f"{axis}_offset"] if component == axis else detail[axis], float
    )


def component_coordinate_audit(grid: dict, detail: dict) -> dict[str, object]:
    components = {}
    maximum = 0.0
    for component in "xyz":
        per_axis = {}
        for axis in "xyz":
            field_coordinate = np.asarray(grid[axis], float).copy()
            if component == axis:
                field_coordinate += np.asarray(grid[f"delta_{axis}"], float)
            index_coordinate = detail_coordinate(detail, component, axis)
            mismatch = (
                float("inf")
                if field_coordinate.shape != index_coordinate.shape
                else float(np.max(np.abs(field_coordinate - index_coordinate)))
            )
            maximum = max(maximum, mismatch)
            per_axis[axis] = {
                "mismatch_m": mismatch,
                "field_bounds_m": [
                    float(field_coordinate[0]),
                    float(field_coordinate[-1]),
                ],
                "index_bounds_m": [
                    float(index_coordinate[0]),
                    float(index_coordinate[-1]),
                ],
            }
        components[component] = per_axis
    return {"components": components, "maximum_mismatch_m": maximum}


def remeshed_detail(fdtd, project: Path, half_x_m: float) -> dict:
    fdtd.load(str(project))
    fdtd.switchtolayout()
    if int(fdtd.getnamednumber(AU_OBJECT)) != 1:
        raise RuntimeError(f"expected exactly one {AU_OBJECT!r}")
    fdtd.setnamed(
        AU_OBJECT,
        "vertices",
        stage16.ellipse_vertices(half_x_m, HALF_Y_M, VERTEX_COUNT),
    )
    fdtd.runsetup()
    return stage12.index_detail(fdtd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--adjoint-dir",
        type=Path,
        default=DEFAULT_RAW
        / "pva5_fixedgrid_smooth_ellipse_external_field_adjoint_gpu0",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument(
        "--steps-nm", type=float, nargs="+", default=[100.0, 50.0, 25.0, 12.5]
    )
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "au_pva_native_yee_geometry_deps_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_AU_PVA_NATIVE_YEE_GEOMETRY_DEPS",
        "passed": False,
        "Maxwell_forward_solves_this_invocation": 0,
        "Maxwell_adjoint_solves_this_invocation": 0,
        "runsetup_only_remeshes": 0,
        "thermal_solves": 0,
        "optimizer_started": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "finite_difference_used_to_fit_AD": False,
    }
    fdtd = None
    try:
        baseline_project, _baseline_result = stage12.pq.checked_project(
            args.raw_root / BASELINE_CASE
        )
        source_result = json.loads(
            (args.adjoint_dir / "au_sharp_interface_external_field_result.json").read_text()
        )
        profile_scale = float(source_result["source"]["profile_scale"])
        base_amplitude = float(source_result["source"]["fieldregion_base_amplitude"])
        adjoint_project = Path(source_result["adjoint"]["project"]["path"])

        fdtd, _audit, _runtime = stage12.open_fdtd(args.gpu_device)
        fdtd.load(str(baseline_project))
        fdtd.runanalysis(stage12.PABS_GROUP)
        forward, grid = stage12.monitor_electric(fdtd, stage12.PABS_FIELD)
        baseline_detail = stage12.index_detail(fdtd)
        baseline_coordinate_audit = component_coordinate_audit(grid, baseline_detail)

        fdtd.load(str(adjoint_project))
        fdtd.cwnorm(1)
        adjoint_first, adjoint_grid = stage12.monitor_electric(fdtd, stage12.PABS_FIELD)
        fdtd.cwnorm(2)
        adjoint_average, average_grid = stage12.monitor_electric(fdtd, stage12.PABS_FIELD)
        adjoint, normalization = stage12.reconstruct_fieldregion_only_cw(
            adjoint_first, adjoint_average
        )
        adjoint *= profile_scale / base_amplitude
        forward_adjoint_mismatch = stage21.maximum_grid_mismatch(grid, adjoint_grid)
        normalization_grid_mismatch = stage21.maximum_grid_mismatch(
            adjoint_grid, average_grid
        )

        derivatives = []
        coordinate_audits = []
        for step_nm in args.steps_nm:
            step_m = float(step_nm) * 1.0e-9
            plus = remeshed_detail(
                fdtd, baseline_project, BASELINE_HALF_X_M + step_m
            )
            result["runsetup_only_remeshes"] += 1
            minus = remeshed_detail(
                fdtd, baseline_project, BASELINE_HALF_X_M - step_m
            )
            result["runsetup_only_remeshes"] += 1
            plus_minus_mismatch = stage21.maximum_index_detail_mismatch(plus, minus)
            plus_baseline_mismatch = stage21.maximum_index_detail_mismatch(
                baseline_detail, plus
            )
            minus_baseline_mismatch = stage21.maximum_index_detail_mismatch(
                baseline_detail, minus
            )
            plus_component_audit = component_coordinate_audit(grid, plus)
            minus_component_audit = component_coordinate_audit(grid, minus)
            maximum = max(
                plus_minus_mismatch,
                plus_baseline_mismatch,
                minus_baseline_mismatch,
                plus_component_audit["maximum_mismatch_m"],
                minus_component_audit["maximum_mismatch_m"],
            )
            coordinate_audits.append(
                {
                    "step_nm": step_nm,
                    "plus_minus_index_detail_mismatch_m": plus_minus_mismatch,
                    "plus_baseline_index_detail_mismatch_m": plus_baseline_mismatch,
                    "minus_baseline_index_detail_mismatch_m": minus_baseline_mismatch,
                    "plus_field_index_component_audit": plus_component_audit,
                    "minus_field_index_component_audit": minus_component_audit,
                    "maximum_mismatch_m": maximum,
                }
            )
            if maximum >= 2.0e-18:
                raise RuntimeError(
                    f"native Yee geometry d-epsilon grid mismatch {maximum:.6e} m"
                )
            epsilon_plus = stage21.epsilon_from_detail(plus)
            epsilon_minus = stage21.epsilon_from_detail(minus)
            d_epsilon = (epsilon_plus - epsilon_minus) / (2.0 * step_m)
            variants = stage21.contraction_variants(
                forward, adjoint, d_epsilon, grid
            )
            official = variants["official_Ef_times_Ea"]
            derivatives.append(
                {
                    "step_nm": step_nm,
                    "official_AD_J_proxy_per_m": official[
                        "real_derivative_J_proxy_per_relative_epsilon"
                    ],
                    "official_AD_J_proxy_per_um": official[
                        "real_derivative_J_proxy_per_relative_epsilon"
                    ]
                    * 1.0e-6,
                    "complex_convention_audit": variants,
                    "nonzero_d_epsilon_by_component": {
                        component: int(
                            np.count_nonzero(
                                np.abs(d_epsilon[..., 0, index]) > 0.0
                            )
                        )
                        for index, component in enumerate("xyz")
                    },
                }
            )

        changes = []
        for coarse, fine in zip(derivatives[:-1], derivatives[1:]):
            changes.append(
                {
                    "coarse_step_nm": coarse["step_nm"],
                    "fine_step_nm": fine["step_nm"],
                    "relative_change": stage12.relative(
                        coarse["official_AD_J_proxy_per_um"],
                        fine["official_AD_J_proxy_per_um"],
                    ),
                }
            )
        selected = derivatives[-1]["official_AD_J_proxy_per_um"]
        comparisons = {}
        for h_um in (0.10, 0.05):
            minus_name, plus_name = FD_CASES[h_um]
            minus = float(
                source_result["objective_cases"][minus_name]["objective_J_proxy"]
            )
            plus = float(
                source_result["objective_cases"][plus_name]["objective_J_proxy"]
            )
            fd_value = (plus - minus) / (2.0 * h_um)
            comparisons[f"h_{h_um:g}_um"] = {
                "FD_J_proxy_per_um": fd_value,
                "AD_J_proxy_per_um": selected,
                "relative_error": stage12.relative(selected, fd_value),
                "sign_agrees": bool(selected * fd_value > 0.0),
            }
        final_change = changes[-1]["relative_change"]
        maximum_coordinate_mismatch = max(
            baseline_coordinate_audit["maximum_mismatch_m"],
            forward_adjoint_mismatch,
            normalization_grid_mismatch,
            *(row["maximum_mismatch_m"] for row in coordinate_audits),
        )
        strong = comparisons["h_0.05_um"]
        gates = {
            "strong_AD_FD_relative_error_lt_1pct": strong["relative_error"] < 0.01,
            "strong_sign_agrees": strong["sign_agrees"],
            "d_epsilon_step_change_lt_0p5pct": final_change < 0.005,
            "component_Yee_coordinate_mismatch_lt_2e_18_m": (
                maximum_coordinate_mismatch < 2.0e-18
            ),
            "source_normalization_residual_lt_1e_12": normalization[
                "two_normalization_state_spatial_residual"
            ]
            < 1.0e-12,
        }
        passed = bool(all(gates.values()))
        result.update(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": (
                    "PVA5 exact scalar-Au smooth ellipse x-semi-axis derivative "
                    "using native component-specific Yee E and index_detail d-epsilon"
                ),
                "baseline_case": BASELINE_CASE,
                "baseline_coordinate_audit": baseline_coordinate_audit,
                "coordinate_audits": coordinate_audits,
                "forward_adjoint_grid_mismatch_m": forward_adjoint_mismatch,
                "normalization_grid_mismatch_m": normalization_grid_mismatch,
                "maximum_coordinate_mismatch_m": maximum_coordinate_mismatch,
                "normalization": normalization,
                "native_Yee_geometry_derivatives": derivatives,
                "d_epsilon_step_convergence": changes,
                "d_epsilon_final_relative_change": final_change,
                "AD_FD_comparison": comparisons,
                "gates": gates,
                "passed": passed,
                "status": (
                    "VALIDATED_AU_PVA_NATIVE_YEE_GEOMETRY_DEPS"
                    if passed
                    else "FAILED_AU_PVA_NATIVE_YEE_GEOMETRY_DEPS_ADFD"
                ),
                "production_Au_optimization_permitted": passed,
                "common_grid_stage20_is_not_used_for_promotion": True,
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
    return 0 if result.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
