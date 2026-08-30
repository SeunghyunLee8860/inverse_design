#!/usr/bin/env python3
"""Separate the Au adjoint-source convention from moving-boundary effects.

The ellipse, PVA settings, local mesh, monitors, source, and Maxwell domain are
identical in every case.  Only an additive real shift of the scalar Au relative
permittivity is changed.  Completed forward and adjoint FSPs are reused; this
script performs no Maxwell, thermal, electrical, PTE, or optimization solve.
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
DEFAULT_RAW = Path("/home/seunghyun/tairte4/raw_artifacts/au_topology_validation")
BASELINE_CASE = "pva5_fixedgrid_smooth_ellipse_a8p0_b18_edge50_forward"
MATERIAL_CASES = {
    100.0: (
        "pva5_fixedgrid_material_eps_minus100_forward",
        "pva5_fixedgrid_material_eps_plus100_forward",
    ),
    50.0: (
        "pva5_fixedgrid_material_eps_minus50_forward",
        "pva5_fixedgrid_material_eps_plus50_forward",
    ),
}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage12 = load("au_material_source_control", STAGE12)


def maximum_grid_mismatch(left: dict, right: dict) -> float:
    maximum = 0.0
    for key in ("x", "y", "z", "delta_x", "delta_y", "delta_z"):
        a = np.asarray(left[key], float)
        b = np.asarray(right[key], float)
        if a.shape != b.shape:
            return float("inf")
        maximum = max(maximum, float(np.max(np.abs(a - b))))
    return maximum


def maximum_index_detail_mismatch(left: dict, right: dict) -> float:
    maximum = 0.0
    for key in ("x", "x_offset", "y", "y_offset", "z", "z_offset"):
        a = np.asarray(left[key], float)
        b = np.asarray(right[key], float)
        if a.shape != b.shape:
            return float("inf")
        maximum = max(maximum, float(np.max(np.abs(a - b))))
    return maximum


def epsilon_from_detail(detail: dict) -> np.ndarray:
    return np.stack(
        [np.asarray(detail[f"epsilon_{component}"], complex) for component in "xyz"],
        axis=-1,
    )[..., None, :]


def contraction_variants(
    forward: np.ndarray,
    adjoint: np.ndarray,
    d_epsilon: np.ndarray,
    grid: dict,
) -> dict[str, object]:
    volumes = stage12.component_volumes(grid)
    variants = {
        "official_Ef_times_Ea": (forward, adjoint),
        "Ef_times_conj_Ea": (forward, np.conj(adjoint)),
        "conj_Ef_times_Ea": (np.conj(forward), adjoint),
        "conj_Ef_times_conj_Ea": (np.conj(forward), np.conj(adjoint)),
    }
    result: dict[str, object] = {}
    for name, (left, right) in variants.items():
        terms = {}
        total = 0.0
        complex_total = 0.0j
        for index, component in enumerate("xyz"):
            integrand = (
                2.0
                * stage12.EPS0
                * left[..., 0, index]
                * right[..., 0, index]
                * d_epsilon[..., 0, index]
            )
            complex_value = complex(np.sum(integrand * volumes[index]))
            value = float(np.real(complex_value))
            terms[component] = {
                "real": value,
                "imag": float(np.imag(complex_value)),
            }
            total += value
            complex_total += complex_value
        result[name] = {
            "component_terms": terms,
            "real_derivative_J_proxy_per_relative_epsilon": total,
            "complex_sum": [float(complex_total.real), float(complex_total.imag)],
        }
    return result


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
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "au_fixed_geometry_material_adjoint_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_AU_FIXED_GEOMETRY_MATERIAL_ADJOINT",
        "passed": False,
        "Maxwell_forward_solves_this_invocation": 0,
        "Maxwell_adjoint_solves_this_invocation": 0,
        "runsetup_calls_this_invocation": 0,
        "thermal_solves": 0,
        "optimizer_started": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "finite_difference_used_to_fit_AD": False,
    }
    fdtd = None
    try:
        case_projects = {}
        case_results = {}
        for name in {
            BASELINE_CASE,
            *(case for pair in MATERIAL_CASES.values() for case in pair),
        }:
            case_projects[name], case_results[name] = stage12.pq.checked_project(
                args.raw_root / name
            )

        source_result = json.loads(
            (args.adjoint_dir / "au_sharp_interface_external_field_result.json").read_text()
        )
        adjoint_project = Path(source_result["adjoint"]["project"]["path"])
        profile_scale = float(source_result["source"]["profile_scale"])
        base_amplitude = float(source_result["source"]["fieldregion_base_amplitude"])

        fdtd, audit, _runtime = stage12.open_fdtd(args.gpu_device)

        states = {}
        for name, project in case_projects.items():
            fdtd.load(str(project))
            fdtd.runanalysis(stage12.PABS_GROUP)
            electric, grid = stage12.monitor_electric(fdtd, stage12.PABS_FIELD)
            objective, _source, metadata = stage12.fixed_air_objective_and_source(
                electric, grid
            )
            detail = stage12.index_detail(fdtd)
            epsilon = epsilon_from_detail(detail)
            if epsilon.shape != electric.shape:
                raise RuntimeError(f"{name}: E/index shape mismatch")
            states[name] = {
                "electric": electric,
                "grid": grid,
                "detail": detail,
                "epsilon": epsilon,
                "objective": objective,
                "objective_components": metadata["component_value_J_proxy"],
            }

        baseline = states[BASELINE_CASE]
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

        grid_mismatches = {
            name: maximum_grid_mismatch(baseline["grid"], state["grid"])
            for name, state in states.items()
        }
        index_mismatches = {
            name: maximum_index_detail_mismatch(
                baseline["detail"], state["detail"]
            )
            for name, state in states.items()
        }
        adjoint_grid_mismatch = maximum_grid_mismatch(
            baseline["grid"], adjoint_grid
        )
        normalization_grid_mismatch = maximum_grid_mismatch(
            adjoint_grid, average_grid
        )
        maximum_mismatch = max(
            *grid_mismatches.values(),
            *index_mismatches.values(),
            adjoint_grid_mismatch,
            normalization_grid_mismatch,
        )
        if maximum_mismatch >= 2.0e-18:
            raise RuntimeError(f"fixed material-control grid mismatch {maximum_mismatch}")

        steps = []
        for step, (minus_name, plus_name) in sorted(
            MATERIAL_CASES.items(), reverse=True
        ):
            minus = states[minus_name]
            plus = states[plus_name]
            fd_value = (plus["objective"] - minus["objective"]) / (2.0 * step)
            d_epsilon = (plus["epsilon"] - minus["epsilon"]) / (2.0 * step)
            variants = contraction_variants(
                baseline["electric"], adjoint, d_epsilon, baseline["grid"]
            )
            official = float(
                variants["official_Ef_times_Ea"][
                    "real_derivative_J_proxy_per_relative_epsilon"
                ]
            )
            nonzero = np.abs(d_epsilon) > 0.0
            interior_derivatives = {}
            for component_index, component in enumerate("xyz"):
                values = d_epsilon[..., 0, component_index]
                active = nonzero[..., 0, component_index]
                interior_derivatives[component] = {
                    "nonzero_cells": int(np.count_nonzero(active)),
                    "real_median_over_nonzero": float(np.median(values.real[active])),
                    "imag_max_abs_over_nonzero": float(np.max(np.abs(values.imag[active]))),
                }
            steps.append(
                {
                    "step_relative_epsilon": step,
                    "minus_case": minus_name,
                    "plus_case": plus_name,
                    "minus_objective_J_proxy": minus["objective"],
                    "plus_objective_J_proxy": plus["objective"],
                    "FD_J_proxy_per_relative_epsilon": fd_value,
                    "official_AD_J_proxy_per_relative_epsilon": official,
                    "official_relative_error": stage12.relative(official, fd_value),
                    "official_sign_agrees": bool(official * fd_value > 0.0),
                    "d_epsilon_audit": interior_derivatives,
                    "complex_convention_audit": variants,
                }
            )

        fd_step_change = stage12.relative(
            steps[0]["FD_J_proxy_per_relative_epsilon"],
            steps[1]["FD_J_proxy_per_relative_epsilon"],
        )
        ad_step_change = stage12.relative(
            steps[0]["official_AD_J_proxy_per_relative_epsilon"],
            steps[1]["official_AD_J_proxy_per_relative_epsilon"],
        )
        selected = steps[-1]
        gates = {
            "FD_step_change_lt_1pct": fd_step_change < 0.01,
            "AD_step_change_lt_0p5pct": ad_step_change < 0.005,
            "official_AD_FD_relative_error_lt_1pct": selected[
                "official_relative_error"
            ]
            < 0.01,
            "official_sign_agrees": selected["official_sign_agrees"],
            "all_coordinates_match_lt_2e_18_m": maximum_mismatch < 2.0e-18,
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
                    "fixed smooth scalar-Au ellipse, PVA5, fixed mesh; additive "
                    "real-epsilon material derivative only"
                ),
                "baseline_case": BASELINE_CASE,
                "material_steps": steps,
                "FD_step_relative_change": fd_step_change,
                "AD_step_relative_change": ad_step_change,
                "coordinate_audit": {
                    "forward_case_grid_mismatch_m": grid_mismatches,
                    "forward_case_index_detail_mismatch_m": index_mismatches,
                    "forward_adjoint_grid_mismatch_m": adjoint_grid_mismatch,
                    "normalization_grid_mismatch_m": normalization_grid_mismatch,
                    "maximum_mismatch_m": maximum_mismatch,
                },
                "source": {
                    "profile_scale": profile_scale,
                    "fieldregion_base_amplitude": base_amplitude,
                    "normalization": normalization,
                    "formula_selected": "2*eps0*Re integral(Ef*Ea*d_epsilon)dV",
                },
                "gates": gates,
                "passed": passed,
                "status": (
                    "VALIDATED_AU_FIXED_GEOMETRY_MATERIAL_ADJOINT"
                    if passed
                    else "FAILED_AU_FIXED_GEOMETRY_MATERIAL_ADJOINT_ADFD"
                ),
                "interpretation_if_passed": (
                    "the FieldRegion source normalization and unconjugated "
                    "Ef*Ea contraction are valid; the remaining failure is "
                    "specific to moving high-contrast PVA Au boundaries"
                ),
                "production_Au_optimization_permitted": False,
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
