#!/usr/bin/env python3
"""Validate a solver-discrete Au geometry Jacobian against central FD.

This diagnostic keeps the exact scalar-Au rectangular geometry but does not
sample the singular metal boundary field.  Instead it follows the bundled
v261 ``use_deps`` route: remesh plus/minus geometry perturbations without a
Maxwell solve, form the component-wise derivative of the conformal-mesh
permittivity, and contract it with collocated forward/adjoint electric fields.

The optical objective and completed GPU forward/adjoint fields are inherited
from the corner-free fixed-external-field control.  No empirical scaling,
gradient fitting, thermal solve, PTE solve, or optimization is permitted.
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
SOURCE_SCRIPT = HERE / "12_run_au_sharp_interface_external_field_adjoint.py"
DEFAULT_RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/au_topology_validation"
)
BASELINE_CASE = "corner_free_y18_width_8p0_edge25_forward"
AU_OBJECT = "rho1_scalar_complex_block"
INDEX_MONITOR = "au_solver_discrete_deps_index"
EPS0 = 8.8541878128e-12
BASELINE_HALF_WIDTH_M = 8.0e-6


def load_source_module():
    spec = importlib.util.spec_from_file_location("au_external_field", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = load_source_module()

from photothermal_pte.finite_inverse_design.run_v261_large_background_mixed_optical_adfd import (  # noqa: E402
    frequency_slice,
)


def relative(left: complex | float, right: complex | float) -> float:
    return float(abs(left - right) / max(abs(left), abs(right), 1.0e-300))


def trapezoid_weights(coordinate: np.ndarray) -> np.ndarray:
    coordinate = np.asarray(coordinate, float).reshape(-1)
    if coordinate.size < 2 or not np.all(np.diff(coordinate) > 0.0):
        raise ValueError("quadrature coordinate must be strictly increasing")
    weights = np.empty_like(coordinate)
    weights[0] = 0.5 * (coordinate[1] - coordinate[0])
    weights[-1] = 0.5 * (coordinate[-1] - coordinate[-2])
    weights[1:-1] = 0.5 * (coordinate[2:] - coordinate[:-2])
    return weights


def collocated_electric(fdtd: object, monitor: str) -> tuple[np.ndarray, dict]:
    dataset = fdtd.getresult(monitor, "E")
    electric = np.asarray(dataset["E"], np.complex128)
    coordinates = {
        axis: np.asarray(dataset[axis], float).reshape(-1) for axis in "xyz"
    }
    expected = tuple(coordinates[axis].size for axis in "xyz") + (1, 3)
    if electric.shape != expected:
        raise RuntimeError(f"collocated E shape {electric.shape} != {expected}")
    return electric, coordinates


def coordinate_mismatch(left: dict, right: dict) -> float:
    mismatch = 0.0
    for axis in "xyz":
        a = np.asarray(left[axis], float).reshape(-1)
        b = np.asarray(right[axis], float).reshape(-1)
        if a.shape != b.shape:
            return float("inf")
        mismatch = max(mismatch, float(np.max(np.abs(a - b))))
    return mismatch


def native_grid_mismatch(left: dict, right: dict) -> float:
    mismatch = coordinate_mismatch(left, right)
    for axis in "xyz":
        key = f"delta_{axis}"
        a = np.asarray(left[key], float).reshape(-1)
        b = np.asarray(right[key], float).reshape(-1)
        if a.shape != b.shape:
            return float("inf")
        mismatch = max(mismatch, float(np.max(np.abs(a - b))))
    return mismatch


def component_coordinate_mismatch(left: dict, right: dict) -> dict[str, float]:
    result = {}
    for component in "xyz":
        component_maximum = 0.0
        for axis in "xyz":
            left_coordinate = np.asarray(left[axis], float).copy()
            right_coordinate = np.asarray(right[axis], float).copy()
            if component == axis:
                left_coordinate += np.asarray(left[f"delta_{axis}"], float)
                right_coordinate += np.asarray(right[f"delta_{axis}"], float)
            if left_coordinate.shape != right_coordinate.shape:
                component_maximum = float("inf")
                break
            component_maximum = max(
                component_maximum,
                float(np.max(np.abs(left_coordinate - right_coordinate))),
            )
        result[component] = component_maximum
    return result


def add_matching_index_monitor(fdtd: object, field_monitor: str) -> None:
    if int(fdtd.getnamednumber(INDEX_MONITOR)):
        fdtd.select(INDEX_MONITOR)
        fdtd.delete()
    monitor = fdtd.addindex()
    monitor["name"] = INDEX_MONITOR
    monitor["override global monitor settings"] = True
    monitor["use source limits"] = True
    monitor["frequency points"] = 1
    monitor["record conformal mesh when possible"] = True
    monitor["spatial interpolation"] = "none"
    monitor["monitor type"] = str(fdtd.getnamed(field_monitor, "monitor type"))
    parent = field_monitor.rsplit("::", 1)[0] if "::" in field_monitor else None
    for axis in "xyz":
        parent_offset = (
            float(fdtd.getnamed(parent, axis)) if parent is not None else 0.0
        )
        monitor[axis] = float(fdtd.getnamed(field_monitor, axis)) + parent_offset
        monitor[f"{axis} span"] = float(fdtd.getnamed(field_monitor, f"{axis} span"))


def remeshed_epsilon(
    fdtd: object,
    *,
    project: Path,
    half_width_m: float,
    field_monitor: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return collocated component epsilon after runsetup, without FDTD run."""

    fdtd.load(str(project))
    fdtd.switchtolayout()
    if int(fdtd.getnamednumber(AU_OBJECT)) != 1:
        raise RuntimeError(f"expected exactly one {AU_OBJECT!r}")
    fdtd.setnamed(AU_OBJECT, "x min", -float(half_width_m))
    fdtd.setnamed(AU_OBJECT, "x max", float(half_width_m))
    add_matching_index_monitor(fdtd, field_monitor)
    fdtd.runsetup()
    dataset = fdtd.getresult(INDEX_MONITOR, "index")
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


def contract_discrete_deps(
    forward: np.ndarray,
    adjoint: np.ndarray,
    epsilon_plus: dict[str, np.ndarray],
    epsilon_minus: dict[str, np.ndarray],
    coordinates: dict[str, np.ndarray],
    *,
    step_m: float,
) -> dict[str, object]:
    """Contract 2 eps0 E_f E_a with the centered conformal d-epsilon."""

    if forward.shape != adjoint.shape:
        raise ValueError("forward/adjoint collocated shape mismatch")
    spatial_shape = forward.shape[:3]
    weights = [trapezoid_weights(coordinates[axis]) for axis in "xyz"]
    volume = (
        weights[0][:, None, None]
        * weights[1][None, :, None]
        * weights[2][None, None, :]
    )
    total = 0.0
    terms: dict[str, object] = {}
    for component_index, component in enumerate("xyz"):
        plus = np.asarray(epsilon_plus[component], np.complex128)
        minus = np.asarray(epsilon_minus[component], np.complex128)
        if plus.shape != spatial_shape or minus.shape != spatial_shape:
            raise ValueError(f"epsilon {component} shape mismatch")
        d_epsilon = (plus - minus) / (2.0 * step_m)
        integrand = (
            2.0
            * EPS0
            * forward[..., 0, component_index]
            * adjoint[..., 0, component_index]
            * d_epsilon
        )
        if volume.shape != spatial_shape:
            raise ValueError(f"volume shape {volume.shape} != {spatial_shape}")
        value = float(np.sum(np.real(integrand) * volume))
        nonzero = np.abs(d_epsilon) > 0.0
        terms[component] = {
            "derivative_J_proxy_per_m": value,
            "nonzero_d_epsilon_cells": int(np.count_nonzero(nonzero)),
            "maximum_abs_d_epsilon_per_m": float(np.max(np.abs(d_epsilon))),
            "all_finite": bool(
                np.all(np.isfinite(d_epsilon)) and np.all(np.isfinite(integrand))
            ),
        }
        total += value
    return {
        "step_m": float(step_m),
        "rule": (
            "v261-style solver-discrete central d-epsilon remesh contracted "
            "with collocated 2*eps0*E_forward*E_adjoint"
        ),
        "component_terms": terms,
        "total_J_proxy_per_m": total,
        "total_J_proxy_per_um": total * 1.0e-6,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--adjoint-dir",
        type=Path,
        default=DEFAULT_RAW / "corner_free_y18_external_field_adjoint_gpu0",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument(
        "--steps-nm", type=float, nargs="+", default=[100.0, 50.0, 25.0, 12.5]
    )
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "au_solver_discrete_deps_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_AU_SOLVER_DISCRETE_DEPS_ADFD",
        "passed": False,
        "Maxwell_forward_solves_this_invocation": 0,
        "Maxwell_adjoint_solves_this_invocation": 0,
        "runsetup_only_remeshes": 0,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "finite_difference_used_to_fit_AD": False,
        "thermal_solves": 0,
        "optimizer_started": False,
    }
    fdtd = None
    try:
        source_result_path = args.adjoint_dir / "au_sharp_interface_external_field_result.json"
        source_result = json.loads(source_result_path.read_text())
        baseline_project, baseline_case = source.pq.checked_project(
            args.raw_root / BASELINE_CASE
        )
        adjoint_project = Path(source_result["adjoint"]["project"]["path"])
        profile_scale = float(source_result["source"]["profile_scale"])
        base_amplitude = float(
            source_result["source"]["fieldregion_base_amplitude"]
        )

        fdtd, audit, _runtime = source.open_fdtd(args.gpu_device)

        fdtd.load(str(baseline_project))
        fdtd.runanalysis(source.PABS_GROUP)
        forward, forward_coordinates = collocated_electric(fdtd, source.PABS_FIELD)

        fdtd.load(str(adjoint_project))
        fdtd.cwnorm(1)
        adjoint_first, adjoint_coordinates = collocated_electric(
            fdtd, source.PABS_FIELD
        )
        fdtd.cwnorm(2)
        adjoint_average, average_coordinates = collocated_electric(
            fdtd, source.PABS_FIELD
        )
        normalization_grid_mismatch = coordinate_mismatch(
            adjoint_coordinates, average_coordinates
        )
        adjoint, normalization = source.reconstruct_fieldregion_only_cw(
            adjoint_first, adjoint_average
        )
        adjoint *= profile_scale / base_amplitude
        forward_adjoint_grid_mismatch = coordinate_mismatch(
            forward_coordinates, adjoint_coordinates
        )
        del adjoint_first, adjoint_average

        quadratures = []
        coordinate_audits = []
        for step_nm in args.steps_nm:
            step_m = float(step_nm) * 1.0e-9
            epsilon_plus, plus_coordinates = remeshed_epsilon(
                fdtd,
                project=baseline_project,
                half_width_m=BASELINE_HALF_WIDTH_M + step_m,
                field_monitor=source.PABS_FIELD,
            )
            result["runsetup_only_remeshes"] = int(
                result["runsetup_only_remeshes"]
            ) + 1
            epsilon_minus, minus_coordinates = remeshed_epsilon(
                fdtd,
                project=baseline_project,
                half_width_m=BASELINE_HALF_WIDTH_M - step_m,
                field_monitor=source.PABS_FIELD,
            )
            result["runsetup_only_remeshes"] = int(
                result["runsetup_only_remeshes"]
            ) + 1
            plus_minus_mismatch = coordinate_mismatch(
                plus_coordinates, minus_coordinates
            )
            field_index_mismatch = max(
                coordinate_mismatch(forward_coordinates, plus_coordinates),
                coordinate_mismatch(forward_coordinates, minus_coordinates),
            )
            coordinate_audits.append(
                {
                    "step_nm": step_nm,
                    "plus_minus_maximum_coordinate_mismatch_m": plus_minus_mismatch,
                    "field_index_maximum_coordinate_mismatch_m": field_index_mismatch,
                    "shape_xyz": [
                        int(plus_coordinates[axis].size) for axis in "xyz"
                    ],
                    "per_axis_field_index": {
                        axis: {
                            "maximum_mismatch_m": float(
                                np.max(
                                    np.abs(
                                        np.asarray(forward_coordinates[axis])
                                        - np.asarray(plus_coordinates[axis])
                                    )
                                )
                            ),
                            "field_bounds_m": [
                                float(forward_coordinates[axis][0]),
                                float(forward_coordinates[axis][-1]),
                            ],
                            "index_bounds_m": [
                                float(plus_coordinates[axis][0]),
                                float(plus_coordinates[axis][-1]),
                            ],
                        }
                        for axis in "xyz"
                    },
                }
            )
            result["coordinate_audits"] = coordinate_audits
            if max(plus_minus_mismatch, field_index_mismatch) >= 2.0e-18:
                raise RuntimeError(
                    "solver-discrete epsilon and collocated E coordinates do not "
                    f"match: plus/minus={plus_minus_mismatch:.6e} m, "
                    f"field/index={field_index_mismatch:.6e} m, "
                    f"field shape={forward.shape[:3]}, "
                    "index shape="
                    f"{tuple(plus_coordinates[axis].size for axis in 'xyz')}"
                )
            quadratures.append(
                contract_discrete_deps(
                    forward,
                    adjoint,
                    epsilon_plus,
                    epsilon_minus,
                    forward_coordinates,
                    step_m=step_m,
                )
            )
            del epsilon_plus, epsilon_minus

        ad_step_changes = []
        for coarse, fine in zip(quadratures[:-1], quadratures[1:]):
            ad_step_changes.append(
                {
                    "coarse_step_nm": float(coarse["step_m"]) * 1.0e9,
                    "fine_step_nm": float(fine["step_m"]) * 1.0e9,
                    "relative_change": relative(
                        coarse["total_J_proxy_per_um"],
                        fine["total_J_proxy_per_um"],
                    ),
                }
            )
        selected_ad = float(quadratures[-1]["total_J_proxy_per_um"])
        fd = source_result["finite_difference"]
        comparisons = {}
        for key in ("h_0.1_um", "h_0.05_um"):
            fd_value = float(fd[key]["derivative_J_proxy_per_um"])
            comparisons[key] = {
                "FD_J_proxy_per_um": fd_value,
                "AD_J_proxy_per_um": selected_ad,
                "relative_error": relative(selected_ad, fd_value),
                "sign_agrees": bool(np.sign(selected_ad) == np.sign(fd_value)),
            }
        final_step_change = (
            float(ad_step_changes[-1]["relative_change"])
            if ad_step_changes
            else float("inf")
        )
        strong = comparisons["h_0.05_um"]
        all_finite = all(
            row["all_finite"]
            for quadrature in quadratures
            for row in quadrature["component_terms"].values()
        )
        gates = {
            "strong_h0p05_relative_error_lt_1pct": strong["relative_error"] < 0.01,
            "strong_sign_agrees": strong["sign_agrees"],
            "d_epsilon_step_change_lt_0p5pct": final_step_change < 0.005,
            "coordinate_mismatch_lt_2e_18_m": max(
                forward_adjoint_grid_mismatch,
                normalization_grid_mismatch,
                *(row["field_index_maximum_coordinate_mismatch_m"] for row in coordinate_audits),
            )
            < 2.0e-18,
            "all_finite": all_finite,
        }
        passed = bool(all(gates.values()))
        result.update(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": (
                    "exact scalar-Au rectangular width derivative using solver-discrete "
                    "conformal-mesh d-epsilon; no Maxwell solve in this invocation"
                ),
                "source_failure_preserved": {
                    "status": source_result["status"],
                    "official_center_depth_final_relative_change": source_result[
                        "official_center_depth_final_relative_change"
                    ],
                    "full_surface_midpoint_final_relative_change": source_result[
                        "full_surface_midpoint_final_relative_change"
                    ],
                },
                "baseline_case": baseline_case,
                "baseline_project": source_result["objective_cases"][BASELINE_CASE][
                    "project"
                ],
                "adjoint_project": source_result["adjoint"]["project"],
                "normalization": normalization,
                "profile_scale": profile_scale,
                "fieldregion_base_amplitude": base_amplitude,
                "forward_adjoint_coordinate_mismatch_m": forward_adjoint_grid_mismatch,
                "normalization_state_coordinate_mismatch_m": normalization_grid_mismatch,
                "coordinate_audits": coordinate_audits,
                "solver_discrete_deps": quadratures,
                "d_epsilon_step_convergence": ad_step_changes,
                "d_epsilon_final_relative_change": final_step_change,
                "AD_FD_comparison": comparisons,
                "gates": gates,
                "passed": passed,
                "status": (
                    "VALIDATED_AU_SOLVER_DISCRETE_DEPS_BOUNDARY_GRADIENT"
                    if passed
                    else "FAILED_AU_SOLVER_DISCRETE_DEPS_BOUNDARY_GRADIENT_ADFD"
                ),
                "production_Au_optimization_permitted": False,
                "remaining_blocker": (
                    "the direct moving-Au P_Q material/domain derivative remains "
                    "uncertified even if this field-mediated kernel passes"
                ),
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
