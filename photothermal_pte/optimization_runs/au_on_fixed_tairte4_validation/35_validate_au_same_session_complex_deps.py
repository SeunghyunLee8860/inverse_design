#!/usr/bin/env python3
"""Audit the exact-Au CAD shape Jacobian in one Lumerical session.

This control deliberately avoids the v261 temperature/index-perturbation
material wrapper.  It keeps the stable scalar ``(n,k)`` Au material and moves
only the x semi-axis of the smooth 3-D ellipsoid.  For every requested step,
the plus/minus component-wise complex Yee permittivity is read from the same
CAD session and contracted with the already validated forward/adjoint fields.

The imaginary part of epsilon is retained.  This is important for Au and is a
deliberate difference from the generic lumopt2 ``DEpsCalculator`` path, which
currently applies ``real(index_c**2)`` while constructing dEps/dP.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

import importlib.util


HERE = Path(__file__).resolve().parent
STAGE12 = HERE / "12_run_au_sharp_interface_external_field_adjoint.py"
STAGE15 = HERE / "15_validate_solver_discrete_deps_boundary_gradient.py"
STAGE21 = HERE / "21_validate_fixed_geometry_au_material_adjoint.py"
STAGE26 = HERE / "26_validate_au_smooth_3d_ellipsoid_boundary_adjoint.py"
STAGE33 = HERE / "33_validate_au_smooth3d_discrete_epsilon_shape_adjoint.py"
DEFAULT_RAW = Path("/data/seunghyun/tairte4/raw_artifacts/au_topology_validation")
DEFAULT_ADJOINT = DEFAULT_RAW / "pva5_smooth3d_ellipsoid_boundary_adjoint_gpu0"
ELLIPSOID_NAME = "rho1_scalar_complex_block"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage12 = load("au_same_session_stage12", STAGE12)
stage15 = load("au_same_session_stage15", STAGE15)
stage21 = load("au_same_session_stage21", STAGE21)
stage26 = load("au_same_session_stage26", STAGE26)
stage33 = load("au_same_session_stage33", STAGE33)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(
        abs(float(left)), abs(float(right)), np.finfo(float).tiny
    )


def complex_epsilon_dataset(fdtd, grid: dict) -> tuple[list[np.ndarray], float]:
    """Read complex Yee epsilon through LumOpt's layout-mode result route."""

    dataset = fdtd.getresult(stage15.INDEX_MONITOR, "index")
    mismatch = 0.0
    spatial_shape = tuple(np.asarray(grid[axis]).size for axis in "xyz")
    for axis in "xyz":
        field_coordinate = np.asarray(grid[axis], float).reshape(-1)
        index_coordinate = np.asarray(dataset[axis], float).reshape(-1)
        if field_coordinate.shape != index_coordinate.shape:
            raise RuntimeError(
                f"field/index {axis} shape mismatch: "
                f"{field_coordinate.shape} != {index_coordinate.shape}"
            )
        mismatch = max(
            mismatch,
            float(np.max(np.abs(field_coordinate - index_coordinate))),
        )

    epsilon = []
    for component in "xyz":
        index = np.asarray(dataset[f"index_{component}"], np.complex128)
        if index.shape[:3] != spatial_shape:
            raise RuntimeError(
                f"index_{component} spatial shape {index.shape[:3]} != {spatial_shape}"
            )
        while index.ndim > 3:
            if index.shape[-1] != 1:
                raise RuntimeError(
                    f"index_{component} has non-singleton trailing shape {index.shape}"
                )
            index = index[..., 0]
        epsilon.append(np.asarray(index**2, np.complex128))
    return epsilon, mismatch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--adjoint-dir", type=Path, default=DEFAULT_ADJOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 3")
    parser.add_argument(
        "--steps-nm",
        type=float,
        nargs="+",
        default=[100.0, 50.0, 25.0, 10.0, 5.0, 2.5, 1.0],
    )
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "au_same_session_complex_deps_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_AU_SAME_SESSION_COMPLEX_DEPS",
        "passed": False,
        "Maxwell_forward_solves_this_invocation": 0,
        "Maxwell_adjoint_solves_this_invocation": 0,
        "layout_only_index_remeshes": 0,
        "thermal_solves": 0,
        "optimizer_started": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "finite_difference_used_to_fit_AD": False,
        "production_Au_optimization_permitted": False,
    }
    fdtd = None
    try:
        baseline_project, baseline_case = stage26.checked_ellipsoid_project(
            args.raw_root / stage26.BASELINE_CASE
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

        fdtd, session_audit, runtime = stage12.open_fdtd(args.gpu_device)
        fdtd.load(str(baseline_project))
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
            raise RuntimeError(
                f"forward/adjoint grid mismatch {field_mismatch:.3e} m"
            )

        volumes = stage12.component_volumes(grid)
        fdtd.load(str(baseline_project))
        fdtd.switchtolayout()
        fdtd.redrawoff()
        # Analysis-group d-cards disappear in layout mode.  LumOpt's
        # documented CAD-dEps workflow uses a standalone, collocated index
        # monitor for each runsetup remesh, so reproduce that contract here.
        stage15.add_matching_index_monitor(fdtd, stage12.PABS_FIELD)
        stage33.PABS_INDEX = stage15.INDEX_MONITOR
        radius0_m = float(fdtd.getnamed(ELLIPSOID_NAME, "radius"))
        if abs(radius0_m - stage26.A_M) > 1.0e-15:
            raise RuntimeError(
                f"ellipsoid radius readback {radius0_m:.9e} != {stage26.A_M:.9e} m"
            )

        comparisons: dict[str, object] = {}
        index_coordinate_mismatch_m = 0.0
        for step_nm in sorted(set(float(value) for value in args.steps_nm), reverse=True):
            if not (step_nm > 0.0):
                raise ValueError("all dEps steps must be positive")
            h_m = step_nm * 1.0e-9
            h_um = step_nm * 1.0e-3
            eps_by_sign: dict[str, list[np.ndarray]] = {}
            for label, radius in (("plus", radius0_m + h_m), ("minus", radius0_m - h_m)):
                fdtd.setnamed(ELLIPSOID_NAME, "radius", radius)
                # ``switchtolayout`` clears the solved d-cards.  Rebuild only
                # the CAD mesh/index data; this is not a Maxwell solve.
                fdtd.runsetup()
                eps_by_sign[label], mismatch = complex_epsilon_dataset(fdtd, grid)
                index_coordinate_mismatch_m = max(
                    index_coordinate_mismatch_m, mismatch
                )
                result["layout_only_index_remeshes"] = int(
                    result["layout_only_index_remeshes"]
                ) + 1

            component_terms = {}
            total = 0.0
            complex_total = 0.0j
            for component_index, component in enumerate("xyz"):
                d_epsilon_per_um = (
                    eps_by_sign["plus"][component_index]
                    - eps_by_sign["minus"][component_index]
                ) / (2.0 * h_um)
                contraction = complex(
                    np.sum(
                        2.0
                        * stage12.EPS0
                        * forward[..., 0, component_index]
                        * adjoint[..., 0, component_index]
                        * d_epsilon_per_um
                        * volumes[component_index]
                    )
                )
                significant = np.abs(d_epsilon_per_um) > (
                    1.0e-12 * max(float(np.max(np.abs(d_epsilon_per_um))), 1.0)
                )
                component_terms[component] = {
                    "real_J_proxy_per_um": float(contraction.real),
                    "imag_J_proxy_per_um": float(contraction.imag),
                    "d_epsilon_significant_cell_count": int(
                        np.count_nonzero(significant)
                    ),
                    "d_epsilon_l2_per_um": float(np.linalg.norm(d_epsilon_per_um)),
                    "d_epsilon_max_abs_per_um": float(
                        np.max(np.abs(d_epsilon_per_um))
                    ),
                    "d_Re_epsilon_l2_per_um": float(
                        np.linalg.norm(d_epsilon_per_um.real)
                    ),
                    "d_Im_epsilon_l2_per_um": float(
                        np.linalg.norm(d_epsilon_per_um.imag)
                    ),
                    "all_finite": bool(np.all(np.isfinite(d_epsilon_per_um))),
                }
                total += float(contraction.real)
                complex_total += contraction

            key = f"h_{step_nm:g}_nm"
            comparison: dict[str, object] = {
                "step_nm": step_nm,
                "same_session_complex_deps_AD_J_proxy_per_um": total,
                "complex_contraction_J_proxy_per_um": [
                    float(complex_total.real),
                    float(complex_total.imag),
                ],
                "component_terms": component_terms,
            }
            fd_key = f"h_{h_um:g}_um"
            if fd_key in source_result.get("finite_difference", {}):
                fd_value = float(
                    source_result["finite_difference"][fd_key][
                        "derivative_J_proxy_per_um"
                    ]
                )
                comparison.update(
                    {
                        "FD_J_proxy_per_um": fd_value,
                        "relative_error_vs_Maxwell_FD": relative(total, fd_value),
                        "sign_agrees_with_Maxwell_FD": bool(total * fd_value > 0.0),
                    }
                )
            comparisons[key] = comparison
            result["steps"] = comparisons
            result["last_completed_step_nm"] = step_nm
            result_path.write_text(
                json.dumps(result, indent=2, default=str) + "\n"
            )

        fdtd.setnamed(ELLIPSOID_NAME, "radius", radius0_m)
        fdtd.redrawon()

        ordered = sorted(
            comparisons.values(), key=lambda item: float(item["step_nm"]), reverse=True
        )
        finest = ordered[-1]
        for item in ordered:
            item["relative_change_vs_finest"] = relative(
                item["same_session_complex_deps_AD_J_proxy_per_um"],
                finest["same_session_complex_deps_AD_J_proxy_per_um"],
            )
        tail = ordered[-3:] if len(ordered) >= 3 else ordered
        tail_max_change = max(float(item["relative_change_vs_finest"]) for item in tail)
        converged = bool(tail_max_change < 0.01)
        available_fd = [item for item in ordered if "FD_J_proxy_per_um" in item]
        sign_agrees = bool(
            available_fd
            and all(bool(item["sign_agrees_with_Maxwell_FD"]) for item in available_fd)
        )
        passed = bool(converged and sign_agrees)

        result.update(
            {
                "status": (
                    "VALIDATED_AU_SAME_SESSION_COMPLEX_DEPS"
                    if passed
                    else "FAILED_AU_SAME_SESSION_COMPLEX_DEPS"
                ),
                "passed": passed,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "one-parameter exact scalar-Au smooth-ellipsoid CAD shape derivative",
                "session": {
                    "source_name": str(session_audit.SOURCE_NAME),
                    "gpu_device_requested": args.gpu_device,
                },
                "runtime": runtime,
                "baseline": {
                    "project": {
                        "path": str(baseline_project),
                        "bytes": baseline_project.stat().st_size,
                        "sha256": sha256(baseline_project),
                    },
                    "case_result": baseline_case,
                    "ellipsoid_name": ELLIPSOID_NAME,
                    "x_semi_axis_m": radius0_m,
                },
                "adjoint_project": {
                    "path": str(adjoint_project),
                    "bytes": adjoint_project.stat().st_size,
                    "sha256": sha256(adjoint_project),
                },
                "field_grid_maximum_coordinate_mismatch_m": field_mismatch,
                "index_grid_maximum_coordinate_mismatch_m": index_coordinate_mismatch_m,
                "fieldregion_cw_reconstruction": normalization,
                "steps": comparisons,
                "tail_max_relative_change": tail_max_change,
                "dEps_converged_below_1_percent": converged,
                "sign_agrees_with_available_Maxwell_FD": sign_agrees,
                "complex_epsilon_retained": True,
                "lumopt2_generic_real_only_path_used": False,
                "production_Au_optimization_permitted": passed,
                "interpretation": (
                    "Same-session complex CAD dEps/dp converged and has the Maxwell-FD sign."
                    if passed
                    else "The exact-Au conformal CAD dEps/dp did not satisfy both convergence and Maxwell-FD sign gates."
                ),
            }
        )
    except Exception as exc:
        result.update(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return 0 if result.get("passed", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
