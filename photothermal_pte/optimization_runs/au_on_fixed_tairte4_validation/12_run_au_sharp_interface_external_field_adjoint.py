#!/usr/bin/env python3
"""Isolate the v261 Au boundary kernel with a fixed external-field objective.

The objective is an electric-field-energy proxy integrated over two fixed air
slabs below the moving Au edges.  The slabs do not intersect Au for any width
in the finite-difference sweep, so the objective has no moving-domain or
explicit material-loss term.  Its shape derivative must therefore be supplied
only by the Maxwell field-mediated boundary kernel.

Existing exact-binary 25-nm forward FSPs provide independent central finite
differences.  At most one new GPU FieldRegion adjoint is run.  No finite-
difference fitting, empirical normalization, gradient rescaling, thermal,
electrical, PTE, or optimization calculation is allowed here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
PQ_SCRIPT = HERE / "10_run_au_sharp_interface_pq_adjoint.py"


def load_pq_module():
    spec = importlib.util.spec_from_file_location("au_pq_shape_adjoint", PQ_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(PQ_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pq = load_pq_module()

from photothermal_pte.finite_inverse_design.native_yee_q import EPS0  # noqa: E402
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_FIELD,
    PABS_GROUP,
)
from photothermal_pte.finite_inverse_design.run_v261_large_background_mixed_optical_adfd import (  # noqa: E402
    component_volumes,
    fieldregion_profile,
    import_named_fieldregion_profile,
    monitor_electric,
)
from build_nonuniform_complex_yee_jacobian import index_detail  # noqa: E402
from run_production_combined_adfd_smoke import (  # noqa: E402
    FIELD_REGION,
    FREQUENCY_HZ,
    open_fdtd,
    reconstruct_fieldregion_only_cw,
    relative,
    run_adjoint,
)


DEFAULT_RAW = pq.DEFAULT_RAW
FD_CASES = pq.FD_CASES
CORNER_FREE_HALF_Y_M = 18.0e-6
CORNER_FREE_FD_CASES = {
    0.10: (
        "corner_free_y18_width_7p9_edge25_forward",
        "corner_free_y18_width_8p1_edge25_forward",
    ),
    0.05: (
        "corner_free_y18_width_7p95_edge25_forward",
        "corner_free_y18_width_8p05_edge25_forward",
    ),
}
WAVELENGTH_M = pq.WAVELENGTH_M
SPEED_OF_LIGHT_M_S = 299792458.0
AIR_EPSILON = pq.AIR_EPSILON
AU_HALF_Y_M = pq.AU_HALF_Y_M
AU_Z_MIN_M = pq.AU_Z_MIN_M
AU_Z_MAX_M = pq.AU_Z_MAX_M

# The smooth weight is fixed in space and remains at least 150 nm below the Au
# film. Two Gaussian x lobes straddle the moving edges; a compact sin^2 z
# window avoids a Boolean cut through the Yee grid while retaining exact zero
# support outside the air slab.
ROI = {
    "x_lobe_center_abs_m": 8.0e-6,
    "x_sigma_m": 1.5e-6,
    "y_sigma_m": 4.0e-6,
    "z_min_m": -0.45e-6,
    "z_max_m": -0.10e-6,
}


def component_coordinates(
    grid: dict[str, np.ndarray], component: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = [np.asarray(grid[axis], float).copy() for axis in "xyz"]
    coordinates[component] += np.asarray(
        grid[f"delta_{'xyz'[component]}"], float
    )
    return tuple(coordinates)


def fixed_air_objective_and_source(
    electric: np.ndarray,
    grid: dict[str, np.ndarray],
) -> tuple[float, np.ndarray, dict[str, object]]:
    """Return 0.5*eps0*integral_ROI |E|^2 dV and dF/dE*."""

    expected = (
        np.asarray(grid["x"]).size,
        np.asarray(grid["y"]).size,
        np.asarray(grid["z"]).size,
        1,
        3,
    )
    if electric.shape != expected:
        raise ValueError(f"electric shape {electric.shape} != {expected}")
    volumes = component_volumes(grid)
    source = np.zeros_like(electric)
    component_values: dict[str, float] = {}
    component_counts: dict[str, int] = {}
    component_bounds: dict[str, object] = {}
    for component, label in enumerate("xyz"):
        x, y, z = component_coordinates(grid, component)
        wx = np.exp(
            -0.5
            * ((np.abs(x) - ROI["x_lobe_center_abs_m"]) / ROI["x_sigma_m"])
            ** 2
        )
        wy = np.exp(-0.5 * (y / ROI["y_sigma_m"]) ** 2)
        z_fraction = (z - ROI["z_min_m"]) / (
            ROI["z_max_m"] - ROI["z_min_m"]
        )
        wz = np.where(
            (z_fraction > 0.0) & (z_fraction < 1.0),
            np.sin(np.pi * z_fraction) ** 2,
            0.0,
        )
        spatial_weight = (
            wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
        )
        support = spatial_weight > 0.0
        if not np.any(support):
            raise RuntimeError(f"empty fixed external ROI on {label} grid")
        field = np.asarray(electric[..., 0, component], complex)
        weight = (
            0.5
            * EPS0
            * np.asarray(volumes[component])
            * spatial_weight
        )
        value = float(np.sum(weight * np.abs(field) ** 2))
        source[..., 0, component] = weight * field
        component_values[label] = value
        component_counts[label] = int(np.count_nonzero(support))
        indices = np.where(support)
        component_bounds[label] = {
            "x_m": [float(x[indices[0].min()]), float(x[indices[0].max()])],
            "y_m": [float(y[indices[1].min()]), float(y[indices[1].max()])],
            "z_m": [float(z[indices[2].min()]), float(z[indices[2].max()])],
        }
    total = float(sum(component_values.values()))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError(f"invalid fixed external objective {total}")
    metadata = {
        "definition": (
            "0.5*eps0*sum_c integral_fixed_air_ROI "
            "w_fixed(x,y,z)*|E_c|^2 dV"
        ),
        "weight": (
            "two fixed Gaussian x lobes times Gaussian y times compact "
            "sin^2 z window"
        ),
        "meaning": "fixed external electric-field-energy proxy; not total EM energy",
        "units": "J_proxy",
        "requested_bounds": ROI,
        "component_value_J_proxy": component_values,
        "component_sample_count": component_counts,
        "component_realized_bounds_m": component_bounds,
        "minimum_vertical_clearance_from_Au_m": AU_Z_MIN_M - ROI["z_max_m"],
        "moving_domain_or_direct_material_term": False,
    }
    return total, source, metadata


def fitted_au_epsilon(fdtd: object, object_name: str) -> tuple[complex, str]:
    material_name = str(fdtd.getnamed(object_name, "material"))
    wavelength_start = float(fdtd.getglobalsource("wavelength start"))
    wavelength_stop = float(fdtd.getglobalsource("wavelength stop"))
    frequencies = SPEED_OF_LIGHT_M_S / np.asarray(
        [wavelength_start, wavelength_stop]
    )
    fitted_index = complex(
        np.asarray(
            fdtd.getfdtdindex(
                material_name,
                np.asarray([FREQUENCY_HZ]),
                float(np.min(frequencies)),
                float(np.max(frequencies)),
                1,
            )
        ).reshape(-1)[0]
    )
    return fitted_index**2, material_name


def official_center_depth_integral(
    forward_fields,
    adjoint_fields,
    *,
    half_width_m: float,
    epsilon_au: complex,
    n_points: int,
    half_y_m: float = AU_HALF_Y_M,
) -> dict[str, object]:
    """Replicate the bundled v261 Polygon 3-D center-plane/depth rule."""

    if n_points < 3 or n_points % 2 == 0:
        raise ValueError("n_points must be an odd integer >=3")
    y = np.linspace(-half_y_m, half_y_m, n_points)
    z = np.full_like(y, 0.5 * (AU_Z_MIN_M + AU_Z_MAX_M))
    wavelength = np.full_like(y, WAVELENGTH_M)
    total = 0.0
    faces = {}
    all_finite = True
    for label, x, normal_x in (
        ("x_min", -half_width_m, -1.0),
        ("x_max", half_width_m, 1.0),
    ):
        xx = np.full_like(y, x)
        ef = np.asarray(forward_fields.getfield(xx, y, z, wavelength), complex)
        df = np.asarray(forward_fields.getDfield(xx, y, z, wavelength), complex)
        ea = np.asarray(adjoint_fields.getfield(xx, y, z, wavelength), complex)
        da = np.asarray(adjoint_fields.getDfield(xx, y, z, wavelength), complex)
        if any(value.shape != (n_points, 3) for value in (ef, df, ea, da)):
            raise RuntimeError("unexpected vectorized center-depth field shape")
        all_finite = all_finite and bool(
            all(np.all(np.isfinite(value)) for value in (ef, df, ea, da))
        )
        # x-normal boundary: tangential components are y,z and normal D is x.
        ef_parallel = np.array(ef, copy=True)
        ea_parallel = np.array(ea, copy=True)
        ef_parallel[:, 0] = 0.0
        ea_parallel[:, 0] = 0.0
        df_perp = np.zeros_like(df)
        da_perp = np.zeros_like(da)
        df_perp[:, 0] = df[:, 0]
        da_perp[:, 0] = da[:, 0]
        kernel = (
            2.0
            * EPS0
            * (epsilon_au - AIR_EPSILON)
            * np.sum(ef_parallel * ea_parallel, axis=-1)
            + (1.0 / AIR_EPSILON - 1.0 / epsilon_au)
            / EPS0
            * np.sum(df_perp * da_perp, axis=-1)
        )
        value = float(np.trapezoid(np.real(kernel), x=y)) * (
            AU_Z_MAX_M - AU_Z_MIN_M
        )
        # Increasing half-width moves both faces along their outward normals;
        # each normal velocity is +1 m/m. normal_x is retained in provenance.
        faces[label] = {
            "normal": [normal_x, 0.0, 0.0],
            "normal_velocity_m_per_m": 1.0,
            "derivative_J_proxy_per_m": value,
        }
        total += value
    return {
        "n_points_per_edge": n_points,
        "dy_m": float(y[1] - y[0]),
        "z_center_m": float(z[0]),
        "fixed_depth_m": AU_Z_MAX_M - AU_Z_MIN_M,
        "half_y_m": half_y_m,
        "faces": faces,
        "total_J_proxy_per_m": total,
        "all_finite": all_finite,
    }


def midpoint_surface_integral(
    forward_fields,
    adjoint_fields,
    *,
    half_width_m: float,
    half_y_m: float,
    epsilon_au: complex,
    dy_m: float,
    dz_m: float,
) -> dict[str, object]:
    """Integrate the complete 3-D moving face without sampling its corners.

    Unlike the bundled extruded-polygon shortcut, this rule integrates both
    in-plane and film-depth directions. Midpoints avoid assigning a finite
    trapezoid weight to a mathematically singular sharp-metal vertex.
    """

    ny = int(round((2.0 * half_y_m) / dy_m))
    nz = int(round((AU_Z_MAX_M - AU_Z_MIN_M) / dz_m))
    if ny < 2 or nz < 1:
        raise ValueError("midpoint surface quadrature is too coarse")
    dy = 2.0 * half_y_m / ny
    dz = (AU_Z_MAX_M - AU_Z_MIN_M) / nz
    y = -half_y_m + (np.arange(ny) + 0.5) * dy
    z = AU_Z_MIN_M + (np.arange(nz) + 0.5) * dz
    yy, zz = np.meshgrid(y, z, indexing="ij")
    wavelength = np.full_like(yy, WAVELENGTH_M)
    total = 0.0
    faces: dict[str, object] = {}
    all_finite = True
    for label, x, normal_x in (
        ("x_min", -half_width_m, -1.0),
        ("x_max", half_width_m, 1.0),
    ):
        xx = np.full_like(yy, x)
        ef = np.asarray(
            forward_fields.getfield(xx, yy, zz, wavelength), complex
        )
        df = np.asarray(
            forward_fields.getDfield(xx, yy, zz, wavelength), complex
        )
        ea = np.asarray(
            adjoint_fields.getfield(xx, yy, zz, wavelength), complex
        )
        da = np.asarray(
            adjoint_fields.getDfield(xx, yy, zz, wavelength), complex
        )
        expected = (*yy.shape, 3)
        if any(value.shape != expected for value in (ef, df, ea, da)):
            raise RuntimeError("unexpected midpoint surface field shape")
        all_finite = all_finite and bool(
            all(np.all(np.isfinite(value)) for value in (ef, df, ea, da))
        )
        tangential = ef[..., 1] * ea[..., 1] + ef[..., 2] * ea[..., 2]
        normal_d = df[..., 0] * da[..., 0]
        kernel = (
            2.0 * EPS0 * (epsilon_au - AIR_EPSILON) * tangential
            + (1.0 / AIR_EPSILON - 1.0 / epsilon_au)
            / EPS0
            * normal_d
        )
        value = float(np.sum(np.real(kernel)) * dy * dz)
        faces[label] = {
            "normal": [normal_x, 0.0, 0.0],
            "normal_velocity_m_per_m": 1.0,
            "derivative_J_proxy_per_m": value,
        }
        total += value
    return {
        "rule": "two-dimensional midpoint surface quadrature",
        "ny": ny,
        "nz": nz,
        "dy_m": dy,
        "dz_m": dz,
        "half_y_m": half_y_m,
        "y_endpoints_sampled": False,
        "z_endpoints_sampled": False,
        "faces": faces,
        "total_J_proxy_per_m": total,
        "all_finite": all_finite,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--baseline-case")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 6")
    parser.add_argument(
        "--corner-free-control",
        action="store_true",
        help=(
            "use Au y=+-18 um forward controls so the fixed sharp corners "
            "are outside the active source/adjoint region"
        ),
    )
    parser.add_argument("--fd-precheck-only", action="store_true")
    parser.add_argument("--resume-completed-adjoint", action="store_true")
    args = parser.parse_args()
    fd_cases = CORNER_FREE_FD_CASES if args.corner_free_control else FD_CASES
    au_half_y_m = CORNER_FREE_HALF_Y_M if args.corner_free_control else AU_HALF_Y_M
    baseline_case = args.baseline_case or (
        "corner_free_y18_width_8p0_edge25_forward"
        if args.corner_free_control
        else "sharp_width_8p0_edge25_forward"
    )
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not (
        args.fd_precheck_only or args.resume_completed_adjoint
    ):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "au_sharp_interface_external_field_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_AU_SHARP_INTERFACE_EXTERNAL_FIELD_ADJOINT",
        "passed": False,
        "Maxwell_forward_solves_this_invocation": 0,
        "Maxwell_adjoint_solves_this_invocation": 0,
        "thermal_solves": 0,
        "optimizer_started": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "finite_difference_used_to_fit_AD": False,
        "CPU_FDTD_fallback": False,
    }
    fdtd = None
    try:
        projects: dict[str, Path] = {}
        forward_results: dict[str, dict[str, object]] = {}
        all_cases = {baseline_case}
        for pair in fd_cases.values():
            all_cases.update(pair)
        for name in sorted(all_cases):
            projects[name], forward_results[name] = pq.checked_project(
                args.raw_root / name
            )

        fdtd, audit, runtime = open_fdtd(args.gpu_device)
        objective_cases: dict[str, object] = {}
        baseline_electric = None
        baseline_grid = None
        baseline_source = None
        baseline_roi = None
        for name in sorted(all_cases):
            fdtd.load(str(projects[name]))
            fdtd.runanalysis(PABS_GROUP)
            electric, grid = monitor_electric(fdtd, PABS_FIELD)
            objective, source, roi_meta = fixed_air_objective_and_source(
                electric, grid
            )
            objective_cases[name] = {
                "half_width_um": forward_results[name]["shape_parameter"][
                    "value_um"
                ],
                "objective_J_proxy": objective,
                "component_value_J_proxy": roi_meta[
                    "component_value_J_proxy"
                ],
                "project": {
                    "path": str(projects[name]),
                    "size_bytes": projects[name].stat().st_size,
                    "sha256": pq.sha256(projects[name]),
                },
            }
            if name == baseline_case:
                baseline_electric = np.array(electric, copy=True)
                baseline_grid = {
                    key: np.array(value, copy=True) for key, value in grid.items()
                }
                baseline_source = np.array(source, copy=True)
                baseline_roi = roi_meta
            del electric, source
        if baseline_electric is None or baseline_grid is None or baseline_source is None:
            raise RuntimeError("baseline external objective was not captured")

        finite_differences = {}
        for h_um, (minus_name, plus_name) in fd_cases.items():
            minus = float(objective_cases[minus_name]["objective_J_proxy"])
            plus = float(objective_cases[plus_name]["objective_J_proxy"])
            derivative = (plus - minus) / (2.0 * h_um)
            finite_differences[f"h_{h_um:g}_um"] = {
                "h_um": h_um,
                "minus_case": minus_name,
                "plus_case": plus_name,
                "minus_objective_J_proxy": minus,
                "plus_objective_J_proxy": plus,
                "derivative_J_proxy_per_um": derivative,
            }
        fd_step_change = relative(
            finite_differences["h_0.05_um"]["derivative_J_proxy_per_um"],
            finite_differences["h_0.1_um"]["derivative_J_proxy_per_um"],
        )
        baseline_objective = float(
            objective_cases[baseline_case]["objective_J_proxy"]
        )
        strong_fd = float(
            finite_differences["h_0.05_um"]["derivative_J_proxy_per_um"]
        )
        strong_step_fraction = abs(strong_fd) * 0.05 / baseline_objective
        precheck_pass = bool(
            fd_step_change < 0.01
            and strong_step_fraction > 1.0e-6
            and baseline_roi["minimum_vertical_clearance_from_Au_m"] >= 0.15e-6
        )
        result.update(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": (
                    "fixed external air-field objective for isolated Au boundary-"
                    "kernel AD-FD; no P_Q, thermal, electrical, PTE, or optimization"
                ),
                "geometry_control": {
                    "Au_half_y_m": au_half_y_m,
                    "fixed_sharp_corners_y_m": [-au_half_y_m, au_half_y_m],
                    "corner_free_active_face": args.corner_free_control,
                    "purpose": (
                        "isolate the smooth x-normal face by moving the fixed "
                        "y-end corners outside the active illumination and "
                        "adjoint support"
                        if args.corner_free_control
                        else "legacy finite sharp-corner rectangle"
                    ),
                },
                "objective": baseline_roi,
                "objective_cases": objective_cases,
                "finite_difference": finite_differences,
                "FD_step_plateau_relative_change": fd_step_change,
                "strong_h0p05_step_fraction_of_baseline": strong_step_fraction,
                "precheck_gates": {
                    "FD_h0p1_to_h0p05_change_lt_1pct": fd_step_change < 0.01,
                    "strong_step_fraction_gt_1e_6": strong_step_fraction > 1.0e-6,
                    "fixed_ROI_vertical_clearance_ge_150nm": baseline_roi[
                        "minimum_vertical_clearance_from_Au_m"
                    ]
                    >= 0.15e-6,
                },
            }
        )
        if args.fd_precheck_only:
            result["status"] = (
                "READY_AU_EXTERNAL_FIELD_ADJOINT_PRECHECK"
                if precheck_pass
                else "FAILED_AU_EXTERNAL_FIELD_FD_PRECHECK"
            )
            result["precheck_passed"] = precheck_pass
            result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
            print(json.dumps(result, indent=2, default=str))
            return 0 if precheck_pass else 2
        if not precheck_pass:
            raise RuntimeError("fixed external objective FD precheck did not pass")

        # Reload the baseline before creating the adjoint template.
        fdtd.load(str(projects[baseline_case]))
        fdtd.runanalysis(PABS_GROUP)
        forward, grid = monitor_electric(fdtd, PABS_FIELD)
        objective_check, native_source, _ = fixed_air_objective_and_source(
            forward, grid
        )
        if relative(objective_check, baseline_objective) >= 1.0e-12:
            raise RuntimeError("baseline objective changed on reload")
        detail = index_detail(fdtd)
        epsilon = np.stack(
            [detail[f"epsilon_{component}"] for component in "xyz"], axis=-1
        )[..., None, :]
        if epsilon.shape != forward.shape:
            raise RuntimeError("forward E/index component grids do not match")
        epsilon_au, material_name = fitted_au_epsilon(
            fdtd, str(forward_results[baseline_case]["material"]["name"])
        )
        epsilon_fit_error = relative(epsilon_au, pq.AU_EPSILON)

        profile, profile_scale = fieldregion_profile(native_source)
        original_amplitude = float(fdtd.getnamed(audit.SOURCE_NAME, "amplitude"))
        template = output / "au_external_field_adjoint_template.fsp"
        adjoint_project = output / "au_external_field_adjoint_gpu.fsp"
        if args.resume_completed_adjoint:
            if not template.is_file() or not adjoint_project.is_file():
                raise FileNotFoundError("resume requires completed template and adjoint FSP")
            fdtd.load(str(adjoint_project))
            base_amplitude = float(fdtd.getnamed(FIELD_REGION, "base amplitude"))
            imported = np.asarray(
                fdtd.getresult(FIELD_REGION, "source profile")["E"], complex
            )
            roundtrip = float(np.max(np.abs(imported - profile)))
            del imported
            fdtd.cwnorm(1)
            electric_first, adjoint_grid = monitor_electric(fdtd, PABS_FIELD)
            fdtd.cwnorm(2)
            electric_average, average_grid = monitor_electric(fdtd, PABS_FIELD)
            average_mismatch = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(adjoint_grid[key])
                            - np.asarray(average_grid[key])
                        )
                    )
                )
                for key in ("x", "y", "z", "delta_x", "delta_y", "delta_z")
            )
            electric, normalization = reconstruct_fieldregion_only_cw(
                electric_first, electric_average
            )
            normalization["grid_mismatch_m"] = average_mismatch
            adjoint = {
                "electric": electric,
                "grid": adjoint_grid,
                "resources": {"reused_completed_GPU_adjoint": True},
                "resource_used": "REUSED_COMPLETED_GPU_ADJOINT",
                "solver_mode": "GPU",
                "named_source_normalization": normalization,
                "log_audit": audit.log_audit(output),
                "wall_s": 0.0,
                "project": {
                    "path": str(adjoint_project),
                    "size_bytes": adjoint_project.stat().st_size,
                    "sha256": pq.sha256(adjoint_project),
                },
                "reused_without_new_Maxwell_solve": True,
            }
        else:
            fdtd.switchtolayout()
            fdtd.setnamed(audit.SOURCE_NAME, "amplitude", 0.0)
            fdtd.setnamed(audit.SOURCE_NAME, "enabled", True)
            pq.add_adjoint_fieldregion(fdtd, grid)
            roundtrip = import_named_fieldregion_profile(
                fdtd, FIELD_REGION, grid, profile
            )
            base_amplitude = float(fdtd.getnamed(FIELD_REGION, "base amplitude"))
            fdtd.save(str(template))
            adjoint = run_adjoint(
                fdtd,
                audit,
                runtime,
                template=template,
                project=adjoint_project,
            )
            result["Maxwell_adjoint_solves_this_invocation"] = 1

        mismatch = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(grid[key])
                        - np.asarray(adjoint["grid"][key])
                    )
                )
            )
            for key in ("x", "y", "z", "delta_x", "delta_y", "delta_z")
        )
        if mismatch >= 2.0e-18:
            raise RuntimeError(f"forward/adjoint grid mismatch {mismatch:.3e} m")
        forward_fields = pq.build_nointerp_fields(forward, epsilon, grid)
        scaled_adjoint = np.asarray(adjoint["electric"]) * (
            profile_scale / base_amplitude
        )
        adjoint_fields = pq.build_nointerp_fields(scaled_adjoint, epsilon, grid)
        center_depth_quadratures = [
            official_center_depth_integral(
                forward_fields,
                adjoint_fields,
                half_width_m=8.0e-6,
                epsilon_au=epsilon_au,
                n_points=n_points,
                half_y_m=au_half_y_m,
            )
            for n_points in (201, 401, 801, 1601)
        ]
        midpoint_center_depth_quadratures = [
            midpoint_surface_integral(
                forward_fields,
                adjoint_fields,
                half_width_m=8.0e-6,
                half_y_m=au_half_y_m,
                epsilon_au=epsilon_au,
                dy_m=dy_m,
                dz_m=AU_Z_MAX_M - AU_Z_MIN_M,
            )
            for dy_m in (100e-9, 50e-9, 25e-9, 12.5e-9)
        ]
        surface_quadratures = [
            midpoint_surface_integral(
                forward_fields,
                adjoint_fields,
                half_width_m=8.0e-6,
                half_y_m=au_half_y_m,
                epsilon_au=epsilon_au,
                dy_m=dy_m,
                dz_m=dz_m,
            )
            for dy_m, dz_m in (
                (100e-9, 10e-9),
                (50e-9, 5e-9),
                (25e-9, 2.5e-9),
                (12.5e-9, 1.25e-9),
            )
        ]
        selected = surface_quadratures[-1]
        ad_per_um = float(selected["total_J_proxy_per_m"]) * 1.0e-6
        center_depth_change = relative(
            center_depth_quadratures[-1]["total_J_proxy_per_m"],
            center_depth_quadratures[-2]["total_J_proxy_per_m"],
        )
        midpoint_center_depth_change = relative(
            midpoint_center_depth_quadratures[-1]["total_J_proxy_per_m"],
            midpoint_center_depth_quadratures[-2]["total_J_proxy_per_m"],
        )
        surface_quadrature_change = relative(
            surface_quadratures[-1]["total_J_proxy_per_m"],
            surface_quadratures[-2]["total_J_proxy_per_m"],
        )
        comparisons = {}
        for key, row in finite_differences.items():
            fd_value = float(row["derivative_J_proxy_per_um"])
            comparisons[key] = {
                "FD_J_proxy_per_um": fd_value,
                "AD_J_proxy_per_um": ad_per_um,
                "relative_error": relative(ad_per_um, fd_value),
                "sign_agrees": ad_per_um * fd_value > 0.0,
            }
        strong = comparisons["h_0.05_um"]
        passed = bool(
            strong["relative_error"] < 0.01
            and strong["sign_agrees"]
            and surface_quadrature_change < 5.0e-3
            and roundtrip == 0.0
            and mismatch < 2.0e-18
            and all(bool(row["all_finite"]) for row in surface_quadratures)
            and float(adjoint["log_audit"]["final_auto_shutoff"]) < 1.0e-5
        )
        result.update(
            {
                "status": (
                    "VALIDATED_AU_SHARP_INTERFACE_EXTERNAL_FIELD_BOUNDARY_KERNEL"
                    if passed
                    else "FAILED_AU_SHARP_INTERFACE_EXTERNAL_FIELD_BOUNDARY_KERNEL_ADFD"
                ),
                "passed": passed,
                "precheck_passed": True,
                "material_readback": {
                    "material_name": material_name,
                    "requested_epsilon": [pq.AU_EPSILON.real, pq.AU_EPSILON.imag],
                    "fitted_epsilon": [epsilon_au.real, epsilon_au.imag],
                    "relative_error": epsilon_fit_error,
                },
                "source": {
                    "method": "fixed-air-ROI native-Yee FieldRegion vector source",
                    "profile_scale": profile_scale,
                    "fieldregion_base_amplitude": base_amplitude,
                    "source_profile_roundtrip_max_abs_error": roundtrip,
                    "forward_Gaussian_original_amplitude": original_amplitude,
                    "forward_Gaussian_adjoint_amplitude": 0.0,
                    "template": {
                        "path": str(template),
                        "size_bytes": template.stat().st_size,
                        "sha256": pq.sha256(template),
                    },
                },
                "boundary_quadrature_method_selected": (
                    "two-dimensional midpoint integration over the full "
                    "moving y-z face"
                ),
                "official_center_depth_endpoint_quadrature": (
                    center_depth_quadratures
                ),
                "official_center_depth_final_relative_change": (
                    center_depth_change
                ),
                "midpoint_center_depth_quadrature": (
                    midpoint_center_depth_quadratures
                ),
                "midpoint_center_depth_final_relative_change": (
                    midpoint_center_depth_change
                ),
                "full_surface_midpoint_quadrature": surface_quadratures,
                "full_surface_midpoint_final_relative_change": (
                    surface_quadrature_change
                ),
                # Backward-compatible aliases refer to the selected rule.
                "boundary_quadrature": surface_quadratures,
                "boundary_quadrature_final_relative_change": (
                    surface_quadrature_change
                ),
                "AD_FD_comparison": comparisons,
                "forward_adjoint_maximum_coordinate_mismatch_m": mismatch,
                "adjoint": {
                    key: value
                    for key, value in adjoint.items()
                    if key not in {"electric", "grid"}
                },
                "gates": {
                    "strong_h0p05_relative_error_lt_1pct": strong[
                        "relative_error"
                    ]
                    < 0.01,
                    "strong_sign_agrees": strong["sign_agrees"],
                    "boundary_quadrature_change_lt_0p5pct": surface_quadrature_change
                    < 5.0e-3,
                    "source_roundtrip_exact": roundtrip == 0.0,
                    "coordinate_mismatch_lt_2e_18_m": mismatch < 2.0e-18,
                    "adjoint_auto_shutoff_lt_1e_5": float(
                        adjoint["log_audit"]["final_auto_shutoff"]
                    )
                    < 1.0e-5,
                },
                "moving_domain_or_direct_material_term_used": False,
                "gray_Au_air_material_used": False,
                "clipping_smoothing_gain_or_rescaling": False,
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
