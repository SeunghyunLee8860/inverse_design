#!/usr/bin/env python3
"""AD--FD root-cause control for a fully smooth 3-D scalar-Au ellipsoid.

The production 50-nm film has a lateral boundary joined to top and bottom
faces by sharp rims.  This mathematical control removes all such rims while
retaining a lossy, high-contrast, exact scalar Au endpoint.  It certifies only
the field-mediated derivative of a fixed external-field objective with
respect to the ellipsoid x semi-axis.  It is not a physical electrode model
and cannot promote Au topology optimization by itself.
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
DEFAULT_RAW = Path("/data/seunghyun/tairte4/raw_artifacts/au_topology_validation")
BASELINE_CASE = "pva5_smooth3d_ellipsoid_a8_b18_c1_forward"
FD_CASES = {
    0.10: (
        "pva5_smooth3d_ellipsoid_a7p9_b18_c1_forward",
        "pva5_smooth3d_ellipsoid_a8p1_b18_c1_forward",
    ),
    0.05: (
        "pva5_smooth3d_ellipsoid_a7p95_b18_c1_forward",
        "pva5_smooth3d_ellipsoid_a8p05_b18_c1_forward",
    ),
}
A_M = 8.0e-6
B_M = 18.0e-6
C_M = 1.0e-6
Z_CENTER_M = 0.075e-6
Z_MIN_M = Z_CENTER_M - C_M
Z_MAX_M = Z_CENTER_M + C_M
ROI = {
    "x_lobe_center_abs_m": A_M,
    "x_sigma_m": 1.5e-6,
    "y_sigma_m": 4.0e-6,
    "z_min_m": -1.80e-6,
    "z_max_m": -1.10e-6,
}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage12 = load("au_smooth3d_stage12", STAGE12)


def result_half_width_um(result: dict[str, object]) -> float:
    shape = result.get("shape_parameter", {})
    if "value_um" in shape:
        return float(shape["value_um"])
    if "value_m" in shape:
        return float(shape["value_m"]) * 1.0e6
    raise KeyError("shape_parameter requires value_um or value_m")


def checked_ellipsoid_project(case_dir: Path) -> tuple[Path, dict[str, object]]:
    """Accept ordinary results or a separately preserved postprocess recovery."""

    recovered = case_dir / "case_result_recovered.json"
    result_path = recovered if recovered.is_file() else case_dir / "case_result.json"
    result = json.loads(result_path.read_text())
    if not bool(result.get("passed", False)):
        raise RuntimeError(f"forward case did not pass: {result_path}")
    project = case_dir / "complex_material_control.fsp"
    if not project.is_file():
        raise FileNotFoundError(project)
    stored = {
        str(Path(row["path"]).resolve()): row
        for row in result.get("raw_artifacts", [])
    }.get(str(project.resolve()))
    if stored is None:
        raise RuntimeError(f"project is absent from raw artifact manifest: {result_path}")
    if int(stored["size_bytes"]) != project.stat().st_size:
        raise RuntimeError(f"project size mismatch: {project}")
    if str(stored["sha256"]) != stage12.pq.sha256(project):
        raise RuntimeError(f"project SHA-256 mismatch: {project}")
    result["selected_result_path"] = str(result_path)
    return project, result


def maximum_grid_mismatch(left: dict, right: dict) -> float:
    return max(
        float(np.max(np.abs(np.asarray(left[key]) - np.asarray(right[key]))))
        for key in ("x", "y", "z", "delta_x", "delta_y", "delta_z")
    )


def evaluate_vector(function, x, y, z):
    wavelength = np.full_like(np.asarray(x, float), stage12.WAVELENGTH_M)
    value = np.asarray(function(x, y, z, wavelength), complex)
    expected = (*np.asarray(x).shape, 3)
    if value.shape == (3, *np.asarray(x).shape):
        value = np.moveaxis(value, 0, -1)
    if value.shape != expected:
        raise RuntimeError(f"unexpected vector field shape {value.shape} != {expected}")
    return value


def ellipsoid_volume_shape_derivative_quadrature(
    *, mu_order: int, phi_count: int
) -> float:
    """Return integral_surface (dr/da dot n) dA for a geometry-only test."""

    mu, w_mu = np.polynomial.legendre.leggauss(mu_order)
    phi = (np.arange(phi_count, dtype=float) + 0.5) * (2.0 * np.pi / phi_count)
    mm, pp = np.meshgrid(mu, phi, indexing="ij")
    sin_theta = np.sqrt(np.maximum(1.0 - mm**2, np.finfo(float).tiny))
    dr_dmu = np.stack(
        (
            -A_M * mm / sin_theta * np.cos(pp),
            -B_M * mm / sin_theta * np.sin(pp),
            np.full_like(mm, C_M),
        ),
        axis=-1,
    )
    dr_dphi = np.stack(
        (
            -A_M * sin_theta * np.sin(pp),
            B_M * sin_theta * np.cos(pp),
            np.zeros_like(mm),
        ),
        axis=-1,
    )
    surface_vector = np.cross(dr_dphi, dr_dmu)
    dr_da = np.stack(
        (sin_theta * np.cos(pp), np.zeros_like(mm), np.zeros_like(mm)), axis=-1
    )
    return float(
        np.sum(
            np.sum(dr_da * surface_vector, axis=-1)
            * w_mu[:, None]
            * (2.0 * np.pi / phi_count)
        )
    )


def smooth_ellipsoid_surface_integral(
    forward_fields,
    adjoint_fields,
    *,
    epsilon_au: complex,
    mu_order: int,
    phi_count: int,
) -> dict[str, object]:
    """Endpoint-free tensor quadrature on the exact smooth ellipsoid."""

    if mu_order < 2 or phi_count < 8:
        raise ValueError("ellipsoid quadrature is too coarse")
    mu, w_mu = np.polynomial.legendre.leggauss(mu_order)
    phi = (np.arange(phi_count, dtype=float) + 0.5) * (2.0 * np.pi / phi_count)
    w_phi = 2.0 * np.pi / phi_count
    mm, pp = np.meshgrid(mu, phi, indexing="ij")
    sin_theta = np.sqrt(np.maximum(1.0 - mm**2, np.finfo(float).tiny))

    x = A_M * sin_theta * np.cos(pp)
    y = B_M * sin_theta * np.sin(pp)
    z = Z_CENTER_M + C_M * mm
    dr_dmu = np.stack(
        (
            -A_M * mm / sin_theta * np.cos(pp),
            -B_M * mm / sin_theta * np.sin(pp),
            np.full_like(mm, C_M),
        ),
        axis=-1,
    )
    dr_dphi = np.stack(
        (
            -A_M * sin_theta * np.sin(pp),
            B_M * sin_theta * np.cos(pp),
            np.zeros_like(mm),
        ),
        axis=-1,
    )
    # dphi x dmu points outward (for example +x at mu=0, phi=0).
    surface_vector = np.cross(dr_dphi, dr_dmu)
    jacobian = np.linalg.norm(surface_vector, axis=-1)
    normal = surface_vector / jacobian[..., None]
    dr_da = np.stack(
        (sin_theta * np.cos(pp), np.zeros_like(mm), np.zeros_like(mm)), axis=-1
    )
    normal_velocity = np.sum(dr_da * normal, axis=-1)
    surface_weight = w_mu[:, None] * w_phi * jacobian

    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    flat_z = z.reshape(-1)
    flat_normal = normal.reshape(-1, 3)
    flat_velocity = normal_velocity.reshape(-1)
    flat_weight = surface_weight.reshape(-1)
    positive = 0.0
    negative = 0.0
    tangential_total = 0.0
    normal_total = 0.0
    batch_size = 8192
    all_finite = True
    for start in range(0, flat_x.size, batch_size):
        stop = min(start + batch_size, flat_x.size)
        section = slice(start, stop)
        ef = evaluate_vector(
            forward_fields.getfield,
            flat_x[section],
            flat_y[section],
            flat_z[section],
        )
        df = evaluate_vector(
            forward_fields.getDfield,
            flat_x[section],
            flat_y[section],
            flat_z[section],
        )
        ea = evaluate_vector(
            adjoint_fields.getfield,
            flat_x[section],
            flat_y[section],
            flat_z[section],
        )
        da = evaluate_vector(
            adjoint_fields.getDfield,
            flat_x[section],
            flat_y[section],
            flat_z[section],
        )
        finite = all(np.all(np.isfinite(v)) for v in (ef, df, ea, da))
        all_finite = all_finite and finite
        if not finite:
            raise RuntimeError("non-finite smooth-ellipsoid boundary field")
        local_normal = flat_normal[section]
        ef_parallel = ef - np.sum(ef * local_normal, axis=-1)[:, None] * local_normal
        ea_parallel = ea - np.sum(ea * local_normal, axis=-1)[:, None] * local_normal
        df_normal = np.sum(df * local_normal, axis=-1)
        da_normal = np.sum(da * local_normal, axis=-1)
        tangential_kernel = np.real(
            2.0
            * stage12.EPS0
            * (epsilon_au - stage12.AIR_EPSILON)
            * np.sum(ef_parallel * ea_parallel, axis=-1)
        )
        normal_kernel = np.real(
            (1.0 / stage12.AIR_EPSILON - 1.0 / epsilon_au)
            / stage12.EPS0
            * df_normal
            * da_normal
        )
        measure = flat_velocity[section] * flat_weight[section]
        tangential = tangential_kernel * measure
        normal_term = normal_kernel * measure
        weighted = tangential + normal_term
        tangential_total += float(np.sum(tangential))
        normal_total += float(np.sum(normal_term))
        positive += float(np.sum(weighted[weighted > 0.0]))
        negative += float(np.sum(weighted[weighted < 0.0]))

    total = tangential_total + normal_total
    return {
        "rule": "Gauss-Legendre(mu) x midpoint(phi) exact smooth ellipsoid",
        "mu_order": int(mu_order),
        "phi_count": int(phi_count),
        "sample_count": int(flat_x.size),
        "ellipsoid_semi_axes_m": [A_M, B_M, C_M],
        "center_z_m": Z_CENTER_M,
        "surface_has_edges_or_rims": False,
        "mu_endpoints_sampled": False,
        "phi_seam_sampled": False,
        "normal_velocity_range_m_per_m": [
            float(np.min(normal_velocity)),
            float(np.max(normal_velocity)),
        ],
        "surface_area_quadrature_m2": float(np.sum(surface_weight)),
        "tangential_E_term_J_proxy_per_m": tangential_total,
        "normal_D_term_J_proxy_per_m": normal_total,
        "positive_contribution_J_proxy_per_m": positive,
        "negative_contribution_J_proxy_per_m": negative,
        "total_J_proxy_per_m": total,
        "all_finite": bool(all_finite),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument("--fd-precheck-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "au_smooth3d_ellipsoid_boundary_adjoint_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_AU_SMOOTH3D_ELLIPSOID_BOUNDARY_ADJOINT",
        "passed": False,
        "Maxwell_forward_solves_this_invocation": 0,
        "Maxwell_adjoint_solves_this_invocation": 0,
        "CPU_FDTD_fallback": False,
        "thermal_solves": 0,
        "optimizer_started": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "finite_difference_used_to_fit_AD": False,
        "production_Au_optimization_permitted": False,
    }
    fdtd = None
    try:
        stage12.ROI = dict(ROI)
        stage12.AU_Z_MIN_M = Z_MIN_M
        stage12.AU_Z_MAX_M = Z_MAX_M
        projects: dict[str, Path] = {}
        forward_results: dict[str, dict[str, object]] = {}
        cases = {BASELINE_CASE}
        for pair in FD_CASES.values():
            cases.update(pair)
        for name in sorted(cases):
            projects[name], forward_results[name] = checked_ellipsoid_project(
                args.raw_root / name
            )

        fdtd, audit, runtime = stage12.open_fdtd(args.gpu_device)
        objective_cases: dict[str, object] = {}
        for name in sorted(cases):
            fdtd.load(str(projects[name]))
            fdtd.runanalysis(stage12.PABS_GROUP)
            electric, grid = stage12.monitor_electric(fdtd, stage12.PABS_FIELD)
            objective, _source, metadata = stage12.fixed_air_objective_and_source(
                electric, grid
            )
            objective_cases[name] = {
                "half_width_um": result_half_width_um(forward_results[name]),
                "objective_J_proxy": objective,
                "component_value_J_proxy": metadata["component_value_J_proxy"],
                "project": {
                    "path": str(projects[name]),
                    "size_bytes": projects[name].stat().st_size,
                    "sha256": stage12.pq.sha256(projects[name]),
                },
            }
        finite_differences = {}
        for h_um, (minus_name, plus_name) in FD_CASES.items():
            minus = float(objective_cases[minus_name]["objective_J_proxy"])
            plus = float(objective_cases[plus_name]["objective_J_proxy"])
            finite_differences[f"h_{h_um:g}_um"] = {
                "h_um": h_um,
                "minus_case": minus_name,
                "plus_case": plus_name,
                "minus_objective_J_proxy": minus,
                "plus_objective_J_proxy": plus,
                "derivative_J_proxy_per_um": (plus - minus) / (2.0 * h_um),
            }
        fd_step_change = stage12.relative(
            finite_differences["h_0.05_um"]["derivative_J_proxy_per_um"],
            finite_differences["h_0.1_um"]["derivative_J_proxy_per_um"],
        )
        baseline_objective = float(objective_cases[BASELINE_CASE]["objective_J_proxy"])
        strong_fd = float(
            finite_differences["h_0.05_um"]["derivative_J_proxy_per_um"]
        )
        strong_fraction = abs(strong_fd) * 0.05 / baseline_objective
        precheck_passed = bool(fd_step_change < 0.01 and strong_fraction > 1.0e-6)
        result.update(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": (
                    "fully smooth 3-D scalar-Au ellipsoid field-mediated "
                    "shape derivative; mathematical root-cause control only"
                ),
                "geometry": {
                    "representation": "exact_scalar_Au_smooth_3D_ellipsoid",
                    "semi_axes_m": [A_M, B_M, C_M],
                    "center_z_m": Z_CENTER_M,
                    "shape_parameter": "x semi-axis",
                    "surface_has_lateral_corners": False,
                    "surface_has_top_bottom_rims": False,
                    "physical_promotion_permitted": False,
                },
                "objective_contract": {
                    "definition": "fixed external air-field energy proxy",
                    "requested_bounds": ROI,
                    "minimum_vertical_clearance_from_ellipsoid_m": Z_MIN_M
                    - ROI["z_max_m"],
                    "moving_domain_or_direct_material_term": False,
                },
                "objective_cases": objective_cases,
                "finite_difference": finite_differences,
                "FD_step_plateau_relative_change": fd_step_change,
                "strong_h0p05_step_fraction_of_baseline": strong_fraction,
                "precheck_gates": {
                    "FD_h0p1_to_h0p05_change_lt_1pct": fd_step_change < 0.01,
                    "strong_step_fraction_gt_1e_6": strong_fraction > 1.0e-6,
                    "fixed_ROI_outside_ellipsoid": Z_MIN_M > ROI["z_max_m"],
                },
            }
        )
        if args.fd_precheck_only:
            result["status"] = (
                "READY_AU_SMOOTH3D_ELLIPSOID_BOUNDARY_ADJOINT_PRECHECK"
                if precheck_passed
                else "FAILED_AU_SMOOTH3D_ELLIPSOID_FD_PRECHECK"
            )
            result["precheck_passed"] = precheck_passed
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
            return 0 if precheck_passed else 2
        if not precheck_passed:
            raise RuntimeError("smooth-ellipsoid FD precheck did not pass")

        fdtd.load(str(projects[BASELINE_CASE]))
        fdtd.runanalysis(stage12.PABS_GROUP)
        forward, grid = stage12.monitor_electric(fdtd, stage12.PABS_FIELD)
        objective_check, native_source, objective_metadata = (
            stage12.fixed_air_objective_and_source(forward, grid)
        )
        if stage12.relative(objective_check, baseline_objective) >= 1.0e-12:
            raise RuntimeError("baseline objective changed on reload")
        detail = stage12.index_detail(fdtd)
        epsilon = np.stack(
            [detail[f"epsilon_{component}"] for component in "xyz"], axis=-1
        )[..., None, :]
        if epsilon.shape != forward.shape:
            raise RuntimeError("forward E/index component grids do not match")
        epsilon_au, material_name = stage12.fitted_au_epsilon(
            fdtd, "rho1_scalar_complex_block"
        )
        profile, profile_scale = stage12.fieldregion_profile(native_source)
        original_amplitude = float(fdtd.getnamed(audit.SOURCE_NAME, "amplitude"))
        fdtd.switchtolayout()
        fdtd.setnamed(audit.SOURCE_NAME, "amplitude", 0.0)
        fdtd.setnamed(audit.SOURCE_NAME, "enabled", True)
        stage12.pq.add_adjoint_fieldregion(fdtd, grid)
        roundtrip = stage12.import_named_fieldregion_profile(
            fdtd, stage12.FIELD_REGION, grid, profile
        )
        base_amplitude = float(fdtd.getnamed(stage12.FIELD_REGION, "base amplitude"))
        template = output / "au_smooth3d_ellipsoid_adjoint_template.fsp"
        adjoint_project = output / "au_smooth3d_ellipsoid_adjoint_gpu.fsp"
        fdtd.save(str(template))
        adjoint = stage12.run_adjoint(
            fdtd, audit, runtime, template=template, project=adjoint_project
        )
        result["Maxwell_adjoint_solves_this_invocation"] = 1
        mismatch = maximum_grid_mismatch(grid, adjoint["grid"])
        if mismatch >= 2.0e-18:
            raise RuntimeError(f"forward/adjoint grid mismatch {mismatch:.3e} m")
        forward_fields = stage12.pq.build_nointerp_fields(forward, epsilon, grid)
        scaled_adjoint = np.asarray(adjoint["electric"]) * (
            profile_scale / base_amplitude
        )
        adjoint_fields = stage12.pq.build_nointerp_fields(
            scaled_adjoint, epsilon, grid
        )
        quadratures = [
            smooth_ellipsoid_surface_integral(
                forward_fields,
                adjoint_fields,
                epsilon_au=epsilon_au,
                mu_order=mu_order,
                phi_count=phi_count,
            )
            for mu_order, phi_count in ((12, 48), (16, 64), (24, 96), (32, 128))
        ]
        quadrature_change = stage12.relative(
            quadratures[-1]["total_J_proxy_per_m"],
            quadratures[-2]["total_J_proxy_per_m"],
        )
        ad_per_um = float(quadratures[-1]["total_J_proxy_per_m"]) * 1.0e-6
        comparisons = {}
        for key, row in finite_differences.items():
            fd_value = float(row["derivative_J_proxy_per_um"])
            comparisons[key] = {
                "FD_J_proxy_per_um": fd_value,
                "AD_J_proxy_per_um": ad_per_um,
                "relative_error": stage12.relative(ad_per_um, fd_value),
                "sign_agrees": bool(ad_per_um * fd_value > 0.0),
            }
        strong = comparisons["h_0.05_um"]
        passed = bool(
            strong["relative_error"] < 0.01
            and strong["sign_agrees"]
            and quadrature_change < 5.0e-3
            and roundtrip == 0.0
            and mismatch < 2.0e-18
            and all(row["all_finite"] for row in quadratures)
            and float(adjoint["log_audit"]["final_auto_shutoff"]) < 1.0e-5
        )
        result.update(
            {
                "status": (
                    "VALIDATED_AU_SMOOTH3D_ELLIPSOID_BOUNDARY_ADJOINT"
                    if passed
                    else "FAILED_AU_SMOOTH3D_ELLIPSOID_BOUNDARY_ADJOINT"
                ),
                "passed": passed,
                "precheck_passed": True,
                "objective_metadata": objective_metadata,
                "material_readback": {
                    "material_name": material_name,
                    "requested_epsilon": [
                        stage12.pq.AU_EPSILON.real,
                        stage12.pq.AU_EPSILON.imag,
                    ],
                    "fitted_epsilon": [epsilon_au.real, epsilon_au.imag],
                    "relative_error": stage12.relative(
                        epsilon_au, stage12.pq.AU_EPSILON
                    ),
                },
                "source": {
                    "method": "fixed-air-ROI native-Yee FieldRegion vector source",
                    "profile_scale": profile_scale,
                    "fieldregion_base_amplitude": base_amplitude,
                    "source_profile_roundtrip_max_abs_error": roundtrip,
                    "forward_Gaussian_original_amplitude": original_amplitude,
                    "forward_Gaussian_adjoint_amplitude": 0.0,
                },
                "boundary_quadrature": quadratures,
                "boundary_quadrature_final_relative_change": quadrature_change,
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
                    "boundary_quadrature_change_lt_0p5pct": quadrature_change
                    < 5.0e-3,
                    "source_roundtrip_exact": roundtrip == 0.0,
                    "coordinate_mismatch_lt_2e_18_m": mismatch < 2.0e-18,
                    "adjoint_auto_shutoff_lt_1e_5": float(
                        adjoint["log_audit"]["final_auto_shutoff"]
                    )
                    < 1.0e-5,
                },
                "remaining_blocker_if_passed": (
                    "a realistic finite-thickness rounded Au endpoint and its "
                    "direct moving-material P_Q term remain uncertified"
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
    return 0 if bool(result.get("passed", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
