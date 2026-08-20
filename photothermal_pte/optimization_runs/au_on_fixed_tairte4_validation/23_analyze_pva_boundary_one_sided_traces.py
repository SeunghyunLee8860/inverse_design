#!/usr/bin/env python3
"""Resolve the PVA Au shape-gradient failure into normal-trace and z-rim terms.

This is a diagnostic, not a promoted gradient.  It reuses completed forward
and adjoint projects and performs no Maxwell solve.  The continuous boundary
kernel is sampled on the geometric ellipse, and on matched one-sided offsets
inside/outside the Au.  Contributions are retained separately by z bin and by
the tangential-E and normal-D terms.
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
STAGE17 = HERE / "17_run_au_smooth_ellipse_external_field_adjoint.py"
STAGE21 = HERE / "21_validate_fixed_geometry_au_material_adjoint.py"
DEFAULT_RAW = Path("/home/seunghyun/tairte4/raw_artifacts/au_topology_validation")
BASELINE_CASE = "pva5_fixedgrid_smooth_ellipse_a8p0_b18_edge50_forward"
HALF_X_M = 8.0e-6
HALF_Y_M = 18.0e-6


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage12 = load("au_trace_source", STAGE12)
stage17 = load("au_trace_geometry", STAGE17)
stage21 = load("au_trace_material_control", STAGE21)


def relative(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(
        abs(float(left)), abs(float(right)), np.finfo(float).tiny
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


def trace_integral(
    forward_fields,
    adjoint_fields,
    *,
    epsilon_au: complex,
    normal_offset_m: float,
    edge_order: int,
    z_bins: int,
) -> dict[str, object]:
    quadrature = stage17.ellipse_boundary_quadrature(
        half_width_m=HALF_X_M,
        gauss_order_per_edge=edge_order,
    )
    points = quadrature["points_xy"].reshape(-1, 2)
    normals = quadrature["normals_xyz"].reshape(-1, 3)
    velocity = quadrature["normal_velocity_m_per_m"].reshape(-1)
    arc_weight = quadrature["arc_weights_m"].reshape(-1)
    points = points + float(normal_offset_m) * normals[:, :2]

    z_edges = np.linspace(stage12.AU_Z_MIN_M, stage12.AU_Z_MAX_M, z_bins + 1)
    z_center = 0.5 * (z_edges[:-1] + z_edges[1:])
    dz = np.diff(z_edges)
    tangential_by_z = []
    normal_by_z = []
    total_by_z = []
    for z_value, dz_value in zip(z_center, dz):
        x = points[:, 0]
        y = points[:, 1]
        z = np.full_like(x, z_value)
        ef = evaluate_vector(forward_fields.getfield, x, y, z)
        df = evaluate_vector(forward_fields.getDfield, x, y, z)
        ea = evaluate_vector(adjoint_fields.getfield, x, y, z)
        da = evaluate_vector(adjoint_fields.getDfield, x, y, z)
        if not all(np.all(np.isfinite(v)) for v in (ef, df, ea, da)):
            raise RuntimeError("non-finite one-sided boundary trace")

        ef_parallel = ef - np.sum(ef * normals, axis=-1)[:, None] * normals
        ea_parallel = ea - np.sum(ea * normals, axis=-1)[:, None] * normals
        df_normal = np.sum(df * normals, axis=-1)
        da_normal = np.sum(da * normals, axis=-1)
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
        measure = arc_weight * velocity * float(dz_value)
        tangential_value = float(np.sum(tangential_kernel * measure))
        normal_value = float(np.sum(normal_kernel * measure))
        tangential_by_z.append(tangential_value)
        normal_by_z.append(normal_value)
        total_by_z.append(tangential_value + normal_value)

    total = float(np.sum(total_by_z))
    middle = (z_center >= stage12.AU_Z_MIN_M + 10.0e-9) & (
        z_center <= stage12.AU_Z_MAX_M - 10.0e-9
    )
    rim = ~middle
    return {
        "normal_offset_m": float(normal_offset_m),
        "trace_side": (
            "geometric_boundary"
            if normal_offset_m == 0.0
            else ("Au_inside" if normal_offset_m < 0.0 else "air_outside")
        ),
        "edge_gauss_order": int(edge_order),
        "z_bins": int(z_bins),
        "z_center_m": z_center.tolist(),
        "tangential_E_term_J_proxy_per_m_by_z": tangential_by_z,
        "normal_D_term_J_proxy_per_m_by_z": normal_by_z,
        "total_J_proxy_per_m_by_z": total_by_z,
        "tangential_E_term_J_proxy_per_m": float(np.sum(tangential_by_z)),
        "normal_D_term_J_proxy_per_m": float(np.sum(normal_by_z)),
        "total_J_proxy_per_m": total,
        "total_J_proxy_per_um": total * 1.0e-6,
        "middle_10nm_trimmed_J_proxy_per_m": float(np.sum(np.asarray(total_by_z)[middle])),
        "top_bottom_10nm_rims_J_proxy_per_m": float(np.sum(np.asarray(total_by_z)[rim])),
        "rim_fraction_of_absolute_total": float(
            abs(np.sum(np.asarray(total_by_z)[rim]))
            / max(abs(total), np.finfo(float).tiny)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--adjoint-dir",
        type=Path,
        default=DEFAULT_RAW / "pva5_fixedgrid_smooth_ellipse_external_field_adjoint_gpu0",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument(
        "--normal-offsets-nm",
        type=float,
        nargs="+",
        default=[-50.0, -25.0, -12.5, -5.0, 0.0, 5.0, 12.5, 25.0, 50.0],
    )
    parser.add_argument("--edge-order", type=int, default=4)
    parser.add_argument("--z-bins", type=int, default=100)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "au_pva_boundary_one_sided_trace_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_AU_PVA_BOUNDARY_TRACE_DIAGNOSTIC",
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
        baseline_project, _ = stage12.pq.checked_project(args.raw_root / BASELINE_CASE)
        source_result = json.loads(
            (args.adjoint_dir / "au_sharp_interface_external_field_result.json").read_text()
        )
        adjoint_project = Path(source_result["adjoint"]["project"]["path"])
        profile_scale = float(source_result["source"]["profile_scale"])
        base_amplitude = float(source_result["source"]["fieldregion_base_amplitude"])
        fd_target = float(
            source_result["AD_FD_comparison"]["h_0.05_um"]["FD_J_proxy_per_um"]
        )

        fdtd, _audit, _runtime = stage12.open_fdtd(args.gpu_device)
        fdtd.load(str(baseline_project))
        fdtd.runanalysis(stage12.PABS_GROUP)
        forward, grid = stage12.monitor_electric(fdtd, stage12.PABS_FIELD)
        detail = stage12.index_detail(fdtd)
        epsilon = stage21.epsilon_from_detail(detail)
        epsilon_au, material_name = stage12.fitted_au_epsilon(fdtd, stage12.AU_OBJECT if hasattr(stage12, "AU_OBJECT") else "rho1_scalar_complex_block")

        fdtd.load(str(adjoint_project))
        fdtd.cwnorm(1)
        adjoint_first, adjoint_grid = stage12.monitor_electric(fdtd, stage12.PABS_FIELD)
        fdtd.cwnorm(2)
        adjoint_average, average_grid = stage12.monitor_electric(fdtd, stage12.PABS_FIELD)
        adjoint, normalization = stage12.reconstruct_fieldregion_only_cw(
            adjoint_first, adjoint_average
        )
        adjoint *= profile_scale / base_amplitude
        maximum_grid_mismatch = max(
            stage21.maximum_grid_mismatch(grid, adjoint_grid),
            stage21.maximum_grid_mismatch(adjoint_grid, average_grid),
        )
        if maximum_grid_mismatch >= 2.0e-18:
            raise RuntimeError(f"forward/adjoint grid mismatch {maximum_grid_mismatch}")

        forward_fields = stage12.pq.build_nointerp_fields(forward, epsilon, grid)
        adjoint_fields = stage12.pq.build_nointerp_fields(adjoint, epsilon, grid)
        traces = [
            trace_integral(
                forward_fields,
                adjoint_fields,
                epsilon_au=epsilon_au,
                normal_offset_m=float(offset_nm) * 1.0e-9,
                edge_order=int(args.edge_order),
                z_bins=int(args.z_bins),
            )
            for offset_nm in args.normal_offsets_nm
        ]
        by_offset = {round(row["normal_offset_m"] * 1.0e9, 9): row for row in traces}
        symmetric_pairs = []
        for offset_nm in sorted({abs(float(v)) for v in args.normal_offsets_nm if v != 0.0}):
            if -offset_nm not in by_offset or offset_nm not in by_offset:
                continue
            inside = by_offset[-offset_nm]
            outside = by_offset[offset_nm]
            symmetric = 0.5 * (
                inside["total_J_proxy_per_um"] + outside["total_J_proxy_per_um"]
            )
            symmetric_pairs.append(
                {
                    "offset_nm": offset_nm,
                    "inside_J_proxy_per_um": inside["total_J_proxy_per_um"],
                    "outside_J_proxy_per_um": outside["total_J_proxy_per_um"],
                    "inside_outside_relative_mismatch": relative(
                        inside["total_J_proxy_per_um"], outside["total_J_proxy_per_um"]
                    ),
                    "symmetric_average_J_proxy_per_um": symmetric,
                    "symmetric_average_FD_relative_error": relative(symmetric, fd_target),
                    "symmetric_average_sign_agrees": bool(symmetric * fd_target > 0.0),
                }
            )

        result.update(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": (
                    "offline one-sided trace and z-rim decomposition of the PVA5 "
                    "smooth exact-Au boundary kernel; not a promoted gradient"
                ),
                "baseline_case": BASELINE_CASE,
                "material_name": material_name,
                "epsilon_au": [float(epsilon_au.real), float(epsilon_au.imag)],
                "FD_target_h0p05_J_proxy_per_um": fd_target,
                "maximum_forward_adjoint_grid_mismatch_m": maximum_grid_mismatch,
                "normalization": normalization,
                "traces": traces,
                "symmetric_trace_pairs": symmetric_pairs,
                "interpretation_contract": {
                    "negative_offset": "sample on Au-inside side of lateral boundary",
                    "positive_offset": "sample on air-outside side of lateral boundary",
                    "zero_offset": "direct interpolation on geometric boundary",
                    "no_trace_selected_by_FD_fit": True,
                },
                "status": "DIAGNOSED_AU_PVA_BOUNDARY_ONE_SIDED_TRACES",
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
    return 0 if result["status"] == "DIAGNOSED_AU_PVA_BOUNDARY_ONE_SIDED_TRACES" else 2


if __name__ == "__main__":
    raise SystemExit(main())
