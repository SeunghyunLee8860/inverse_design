#!/usr/bin/env python3
"""Offline one-sided field-trace audit for the smooth 3-D Au control.

This script opens already-completed forward and adjoint FSP files but performs
no Maxwell solve.  The exact ellipsoid surface, shape velocity, and quadrature
measure remain fixed; only the field evaluation points move a signed distance
along the outward normal.  It therefore diagnoses interpolation of the
continuum boundary kernel without fitting any trace to finite differences.
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
DEFAULT_ADJOINT = (
    DEFAULT_RAW / "pva5_smooth3d_ellipsoid_boundary_adjoint_gpu0"
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage12 = load("au_smooth3d_trace_stage12", STAGE12)
stage21 = load("au_smooth3d_trace_stage21", STAGE21)
stage26 = load("au_smooth3d_trace_stage26", STAGE26)


def relative(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(
        abs(float(left)), abs(float(right)), np.finfo(float).tiny
    )


def checked_artifact(path: Path, record: dict[str, object]) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual_size = resolved.stat().st_size
    actual_sha = stage12.pq.sha256(resolved)
    if actual_size != int(record["size_bytes"]):
        raise RuntimeError(f"size mismatch for {resolved}")
    if actual_sha != str(record["sha256"]):
        raise RuntimeError(f"SHA-256 mismatch for {resolved}")
    return {"path": str(resolved), "size_bytes": actual_size, "sha256": actual_sha}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--adjoint-dir", type=Path, default=DEFAULT_ADJOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument(
        "--normal-offsets-nm",
        type=float,
        nargs="+",
        default=[-100.0, -50.0, -25.0, -12.5, -5.0, 0.0, 5.0, 12.5, 25.0, 50.0, 100.0],
    )
    parser.add_argument("--mu-order", type=int, default=32)
    parser.add_argument("--phi-count", type=int, default=128)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "au_smooth3d_one_sided_trace_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_AU_SMOOTH3D_ONE_SIDED_TRACE_AUDIT",
        "passed": False,
        "Maxwell_forward_solves_this_invocation": 0,
        "Maxwell_adjoint_solves_this_invocation": 0,
        "thermal_solves": 0,
        "optimizer_started": False,
        "finite_difference_used_to_fit_AD": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "production_Au_optimization_permitted": False,
    }
    fdtd = None
    try:
        baseline_project, baseline_result = stage26.checked_ellipsoid_project(
            args.raw_root / stage26.BASELINE_CASE
        )
        source_result_path = (
            args.adjoint_dir / "au_smooth3d_ellipsoid_boundary_adjoint_result.json"
        )
        source_result = json.loads(source_result_path.read_text())
        adjoint_record = source_result["adjoint"]["project"]
        adjoint_project = Path(adjoint_record["path"])
        checked_inputs = {
            "forward_project": checked_artifact(
                baseline_project,
                next(
                    row
                    for row in baseline_result["raw_artifacts"]
                    if Path(row["path"]).resolve() == baseline_project.resolve()
                ),
            ),
            "adjoint_project": checked_artifact(adjoint_project, adjoint_record),
        }
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
        epsilon_au, material_name = stage12.fitted_au_epsilon(
            fdtd, "rho1_scalar_complex_block"
        )

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
        mismatch = max(
            stage21.maximum_grid_mismatch(grid, adjoint_grid),
            stage21.maximum_grid_mismatch(adjoint_grid, average_grid),
        )
        if mismatch >= 2.0e-18:
            raise RuntimeError(f"forward/adjoint grid mismatch {mismatch:.3e} m")

        forward_fields = stage12.pq.build_nointerp_fields(forward, epsilon, grid)
        adjoint_fields = stage12.pq.build_nointerp_fields(adjoint, epsilon, grid)
        traces = []
        for offset_nm in args.normal_offsets_nm:
            row = stage26.smooth_ellipsoid_surface_integral(
                forward_fields,
                adjoint_fields,
                epsilon_au=epsilon_au,
                mu_order=int(args.mu_order),
                phi_count=int(args.phi_count),
                normal_offset_m=float(offset_nm) * 1.0e-9,
            )
            ad_per_um = float(row["total_J_proxy_per_m"]) * 1.0e-6
            row["AD_J_proxy_per_um"] = ad_per_um
            row["FD_J_proxy_per_um"] = fd_target
            row["FD_relative_error"] = relative(ad_per_um, fd_target)
            row["FD_sign_agrees"] = bool(ad_per_um * fd_target > 0.0)
            traces.append(row)

        by_offset = {
            round(float(row["field_trace_normal_offset_m"]) * 1.0e9, 9): row
            for row in traces
        }
        symmetric_pairs = []
        magnitudes = sorted(
            {abs(float(value)) for value in args.normal_offsets_nm if value != 0.0}
        )
        for offset_nm in magnitudes:
            key = round(offset_nm, 9)
            if -key not in by_offset or key not in by_offset:
                continue
            inside = by_offset[-key]
            outside = by_offset[key]
            average = 0.5 * (
                float(inside["AD_J_proxy_per_um"])
                + float(outside["AD_J_proxy_per_um"])
            )
            symmetric_pairs.append(
                {
                    "offset_nm": offset_nm,
                    "inside_AD_J_proxy_per_um": inside["AD_J_proxy_per_um"],
                    "outside_AD_J_proxy_per_um": outside["AD_J_proxy_per_um"],
                    "inside_outside_relative_mismatch": relative(
                        inside["AD_J_proxy_per_um"], outside["AD_J_proxy_per_um"]
                    ),
                    "symmetric_average_AD_J_proxy_per_um": average,
                    "FD_relative_error": relative(average, fd_target),
                    "FD_sign_agrees": bool(average * fd_target > 0.0),
                }
            )

        result.update(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "DIAGNOSED_AU_SMOOTH3D_ONE_SIDED_TRACES",
                "scope": (
                    "offline field-trace audit on an unchanged exact ellipsoid "
                    "surface; no Maxwell solve and no FD-fitted trace selection"
                ),
                "official_formula_source": (
                    "/opt/lumerical/v261/api/python/lumopt/utilities/gradients.py"
                ),
                "material_name": material_name,
                "epsilon_au": [float(epsilon_au.real), float(epsilon_au.imag)],
                "FD_target_h0p05_J_proxy_per_um": fd_target,
                "maximum_forward_adjoint_grid_mismatch_m": mismatch,
                "normalization": normalization,
                "input_artifacts": checked_inputs,
                "quadrature": {
                    "mu_order": int(args.mu_order),
                    "phi_count": int(args.phi_count),
                },
                "traces": traces,
                "symmetric_trace_pairs": symmetric_pairs,
                "interpretation_contract": {
                    "negative_offset": "sample fields on Au-inside side",
                    "positive_offset": "sample fields on air-outside side",
                    "surface_geometry_and_shape_velocity_change": False,
                    "trace_selected_by_finite_difference_fit": False,
                    "a_matching_offset_would_not_be_a_production_gradient": True,
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
    return 0 if result["status"] == "DIAGNOSED_AU_SMOOTH3D_ONE_SIDED_TRACES" else 2


if __name__ == "__main__":
    raise SystemExit(main())
