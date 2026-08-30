#!/usr/bin/env python3
"""Compare exact-Au Maxwell FD and complex d-epsilon at the same CAD step.

The two forward projects must already be completed.  This program only loads
their saved fields and evaluates the fixed external-air objective; it never
calls ``run`` or ``runsetup`` and therefore performs no Maxwell solve.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
STAGE12 = HERE / "12_run_au_sharp_interface_external_field_adjoint.py"
STAGE26 = HERE / "26_validate_au_smooth_3d_ellipsoid_boundary_adjoint.py"
DEFAULT_RAW = Path("/data/seunghyun/tairte4/raw_artifacts/au_topology_validation")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage12 = load("au_same_step_fd_stage12", STAGE12)
stage26 = load("au_same_step_fd_stage26", STAGE26)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_project(case_dir: Path, expected_half_width_um: float) -> tuple[Path, dict]:
    recovered = case_dir / "case_result_recovered.json"
    result_path = recovered if recovered.is_file() else case_dir / "case_result.json"
    result = json.loads(result_path.read_text())
    if not bool(result.get("passed", False)):
        raise RuntimeError(f"forward case did not pass: {result_path}")
    actual = stage26.result_half_width_um(result)
    if not np.isclose(actual, expected_half_width_um, rtol=0.0, atol=1.0e-9):
        raise RuntimeError(f"half width {actual} um != {expected_half_width_um} um")
    project = case_dir / "complex_material_control.fsp"
    records = {
        str(Path(row["path"]).resolve()): row for row in result.get("raw_artifacts", [])
    }
    record = records.get(str(project.resolve()))
    if record is None:
        raise RuntimeError(f"project absent from raw manifest: {result_path}")
    if int(record["size_bytes"]) != project.stat().st_size:
        raise RuntimeError(f"project size mismatch: {project}")
    digest = sha256(project)
    if digest != str(record["sha256"]):
        raise RuntimeError(f"project SHA-256 mismatch: {project}")
    result["selected_result_path"] = str(result_path)
    return project, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minus-dir",
        type=Path,
        default=DEFAULT_RAW / "pva5_smooth3d_ellipsoid_a7p999_b18_c1_forward_retry2_gpu4",
    )
    parser.add_argument(
        "--plus-dir",
        type=Path,
        default=DEFAULT_RAW / "pva5_smooth3d_ellipsoid_a8p001_b18_c1_forward_gpu4",
    )
    parser.add_argument(
        "--deps-result",
        type=Path,
        default=(
            DEFAULT_RAW
            / "same_session_complex_deps_subnm_gpu4"
            / "au_same_session_complex_deps_result.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RAW / "same_step_local_maxwell_fd_h1nm",
    )
    parser.add_argument("--minus-half-width-um", type=float, default=7.999)
    parser.add_argument("--plus-half-width-um", type=float, default=8.001)
    parser.add_argument("--deps-step-key", default="h_1_nm")
    parser.add_argument("--gpu-device", default="GPU 3")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "au_same_step_local_maxwell_fd_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_AU_SAME_STEP_LOCAL_MAXWELL_FD",
        "passed": False,
        "scope": "exact scalar-Au smooth-ellipsoid x-semi-axis derivative",
        "Maxwell_forward_solves_this_invocation": 0,
        "Maxwell_adjoint_solves_this_invocation": 0,
        "runsetup_calls_this_invocation": 0,
        "empirical_sign_flip_or_gradient_rescaling": False,
    }
    fdtd = None
    try:
        # Stage 12 defaults to the thin-film ROI.  The completed ellipsoid
        # certificate deliberately moved the fixed objective farther below
        # the smooth 3-D Au body; reproduce that exact objective here.
        stage12.ROI = dict(stage26.ROI)
        stage12.AU_Z_MIN_M = stage26.Z_MIN_M
        stage12.AU_Z_MAX_M = stage26.Z_MAX_M
        if not args.plus_half_width_um > args.minus_half_width_um:
            raise ValueError("plus half width must exceed minus half width")
        minus_project, minus_result = checked_project(
            args.minus_dir.resolve(), args.minus_half_width_um
        )
        plus_project, plus_result = checked_project(
            args.plus_dir.resolve(), args.plus_half_width_um
        )
        deps = json.loads(args.deps_result.resolve().read_text())
        deps_row = deps["steps"][args.deps_step_key]
        complex_deps_ad = float(deps_row["same_session_complex_deps_AD_J_proxy_per_um"])

        fdtd, _audit, runtime = stage12.open_fdtd(args.gpu_device)
        objectives: dict[str, object] = {}
        grids: dict[str, dict] = {}
        for name, project, metadata in (
            ("minus", minus_project, minus_result),
            ("plus", plus_project, plus_result),
        ):
            fdtd.load(str(project))
            fdtd.runanalysis(stage12.PABS_GROUP)
            electric, grid = stage12.monitor_electric(fdtd, stage12.PABS_FIELD)
            objective, _source, objective_meta = stage12.fixed_air_objective_and_source(
                electric, grid
            )
            grids[name] = grid
            objectives[name] = {
                "half_width_um": stage26.result_half_width_um(metadata),
                "objective_J_proxy": objective,
                "component_value_J_proxy": objective_meta["component_value_J_proxy"],
                "project": {
                    "path": str(project),
                    "size_bytes": project.stat().st_size,
                    "sha256": sha256(project),
                },
                "result": {
                    "path": metadata["selected_result_path"],
                    "status": metadata["status"],
                },
            }
        grid_mismatch = stage26.maximum_grid_mismatch(grids["minus"], grids["plus"])
        denominator_um = args.plus_half_width_um - args.minus_half_width_um
        h_nm = 0.5 * denominator_um * 1.0e3
        maxwell_fd = (
            float(objectives["plus"]["objective_J_proxy"])
            - float(objectives["minus"]["objective_J_proxy"])
        ) / denominator_um
        relative_error = abs(complex_deps_ad - maxwell_fd) / max(
            abs(maxwell_fd), np.finfo(float).tiny
        )
        same_sign = bool(complex_deps_ad * maxwell_fd > 0.0)
        gates = {
            "same_component_grid_max_mismatch_lt_1e_15_m": grid_mismatch < 1.0e-15,
            "same_step_complex_dEps_vs_Maxwell_FD_relative_error_lt_1pct": relative_error
            < 0.01,
            "same_sign": same_sign,
        }
        passed = bool(all(gates.values()))
        result.update(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": runtime,
                "parameter_step": {
                    "h_nm": h_nm,
                    "minus_half_width_um": args.minus_half_width_um,
                    "plus_half_width_um": args.plus_half_width_um,
                    "central_difference_denominator_um": denominator_um,
                    "complex_dEps_step_key": args.deps_step_key,
                },
                "objective_contract": {
                    "definition": "fixed external air-field energy proxy",
                    "requested_bounds": stage26.ROI,
                    "minimum_vertical_clearance_from_ellipsoid_m": (
                        stage26.Z_MIN_M - stage26.ROI["z_max_m"]
                    ),
                },
                "objective_cases": objectives,
                "component_grid_maximum_mismatch_m": grid_mismatch,
                "Maxwell_central_FD_J_proxy_per_um": maxwell_fd,
                "same_session_complex_dEps_AD_J_proxy_per_um": complex_deps_ad,
                "AD_FD_relative_error": relative_error,
                "AD_FD_same_sign": same_sign,
                "gates": gates,
                "passed": passed,
                "status": (
                    "VALIDATED_AU_SAME_STEP_LOCAL_COMPLEX_DEPS"
                    if passed
                    else "FAILED_AU_SAME_STEP_LOCAL_COMPLEX_DEPS"
                ),
                "interpretation": (
                    "The exact-binary reduced-parameter shape route passes this control."
                    if passed
                    else (
                        "Diagonal index-monitor complex d-epsilon is not a complete "
                        "moving-boundary Jacobian for this conformal Au control."
                    )
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
