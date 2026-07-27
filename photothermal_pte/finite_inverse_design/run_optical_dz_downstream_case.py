#!/usr/bin/env python3
"""Propagate one optical dz checkpoint through thermal/PTE and adjoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .explicit_thermal import build_explicit_geometry, evaluate_explicit_thermal
from .probe_v261_cpu_tfsf_device import PABS_FIELD, SOURCE_NAME
from .probe_v261_gpu_plane_wave_roi import json_default, load_lumapi
from .run_combined_physical_rho_pte_adfd import (
    build_native_thermal_mapping,
    compact_forward,
    coupling_for_geometry,
    native_weight_and_source,
    physical_state,
    run_forward_density,
)
from .run_v261_large_background_mixed_optical_adfd import (
    FIELD_REGION,
    as_e5,
    fieldregion_profile,
    monitor_electric,
    prepare_adjoint_layout,
    run_adjoint,
)
from .run_v261_large_background_tfsf_forward import sha256


STATUS_PASS = "GENERATED_OPTICAL_DZ_DOWNSTREAM_CASE"
STATUS_FAIL = "FAILED_OPTICAL_DZ_DOWNSTREAM_CASE"
CLOSURE_LIMIT = 5.0e-3
POWER_LIMIT = 5.0e-3
ENERGY_LIMIT = 1.0e-2
RESIDUAL_LIMIT = 1.0e-8
FLUX_SIGNS = {
    f"device_flux_{axis}_{side}": (-1.0 if side == "min" else 1.0)
    for axis in "xyz"
    for side in ("min", "max")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-project", required=True)
    parser.add_argument("--forward-sha256", required=True)
    parser.add_argument("--flake-dz-nm", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gpu-device", default="GPU 1")
    for flake_um in (4, 6):
        parser.add_argument(f"--resume-adjoint-{flake_um}um")
        parser.add_argument(f"--resume-adjoint-{flake_um}um-sha256")
    return parser.parse_args()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
    }


def same_grid(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> float:
    return max(
        float(
            np.max(
                np.abs(
                    np.asarray(left[key], float)
                    - np.asarray(right[key], float)
                )
            )
        )
        for key in ("x", "y", "z", "delta_x", "delta_y", "delta_z")
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "optical_dz_downstream_case_result.json"
    forward_project = Path(args.forward_project).expanduser().resolve()
    result: dict[str, object] = {
        "status": "BLOCKED_OPTICAL_DZ_DOWNSTREAM_CASE_NOT_RUN",
        "passed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "one nonuniform physical-rho optical dz source propagated to "
            "named 4/6 um explicit thermal/PTE scenarios and spatially "
            "weighted GPU Maxwell adjoints"
        ),
        "flake_dz_nm": args.flake_dz_nm,
        "finite_difference_run": False,
        "optimization_run": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
    }
    fdtd = None
    started = time.monotonic()
    try:
        if not forward_project.is_file():
            raise FileNotFoundError(forward_project)
        if sha256(forward_project) != args.forward_sha256:
            raise RuntimeError("forward FSP SHA-256 mismatch")
        rho, direction = physical_state()
        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        forward = run_forward_density(
            fdtd,
            rho=rho,
            project=forward_project,
            threads=args.threads,
            flux_signs=FLUX_SIGNS,
            reuse_completed=True,
        )
        scenario_records = {}
        all_artifacts = []
        for flake_um in (4.0, 6.0):
            initial_geometry = build_explicit_geometry(
                np.full((20, 20), 0.5),
                lateral_domain_m=32.0e-6,
                si_depth_m=20.0e-6,
                flake_span_m=flake_um * 1.0e-6,
                core_xy_cell_size_m=100.0e-9,
                flake_dz_m=25.0e-9,
                design_dz_m=100.0e-9,
            )
            coupling = coupling_for_geometry(initial_geometry)
            thermal_rho = coupling.thermal(rho)
            geometry = build_explicit_geometry(
                thermal_rho,
                lateral_domain_m=32.0e-6,
                si_depth_m=20.0e-6,
                flake_span_m=flake_um * 1.0e-6,
                core_xy_cell_size_m=100.0e-9,
                flake_dz_m=25.0e-9,
                design_dz_m=100.0e-9,
            )
            mapping = build_native_thermal_mapping(
                forward["native"], geometry
            )
            evaluation = evaluate_explicit_thermal(
                rho=thermal_rho,
                source_W_m3=mapping["source_W_m3"],
                lateral_domain_m=32.0e-6,
                si_depth_m=20.0e-6,
                flake_span_m=flake_um * 1.0e-6,
                core_xy_cell_size_m=100.0e-9,
                flake_dz_m=25.0e-9,
                design_dz_m=100.0e-9,
            )
            coefficient, weighted_source, pullback = (
                native_weight_and_source(
                    evaluation=evaluation,
                    native=forward["native"],
                    mapping=mapping,
                    electric=forward["electric"],
                    epsilon=forward["epsilon"],
                    frequency_Hz=float(forward["grid"]["f"][0]),
                )
            )
            thermal_gradient_nodal = coupling.thermal_vjp(
                evaluation.gradient_rho_A
            )
            profile, profile_scale = fieldregion_profile(weighted_source)
            resume_path_text = getattr(
                args, f"resume_adjoint_{int(flake_um)}um"
            )
            resume_sha = getattr(
                args, f"resume_adjoint_{int(flake_um)}um_sha256"
            )
            if bool(resume_path_text) != bool(resume_sha):
                raise ValueError(
                    f"{flake_um:g} um resume adjoint path/SHA must "
                    "be supplied together"
                )
            if resume_path_text:
                adjoint_project = (
                    Path(resume_path_text).expanduser().resolve()
                )
                if not adjoint_project.is_file():
                    raise FileNotFoundError(adjoint_project)
                if sha256(adjoint_project) != resume_sha:
                    raise RuntimeError(
                        f"{flake_um:g} um resume adjoint SHA mismatch"
                    )
                fdtd.load(str(adjoint_project))
                if int(fdtd.getnamednumber(SOURCE_NAME)) != 0:
                    raise RuntimeError(
                        "reused adjoint unexpectedly contains TFSF"
                    )
                imported_profile = as_e5(
                    fdtd.getresult(FIELD_REGION, "source profile")["E"]
                )
                profile_error = float(
                    np.max(np.abs(imported_profile - profile))
                )
                if profile_error != 0.0:
                    raise RuntimeError(
                        "reused adjoint source profile differs from "
                        f"recomputed profile: {profile_error:.3e}"
                    )
                source_meta = {
                    "source_profile_roundtrip_max_abs_error": profile_error,
                    "fieldregion_base_amplitude": float(
                        fdtd.getnamed(FIELD_REGION, "base amplitude")
                    ),
                    "reused_completed_project": True,
                }
                adjoint = {
                    "engine": "GPU",
                    "run_wall_s": 0.0,
                    "resource_name": "REUSED_COMPLETED_FSP",
                    "resources": {"reuse": True},
                    "project": artifact(adjoint_project),
                }
                template = None
            else:
                fdtd.switchtolayout()
                fdtd.load(str(forward_project))
                template = (
                    output
                    / f"flake_{flake_um:g}um_adjoint_template.fsp"
                )
                source_meta = prepare_adjoint_layout(
                    fdtd,
                    grid=forward["grid"],
                    profile=profile,
                    template=template,
                )
                adjoint_project = (
                    output / f"flake_{flake_um:g}um_adjoint_gpu.fsp"
                )
                adjoint = run_adjoint(
                    fdtd,
                    template=template,
                    project=adjoint_project,
                    engine="GPU",
                    threads=args.threads,
                    gpu_device=args.gpu_device,
                )
            fdtd.load(str(adjoint_project))
            adjoint_electric, adjoint_grid = monitor_electric(
                fdtd, PABS_FIELD
            )
            coordinate_mismatch = same_grid(
                forward["grid"], adjoint_grid
            )
            if coordinate_mismatch > 2.0e-18:
                raise RuntimeError(
                    "forward/adjoint PABS component coordinates differ"
                )
            raw_path = output / f"flake_{flake_um:g}um_downstream_raw.npz"
            np.savez_compressed(
                raw_path,
                rho=rho,
                direction=direction,
                mapped_Q_W_m3=mapping["source_W_m3"],
                temperature_K=evaluation.solved.temperature_K,
                flake_mask=geometry.flake_mask,
                x_edges_m=geometry.x_edges_m,
                y_edges_m=geometry.y_edges_m,
                z_edges_m=geometry.z_edges_m,
                forward_electric=forward["electric"],
                adjoint_electric=adjoint_electric,
                thermal_gradient_nodal_A=thermal_gradient_nodal,
                profile_scale=np.asarray([profile_scale]),
                fieldregion_base_amplitude=np.asarray(
                    [source_meta["fieldregion_base_amplitude"]]
                ),
                pabs_x_m=forward["grid"]["x"],
                pabs_y_m=forward["grid"]["y"],
                pabs_z_m=forward["grid"]["z"],
                pabs_delta_x_m=forward["grid"]["delta_x"],
                pabs_delta_y_m=forward["grid"]["delta_y"],
                pabs_delta_z_m=forward["grid"]["delta_z"],
            )
            raw_record = artifact(raw_path)
            if template is not None:
                all_artifacts.append(artifact(template))
            all_artifacts.extend([artifact(adjoint_project), raw_record])
            theta = evaluation.solved.temperature_K
            flake_theta = theta[geometry.flake_mask]
            scenario_records[f"{flake_um:g}um"] = {
                "flake_span_um": flake_um,
                "thermal_controls": {
                    "lateral_domain_um": 32.0,
                    "si_depth_um": 20.0,
                    "core_xy_cell_size_nm": 100.0,
                    "flake_dz_nm": 25.0,
                    "design_dz_nm": 100.0,
                },
                "mapped_Q_power_W": mapping["mapped_power_W"],
                "Q_mapping_relative_power_error": mapping[
                    "relative_power_error"
                ],
                "Q_outside_flake_nonzero_count": mapping[
                    "outside_flake_nonzero_count"
                ],
                "Tmax_DeltaT_K": float(np.max(theta)),
                "TaIrTe4_Tmax_DeltaT_K": float(np.max(flake_theta)),
                "PTE_objective_A": float(evaluation.objective_A),
                "thermal_directional_gradient_A": float(
                    np.sum(thermal_gradient_nodal * direction)
                ),
                "thermal_energy_balance_relative_error": float(
                    evaluation.solved.energy_balance_relative_error
                ),
                "thermal_forward_linear_residual_relative": float(
                    evaluation.solved.linear_residual_relative
                ),
                "thermal_adjoint_linear_residual_relative": float(
                    evaluation.adjoint_linear_residual_relative
                ),
                "native_Q_pullback": pullback,
                "forward_adjoint_coordinate_mismatch_m": coordinate_mismatch,
                "source_profile_roundtrip_max_abs_error": source_meta[
                    "source_profile_roundtrip_max_abs_error"
                ],
                "profile_scale": profile_scale,
                "fieldregion_base_amplitude": source_meta[
                    "fieldregion_base_amplitude"
                ],
                "adjoint": {
                    key: value
                    for key, value in adjoint.items()
                    if key not in {"electric", "grid"}
                },
                "raw_artifact": raw_record,
            }
            print(
                "OPTICAL_DZ_DOWNSTREAM "
                f"dz={args.flake_dz_nm:g}nm flake={flake_um:g}um "
                f"PTE={evaluation.objective_A:.9e} "
                f"Tmax={np.max(theta):.9e}",
                flush=True,
            )
        worst_mapping = max(
            value["Q_mapping_relative_power_error"]
            for value in scenario_records.values()
        )
        worst_energy = max(
            value["thermal_energy_balance_relative_error"]
            for value in scenario_records.values()
        )
        worst_residual = max(
            max(
                value["thermal_forward_linear_residual_relative"],
                value["thermal_adjoint_linear_residual_relative"],
            )
            for value in scenario_records.values()
        )
        passed = bool(
            forward["six_face_closure_relative_error"] < CLOSURE_LIMIT
            and worst_mapping < POWER_LIMIT
            and worst_energy < ENERGY_LIMIT
            and worst_residual < RESIDUAL_LIMIT
        )
        result.update(
            {
                "status": STATUS_PASS if passed else STATUS_FAIL,
                "passed": passed,
                "forward": compact_forward(forward),
                "forward_FSP": artifact(forward_project),
                "scenarios": scenario_records,
                "gates": {
                    "six_face_closure_relative_error": forward[
                        "six_face_closure_relative_error"
                    ],
                    "six_face_closure_limit": CLOSURE_LIMIT,
                    "worst_Q_mapping_relative_power_error": worst_mapping,
                    "Q_mapping_limit": POWER_LIMIT,
                    "worst_thermal_energy_balance_relative_error": (
                        worst_energy
                    ),
                    "thermal_energy_balance_limit": ENERGY_LIMIT,
                    "worst_linear_residual_relative": worst_residual,
                    "linear_residual_limit": RESIDUAL_LIMIT,
                },
                "raw_artifacts": all_artifacts,
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": STATUS_FAIL,
                "passed": False,
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
        result["wall_s"] = time.monotonic() - started
        result_path.write_text(
            json.dumps(result, indent=2, default=json_default) + "\n"
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "flake_dz_nm": args.flake_dz_nm,
                "result_path": str(result_path),
            }
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
