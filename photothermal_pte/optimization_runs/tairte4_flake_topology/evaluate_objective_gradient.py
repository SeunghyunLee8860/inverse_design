#!/usr/bin/env python3
"""Evaluate signed PTE current and its exact current-density gradient."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from photothermal_pte.finite_inverse_design.native_yee_q import EPS0
from photothermal_pte.optimization_runs.tairte4_flake_topology.validate_combined_adfd import (
    FIELD_REGION,
    FREQUENCY_HZ,
    checked,
    compact_forward,
    load_operator,
    open_fdtd,
    lumerical_gpu_engine_lock,
    polarization_angle,
    pullback_q,
    run_forward,
    sha256,
    solve_coupled,
)

import build_nonuniform_complex_yee_jacobian as jacobian_builder
import run_production_combined_adfd_smoke as legacy_combined
from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.ansys_minimum_feature import (
    evaluate_on_cad as evaluate_ansys_minimum_feature,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.thermal import (
    thermal_interface_contract,
)


RHO_ROUNDOFF_TOLERANCE = 1.0e-12


def discard_regenerable_projects(paths: tuple[Path, ...]) -> list[dict[str, object]]:
    """Delete successful per-evaluation FSPs after recording provenance."""
    discarded: list[dict[str, object]] = []
    for project_path in paths:
        if not project_path.is_file():
            continue
        discarded.append(
            {
                "path": str(project_path),
                "size_bytes": project_path.stat().st_size,
                "sha256": sha256(project_path),
            }
        )
        project_path.unlink()
    return discarded


def load_rho(path: Path) -> np.ndarray:
    source = path.expanduser().resolve()
    with np.load(source) as data:
        if "rho" not in data:
            raise RuntimeError("density artifact must contain rho")
        rho = np.asarray(data["rho"], dtype=np.float64)
    if rho.shape != CONTRACT.design_node_shape:
        raise RuntimeError(f"rho shape {rho.shape} != {CONTRACT.design_node_shape}")
    if not np.all(np.isfinite(rho)):
        raise RuntimeError("rho must be finite in [0,1]")
    if np.any(rho < -RHO_ROUNDOFF_TOLERANCE) or np.any(
        rho > 1.0 + RHO_ROUNDOFF_TOLERANCE
    ):
        raise RuntimeError("rho must be finite in [0,1]")
    # The analytical tanh projection can differ from its mathematical [0,1]
    # range by one floating-point ulp at high beta (observed: -5.55e-17 at
    # beta=16). Canonicalize only this bounded roundoff; material densities
    # outside the explicit tolerance still fail closed above.
    return np.clip(rho, 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-fsp", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--jacobian-dir", required=True, type=Path)
    parser.add_argument("--rho-npz", required=True, type=Path)
    parser.add_argument("--latent-npz", type=Path)
    parser.add_argument("--dfm-beta", type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--gpu-device", default="GPU 5")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--discard-fsp-after-success",
        action="store_true",
        help=(
            "Remove per-evaluation forward/adjoint/template FSP projects only "
            "after the validated NPZ and their path/size/SHA provenance have "
            "been recorded. Intended for long optimization runs."
        ),
    )
    args = parser.parse_args()
    base_fsp = checked(args.base_fsp, args.base_sha256)
    _, _, operator_meta = load_operator(args.jacobian_dir)
    rho = CONTRACT.apply_fixed_contact_density(load_rho(args.rho_npz))
    latent = None
    if (args.latent_npz is None) != (args.dfm_beta is None):
        raise RuntimeError("--latent-npz and --dfm-beta must be supplied together")
    if args.latent_npz is not None:
        with np.load(args.latent_npz.expanduser().resolve()) as data:
            latent = np.asarray(data["latent"], dtype=np.float64)
        if latent.shape != CONTRACT.design_node_shape:
            raise RuntimeError("latent DFM shape does not match the design contract")
    angle = polarization_angle(args.polarization)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "objective_gradient_result.json"
    result = {
        "status": "FAILED_TAIRTE4_FLAKE_OBJECTIVE_GRADIENT",
        "passed": False,
        "optimization_iterations": 0,
    }
    fdtd = None
    started = time.monotonic()
    try:
        fdtd, audit, runtime = open_fdtd(args.gpu_device)
        ansys_dfm_indicators = np.zeros(2, dtype=np.float64)
        ansys_dfm_gradient = np.zeros_like(rho)
        ansys_dfm = None
        if latent is not None:
            latent = CONTRACT.apply_fixed_contact_density(latent)
            ansys_dfm_indicators, ansys_dfm_gradient, ansys_dfm = (
                evaluate_ansys_minimum_feature(fdtd, latent, float(args.dfm_beta))
            )
            ansys_dfm_gradient = CONTRACT.zero_fixed_contact_gradient(
                ansys_dfm_gradient
            )
        forward = run_forward(
            fdtd,
            audit,
            runtime,
            template=base_fsp,
            rho=rho,
            role=f"forward_{args.polarization}",
            output=output,
            polarization_angle_deg=angle,
        )
        fdtd.switchtolayout()
        local_operator, local_meta = jacobian_builder.build_tairte4_local_epsilon_operator(
            fdtd, rho
        )
        coupled = solve_coupled(forward, rho, args.cuda_device, need_adjoint=True)
        pulled, pullback_meta = pullback_q(forward, coupled)
        native_source = np.zeros_like(forward["electric"], dtype=np.complex128)
        for index, component in enumerate("xyz"):
            native_source[..., 0, index] = (
                0.5
                * EPS0
                * (2.0 * np.pi * FREQUENCY_HZ)
                * np.imag(forward["epsilon"][..., 0, index])
                * pulled[component]
                * forward["electric"][..., 0, index]
            )
        template = output / "adjoint_template.fsp"
        profile_scale, base_amplitude, source_meta = legacy_combined.prepare_common_grid_source(
            fdtd,
            audit,
            base_project=Path(forward["project"]["path"]),
            grid=forward["grid"],
            native_source=native_source,
            template=template,
        )
        with lumerical_gpu_engine_lock() as adjoint_lock_metadata:
            adjoint = legacy_combined.run_adjoint(
                fdtd,
                audit,
                runtime,
                template=template,
                project=output / "adjoint_gpu.fsp",
            )
        adjoint["global_gpu_engine_lock"] = adjoint_lock_metadata
        gradient_optical, optical_meta = legacy_combined.optical_gradient(
            local_operator,
            forward=forward,
            adjoint=adjoint,
            pulled=pulled,
            profile_scale=profile_scale,
            base_amplitude=base_amplitude,
        )
        gradient_optical = CONTRACT.zero_fixed_contact_gradient(gradient_optical)
        gradient_thermal = CONTRACT.zero_fixed_contact_gradient(
            coupled["gradient_thermal"]
        )
        gradient_electrical = CONTRACT.zero_fixed_contact_gradient(
            coupled["gradient_electrical"]
        )
        gradient_terminal_conductance = CONTRACT.zero_fixed_contact_gradient(
            coupled["gradient_terminal_conductance"]
        )
        gradient_total = gradient_optical + gradient_thermal + gradient_electrical
        if not np.all(np.isfinite(gradient_total)) or np.max(np.abs(gradient_total)) == 0.0:
            raise RuntimeError("objective gradient is zero or nonfinite")
        raw = output / "objective_gradient.npz"
        np.savez_compressed(
            raw,
            rho=rho,
            objective_A=np.asarray(coupled["electrical"].current_A),
            gradient_total_A=gradient_total,
            gradient_optical_A=gradient_optical,
            gradient_thermal_A=gradient_thermal,
            gradient_electrical_A=gradient_electrical,
            terminal_conductance_S=np.asarray(coupled["electrical"].terminal_conductance_S),
            gradient_terminal_conductance_S=gradient_terminal_conductance,
            temperature_K=coupled["temperature"],
            ansys_dfm_indicators=ansys_dfm_indicators,
            ansys_dfm_gradient_latent=ansys_dfm_gradient,
        )
        passed = bool(
            forward["closure"] < 0.005
            and coupled["mapping"]["relative_mapping_error"] < 0.005
            and coupled["thermal_forward"].explicit_relative_residual < 1e-8
            and coupled["thermal_adjoint"].explicit_relative_residual < 1e-8
            and coupled["energy"] < 0.01
            and adjoint["log_audit"]["final_auto_shutoff"] < 1e-5
            and optical_meta["forward_adjoint_coordinate_mismatch_m"] < 2e-18
        )
        result = {
            "status": "VALIDATED_TAIRTE4_FLAKE_OBJECTIVE_GRADIENT" if passed else "FAILED_TAIRTE4_FLAKE_OBJECTIVE_GRADIENT",
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "polarization": args.polarization,
            "polarization_angle_deg": angle,
            "axis_contract": "Lumerical x=b, y=a, z=c",
            "thermal_interface_contract": thermal_interface_contract(),
            "objective": "signed full-flake terminal PTE current",
            "objective_A": coupled["electrical"].current_A,
            "rho_range": [float(np.min(rho)), float(np.max(rho))],
            "gradient_norms_A": {
                "total": float(np.linalg.norm(gradient_total)),
                "optical": float(np.linalg.norm(gradient_optical)),
                "thermal": float(np.linalg.norm(gradient_thermal)),
                "electrical": float(np.linalg.norm(gradient_electrical)),
            },
            "terminal_conductance_S": coupled["electrical"].terminal_conductance_S,
            "current_density_local_Yee_J": local_meta,
            "baseline_operator_provenance": operator_meta,
            "forward": compact_forward(forward),
            "Q_mapping": coupled["mapping"],
            "thermal": {
                "forward_residual": coupled["thermal_forward"].explicit_relative_residual,
                "adjoint_residual": coupled["thermal_adjoint"].explicit_relative_residual,
                "energy_balance": coupled["energy"],
            },
            "pullback": pullback_meta,
            "adjoint_source": source_meta,
            "adjoint": {key: value for key, value in adjoint.items() if key not in {"electric", "grid"}},
            "optical_gradient": optical_meta,
            "ansys_minimum_feature": ansys_dfm,
            "raw_artifact": {
                "path": str(raw),
                "size_bytes": raw.stat().st_size,
                "sha256": sha256(raw),
            },
            "Maxwell_solves": {"forward": 1, "adjoint": 1},
            "optimization_iterations": 0,
            "empirical_normalization": False,
            "gradient_rescaling": False,
            "Q_clipping_smoothing_gain_or_rescaling": False,
            "CPU_FDTD_fallback": False,
            "CPU_thermal_linear_solve_fallback": False,
            "wall_s": time.monotonic() - started,
        }
        if args.discard_fsp_after_success:
            discarded_projects = discard_regenerable_projects(
                (
                    Path(forward["project"]["path"]),
                    template,
                    Path(adjoint["project"]["path"]),
                )
            )
            result["large_project_retention"] = {
                "policy": "discard_regenerable_per_evaluation_fsp_after_success",
                "retained": False,
                "discarded_projects": discarded_projects,
                "objective_gradient_npz_retained": True,
                "density_and_optimizer_checkpoints_retained": True,
            }
    except Exception as exc:
        result.update(
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            wall_s=time.monotonic() - started,
        )
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    if os.environ.get("TAIRTE4_VERBOSE_RESULT", "0") == "1":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(
            json.dumps(
                {
                    "status": result.get("status"),
                    "passed": result.get("passed"),
                    "polarization": result.get("polarization"),
                    "objective_A": result.get("objective_A"),
                    "Q_mapping_relative_error": result.get("Q_mapping", {}).get(
                        "relative_mapping_error"
                    ),
                    "wall_s": result.get("wall_s"),
                    "result_path": str(result_path),
                },
                default=str,
            )
        )
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
