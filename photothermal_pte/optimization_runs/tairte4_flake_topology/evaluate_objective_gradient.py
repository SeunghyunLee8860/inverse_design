#!/usr/bin/env python3
"""Evaluate signed PTE current and its exact current-density gradient."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
    pullback_q,
    run_forward,
    sha256,
    solve_coupled,
)

import build_nonuniform_complex_yee_jacobian as jacobian_builder
import run_production_combined_adfd_smoke as legacy_combined


def load_rho(path: Path) -> np.ndarray:
    source = path.expanduser().resolve()
    with np.load(source) as data:
        if "rho" not in data:
            raise RuntimeError("density artifact must contain rho")
        rho = np.asarray(data["rho"], dtype=np.float64)
    if rho.shape != (161, 161):
        raise RuntimeError(f"rho shape {rho.shape} != (161, 161)")
    if not np.all(np.isfinite(rho)) or np.any((rho < 0.0) | (rho > 1.0)):
        raise RuntimeError("rho must be finite in [0,1]")
    return rho


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-fsp", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--jacobian-dir", required=True, type=Path)
    parser.add_argument("--rho-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--gpu-device", default="GPU 5")
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    base_fsp = checked(args.base_fsp, args.base_sha256)
    _, _, operator_meta = load_operator(args.jacobian_dir)
    rho = load_rho(args.rho_npz)
    angle = 90.0 if args.polarization == "Ea" else 0.0
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
        adjoint = legacy_combined.run_adjoint(
            fdtd,
            audit,
            runtime,
            template=template,
            project=output / "adjoint_gpu.fsp",
        )
        gradient_optical, optical_meta = legacy_combined.optical_gradient(
            local_operator,
            forward=forward,
            adjoint=adjoint,
            pulled=pulled,
            profile_scale=profile_scale,
            base_amplitude=base_amplitude,
        )
        gradient_thermal = coupled["gradient_thermal"]
        gradient_electrical = coupled["gradient_electrical"]
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
            temperature_K=coupled["temperature"],
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
            "objective": "signed full-flake terminal PTE current",
            "objective_A": coupled["electrical"].current_A,
            "rho_range": [float(np.min(rho)), float(np.max(rho))],
            "gradient_norms_A": {
                "total": float(np.linalg.norm(gradient_total)),
                "optical": float(np.linalg.norm(gradient_optical)),
                "thermal": float(np.linalg.norm(gradient_thermal)),
                "electrical": float(np.linalg.norm(gradient_electrical)),
            },
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
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
