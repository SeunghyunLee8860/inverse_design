#!/usr/bin/env python3
"""Fresh GPU-Maxwell/CUDA-thermal/electrical evaluation of an exact binary design."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.evaluate_objective_gradient import (
    load_rho,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.validate_combined_adfd import (
    checked,
    compact_forward,
    open_fdtd,
    polarization_angle,
    run_forward,
    sha256,
    solve_coupled,
)


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-fsp", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--rho-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--gpu-device", default="GPU 5")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--reference-objective-A", type=float, default=None)
    args = parser.parse_args()

    base_fsp = checked(args.base_fsp, args.base_sha256)
    rho_path = args.rho_npz.expanduser().resolve()
    rho = load_rho(rho_path)
    if not np.all((rho == 0.0) | (rho == 1.0)):
        raise RuntimeError("objective-only final evaluation requires exact 0/1 density")
    angle = polarization_angle(args.polarization)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "binary_objective_result.json"
    result: dict[str, object] = {
        "status": "FAILED_EXACT_BINARY_OBJECTIVE_VALIDATION",
        "passed": False,
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
            role=f"exact_binary_forward_{args.polarization}",
            output=output,
            polarization_angle_deg=angle,
        )
        # No endpoint Jacobian or Maxwell adjoint is needed for an exact-binary
        # objective certificate.  Release the largest unused field arrays.
        forward.pop("electric", None)
        forward.pop("epsilon", None)
        forward.pop("index", None)
        coupled = solve_coupled(forward, rho, args.cuda_device, need_adjoint=False)
        objective = float(coupled["electrical"].current_A)
        reference_change = None
        if args.reference_objective_A is not None:
            reference_change = (objective - args.reference_objective_A) / abs(
                args.reference_objective_A
            )
        raw = output / "binary_objective_fields.npz"
        np.savez_compressed(
            raw,
            rho_binary=rho.astype(np.uint8),
            mapped_Q_W_m3=np.asarray(coupled["mapped_q"], dtype=np.float64),
            nodal_temperature_K=np.asarray(coupled["temperature"], dtype=np.float64),
            weighting_potential=np.asarray(
                coupled["electrical"].weighting_potential, dtype=np.float64
            ),
            weighting_gradient_element_m_inv=np.asarray(
                coupled["electrical"].weighting_gradient_element_m_inv,
                dtype=np.float64,
            ),
        )
        passed = bool(
            forward["closure"] < 0.005
            and coupled["mapping"]["relative_mapping_error"] < 0.005
            and coupled["thermal_forward"].explicit_relative_residual < 1e-8
            and coupled["energy"] < 0.01
            and coupled["electrical"].weighting_residual < 1e-8
            and np.isfinite(objective)
            and coupled["electrical"].terminal_conductance_S > 0.0
        )
        result = {
            "schema": "contact-anchored-exact-binary-objective-v1",
            "status": (
                "VALIDATED_EXACT_BINARY_GPU_MAXWELL_CUDA_THERMAL_ELECTRICAL_OBJECTIVE"
                if passed
                else "FAILED_EXACT_BINARY_OBJECTIVE_VALIDATION"
            ),
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "polarization": args.polarization,
            "polarization_angle_deg": angle,
            "axis_contract": "Lumerical x=b, y=a, z=c",
            "objective_A": objective,
            "reference_continuous_objective_A": args.reference_objective_A,
            "relative_objective_change_from_continuous": reference_change,
            "equivalent_objective_at_285uW_A": objective * 285.0e-6 / forward["source_power_W"],
            "responsivity_A_W": objective / forward["source_power_W"],
            "terminal_conductance_S": coupled["electrical"].terminal_conductance_S,
            "rho_values": np.unique(rho).tolist(),
            "forward": compact_forward(forward),
            "gates": {
                "optical_closure": forward["closure"],
                "Q_mapping_error": coupled["mapping"]["relative_mapping_error"],
                "thermal_forward_residual": coupled["thermal_forward"].explicit_relative_residual,
                "thermal_energy_balance": coupled["energy"],
                "electrical_weighting_residual": coupled["electrical"].weighting_residual,
            },
            "inputs": {"base_FSP": artifact(base_fsp), "binary_density": artifact(rho_path)},
            "raw_artifact": artifact(raw),
            "Maxwell_solves": {"forward": 1, "adjoint": 0},
            "posthoc_objective_or_gradient_rescaling": False,
            "Q_clipping_smoothing_gain_or_rescaling": False,
            "CPU_FDTD_fallback": False,
            "CPU_thermal_linear_solve_fallback": False,
            "wall_s": time.monotonic() - started,
        }
    except Exception as error:
        result.update(
            error=f"{type(error).__name__}: {error}",
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
