#!/usr/bin/env python3
"""Centered-FD validation of one full latent/filter/projection direction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy import ndimage


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
for path in (HERE, REPOSITORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from photothermal_pte.optimization_runs.cuda_thermal_adjoint import PersistentCudaCSR  # noqa: E402
from production_density_mapping import ProductionDensityMapping  # noqa: E402
from run_production_combined_adfd_smoke import (  # noqa: E402
    SCENARIO,
    boundary_energy,
    checked,
    compact_forward,
    contract_configuration,
    map_q,
    open_fdtd,
    run_forward,
)
from validate_production_thermal_material_adfd import build_state  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(value: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(value)))
    if not np.isfinite(scale) or scale == 0.0:
        raise RuntimeError("zero or nonfinite latent direction")
    return np.asarray(value, float) / scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation-result", type=Path, required=True)
    parser.add_argument("--preparation-raw", type=Path, required=True)
    parser.add_argument("--preparation-raw-sha256", required=True)
    parser.add_argument("--base-fsp", type=Path, required=True)
    parser.add_argument("--base-fsp-sha256", required=True)
    parser.add_argument("--direction", choices=("adjoint_aligned", "fixed_seed_random"), required=True)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "selected_full_latent_direction_adfd_result.json"
    result: dict[str, object] = {
        "status": "FAILED_SELECTED_FULL_LATENT_DIRECTION_ADFD",
        "passed": False,
        "direction": args.direction,
        "step": args.step,
        "Maxwell_forward_solves": 0,
        "Maxwell_adjoint_solves": 0,
        "thermal_forward_solves": 0,
        "thermal_adjoint_solves": 0,
        "optimizer_started": False,
        "CPU_FDTD_fallback": False,
        "CPU_thermal_solve_fallback": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
    }
    fdtd = None
    started = time.monotonic()
    try:
        prep_result_path = args.preparation_result.expanduser().resolve()
        prep = json.loads(prep_result_path.read_text())
        if prep.get("status") != "COMPLETED_SELECTED_FULL_LATENT_ADJOINT_PREPARATION" or not prep.get("passed"):
            raise RuntimeError("full-latent preparation did not pass")
        raw_path = checked(args.preparation_raw, args.preparation_raw_sha256, "full-latent preparation raw")
        base_fsp = checked(args.base_fsp, args.base_fsp_sha256, "base FSP")
        raw = np.load(raw_path)
        latent = np.asarray(raw["latent"], float)
        gradient_latent = np.asarray(raw["gradient_latent_A"], float)
        beta = float(prep["beta"])
        if latent.shape != gradient_latent.shape:
            raise RuntimeError("latent/gradient shape mismatch")
        if args.direction == "adjoint_aligned":
            direction = normalized(gradient_latent)
        else:
            rng = np.random.default_rng(2026080602)
            direction = ndimage.gaussian_filter(rng.normal(size=latent.shape), sigma=4.0)
            direction -= np.mean(direction)
            direction = normalized(direction)
        if args.step <= 0.0:
            raise ValueError("step must be positive")
        if np.min(latent - args.step * direction) <= 0.0 or np.max(latent + args.step * direction) >= 1.0:
            raise RuntimeError("latent FD perturbation would require clipping")
        mapping_operator = ProductionDensityMapping()
        config = contract_configuration("selected_production")
        fdtd, audit, runtime = open_fdtd(args.gpu_device)
        pair: dict[str, object] = {}
        objectives: dict[str, float] = {}
        rho_pair: dict[str, np.ndarray] = {}
        for sign, label in ((1.0, "plus"), (-1.0, "minus")):
            local_latent = latent + sign * args.step * direction
            local_rho = mapping_operator.physical(local_latent, beta)
            rho_pair[label] = local_rho
            forward = run_forward(
                fdtd,
                audit,
                runtime,
                base_fsp=base_fsp,
                rho=local_rho,
                role=f"full_latent_{args.direction}_h{args.step:g}_{label}",
                output=output,
                imported_object=str(config["imported_object"]),
                nodes=config["nodes"],
            )
            result["Maxwell_forward_solves"] = int(result["Maxwell_forward_solves"]) + 1
            data, q_mapping = map_q(forward["q"], design_half_span_m=float(config["design_half_span_m"]))
            state = build_state(data, SCENARIO, config["density_forward"](local_rho))
            solve = PersistentCudaCSR(state.system.matrix_W_K, cuda_device=args.cuda_device).solve(
                state.source_power_W,
                relative_tolerance=1.0e-10,
                max_iterations=30000,
            )
            result["thermal_forward_solves"] = int(result["thermal_forward_solves"]) + 1
            objective = float(np.dot(state.c_A_K, solve.solution))
            objectives[label] = objective
            pair[label] = {
                "objective_A": objective,
                "objective_A_per_incident_W": objective / float(prep["incident_power_W"]),
                "latent_range": [float(np.min(local_latent)), float(np.max(local_latent))],
                "physical_density_range": [float(np.min(local_rho)), float(np.max(local_rho))],
                "forward": compact_forward(forward),
                "mapping": q_mapping,
                "thermal_residual": solve.explicit_relative_residual,
                "thermal_energy_balance": boundary_energy(state, solve.solution),
                "thermal_solve_seconds": solve.solve_seconds,
            }
        fd = (objectives["plus"] - objectives["minus"]) / (2.0 * args.step)
        ad = float(np.sum(gradient_latent * direction))
        absolute_error = abs(ad - fd)
        relative_error = absolute_error / max(abs(ad), abs(fd), np.finfo(float).tiny)
        normalized_error = absolute_error / max(float(np.linalg.norm(gradient_latent) * np.linalg.norm(direction)), np.finfo(float).tiny)
        worst_closure = max(float(pair[label]["forward"]["closure"]) for label in pair)
        worst_mapping = max(float(pair[label]["mapping"]["internal_relative_power_error"]) for label in pair)
        worst_residual = max(float(pair[label]["thermal_residual"]) for label in pair)
        worst_energy = max(float(pair[label]["thermal_energy_balance"]) for label in pair)
        worst_shutoff = max(float(pair[label]["forward"]["log_audit"]["final_auto_shutoff"]) for label in pair)
        strong = max(abs(ad), abs(fd)) >= 1.0e-22
        passed = bool(
            (relative_error < 0.01 if strong else normalized_error < 0.01)
            and normalized_error < 0.01
            and worst_closure < 0.005
            and worst_mapping < 0.005
            and worst_residual < 1.0e-8
            and worst_energy < 0.01
            and worst_shutoff < 1.0e-5
        )
        raw_out = output / "selected_full_latent_direction_adfd.npz"
        np.savez_compressed(
            raw_out,
            latent=latent,
            direction=direction,
            rho_base=mapping_operator.physical(latent, beta),
            rho_plus=rho_pair["plus"],
            rho_minus=rho_pair["minus"],
            gradient_latent_A=gradient_latent,
        )
        result.update({
            "status": "VALIDATED_SELECTED_FULL_LATENT_DIRECTION_ADFD" if passed else "FAILED_SELECTED_FULL_LATENT_DIRECTION_ADFD",
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "beta": beta,
            "adjoint_directional_A": ad,
            "finite_difference_directional_A": fd,
            "relative_error": relative_error,
            "multi_direction_normalized_error": normalized_error,
            "direction_is_strong": strong,
            "pair": pair,
            "gates": {
                "directional_relative_error": relative_error,
                "multi_direction_normalized_error": normalized_error,
                "limit": 0.01,
                "worst_optical_closure": worst_closure,
                "worst_Q_mapping_error": worst_mapping,
                "worst_thermal_residual": worst_residual,
                "worst_thermal_energy_balance": worst_energy,
                "worst_auto_shutoff": worst_shutoff,
            },
            "inputs": {
                "preparation_result": {"path": str(prep_result_path), "sha256": sha256(prep_result_path)},
                "preparation_raw": {"path": str(raw_path), "sha256": sha256(raw_path)},
                "base_FSP": {"path": str(base_fsp), "sha256": sha256(base_fsp)},
            },
            "raw_artifact": {"path": str(raw_out), "size_bytes": raw_out.stat().st_size, "sha256": sha256(raw_out)},
            "wall_s": time.monotonic() - started,
        })
    except Exception as exc:
        result.update({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "wall_s": time.monotonic() - started})
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
