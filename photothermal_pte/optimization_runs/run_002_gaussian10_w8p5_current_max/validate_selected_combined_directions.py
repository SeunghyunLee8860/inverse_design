#!/usr/bin/env python3
"""Validate independent selected-grid combined physical-density directions.

The corrected optical and already validated fixed-Q thermal gradients are
SHA-pinned inputs.  Each invocation performs exactly two GPU Maxwell forward
solves and two CUDA thermal solves for one centered-FD direction.
"""

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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.optimization_runs.cuda_thermal_adjoint import (  # noqa: E402
    PersistentCudaCSR,
)
from run_production_combined_adfd_smoke import (  # noqa: E402
    SCENARIO,
    boundary_energy,
    checked,
    compact_forward,
    contract_configuration,
    map_q,
    open_fdtd,
    relative,
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
        raise RuntimeError("zero or nonfinite validation direction")
    return np.asarray(value, float) / scale


def direction_field(name: str, shape: tuple[int, int]) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, shape[0])
    y = np.linspace(-1.0, 1.0, shape[1])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    if name == "smooth_asymmetric":
        value = (
            np.sin(1.2 * np.pi * xx)
            + 0.43 * np.cos(0.7 * np.pi * yy)
            + 0.17 * xx * yy
        )
    elif name == "central_localized":
        value = np.exp(-(xx**2 + yy**2) / (2.0 * 0.12**2))
    elif name == "design_edge_localized":
        value = np.exp(
            -((xx + 0.91) ** 2 + (yy - 0.22) ** 2) / (2.0 * 0.08**2)
        )
    elif name == "fixed_seed_random":
        rng = np.random.default_rng(20260806)
        value = ndimage.gaussian_filter(rng.normal(size=shape), sigma=4.0)
        value -= np.mean(value)
    else:
        raise ValueError(f"unknown direction {name!r}")
    return normalized(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-result", type=Path, required=True)
    parser.add_argument("--combined-raw", type=Path, required=True)
    parser.add_argument("--combined-raw-sha256", required=True)
    parser.add_argument("--corrected-optical-gradient", type=Path, required=True)
    parser.add_argument("--corrected-optical-gradient-sha256", required=True)
    parser.add_argument("--base-fsp", type=Path, required=True)
    parser.add_argument("--base-fsp-sha256", required=True)
    parser.add_argument(
        "--direction",
        choices=(
            "smooth_asymmetric",
            "central_localized",
            "design_edge_localized",
            "fixed_seed_random",
        ),
        required=True,
    )
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--incident-power-W", type=float, default=1.3822261103022194e-13)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "selected_combined_direction_adfd_result.json"
    result: dict[str, object] = {
        "status": "FAILED_SELECTED_COMBINED_DIRECTION_ADFD",
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
        combined_result = args.combined_result.expanduser().resolve()
        combined = json.loads(combined_result.read_text())
        combined_raw = checked(
            args.combined_raw,
            args.combined_raw_sha256,
            "combined physical-density raw",
        )
        optical_path = checked(
            args.corrected_optical_gradient,
            args.corrected_optical_gradient_sha256,
            "corrected optical gradient",
        )
        base_fsp = checked(args.base_fsp, args.base_fsp_sha256, "base FSP")
        raw = np.load(combined_raw)
        optical = np.load(optical_path)
        rho = np.asarray(raw["rho"], float)
        thermal_gradient = np.asarray(raw["gradient_thermal_A"], float)
        optical_gradient = np.asarray(optical["gradient_total_optical_A"], float)
        if optical_gradient.shape != rho.shape or thermal_gradient.shape != rho.shape:
            raise RuntimeError("gradient/density shape mismatch")
        total_gradient = optical_gradient + thermal_gradient
        direction = direction_field(args.direction, rho.shape)
        if args.step <= 0.0:
            raise ValueError("step must be positive")
        if (
            float(np.min(rho - args.step * direction)) <= 0.0
            or float(np.max(rho + args.step * direction)) >= 1.0
        ):
            raise RuntimeError("FD perturbation would require clipping")

        config = contract_configuration("selected_production")
        fdtd, audit, runtime = open_fdtd(args.gpu_device)
        pair: dict[str, object] = {}
        objectives: dict[str, float] = {}
        for sign, label in ((1.0, "plus"), (-1.0, "minus")):
            local_rho = rho + sign * args.step * direction
            forward = run_forward(
                fdtd,
                audit,
                runtime,
                base_fsp=base_fsp,
                rho=local_rho,
                role=f"{args.direction}_h{args.step:g}_{label}",
                output=output,
                imported_object=str(config["imported_object"]),
                nodes=config["nodes"],
            )
            result["Maxwell_forward_solves"] = int(result["Maxwell_forward_solves"]) + 1
            data, mapping = map_q(
                forward["q"],
                design_half_span_m=float(config["design_half_span_m"]),
            )
            state = build_state(
                data,
                SCENARIO,
                config["density_forward"](local_rho),
            )
            solve = PersistentCudaCSR(
                state.system.matrix_W_K,
                cuda_device=args.cuda_device,
            ).solve(
                state.source_power_W,
                relative_tolerance=1.0e-10,
                max_iterations=30000,
            )
            result["thermal_forward_solves"] = int(result["thermal_forward_solves"]) + 1
            objective = float(np.dot(state.c_A_K, solve.solution))
            objectives[label] = objective
            pair[label] = {
                "objective_A": objective,
                "objective_A_per_incident_W": objective / args.incident_power_W,
                "forward": compact_forward(forward),
                "mapping": mapping,
                "thermal_residual": solve.explicit_relative_residual,
                "thermal_energy_balance": boundary_energy(state, solve.solution),
                "thermal_solve_seconds": solve.solve_seconds,
            }

        finite_difference = (objectives["plus"] - objectives["minus"]) / (
            2.0 * args.step
        )
        adjoint = float(np.sum(total_gradient * direction))
        absolute_error = abs(adjoint - finite_difference)
        strong_scale = max(
            abs(adjoint), abs(finite_difference), np.finfo(float).tiny
        )
        relative_error = absolute_error / strong_scale
        gradient_scale = max(
            float(np.linalg.norm(total_gradient) * np.linalg.norm(direction)),
            np.finfo(float).tiny,
        )
        normalized_error = absolute_error / gradient_scale
        worst_closure = max(
            float(pair[label]["forward"]["closure"]) for label in ("plus", "minus")
        )
        worst_mapping = max(
            float(pair[label]["mapping"]["internal_relative_power_error"])
            for label in ("plus", "minus")
        )
        worst_residual = max(
            float(pair[label]["thermal_residual"]) for label in ("plus", "minus")
        )
        worst_energy = max(
            float(pair[label]["thermal_energy_balance"])
            for label in ("plus", "minus")
        )
        worst_shutoff = max(
            float(pair[label]["forward"]["log_audit"]["final_auto_shutoff"])
            for label in ("plus", "minus")
        )
        strong = max(abs(adjoint), abs(finite_difference)) >= 1.0e-22
        passed = bool(
            (relative_error < 0.01 if strong else normalized_error < 0.01)
            and normalized_error < 0.01
            and worst_closure < 0.005
            and worst_mapping < 0.005
            and worst_residual < 1.0e-8
            and worst_energy < 0.01
            and worst_shutoff < 1.0e-5
        )
        direction_path = output / "selected_combined_direction_adfd.npz"
        np.savez_compressed(
            direction_path,
            rho=rho,
            direction=direction,
            gradient_total_A=total_gradient,
            gradient_optical_A=optical_gradient,
            gradient_thermal_A=thermal_gradient,
        )
        result.update(
            {
                "status": (
                    "VALIDATED_SELECTED_COMBINED_DIRECTION_ADFD"
                    if passed
                    else "FAILED_SELECTED_COMBINED_DIRECTION_ADFD"
                ),
                "passed": passed,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "adjoint_directional_A": adjoint,
                "finite_difference_directional_A": finite_difference,
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
                    "combined_result": {
                        "path": str(combined_result),
                        "sha256": sha256(combined_result),
                    },
                    "combined_raw": {
                        "path": str(combined_raw),
                        "sha256": sha256(combined_raw),
                    },
                    "corrected_optical_gradient": {
                        "path": str(optical_path),
                        "sha256": sha256(optical_path),
                    },
                    "base_FSP": {"path": str(base_fsp), "sha256": sha256(base_fsp)},
                },
                "raw_artifact": {
                    "path": str(direction_path),
                    "size_bytes": direction_path.stat().st_size,
                    "sha256": sha256(direction_path),
                },
                "wall_s": time.monotonic() - started,
            }
        )
    except Exception as exc:
        result.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "wall_s": time.monotonic() - started,
            }
        )
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
