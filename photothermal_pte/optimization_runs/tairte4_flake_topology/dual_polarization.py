"""Shared-geometry E||a/E||b objective evaluation and exact validation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.run_true_mma_optimization import (
    REFERENCE_INCIDENT_POWER_W,
    REPOSITORY,
    evaluate,
    sha256,
    write_json,
)


POLARIZATIONS = ("Ea", "Eb")
SOFTMIN_TEMPERATURE_AT_REFERENCE_POWER_A = 5.0e-9


def smooth_min_and_gradient(
    objectives_A: dict[str, float],
    gradients_A: dict[str, np.ndarray],
    temperature_A: float,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Return a stable log-mean-exp soft minimum and analytic gradient."""

    if set(objectives_A) != set(POLARIZATIONS):
        raise ValueError("dual objective requires exactly Ea and Eb")
    if set(gradients_A) != set(POLARIZATIONS):
        raise ValueError("dual gradient requires exactly Ea and Eb")
    ia = float(objectives_A["Ea"])
    ib = float(objectives_A["Eb"])
    if not np.isfinite(ia) or not np.isfinite(ib):
        raise RuntimeError("dual smooth-min objective requires finite currents")
    temperature = float(temperature_A)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("smooth-min temperature must be positive and finite")
    ga = np.asarray(gradients_A["Ea"], dtype=np.float64)
    gb = np.asarray(gradients_A["Eb"], dtype=np.float64)
    if (
        ga.shape != gb.shape
        or not np.all(np.isfinite(ga))
        or not np.all(np.isfinite(gb))
    ):
        raise RuntimeError("dual polarization gradients are incompatible or nonfinite")
    logits = -np.asarray((ia, ib), dtype=np.float64) / temperature
    maximum = float(np.max(logits))
    exponentials = np.exp(logits - maximum)
    normalized = exponentials / np.sum(exponentials)
    weights = {"Ea": float(normalized[0]), "Eb": float(normalized[1])}
    objective = -temperature * (
        maximum + float(np.log(0.5 * np.sum(exponentials)))
    )
    gradient = weights["Ea"] * ga + weights["Eb"] * gb
    return float(objective), gradient, weights


def _load_cached_combined(
    result_path: Path,
) -> tuple[dict[str, object], np.ndarray, np.ndarray] | None:
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text())
    if not result.get("passed"):
        return None
    raw_path = Path(result["raw_artifact"]["path"])
    if not raw_path.is_file() or sha256(raw_path) != result["raw_artifact"]["sha256"]:
        raise RuntimeError("cached dual-polarization raw artifact is absent or changed")
    with np.load(raw_path) as raw:
        return (
            result,
            np.asarray(raw["gradient_total_A"], dtype=np.float64),
            np.asarray(raw["gradient_terminal_conductance_S"], dtype=np.float64),
        )


def evaluate_dual_polarization(
    rho: np.ndarray,
    *,
    polarization: str,
    output: Path,
    gpu: int,
    events: Path,
    base_fsp: Path,
    base_sha256: str,
    jacobian_dir: Path,
    latent: np.ndarray | None = None,
    dfm_beta: float | None = None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Evaluate both incident polarizations on one density and combine them."""

    del polarization
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "objective_gradient_result.json"
    cached = _load_cached_combined(result_path)
    if cached is not None:
        return cached

    results: dict[str, dict[str, object]] = {}
    gradients: dict[str, np.ndarray] = {}
    conductance_gradients: dict[str, np.ndarray] = {}
    for name in POLARIZATIONS:
        result, gradient, conductance_gradient = evaluate(
            rho,
            polarization=name,
            output=output / name,
            gpu=gpu,
            events=events,
            base_fsp=base_fsp,
            base_sha256=base_sha256,
            jacobian_dir=jacobian_dir,
            latent=latent if name == "Ea" else None,
            dfm_beta=dfm_beta if name == "Ea" else None,
        )
        results[name] = result
        gradients[name] = gradient
        conductance_gradients[name] = conductance_gradient

    source_powers = {
        name: float(results[name]["forward"]["source_power_W"])
        for name in POLARIZATIONS
    }
    source_spread = abs(source_powers["Ea"] - source_powers["Eb"]) / max(
        min(source_powers.values()), np.finfo(float).tiny
    )
    if source_spread >= 0.005:
        raise RuntimeError("Ea/Eb source powers disagree by at least 0.5%")
    objectives = {
        name: float(results[name]["objective_A"]) for name in POLARIZATIONS
    }
    source_power_mean = 0.5 * (source_powers["Ea"] + source_powers["Eb"])
    temperature_A = (
        SOFTMIN_TEMPERATURE_AT_REFERENCE_POWER_A
        * source_power_mean
        / REFERENCE_INCIDENT_POWER_W
    )
    objective, combined_gradient, weights = smooth_min_and_gradient(
        objectives, gradients, temperature_A
    )
    conductances = {
        name: float(results[name]["terminal_conductance_S"])
        for name in POLARIZATIONS
    }
    conductance_spread = abs(conductances["Ea"] - conductances["Eb"]) / max(
        min(conductances.values()), np.finfo(float).tiny
    )
    if conductance_spread >= 1.0e-6:
        raise RuntimeError("polarization-independent terminal conductance changed")
    conductance = 0.5 * (conductances["Ea"] + conductances["Eb"])
    conductance_gradient = 0.5 * (
        conductance_gradients["Ea"] + conductance_gradients["Eb"]
    )

    ea_raw = Path(results["Ea"]["raw_artifact"]["path"])
    with np.load(ea_raw) as raw:
        indicators = np.asarray(raw["ansys_dfm_indicators"], dtype=np.float64)
        indicator_gradient = np.asarray(
            raw["ansys_dfm_gradient_latent"], dtype=np.float64
        )
    raw_path = output / "objective_gradient.npz"
    np.savez_compressed(
        raw_path,
        rho=np.asarray(rho, dtype=np.float64),
        objective_A=np.asarray(objective),
        objective_Ea_A=np.asarray(objectives["Ea"]),
        objective_Eb_A=np.asarray(objectives["Eb"]),
        gradient_total_A=combined_gradient,
        gradient_Ea_A=gradients["Ea"],
        gradient_Eb_A=gradients["Eb"],
        terminal_conductance_S=np.asarray(conductance),
        gradient_terminal_conductance_S=conductance_gradient,
        ansys_dfm_indicators=indicators,
        ansys_dfm_gradient_latent=indicator_gradient,
    )
    result: dict[str, object] = {
        "status": "VALIDATED_DUAL_POLARIZATION_OBJECTIVE_GRADIENT",
        "passed": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "polarization": "Ea+Eb",
        "axis_contract": "Lumerical x=b, y=a, z=c",
        "objective": "smooth worst-case of signed Ea and Eb terminal PTE currents",
        "objective_aggregation": {
            "type": "log_mean_exp_soft_minimum",
            "formula": "-tau*log(mean(exp(-I/tau)))",
            "temperature_A": temperature_A,
            "temperature_at_285uW_A": SOFTMIN_TEMPERATURE_AT_REFERENCE_POWER_A,
            "gradient_weights": weights,
        },
        "objective_A": objective,
        "polarization_objectives_A": objectives,
        "source_powers_W": source_powers,
        "source_power_relative_spread": source_spread,
        "terminal_conductance_S": conductance,
        "polarization_terminal_conductances_S": conductances,
        "thermal_interface_contract": results["Ea"]["thermal_interface_contract"],
        "ansys_minimum_feature": results["Ea"].get("ansys_minimum_feature"),
        "forward": results["Ea"]["forward"],
        "polarization_results": {
            name: {
                "result_path": str(output / name / "objective_gradient_result.json"),
                "objective_A": objectives[name],
                "terminal_conductance_S": conductances[name],
                "raw_artifact": results[name]["raw_artifact"],
                "forward": results[name]["forward"],
            }
            for name in POLARIZATIONS
        },
        "raw_artifact": {
            "path": str(raw_path),
            "size_bytes": raw_path.stat().st_size,
            "sha256": sha256(raw_path),
        },
        "Maxwell_solves": {"forward": 2, "adjoint": 2},
        "empirical_normalization": False,
        "gradient_rescaling": False,
    }
    write_json(result_path, result)
    return result, combined_gradient, conductance_gradient


def evaluate_dual_exact_candidate(
    *,
    rho: np.ndarray,
    rank: int,
    candidate_root: Path,
    gpu: int,
    base_fsp: Path,
    base_sha256: str,
    reference_objectives_A: dict[str, float],
) -> dict[str, object]:
    """Freshly validate one exact binary candidate for both polarizations."""

    candidate_root.mkdir(parents=True, exist_ok=True)
    density = candidate_root / f"exact_candidate_{rank:02d}.npz"
    output = candidate_root / f"exact_candidate_{rank:02d}_dual_physics"
    np.savez_compressed(density, rho=np.asarray(rho, dtype=np.float64))
    results: dict[str, dict[str, Any]] = {}
    for name in POLARIZATIONS:
        polarization_output = output / name
        command = [
            sys.executable,
            "-m",
            "photothermal_pte.optimization_runs.tairte4_flake_topology.evaluate_binary_objective",
            "--base-fsp",
            str(base_fsp),
            "--base-sha256",
            base_sha256,
            "--rho-npz",
            str(density),
            "--output-dir",
            str(polarization_output),
            "--polarization",
            name,
            "--gpu-device",
            f"GPU {gpu}",
            "--cuda-device",
            "0",
            "--reference-objective-A",
            str(float(reference_objectives_A[name])),
        ]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        completed = subprocess.run(command, cwd=REPOSITORY, env=environment)
        result_path = polarization_output / "binary_objective_result.json"
        if not result_path.is_file():
            raise RuntimeError(f"exact candidate {rank} {name} produced no result")
        result = json.loads(result_path.read_text())
        physical = bool(result.get("physical_gates_passed", False))
        if completed.returncode not in (0, 1) or not physical:
            raise RuntimeError(f"exact candidate {rank} {name} failed physical gates")
        results[name] = result

    objectives = {
        name: float(results[name]["objective_A"]) for name in POLARIZATIONS
    }
    source_power_mean = 0.5 * sum(
        float(results[name]["forward"]["source_power_W"])
        for name in POLARIZATIONS
    )
    temperature_A = (
        SOFTMIN_TEMPERATURE_AT_REFERENCE_POWER_A
        * source_power_mean
        / REFERENCE_INCIDENT_POWER_W
    )
    dummy = {name: np.zeros(1, dtype=np.float64) for name in POLARIZATIONS}
    objective, _, _ = smooth_min_and_gradient(objectives, dummy, temperature_A)
    physical_passed = all(
        bool(results[name]["physical_gates_passed"]) for name in POLARIZATIONS
    )
    per_polarization_gate = {
        name: bool(results[name]["binary_objective_preserved_within_one_percent"])
        for name in POLARIZATIONS
    }
    objective_gate_passed = all(per_polarization_gate.values())
    combined = {
        "schema": "dual-polarization-exact-binary-objective-v1",
        "status": (
            "VALIDATED_DUAL_POLARIZATION_EXACT_BINARY_OBJECTIVE"
            if physical_passed and objective_gate_passed
            else "FAILED_DUAL_POLARIZATION_EXACT_BINARY_OBJECTIVE"
        ),
        "passed": bool(physical_passed and objective_gate_passed),
        "objective_A": objective,
        "polarization_objectives_A": objectives,
        "physical_gates_passed": physical_passed,
        "objective_gate_passed": objective_gate_passed,
        "per_polarization_objective_gate_passed": per_polarization_gate,
        "polarization_results": results,
    }
    combined_path = output / "dual_binary_objective_result.json"
    write_json(combined_path, combined)
    return {
        "rank": rank,
        "density": {
            "path": str(density),
            "size_bytes": density.stat().st_size,
            "sha256": sha256(density),
        },
        "result_path": str(combined_path),
        "objective_A": objective,
        "physical_gates_passed": physical_passed,
        "objective_gate_passed": objective_gate_passed,
        "result": combined,
    }
