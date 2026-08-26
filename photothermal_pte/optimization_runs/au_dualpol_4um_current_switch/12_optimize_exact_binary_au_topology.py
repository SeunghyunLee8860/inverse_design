#!/usr/bin/env python3
"""Adjoint-guided exact-binary Au topology finishing.

The continuous LD_MMA run found the requested opposite currents, but all
post-processed exact binaries lost the Eb sign.  This finishing stage never
uses a gray material.  Each update adds/removes a physical 500 nm disk,
repairs both phases with exact morphology, rejects any exact-DFM violation,
and ranks a bounded set of candidates with the full Maxwell--thermal--
electrical operator.  The combined adjoint is used only to propose candidates;
promotion is always decided by forward physics.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.combined_4um import (
    CompiledOpticalRunner,
    combined_gradient,
    evaluate_forward_multiphysics,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import CONTRACT
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.dfm import (
    exact_500nm_audit,
    physical_disk_footprint,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    N_DESIGN,
    N_TA,
    current_integrand,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    audit as material_fraction_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.production_readiness import (
    calibrated_source_scales,
    require_production_readiness,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.paths import (
    raw_path,
)
HERE = Path(__file__).resolve().parent
OUT = HERE / "results_4um_dualpol_au_exact_binary_search"
RAW = raw_path("exact_binary_search")
INITIAL = Path(
    os.environ.get(
        "AU_EXACT_INITIAL_NPZ",
        str(
            raw_path(
                "optimization_ld_mma",
                "final_binary_threshold_0.50_radius_450nm_close_open.npz",
            )
        ),
    )
)
MAX_STEPS = int(os.environ.get("AU_EXACT_MAX_STEPS", "12"))
PROPOSALS_PER_STEP = int(os.environ.get("AU_EXACT_PROPOSALS", "6"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def optical_closure(result, runner, scale: float) -> tuple[float, float, float]:
    eta0 = float(runner.model["fdtdx"].constants.eta0)
    p_six = scale * eta0 * float(
        np.mean(
            np.asarray(
                result["optical_output"].detector_states["material_flux_td"][
                    "poynting_flux"
                ]
            )[:, 0]
        )
    )
    p_q = float(np.sum(result["source_power_W"]))
    return p_q, p_six, abs(p_q - p_six) / max(abs(p_six), np.finfo(float).tiny)


def audit_gates(result, closure: float) -> dict[str, bool]:
    q_finite = all(
        np.all(np.isfinite(value)) and float(np.min(value)) >= 0.0
        for value in result["q_fields_W_m3"].values()
    )
    return {
        "optical_closure_lt_0p5pct": bool(closure < 0.005),
        "thermal_energy_balance_lt_1pct": bool(
            result["thermal_audit"]["energy_balance_relative"] < 0.01
        ),
        "thermal_residual_lt_1e8": bool(
            result["thermal_audit"]["relative_residual"] < 1.0e-8
        ),
        "electrical_residual_lt_1e8": bool(
            result["electrical_audit"]["relative_residual"] < 1.0e-8
        ),
        "finite_nonnegative_q": q_finite,
    }


def evaluate_forward(runners, rho, source_scales, cuda_device, raw_tag=None):
    cases = {}
    for pol in ("Ea", "Eb"):
        source_scale = float(source_scales[pol])
        result = evaluate_forward_multiphysics(
            runners[pol], rho, source_scale, cuda_device, need_gradient=False
        )
        p_q, p_six, closure = optical_closure(result, runners[pol], source_scale)
        gates = audit_gates(result, closure)
        if not all(gates.values()):
            raise RuntimeError(f"fail-closed exact search {pol}: {gates}")
        case = {
            "current_A": float(result["objective_A"]),
            "P_Q_W": p_q,
            "P_six_W": p_six,
            "closure_relative": closure,
            "Tmax_K": float(np.max(result["temperature"])),
            "gates": gates,
        }
        if raw_tag is not None:
            weighting = np.asarray(result["weighting"], dtype=np.float64)
            raw = RAW / f"{raw_tag}_{pol}_fields.npz"
            np.savez_compressed(
                raw,
                physical_density=rho,
                q_au_W_m3=np.asarray(result["q_fields_W_m3"]["au"]),
                q_tairte4_W_m3=np.asarray(result["q_fields_W_m3"]["tairte4"]),
                dual_volume_au_m3=np.asarray(runners[pol].volumes["au"]),
                dual_volume_tairte4_m3=np.asarray(runners[pol].volumes["tairte4"]),
                ta_temperature_K=np.asarray(result["ta_temperature"]),
                weighting_tairte4=weighting[: N_TA * N_TA].reshape(N_TA, N_TA),
                weighting_au=weighting[N_TA * N_TA :].reshape(N_DESIGN, N_DESIGN),
                current_integrand_A_m2=current_integrand(
                    np.asarray(result["ta_temperature"]),
                    weighting,
                    electrical_system=result["electrical_system"],
                ),
            )
            case["raw_fields"] = {
                "path": str(raw.resolve()),
                "bytes": raw.stat().st_size,
                "sha256": sha256(raw),
                "committed_to_git": False,
            }
        cases[pol] = case
        del result
        gc.collect()
    ia = float(cases["Ea"]["current_A"])
    ib = float(cases["Eb"]["current_A"])
    return {
        "I_a_A": ia,
        "I_b_A": ib,
        "balanced_utility_A": min(ia, -ib),
        "cases": cases,
    }


def evaluate_gradient(runners, rho, source_scales, cuda_device):
    values = {}
    for pol in ("Ea", "Eb"):
        source_scale = float(source_scales[pol])
        result = combined_gradient(runners[pol], rho, source_scale, cuda_device)
        _, _, closure = optical_closure(result, runners[pol], source_scale)
        gates = audit_gates(result, closure)
        gates["mapping_transpose_lt_1e12"] = bool(
            result["weighted_contraction_relative_error"] < 1.0e-12
        )
        if not all(gates.values()):
            raise RuntimeError(f"fail-closed exact-search gradient {pol}: {gates}")
        values[pol] = {
            "current_A": float(result["objective_A"]),
            "gradient_A": np.asarray(result["gradient_total_A"], dtype=np.float64),
            "gradient_optical_A": np.asarray(result["gradient_optical_A"], dtype=np.float64),
            "gradient_thermal_A": np.asarray(result["gradient_thermal_A"], dtype=np.float64),
            "gradient_electrical_A": np.asarray(result["gradient_electrical_A"], dtype=np.float64),
            "gates": gates,
        }
        del result
        gc.collect()
    return values


def shifted_footprint(center, footprint, shape):
    result = np.zeros(shape, dtype=bool)
    hx, hy = footprint.shape[0] // 2, footprint.shape[1] // 2
    ci, cj = center
    lo_i, hi_i = max(0, ci - hx), min(shape[0], ci + hx + 1)
    lo_j, hi_j = max(0, cj - hy), min(shape[1], cj + hy + 1)
    fi0, fj0 = lo_i - (ci - hx), lo_j - (cj - hy)
    result[lo_i:hi_i, lo_j:hi_j] = footprint[
        fi0 : fi0 + hi_i - lo_i, fj0 : fj0 + hi_j - lo_j
    ]
    return result


def cleanup_variants(seed: np.ndarray, footprint: np.ndarray):
    variants = []
    for name, sequence in (
        ("none", ()),
        ("open_close_open", ("open", "close", "open")),
        ("close_open", ("close", "open")),
        ("close_open_close", ("close", "open", "close")),
    ):
        binary = seed.copy()
        for operation in sequence:
            if operation == "open":
                binary = ndimage.binary_opening(binary, structure=footprint, border_value=0)
            else:
                binary = ~ndimage.binary_opening(~binary, structure=footprint, border_value=1)
        audit = exact_500nm_audit(binary.astype(float))
        if audit["solid_pass"] and audit["void_pass"]:
            variants.append((name, binary))
    return variants


def separated_top_centers(score: np.ndarray, valid: np.ndarray, count: int):
    work = np.where(valid, score, -np.inf).copy()
    centers = []
    exclusion = 5
    for _ in range(count):
        index = int(np.argmax(work))
        if not np.isfinite(work.ravel()[index]):
            break
        i, j = np.unravel_index(index, work.shape)
        centers.append((i, j))
        work[max(0, i-exclusion):i+exclusion+1, max(0, j-exclusion):j+exclusion+1] = -np.inf
    return centers


def propose(binary: np.ndarray, active_gradient: np.ndarray, limit: int):
    footprint = physical_disk_footprint(
        0.5 * CONTRACT.minimum_solid_feature_m, CONTRACT.design_pitch_m
    )
    kernel = footprint.astype(float)
    disk_score = ndimage.convolve(active_gradient, kernel, mode="constant", cval=0.0)
    phase_fraction = ndimage.convolve(binary.astype(float), kernel, mode="constant", cval=0.0)
    disk_count = float(np.count_nonzero(footprint))
    add_centers = separated_top_centers(
        disk_score, phase_fraction < disk_count - 0.5, 5
    )
    remove_centers = separated_top_centers(
        -disk_score, phase_fraction > 0.5, 5
    )
    seeds = []
    for operation, centers in (("add", add_centers), ("remove", remove_centers)):
        for center in centers:
            disk = shifted_footprint(center, footprint, binary.shape)
            changed = binary | disk if operation == "add" else binary & ~disk
            seeds.append((f"{operation}_{center[0]}_{center[1]}", changed))
    if add_centers and remove_centers:
        for add_center in add_centers[:2]:
            for remove_center in remove_centers[:2]:
                changed = binary | shifted_footprint(add_center, footprint, binary.shape)
                changed &= ~shifted_footprint(remove_center, footprint, binary.shape)
                seeds.append(
                    (f"add_{add_center[0]}_{add_center[1]}_remove_{remove_center[0]}_{remove_center[1]}", changed)
                )
    ranked = []
    for seed_name, seed in seeds:
        for cleanup_name, candidate in cleanup_variants(seed, footprint):
            if np.array_equal(candidate, binary):
                continue
            predicted = float(np.sum(active_gradient * (candidate.astype(float) - binary)))
            ranked.append((predicted, f"{seed_name}_{cleanup_name}", candidate.astype(float)))
    unique = []
    for item in sorted(ranked, key=lambda row: (-row[0], row[1])):
        if not any(np.array_equal(item[2], prior[2]) for prior in unique):
            unique.append(item)
    return unique[:limit]


def plot_step(history, rho, active_gradient, step):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    axes[0].imshow(rho.T, origin="lower", cmap="gray_r", vmin=0, vmax=1, extent=(-4,4,-4,4))
    axes[0].set_title("exact binary: black=Au")
    axes[0].set_xlabel("x=b (um)"); axes[0].set_ylabel("y=a (um)")
    limit = max(float(np.max(np.abs(active_gradient))), np.finfo(float).tiny)
    image = axes[1].imshow(active_gradient.T, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit, extent=(-4,4,-4,4))
    axes[1].set_title("active combined adjoint gradient")
    axes[1].set_xlabel("x=b (um)"); axes[1].set_ylabel("y=a (um)")
    fig.colorbar(image, ax=axes[1], fraction=0.046)
    x = [row["step"] for row in history]
    axes[2].plot(x, [1e9*row["I_a_A"] for row in history], "o-", label="I_a")
    axes[2].plot(x, [-1e9*row["I_b_A"] for row in history], "o-", label="-I_b")
    axes[2].plot(x, [1e9*row["balanced_utility_A"] for row in history], "ko-", label="min")
    axes[2].axhline(0, color="black", linewidth=.8)
    axes[2].set_xlabel("accepted exact-binary update"); axes[2].set_ylabel("useful current (nA)")
    axes[2].grid(alpha=.25); axes[2].legend()
    fig.suptitle(f"Exact 500 nm Au topology search — accepted step {step}")
    fig.savefig(OUT / f"accepted_step_{step:02d}.png", dpi=170)
    fig.savefig(OUT / "latest_exact_binary_search.png", dpi=170)
    plt.close(fig)


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("GPU-only exact topology search requires CUDA_VISIBLE_DEVICES")
    readiness = require_production_readiness()
    cuda_device = int(os.environ.get("THERMAL_CUDA_DEVICE", "0"))
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    source_scales = calibrated_source_scales(
        readiness, CONTRACT.reporting_incident_power_W
    )
    with np.load(INITIAL, allow_pickle=False) as data:
        binary = np.asarray(data["physical_density"], dtype=float) >= 0.5
    audit = exact_500nm_audit(binary.astype(float))
    if not (audit["solid_pass"] and audit["void_pass"]):
        raise RuntimeError("initial exact-binary candidate fails 500 nm audit")
    start = time.perf_counter()
    runners = {
        pol: CompiledOpticalRunner.create(
            pol,
            binary.astype(float),
            numerical_contract=readiness["selected_numerical_contract"],
        )
        for pol in ("Ea", "Eb")
    }
    history = []
    current = evaluate_forward(runners, binary.astype(float), source_scales, cuda_device, raw_tag="accepted_step_00")
    current.update(step=0, name="initial_threshold_0.50_exact", exact_bad_cells=0)
    history.append(current)
    for step in range(1, MAX_STEPS + 1):
        gradients = evaluate_gradient(runners, binary.astype(float), source_scales, cuda_device)
        ia, ib = gradients["Ea"]["current_A"], gradients["Eb"]["current_A"]
        active = gradients["Ea"]["gradient_A"] if ia <= -ib else -gradients["Eb"]["gradient_A"]
        proposals = propose(binary, active, PROPOSALS_PER_STEP)
        if not proposals:
            history[-1]["termination"] = "no_new_exact_500nm_proposal"
            break
        evaluated = []
        for predicted, name, candidate in proposals:
            result = evaluate_forward(runners, candidate, source_scales, cuda_device)
            result.update(name=name, predicted_delta_A=predicted)
            evaluated.append((result, candidate))
            print(f"[candidate step={step}] {name}: Ia={1e9*result['I_a_A']:.5f} nA Ib={1e9*result['I_b_A']:.5f} nA min={1e9*result['balanced_utility_A']:.5f} nA", flush=True)
        best, best_binary = max(evaluated, key=lambda item: item[0]["balanced_utility_A"])
        if best["balanced_utility_A"] <= current["balanced_utility_A"] + 1.0e-13:
            history[-1]["termination"] = "no_forward_verified_improving_exact_move"
            history[-1]["best_rejected"] = best
            break
        binary = best_binary.astype(bool)
        current = evaluate_forward(runners, binary.astype(float), source_scales, cuda_device, raw_tag=f"accepted_step_{step:02d}")
        current.update(step=step, name=best["name"], predicted_delta_A=best["predicted_delta_A"], exact_bad_cells=0)
        history.append(current)
        plot_step(history, binary.astype(float), active, step)
        np.savez_compressed(RAW / f"accepted_step_{step:02d}.npz", physical_density=binary.astype(float))
        write_json(OUT / "exact_binary_search_history.json", history)
        if current["I_a_A"] > 0 and current["I_b_A"] < 0:
            history[-1]["termination"] = "opposite_current_direction_achieved"
            break
    success = bool(history[-1]["I_a_A"] > 0 and history[-1]["I_b_A"] < 0)
    final = {
        "status": "VALIDATED_4UM_DUALPOL_AU_CURRENT_SWITCH_EXACT_BINARY" if success else "BLOCKED_EXACT_BINARY_LOCAL_SEARCH_NO_OPPOSITE_SIGN",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "contract": CONTRACT.audit(),
        "au_material_fraction": material_fraction_audit(),
        "production_readiness": readiness,
        "algorithm": "combined-adjoint-guided exact 500 nm disk add/remove; forward-verified acceptance",
        "initial_raw": str(INITIAL),
        "history": history,
        "accepted_steps": len(history)-1,
        "runtime_s": time.perf_counter()-start,
        "opposite_current_direction_gate": success,
        "exact_500nm_solid_void_gate": True,
        "no_gray_material_in_search": True,
    }
    final_raw = RAW / "FINAL_EXACT_BINARY.npz"
    np.savez_compressed(final_raw, physical_density=binary.astype(float))
    final["final_raw"] = {"path": str(final_raw.resolve()), "bytes": final_raw.stat().st_size, "sha256": sha256(final_raw), "committed_to_git": False}
    write_json(OUT / "FINAL_EXACT_BINARY_SEARCH.json", final)
    write_json(OUT / "exact_binary_search_history.json", history)
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
