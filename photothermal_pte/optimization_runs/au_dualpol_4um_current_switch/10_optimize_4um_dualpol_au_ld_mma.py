#!/usr/bin/env python3
"""Optimize a binary Au pattern for opposite-sign dual-polarization PTE current.

The exact production problem is

    maximize t
    subject to t <= I(E||a),  t <= -I(E||b),

plus differentiable 500 nm solid and void opening constraints.  The design
starts from a uniform latent density of 0.5.  There is no symmetry, volume,
connectivity, hand-written sign step, Adam update, or post-update clipping.
NLopt LD_MMA owns every continuous design update.  Projection continuation is
performed only between completed MMA stages; an exact thresholded 500 nm
audit and a separately evaluated binary candidate close the run.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
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
import nlopt
import numpy as np
from scipy import ndimage

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.combined_4um import (
    CompiledOpticalRunner,
    combined_gradient,
    evaluate_forward_multiphysics,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.dfm import (
    MAPPING,
    density_metrics,
    exact_500nm_audit,
    physical_disk_footprint,
    smooth_500nm_constraints,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    audit as material_fraction_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    N_DESIGN,
    N_TA,
    current_integrand,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get(
    "AU_DUALPOL_OUTPUT_NAME", "results_4um_dualpol_au_ld_mma"
)
RAW = Path(
    os.environ.get(
        "AU_DUALPOL_RAW_DIR",
        "/home/seunghyun/tairte4/raw/au_dualpol_4um_current_switch/optimization_ld_mma",
    )
)
CALIBRATION = (
    HERE
    / "results_fdtdx_4um_source_calibration"
    / "fdtdx_4um_source_calibration.json"
)
CURRENT_SCALE_A = 1.0e-9
BETAS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
STAGE_MAXEVAL = {
    1.0: 24,
    2.0: 16,
    4.0: 14,
    8.0: 12,
    16.0: 16,
    32.0: 14,
    64.0: 12,
    128.0: 12,
    12.0: 16,
    24.0: 14,
    48.0: 12,
    96.0: 14,
    80.0: 14,
    112.0: 12,
}
CAP_REDUCTION = {
    1.0: 1.01,
    2.0: 0.95,
    4.0: 0.90,
    8.0: 0.85,
    16.0: 0.95,
    32.0: 0.92,
    64.0: 0.90,
    128.0: 0.95,
    # Adaptive continuation used after the validated beta=8 checkpoint.  A
    # beta change already perturbs both current signs and morphology residuals;
    # imposing a second 20--35% residual jump at the same instant made the
    # beta=16 entry infeasible.  These reductions are deliberately gradual,
    # while the exact binary repair below remains the zero-violation authority.
    12.0: 0.95,
    24.0: 0.92,
    48.0: 0.90,
    96.0: 0.95,
    80.0: 0.95,
    112.0: 0.95,
}
# A smooth opening has a finite soft-min/soft-max boundary layer even for an
# exact-audit-passing binary pattern.  These floors were measured on exact
# 500 nm controls; using a smaller number would reject valid large features.
# The discontinuous final audit remains the zero-violation authority.
DFM_CAP_FLOOR = np.asarray((1.0e-1, 5.0e-3), dtype=np.float64)
STAGE_PERFORMANCE_RETENTION = 0.90
NLOPT_CONSTRAINT_TOL = 2.0e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def finite_float(value: object) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeError("non-finite optimizer metric")
    return result


def optical_closure(result: dict[str, object], runner: CompiledOpticalRunner, scale: float) -> tuple[float, float, float]:
    output = result["optical_output"]
    eta0 = float(runner.model["fdtdx"].constants.eta0)
    p_six = scale * eta0 * float(
        np.mean(
            np.asarray(output.detector_states["material_flux_td"]["poynting_flux"])[
                :, 0
            ]
        )
    )
    p_q = float(np.sum(result["source_power_W"]))
    closure = abs(p_q - p_six) / max(abs(p_six), np.finfo(float).tiny)
    return p_q, p_six, closure


def audit_physics(result: dict[str, object], closure: float, polarization: str) -> dict[str, object]:
    gates = {
        "optical_closure_lt_0p5pct": closure < 0.005,
        "thermal_energy_balance_lt_1pct": finite_float(
            result["thermal_audit"]["energy_balance_relative"]
        )
        < 0.01,
        "thermal_residual_lt_1e8": finite_float(
            result["thermal_audit"]["relative_residual"]
        )
        < 1e-8,
        "thermal_adjoint_residual_lt_1e8": finite_float(
            result["thermal_adjoint_audit"]["relative_residual"]
        )
        < 1e-8,
        "electrical_residual_lt_1e8": finite_float(
            result["electrical_audit"]["relative_residual"]
        )
        < 1e-8,
        "electrical_adjoint_residual_lt_1e8": finite_float(
            result["electrical_adjoint_audit"]["relative_residual"]
        )
        < 1e-8,
        "mapping_transpose_lt_1e12": finite_float(
            result["weighted_contraction_relative_error"]
        )
        < 1e-12,
        "finite_nonnegative_q": all(
            np.all(np.isfinite(value)) and float(np.min(value)) >= 0.0
            for value in result["q_fields_W_m3"].values()
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"fail-closed {polarization} physics gate: {gates}")
    return gates


@dataclass
class PhysicsPoint:
    latent: np.ndarray
    rho: np.ndarray
    current_a_A: float
    current_b_A: float
    gradient_a_latent_A: np.ndarray
    gradient_b_latent_A: np.ndarray
    smooth_values: np.ndarray
    smooth_gradients: np.ndarray
    smooth_fields: dict[str, np.ndarray]
    density: dict[str, object]
    audit: dict[str, dict[str, object]]
    diagnostics: dict[str, dict[str, object]]


class DualPolarizationEvaluator:
    def __init__(
        self,
        runner_a: CompiledOpticalRunner,
        runner_b: CompiledOpticalRunner,
        source_scale: float,
        cuda_device: int,
        beta: float,
        dfm_caps: np.ndarray,
        history: list[dict[str, object]],
        manifest: dict[str, object],
    ) -> None:
        self.runners = {"Ea": runner_a, "Eb": runner_b}
        self.source_scale = float(source_scale)
        self.cuda_device = int(cuda_device)
        self.beta = float(beta)
        self.dfm_caps = np.asarray(dfm_caps, dtype=np.float64)
        self.history = history
        self.manifest = manifest
        self.cached_latent: np.ndarray | None = None
        self.cached_point: PhysicsPoint | None = None

    def _evaluate(self, latent: np.ndarray) -> PhysicsPoint:
        latent = np.asarray(latent, dtype=np.float64).reshape(CONTRACT.design_shape)
        if self.cached_latent is not None and np.array_equal(latent, self.cached_latent):
            assert self.cached_point is not None
            return self.cached_point
        rho = MAPPING.physical(latent, self.beta)
        results: dict[str, dict[str, object]] = {}
        audit: dict[str, dict[str, object]] = {}
        diagnostics: dict[str, dict[str, object]] = {}
        for pol in ("Ea", "Eb"):
            result = combined_gradient(
                self.runners[pol], rho, self.source_scale, self.cuda_device
            )
            p_q, p_six, closure = optical_closure(
                result, self.runners[pol], self.source_scale
            )
            audit[pol] = audit_physics(result, closure, pol)
            diagnostics[pol] = {
                "current_A": finite_float(result["objective_A"]),
                "P_Q_W": p_q,
                "P_six_W": p_six,
                "closure_relative": closure,
                "Tmax_K": float(np.max(result["temperature"])),
                "forward_s": finite_float(result["forward_s"]),
                "adjoint_s": finite_float(result["adjoint_s"]),
                "gradient_norm_A": float(np.linalg.norm(result["gradient_total_A"])),
                "gradient_optical_norm_A": float(
                    np.linalg.norm(result["gradient_optical_A"])
                ),
                "gradient_thermal_contact_norm_A": float(
                    np.linalg.norm(result["gradient_thermal_A"])
                ),
                "gradient_electrical_weighting_norm_A": float(
                    np.linalg.norm(result["gradient_electrical_A"])
                ),
            }
            results[pol] = result
        smooth_values, smooth_gradients, smooth_fields = smooth_500nm_constraints(
            latent, self.beta
        )
        point = PhysicsPoint(
            latent=latent.copy(),
            rho=rho,
            current_a_A=finite_float(results["Ea"]["objective_A"]),
            current_b_A=finite_float(results["Eb"]["objective_A"]),
            gradient_a_latent_A=MAPPING.vjp(
                latent, np.asarray(results["Ea"]["gradient_total_A"]), self.beta
            ),
            gradient_b_latent_A=MAPPING.vjp(
                latent, np.asarray(results["Eb"]["gradient_total_A"]), self.beta
            ),
            smooth_values=smooth_values,
            smooth_gradients=smooth_gradients,
            smooth_fields=smooth_fields,
            density=density_metrics(latent, self.beta),
            audit=audit,
            diagnostics=diagnostics,
        )
        self._record(point, results)
        self.cached_latent = latent.copy()
        self.cached_point = point
        del results
        gc.collect()
        return point

    def _record(
        self, point: PhysicsPoint, results: dict[str, dict[str, object]]
    ) -> None:
        evaluation = len(self.history) + 1
        utility_a = point.current_a_A
        utility_b = -point.current_b_A
        row: dict[str, object] = {
            "evaluation": evaluation,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "beta": self.beta,
            "I_a_A": point.current_a_A,
            "I_b_A": point.current_b_A,
            "utility_a_A": utility_a,
            "utility_b_A": utility_b,
            "balanced_utility_A": min(utility_a, utility_b),
            "active_polarization": "Ea" if utility_a <= utility_b else "Eb",
            "dfm_solid_value": float(point.smooth_values[0]),
            "dfm_void_value": float(point.smooth_values[1]),
            "dfm_solid_cap": float(self.dfm_caps[0]),
            "dfm_void_cap": float(self.dfm_caps[1]),
            "dfm_solid_g": float(point.smooth_values[0] / self.dfm_caps[0] - 1.0),
            "dfm_void_g": float(point.smooth_values[1] / self.dfm_caps[1] - 1.0),
            **point.density,
            "polarizations": point.diagnostics,
            "gates": point.audit,
            "optimizer": "NLopt LD_MMA",
            "manual_move_limit": None,
            "symmetry_constraint": False,
            "volume_fraction_constraint": False,
            "connectivity_constraint": False,
        }
        self.history.append(row)
        RAW.mkdir(parents=True, exist_ok=True)
        raw = RAW / f"evaluation_{evaluation:04d}.npz"
        np.savez_compressed(
            raw,
            latent=point.latent,
            physical_density=point.rho,
            gradient_Ia_latent_A=point.gradient_a_latent_A,
            gradient_Ib_latent_A=point.gradient_b_latent_A,
            dfm_values=point.smooth_values,
            dfm_gradients=point.smooth_gradients,
            dfm_solid_residual=point.smooth_fields["solid_residual"],
            dfm_void_residual=point.smooth_fields["void_residual"],
        )
        self.manifest.setdefault("evaluations", {})[f"{evaluation:04d}"] = {
            "path": str(raw),
            "bytes": raw.stat().st_size,
            "sha256": sha256(raw),
        }
        write_json(OUT / "optimization_history.json", self.history)
        write_json(OUT / "RAW_ARTIFACT_MANIFEST.json", self.manifest)
        write_json(OUT / f"evaluation_{evaluation:04d}.json", row)
        plot_evaluation(OUT, self.history, point, evaluation)
        print(
            f"[eval {evaluation:04d}] beta={self.beta:g} "
            f"Ia={1e9*point.current_a_A:.6f} nA "
            f"Ib={1e9*point.current_b_A:.6f} nA "
            f"min={1e9*min(utility_a,utility_b):.6f} nA "
            f"gray={point.density['gray_fraction_0p01_0p99']:.4f} "
            f"bad={point.density['exact_bad_cell_count']} "
            f"gdfm=({row['dfm_solid_g']:.3e},{row['dfm_void_g']:.3e})",
            flush=True,
        )

    def objective(self, vector: np.ndarray, gradient: np.ndarray) -> float:
        if gradient.size:
            gradient[:] = 0.0
            gradient[-1] = 1.0
        return float(vector[-1])

    def constraints(
        self, values: np.ndarray, vector: np.ndarray, gradient: np.ndarray
    ) -> None:
        point = self._evaluate(vector[:-1])
        t = float(vector[-1])
        values[:] = (
            t - point.current_a_A / CURRENT_SCALE_A,
            t + point.current_b_A / CURRENT_SCALE_A,
            point.smooth_values[0] / self.dfm_caps[0] - 1.0,
            point.smooth_values[1] / self.dfm_caps[1] - 1.0,
        )
        if gradient.size:
            gradient[:] = 0.0
            gradient[0, :-1] = -point.gradient_a_latent_A.ravel() / CURRENT_SCALE_A
            gradient[0, -1] = 1.0
            gradient[1, :-1] = point.gradient_b_latent_A.ravel() / CURRENT_SCALE_A
            gradient[1, -1] = 1.0
            gradient[2, :-1] = point.smooth_gradients[0].ravel() / self.dfm_caps[0]
            gradient[3, :-1] = point.smooth_gradients[1].ravel() / self.dfm_caps[1]
        # Record the actual epigraph coordinate and its slack.  NLopt can call
        # the constraints repeatedly at the same density while updating only
        # t, so this diagnostic is updated without rerunning Maxwell.
        if self.history:
            self.history[-1]["epigraph_t_scaled"] = t
            self.history[-1]["epigraph_t_A"] = t * CURRENT_SCALE_A
            self.history[-1]["epigraph_slack_a_A"] = (
                point.current_a_A - t * CURRENT_SCALE_A
            )
            self.history[-1]["epigraph_slack_b_A"] = (
                -point.current_b_A - t * CURRENT_SCALE_A
            )
            write_json(OUT / "optimization_history.json", self.history)
            write_json(
                OUT / f"evaluation_{self.history[-1]['evaluation']:04d}.json",
                self.history[-1],
            )


def plot_evaluation(
    output: Path,
    history: list[dict[str, object]],
    point: PhysicsPoint,
    evaluation: int,
) -> None:
    rows = history
    index = np.asarray([row["evaluation"] for row in rows])
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    image = axes[0, 0].imshow(
        point.rho.T,
        origin="lower",
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        extent=(-4, 4, -4, 4),
        interpolation="nearest",
    )
    axes[0, 0].set_title("physical density: black=Au, white=void")
    axes[0, 0].set_xlabel("x=b (um)")
    axes[0, 0].set_ylabel("y=a (um)")
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046)

    axes[0, 1].hist(point.rho.ravel(), bins=40, range=(0, 1))
    axes[0, 1].set_title("physical-density histogram")
    axes[0, 1].set_xlabel("rho_Au")

    active = (
        point.gradient_a_latent_A
        if point.current_a_A <= -point.current_b_A
        else -point.gradient_b_latent_A
    )
    limit = float(np.max(np.abs(active)))
    gradient_image = axes[0, 2].imshow(
        active.T,
        origin="lower",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        extent=(-4, 4, -4, 4),
    )
    axes[0, 2].set_title("active utility gradient d[min]/d latent (A)")
    axes[0, 2].set_xlabel("x=b (um)")
    axes[0, 2].set_ylabel("y=a (um)")
    fig.colorbar(gradient_image, ax=axes[0, 2], fraction=0.046)

    ia = 1e9 * np.asarray([row["utility_a_A"] for row in rows], dtype=float)
    mb = 1e9 * np.asarray([row["utility_b_A"] for row in rows], dtype=float)
    axes[1, 0].plot(index, ia, label="I_a (right-to-left)")
    axes[1, 0].plot(index, mb, label="-I_b (left-to-right)")
    axes[1, 0].plot(index, np.minimum(ia, mb), "k-", label="min utility")
    axes[1, 0].set_title("opposite-current epigraph objective")
    axes[1, 0].set_xlabel("full-physics evaluation")
    axes[1, 0].set_ylabel("useful current (nA)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.25)

    beta = np.asarray([row["beta"] for row in rows], dtype=float)
    gray = np.asarray([row["gray_fraction_0p01_0p99"] for row in rows])
    axes[1, 1].plot(index, gray, "o-", label="gray fraction")
    beta_axis = axes[1, 1].twinx()
    beta_axis.step(index, beta, where="post", color="black", label="beta")
    beta_axis.set_yscale("log", base=2)
    axes[1, 1].set_title("binarization continuation")
    axes[1, 1].set_xlabel("full-physics evaluation")
    axes[1, 1].set_ylabel("gray fraction")
    beta_axis.set_ylabel("beta")

    solid_g = np.asarray([row["dfm_solid_g"] for row in rows])
    void_g = np.asarray([row["dfm_void_g"] for row in rows])
    bad = np.asarray([row["exact_bad_cell_count"] for row in rows])
    axes[1, 2].plot(index, solid_g, label="solid g<=0")
    axes[1, 2].plot(index, void_g, label="void g<=0")
    axes[1, 2].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    bad_axis = axes[1, 2].twinx()
    bad_axis.plot(index, bad, color="tab:red", alpha=0.55, label="exact bad cells")
    axes[1, 2].set_title("500 nm solid/void constraints")
    axes[1, 2].set_xlabel("full-physics evaluation")
    axes[1, 2].set_ylabel("normalized smooth inequality")
    bad_axis.set_ylabel("exact bad cells")
    axes[1, 2].legend(fontsize=8, loc="upper left")

    latest = rows[-1]
    fig.suptitle(
        "Au dual-polarization current switch; "
        f"eval={evaluation}, beta={latest['beta']:g}, "
        f"Ia={1e9*latest['I_a_A']:.4f} nA, "
        f"Ib={1e9*latest['I_b_A']:.4f} nA, "
        f"min={1e9*latest['balanced_utility_A']:.4f} nA",
        fontsize=14,
    )
    path = output / f"evaluation_{evaluation:04d}.png"
    fig.savefig(path, dpi=150)
    fig.savefig(output / "latest_iteration.png", dpi=150)
    plt.close(fig)


def stage_caps(latent: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    values = smooth_500nm_constraints(latent, beta)[0]
    caps = np.maximum(DFM_CAP_FLOOR, CAP_REDUCTION[beta] * values)
    # A zero phase residual is already feasible and must not be turned into an
    # artificial equality by division through a numerical floor.
    caps = np.where(values < 0.25 * DFM_CAP_FLOOR, DFM_CAP_FLOOR, caps)
    return values, caps


def make_optimizer(
    evaluator: DualPolarizationEvaluator, maxeval: int | None = None
) -> nlopt.opt:
    variable_count = int(np.prod(CONTRACT.design_shape)) + 1
    optimizer = nlopt.opt(nlopt.LD_MMA, variable_count)
    lower = np.concatenate((np.zeros(variable_count - 1), [-100.0]))
    upper = np.concatenate((np.ones(variable_count - 1), [1000.0]))
    optimizer.set_lower_bounds(lower)
    optimizer.set_upper_bounds(upper)
    optimizer.set_max_objective(evaluator.objective)
    optimizer.add_inequality_mconstraint(
        evaluator.constraints,
        np.full(4, NLOPT_CONSTRAINT_TOL, dtype=np.float64),
    )
    optimizer.set_initial_step(
        np.concatenate((np.full(variable_count - 1, 0.05), [0.1]))
    )
    # Relative-x stopping is ill-scaled for 6400 densities plus one epigraph
    # scalar: early iterations can move only t and look stationary in the
    # aggregate x norm.  Each continuation stage therefore uses its audited
    # full-physics evaluation budget; beta promotion, not an accidental xtol,
    # controls termination.
    optimizer.set_ftol_rel(0.0)
    optimizer.set_xtol_rel(0.0)
    optimizer.set_maxeval(
        STAGE_MAXEVAL[evaluator.beta] if maxeval is None else int(maxeval)
    )
    return optimizer


def repaired_binary_candidates(rho: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return nearby exact-DFM binary projections in deterministic rank order.

    Alternating one opening and one closing at a fixed 0.5 threshold can enter
    a two-cycle: repairing a narrow void may create a narrow solid and vice
    versa.  Promotion must not depend on that arbitrary threshold.  We instead
    enumerate a small deterministic family of level sets and physical
    morphology sequences, retain only candidates that independently pass both
    exact openings, remove duplicates, and rank by L1 distance to the actual
    continuous physical density.  This is geometry cleanup, not an optimizer
    update or a global source rescaling.
    """

    rho_array = np.asarray(rho, dtype=np.float64)
    candidates: list[tuple[float, float, str, np.ndarray]] = []
    thresholds = np.linspace(0.05, 0.95, 19)
    repair_radii_m = (250.0e-9, 350.0e-9, 450.0e-9)
    sequences = (
        ("open_close_open", ("open", "close", "open")),
        ("close_open", ("close", "open")),
        ("close_open_close", ("close", "open", "close")),
    )
    for threshold in thresholds:
        seed = rho_array >= float(threshold)
        for radius_m in repair_radii_m:
            footprint = physical_disk_footprint(
                radius_m, CONTRACT.design_pitch_m
            )
            for sequence_name, sequence in sequences:
                binary = seed.copy()
                for operation in sequence:
                    if operation == "open":
                        binary = ndimage.binary_opening(
                            binary, structure=footprint, border_value=0
                        )
                    else:
                        binary = ~ndimage.binary_opening(
                            ~binary, structure=footprint, border_value=1
                        )
                audit = exact_500nm_audit(binary.astype(float))
                if not (audit["solid_pass"] and audit["void_pass"]):
                    continue
                score = float(np.mean(np.abs(binary.astype(float) - rho_array)))
                name = (
                    f"threshold_{threshold:.2f}_radius_{radius_m*1e9:.0f}nm_"
                    f"{sequence_name}"
                )
                candidates.append(
                    (score, float(threshold), name, binary.astype(np.float64))
                )
    unique: list[tuple[float, float, str, np.ndarray]] = []
    for candidate in sorted(candidates, key=lambda item: (item[0], item[2])):
        if not any(np.array_equal(candidate[3], item[3]) for item in unique):
            unique.append(candidate)
    # Three nearest geometries plus representative threshold-0.25, 0.50 and
    # 0.80 level sets separate morphology choice from Maxwell/PTE performance.
    # This avoids both a one-threshold promotion and a costly exhaustive sweep.
    selected = unique[:3]
    for target in (0.25, 0.50, 0.80):
        ranked = sorted(
            unique,
            key=lambda item: (abs(item[1] - target), item[0], item[2]),
        )
        for candidate in ranked:
            if not any(
                np.array_equal(candidate[3], item[3]) for item in selected
            ):
                selected.append(candidate)
                break
    return [(name, binary) for _, _, name, binary in selected]


def evaluate_binary_candidate(
    name: str,
    rho: np.ndarray,
    runners: dict[str, CompiledOpticalRunner],
    source_scale: float,
    cuda_device: int,
) -> dict[str, object]:
    cases: dict[str, object] = {}
    for pol in ("Ea", "Eb"):
        result = evaluate_forward_multiphysics(
            runners[pol], rho, source_scale, cuda_device, need_gradient=False
        )
        p_q, p_six, closure = optical_closure(result, runners[pol], source_scale)
        gates = {
            "optical_closure_lt_0p5pct": closure < 0.005,
            "thermal_energy_balance_lt_1pct": finite_float(
                result["thermal_audit"]["energy_balance_relative"]
            )
            < 0.01,
            "thermal_residual_lt_1e8": finite_float(
                result["thermal_audit"]["relative_residual"]
            )
            < 1e-8,
            "electrical_residual_lt_1e8": finite_float(
                result["electrical_audit"]["relative_residual"]
            )
            < 1e-8,
        }
        cases[pol] = {
            "current_A": finite_float(result["objective_A"]),
            "P_Q_W": p_q,
            "P_six_W": p_six,
            "closure_relative": closure,
            "Tmax_K": float(np.max(result["temperature"])),
            "gates": gates,
        }
        if not all(gates.values()):
            raise RuntimeError(f"fail-closed final binary {name} {pol}: {gates}")
        raw_case = RAW / f"final_binary_{name}_{pol}_fields.npz"
        weighting = np.asarray(result["weighting"], dtype=np.float64)
        np.savez_compressed(
            raw_case,
            physical_density=np.asarray(rho, dtype=np.float64),
            q_au_W_m3=np.asarray(result["q_fields_W_m3"]["au"]),
            q_tairte4_W_m3=np.asarray(result["q_fields_W_m3"]["tairte4"]),
            dual_volume_au_m3=np.asarray(runners[pol].volumes["au"]),
            dual_volume_tairte4_m3=np.asarray(
                runners[pol].volumes["tairte4"]
            ),
            ta_temperature_K=np.asarray(result["ta_temperature"]),
            weighting_tairte4=weighting[: N_TA * N_TA].reshape(N_TA, N_TA),
            weighting_au=weighting[N_TA * N_TA :].reshape(N_DESIGN, N_DESIGN),
            current_integrand_A_m2=current_integrand(
                np.asarray(result["ta_temperature"]), weighting
            ),
        )
        cases[pol]["raw_fields"] = {
            "path": str(raw_case.resolve()),
            "bytes": int(raw_case.stat().st_size),
            "sha256": sha256(raw_case),
            "committed_to_git": False,
        }
        del result
        gc.collect()
    ia = float(cases["Ea"]["current_A"])
    ib = float(cases["Eb"]["current_A"])
    return {
        "name": name,
        "I_a_A": ia,
        "I_b_A": ib,
        "balanced_utility_A": min(ia, -ib),
        "cases": cases,
        "exact_audit": {
            key: value
            for key, value in exact_500nm_audit(rho).items()
            if not isinstance(value, np.ndarray)
        },
    }


def write_history_csv(history: list[dict[str, object]]) -> None:
    fields = (
        "evaluation",
        "beta",
        "I_a_A",
        "I_b_A",
        "balanced_utility_A",
        "active_polarization",
        "gray_fraction_0p01_0p99",
        "binarization_mean_4rho1mrho",
        "exact_bad_cell_count",
        "dfm_solid_value",
        "dfm_void_value",
        "dfm_solid_cap",
        "dfm_void_cap",
        "dfm_solid_g",
        "dfm_void_g",
    )
    with (OUT / "optimization_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in history:
            writer.writerow({name: row[name] for name in fields})


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("GPU-only optimization requires CUDA_VISIBLE_DEVICES")
    cuda_device = int(os.environ.get("THERMAL_CUDA_DEVICE", "0"))
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if calibration.get("status") != "VALIDATED_FDTDX_4UM_SOURCE_POWER_CALIBRATION":
        raise RuntimeError("validated source-power calibration is unavailable")
    source_scale = CONTRACT.reporting_incident_power_W / float(
        calibration["common_reference_incident_power_W"]
    )
    resume_path_text = os.environ.get("AU_DUALPOL_RESUME_STAGE_NPZ", "").strip()
    resume_evaluation = int(os.environ.get("AU_DUALPOL_RESUME_EVALUATION", "0"))
    finalize_only = os.environ.get("AU_DUALPOL_FINALIZE_ONLY", "0") == "1"
    requested_betas = os.environ.get("AU_DUALPOL_BETAS", "").strip()
    run_betas = (
        ()
        if finalize_only
        else (
            tuple(float(value) for value in requested_betas.split(",") if value.strip())
            if requested_betas
            else BETAS
        )
    )
    if not run_betas and not finalize_only:
        raise RuntimeError("empty beta continuation schedule")

    latent = np.full(CONTRACT.design_shape, 0.5, dtype=np.float64)
    vector = np.concatenate((latent.ravel(), [0.0]))
    history: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "schema": "au-dualpol-4um-ld-mma-raw-manifest-v1",
        "raw_artifacts_committed_to_git": False,
        "contract": CONTRACT.audit(),
        "au_material_fraction": material_fraction_audit(),
        "mapping": MAPPING.audit(),
        "optimizer": {
            "library": "NLopt",
            "version": nlopt.__version__,
            "algorithm": "LD_MMA",
            "objective": "maximize t subject to Ia>=t and -Ib>=t",
            "manual_move_limit": None,
            "custom_update": False,
            "beta_schedule": list(run_betas),
            "stage_maxeval": {str(key): value for key, value in STAGE_MAXEVAL.items()},
            "stage_feasibility_gate": (
                "Ia>0, Ib<0, epigraph feasible, both smooth DFM inequalities <=0, "
                "and returned min(Ia,-Ib) >= 90% of the prior promoted objective"
            ),
            "maximum_stage_attempts": 3,
        },
        "evaluations": {},
    }
    if resume_path_text:
        resume_path = Path(resume_path_text).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        if resume_evaluation <= 0:
            raise RuntimeError(
                "AU_DUALPOL_RESUME_EVALUATION must identify the completed checkpoint"
            )
        with np.load(resume_path, allow_pickle=False) as checkpoint:
            vector = np.asarray(checkpoint["vector"], dtype=np.float64)
            resume_beta = float(np.asarray(checkpoint["beta"]).item())
        expected = int(np.prod(CONTRACT.design_shape)) + 1
        if vector.size != expected:
            raise RuntimeError(
                f"resume vector has {vector.size} entries; expected {expected}"
            )
        latent = vector[:-1].reshape(CONTRACT.design_shape)
        history_path = OUT / "optimization_history.json"
        stages_path = OUT / "continuation_stages.json"
        manifest_path = OUT / "RAW_ARTIFACT_MANIFEST.json"
        if not (history_path.is_file() and stages_path.is_file() and manifest_path.is_file()):
            raise RuntimeError("published checkpoint provenance is incomplete")
        loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
        history = [
            row
            for row in loaded_history
            if int(row["evaluation"]) <= resume_evaluation
        ]
        if len(history) != resume_evaluation:
            raise RuntimeError(
                f"resume history contains {len(history)} rows; expected {resume_evaluation}"
            )
        loaded_stages = json.loads(stages_path.read_text(encoding="utf-8"))
        stage_rows = [
            row
            for row in loaded_stages
            if int(row["evaluations_total"]) <= resume_evaluation
        ]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("au_material_fraction") != material_fraction_audit():
            raise RuntimeError(
                "resume checkpoint uses a different or undocumented Au material "
                "fraction; start a new run under the shared linear contract"
            )
        manifest["evaluations"] = {
            key: value
            for key, value in manifest.get("evaluations", {}).items()
            if int(key) <= resume_evaluation
        }
        completed_betas = [float(row["beta"]) for row in stage_rows]
        manifest["optimizer"]["beta_schedule"] = completed_betas + list(run_betas)
        manifest["optimizer"]["resume"] = {
            "checkpoint_path": str(resume_path),
            "checkpoint_sha256": sha256(resume_path),
            "resume_evaluation": resume_evaluation,
            "completed_betas": completed_betas,
            "finalize_only": finalize_only,
        }
        write_history_csv(history)
        write_json(OUT / "optimization_history.json", history)
        write_json(OUT / "continuation_stages.json", stage_rows)
        write_json(OUT / "RAW_ARTIFACT_MANIFEST.json", manifest)
    write_json(OUT / "RUN_STATE.json", {"stage": "compiling", "gpu": os.environ["CUDA_VISIBLE_DEVICES"]})
    runners = {
        pol: CompiledOpticalRunner.create(pol, latent) for pol in ("Ea", "Eb")
    }
    initial_stage_index = len(stage_rows)
    last_completed_beta = float(stage_rows[-1]["beta"]) if stage_rows else None
    previous_stage_objective = (
        float(stage_rows[-1]["nlopt_max_objective"]) if stage_rows else None
    )
    for local_stage_index, beta in enumerate(run_betas):
        stage_index = initial_stage_index + local_stage_index
        latent = vector[:-1].reshape(CONTRACT.design_shape)
        entry_values, caps = stage_caps(latent, beta)
        evaluator = DualPolarizationEvaluator(
            runners["Ea"],
            runners["Eb"],
            source_scale,
            cuda_device,
            beta,
            caps,
            history,
            manifest,
        )
        # Put the epigraph variable on a feasible active scale before LD_MMA.
        # Otherwise the first cheap t-only steps can trigger a false relative-x
        # stop before any density update is proposed.
        entry_point = evaluator._evaluate(latent)
        vector[-1] = min(
            entry_point.current_a_A, -entry_point.current_b_A
        ) / CURRENT_SCALE_A - 1.0e-6
        write_json(
            OUT / "RUN_STATE.json",
            {
                "stage": "optimizing",
                "stage_index": stage_index,
                "beta": beta,
                "stage_entry_dfm_values": entry_values.tolist(),
                "stage_dfm_caps": caps.tolist(),
                "evaluations_completed": len(history),
                "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
            },
        )
        start = time.perf_counter()
        attempts: list[dict[str, object]] = []
        stage_feasible = False
        stage_performance_retained = False
        stage_promotable = False
        optimizer: nlopt.opt | None = None
        performance_floor = (
            STAGE_PERFORMANCE_RETENTION * previous_stage_objective
            if previous_stage_objective is not None
            else -np.inf
        )
        for attempt in range(3):
            optimizer = make_optimizer(evaluator)
            vector = optimizer.optimize(vector)
            returned = evaluator._evaluate(vector[:-1])
            t_scaled = float(vector[-1])
            constraint_values = np.asarray(
                (
                    t_scaled - returned.current_a_A / CURRENT_SCALE_A,
                    t_scaled + returned.current_b_A / CURRENT_SCALE_A,
                    returned.smooth_values[0] / caps[0] - 1.0,
                    returned.smooth_values[1] / caps[1] - 1.0,
                ),
                dtype=np.float64,
            )
            stage_feasible = bool(
                returned.current_a_A > 0.0
                and returned.current_b_A < 0.0
                and np.max(constraint_values) <= 10.0 * NLOPT_CONSTRAINT_TOL
            )
            returned_balanced_nA = min(
                returned.current_a_A, -returned.current_b_A
            ) / CURRENT_SCALE_A
            stage_performance_retained = bool(
                returned_balanced_nA >= performance_floor
                and t_scaled >= performance_floor
            )
            stage_promotable = stage_feasible and stage_performance_retained
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "nlopt_result": int(optimizer.last_optimize_result()),
                    "nlopt_function_evaluations": int(optimizer.get_numevals()),
                    "nlopt_max_objective": float(optimizer.last_optimum_value()),
                    "constraint_values": constraint_values.tolist(),
                    "I_a_A": returned.current_a_A,
                    "I_b_A": returned.current_b_A,
                    "feasible": stage_feasible,
                    "returned_balanced_utility_nA": returned_balanced_nA,
                    "performance_floor_nA": performance_floor,
                    "performance_retained": stage_performance_retained,
                    "promotable": stage_promotable,
                }
            )
            if stage_promotable:
                break
            # A continuation stage is never allowed to promote an infeasible
            # density.  Re-anchor t to the actual current bottleneck and let a
            # fresh MMA subproblem continue at the same beta/caps.
            vector[-1] = min(
                returned.current_a_A, -returned.current_b_A
            ) / CURRENT_SCALE_A - 1.0e-6
            print(
                f"[stage] beta={beta:g} attempt={attempt + 1} not promotable; "
                f"feasible={stage_feasible}, retained={stage_performance_retained}, "
                f"balanced={returned_balanced_nA:.6f} nA, "
                f"floor={performance_floor:.6f} nA, "
                f"constraints={constraint_values.tolist()}",
                flush=True,
            )
        if optimizer is None or not stage_promotable:
            write_json(
                OUT / "RUN_STATE.json",
                {
                    "stage": "blocked_continuation_stage_not_promotable",
                    "stage_index": stage_index,
                    "beta": beta,
                    "attempts": attempts,
                    "evaluations_completed": len(history),
                    "last_completed_beta": last_completed_beta,
                },
            )
            raise RuntimeError(
                f"beta={beta:g} remained non-promotable after {len(attempts)} MMA attempts"
            )
        stage = {
            "stage_index": stage_index,
            "beta": beta,
            "entry_dfm_values": entry_values.tolist(),
            "dfm_caps": caps.tolist(),
            "nlopt_result": int(optimizer.last_optimize_result()),
            "nlopt_function_evaluations": int(
                sum(int(row["nlopt_function_evaluations"]) for row in attempts)
            ),
            "nlopt_max_objective": float(optimizer.last_optimum_value()),
            "stage_runtime_s": time.perf_counter() - start,
            "evaluations_total": len(history),
            "stage_feasible": stage_feasible,
            "stage_performance_retained": stage_performance_retained,
            "stage_promotable": stage_promotable,
            "performance_floor_nA": performance_floor,
            "attempts": attempts,
        }
        stage_rows.append(stage)
        np.savez_compressed(
            RAW / f"stage_{stage_index:02d}_beta_{beta:g}.npz",
            vector=vector,
            latent=vector[:-1].reshape(CONTRACT.design_shape),
            beta=np.asarray(beta),
            dfm_caps=caps,
        )
        write_json(OUT / "continuation_stages.json", stage_rows)
        write_history_csv(history)
        print(f"[stage] completed beta={beta:g}: {stage}", flush=True)
        last_completed_beta = beta
        previous_stage_objective = float(optimizer.last_optimum_value())

    final_latent = vector[:-1].reshape(CONTRACT.design_shape)
    final_beta = (
        run_betas[-1]
        if run_betas
        else float(stage_rows[-1]["beta"] if stage_rows else resume_beta)
    )
    final_projected = MAPPING.physical(final_latent, final_beta)
    candidates = repaired_binary_candidates(final_projected)
    if not candidates:
        raise RuntimeError("no exact 500 nm solid/void binary repair candidate")
    binary_results = []
    for name, rho in candidates:
        result = evaluate_binary_candidate(
            name, rho, runners, source_scale, cuda_device
        )
        raw = RAW / f"final_binary_{name}.npz"
        np.savez_compressed(raw, physical_density=rho)
        result["raw"] = {
            "path": str(raw),
            "bytes": raw.stat().st_size,
            "sha256": sha256(raw),
        }
        binary_results.append(result)
    best_diagnostic = max(
        binary_results, key=lambda row: row["balanced_utility_A"]
    )
    opposite_sign_pass = bool(
        best_diagnostic["I_a_A"] > 0.0
        and best_diagnostic["I_b_A"] < 0.0
        and best_diagnostic["balanced_utility_A"] > 0.0
    )
    exact_dfm_pass = bool(
        int(best_diagnostic["exact_audit"]["solid_bad_cell_count"]) == 0
        and int(best_diagnostic["exact_audit"]["void_bad_cell_count"]) == 0
    )
    promoted = best_diagnostic if opposite_sign_pass and exact_dfm_pass else None
    promoted_raw = None
    if promoted is not None:
        promoted_rho = dict(candidates)[promoted["name"]]
        final_raw = RAW / "PROMOTED_FINAL_BINARY.npz"
        np.savez_compressed(final_raw, physical_density=promoted_rho)
        promoted_raw = {
            "path": str(final_raw),
            "bytes": final_raw.stat().st_size,
            "sha256": sha256(final_raw),
        }
    final = {
        "status": (
            "VALIDATED_4UM_DUALPOL_AU_CURRENT_SWITCH_EXACT_BINARY"
            if opposite_sign_pass and exact_dfm_pass
            else (
                "FAILED_4UM_DUALPOL_EXACT_500NM_DFM"
                if opposite_sign_pass
                else "FAILED_4UM_DUALPOL_OPPOSITE_CURRENT_DIRECTION"
            )
        ),
        "objective": "maximize min(Ia,-Ib)",
        "sign_convention": {
            "psi_left": 0,
            "psi_right": 1,
            "I_a_target": "positive, internal conventional current right-to-left",
            "I_b_target": "negative, internal conventional current left-to-right",
        },
        "continuous_last": history[-1],
        "binary_candidates": binary_results,
        "best_diagnostic_candidate": best_diagnostic,
        "promoted": promoted,
        "opposite_current_direction_gate": opposite_sign_pass,
        "exact_500nm_solid_void_gate": exact_dfm_pass,
        "continuous_checkpoint_beta": final_beta,
        "promoted_raw": promoted_raw,
        "no_clipping_smoothing_gain_or_global_q_rescaling": True,
    }
    write_json(OUT / "FINAL_RESULT.json", final)
    write_history_csv(history)
    write_json(OUT / "RAW_ARTIFACT_MANIFEST.json", manifest)
    write_json(
        OUT / "RUN_STATE.json",
        {
            "stage": "complete" if promoted is not None else "blocked_exact_binary",
            "status": final["status"],
            "evaluations_completed": len(history),
            "promoted": promoted,
            "best_diagnostic_candidate": best_diagnostic,
        },
    )
    report = [
        "# 4 um Au dual-polarization PTE current-switch inverse design",
        "",
        f"Status: **{final['status']}**",
        "",
        "The exact epigraph objective maximizes `min(I_a,-I_b)` with `psi=0`",
        "on the left terminal and `psi=1` on the right terminal.  Therefore",
        "the two requested illumination states drive opposite internal conventional",
        "current directions.  Both 500 nm solid and void constraints are included",
        "in LD_MMA and every binary candidate is separately exact-audited.",
        "A candidate is promoted only when both requested current signs survive",
        "the exact-binary conversion; otherwise the result remains fail-closed.",
        "",
        "| result | I_a (nA) | I_b (nA) | min(I_a,-I_b) (nA) | exact bad cells |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in binary_results:
        report.append(
            f"| {row['name']} | {1e9*row['I_a_A']:.6f} | "
            f"{1e9*row['I_b_A']:.6f} | {1e9*row['balanced_utility_A']:.6f} | "
            f"{row['exact_audit']['solid_bad_cell_count'] + row['exact_audit']['void_bad_cell_count']} |"
        )
    (OUT / "DUALPOL_AU_INVERSE_DESIGN_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(final, indent=2), flush=True)
    return 0 if opposite_sign_pass and exact_dfm_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
