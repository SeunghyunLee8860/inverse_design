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


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_4um_dualpol_au_ld_mma"
RAW = Path(
    "/home/seunghyun/tairte4/raw/au_dualpol_4um_current_switch/optimization_ld_mma"
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
    16.0: 10,
    32.0: 8,
    64.0: 8,
    128.0: 6,
}
CAP_REDUCTION = {
    1.0: 1.01,
    2.0: 0.95,
    4.0: 0.90,
    8.0: 0.85,
    16.0: 0.80,
    32.0: 0.75,
    64.0: 0.70,
    128.0: 0.65,
}
# A smooth opening has a finite soft-min/soft-max boundary layer even for an
# exact-audit-passing binary pattern.  These floors were measured on exact
# 500 nm controls; using a smaller number would reject valid large features.
# The discontinuous final audit remains the zero-violation authority.
DFM_CAP_FLOOR = np.asarray((1.0e-1, 5.0e-3), dtype=np.float64)
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


def make_optimizer(evaluator: DualPolarizationEvaluator) -> nlopt.opt:
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
    optimizer.set_maxeval(STAGE_MAXEVAL[evaluator.beta])
    return optimizer


def repaired_binary_candidates(rho: np.ndarray) -> list[tuple[str, np.ndarray]]:
    footprint = physical_disk_footprint(
        0.5 * CONTRACT.minimum_solid_feature_m, CONTRACT.design_pitch_m
    )
    seed = np.asarray(rho) >= 0.5
    candidates: list[tuple[str, np.ndarray]] = []
    for name, first_phase in (("remove_then_fill", "solid"), ("fill_then_remove", "void")):
        binary = seed.copy()
        order = ("solid", "void") if first_phase == "solid" else ("void", "solid")
        for _ in range(32):
            before = binary.copy()
            for phase in order:
                if phase == "solid":
                    binary = ndimage.binary_opening(
                        binary, structure=footprint, border_value=0
                    )
                else:
                    binary = ~ndimage.binary_opening(
                        ~binary, structure=footprint, border_value=1
                    )
            audit = exact_500nm_audit(binary.astype(float))
            if audit["solid_pass"] and audit["void_pass"]:
                break
            if np.array_equal(binary, before):
                break
        audit = exact_500nm_audit(binary.astype(float))
        if audit["solid_pass"] and audit["void_pass"]:
            candidates.append((name, binary.astype(np.float64)))
    unique: list[tuple[str, np.ndarray]] = []
    for name, binary in candidates:
        if not any(np.array_equal(binary, item) for _, item in unique):
            unique.append((name, binary))
    return unique


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
        writer = csv.DictWriter(stream, fieldnames=fields)
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
    latent = np.full(CONTRACT.design_shape, 0.5, dtype=np.float64)
    vector = np.concatenate((latent.ravel(), [0.0]))
    history: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "schema": "au-dualpol-4um-ld-mma-raw-manifest-v1",
        "raw_artifacts_committed_to_git": False,
        "contract": CONTRACT.audit(),
        "mapping": MAPPING.audit(),
        "optimizer": {
            "library": "NLopt",
            "version": nlopt.__version__,
            "algorithm": "LD_MMA",
            "objective": "maximize t subject to Ia>=t and -Ib>=t",
            "manual_move_limit": None,
            "custom_update": False,
            "beta_schedule": list(BETAS),
            "stage_maxeval": {str(key): value for key, value in STAGE_MAXEVAL.items()},
        },
        "evaluations": {},
    }
    write_json(OUT / "RUN_STATE.json", {"stage": "compiling", "gpu": os.environ["CUDA_VISIBLE_DEVICES"]})
    runners = {
        pol: CompiledOpticalRunner.create(pol, latent) for pol in ("Ea", "Eb")
    }
    stage_rows: list[dict[str, object]] = []
    for stage_index, beta in enumerate(BETAS):
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
        vector[-1] = 0.98 * min(
            entry_point.current_a_A, -entry_point.current_b_A
        ) / CURRENT_SCALE_A
        optimizer = make_optimizer(evaluator)
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
        vector = optimizer.optimize(vector)
        stage = {
            "stage_index": stage_index,
            "beta": beta,
            "entry_dfm_values": entry_values.tolist(),
            "dfm_caps": caps.tolist(),
            "nlopt_result": int(optimizer.last_optimize_result()),
            "nlopt_function_evaluations": int(optimizer.get_numevals()),
            "nlopt_max_objective": float(optimizer.last_optimum_value()),
            "stage_runtime_s": time.perf_counter() - start,
            "evaluations_total": len(history),
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

    final_latent = vector[:-1].reshape(CONTRACT.design_shape)
    final_projected = MAPPING.physical(final_latent, BETAS[-1])
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
    promoted = max(binary_results, key=lambda row: row["balanced_utility_A"])
    promoted_rho = dict(candidates)[promoted["name"]]
    final_raw = RAW / "PROMOTED_FINAL_BINARY.npz"
    np.savez_compressed(final_raw, physical_density=promoted_rho)
    opposite_sign_pass = bool(
        promoted["I_a_A"] > 0.0
        and promoted["I_b_A"] < 0.0
        and promoted["balanced_utility_A"] > 0.0
    )
    final = {
        "status": (
            "VALIDATED_4UM_DUALPOL_AU_CURRENT_SWITCH_EXACT_BINARY"
            if opposite_sign_pass
            else "FAILED_4UM_DUALPOL_OPPOSITE_CURRENT_DIRECTION"
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
        "promoted": promoted,
        "opposite_current_direction_gate": opposite_sign_pass,
        "promoted_raw": {
            "path": str(final_raw),
            "bytes": final_raw.stat().st_size,
            "sha256": sha256(final_raw),
        },
        "no_clipping_smoothing_gain_or_global_q_rescaling": True,
    }
    write_json(OUT / "FINAL_RESULT.json", final)
    write_history_csv(history)
    write_json(OUT / "RAW_ARTIFACT_MANIFEST.json", manifest)
    write_json(
        OUT / "RUN_STATE.json",
        {
            "stage": "complete",
            "status": final["status"],
            "evaluations_completed": len(history),
            "promoted": promoted,
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
        "in LD_MMA and the promoted final geometry is separately exact-audited.",
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
    return 0 if opposite_sign_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
