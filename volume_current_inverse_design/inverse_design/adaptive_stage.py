"""Adaptive beta-stage controller (pure logic, no FDTD/nlopt imports).

Decides, after every objective evaluation, whether the current beta stage
should CONTINUE, ADVANCE to the next beta early, or ABORT the whole run.
Keeping this FDTD-free lets the policy be regression-tested without Lumerical.

Policy (fixed ladder 2,4,8,16,32,64 stays; only the per-stage eval count is
adaptive):

  early advance (all must hold):
    * evaluations >= min_evals_per_stage and >= convergence_window
    * relative span of the last `convergence_window` objectives
      < objective_rel_tol            (plateau; also bounds improvement)
    * g_solid <= 0 AND g_void <= 0 for every eval in the window
    * consecutive latent changes in the window are quiet:
      max RMS < latent_rms_tol AND max |Delta| < latent_max_tol
    -> reason "adaptive_converged_feasible"

  plateau while INFEASIBLE is NOT convergence:
    -> reason "stage_stalled_infeasible"; the run must ABORT (advancing beta
       from an infeasible stall would just harden an infeasible design).

  max_evals_per_stage reached:
    * stage saw at least one constraint-feasible eval -> advance,
      reason "maxeval_feasible"
    * otherwise -> ABORT, reason "maxeval_infeasible"

The runner realises a stop via nlopt's force_stop(); the resulting
nlopt.ForcedStop is INTENTIONAL whenever ``stop_reason`` is set and must never
be classified as a solver/numerical failure.  ``last_latent`` always holds the
most recently observed latent so a force-stopped stage hands a valid vector to
the next stage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

REASON_CONVERGED = "adaptive_converged_feasible"
REASON_STALLED_INFEASIBLE = "stage_stalled_infeasible"
REASON_MAXEVAL_FEASIBLE = "maxeval_feasible"
REASON_MAXEVAL_INFEASIBLE = "maxeval_infeasible"
ADVANCE_REASONS = (REASON_CONVERGED, REASON_MAXEVAL_FEASIBLE)
ABORT_REASONS = (REASON_STALLED_INFEASIBLE, REASON_MAXEVAL_INFEASIBLE)
CONTINUE = "continue"

POLICY_VERSION = "adaptive_stage/v1"


@dataclass(frozen=True)
class AdaptiveConfig:
    min_evals_per_stage: int = 3
    max_evals_per_stage: int = 12
    convergence_window: int = 3
    objective_rel_tol: float = 5e-3
    latent_rms_tol: float = 1e-3
    latent_max_tol: float = 1e-2

    def validate(self) -> "AdaptiveConfig":
        if self.min_evals_per_stage < 1:
            raise ValueError("min_evals_per_stage must be >= 1")
        if self.max_evals_per_stage < self.min_evals_per_stage:
            raise ValueError("max_evals_per_stage must be >= min_evals_per_stage")
        if self.convergence_window < 2:
            raise ValueError("convergence_window must be >= 2 (needs latent deltas)")
        for name in ("objective_rel_tol", "latent_rms_tol", "latent_max_tol"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        return self

    def to_dict(self) -> dict:
        # Included verbatim in the run contract BEFORE config_hash is computed,
        # so any change to the adaptive policy refuses to resume older attempts.
        return {
            "policy": POLICY_VERSION,
            "min_evals_per_stage": int(self.min_evals_per_stage),
            "max_evals_per_stage": int(self.max_evals_per_stage),
            "convergence_window": int(self.convergence_window),
            "objective_rel_tol": float(self.objective_rel_tol),
            "latent_rms_tol": float(self.latent_rms_tol),
            "latent_max_tol": float(self.latent_max_tol),
        }


class StageController:
    """Per-beta-stage decision maker; feed one record per objective eval."""

    def __init__(self, config: AdaptiveConfig, beta: float):
        self.config = config.validate()
        self.beta = float(beta)
        self.objectives: list[float] = []
        self.g_solid: list[float] = []
        self.g_void: list[float] = []
        self._latents: deque = deque(maxlen=config.convergence_window)
        self.stop_reason: str | None = None
        self.last_latent: np.ndarray | None = None

    # -- feeding -------------------------------------------------------------
    def record(self, objective, g_solid, g_void, latent) -> str:
        """Register one evaluation; returns CONTINUE / an ADVANCE_/ABORT_ reason."""
        if self.stop_reason is not None:
            return self.stop_reason  # decision is final for this stage
        latent = np.array(latent, dtype=float, copy=True).reshape(-1)
        self.last_latent = latent
        self.objectives.append(float(objective))
        self.g_solid.append(float(g_solid))
        self.g_void.append(float(g_void))
        self._latents.append(latent)

        cfg = self.config
        n = len(self.objectives)
        if n >= cfg.max_evals_per_stage:
            self.stop_reason = (
                REASON_MAXEVAL_FEASIBLE if self.stage_saw_feasible
                else REASON_MAXEVAL_INFEASIBLE
            )
            return self.stop_reason
        if n < cfg.min_evals_per_stage or n < cfg.convergence_window:
            return CONTINUE

        metrics = self.window_metrics()
        plateau = metrics["objective_rel_span"] < cfg.objective_rel_tol
        quiet = (
            metrics["latent_rms_change"] < cfg.latent_rms_tol
            and metrics["latent_max_change"] < cfg.latent_max_tol
        )
        if plateau and quiet:
            self.stop_reason = (
                REASON_CONVERGED if metrics["window_feasible"]
                else REASON_STALLED_INFEASIBLE
            )
            return self.stop_reason
        return CONTINUE

    # -- introspection -------------------------------------------------------
    @property
    def evaluations(self) -> int:
        return len(self.objectives)

    @property
    def stage_saw_feasible(self) -> bool:
        return any(
            gs <= 0.0 and gv <= 0.0
            for gs, gv in zip(self.g_solid, self.g_void)
        )

    def window_metrics(self) -> dict:
        w = self.config.convergence_window
        obj = self.objectives[-w:]
        gs = self.g_solid[-w:]
        gv = self.g_void[-w:]
        hi, lo = max(obj), min(obj)
        rel_span = (hi - lo) / max(abs(hi), abs(lo), np.finfo(float).tiny)
        rms_change = float("inf")
        max_change = float("inf")
        if len(self._latents) >= 2:
            rms_list, max_list = [], []
            latents = list(self._latents)
            for previous, current in zip(latents[:-1], latents[1:]):
                delta = current - previous
                rms_list.append(float(np.sqrt(np.mean(delta * delta))))
                max_list.append(float(np.max(np.abs(delta))))
            rms_change = max(rms_list)
            max_change = max(max_list)
        return {
            "objective_window": obj,
            "objective_rel_span": float(rel_span),
            "g_solid_window": gs,
            "g_void_window": gv,
            "window_feasible": all(
                a <= 0.0 and b <= 0.0 for a, b in zip(gs, gv)
            ),
            "latent_rms_change": rms_change,
            "latent_max_change": max_change,
        }

    def summary(self, stop_reason: str | None = None) -> dict:
        """Stage-end record for history.jsonl / stop.json."""
        metrics = self.window_metrics() if self.objectives else {}
        return {
            "beta": self.beta,
            "evaluations": self.evaluations,
            "stop_reason": stop_reason or self.stop_reason,
            "stage_saw_feasible": self.stage_saw_feasible,
            "adaptive_config": self.config.to_dict(),
            **metrics,
        }


def is_intentional_stop(exc: BaseException, controller: StageController) -> bool:
    """True iff `exc` is nlopt's ForcedStop raised by OUR force_stop() call.

    An intentional adaptive stop must never fall through to the
    license/solver/numerical failure classifier.  nlopt is imported lazily so
    this module (and its tests) stay importable without it.
    """
    if controller is None or controller.stop_reason is None:
        return False
    try:
        import nlopt
    except Exception:  # pragma: no cover - nlopt always present in production
        return False
    return isinstance(exc, nlopt.ForcedStop)
