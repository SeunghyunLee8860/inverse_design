"""Thin audited wrapper around the Ansys v261 LumOpt DFM indicator.

The production topology optimizer uses the same conic-filter, Heaviside, and
minimum-feature parameters for the optical material map and for the official
``topoparamstominfeaturesize*`` geometry penalty.  The calls are CAD-only and
run inside an already-open FDTD session; they do not launch another Maxwell
solve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AnsysMinimumFeatureContract:
    filter_radius_m: float = 300.0e-9
    minimum_feature_m: float = 500.0e-9
    eta: float = 0.5
    dx_m: float = 100.0e-9
    dy_m: float = 100.0e-9
    activation_beta: float = 12.0
    penalty_factor: float = 100.0
    maximum_penalty: float = 1.0e4

    @property
    def delta_eta(self) -> float:
        scaled = self.minimum_feature_m / (2.0 * self.filter_radius_m)
        if self.minimum_feature_m < self.filter_radius_m:
            return float(scaled * scaled)
        return float(0.5 - (1.0 - scaled) ** 2)

    @property
    def eta_d(self) -> float:
        return float(self.eta - self.delta_eta)

    @property
    def eta_e(self) -> float:
        return float(self.eta + self.delta_eta)

    def penalty_scaling(self, beta: float) -> float:
        value = float(beta)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("beta must be finite and positive")
        if value <= self.activation_beta:
            return 0.0
        return float(
            min(
                self.penalty_factor * (value - self.activation_beta) ** 2,
                self.maximum_penalty,
            )
        )

    def validate(self) -> None:
        if not 0.0 < self.minimum_feature_m < 2.0 * self.filter_radius_m:
            raise ValueError(
                "Ansys DFM requires 0 < minimum_feature < 2*filter_radius"
            )
        if min(self.dx_m, self.dy_m, self.activation_beta) <= 0.0:
            raise ValueError("DFM grid and activation beta must be positive")
        if not 0.0 < self.eta_d < self.eta < self.eta_e < 1.0:
            raise ValueError("invalid eroded/intermediate/dilated thresholds")

    def audit(self) -> dict[str, object]:
        self.validate()
        return {
            "source": "/opt/lumerical/v261/api/python/lumopt/geometries/topology.py",
            "implementation": (
                "topoparamstominfeaturesizeindicator and "
                "topoparamstominfeaturesizegradient"
            ),
            "contract": asdict(self),
            "derived": {
                "delta_eta": self.delta_eta,
                "eta_d": self.eta_d,
                "eta_e": self.eta_e,
            },
            "extra_Maxwell_solves": 0,
        }


CONTRACT = AnsysMinimumFeatureContract()


def evaluate_on_cad(
    fdtd: Any,
    latent: np.ndarray,
    beta: float,
    *,
    contract: AnsysMinimumFeatureContract = CONTRACT,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return official solid/void indicators and their summed raw gradient."""

    contract.validate()
    value = np.asarray(latent, dtype=np.float64)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("latent DFM field must be a finite two-dimensional array")
    if np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("latent DFM field must lie in [0,1]")
    scaling = contract.penalty_scaling(beta)
    if scaling == 0.0:
        return (
            np.zeros(2, dtype=np.float64),
            np.zeros_like(value),
            {**contract.audit(), "beta": float(beta), "penalty_scaling": 0.0},
        )

    fdtd.putv("codex_dfm_topo_rho", value)
    fdtd.eval(
        "codex_dfm_params=struct;"
        "codex_dfm_params.eps_levels=[1,2];"
        f"codex_dfm_params.filter_radius={contract.filter_radius_m:.17g};"
        f"codex_dfm_params.beta={float(beta):.17g};"
        f"codex_dfm_params.eta={contract.eta:.17g};"
        f"codex_dfm_params.eta_e={contract.eta_e:.17g};"
        f"codex_dfm_params.eta_d={contract.eta_d:.17g};"
        f"codex_dfm_params.dx={contract.dx_m:.17g};"
        f"codex_dfm_params.dy={contract.dy_m:.17g};"
        "codex_dfm_params.dz=0.0;"
        "codex_dfm_indicators=topoparamstominfeaturesizeindicator("
        "codex_dfm_params,codex_dfm_topo_rho);"
        "codex_dfm_gradient=topoparamstominfeaturesizegradient("
        "codex_dfm_params,codex_dfm_topo_rho);"
    )
    indicators = np.asarray(fdtd.getv("codex_dfm_indicators"), dtype=np.float64).reshape(-1)
    gradient = np.asarray(fdtd.getv("codex_dfm_gradient"), dtype=np.float64).squeeze()
    if indicators.shape != (2,):
        raise RuntimeError(f"Ansys DFM returned indicator shape {indicators.shape}")
    if gradient.shape != value.shape:
        raise RuntimeError(
            f"Ansys DFM returned gradient shape {gradient.shape}, expected {value.shape}"
        )
    if not np.all(np.isfinite(indicators)) or not np.all(np.isfinite(gradient)):
        raise RuntimeError("Ansys DFM returned NaN or Inf")
    return (
        indicators,
        gradient,
        {
            **contract.audit(),
            "beta": float(beta),
            "penalty_scaling": scaling,
            "indicator_solid": float(indicators[0]),
            "indicator_void": float(indicators[1]),
            "indicator_sum": float(np.sum(indicators)),
            "gradient_l2": float(np.linalg.norm(gradient)),
            "gradient_max_abs": float(np.max(np.abs(gradient))),
        },
    )
