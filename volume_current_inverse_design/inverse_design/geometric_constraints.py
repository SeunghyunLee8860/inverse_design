"""Differentiable minimum-length-scale constraints (Zhou et al., CMAME 2015).

Wraps ``msopt.Filters.constraint_solid`` / ``constraint_void`` around the SAME
periodic conic filter and tanh(0.5) projection used by
``PeriodicConstrainedMapping``, on the 240x240 unique torus.  The solid
constraint penalises solid features thinner than the length scale; the void
constraint penalises thin voids/gaps.  Both are autograd-differentiable, so the
constrained optimiser (NLopt MMA/CCSA) receives exact gradients.

These are the *optimisation-side* length-scale drivers.  They raise the odds of
passing the independent ``geometry_drc`` gate but are NOT the operational
guarantee -- the DRC is (see run_constrained_inverse_design finalisation).

Constraint form (recorded in the run contract):

    g_solid(latent, beta) = C_solid(latent, beta) - tol_solid   <= 0
    g_void (latent, beta) = C_void (latent, beta) - tol_void    <= 0

C_solid/C_void are the Zhou indicator-weighted penalties; a fully binary design
respecting the length scale drives them to ~0.  ``tol`` gives the optimiser a
small feasible band so it is not forced to an exact-equality manifold.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import autograd.numpy as anp
import numpy as np
from autograd import grad

from msopt.Filters import constraint_solid, constraint_void, get_conic_radius_from_eta_e
from msopt.Sub_Mapping import tanh_projection_m


@dataclass
class ConstraintConfig:
    c_decay: float
    eta_erode: float
    eta_dilate: float
    tol_solid: float
    tol_void: float
    resolution: float  # design nodes per micrometre (1/dx_um)

    def to_dict(self) -> dict:
        return {
            "form": "g = C - tol (<=0 feasible); C = Zhou2015 indicator penalty",
            "c_decay": self.c_decay,
            "eta_erode": self.eta_erode,
            "eta_dilate": self.eta_dilate,
            "tol_solid": self.tol_solid,
            "tol_void": self.tol_void,
            "resolution_per_um": self.resolution,
        }


class LengthScaleConstraints:
    """Solid/void minimum-length-scale constraints for a mapping's filter."""

    def __init__(
        self,
        mapping,
        c_decay: float | None = None,
        tol_solid: float | None = None,
        tol_void: float | None = None,
    ):
        self.mapping = mapping
        cfg = mapping.config
        self.periodic_axes = [0, 1]
        self.resolution = 1.0 / cfg.dx_um
        self.eta_erode = cfg.eta_erode
        self.eta_dilate = cfg.eta_dilate
        self.c_decay = float(
            c_decay if c_decay is not None else os.environ.get("VC_CONSTRAINT_C", 400.0)
        )
        self.tol_solid = float(
            tol_solid if tol_solid is not None else os.environ.get("VC_TOL_SOLID", 1e-5)
        )
        self.tol_void = float(
            tol_void if tol_void is not None else os.environ.get("VC_TOL_VOID", 1e-5)
        )

    @property
    def config(self) -> ConstraintConfig:
        return ConstraintConfig(
            c_decay=self.c_decay,
            eta_erode=self.eta_erode,
            eta_dilate=self.eta_dilate,
            tol_solid=self.tol_solid,
            tol_void=self.tol_void,
            resolution=self.resolution,
        )

    def _filter_f(self, x2d):
        return self.mapping.filter_unique(x2d)

    def _threshold_f(self, beta):
        return lambda xf: tanh_projection_m(xf, beta, self.mapping.config.eta_nominal)

    def _as2d(self, latent):
        return anp.reshape(latent, (self.mapping.Nux, self.mapping.Nuy))

    # -- raw penalties -------------------------------------------------------
    def solid_penalty(self, latent, beta):
        x2d = self._as2d(latent)
        return constraint_solid(
            x2d, self.c_decay, self.eta_erode, self._filter_f,
            self._threshold_f(beta), self.resolution, self.periodic_axes,
        )

    def void_penalty(self, latent, beta):
        x2d = self._as2d(latent)
        return constraint_void(
            x2d, self.c_decay, self.eta_dilate, self._filter_f,
            self._threshold_f(beta), self.resolution, self.periodic_axes,
        )

    # -- residuals (<=0 feasible) and gradients ------------------------------
    def residuals(self, latent, beta):
        return (
            float(self.solid_penalty(latent, beta)) - self.tol_solid,
            float(self.void_penalty(latent, beta)) - self.tol_void,
        )

    def solid_residual_and_grad(self, latent, beta):
        val = float(self.solid_penalty(latent, beta)) - self.tol_solid
        g = grad(lambda z: self.solid_penalty(z, beta))(np.asarray(latent, float))
        return val, np.asarray(g, float).reshape(-1)

    def void_residual_and_grad(self, latent, beta):
        val = float(self.void_penalty(latent, beta)) - self.tol_void
        g = grad(lambda z: self.void_penalty(z, beta))(np.asarray(latent, float))
        return val, np.asarray(g, float).reshape(-1)


def suggested_filter_radius_um(min_length_um: float, eta_erode: float = 0.75) -> float:
    """Conic radius whose eroded/dilated threshold pair targets `min_length_um`."""
    return float(get_conic_radius_from_eta_e(min_length_um, eta_erode))
