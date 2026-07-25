"""Unique-periodic, three-field density mapping for length-scale control.

Replaces the single conic+tanh ``FilterProjectMapping`` (which does not enforce
a minimum length scale) and the gray-locked ``SymmetricSeamMapping``.

Design representation
---------------------
* optimisation variable: ``(Nux, Nuy)`` = 240x240 unique periodic latent
  (57,600 variables), NOT the 241x241 physical node set.
* everything -- filter, projection, geometric constraints -- is computed on the
  240x240 torus.
* only when handing a density to the FDTD solver is the fencepost appended
  (241x241) and the 2-D pattern extruded to the ``Nz`` z-layers.

Three fields
------------
From one filtered field ``xf`` three projections are formed at fixed thresholds:

    rho_dilated = tanh(xf, beta, eta_d),  eta_d < 0.5   (largest solid)
    rho_nominal = tanh(xf, beta, 0.5)
    rho_eroded  = tanh(xf, beta, eta_e),  eta_e > 0.5   (smallest solid)

``tanh_projection_m`` is monotonically DECREASING in eta, so for every node

    rho_eroded <= rho_nominal <= rho_dilated

holds to machine precision -- the ordering a robust (Wang-Sigmund-Jensen 2011)
formulation needs.  The nominal field is the default physical density; the
eroded/dilated fields feed the robust objective and the final fabrication
variants.

A uniform gray latent maps to 0.5 (no solid bias) because a constant survives
the conic filter unchanged and ``tanh(0.5, beta, 0.5) = 0.5``.

Minimum length scale is NOT claimed to be guaranteed by this mapping alone; it
is *enforced* by the differentiable solid/void constraints
(``inverse_design/geometric_constraints.py``) during optimisation and *verified*
by the independent DRC (``inverse_design/geometry_drc.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import autograd.numpy as anp
import numpy as np

from periodic_filter import periodic_conic_filter_unique
from msopt.Sub_Mapping import tanh_projection_m


@dataclass(frozen=True)
class MappingConfig:
    period_um: float
    dx_um: float
    dy_um: float
    filter_radius_um: float
    eta_dilate: float
    eta_nominal: float
    eta_erode: float

    def to_dict(self) -> dict:
        return {
            "period_um": self.period_um,
            "dx_um": self.dx_um,
            "dy_um": self.dy_um,
            "filter_radius_um": self.filter_radius_um,
            "eta_dilate": self.eta_dilate,
            "eta_nominal": self.eta_nominal,
            "eta_erode": self.eta_erode,
        }


# mapping implementation version -- recorded in run provenance so a resume with
# a changed mapping is never treated as the same attempt.
MAPPING_VERSION = "periodic_constrained_mapping/v1"

_FIELDS = ("eroded", "nominal", "dilated")


class PeriodicConstrainedMapping:
    """latent (Nux*Nuy) -> physical (Nx*Ny*Nz), three-field, seam-exact."""

    Is_freeform = [True, False, False]

    def __init__(
        self,
        Nx: int,
        Ny: int,
        Nz: int,
        period_um: float,
        filter_radius_um: float,
        eta_dilate: float = 0.25,
        eta_erode: float = 0.75,
        isolation_gap_um: float | None = None,
    ):
        self.Nx, self.Ny, self.Nz = int(Nx), int(Ny), int(Nz)
        self.Nux, self.Nuy = self.Nx - 1, self.Ny - 1
        if not (0.0 < eta_dilate < 0.5 < eta_erode < 1.0):
            raise ValueError("require eta_dilate < 0.5 < eta_erode in (0,1)")
        self.config = MappingConfig(
            period_um=float(period_um),
            dx_um=float(period_um) / self.Nux,
            dy_um=float(period_um) / self.Nuy,
            filter_radius_um=float(filter_radius_um),
            eta_dilate=float(eta_dilate),
            eta_nominal=0.5,
            eta_erode=float(eta_erode),
        )
        # Optional periodic isolation moat: force a void frame of half-width
        # gap/2 inside every unit-cell edge.  Under periodic tiling neighbouring
        # cells are then separated by a full `gap` void band.  Imposed on the
        # LATENT (before the filter) so the conic filter tapers material to void
        # over the length scale -- no hard cut, no post-filter floor.
        if isolation_gap_um is None:
            isolation_gap_um = float(os.environ.get("PERIODIC_ISOLATION_GAP_UM", "0.0"))
        self.isolation_gap_um = float(isolation_gap_um)
        self._mask = None
        if self.isolation_gap_um > 0:
            half = 0.5 * self.isolation_gap_um
            xs = (np.arange(self.Nux) + 0.5) * self.config.dx_um  # cell centres, 0..period
            ys = (np.arange(self.Nuy) + 0.5) * self.config.dy_um
            keep_x = (xs >= half) & (xs <= self.config.period_um - half)
            keep_y = (ys >= half) & (ys <= self.config.period_um - half)
            self._mask = (keep_x[:, None] & keep_y[None, :]).astype(float)

    # -- building blocks (all operate on the 240x240 unique torus) -----------
    def filter_unique(self, latent):
        x2d = anp.reshape(latent, (self.Nux, self.Nuy))
        if self._mask is not None:
            x2d = x2d * self._mask
        return periodic_conic_filter_unique(
            x2d, self.config.filter_radius_um, self.config.dx_um, self.config.dy_um
        )

    def project_unique(self, filtered, beta, eta):
        return tanh_projection_m(filtered, beta, eta)

    def three_fields_unique(self, latent, beta):
        xf = self.filter_unique(latent)
        eroded = self.project_unique(xf, beta, self.config.eta_erode)
        nominal = self.project_unique(xf, beta, self.config.eta_nominal)
        dilated = self.project_unique(xf, beta, self.config.eta_dilate)
        return eroded, nominal, dilated

    def field_unique(self, latent, beta, field="nominal"):
        xf = self.filter_unique(latent)
        eta = {
            "eroded": self.config.eta_erode,
            "nominal": self.config.eta_nominal,
            "dilated": self.config.eta_dilate,
        }[field]
        return self.project_unique(xf, beta, eta)

    # -- unique -> physical --------------------------------------------------
    def append_fencepost(self, unique2d):
        ref = anp.concatenate([unique2d, unique2d[:1, :]], axis=0)  # -> Nx rows
        ref = anp.concatenate([ref, ref[:, :1]], axis=1)            # -> Ny cols
        return ref

    def extrude(self, full2d):
        return anp.repeat(full2d[:, :, None], self.Nz, axis=2)

    def physical_from_unique(self, unique2d):
        return self.extrude(self.append_fencepost(unique2d))

    def physical(self, latent, beta, field="nominal"):
        """Return flat physical density (Nx*Ny*Nz) for the requested field."""
        unique = self.field_unique(latent, beta, field)
        return anp.reshape(self.physical_from_unique(unique), (-1,))

    def three_fields_physical(self, latent, beta):
        eroded, nominal, dilated = self.three_fields_unique(latent, beta)
        return (
            anp.reshape(self.physical_from_unique(eroded), (-1,)),
            anp.reshape(self.physical_from_unique(nominal), (-1,)),
            anp.reshape(self.physical_from_unique(dilated), (-1,)),
        )

    # -- drop-in call: nominal physical (matches legacy mapping(x, beta)) ----
    def __call__(self, x, beta, Is_opt=True):
        return self.physical(x, beta, field="nominal")
