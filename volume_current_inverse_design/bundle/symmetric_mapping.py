"""[LEGACY 2026-07-24] Reproduce old runs only; NO minimum-length-scale
guarantee. Production uses periodic_constrained_mapping.PeriodicConstrainedMapping.

Symmetric open-close density mapping (unbiased MFS/MGS enforcement).

Drop-in replacement for the dilation-biased morphology cascade.  Chain:

    latent(2D) --conic smooth--> --tanh(0.5) binarize-->
      --OPENING (erode then dilate, radius r_mfs)-->   # min feature (MFS)
      --CLOSING (dilate then erode, radius r_mgs)-->    # min gap    (MGS)
      --vertical extrude to Nz--> --seam wrap-->

Erode = conic_filter + tanh at eta=0.75 ; dilate = conic_filter + tanh at
eta=0.25.  Opening and closing each use one erode and one dilate, so the
erode/dilate counts are balanced and there is NO solid bias: a uniform gray
latent does not drift toward solid, and MFS/MGS are enforced symmetrically for
material and void.

The conic radius per target length is calibrated so the MEASURED minimum
feature/gap equals the requested MFS/MGS (see verify_symmetric_mapping.py).
Everything is autograd.numpy so the whole map is differentiable and the
mapping VJP is exact.
"""

from __future__ import annotations

import os

import autograd.numpy as npa
import numpy as np

from msopt import Filters as Ft
from msopt.Sub_Mapping import get_conic_radius, tanh_projection_m


# Measured target->achieved length calibration (bar test, high beta):
# requesting an achieved length L needs an internal conic target of L*CAL.
# CAL=0.75 gives measured MFS=MGS=400 nm for a 400 nm request (see verifier).
LENGTH_CALIBRATION = float(os.environ.get("SYM_LENGTH_CAL", "0.75"))
ETA_ERODE = 0.75
ETA_DILATE = 0.25


def _conic(x2d, radius, L, res):
    if radius <= 0:
        return x2d
    flat = Ft.conic_filter(
        npa.clip(x2d, 0.0, 1.0).flatten(), radius, L, L, res, periodic_axes=[0, 1]
    )
    return npa.reshape(flat, x2d.shape)


def _erode(x2d, radius, beta, L, res):
    return tanh_projection_m(_conic(x2d, radius, L, res), beta, ETA_ERODE)


def _dilate(x2d, radius, beta, L, res):
    return tanh_projection_m(_conic(x2d, radius, L, res), beta, ETA_DILATE)


def _open_then_close(y, r_mfs, r_mgs, beta, L, res):
    y = _dilate(_erode(y, r_mfs, beta, L, res), r_mfs, beta, L, res)   # opening: MFS
    y = _erode(_dilate(y, r_mgs, beta, L, res), r_mgs, beta, L, res)   # closing: MGS
    return y


def _close_then_open(y, r_mfs, r_mgs, beta, L, res):
    y = _erode(_dilate(y, r_mgs, beta, L, res), r_mgs, beta, L, res)   # closing: MGS
    y = _dilate(_erode(y, r_mfs, beta, L, res), r_mfs, beta, L, res)   # opening: MFS
    return y


def symmetric_open_close_2d(latent2d, beta, mfs_um, mgs_um, period_um, res):
    """Unbiased MFS/MGS-aware 2D map (autograd differentiable).

    Averages open-close and close-open so neither the solid-completing nor the
    void-carving ordering dominates: a uniform gray latent maps to ~0.5 and the
    seed keeps a balanced solid fraction across beta.  Length-scale control is
    soft at this (design) stage; strict MFS/MGS is applied at final projection.
    """
    r_mfs = get_conic_radius(max(mfs_um * LENGTH_CALIBRATION, 1e-6), 1 - ETA_DILATE)
    r_mgs = get_conic_radius(max(mgs_um * LENGTH_CALIBRATION, 1e-6), 1 - ETA_DILATE)
    r_smooth = 0.5 * min(mfs_um, mgs_um) * LENGTH_CALIBRATION
    y = tanh_projection_m(_conic(latent2d, r_smooth, period_um, res), beta, 0.5)
    oc = _open_then_close(y, r_mfs, r_mgs, beta, period_um, res)
    co = _close_then_open(y, r_mfs, r_mgs, beta, period_um, res)
    return 0.5 * (oc + co)


class SymmetricSeamMapping:
    """latent (Nx*Ny) -> physical (Nx*Ny*Nz), seam-wrapped, z-extruded."""

    def __init__(self, Nx, Ny, Nz, res, period_um, mfs_um, mgs_um):
        self.Nx, self.Ny, self.Nz = int(Nx), int(Ny), int(Nz)
        self.res = float(res)
        self.period_um = float(period_um)
        self.mfs_um = float(mfs_um)
        self.mgs_um = float(mgs_um)
        self.Is_freeform = [True, False, False]

    def __call__(self, x, beta, Is_opt=True):
        latent2d = npa.reshape(x, (self.Nx, self.Ny))
        ref = symmetric_open_close_2d(
            latent2d, beta, self.mfs_um, self.mgs_um, self.period_um, self.res
        )
        # exact periodic seam wrap (last x/y plane copies the first)
        ref = npa.concatenate([ref[: self.Nx - 1], ref[:1]], axis=0)
        ref = npa.concatenate([ref[:, : self.Ny - 1], ref[:, :1]], axis=1)
        # vertical extrusion: identical reference layer on every z node
        volume = npa.repeat(ref[:, :, None], self.Nz, axis=2)
        return npa.reshape(volume, (-1,))


# --------------------------------------------------------------------------
# Standard filter + projection mapping (unbiased, binarizes cleanly).
# This is the recommended design mapping: a single periodic conic filter sets
# the smoothing length scale (softly enforcing min feature AND min gap
# symmetrically), and one tanh(eta=0.5) projection binarizes.  Uniform gray
# latent maps to 0.5 (no bias); high-contrast designs binarize as beta grows.
# --------------------------------------------------------------------------

def filter_project_2d(latent2d, beta, radius_um, period_um, res):
    filt = _conic(latent2d, radius_um, period_um, res)
    return tanh_projection_m(filt, beta, 0.5)


class FilterProjectMapping:
    """latent (Nx*Ny) -> physical (Nx*Ny*Nz), seam-wrapped, z-extruded.

    radius_um is the conic filter radius; the softly enforced minimum feature
    and minimum gap are both ~radius_um (measured in verify_symmetric_mapping).
    """

    def __init__(self, Nx, Ny, Nz, res, period_um, mfs_um, mgs_um,
                 border_air_um=None):
        self.Nx, self.Ny, self.Nz = int(Nx), int(Ny), int(Nz)
        self.res = float(res)
        self.period_um = float(period_um)
        self.mfs_um = float(mfs_um)
        self.mgs_um = float(mgs_um)
        # single symmetric length scale; MFS and MGS are enforced equally
        self.radius_um = max(self.mfs_um, self.mgs_um)
        self.Is_freeform = [True, False, False]
        # Optional forced-air border: the outer `border_air_um` on all four
        # x/y edges of the unit cell is clamped to void.  Under periodic
        # tiling this isolates each cell into a (period-2*border) island
        # separated from its neighbours by a 2*border air moat.
        if border_air_um is None:
            border_air_um = float(os.environ.get("MSOPT_BORDER_AIR_UM", "0.0"))
        self.border_air_um = float(border_air_um)
        # During gradient optimization the border is clamped to a tiny air-like
        # floor (not exactly 0) so the centered rho->epsilon Jacobian probe
        # stays inside [rho_step, 1-rho_step]; the border gradient is zeroed by
        # the mask regardless, so this only keeps the fail-closed guards happy.
        # n(rho=0.003) = 1.02, epsilon = 1.045 -- essentially air.  The final
        # fabricable design can be clamped to exactly 0 at projection time.
        self.border_floor = float(os.environ.get("MSOPT_BORDER_AIR_FLOOR", "0.003"))
        self._mask = None
        if self.border_air_um > 0:
            xs = (np.arange(self.Nx) - 0.5 * (self.Nx - 1)) / self.res
            ys = (np.arange(self.Ny) - 0.5 * (self.Ny - 1)) / self.res
            inner = 0.5 * self.period_um - self.border_air_um
            keep = (np.abs(xs)[:, None] < inner) & (np.abs(ys)[None, :] < inner)
            self._mask = keep.astype(float)

    def __call__(self, x, beta, Is_opt=True):
        latent2d = npa.reshape(x, (self.Nx, self.Ny))
        if self._mask is not None:
            # 1) impose air in the frame BEFORE the conic filter so the filter
            #    smooths the material->air-frame transition over the length
            #    scale (MFS respected at the island edge) instead of a hard cut.
            latent2d = latent2d * self._mask
        ref = filter_project_2d(
            latent2d, beta, self.radius_um, self.period_um, self.res
        )
        if self._mask is not None:
            # The forced-air frame is imposed at the LATENT (step 1), so the
            # conic filter + projection process the air region TOGETHER with the
            # interior structure -- the material tapers smoothly to air over the
            # length scale, with no post-hoc hard cut at the frame edge.  The
            # only extra step is a uniform floor so every node stays >=
            # border_floor > rho_step (centered Jacobian probe safe); void and
            # near-air sit at n=1.02, and exact 0 comes at final projection.
            ref = npa.maximum(ref, self.border_floor)
        ref = npa.concatenate([ref[: self.Nx - 1], ref[:1]], axis=0)
        ref = npa.concatenate([ref[:, : self.Ny - 1], ref[:, :1]], axis=1)
        volume = npa.repeat(ref[:, :, None], self.Nz, axis=2)
        return npa.reshape(volume, (-1,))
