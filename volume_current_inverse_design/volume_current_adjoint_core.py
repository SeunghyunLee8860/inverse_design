"""Validated discrete operators for coherent FieldRegion volume-current adjoints.

This module contains no solver launch code.  Its purpose is to make the
mathematical contract explicit and testable: coherent complex differentiation,
periodic quadrature, raw-Yee volumes from solver half-steps, periodic seam
transpose, Richardson differentiation, and gradient assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.constants import epsilon_0


def as_e5(value: np.ndarray) -> np.ndarray:
    """Canonicalize E to (nx, ny, nz, nf, 3)."""
    a = np.asarray(value, np.complex128)
    if a.ndim == 4:
        a = a[:, :, :, None, :]
    if a.ndim != 5 or a.shape[-1] != 3:
        raise ValueError(f"expected (..., nf, 3) electric field, got {a.shape}")
    return a


def axis_trapezoid(coords: np.ndarray) -> np.ndarray:
    """Integration weights for one closed, non-periodic coordinate interval."""
    x = np.asarray(coords, float).reshape(-1)
    if x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("coordinates must contain at least two increasing points")
    w = np.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def coherent_fom_and_wirtinger(
    electric_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    coordinate_scale: float = 1e6,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return F, overlap S, and q=dF/dE* for |int Ex Ez* dV|^2.

    x/y contain both endpoints of the periodic unit cell.  Trapezoid endpoint
    halves are therefore the periodic quotient weights when endpoint fields are
    identical.  z is a closed physical interval and uses the same rule.
    """
    e = as_e5(electric_field)
    wx = axis_trapezoid(np.asarray(x) * coordinate_scale)
    wy = axis_trapezoid(np.asarray(y) * coordinate_scale)
    wz = axis_trapezoid(np.asarray(z) * coordinate_scale)
    w = (wx[:, None, None] * wy[None, :, None] * wz[None, None, :])[..., None]
    if e.shape[:3] != w.shape[:3]:
        raise ValueError(f"field grid {e.shape[:3]} != weight grid {w.shape[:3]}")
    ex, ez = e[..., 0], e[..., 2]
    overlap = np.sum(w * ex * np.conj(ez), axis=(0, 1, 2))
    fom = float(np.sum(np.abs(overlap) ** 2))
    q = np.zeros_like(e)
    q[..., 0] = overlap * w * ez
    q[..., 2] = np.conj(overlap) * w * ex
    return fom, overlap, q


def fieldregion_source_profile(q_d_f_d_e_conj: np.ndarray) -> tuple[np.ndarray, float]:
    """Return normalized R1.02 import profile dF/dE and its exact scale."""
    q = as_e5(q_d_f_d_e_conj)
    profile = np.conj(q)
    scale = float(np.max(np.abs(profile)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid source scale {scale}")
    return np.ascontiguousarray(profile / scale), scale


def periodic_yee_volumes_from_half_steps(
    delta_x_plus: np.ndarray,
    delta_y_plus: np.ndarray,
    delta_z_plus: np.ndarray,
) -> dict[int, np.ndarray]:
    """Build component Yee volumes from solver raw delta arrays.

    The delta arrays are half steps toward the next node.  x/y are periodic and
    include both endpoints, so every component receives half weight at both
    copies of the seam.  The first design-z node has equal 5 nm half-steps on
    both sides in this validated stack; the final 50 nm global step continues
    above the design region.
    """
    dxp = np.asarray(delta_x_plus, float).reshape(-1)
    dyp = np.asarray(delta_y_plus, float).reshape(-1)
    dzp = np.asarray(delta_z_plus, float).reshape(-1)
    if min(dxp.size, dyp.size, dzp.size) < 2:
        raise ValueError("each delta axis must contain at least two samples")
    if min(np.min(dxp), np.min(dyp), np.min(dzp)) <= 0:
        raise ValueError("solver deltas must be positive")

    dxm = np.roll(dxp, 1)
    dym = np.roll(dyp, 1)
    dzm = np.empty_like(dzp)
    dzm[1:] = dzp[:-1]
    dzm[0] = dzp[0]

    dual_x, dual_y, dual_z = dxm + dxp, dym + dyp, dzm + dzp
    primal_x, primal_y, primal_z = 2 * dxp, 2 * dyp, 2 * dzp
    outer = lambda a, b, c: a[:, None, None] * b[None, :, None] * c[None, None, :]
    full = {
        0: outer(primal_x, dual_y, dual_z),
        1: outer(dual_x, primal_y, dual_z),
        2: outer(dual_x, dual_y, primal_z),
    }
    seam_x = np.ones(dxp.size)
    seam_y = np.ones(dyp.size)
    seam_x[[0, -1]] = 0.5
    seam_y[[0, -1]] = 0.5
    quotient = seam_x[:, None, None] * seam_y[None, :, None]
    return {c: full[c] * quotient for c in range(3)}


def fold_periodic_seam(full_gradient: np.ndarray) -> np.ndarray:
    """Transpose a duplicated (nx+1,ny+1,...) grid to unique periodic nodes."""
    g = np.asarray(full_gradient)
    if g.shape[0] < 2 or g.shape[1] < 2:
        raise ValueError("periodic grid must have duplicated endpoints")
    out = np.array(g[:-1, :-1, ...], copy=True)
    out[0, :, ...] += g[-1, :-1, ...]
    out[:, 0, ...] += g[:-1, -1, ...]
    out[0, 0, ...] += g[-1, -1, ...]
    return out


def expand_periodic_unique(unique_density: np.ndarray) -> np.ndarray:
    """Forward map from unique periodic nodes to a duplicated endpoint grid."""
    u = np.asarray(unique_density)
    out = np.empty((u.shape[0] + 1, u.shape[1] + 1) + u.shape[2:], dtype=u.dtype)
    out[:-1, :-1, ...] = u
    out[-1, :-1, ...] = u[0, :, ...]
    out[:-1, -1, ...] = u[:, 0, ...]
    out[-1, -1, ...] = u[0, 0, ...]
    return out


def richardson(coarse, fine):
    """Fourth-order estimate from centered derivatives at h and h/2."""
    return fine + (fine - coarse) / 3.0


@dataclass(frozen=True)
class DerivativeTrust:
    step_change: float
    channel_step_change: Mapping[str, float]
    trustworthy: bool


def derivative_trust(
    coarse_total: float,
    fine_total: float,
    rich_total: float,
    coarse_channels: Mapping[str, float],
    fine_channels: Mapping[str, float],
    rich_channels: Mapping[str, float],
    total_tolerance: float = 0.02,
    channel_tolerance: float = 0.05,
) -> DerivativeTrust:
    step = abs(fine_total - coarse_total) / max(abs(rich_total), 1e-300)
    channel = {
        k: abs(fine_channels[k] - coarse_channels[k])
        / max(abs(rich_channels[k]), 1e-300)
        for k in rich_channels
    }
    return DerivativeTrust(
        step_change=float(step),
        channel_step_change=channel,
        trustworthy=bool(step <= total_tolerance and max(channel.values()) <= channel_tolerance),
    )


def assemble_overlap(
    forward_e: np.ndarray,
    adjoint_e_raw: np.ndarray,
    d_eps: np.ndarray,
    volumes: Mapping[int, np.ndarray],
    profile_scale: float,
    base_amplitude: float,
) -> tuple[float, list[float]]:
    """Assemble 2 eps0/base_amp Re sum dV E_f E_a d-epsilon."""
    ef = as_e5(forward_e)
    ea = as_e5(adjoint_e_raw) * float(profile_scale)
    de = as_e5(d_eps)
    if ef.shape != ea.shape or ef.shape != de.shape:
        raise ValueError(f"shape mismatch: forward={ef.shape}, adjoint={ea.shape}, deps={de.shape}")
    terms = [
        float(np.real(np.sum(
            (2 * epsilon_0 / float(base_amplitude))
            * np.asarray(volumes[c])[..., None]
            * ef[..., c] * ea[..., c] * de[..., c]
        )))
        for c in range(3)
    ]
    return float(sum(terms)), terms
