"""Matrix-free colored transpose of the solver-measured rho->epsilon map.

Independent variables live on ``rho[:-1, :-1, :]``; the last x/y planes are
periodic fencepost copies.  For a measured stencil of radius at most one,
modulo-three colors have disjoint output support.  A centered layout-only
epsilon probe per color can therefore apply J.T without storing the huge J.
"""

from __future__ import annotations

from itertools import product
import numpy as np

COLOR_STRIDE = 3


def expand_periodic_unique(unique: np.ndarray) -> np.ndarray:
    u = np.asarray(unique)
    if u.ndim != 3:
        raise ValueError(f"expected unique rho (nx,ny,nz), got {u.shape}")
    out = np.empty((u.shape[0] + 1, u.shape[1] + 1, u.shape[2]), dtype=u.dtype)
    out[:-1, :-1] = u
    out[-1, :-1] = u[0]
    out[:-1, -1] = u[:, 0]
    out[-1, -1] = u[0, 0]
    return out


def unique_from_periodic(full: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    a = np.asarray(full)
    if a.ndim != 3 or min(a.shape[:2]) < 2:
        raise ValueError(f"expected periodic rho (nx+1,ny+1,nz), got {a.shape}")
    scale = max(float(np.max(np.abs(a))), 1.0)
    if np.max(np.abs(a[-1] - a[0])) > tolerance * scale:
        raise ValueError("rho x fencepost is not periodic")
    if np.max(np.abs(a[:, -1] - a[:, 0])) > tolerance * scale:
        raise ValueError("rho y fencepost is not periodic")
    return np.array(a[:-1, :-1], copy=True)


def colors(stride: int = COLOR_STRIDE):
    return product(range(stride), repeat=3)


def color_direction(unique_shape, color, stride: int = COLOR_STRIDE) -> np.ndarray:
    if len(unique_shape) != 3 or len(color) != 3:
        raise ValueError("unique_shape and color must be 3-D")
    u = np.zeros(unique_shape, float)
    u[color[0]::stride, color[1]::stride, color[2]::stride] = 1.0
    return expand_periodic_unique(u)


def _periodic_owner(n_unique, n_fine, color, stride):
    candidates = np.arange(color, n_unique, stride, dtype=int)
    if candidates.size == 0:
        raise ValueError(f"empty periodic color {color} for size {n_unique}")
    fine = np.arange(n_fine, dtype=int) % n_unique
    delta = np.abs(fine[:, None] - candidates[None, :])
    delta = np.minimum(delta, n_unique - delta)
    return candidates[np.argmin(delta, axis=1)]


def _z_owner(design_z, fine_z, color, stride):
    zc = np.asarray(design_z, float).reshape(-1)
    zf = np.asarray(fine_z, float).reshape(-1)
    candidates = np.arange(color, zc.size, stride, dtype=int)
    if candidates.size == 0:
        raise ValueError(f"empty z color {color} for size {zc.size}")
    return candidates[np.argmin(np.abs(zf[:, None] - zc[candidates][None, :]), axis=1)]


def owner_axes(unique_shape, fine_shape, design_z, fine_z, color,
               stride: int = COLOR_STRIDE):
    return (
        _periodic_owner(unique_shape[0], fine_shape[0], color[0], stride),
        _periodic_owner(unique_shape[1], fine_shape[1], color[1], stride),
        _z_owner(design_z, fine_z, color[2], stride),
    )


def accumulate_color_transpose(gradient, fine_sensitivity, d_eps_color, owners):
    """Accumulate one color for one or several fine sensitivity arrays."""
    de = np.asarray(d_eps_color)
    s = np.asarray(fine_sensitivity)
    single = s.ndim == 5
    if single:
        s = s[None]
    if de.ndim != 5 or de.shape[-1] != 3:
        raise ValueError(f"expected d-epsilon (nx,ny,nz,nf,3), got {de.shape}")
    if s.ndim != 6 or s.shape[1:] != de.shape:
        raise ValueError(f"sensitivity {s.shape} incompatible with d-epsilon {de.shape}")
    target = gradient[None] if single else gradient
    if target.shape[0] != s.shape[0]:
        raise ValueError("gradient/sensitivity channel count mismatch")
    contribution = np.real(np.sum(s * de[None], axis=(-1, -2)))
    ox, oy, oz = owners
    owner_grid = np.broadcast_arrays(
        ox[:, None, None], oy[None, :, None], oz[None, None, :])
    for q in range(s.shape[0]):
        np.add.at(target[q], tuple(owner_grid), contribution[q])


def direction_inner_product(gradient_unique, direction_full) -> float:
    return float(np.sum(np.asarray(gradient_unique) * unique_from_periodic(direction_full)))
