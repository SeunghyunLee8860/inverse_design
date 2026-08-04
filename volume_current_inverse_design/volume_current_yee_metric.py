"""Component-correct raw-Yee metric for a duplicated periodic monitor grid.

Raw R1.02 no-interpolation data uses different periodic endpoint conventions:

* Ex is primal in x: x[0] is the physical wrap interval and x[-1] is a zero
  sentinel; y remains a duplicated dual-node axis.
* Ey is primal in y with the analogous convention; x is dual-node.
* Ez is dual-node in both periodic x and y axes.

The z widths are built directly from solver half-step DELTA_Z data.  No extra
half-cell averaging of epsilon or field products is performed: index_c and E_c
are already registered at coordinate_c + DELTA_c by R1.02 FieldsNoInterp.
"""

from __future__ import annotations

import numpy as np


def component_periodic_yee_volumes(
    delta_x_plus: np.ndarray,
    delta_y_plus: np.ndarray,
    delta_z_plus: np.ndarray,
) -> dict[int, np.ndarray]:
    dxp = np.asarray(delta_x_plus, float).reshape(-1)
    dyp = np.asarray(delta_y_plus, float).reshape(-1)
    dzp = np.asarray(delta_z_plus, float).reshape(-1)
    if min(dxp.size, dyp.size, dzp.size) < 2:
        raise ValueError("each solver half-step axis needs at least two samples")
    if min(np.min(dxp), np.min(dyp), np.min(dzp)) <= 0:
        raise ValueError("solver half steps must be positive")

    dxm = np.roll(dxp, 1)
    dym = np.roll(dyp, 1)
    dzm = np.empty_like(dzp)
    dzm[0] = dzp[0]
    dzm[1:] = dzp[:-1]

    dual_x, dual_y, dual_z = dxm + dxp, dym + dyp, dzm + dzp
    primal_x, primal_y, primal_z = 2 * dxp, 2 * dyp, 2 * dzp
    outer = lambda a, b, c: a[:, None, None] * b[None, :, None] * c[None, None, :]

    x_dual_q = np.ones(dxp.size)
    y_dual_q = np.ones(dyp.size)
    x_dual_q[[0, -1]] = 0.5
    y_dual_q[[0, -1]] = 0.5
    x_primal_q = np.ones(dxp.size)
    y_primal_q = np.ones(dyp.size)
    x_primal_q[-1] = 0.0
    y_primal_q[-1] = 0.0

    return {
        0: outer(primal_x * x_primal_q, dual_y * y_dual_q, dual_z),
        1: outer(dual_x * x_dual_q, primal_y * y_primal_q, dual_z),
        2: outer(dual_x * x_dual_q, dual_y * y_dual_q, primal_z),
    }


def seam_contract(deps: np.ndarray, tolerance: float = 1e-10) -> dict[str, float | bool]:
    """Validate the observed R1.02 component endpoint convention."""
    a = np.asarray(deps)
    if a.ndim != 5 or a.shape[-1] != 3:
        raise ValueError(f"expected (nx,ny,nz,nf,3), got {a.shape}")

    def rel(x, y):
        return float(np.linalg.norm((x - y).ravel()) /
                     max(np.linalg.norm(x.ravel()), np.linalg.norm(y.ravel()), 1e-300))

    ex_first = float(np.linalg.norm(a[0, ..., 0].ravel()))
    ey_first = float(np.linalg.norm(a[:, 0, ..., 1].ravel()))
    values = {
        "ex_x_last_over_first": float(
            np.linalg.norm(a[-1, ..., 0].ravel()) / max(ex_first, 1e-300)),
        "ey_y_last_over_first": float(
            np.linalg.norm(a[:, -1, ..., 1].ravel()) / max(ey_first, 1e-300)),
        "ex_y_dual_seam_relative": rel(a[:, 0, ..., 0], a[:, -1, ..., 0]),
        "ey_x_dual_seam_relative": rel(a[0, ..., 1], a[-1, ..., 1]),
        "ez_x_dual_seam_relative": rel(a[0, ..., 2], a[-1, ..., 2]),
        "ez_y_dual_seam_relative": rel(a[:, 0, ..., 2], a[:, -1, ..., 2]),
    }
    values["valid"] = bool(max(values.values()) <= tolerance)
    return values
