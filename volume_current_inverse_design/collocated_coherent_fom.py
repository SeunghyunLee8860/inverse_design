"""Physically collocated Ex/Ez coherent FOM on the raw R1.02 Yee grid.

Raw no-interpolation fields use the component registration measured in this
workspace: Ex is primal in x and dual in y/z, while Ez is dual in x/y and
primal in z.  Both components are therefore collocated at x/z face centers:

    Ex_c(i,j,k) = (Ex(i,j,k) + Ex(i,j,k+1))/2
    Ez_c(i,j,k) = (Ez(i,j,k) + Ez(i+1,j,k))/2

The returned q is dF/dE* on the original full FieldRegion array.  It includes
the exact transpose of both averages and splits dual-y periodic seam weights
between the duplicated endpoints used by FieldRegion source profiles.
"""

from __future__ import annotations

import numpy as np

from volume_current_adjoint_core import as_e5


def _coordinates(value: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(value, float).reshape(-1)
    if out.size < 2 or np.any(np.diff(out) <= 0):
        raise ValueError(f"{name} must contain at least two increasing points")
    return out


def _dual_periodic_width(intervals: np.ndarray) -> np.ndarray:
    """Periodic dual-node widths from the unique primal interval widths."""
    d = np.asarray(intervals, float).reshape(-1)
    return 0.5 * (d + np.roll(d, 1))


def collocated_coherent_fom_and_wirtinger(
    electric_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    coordinate_scale: float = 1e6,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, object]]:
    """Return F, S and q=dF/dE* for collocated |int Ex Ez* dV|^2.

    x and y include duplicated periodic endpoints.  z contains the closed FOM
    interval endpoints.  The common grid has shape (nx-1, ny-1, nz-1): x and
    z are interval centers and y is the set of unique periodic dual nodes.
    """
    e = as_e5(electric_field)
    xx = _coordinates(x, "x") * coordinate_scale
    yy = _coordinates(y, "y") * coordinate_scale
    zz = _coordinates(z, "z") * coordinate_scale
    if e.shape[:3] != (xx.size, yy.size, zz.size):
        raise ValueError(
            f"field grid {e.shape[:3]} != coordinates {(xx.size, yy.size, zz.size)}")

    # Meaningful raw component samples under the measured R1.02 registration.
    # Ex[-1,:,:] is the x-primal zero sentinel.  The final y sample duplicates
    # the periodic dual seam.  Ez has duplicated dual x/y seam endpoints; its
    # final z-primal sample lies beyond the final closed z interval.
    ex = e[:-1, :-1, :, :, 0]
    ez = e[:, :-1, :-1, :, 2]

    # Common physical location: (x interval center, y dual node, z interval center).
    ex_c = 0.5 * (ex[:, :, :-1, :] + ex[:, :, 1:, :])
    ez_c = 0.5 * (ez[:-1, :, :, :] + ez[1:, :, :, :])
    if ex_c.shape != ez_c.shape:
        raise AssertionError(f"collocated shape mismatch {ex_c.shape} != {ez_c.shape}")

    dx = np.diff(xx)
    dy = _dual_periodic_width(np.diff(yy))
    dz = np.diff(zz)
    weight = (dx[:, None, None] * dy[None, :, None] * dz[None, None, :])[..., None]
    overlap = np.sum(weight * ex_c * np.conj(ez_c), axis=(0, 1, 2))
    fom = float(np.sum(np.abs(overlap) ** 2))

    qex_c = overlap * weight * ez_c
    qez_c = np.conj(overlap) * weight * ex_c
    q = np.zeros_like(e)

    # C_ex^H: z-center average back to the two neighboring Ex z nodes.
    q[:-1, :-1, :-1, :, 0] += 0.5 * qex_c
    q[:-1, :-1, 1:, :, 0] += 0.5 * qex_c

    # C_ez^H: x-center average back to the two neighboring Ez x nodes.
    q[:-1, :-1, :-1, :, 2] += 0.5 * qez_c
    q[1:, :-1, :-1, :, 2] += 0.5 * qez_c

    # y is dual and its physical periodic seam is duplicated by FieldRegion.
    # Split the unique seam coefficient equally between first and last copies.
    for component in (0, 2):
        seam = np.array(q[:, 0, :, :, component], copy=True)
        q[:, 0, :, :, component] = 0.5 * seam
        q[:, -1, :, :, component] = 0.5 * seam

    contract = {
        "common_location": "x/z face center",
        "common_shape": list(ex_c.shape),
        "ex_operator": "z-neighbor arithmetic average",
        "ez_operator": "x-neighbor arithmetic average",
        "periodic_y_source_seam": "transpose split 1/2 + 1/2",
        "x_interval_um": [float(np.min(dx)), float(np.max(dx))],
        "y_dual_width_um": [float(np.min(dy)), float(np.max(dy))],
        "z_interval_um": [float(np.min(dz)), float(np.max(dz))],
    }
    return fom, overlap, q, contract



def collocated_coherent_fom_and_wirtinger_y(
    electric_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    coordinate_scale: float = 1e6,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, object]]:
    """Return F, S and q=dF/dE* for |int Ey_c Ez_c* dV|^2.

    Raw Ey is dual in x, primal in y and dual in z. Raw Ez is dual in
    x/y and primal in z. They are collocated at x-dual / y-z face centers:

        Ey_c(i,j,k) = [Ey(i,j,k) + Ey(i,j,k+1)]/2
        Ez_c(i,j,k) = [Ez(i,j,k) + Ez(i,j+1,k)]/2.
    """
    e = as_e5(electric_field)
    xx = _coordinates(x, "x") * coordinate_scale
    yy = _coordinates(y, "y") * coordinate_scale
    zz = _coordinates(z, "z") * coordinate_scale
    if e.shape[:3] != (xx.size, yy.size, zz.size):
        raise ValueError(
            f"field grid {e.shape[:3]} != coordinates {(xx.size, yy.size, zz.size)}"
        )

    ey = e[:-1, :-1, :, :, 1]
    ez = e[:-1, :, :-1, :, 2]
    ey_c = 0.5 * (ey[:, :, :-1, :] + ey[:, :, 1:, :])
    ez_c = 0.5 * (ez[:, :-1, :, :] + ez[:, 1:, :, :])
    if ey_c.shape != ez_c.shape:
        raise AssertionError(f"collocated shape mismatch {ey_c.shape} != {ez_c.shape}")

    dx = _dual_periodic_width(np.diff(xx))
    dy = np.diff(yy)
    dz = np.diff(zz)
    weight = (dx[:, None, None] * dy[None, :, None] * dz[None, None, :])[..., None]
    overlap = np.sum(weight * ey_c * np.conj(ez_c), axis=(0, 1, 2))
    fom = float(np.sum(np.abs(overlap) ** 2))

    qey_c = overlap * weight * ez_c
    qez_c = np.conj(overlap) * weight * ey_c
    q = np.zeros_like(e)

    q[:-1, :-1, :-1, :, 1] += 0.5 * qey_c
    q[:-1, :-1, 1:, :, 1] += 0.5 * qey_c
    q[:-1, :-1, :-1, :, 2] += 0.5 * qez_c
    q[:-1, 1:, :-1, :, 2] += 0.5 * qez_c

    # x is the duplicated periodic dual-node axis for both components.
    for component in (1, 2):
        seam = np.array(q[0, :, :, :, component], copy=True)
        q[0, :, :, :, component] = 0.5 * seam
        q[-1, :, :, :, component] = 0.5 * seam

    contract = {
        "common_location": "x-dual / y-z face center",
        "common_shape": list(ey_c.shape),
        "ey_operator": "z-neighbor arithmetic average",
        "ez_operator": "y-neighbor arithmetic average",
        "periodic_x_source_seam": "transpose split 1/2 + 1/2",
        "x_dual_width_um": [float(np.min(dx)), float(np.max(dx))],
        "y_interval_um": [float(np.min(dy)), float(np.max(dy))],
        "z_interval_um": [float(np.min(dz)), float(np.max(dz))],
    }
    return fom, overlap, q, contract

def collocated_component_energy_and_wirtinger(
    electric_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    component: str,
    coordinate_scale: float = 1e6,
) -> tuple[float, np.ndarray]:
    """Return int|C_component E_component|^2 dV and q=dF/dE*.

    This diagnostic uses the same x/z face-center collocation and exact
    transpose as the coherent FOM, without Ex<->Ez cross-component mapping.
    """
    e = as_e5(electric_field)
    xx = _coordinates(x, "x") * coordinate_scale
    yy = _coordinates(y, "y") * coordinate_scale
    zz = _coordinates(z, "z") * coordinate_scale
    if e.shape[:3] != (xx.size, yy.size, zz.size):
        raise ValueError(
            f"field grid {e.shape[:3]} != coordinates {(xx.size, yy.size, zz.size)}")
    dx = np.diff(xx)
    dy = _dual_periodic_width(np.diff(yy))
    dz = np.diff(zz)
    weight = (dx[:, None, None] * dy[None, :, None] * dz[None, None, :])[..., None]
    q = np.zeros_like(e)

    if component == "x":
        raw = e[:-1, :-1, :, :, 0]
        common = 0.5 * (raw[:, :, :-1, :] + raw[:, :, 1:, :])
        q_common = weight * common
        q[:-1, :-1, :-1, :, 0] += 0.5 * q_common
        q[:-1, :-1, 1:, :, 0] += 0.5 * q_common
        c = 0
    elif component == "z":
        raw = e[:, :-1, :-1, :, 2]
        common = 0.5 * (raw[:-1, :, :, :] + raw[1:, :, :, :])
        q_common = weight * common
        q[:-1, :-1, :-1, :, 2] += 0.5 * q_common
        q[1:, :-1, :-1, :, 2] += 0.5 * q_common
        c = 2
    else:
        raise ValueError("component must be 'x' or 'z'")

    seam = np.array(q[:, 0, :, :, c], copy=True)
    q[:, 0, :, :, c] = 0.5 * seam
    q[:, -1, :, :, c] = 0.5 * seam
    energy = float(np.sum(weight * np.abs(common) ** 2))
    return energy, q


def fieldregion_periodic_source_right_inverse(q_observation: np.ndarray) -> np.ndarray:
    """Map duplicated-monitor coefficients to FieldRegion's active source copy.

    R1.02 monitoring duplicates dual periodic endpoint DOFs, while source mode
    uses the first copy and discards the last.  Therefore the source right-
    inverse folds all duplicate coefficients into the first copy and zeros the
    last.  Component registrations are Ex: dual-y; Ey: dual-x; Ez: dual-x/y.
    """
    q = as_e5(q_observation)
    out = np.array(q, copy=True)

    # Ex is dual in y; its x-primal last entry is already a zero sentinel.
    out[:, 0, :, :, 0] += out[:, -1, :, :, 0]
    out[:, -1, :, :, 0] = 0.0

    # Ey is dual in x; its y-primal last entry is a zero sentinel.
    out[0, :, :, :, 1] += out[-1, :, :, :, 1]
    out[-1, :, :, :, 1] = 0.0

    # Ez is dual in both x and y. Sequential folds sum all four corner copies.
    out[0, :, :, :, 2] += out[-1, :, :, :, 2]
    out[-1, :, :, :, 2] = 0.0
    out[:, 0, :, :, 2] += out[:, -1, :, :, 2]
    out[:, -1, :, :, 2] = 0.0
    return out
