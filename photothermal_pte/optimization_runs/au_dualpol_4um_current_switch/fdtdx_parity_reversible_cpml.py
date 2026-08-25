"""Algebraic inverse of pinned FDTDX CPML auxiliary recurrences.

This avoids the existing reversible path's per-step full-precision E/H
interface recorder.  It is a bounded reconstruction primitive, not yet an
exact-grid custom VJP.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_reversible_ade_step import (
    update_E_reverse_diagonal_c4_free_ade,
)


def exact_pml_interface_recorder_audit(
    *, grid_shape: tuple[int, int, int], time_steps: int, itemsize: int = 4
) -> dict[str, Any]:
    """Memory of FDTDX's six-face E/H interface recorder without compression."""

    nx, ny, nz = (int(value) for value in grid_shape)
    if min(nx, ny, nz, time_steps, itemsize) <= 0:
        raise ValueError("recorder audit inputs must be positive")
    interface_cells = 2 * (ny * nz + nx * nz + nx * ny)
    bytes_per_step = interface_cells * 3 * 2 * itemsize
    total_bytes = bytes_per_step * int(time_steps)
    return {
        "schema": "fdtdx_4um_parity_exact_PML_interface_recorder_v1",
        "status": "BLOCKED_EXACT_FULL_RATE_RECORDER",
        "grid_shape": [nx, ny, nz],
        "time_steps": int(time_steps),
        "six_face_interface_cells_with_corner_duplication": interface_cells,
        "field_components": 3,
        "fields": ["E", "H"],
        "itemsize": int(itemsize),
        "bytes_per_step": bytes_per_step,
        "total_bytes": total_bytes,
        "total_TiB": total_bytes / 2**40,
        "lossy_time_filter_allowed_without_ADFD_revalidation": False,
    }


def cpml_inverse_coefficient_audit(objects: Any) -> dict[str, Any]:
    """Require every placed CPML b coefficient to be finite and nonzero."""

    rows: dict[str, Any] = {}
    passed = bool(objects.pml_objects)
    for pml in objects.pml_objects:
        values = {}
        for label in ("pml_b_E", "pml_b_H"):
            array = np.asarray(getattr(pml, label), dtype=np.float64)
            finite = bool(np.all(np.isfinite(array)))
            nonzero = bool(np.all(array != 0.0))
            passed = passed and finite and nonzero
            values[label] = {
                "minimum": float(np.min(array)),
                "maximum": float(np.max(array)),
                "finite": finite,
                "all_nonzero": nonzero,
                "maximum_single_step_inverse_gain": float(
                    np.max(1.0 / np.abs(array))
                ),
            }
        rows[pml.name] = values
    return {
        "schema": "fdtdx_4um_parity_CPML_inverse_coefficients_v1",
        "status": "PASS" if passed else "FAIL",
        "pml": rows,
        "algebraic_inverse": "psi_old=(psi_new-a*d_field)/b",
        "sliced_exact_resets_required": True,
    }


def _raw_directional_derivatives(field: Any, objects: Any, config: Any, *, curl_E: bool):
    import jax.numpy as jnp
    from fdtdx.core.physics.curl import _metric_scale
    from fdtdx.fdtd.update import pad_fields_for_boundaries

    padded = pad_fields_for_boundaries(field, objects, config)
    shape = tuple(int(value) - 2 for value in padded.shape[1:])
    stencil = "forward" if curl_E else "backward"
    dx = _metric_scale(config, axis=0, shape=shape, stencil=stencil)
    dy = _metric_scale(config, axis=1, shape=shape, stencil=stencil)
    dz = _metric_scale(config, axis=2, shape=shape, stencil=stencil)
    Fx, Fy, Fz = padded
    center = (slice(1, -1), slice(1, -1), slice(1, -1))
    if curl_E:
        dyFz = (Fz[1:-1, 2:, 1:-1] - Fz[center]) * dy
        dzFy = (Fy[1:-1, 1:-1, 2:] - Fy[center]) * dz
        dzFx = (Fx[1:-1, 1:-1, 2:] - Fx[center]) * dz
        dxFz = (Fz[2:, 1:-1, 1:-1] - Fz[center]) * dx
        dxFy = (Fy[2:, 1:-1, 1:-1] - Fy[center]) * dx
        dyFx = (Fx[1:-1, 2:, 1:-1] - Fx[center]) * dy
    else:
        dyFz = (Fz[center] - Fz[1:-1, :-2, 1:-1]) * dy
        dzFy = (Fy[center] - Fy[1:-1, 1:-1, :-2]) * dz
        dzFx = (Fx[center] - Fx[1:-1, 1:-1, :-2]) * dz
        dxFz = (Fz[center] - Fz[:-2, 1:-1, 1:-1]) * dx
        dxFy = (Fy[center] - Fy[:-2, 1:-1, 1:-1]) * dx
        dyFx = (Fx[center] - Fx[1:-1, :-2, 1:-1]) * dy
    return {
        0: (dxFz, dxFy),
        1: (dyFx, dyFz),
        2: (dzFy, dzFx),
    }


def reverse_cpml_auxiliary(
    *, fields: Any, psi_new: Any, objects: Any, config: Any, curl_E: bool
) -> Any:
    """Reverse every local CPML psi recurrence for an E- or H-curl update."""

    import jax.numpy as jnp

    derivatives = _raw_directional_derivatives(
        fields, objects, config, curl_E=curl_E
    )
    result = {}
    for pml in objects.pml_objects:
        d1_full, d2_full = derivatives[pml.axis]
        d1 = d1_full[pml.grid_slice]
        d2 = d2_full[pml.grid_slice]
        psi1_new, psi2_new = psi_new[pml.name]
        if curl_E:
            a, b = pml.pml_a_H, pml.pml_b_H
        else:
            a, b = pml.pml_a_E, pml.pml_b_E
        safe_b = jnp.where(b != 0, b, jnp.ones((), dtype=b.dtype))
        result[pml.name] = (
            (psi1_new - a * d1) / safe_b,
            (psi2_new - a * d2) / safe_b,
        )
    return result


def update_H_reverse_with_cpml(
    *, time_step: Any, arrays: Any, objects: Any, config: Any
) -> Any:
    """Reverse H and its CPML auxiliary state from one standard forward step."""

    from fdtdx.fdtd.update import update_H_reverse

    psi_previous = reverse_cpml_auxiliary(
        fields=arrays.fields.E,
        psi_new=arrays.fields.psi_H,
        objects=objects,
        config=config,
        curl_E=True,
    )
    output = update_H_reverse(
        time_step=time_step,
        arrays=arrays,
        objects=objects,
        config=config,
    )
    return output.aset("fields->psi_H", psi_previous)


def update_E_reverse_ADE_with_cpml(
    *, time_step: Any, arrays: Any, objects: Any, config: Any
) -> Any:
    """Reverse target ADE E and its CPML auxiliary state."""

    psi_previous = reverse_cpml_auxiliary(
        fields=arrays.fields.H,
        psi_new=arrays.fields.psi_E,
        objects=objects,
        config=config,
        curl_E=False,
    )
    output = update_E_reverse_diagonal_c4_free_ade(
        time_step=time_step,
        arrays=arrays,
        objects=objects,
        config=config,
    )
    return output.aset("fields->psi_E", psi_previous)
