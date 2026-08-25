"""Fail-closed coefficient-support audit for regional ADE checkpointing."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_sparse_ade_checkpoint import (
    Region,
    normalize_disjoint_regions,
)


def sparse_ade_coefficient_support_audit(
    arrays: Any,
    *,
    regions: Iterable[Region],
    jax_module: Any,
) -> dict[str, Any]:
    reference = arrays.dispersive_c3
    if reference is None:
        raise ValueError("support audit requires dispersive c3")
    spatial_shape = tuple(int(value) for value in reference.shape[-3:])
    normalized = normalize_disjoint_regions(regions, spatial_shape=spatial_shape)
    jnp = jax_module.numpy
    mask = jnp.zeros(spatial_shape, dtype=jnp.bool_)
    for region in normalized:
        mask = mask.at[region].set(True)

    outside_maxima: dict[str, float] = {}
    inside_nonzero: dict[str, int] = {}
    for name in ("dispersive_c1", "dispersive_c2", "dispersive_c3", "dispersive_c4"):
        value = getattr(arrays, name)
        if value is None:
            continue
        broadcast_mask = mask.reshape((1,) * (value.ndim - 3) + spatial_shape)
        outside = jnp.max(jnp.where(broadcast_mask, 0.0, jnp.abs(value)))
        count = jnp.count_nonzero(jnp.where(broadcast_mask, value, 0.0))
        outside, count = jax_module.block_until_ready((outside, count))
        outside_maxima[name] = float(np.asarray(outside))
        inside_nonzero[name] = int(np.asarray(count))

    checks = {
        "all_coefficients_exact_zero_outside_regions": all(
            value == 0.0 for value in outside_maxima.values()
        ),
        "c3_nonzero_inside_regions": inside_nonzero.get("dispersive_c3", 0) > 0,
    }
    return {
        "schema": "fdtdx_4um_parity_sparse_ADE_support_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "regions": [
            [[part.start, part.stop] for part in region] for region in normalized
        ],
        "maximum_abs_coefficient_outside_regions": outside_maxima,
        "nonzero_coefficient_count_inside_regions": inside_nonzero,
        "outside_dispersion_allowed": False,
    }
