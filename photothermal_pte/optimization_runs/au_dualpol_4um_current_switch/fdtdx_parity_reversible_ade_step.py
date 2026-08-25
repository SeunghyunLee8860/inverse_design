"""Target-only inverse of one pinned FDTDX diagonal Lorentz-ADE E step.

This is a reconstruction primitive, not a custom VJP and not a production
runner.  It intentionally rejects conductivity, c4 coupling, and full-tensor
permittivity so unsupported physics cannot silently take this path.
"""

from __future__ import annotations

from typing import Any


def update_E_reverse_diagonal_c4_free_ade(
    *,
    time_step: Any,
    arrays: Any,
    objects: Any,
    config: Any,
) -> Any:
    """Reconstruct E_n, P_n, and P_(n-1) from the state after one E step."""

    import jax
    import jax.numpy as jnp
    from fdtdx.fdtd.update import (
        _source_uses_default_always_on_switch,
        apply_boundary_post_E_update,
        curl_H,
        pad_fields_for_boundaries,
    )

    if arrays.electric_conductivity is not None:
        raise NotImplementedError(
            "target reversible ADE primitive requires electric_conductivity=None"
        )
    if arrays.dispersive_c4 is not None:
        raise NotImplementedError("target reversible ADE primitive requires c4=None")
    if arrays.inv_permittivities.shape[0] == 9:
        raise NotImplementedError(
            "target reversible ADE primitive does not support full-tensor epsilon"
        )
    P_next = arrays.fields.dispersive_P_curr
    P_curr = arrays.fields.dispersive_P_prev
    c1 = arrays.dispersive_c1
    c2 = arrays.dispersive_c2
    c3 = arrays.dispersive_c3
    if any(value is None for value in (P_next, P_curr, c1, c2, c3)):
        raise ValueError("target reversible ADE primitive requires complete ADE state")

    # Undo the additive source exactly where the pinned non-dispersive reverse
    # update does so.  Boundary interfaces must already have been restored by a
    # sliced reversible driver when PMLs are present.
    E_next = arrays.fields.E
    for source in objects.sources:
        if _source_uses_default_always_on_switch(source):
            E_next = source.update_E(
                E_next,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                time_step=time_step,
                inverse=True,
            )
            continue

        def _remove_source():
            adjusted = source.adjust_time_step_by_on_off(time_step)
            return source.update_E(
                E_next,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                time_step=adjusted,
                inverse=True,
            )

        E_next = jax.lax.cond(
            source.is_on_at_time_step(time_step),
            _remove_source,
            lambda: E_next,
        )

    H_pad = pad_fields_for_boundaries(arrays.fields.H, objects, config)
    curl, _ = curl_H(
        config,
        H_pad,
        arrays.fields.psi_E,
        objects,
        False,
    )
    inv_eps = arrays.inv_permittivities
    E_prev = (
        E_next
        - config.courant_number * curl * inv_eps
        - inv_eps * jnp.sum(P_curr - P_next, axis=0)
    )

    numerator = P_next - c1 * P_curr - c3 * E_prev
    active = c2 != 0
    safe_c2 = jnp.where(active, c2, jnp.ones((), dtype=c2.dtype))
    P_prev = jnp.where(
        active,
        numerator / safe_c2,
        jnp.zeros((), dtype=numerator.dtype),
    )

    E_prev = apply_boundary_post_E_update(E_prev, objects)
    return (
        arrays
        .aset("fields->E", E_prev)
        .aset("fields->dispersive_P_curr", P_curr)
        .aset("fields->dispersive_P_prev", P_prev)
    )
