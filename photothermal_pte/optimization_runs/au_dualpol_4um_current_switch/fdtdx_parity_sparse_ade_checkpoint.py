"""Checkpoint only the exact regional support of the ADE polarization state.

The standard FDTDX update is retained verbatim.  Full-grid P-current and
P-previous arrays are reconstructed immediately before each standard forward
step and sliced back to certified dispersive regions immediately afterward.
Only the regional arrays enter the Equinox checkpoint value.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_dynamic_checkpoint import (
    tree_array_bytes,
)


Region = tuple[slice, slice, slice]


def normalize_disjoint_regions(
    regions: Iterable[Region], *, spatial_shape: tuple[int, int, int]
) -> tuple[Region, ...]:
    normalized: list[Region] = []
    for region in regions:
        if len(region) != 3:
            raise ValueError("each sparse ADE region must have three slices")
        parts = []
        for axis, (part, size) in enumerate(zip(region, spatial_shape, strict=True)):
            if part.step not in (None, 1):
                raise ValueError("sparse ADE slices must have unit step")
            if part.start is None or part.stop is None:
                raise ValueError("sparse ADE slices require explicit bounds")
            start, stop = int(part.start), int(part.stop)
            if not 0 <= start < stop <= size:
                raise ValueError(
                    f"invalid sparse ADE bounds on axis {axis}: {(start, stop)}"
                )
            parts.append(slice(start, stop))
        normalized.append(tuple(parts))
    if not normalized:
        raise ValueError("at least one sparse ADE region is required")

    def overlaps(first: Region, second: Region) -> bool:
        return all(
            max(a.start, b.start) < min(a.stop, b.stop)
            for a, b in zip(first, second, strict=True)
        )

    for index, first in enumerate(normalized):
        for second in normalized[index + 1 :]:
            if overlaps(first, second):
                raise ValueError("sparse ADE regions must not overlap")
    return tuple(normalized)


def region_shape(region: Region) -> tuple[int, int, int]:
    return tuple(int(part.stop) - int(part.start) for part in region)


def sparse_ade_checkpoint_carry_audit(
    arrays: Any,
    *,
    regions: Iterable[Region],
    jax_module: Any,
) -> dict[str, Any]:
    fields = arrays.fields
    if fields.dispersive_P_curr is None or fields.dispersive_P_prev is None:
        raise ValueError("sparse ADE carry requires dispersive P state")
    p_curr = fields.dispersive_P_curr
    spatial_shape = tuple(int(value) for value in p_curr.shape[-3:])
    normalized = normalize_disjoint_regions(regions, spatial_shape=spatial_shape)
    non_p_fields = tree_array_bytes(
        (fields.E, fields.H, fields.psi_E, fields.psi_H),
        jax_module=jax_module,
    )
    full_p_bytes = tree_array_bytes(
        (fields.dispersive_P_curr, fields.dispersive_P_prev),
        jax_module=jax_module,
    )
    leading = math.prod(int(value) for value in p_curr.shape[:-3])
    itemsize = int(p_curr.dtype.itemsize)
    sparse_p_bytes = (
        2
        * leading
        * sum(math.prod(region_shape(region)) for region in normalized)
        * itemsize
    )
    detector_bytes = tree_array_bytes(arrays.detector_states, jax_module=jax_module)
    sparse_dynamic = (
        non_p_fields + sparse_p_bytes + detector_bytes + np.dtype(np.int32).itemsize
    )
    full_dynamic = (
        non_p_fields + full_p_bytes + detector_bytes + np.dtype(np.int32).itemsize
    )
    return {
        "schema": "fdtdx_4um_parity_sparse_ADE_checkpoint_carry_v1",
        "status": "PASS" if 0 < sparse_p_bytes < full_p_bytes else "FAIL",
        "spatial_shape": list(spatial_shape),
        "leading_pole_component_shape": list(p_curr.shape[:-3]),
        "regions": [
            [[part.start, part.stop] for part in region] for region in normalized
        ],
        "region_shapes": [list(region_shape(region)) for region in normalized],
        "non_P_FieldState_bytes": non_p_fields,
        "full_domain_P_curr_prev_bytes": full_p_bytes,
        "sparse_regional_P_curr_prev_bytes": sparse_p_bytes,
        "removed_P_checkpoint_bytes": full_p_bytes - sparse_p_bytes,
        "detector_state_bytes": detector_bytes,
        "full_dynamic_checkpoint_bytes": full_dynamic,
        "sparse_dynamic_checkpoint_bytes": sparse_dynamic,
        "sparse_over_full_dynamic_fraction": sparse_dynamic / full_dynamic,
        "full_P_reconstructed_only_inside_step": True,
        "standard_fdtdx_forward_step_used": True,
        "maxwell_ADE_update_modified": False,
    }


def sparse_ade_checkpointed_fdtd(
    *,
    arrays: Any,
    objects: Any,
    config: Any,
    key: Any,
    regions: Iterable[Region],
    record_detectors: bool = True,
) -> tuple[Any, Any]:
    """Run standard FDTDX steps with only regional P in checkpoint state."""

    import equinox.internal as eqxi
    import jax.numpy as jnp
    from fdtdx.fdtd.container import ArrayContainer, FieldState
    from fdtdx.fdtd.forward import forward

    gradient = config.gradient_config
    if gradient is None or gradient.method != "checkpointed":
        raise ValueError("sparse ADE loop requires checkpointed GradientConfig")
    if gradient.num_checkpoints is None or gradient.num_checkpoints < 1:
        raise ValueError("sparse ADE loop requires a positive checkpoint count")
    if arrays.recording_state is not None:
        raise NotImplementedError("sparse ADE loop does not support Recorder state")
    if config.invertible_optimization:
        raise NotImplementedError("sparse ADE loop never records reversible boundaries")

    reset = arrays.reset()
    if reset.fields.dispersive_P_curr is None or reset.fields.dispersive_P_prev is None:
        raise ValueError("sparse ADE loop requires allocated dispersive P state")
    p_template = reset.fields.dispersive_P_curr
    full_p_shape = tuple(int(value) for value in p_template.shape)
    spatial_shape = full_p_shape[-3:]
    normalized = normalize_disjoint_regions(regions, spatial_shape=spatial_shape)
    p_dtype = p_template.dtype
    p_prefix = (slice(None), slice(None))

    inv_permittivities = reset.inv_permittivities
    inv_permeabilities = reset.inv_permeabilities
    electric_conductivity = reset.electric_conductivity
    magnetic_conductivity = reset.magnetic_conductivity
    dispersive_c1 = reset.dispersive_c1
    dispersive_c2 = reset.dispersive_c2
    dispersive_c3 = reset.dispersive_c3
    dispersive_c4 = reset.dispersive_c4
    initial_inv_permittivities = reset.initial_inv_permittivities

    def extract(full_p):
        return tuple(full_p[p_prefix + region] for region in normalized)

    def expand(regional_p):
        full_p = jnp.zeros(full_p_shape, dtype=p_dtype)
        for region, value in zip(normalized, regional_p, strict=True):
            full_p = full_p.at[p_prefix + region].set(value)
        return full_p

    def assemble(fields, detector_states):
        return ArrayContainer(
            fields=fields,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            detector_states=detector_states,
            recording_state=None,
            electric_conductivity=electric_conductivity,
            magnetic_conductivity=magnetic_conductivity,
            dispersive_c1=dispersive_c1,
            dispersive_c2=dispersive_c2,
            dispersive_c3=dispersive_c3,
            dispersive_c4=dispersive_c4,
            initial_inv_permittivities=initial_inv_permittivities,
        )

    def body(sparse_state):
        (
            time_step,
            E,
            H,
            psi_E,
            psi_H,
            p_curr_regional,
            p_prev_regional,
            detector_states,
        ) = sparse_state
        fields = FieldState(
            E=E,
            H=H,
            psi_E=psi_E,
            psi_H=psi_H,
            dispersive_P_curr=expand(p_curr_regional),
            dispersive_P_prev=expand(p_prev_regional),
        )
        next_time, output = forward(
            state=(time_step, assemble(fields, detector_states)),
            config=config,
            objects=objects,
            key=key,
            record_detectors=record_detectors,
            record_boundaries=False,
            simulate_boundaries=True,
        )
        return (
            next_time,
            output.fields.E,
            output.fields.H,
            output.fields.psi_E,
            output.fields.psi_H,
            extract(output.fields.dispersive_P_curr),
            extract(output.fields.dispersive_P_prev),
            output.detector_states,
        )

    initial = (
        jnp.asarray(0, dtype=jnp.int32),
        reset.fields.E,
        reset.fields.H,
        reset.fields.psi_E,
        reset.fields.psi_H,
        extract(reset.fields.dispersive_P_curr),
        extract(reset.fields.dispersive_P_prev),
        reset.detector_states,
    )
    final = eqxi.while_loop(
        max_steps=config.time_steps_total,
        cond_fun=lambda state: state[0] < config.time_steps_total,
        body_fun=body,
        init_val=initial,
        kind="checkpointed",
        checkpoints=gradient.num_checkpoints,
    )
    (
        final_time,
        final_E,
        final_H,
        final_psi_E,
        final_psi_H,
        final_p_curr,
        final_p_prev,
        final_detector_states,
    ) = final
    final_fields = FieldState(
        E=final_E,
        H=final_H,
        psi_E=final_psi_E,
        psi_H=final_psi_H,
        dispersive_P_curr=expand(final_p_curr),
        dispersive_P_prev=expand(final_p_prev),
    )
    return final_time, assemble(final_fields, final_detector_states)
