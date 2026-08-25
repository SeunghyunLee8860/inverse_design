"""Design-only compensated sparse sliced reversible FDTDX VJP."""

from __future__ import annotations

import math
from typing import Any, Iterable

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_reversible_cpml import (
    cpml_inverse_coefficient_audit,
    update_E_reverse_ADE_with_cpml,
    update_H_reverse_with_cpml,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_sparse_ade_checkpoint import (
    Region,
    normalize_disjoint_regions,
)


def reversible_ade_cpml_phasor_design_sliced_fdtd(
    *,
    arrays: Any,
    objects: Any,
    config: Any,
    key: Any,
    steps_per_slice: int,
    regions: Iterable[Region],
    design_region: Region,
    support_audit: dict[str, Any],
) -> tuple[Any, Any]:
    """Differentiate only design-region c3 with compensated accumulation."""

    import jax
    import jax.numpy as jnp
    from fdtdx.fdtd.container import ArrayContainer, FieldState
    from fdtdx.fdtd.forward import forward
    from fdtdx.objects.detectors.phasor import PhasorDetector

    if not isinstance(steps_per_slice, int) or steps_per_slice <= 0:
        raise ValueError("steps_per_slice must be a positive Python integer")
    cpml_audit = cpml_inverse_coefficient_audit(objects)
    if cpml_audit["status"] != "PASS":
        raise RuntimeError(f"CPML inverse coefficient audit failed: {cpml_audit}")
    unsupported_detectors = [
        detector.name
        for detector in objects.detectors
        if type(detector) is not PhasorDetector
    ]
    if unsupported_detectors:
        raise NotImplementedError(
            "Design sliced VJP only propagates additive PhasorDetector states; "
            f"unsupported={unsupported_detectors}"
        )
    detector_names = tuple(detector.name for detector in objects.detectors)
    if tuple(arrays.detector_states) != detector_names:
        raise ValueError(
            "Placed detector objects and states must have identical order: "
            f"objects={detector_names}, states={tuple(arrays.detector_states)}"
        )
    if arrays.recording_state is not None:
        raise NotImplementedError("Design sliced VJP does not use a boundary recorder")
    if arrays.electric_conductivity is not None or arrays.magnetic_conductivity is not None:
        raise NotImplementedError("Design sliced VJP requires conductivity=None")
    if arrays.dispersive_c4 is not None:
        raise NotImplementedError("Design sliced VJP requires c4=None")
    if arrays.inv_permittivities.shape[0] == 9:
        raise NotImplementedError("Design sliced VJP rejects full-tensor epsilon")
    if arrays.fields.dispersive_P_curr is None or arrays.dispersive_c3 is None:
        raise ValueError("Design sliced VJP requires dispersive ADE state and c3")
    if support_audit["status"] != "PASS":
        raise RuntimeError(f"Sparse ADE coefficient support failed: {support_audit}")

    base = arrays.reset()
    p_template = base.fields.dispersive_P_curr
    c3_template = base.dispersive_c3
    assert p_template is not None and c3_template is not None
    full_p_shape = tuple(int(value) for value in p_template.shape)
    spatial_shape = full_p_shape[-3:]
    normalized_regions = normalize_disjoint_regions(
        regions,
        spatial_shape=spatial_shape,
    )
    normalized_design_region = normalize_disjoint_regions(
        (design_region,),
        spatial_shape=spatial_shape,
    )[0]
    if normalized_design_region not in normalized_regions:
        raise ValueError("design_region must be one of the audited sparse regions")
    normalized_bounds = [
        [[part.start, part.stop] for part in region] for region in normalized_regions
    ]
    requested_support_regions = support_audit.get("regions")
    if requested_support_regions != normalized_bounds:
        raise RuntimeError(
            "Sparse ADE support audit regions do not match requested regions: "
            f"audit={requested_support_regions}, requested={normalized_bounds}"
        )

    p_prefix = (slice(None), slice(None))
    design_index = p_prefix + normalized_design_region
    total_steps = int(config.time_steps_total)
    num_slices = math.ceil(total_steps / steps_per_slice)
    initial_inv_permittivities = jax.lax.stop_gradient(
        base.initial_inv_permittivities
    )
    fixed_inv_eps = jax.lax.stop_gradient(base.inv_permittivities)
    fixed_inv_mu = jax.lax.stop_gradient(base.inv_permeabilities)
    fixed_c1 = jax.lax.stop_gradient(base.dispersive_c1)
    fixed_c2 = jax.lax.stop_gradient(base.dispersive_c2)
    fixed_c3 = jax.lax.stop_gradient(c3_template).at[design_index].set(0.0)
    design_c3_initial = c3_template[design_index]

    def assemble_c3(design_c3):
        return fixed_c3.at[design_index].set(design_c3)

    def extract(full_p):
        return tuple(full_p[p_prefix + region] for region in normalized_regions)

    def expand(regional_p):
        full_p = jnp.zeros(full_p_shape, dtype=p_template.dtype)
        for region, value in zip(normalized_regions, regional_p, strict=True):
            full_p = full_p.at[p_prefix + region].set(value)
        return full_p

    def state_tuple(container):
        return (
            container.fields.E,
            container.fields.H,
            container.fields.psi_E,
            container.fields.psi_H,
            container.fields.dispersive_P_curr,
            container.fields.dispersive_P_prev,
            container.detector_states,
        )

    def sparse_checkpoint(state):
        E, H, psi_E, psi_H, P_curr, P_prev, _ = state
        return E, H, psi_E, psi_H, extract(P_curr), extract(P_prev)

    def from_state_design(state, design_c3):
        E, H, psi_E, psi_H, P_curr, P_prev, detector_states = state
        return ArrayContainer(
            fields=FieldState(
                E=E,
                H=H,
                psi_E=psi_E,
                psi_H=psi_H,
                dispersive_P_curr=P_curr,
                dispersive_P_prev=P_prev,
            ),
            inv_permittivities=fixed_inv_eps,
            inv_permeabilities=fixed_inv_mu,
            detector_states=detector_states,
            recording_state=None,
            electric_conductivity=None,
            magnetic_conductivity=None,
            dispersive_c1=fixed_c1,
            dispersive_c2=fixed_c2,
            dispersive_c3=assemble_c3(design_c3),
            dispersive_c4=None,
            initial_inv_permittivities=initial_inv_permittivities,
        )

    def one_step(time_step, state, design_c3):
        _, output = forward(
            state=(time_step, from_state_design(state, design_c3)),
            config=config,
            objects=objects,
            key=key,
            record_detectors=True,
            record_boundaries=False,
            simulate_boundaries=True,
        )
        return state_tuple(output)

    def run_sliced_forward(initial_state, design_c3):
        slice_starts = jnp.arange(num_slices, dtype=jnp.int32) * steps_per_slice

        def slice_body(state, slice_start):
            checkpoint = sparse_checkpoint(state)

            def step_body(step_index, current_state):
                time_step = slice_start + step_index
                return jax.lax.cond(
                    time_step < total_steps,
                    lambda operand: one_step(*operand),
                    lambda operand: operand[1],
                    (time_step, current_state, design_c3),
                )

            final_state = jax.lax.fori_loop(
                0,
                steps_per_slice,
                step_body,
                state,
            )
            return final_state, checkpoint

        return jax.lax.scan(slice_body, initial_state, slice_starts)

    @jax.custom_vjp
    def primitive(initial_state, design_c3):
        final_state, _ = run_sliced_forward(initial_state, design_c3)
        return final_state

    def primitive_fwd(initial_state, design_c3):
        final_state, checkpoints = run_sliced_forward(initial_state, design_c3)
        return final_state, (final_state, checkpoints, design_c3)

    def primitive_bwd(residual, final_cotangent):
        final_state, checkpoints, design_c3 = residual
        zero_design_cotangent = jnp.zeros_like(design_c3)

        def reverse_slice(reverse_index, carry):
            (
                current_state,
                running_cotangent,
                design_cotangent,
                design_compensation,
            ) = carry
            slice_index = num_slices - 1 - reverse_index
            slice_start = slice_index * steps_per_slice
            slice_stop = jnp.minimum(slice_start + steps_per_slice, total_steps)
            active_steps = slice_stop - slice_start

            def reverse_step(step_index, step_carry):
                def active_branch(active_carry):
                    current, current_cotangent, accumulated, compensation = active_carry
                    previous_time = slice_stop - 1 - step_index
                    current_container = from_state_design(current, design_c3)
                    previous_container = update_H_reverse_with_cpml(
                        time_step=previous_time,
                        arrays=current_container,
                        objects=objects,
                        config=config,
                    )
                    previous_container = update_E_reverse_ADE_with_cpml(
                        time_step=previous_time,
                        arrays=previous_container,
                        objects=objects,
                        config=config,
                    )
                    previous_state = state_tuple(previous_container)
                    _, pullback = jax.vjp(
                        lambda step_state, step_design_c3: one_step(
                            previous_time, step_state, step_design_c3
                        ),
                        previous_state,
                        design_c3,
                    )
                    previous_cotangent, step_design_cotangent = pullback(
                        current_cotangent
                    )
                    adjusted = step_design_cotangent - compensation
                    updated = accumulated + adjusted
                    updated_compensation = (updated - accumulated) - adjusted
                    return (
                        previous_state,
                        previous_cotangent,
                        updated,
                        updated_compensation,
                    )

                return jax.lax.cond(
                    step_index < active_steps,
                    active_branch,
                    lambda inactive_carry: inactive_carry,
                    step_carry,
                )

            (
                current_state,
                running_cotangent,
                design_cotangent,
                design_compensation,
            ) = jax.lax.fori_loop(
                0,
                steps_per_slice,
                reverse_step,
                (
                    current_state,
                    running_cotangent,
                    design_cotangent,
                    design_compensation,
                ),
            )
            exact_checkpoint = jax.tree_util.tree_map(
                lambda values: values[slice_index], checkpoints
            )
            E, H, psi_E, psi_H, P_curr_regional, P_prev_regional = exact_checkpoint
            reset_state = (
                E,
                H,
                psi_E,
                psi_H,
                expand(P_curr_regional),
                expand(P_prev_regional),
                current_state[-1],
            )
            return (
                reset_state,
                running_cotangent,
                design_cotangent,
                design_compensation,
            )

        _, initial_cotangent, design_cotangent, _ = jax.lax.fori_loop(
            0,
            num_slices,
            reverse_slice,
            (
                final_state,
                final_cotangent,
                zero_design_cotangent,
                zero_design_cotangent,
            ),
        )
        return initial_cotangent, design_cotangent

    primitive.defvjp(primitive_fwd, primitive_bwd)
    final_state = primitive(state_tuple(base), design_c3_initial)
    output = from_state_design(final_state, design_c3_initial)
    return jnp.asarray(total_steps, dtype=jnp.int32), output
