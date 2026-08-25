"""Sliced reversible ADE+CPML+phasor custom VJP prototype."""

from __future__ import annotations

import math
from typing import Any

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_reversible_cpml import (
    cpml_inverse_coefficient_audit,
    update_E_reverse_ADE_with_cpml,
    update_H_reverse_with_cpml,
)


def reversible_ade_cpml_phasor_sliced_fdtd_prototype(
    *,
    arrays: Any,
    objects: Any,
    config: Any,
    key: Any,
    steps_per_slice: int,
) -> tuple[Any, Any]:
    """Run a sliced custom VJP with exact primal resets between slices."""

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
            "Prototype only propagates additive PhasorDetector states; "
            f"unsupported={unsupported_detectors}"
        )
    detector_names = tuple(detector.name for detector in objects.detectors)
    if tuple(arrays.detector_states) != detector_names:
        raise ValueError(
            "Placed detector objects and states must have identical order: "
            f"objects={detector_names}, states={tuple(arrays.detector_states)}"
        )
    if arrays.recording_state is not None:
        raise NotImplementedError("Prototype does not use a boundary recorder")
    if arrays.electric_conductivity is not None or arrays.magnetic_conductivity is not None:
        raise NotImplementedError("Prototype requires conductivity=None")
    if arrays.dispersive_c4 is not None:
        raise NotImplementedError("Prototype requires c4=None")
    if arrays.inv_permittivities.shape[0] == 9:
        raise NotImplementedError("Prototype rejects full-tensor epsilon")
    if arrays.fields.dispersive_P_curr is None:
        raise ValueError("Prototype requires dispersive ADE state")

    base = arrays.reset()
    initial_inv_permittivities = base.initial_inv_permittivities
    total_steps = int(config.time_steps_total)
    num_slices = math.ceil(total_steps / steps_per_slice)

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

    def reversible_checkpoint(state):
        return state[:-1]

    def parameter_tuple(container):
        return (
            container.inv_permittivities,
            container.inv_permeabilities,
            container.dispersive_c1,
            container.dispersive_c2,
            container.dispersive_c3,
        )

    def from_state_parameters(state, parameters):
        E, H, psi_E, psi_H, P_curr, P_prev, detector_states = state
        inv_eps, inv_mu, c1, c2, c3 = parameters
        return ArrayContainer(
            fields=FieldState(
                E=E,
                H=H,
                psi_E=psi_E,
                psi_H=psi_H,
                dispersive_P_curr=P_curr,
                dispersive_P_prev=P_prev,
            ),
            inv_permittivities=inv_eps,
            inv_permeabilities=inv_mu,
            detector_states=detector_states,
            recording_state=None,
            electric_conductivity=None,
            magnetic_conductivity=None,
            dispersive_c1=c1,
            dispersive_c2=c2,
            dispersive_c3=c3,
            dispersive_c4=None,
            initial_inv_permittivities=initial_inv_permittivities,
        )

    def one_step(time_step, state, parameters):
        _, output = forward(
            state=(time_step, from_state_parameters(state, parameters)),
            config=config,
            objects=objects,
            key=key,
            record_detectors=True,
            record_boundaries=False,
            simulate_boundaries=True,
        )
        return state_tuple(output)

    def run_sliced_forward(initial_state, parameters):
        slice_starts = jnp.arange(num_slices, dtype=jnp.int32) * steps_per_slice

        def slice_body(state, slice_start):
            checkpoint = reversible_checkpoint(state)

            def step_body(step_index, current_state):
                time_step = slice_start + step_index
                return jax.lax.cond(
                    time_step < total_steps,
                    lambda operand: one_step(*operand),
                    lambda operand: operand[1],
                    (time_step, current_state, parameters),
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
    def primitive(initial_state, parameters):
        final_state, _ = run_sliced_forward(initial_state, parameters)
        return final_state

    def primitive_fwd(initial_state, parameters):
        final_state, checkpoints = run_sliced_forward(initial_state, parameters)
        return final_state, (final_state, checkpoints, parameters)

    def primitive_bwd(residual, final_cotangent):
        final_state, checkpoints, parameters = residual
        zero_parameter_cotangent = jax.tree_util.tree_map(jnp.zeros_like, parameters)

        def reverse_slice(reverse_index, carry):
            current_state, running_cotangent, parameter_cotangent = carry
            slice_index = num_slices - 1 - reverse_index
            slice_start = slice_index * steps_per_slice
            slice_stop = jnp.minimum(slice_start + steps_per_slice, total_steps)
            active_steps = slice_stop - slice_start

            def reverse_step(step_index, step_carry):
                def active_branch(active_carry):
                    current, current_cotangent, accumulated_parameter_cotangent = (
                        active_carry
                    )
                    previous_time = slice_stop - 1 - step_index
                    current_container = from_state_parameters(current, parameters)
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
                        lambda step_state, step_parameters: one_step(
                            previous_time, step_state, step_parameters
                        ),
                        previous_state,
                        parameters,
                    )
                    previous_cotangent, step_parameter_cotangent = pullback(
                        current_cotangent
                    )
                    accumulated_parameter_cotangent = jax.tree_util.tree_map(
                        jnp.add,
                        accumulated_parameter_cotangent,
                        step_parameter_cotangent,
                    )
                    return (
                        previous_state,
                        previous_cotangent,
                        accumulated_parameter_cotangent,
                    )

                return jax.lax.cond(
                    step_index < active_steps,
                    active_branch,
                    lambda inactive_carry: inactive_carry,
                    step_carry,
                )

            current_state, running_cotangent, parameter_cotangent = jax.lax.fori_loop(
                0,
                steps_per_slice,
                reverse_step,
                (current_state, running_cotangent, parameter_cotangent),
            )
            exact_checkpoint = jax.tree_util.tree_map(
                lambda values: values[slice_index], checkpoints
            )
            reset_state = (*exact_checkpoint, current_state[-1])
            return reset_state, running_cotangent, parameter_cotangent

        _, initial_cotangent, parameter_cotangent = jax.lax.fori_loop(
            0,
            num_slices,
            reverse_slice,
            (final_state, final_cotangent, zero_parameter_cotangent),
        )
        return initial_cotangent, parameter_cotangent

    primitive.defvjp(primitive_fwd, primitive_bwd)
    parameters = parameter_tuple(base)
    final_state = primitive(state_tuple(base), parameters)
    output = from_state_parameters(final_state, parameters)
    return jnp.asarray(total_steps, dtype=jnp.int32), output
