"""Exact-checkpoint blockwise FDTDX VJP without algebraic time reversal."""

from __future__ import annotations

import math
from typing import Any

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_sparse_ade_checkpoint import (
    Region,
    normalize_disjoint_regions,
)


def blockwise_exact_ade_cpml_phasor_design_fdtd(
    *,
    arrays: Any,
    objects: Any,
    config: Any,
    key: Any,
    steps_per_block: int,
    design_region: Region,
) -> tuple[Any, Any]:
    """Differentiate regional Au c3 from exact full-state block checkpoints.

    This is a correctness prototype, not a production-memory implementation.
    Its backward pass never reconstructs a previous E/H/ADE-P/CPML state.
    Instead, it restarts each block from the exact state saved by the forward
    pass and differentiates the ordinary pinned FDTDX forward block.
    """

    import jax
    import jax.numpy as jnp
    from fdtdx.fdtd.container import ArrayContainer, FieldState
    from fdtdx.fdtd.forward import forward
    from fdtdx.objects.detectors.phasor import PhasorDetector

    if not isinstance(steps_per_block, int) or steps_per_block <= 0:
        raise ValueError("steps_per_block must be a positive Python integer")
    unsupported_detectors = [
        detector.name
        for detector in objects.detectors
        if type(detector) is not PhasorDetector
    ]
    if unsupported_detectors:
        raise NotImplementedError(
            "Blockwise exact prototype is certified only for PhasorDetector; "
            f"unsupported={unsupported_detectors}"
        )
    detector_names = tuple(detector.name for detector in objects.detectors)
    if tuple(arrays.detector_states) != detector_names:
        raise ValueError(
            "Placed detector objects and states must have identical order: "
            f"objects={detector_names}, states={tuple(arrays.detector_states)}"
        )
    if arrays.recording_state is not None:
        raise NotImplementedError("Blockwise exact prototype has no boundary recorder")
    if arrays.electric_conductivity is not None or arrays.magnetic_conductivity is not None:
        raise NotImplementedError("Blockwise exact prototype requires conductivity=None")
    if arrays.dispersive_c4 is not None:
        raise NotImplementedError("Blockwise exact prototype requires c4=None")
    if arrays.inv_permittivities.shape[0] == 9:
        raise NotImplementedError("Blockwise exact prototype rejects full-tensor epsilon")
    if arrays.fields.dispersive_P_curr is None or arrays.dispersive_c3 is None:
        raise ValueError("Blockwise exact prototype requires dispersive ADE state and c3")

    base = arrays.reset()
    c3_template = base.dispersive_c3
    assert c3_template is not None
    spatial_shape = tuple(int(value) for value in c3_template.shape[-3:])
    normalized_design_region = normalize_disjoint_regions(
        (design_region,),
        spatial_shape=spatial_shape,
    )[0]
    design_index = (slice(None), slice(None)) + normalized_design_region
    total_steps = int(config.time_steps_total)
    num_blocks = math.ceil(total_steps / steps_per_block)

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

    def run_block(block_start, initial_state, design_c3):
        def step_body(step_index, current_state):
            time_step = block_start + step_index
            return jax.lax.cond(
                time_step < total_steps,
                lambda operand: one_step(*operand),
                lambda operand: operand[1],
                (time_step, current_state, design_c3),
            )

        return jax.lax.fori_loop(
            0,
            steps_per_block,
            step_body,
            initial_state,
        )

    def run_blockwise_forward(initial_state, design_c3):
        block_starts = jnp.arange(num_blocks, dtype=jnp.int32) * steps_per_block

        def block_body(state, block_start):
            final_state = run_block(block_start, state, design_c3)
            return final_state, state

        return jax.lax.scan(block_body, initial_state, block_starts)

    @jax.custom_vjp
    def primitive(initial_state, design_c3):
        final_state, _ = run_blockwise_forward(initial_state, design_c3)
        return final_state

    def primitive_fwd(initial_state, design_c3):
        final_state, exact_block_starts = run_blockwise_forward(
            initial_state,
            design_c3,
        )
        return final_state, (exact_block_starts, design_c3)

    def primitive_bwd(residual, final_cotangent):
        exact_block_starts, design_c3 = residual
        zero_design_cotangent = jnp.zeros_like(design_c3)

        def reverse_block(reverse_index, carry):
            running_cotangent, design_cotangent, design_compensation = carry
            block_index = num_blocks - 1 - reverse_index
            block_start = block_index * steps_per_block
            exact_start = jax.tree_util.tree_map(
                lambda values: values[block_index],
                exact_block_starts,
            )
            _, pullback = jax.vjp(
                lambda block_state, block_design_c3: run_block(
                    block_start,
                    block_state,
                    block_design_c3,
                ),
                exact_start,
                design_c3,
            )
            previous_cotangent, block_design_cotangent = pullback(
                running_cotangent
            )
            adjusted = block_design_cotangent - design_compensation
            updated = design_cotangent + adjusted
            updated_compensation = (updated - design_cotangent) - adjusted
            return previous_cotangent, updated, updated_compensation

        initial_cotangent, design_cotangent, _ = jax.lax.fori_loop(
            0,
            num_blocks,
            reverse_block,
            (
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
