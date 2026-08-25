"""Sparse exact-block FDTDX VJP with checkpointed within-block recomputation."""

from __future__ import annotations

import math
from typing import Any, Iterable

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_sparse_ade_checkpoint import (
    Region,
    normalize_disjoint_regions,
    sparse_ade_checkpoint_carry_audit,
)


def blockwise_exact_sparse_checkpoint_audit(
    arrays: Any,
    *,
    regions: Iterable[Region],
    jax_module: Any,
    total_steps: int,
    steps_per_block: int,
    inner_checkpoints: int,
) -> dict[str, Any]:
    """Report exact outer/inner checkpoint payloads, excluding work buffers."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if steps_per_block <= 0:
        raise ValueError("steps_per_block must be positive")
    if not 1 <= inner_checkpoints <= steps_per_block:
        raise ValueError("inner_checkpoints must be in [1, steps_per_block]")
    carry = sparse_ade_checkpoint_carry_audit(
        arrays,
        regions=regions,
        jax_module=jax_module,
    )
    payload = int(carry["sparse_dynamic_checkpoint_bytes"])
    outer_checkpoints = math.ceil(total_steps / steps_per_block)
    outer_bytes = outer_checkpoints * payload
    inner_bytes = inner_checkpoints * payload
    return {
        "schema": "fdtdx_4um_blockwise_exact_sparse_checkpoint_v1",
        "status": carry["status"],
        "total_steps": int(total_steps),
        "steps_per_block": int(steps_per_block),
        "outer_checkpoints": outer_checkpoints,
        "inner_checkpoints": int(inner_checkpoints),
        "sparse_checkpoint_payload_bytes": payload,
        "outer_checkpoint_bytes": outer_bytes,
        "inner_checkpoint_bytes": inner_bytes,
        "outer_plus_inner_checkpoint_bytes": outer_bytes + inner_bytes,
        "excludes_transient_full_P_and_XLA_work_buffers": True,
        "algebraic_time_reversal_used": False,
        "carry_audit": carry,
    }


def blockwise_exact_sparse_ade_cpml_phasor_design_fdtd(
    *,
    arrays: Any,
    objects: Any,
    config: Any,
    key: Any,
    steps_per_block: int,
    inner_checkpoints: int,
    regions: Iterable[Region],
    design_region: Region,
    support_audit: dict[str, Any],
) -> tuple[Any, Any]:
    """Use exact sparse block starts and exact checkpointed block VJPs."""

    import equinox.internal as eqxi
    import jax
    import jax.numpy as jnp
    from fdtdx.fdtd.container import ArrayContainer, FieldState
    from fdtdx.fdtd.forward import forward
    from fdtdx.objects.detectors.phasor import PhasorDetector

    if not isinstance(steps_per_block, int) or steps_per_block <= 0:
        raise ValueError("steps_per_block must be a positive Python integer")
    if not isinstance(inner_checkpoints, int) or not (
        1 <= inner_checkpoints <= steps_per_block
    ):
        raise ValueError("inner_checkpoints must be in [1, steps_per_block]")
    unsupported_detectors = [
        detector.name
        for detector in objects.detectors
        if type(detector) is not PhasorDetector
    ]
    if unsupported_detectors:
        raise NotImplementedError(
            "Sparse exact-block VJP is certified only for PhasorDetector; "
            f"unsupported={unsupported_detectors}"
        )
    detector_names = tuple(detector.name for detector in objects.detectors)
    if tuple(arrays.detector_states) != detector_names:
        raise ValueError(
            "Placed detector objects and states must have identical order: "
            f"objects={detector_names}, states={tuple(arrays.detector_states)}"
        )
    if arrays.recording_state is not None:
        raise NotImplementedError("Sparse exact-block VJP has no boundary recorder")
    if arrays.electric_conductivity is not None or arrays.magnetic_conductivity is not None:
        raise NotImplementedError("Sparse exact-block VJP requires conductivity=None")
    if arrays.dispersive_c4 is not None:
        raise NotImplementedError("Sparse exact-block VJP requires c4=None")
    if arrays.inv_permittivities.shape[0] == 9:
        raise NotImplementedError("Sparse exact-block VJP rejects full-tensor epsilon")
    if arrays.fields.dispersive_P_curr is None or arrays.dispersive_c3 is None:
        raise ValueError("Sparse exact-block VJP requires dispersive ADE state and c3")
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
        [[part.start, part.stop] for part in region]
        for region in normalized_regions
    ]
    if support_audit.get("regions") != normalized_bounds:
        raise RuntimeError(
            "Sparse ADE support audit regions do not match requested regions: "
            f"audit={support_audit.get('regions')}, requested={normalized_bounds}"
        )

    p_prefix = (slice(None), slice(None))
    design_index = p_prefix + normalized_design_region
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

    def extract(full_p):
        return tuple(full_p[p_prefix + region] for region in normalized_regions)

    def expand(regional_p):
        full_p = jnp.zeros(full_p_shape, dtype=p_template.dtype)
        for region, value in zip(normalized_regions, regional_p, strict=True):
            full_p = full_p.at[p_prefix + region].set(value)
        return full_p

    def sparse_state(container):
        return (
            container.fields.E,
            container.fields.H,
            container.fields.psi_E,
            container.fields.psi_H,
            extract(container.fields.dispersive_P_curr),
            extract(container.fields.dispersive_P_prev),
            container.detector_states,
        )

    def from_sparse_state(state, design_c3):
        E, H, psi_E, psi_H, P_curr, P_prev, detector_states = state
        return ArrayContainer(
            fields=FieldState(
                E=E,
                H=H,
                psi_E=psi_E,
                psi_H=psi_H,
                dispersive_P_curr=expand(P_curr),
                dispersive_P_prev=expand(P_prev),
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
            state=(time_step, from_sparse_state(state, design_c3)),
            config=config,
            objects=objects,
            key=key,
            record_detectors=True,
            record_boundaries=False,
            simulate_boundaries=True,
        )
        return sparse_state(output)

    def run_block_lax(block_start, initial_state, design_c3):
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

    def run_block_checkpointed(block_start, initial_state, design_c3):
        active_steps = jnp.minimum(steps_per_block, total_steps - block_start)

        def body(loop_state):
            local_step, state = loop_state
            return local_step + 1, one_step(
                block_start + local_step,
                state,
                design_c3,
            )

        _, final_state = eqxi.while_loop(
            max_steps=steps_per_block,
            cond_fun=lambda loop_state: loop_state[0] < active_steps,
            body_fun=body,
            init_val=(jnp.asarray(0, dtype=jnp.int32), initial_state),
            kind="checkpointed",
            checkpoints=inner_checkpoints,
        )
        return final_state

    def run_blockwise_forward(initial_state, design_c3):
        block_starts = jnp.arange(num_blocks, dtype=jnp.int32) * steps_per_block

        def block_body(state, block_start):
            final_state = run_block_lax(block_start, state, design_c3)
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
                lambda block_state, block_design_c3: run_block_checkpointed(
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
    final_state = primitive(sparse_state(base), design_c3_initial)
    output = from_sparse_state(final_state, design_c3_initial)
    return jnp.asarray(total_steps, dtype=jnp.int32), output
