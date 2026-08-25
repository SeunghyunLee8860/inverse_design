"""Offline two-level exact sparse FDTDX VJP using direct segment VJPs."""

from __future__ import annotations

import math
from typing import Any, Iterable

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_sparse_ade_checkpoint import (
    Region,
    normalize_disjoint_regions,
    sparse_ade_checkpoint_carry_audit,
)


def two_level_exact_sparse_checkpoint_audit(
    arrays: Any,
    *,
    regions: Iterable[Region],
    jax_module: Any,
    total_steps: int,
    outer_block_steps: int,
    segment_steps: int,
) -> dict[str, Any]:
    """Report exact outer/inner start payloads before direct-VJP residuals."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if outer_block_steps <= 0:
        raise ValueError("outer_block_steps must be positive")
    if segment_steps <= 0 or outer_block_steps % segment_steps:
        raise ValueError("segment_steps must divide outer_block_steps")
    carry = sparse_ade_checkpoint_carry_audit(
        arrays,
        regions=regions,
        jax_module=jax_module,
    )
    payload = int(carry["sparse_dynamic_checkpoint_bytes"])
    outer_checkpoints = math.ceil(total_steps / outer_block_steps)
    inner_checkpoints = outer_block_steps // segment_steps
    outer_bytes = outer_checkpoints * payload
    inner_bytes = inner_checkpoints * payload
    return {
        "schema": "fdtdx_4um_two_level_exact_sparse_checkpoint_v1",
        "status": carry["status"],
        "total_steps": int(total_steps),
        "outer_block_steps": int(outer_block_steps),
        "segment_steps": int(segment_steps),
        "outer_checkpoints": outer_checkpoints,
        "inner_checkpoints_reused_per_outer_block": inner_checkpoints,
        "sparse_checkpoint_payload_bytes": payload,
        "outer_checkpoint_bytes": outer_bytes,
        "reused_inner_checkpoint_bytes": inner_bytes,
        "outer_plus_reused_inner_checkpoint_bytes": outer_bytes + inner_bytes,
        "excludes_direct_segment_VJP_residuals_and_XLA_work_buffers": True,
        "algebraic_time_reversal_used": False,
        "online_checkpointed_loop_used": False,
        "carry_audit": carry,
    }


def two_level_exact_sparse_ade_cpml_phasor_design_fdtd(
    *,
    arrays: Any,
    objects: Any,
    config: Any,
    key: Any,
    outer_block_steps: int,
    segment_steps: int,
    regions: Iterable[Region],
    design_region: Region,
    support_audit: dict[str, Any],
) -> tuple[Any, Any]:
    """Differentiate exact sparse blocks using short direct segment VJPs."""

    import jax
    import jax.numpy as jnp
    from fdtdx.fdtd.container import ArrayContainer, FieldState
    from fdtdx.fdtd.forward import forward
    from fdtdx.objects.detectors.phasor import PhasorDetector

    if not isinstance(outer_block_steps, int) or outer_block_steps <= 0:
        raise ValueError("outer_block_steps must be a positive Python integer")
    if (
        not isinstance(segment_steps, int)
        or segment_steps <= 0
        or outer_block_steps % segment_steps
    ):
        raise ValueError("segment_steps must divide outer_block_steps")
    unsupported_detectors = [
        detector.name
        for detector in objects.detectors
        if type(detector) is not PhasorDetector
    ]
    if unsupported_detectors:
        raise NotImplementedError(
            "Two-level exact VJP is certified only for PhasorDetector; "
            f"unsupported={unsupported_detectors}"
        )
    detector_names = tuple(detector.name for detector in objects.detectors)
    if tuple(arrays.detector_states) != detector_names:
        raise ValueError(
            "Placed detector objects and states must have identical order: "
            f"objects={detector_names}, states={tuple(arrays.detector_states)}"
        )
    if arrays.recording_state is not None:
        raise NotImplementedError("Two-level exact VJP has no boundary recorder")
    if arrays.electric_conductivity is not None or arrays.magnetic_conductivity is not None:
        raise NotImplementedError("Two-level exact VJP requires conductivity=None")
    if arrays.dispersive_c4 is not None:
        raise NotImplementedError("Two-level exact VJP requires c4=None")
    if arrays.inv_permittivities.shape[0] == 9:
        raise NotImplementedError("Two-level exact VJP rejects full-tensor epsilon")
    if arrays.fields.dispersive_P_curr is None or arrays.dispersive_c3 is None:
        raise ValueError("Two-level exact VJP requires dispersive ADE state and c3")
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
    num_outer_blocks = math.ceil(total_steps / outer_block_steps)
    segments_per_outer = outer_block_steps // segment_steps
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

    def run_segment(segment_start, initial_state, design_c3):
        def step_body(step_index, current_state):
            time_step = segment_start + step_index
            return jax.lax.cond(
                time_step < total_steps,
                lambda operand: one_step(*operand),
                lambda operand: operand[1],
                (time_step, current_state, design_c3),
            )

        return jax.lax.fori_loop(
            0,
            segment_steps,
            step_body,
            initial_state,
        )

    def run_outer(outer_start, initial_state, design_c3):
        segment_starts = (
            outer_start
            + jnp.arange(segments_per_outer, dtype=jnp.int32) * segment_steps
        )

        def segment_body(state, segment_start):
            final_state = run_segment(segment_start, state, design_c3)
            return final_state, state

        return jax.lax.scan(segment_body, initial_state, segment_starts)

    def run_two_level_forward(initial_state, design_c3):
        outer_starts = (
            jnp.arange(num_outer_blocks, dtype=jnp.int32) * outer_block_steps
        )

        def outer_body(state, outer_start):
            final_state, _ = run_outer(outer_start, state, design_c3)
            return final_state, state

        return jax.lax.scan(outer_body, initial_state, outer_starts)

    @jax.custom_vjp
    def primitive(initial_state, design_c3):
        final_state, _ = run_two_level_forward(initial_state, design_c3)
        return final_state

    def primitive_fwd(initial_state, design_c3):
        final_state, exact_outer_starts = run_two_level_forward(
            initial_state,
            design_c3,
        )
        return final_state, (exact_outer_starts, design_c3)

    def primitive_bwd(residual, final_cotangent):
        exact_outer_starts, design_c3 = residual
        zero_design_cotangent = jnp.zeros_like(design_c3)

        def reverse_outer(reverse_outer_index, carry):
            running_cotangent, design_cotangent, design_compensation = carry
            outer_index = num_outer_blocks - 1 - reverse_outer_index
            outer_start = outer_index * outer_block_steps
            exact_outer_start = jax.tree_util.tree_map(
                lambda values: values[outer_index],
                exact_outer_starts,
            )
            _, exact_segment_starts = run_outer(
                outer_start,
                exact_outer_start,
                design_c3,
            )
            exact_segment_starts = jax.tree_util.tree_map(
                jax.lax.stop_gradient,
                exact_segment_starts,
            )

            def reverse_segment(reverse_segment_index, segment_carry):
                current_cotangent, accumulated, compensation = segment_carry
                segment_index = segments_per_outer - 1 - reverse_segment_index
                segment_start = outer_start + segment_index * segment_steps
                exact_segment_start = jax.tree_util.tree_map(
                    lambda values: values[segment_index],
                    exact_segment_starts,
                )
                _, pullback = jax.vjp(
                    lambda segment_state, segment_design_c3: run_segment(
                        segment_start,
                        segment_state,
                        segment_design_c3,
                    ),
                    exact_segment_start,
                    design_c3,
                )
                previous_cotangent, segment_design_cotangent = pullback(
                    current_cotangent
                )
                adjusted = segment_design_cotangent - compensation
                updated = accumulated + adjusted
                updated_compensation = (updated - accumulated) - adjusted
                return previous_cotangent, updated, updated_compensation

            return jax.lax.fori_loop(
                0,
                segments_per_outer,
                reverse_segment,
                (
                    running_cotangent,
                    design_cotangent,
                    design_compensation,
                ),
            )

        initial_cotangent, design_cotangent, _ = jax.lax.fori_loop(
            0,
            num_outer_blocks,
            reverse_outer,
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
