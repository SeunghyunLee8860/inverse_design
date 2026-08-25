"""Parity-only checkpoint loop with immutable material arrays outside carry.

FDTDX's generic checkpointed entry point carries the complete ArrayContainer.
Equinox therefore stores every full-grid material coefficient in each saved
checkpoint even though those leaves do not change with time.  This wrapper
keeps only the time-varying FieldState and detector states in ``init_val`` and
captures material leaves in the differentiable body closure.

The wrapper is intentionally narrow: fixed-duration checkpointed simulations,
no recorder, and no reversible boundary recording.  It does not change a
single Maxwell update equation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def tree_array_bytes(tree: Any, *, jax_module: Any) -> int:
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax_module.tree_util.tree_leaves(tree)
        if hasattr(leaf, "dtype") and hasattr(leaf, "size")
    )


def checkpoint_carry_audit(arrays: Any, *, jax_module: Any) -> dict[str, Any]:
    fields = tree_array_bytes(arrays.fields, jax_module=jax_module)
    detectors = tree_array_bytes(arrays.detector_states, jax_module=jax_module)
    full = tree_array_bytes(arrays, jax_module=jax_module)
    dynamic = fields + detectors + np.dtype(np.int32).itemsize
    return {
        "schema": "fdtdx_4um_parity_dynamic_checkpoint_carry_v1",
        "status": "PASS" if 0 < dynamic < full else "FAIL",
        "full_ArrayContainer_bytes": full,
        "dynamic_FieldState_bytes": fields,
        "dynamic_detector_state_bytes": detectors,
        "dynamic_time_step_bytes": np.dtype(np.int32).itemsize,
        "dynamic_checkpoint_bytes": dynamic,
        "excluded_immutable_bytes": full - fields - detectors,
        "dynamic_over_full_fraction": dynamic / full,
        "material_arrays_remain_differentiable_closure_inputs": True,
        "maxwell_update_modified": False,
    }


def dynamic_checkpointed_fdtd(
    *,
    arrays: Any,
    objects: Any,
    config: Any,
    key: Any,
    record_detectors: bool = True,
) -> tuple[Any, Any]:
    """Run the standard FDTDX forward step with a dynamic-only loop carry."""

    import equinox.internal as eqxi
    import jax.numpy as jnp
    from fdtdx.fdtd.container import ArrayContainer
    from fdtdx.fdtd.forward import forward

    gradient = config.gradient_config
    if gradient is None or gradient.method != "checkpointed":
        raise ValueError("dynamic parity loop requires checkpointed GradientConfig")
    if gradient.num_checkpoints is None or gradient.num_checkpoints < 1:
        raise ValueError("dynamic parity loop requires a positive checkpoint count")
    if arrays.recording_state is not None:
        raise NotImplementedError("dynamic parity loop does not support Recorder state")
    if config.invertible_optimization:
        raise NotImplementedError(
            "dynamic parity loop never records reversible boundaries"
        )

    reset = arrays.reset()
    inv_permittivities = reset.inv_permittivities
    inv_permeabilities = reset.inv_permeabilities
    electric_conductivity = reset.electric_conductivity
    magnetic_conductivity = reset.magnetic_conductivity
    dispersive_c1 = reset.dispersive_c1
    dispersive_c2 = reset.dispersive_c2
    dispersive_c3 = reset.dispersive_c3
    dispersive_c4 = reset.dispersive_c4
    initial_inv_permittivities = reset.initial_inv_permittivities

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

    def body(dynamic_state):
        time_step, fields, detector_states = dynamic_state
        full_state = (time_step, assemble(fields, detector_states))
        next_time, output = forward(
            state=full_state,
            config=config,
            objects=objects,
            key=key,
            record_detectors=record_detectors,
            record_boundaries=False,
            simulate_boundaries=True,
        )
        return next_time, output.fields, output.detector_states

    dynamic_initial = (
        jnp.asarray(0, dtype=jnp.int32),
        reset.fields,
        reset.detector_states,
    )
    dynamic_final = eqxi.while_loop(
        max_steps=config.time_steps_total,
        cond_fun=lambda state: state[0] < config.time_steps_total,
        body_fun=body,
        init_val=dynamic_initial,
        kind="checkpointed",
        checkpoints=gradient.num_checkpoints,
    )
    final_time, final_fields, final_detector_states = dynamic_final
    return final_time, assemble(final_fields, final_detector_states)
