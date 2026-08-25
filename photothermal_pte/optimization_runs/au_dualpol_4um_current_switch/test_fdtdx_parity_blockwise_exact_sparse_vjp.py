from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_blockwise_exact_sparse_vjp import (
    blockwise_exact_sparse_ade_cpml_phasor_design_fdtd,
    blockwise_exact_sparse_checkpoint_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_sparse_ade_support import (
    sparse_ade_coefficient_support_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.test_fdtdx_parity_reversible_cpml_detector_vjp import (
    _small_cpml_phasor_scene,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.test_fdtdx_parity_reversible_sliced_long import (
    _long_cpml_phasor_scene,
)


@pytest.mark.parametrize(
    ("scene_builder", "steps_per_block", "inner_checkpoints"),
    (
        (_small_cpml_phasor_scene, 6, 3),
        (_long_cpml_phasor_scene, 16, 4),
    ),
)
def test_sparse_checkpointed_exact_blocks_match_direct_unrolled_ad(
    scene_builder,
    steps_per_block: int,
    inner_checkpoints: int,
) -> None:
    fdtdx, jax, objects, arrays, config, key = scene_builder()
    design_region = (slice(3, 9), slice(3, 9), slice(4, 8))
    regions = (design_region,)
    prefix = (slice(None), slice(None))
    c3 = arrays.dispersive_c3
    assert c3 is not None
    design_c3 = c3[prefix + design_region]
    support_audit = sparse_ade_coefficient_support_audit(
        arrays,
        regions=regions,
        jax_module=jax,
    )
    assert support_audit["status"] == "PASS"

    payload_audit = blockwise_exact_sparse_checkpoint_audit(
        arrays,
        regions=regions,
        jax_module=jax,
        total_steps=config.time_steps_total,
        steps_per_block=steps_per_block,
        inner_checkpoints=inner_checkpoints,
    )
    payload = payload_audit["sparse_checkpoint_payload_bytes"]
    assert payload_audit["outer_checkpoints"] == (
        config.time_steps_total + steps_per_block - 1
    ) // steps_per_block
    assert payload_audit["outer_checkpoint_bytes"] == (
        payload_audit["outer_checkpoints"] * payload
    )
    assert payload_audit["inner_checkpoint_bytes"] == inner_checkpoints * payload
    assert payload_audit["algebraic_time_reversal_used"] is False

    def arrays_for_design(value):
        return arrays.aset(
            "dispersive_c3",
            c3.at[prefix + design_region].set(value),
        )

    def detector_loss(output):
        phasor = output.detector_states["late_slab"]["phasor"]
        return jax.numpy.mean(jax.numpy.square(jax.numpy.abs(phasor)))

    def direct_loss(value):
        state = (
            jax.numpy.asarray(0, dtype=jax.numpy.int32),
            arrays_for_design(value),
        )
        for _ in range(config.time_steps_total):
            state = fdtdx.fdtd.forward.forward(
                state=state,
                config=config,
                objects=objects,
                key=key,
                record_detectors=True,
                record_boundaries=False,
                simulate_boundaries=True,
            )
        return detector_loss(state[1])

    def blockwise_loss(value):
        _, output = blockwise_exact_sparse_ade_cpml_phasor_design_fdtd(
            arrays=arrays_for_design(value),
            objects=objects,
            config=config,
            key=key,
            steps_per_block=steps_per_block,
            inner_checkpoints=inner_checkpoints,
            regions=regions,
            design_region=design_region,
            support_audit=support_audit,
        )
        return detector_loss(output)

    direct_value, direct_gradient = jax.value_and_grad(direct_loss)(design_c3)
    blockwise_value, blockwise_gradient = jax.value_and_grad(blockwise_loss)(
        design_c3
    )

    assert float(blockwise_value) == pytest.approx(
        float(direct_value),
        rel=1.0e-6,
        abs=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(blockwise_gradient),
        np.asarray(direct_gradient),
        rtol=2.0e-5,
        atol=3.0e-11,
    )
    assert np.count_nonzero(np.asarray(blockwise_gradient)) > 0
