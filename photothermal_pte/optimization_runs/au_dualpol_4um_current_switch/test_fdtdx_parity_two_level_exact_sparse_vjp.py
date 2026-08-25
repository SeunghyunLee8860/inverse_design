from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_sparse_ade_support import (
    sparse_ade_coefficient_support_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_two_level_exact_sparse_vjp import (
    two_level_exact_sparse_ade_cpml_phasor_design_fdtd,
    two_level_exact_sparse_checkpoint_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.test_fdtdx_parity_reversible_cpml_detector_vjp import (
    _small_cpml_phasor_scene,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.test_fdtdx_parity_reversible_sliced_long import (
    _long_cpml_phasor_scene,
)


@pytest.mark.parametrize(
    ("scene_builder", "outer_block_steps", "segment_steps"),
    (
        (_small_cpml_phasor_scene, 12, 3),
        (_long_cpml_phasor_scene, 32, 8),
    ),
)
def test_two_level_exact_sparse_vjp_matches_direct_unrolled_ad(
    scene_builder,
    outer_block_steps: int,
    segment_steps: int,
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

    payload_audit = two_level_exact_sparse_checkpoint_audit(
        arrays,
        regions=regions,
        jax_module=jax,
        total_steps=config.time_steps_total,
        outer_block_steps=outer_block_steps,
        segment_steps=segment_steps,
    )
    payload = payload_audit["sparse_checkpoint_payload_bytes"]
    assert payload_audit["outer_checkpoints"] == (
        config.time_steps_total + outer_block_steps - 1
    ) // outer_block_steps
    assert payload_audit["inner_checkpoints_reused_per_outer_block"] == (
        outer_block_steps // segment_steps
    )
    assert payload_audit["outer_checkpoint_bytes"] == (
        payload_audit["outer_checkpoints"] * payload
    )
    assert payload_audit["reused_inner_checkpoint_bytes"] == (
        payload_audit["inner_checkpoints_reused_per_outer_block"] * payload
    )
    assert payload_audit["algebraic_time_reversal_used"] is False
    assert payload_audit["online_checkpointed_loop_used"] is False

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

    def two_level_loss(value):
        _, output = two_level_exact_sparse_ade_cpml_phasor_design_fdtd(
            arrays=arrays_for_design(value),
            objects=objects,
            config=config,
            key=key,
            outer_block_steps=outer_block_steps,
            segment_steps=segment_steps,
            regions=regions,
            design_region=design_region,
            support_audit=support_audit,
        )
        return detector_loss(output)

    direct_value, direct_gradient = jax.value_and_grad(direct_loss)(design_c3)
    two_level_value, two_level_gradient = jax.value_and_grad(two_level_loss)(
        design_c3
    )

    assert float(two_level_value) == pytest.approx(
        float(direct_value),
        rel=1.0e-6,
        abs=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(two_level_gradient),
        np.asarray(direct_gradient),
        rtol=2.0e-5,
        atol=3.0e-11,
    )
    assert np.count_nonzero(np.asarray(two_level_gradient)) > 0
