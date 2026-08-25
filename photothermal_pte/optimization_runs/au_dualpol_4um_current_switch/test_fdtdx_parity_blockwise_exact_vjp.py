from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_blockwise_exact_vjp import (
    blockwise_exact_ade_cpml_phasor_design_fdtd,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.test_fdtdx_parity_reversible_cpml_detector_vjp import (
    _small_cpml_phasor_scene,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.test_fdtdx_parity_reversible_sliced_long import (
    _long_cpml_phasor_scene,
)


@pytest.mark.parametrize(
    ("scene_builder", "steps_per_block"),
    (
        (_small_cpml_phasor_scene, 6),
        (_long_cpml_phasor_scene, 16),
    ),
)
def test_blockwise_exact_design_vjp_matches_direct_unrolled_ad(
    scene_builder,
    steps_per_block: int,
) -> None:
    fdtdx, jax, objects, arrays, config, key = scene_builder()
    design_region = (slice(3, 9), slice(3, 9), slice(4, 8))
    prefix = (slice(None), slice(None))
    c3 = arrays.dispersive_c3
    assert c3 is not None
    design_c3 = c3[prefix + design_region]

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
        _, output = blockwise_exact_ade_cpml_phasor_design_fdtd(
            arrays=arrays_for_design(value),
            objects=objects,
            config=config,
            key=key,
            steps_per_block=steps_per_block,
            design_region=design_region,
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
