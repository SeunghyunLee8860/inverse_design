from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_reversible_sliced_vjp import (
    reversible_ade_cpml_phasor_sliced_fdtd_prototype,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.test_fdtdx_parity_reversible_cpml_detector_vjp import (
    _small_cpml_phasor_scene,
)


def test_sliced_ADE_CPML_phasor_vjp_matches_direct_c3_gradient() -> None:
    fdtdx, jax, objects, arrays, config, key = _small_cpml_phasor_scene()
    c3 = arrays.dispersive_c3
    assert c3 is not None

    def detector_loss(output):
        phasor = output.detector_states["late_slab"]["phasor"]
        return jax.numpy.mean(jax.numpy.square(jax.numpy.abs(phasor)))

    def direct_loss(value):
        state = (
            jax.numpy.asarray(0, dtype=jax.numpy.int32),
            arrays.aset("dispersive_c3", value),
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

    def sliced_loss(value):
        _, output = reversible_ade_cpml_phasor_sliced_fdtd_prototype(
            arrays=arrays.aset("dispersive_c3", value),
            objects=objects,
            config=config,
            key=key,
            steps_per_slice=6,
        )
        return detector_loss(output)

    direct_value, direct_gradient = jax.value_and_grad(direct_loss)(c3)
    sliced_value, sliced_gradient = jax.value_and_grad(sliced_loss)(c3)
    assert float(sliced_value) == pytest.approx(
        float(direct_value), rel=1.0e-6, abs=1.0e-10
    )
    np.testing.assert_allclose(
        np.asarray(sliced_gradient),
        np.asarray(direct_gradient),
        rtol=5.0e-4,
        atol=3.0e-10,
    )
    assert np.count_nonzero(np.asarray(sliced_gradient)) > 0
