from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_reversible_ade_vjp import (
    reversible_ade_fdtd_no_pml_prototype,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.test_fdtdx_parity_reversible_ade_step import (
    _small_scene,
)


def test_multistep_custom_vjp_matches_direct_c3_gradient() -> None:
    fdtdx, jax, objects, arrays, config, key = _small_scene()
    config = config.aset("time", 24 * float(config.time_step_duration))
    c3 = arrays.dispersive_c3
    assert c3 is not None

    def direct_loss(value):
        state = (jax.numpy.asarray(0, dtype=jax.numpy.int32), arrays.aset("dispersive_c3", value))
        for _ in range(config.time_steps_total):
            state = fdtdx.fdtd.forward.forward(
                state=state,
                config=config,
                objects=objects,
                key=key,
                record_detectors=False,
                record_boundaries=False,
                simulate_boundaries=True,
            )
        return jax.numpy.mean(jax.numpy.square(state[1].fields.E))

    def reversible_loss(value):
        _, output = reversible_ade_fdtd_no_pml_prototype(
            arrays=arrays.aset("dispersive_c3", value),
            objects=objects,
            config=config,
            key=key,
        )
        return jax.numpy.mean(jax.numpy.square(output.fields.E))

    direct_value, direct_gradient = jax.value_and_grad(direct_loss)(c3)
    reversible_value, reversible_gradient = jax.value_and_grad(reversible_loss)(c3)
    assert float(reversible_value) == pytest.approx(
        float(direct_value), rel=1.0e-6, abs=1.0e-10
    )
    np.testing.assert_allclose(
        np.asarray(reversible_gradient),
        np.asarray(direct_gradient),
        rtol=2.0e-4,
        atol=2.0e-10,
    )
    assert np.count_nonzero(np.asarray(reversible_gradient)) > 0


def test_multistep_custom_vjp_rejects_detector_state() -> None:
    _, _, objects, arrays, config, key = _small_scene()
    arrays = arrays.aset("detector_states", {"forbidden": {"value": arrays.fields.E[0, :1, :1, :1]}})
    with pytest.raises(NotImplementedError, match="detector"):
        reversible_ade_fdtd_no_pml_prototype(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
        )
