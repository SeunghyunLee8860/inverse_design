from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.test_fdtdx_parity_reversible_ade_step import (
    _forward_states,
    _reverse_one,
)


def test_reconstructed_state_reproduces_one_step_c3_vjp() -> None:
    fdtdx, jax, objects, states, config = _forward_states(12)
    true_previous = states[11]
    reconstructed = _reverse_one(fdtdx, objects, config, states[12], 11)
    c3 = true_previous.dispersive_c3
    assert c3 is not None
    key = jax.random.PRNGKey(20260825)

    def loss(value, base):
        _, output = fdtdx.fdtd.forward.forward(
            state=(
                jax.numpy.asarray(11, dtype=jax.numpy.int32),
                base.aset("dispersive_c3", value),
            ),
            config=config,
            objects=objects,
            key=key,
            record_detectors=False,
            record_boundaries=False,
            simulate_boundaries=True,
        )
        return jax.numpy.mean(jax.numpy.square(output.fields.E))

    true_value, true_vjp = jax.value_and_grad(
        lambda value: loss(value, true_previous)
    )(c3)
    reconstructed_value, reconstructed_vjp = jax.value_and_grad(
        lambda value: loss(value, reconstructed)
    )(c3)
    assert float(reconstructed_value) == pytest.approx(
        float(true_value), rel=2.0e-5, abs=1.0e-10
    )
    np.testing.assert_allclose(
        np.asarray(reconstructed_vjp),
        np.asarray(true_vjp),
        rtol=5.0e-5,
        atol=1.0e-10,
    )
    assert np.count_nonzero(np.asarray(true_vjp)) > 0
