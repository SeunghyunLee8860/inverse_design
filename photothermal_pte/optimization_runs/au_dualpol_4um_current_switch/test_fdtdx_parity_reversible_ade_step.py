from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_reversible_ade_step import (
    update_E_reverse_diagonal_c4_free_ade,
)


def _small_scene():
    import fdtdx
    import jax
    import jax.numpy as jnp

    config = fdtdx.SimulationConfig(
        time=1.0e-14,
        grid=fdtdx.UniformGrid(spacing=50.0e-9),
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.5,
        gradient_config=None,
    )
    config = config.aset("time", 32 * float(config.time_step_duration))
    volume = fdtdx.SimulationVolume(name="volume", partial_grid_shape=(8, 8, 8))
    material = fdtdx.Material(
        permittivity=2.0,
        dispersion=fdtdx.DispersionModel(
            poles=(
                fdtdx.LorentzPole(
                    resonance_frequency=2.0e15,
                    damping=1.0e13,
                    delta_epsilon=1.5,
                ),
            )
        ),
    )
    slab = fdtdx.UniformMaterialObject(
        name="slab", partial_grid_shape=(8, 8, 4), material=material
    )
    source = fdtdx.PointDipoleSource(
        name="dipole",
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=800.0e-9),
        polarization=0,
        amplitude=1.0,
    )
    constraints = [
        slab.set_grid_coordinates(
            axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(0, 0, 2)
        ),
        source.set_grid_coordinates(
            axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(4, 4, 4)
        ),
    ]
    key = jax.random.PRNGKey(20260825)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, slab, source],
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
    return fdtdx, jax, objects, arrays, config, key


def _forward_states(num_steps: int):
    fdtdx, jax, objects, arrays, config, key = _small_scene()
    states = [arrays]
    state = (jax.numpy.asarray(0, dtype=jax.numpy.int32), arrays)
    for _ in range(num_steps):
        state = fdtdx.fdtd.forward.forward(
            state=state,
            config=config,
            objects=objects,
            key=key,
            record_detectors=False,
            record_boundaries=False,
            simulate_boundaries=True,
        )
        states.append(state[1])
    return fdtdx, jax, objects, states, config


def _reverse_one(fdtdx, objects, config, arrays, time_step):
    arrays = fdtdx.fdtd.update.update_H_reverse(
        time_step=time_step,
        arrays=arrays,
        objects=objects,
        config=config,
    )
    return update_E_reverse_diagonal_c4_free_ade(
        time_step=time_step,
        arrays=arrays,
        objects=objects,
        config=config,
    )


def _assert_dynamic_close(observed, expected, *, rtol, atol):
    np.testing.assert_allclose(
        np.asarray(observed.fields.E), np.asarray(expected.fields.E), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        np.asarray(observed.fields.H), np.asarray(expected.fields.H), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        np.asarray(observed.fields.dispersive_P_curr),
        np.asarray(expected.fields.dispersive_P_curr),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(observed.fields.dispersive_P_prev),
        np.asarray(expected.fields.dispersive_P_prev),
        rtol=rtol,
        atol=atol,
    )


def test_actual_fdtdx_one_step_ADE_reverse_reconstructs_previous_state() -> None:
    fdtdx, _, objects, states, config = _forward_states(12)
    reconstructed = _reverse_one(fdtdx, objects, config, states[12], 11)
    _assert_dynamic_close(reconstructed, states[11], rtol=3.0e-5, atol=2.0e-8)


def test_actual_fdtdx_eight_step_ADE_reverse_round_trip() -> None:
    fdtdx, _, objects, states, config = _forward_states(8)
    reconstructed = states[-1]
    for time_step in reversed(range(8)):
        reconstructed = _reverse_one(
            fdtdx, objects, config, reconstructed, time_step
        )
    _assert_dynamic_close(reconstructed, states[0], rtol=2.0e-4, atol=5.0e-8)
