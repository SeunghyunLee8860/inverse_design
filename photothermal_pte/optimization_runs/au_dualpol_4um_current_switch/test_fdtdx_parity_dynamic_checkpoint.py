from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_dynamic_checkpoint import (
    checkpoint_carry_audit,
    dynamic_checkpointed_fdtd,
)


def _small_dispersive_scene():
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
    config = config.aset("time", 64 * float(config.time_step_duration))
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
        name="slab",
        partial_grid_shape=(8, 8, 4),
        material=material,
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
    config = config.aset(
        "gradient_config",
        fdtdx.GradientConfig(method="checkpointed", num_checkpoints=4),
    )
    return fdtdx, jax, objects, arrays, config, key


def test_dynamic_carry_excludes_immutable_material_arrays() -> None:
    _, jax, _, arrays, _, _ = _small_dispersive_scene()
    audit = checkpoint_carry_audit(arrays, jax_module=jax)
    assert audit["status"] == "PASS"
    assert audit["excluded_immutable_bytes"] > 0
    assert 0.0 < audit["dynamic_over_full_fraction"] < 1.0
    assert audit["material_arrays_remain_differentiable_closure_inputs"] is True
    assert audit["maxwell_update_modified"] is False


def test_dynamic_loop_matches_generic_forward_and_c3_gradient() -> None:
    fdtdx, jax, objects, arrays, config, key = _small_dispersive_scene()
    c3 = arrays.dispersive_c3
    assert c3 is not None

    def loss_generic(value):
        container = arrays.aset("dispersive_c3", value)
        _, output = fdtdx.run_fdtd(
            arrays=container,
            objects=objects,
            config=config,
            key=key,
            show_progress=False,
        )
        return jax.numpy.mean(jax.numpy.square(output.fields.E))

    def loss_dynamic(value):
        container = arrays.aset("dispersive_c3", value)
        _, output = dynamic_checkpointed_fdtd(
            arrays=container,
            objects=objects,
            config=config,
            key=key,
            record_detectors=True,
        )
        return jax.numpy.mean(jax.numpy.square(output.fields.E))

    generic_value, generic_gradient = jax.value_and_grad(loss_generic)(c3)
    dynamic_value, dynamic_gradient = jax.value_and_grad(loss_dynamic)(c3)
    assert float(generic_value) > 0.0
    assert float(dynamic_value) == pytest.approx(float(generic_value), rel=1.0e-6)
    np.testing.assert_allclose(
        np.asarray(dynamic_gradient),
        np.asarray(generic_gradient),
        rtol=2.0e-5,
        atol=1.0e-12,
    )
    assert np.all(np.isfinite(np.asarray(dynamic_gradient)))
    assert np.count_nonzero(np.asarray(dynamic_gradient)) > 0
