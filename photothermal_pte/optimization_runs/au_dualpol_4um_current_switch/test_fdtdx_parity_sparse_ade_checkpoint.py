from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_sparse_ade_checkpoint import (
    normalize_disjoint_regions,
    sparse_ade_checkpoint_carry_audit,
    sparse_ade_checkpointed_fdtd,
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
    config = config.aset(
        "gradient_config",
        fdtdx.GradientConfig(method="checkpointed", num_checkpoints=4),
    )
    return fdtdx, jax, objects, arrays, config, key, objects["slab"].grid_slice


def test_regions_are_explicit_bounded_and_disjoint() -> None:
    regions = normalize_disjoint_regions(
        (
            (slice(0, 4), slice(0, 4), slice(0, 2)),
            (slice(4, 8), slice(4, 8), slice(2, 4)),
        ),
        spatial_shape=(8, 8, 8),
    )
    assert len(regions) == 2
    with pytest.raises(ValueError, match="must not overlap"):
        normalize_disjoint_regions(
            (
                (slice(0, 4), slice(0, 4), slice(0, 4)),
                (slice(3, 5), slice(3, 5), slice(3, 5)),
            ),
            spatial_shape=(8, 8, 8),
        )


def test_sparse_carry_audit_removes_zero_polarization_volume() -> None:
    _, jax, _, arrays, _, _, slab_slice = _small_dispersive_scene()
    audit = sparse_ade_checkpoint_carry_audit(
        arrays, regions=(slab_slice,), jax_module=jax
    )
    assert audit["status"] == "PASS"
    assert audit["removed_P_checkpoint_bytes"] > 0
    assert (
        audit["sparse_dynamic_checkpoint_bytes"]
        < audit["full_dynamic_checkpoint_bytes"]
    )
    assert audit["maxwell_ADE_update_modified"] is False


def test_sparse_loop_matches_generic_forward_and_regional_c3_gradient() -> None:
    fdtdx, jax, objects, arrays, config, key, slab_slice = _small_dispersive_scene()
    c3 = arrays.dispersive_c3
    assert c3 is not None
    regional_index = (slice(None), slice(None), *slab_slice)
    regional_c3 = c3[regional_index]

    def with_regional_c3(value):
        return arrays.aset("dispersive_c3", c3.at[regional_index].set(value))

    def loss_generic(value):
        _, output = fdtdx.run_fdtd(
            arrays=with_regional_c3(value),
            objects=objects,
            config=config,
            key=key,
            show_progress=False,
        )
        return jax.numpy.mean(jax.numpy.square(output.fields.E))

    def loss_sparse(value):
        _, output = sparse_ade_checkpointed_fdtd(
            arrays=with_regional_c3(value),
            objects=objects,
            config=config,
            key=key,
            regions=(slab_slice,),
            record_detectors=True,
        )
        return jax.numpy.mean(jax.numpy.square(output.fields.E))

    generic_value, generic_gradient = jax.value_and_grad(loss_generic)(regional_c3)
    sparse_value, sparse_gradient = jax.value_and_grad(loss_sparse)(regional_c3)
    assert float(generic_value) > 0.0
    assert float(sparse_value) == pytest.approx(float(generic_value), rel=1.0e-6)
    np.testing.assert_allclose(
        np.asarray(sparse_gradient),
        np.asarray(generic_gradient),
        rtol=2.0e-5,
        atol=1.0e-12,
    )
    assert np.all(np.isfinite(np.asarray(sparse_gradient)))
    assert np.count_nonzero(np.asarray(sparse_gradient)) > 0
