from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_gradient_detectors import (
    filter_gradient_detectors,
)


def _scene_with_two_detectors():
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
    config = config.aset("time", 48 * float(config.time_step_duration))
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
    keep = fdtdx.PhasorDetector(
        name="keep",
        partial_grid_shape=(4, 4, 2),
        wave_characters=(fdtdx.WaveCharacter(wavelength=800.0e-9),),
        components=("Ex",),
        exact_interpolation=False,
        plot=False,
    )
    control = fdtdx.PhasorDetector(
        name="control",
        partial_grid_shape=(2, 2, 2),
        wave_characters=(fdtdx.WaveCharacter(wavelength=800.0e-9),),
        components=("Ey",),
        exact_interpolation=False,
        plot=False,
    )
    constraints = [
        slab.set_grid_coordinates(
            axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(0, 0, 2)
        ),
        source.set_grid_coordinates(
            axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(4, 4, 4)
        ),
        keep.set_grid_coordinates(
            axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(2, 2, 3)
        ),
        control.set_grid_coordinates(
            axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(3, 3, 3)
        ),
    ]
    key = jax.random.PRNGKey(20260825)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, slab, source, keep, control],
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


def test_filter_rejects_unknown_or_duplicate_detector_names() -> None:
    _, jax, objects, arrays, _, _ = _scene_with_two_detectors()
    with pytest.raises(ValueError, match="unknown"):
        filter_gradient_detectors(
            arrays, objects, keep_names=("missing",), jax_module=jax
        )
    with pytest.raises(ValueError, match="duplicates"):
        filter_gradient_detectors(
            arrays, objects, keep_names=("keep", "keep"), jax_module=jax
        )


def test_filter_removes_aligned_object_and_state_only() -> None:
    _, jax, objects, arrays, _, _ = _scene_with_two_detectors()
    filtered_arrays, filtered_objects, audit = filter_gradient_detectors(
        arrays, objects, keep_names=("keep",), jax_module=jax
    )
    assert [detector.name for detector in filtered_objects.detectors] == ["keep"]
    assert tuple(filtered_arrays.detector_states) == ("keep",)
    assert audit["status"] == "PASS"
    assert audit["removed_names"] == ["control"]
    assert audit["removed_detector_state_bytes"] > 0
    assert audit["non_detector_object_count_unchanged"] is True
    assert filtered_objects.volume.name == objects.volume.name == "volume"


def test_removed_control_detector_does_not_change_field_or_c3_gradient() -> None:
    fdtdx, jax, objects, arrays, config, key = _scene_with_two_detectors()
    filtered_arrays, filtered_objects, _ = filter_gradient_detectors(
        arrays, objects, keep_names=("keep",), jax_module=jax
    )
    c3 = arrays.dispersive_c3
    assert c3 is not None

    def loss(value, *, reduced):
        base = filtered_arrays if reduced else arrays
        simulation_objects = filtered_objects if reduced else objects
        _, output = fdtdx.run_fdtd(
            arrays=base.aset("dispersive_c3", value),
            objects=simulation_objects,
            config=config,
            key=key,
            show_progress=False,
        )
        return jax.numpy.mean(jax.numpy.square(output.fields.E))

    full_value, full_gradient = jax.value_and_grad(
        lambda value: loss(value, reduced=False)
    )(c3)
    reduced_value, reduced_gradient = jax.value_and_grad(
        lambda value: loss(value, reduced=True)
    )(c3)
    assert float(reduced_value) == pytest.approx(float(full_value), rel=0.0, abs=0.0)
    np.testing.assert_array_equal(
        np.asarray(reduced_gradient), np.asarray(full_gradient)
    )
