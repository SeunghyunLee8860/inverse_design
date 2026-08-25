from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_reversible_cpml_detector_vjp import (
    reversible_ade_cpml_phasor_fdtd_prototype,
)


def _small_cpml_phasor_scene():
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
    config = config.aset("time", 24 * float(config.time_step_duration))
    volume = fdtdx.SimulationVolume(name="volume", partial_grid_shape=(12, 12, 12))
    boundaries, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig(
            thickness_grid_minx=2,
            thickness_grid_maxx=2,
            thickness_grid_miny=2,
            thickness_grid_maxy=2,
            thickness_grid_minz=2,
            thickness_grid_maxz=2,
        ),
        volume,
    )
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
        name="slab", partial_grid_shape=(6, 6, 4), material=material
    )
    source = fdtdx.PointDipoleSource(
        name="dipole",
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=800.0e-9),
        polarization=0,
        amplitude=1.0,
    )
    detector = fdtdx.PhasorDetector(
        name="late_slab",
        partial_grid_shape=(4, 4, 2),
        wave_characters=(fdtdx.WaveCharacter(wavelength=800.0e-9),),
        components=("Ex", "Ey", "Ez"),
        switch=fdtdx.OnOffSwitch(
            start_time=8 * float(config.time_step_duration)
        ),
        exact_interpolation=False,
        plot=False,
    )
    constraints = list(boundary_constraints)
    constraints.extend(
        (
            slab.set_grid_coordinates(
                axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(3, 3, 4)
            ),
            source.set_grid_coordinates(
                axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(6, 6, 6)
            ),
            detector.set_grid_coordinates(
                axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(4, 4, 5)
            ),
        )
    )
    key = jax.random.PRNGKey(20260825)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, *boundaries.values(), slab, source, detector],
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
    return fdtdx, jax, objects, arrays, config, key


def test_ADE_CPML_phasor_custom_vjp_matches_direct_c3_gradient() -> None:
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

    def reversible_loss(value):
        _, output = reversible_ade_cpml_phasor_fdtd_prototype(
            arrays=arrays.aset("dispersive_c3", value),
            objects=objects,
            config=config,
            key=key,
        )
        return detector_loss(output)

    direct_value, direct_gradient = jax.value_and_grad(direct_loss)(c3)
    reversible_value, reversible_gradient = jax.value_and_grad(reversible_loss)(c3)
    assert float(reversible_value) == pytest.approx(
        float(direct_value), rel=1.0e-6, abs=1.0e-10
    )
    np.testing.assert_allclose(
        np.asarray(reversible_gradient),
        np.asarray(direct_gradient),
        rtol=1.0e-3,
        atol=3.0e-10,
    )
    assert np.count_nonzero(np.asarray(reversible_gradient)) > 0
