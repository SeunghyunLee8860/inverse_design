from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_reversible_cpml import (
    cpml_inverse_coefficient_audit,
    exact_pml_interface_recorder_audit,
    update_E_reverse_ADE_with_cpml,
    update_H_reverse_with_cpml,
)


def _small_cpml_scene():
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
    constraints = list(boundary_constraints)
    constraints.extend(
        (
            slab.set_grid_coordinates(
                axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(3, 3, 4)
            ),
            source.set_grid_coordinates(
                axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(6, 6, 6)
            ),
        )
    )
    key = jax.random.PRNGKey(20260825)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, *boundaries.values(), slab, source],
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
    return fdtdx, jax, objects, arrays, config, key


def _states(num_steps):
    fdtdx, jax, objects, arrays, config, key = _small_cpml_scene()
    result = [arrays]
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
        result.append(state[1])
    return fdtdx, objects, result, config


def _reverse_one(objects, arrays, config, time_step):
    arrays = update_H_reverse_with_cpml(
        time_step=time_step, arrays=arrays, objects=objects, config=config
    )
    return update_E_reverse_ADE_with_cpml(
        time_step=time_step, arrays=arrays, objects=objects, config=config
    )


def _assert_close(observed, expected, *, rtol, atol):
    for name in ("E", "H", "dispersive_P_curr", "dispersive_P_prev"):
        np.testing.assert_allclose(
            np.asarray(getattr(observed.fields, name)),
            np.asarray(getattr(expected.fields, name)),
            rtol=rtol,
            atol=atol,
        )
    for psi_name in ("psi_E", "psi_H"):
        observed_psi = getattr(observed.fields, psi_name)
        expected_psi = getattr(expected.fields, psi_name)
        assert set(observed_psi) == set(expected_psi)
        for pml_name in observed_psi:
            for observed_value, expected_value in zip(
                observed_psi[pml_name], expected_psi[pml_name], strict=True
            ):
                np.testing.assert_allclose(
                    np.asarray(observed_value),
                    np.asarray(expected_value),
                    rtol=rtol,
                    atol=atol,
                )


def test_exact_grid_interface_recorder_is_terabyte_scale() -> None:
    audit = exact_pml_interface_recorder_audit(
        grid_shape=(186, 186, 286), time_steps=256_163
    )
    assert audit["bytes_per_step"] == 6_767_424
    assert audit["total_TiB"] > 1.5
    assert audit["status"] == "BLOCKED_EXACT_FULL_RATE_RECORDER"


def test_placed_cpml_coefficients_have_algebraic_inverse() -> None:
    _, _, objects, _, _, _ = _small_cpml_scene()
    audit = cpml_inverse_coefficient_audit(objects)
    assert audit["status"] == "PASS"
    assert len(audit["pml"]) == 6


def test_actual_fdtdx_cpml_ADE_one_step_reverse_round_trip() -> None:
    _, objects, states, config = _states(12)
    reconstructed = _reverse_one(objects, states[12], config, 11)
    _assert_close(reconstructed, states[11], rtol=5.0e-5, atol=3.0e-8)


def test_actual_fdtdx_cpml_ADE_eight_step_reverse_round_trip() -> None:
    _, objects, states, config = _states(8)
    reconstructed = states[-1]
    for time_step in reversed(range(8)):
        reconstructed = _reverse_one(objects, reconstructed, config, time_step)
    _assert_close(reconstructed, states[0], rtol=5.0e-4, atol=1.0e-7)
