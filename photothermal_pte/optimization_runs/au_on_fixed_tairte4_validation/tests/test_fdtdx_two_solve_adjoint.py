"""Offline contracts for the checkpoint-free FDTDX adjoint helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.sources.profile import SingleFrequencyProfile


MODULE_PATH = Path(__file__).resolve().parents[1] / "fdtdx_two_solve_adjoint.py"
SPEC = importlib.util.spec_from_file_location("fdtdx_two_solve_adjoint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _placed_source(profile: jax.Array):
    config = SimulationConfig(
        time=100e-15,
        grid=UniformGrid(spacing=100e-9),
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.5,
        gradient_config=None,
    )
    source = MODULE.DistributedElectricCurrentSource(
        partial_grid_shape=(1, 1, 1),
        wave_character=WaveCharacter(wavelength=1.0e-6),
        temporal_profile=SingleFrequencyProfile(
            phase_shift=0.0,
            num_startup_periods=1,
        ),
        complex_profile=profile,
    )
    key = jax.random.PRNGKey(9)
    placed = source.place_on_grid(
        grid_slice_tuple=((2, 3), (2, 3), (2, 3)),
        config=config,
        key=key,
    )
    inv_eps = jnp.ones((3, 5, 5, 5), dtype=jnp.float32)
    return placed.apply(key, inv_eps, 1.0), config, inv_eps


def _placed_source_with_conductivity(profile: jax.Array, sigma: float):
    config = SimulationConfig(
        time=100e-15,
        grid=UniformGrid(spacing=100e-9),
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.5,
        gradient_config=None,
    )
    source = MODULE.DistributedElectricCurrentSource(
        partial_grid_shape=(1, 1, 1),
        wave_character=WaveCharacter(wavelength=1.0e-6),
        temporal_profile=SingleFrequencyProfile(
            phase_shift=0.0,
            num_startup_periods=1,
        ),
        complex_profile=profile,
    )
    key = jax.random.PRNGKey(11)
    placed = source.place_on_grid(
        grid_slice_tuple=((2, 3), (2, 3), (2, 3)),
        config=config,
        key=key,
    )
    inv_eps = 0.25 * jnp.ones((3, 5, 5, 5), dtype=jnp.float32)
    conductivity = sigma * jnp.ones_like(inv_eps)
    return (
        placed.apply(key, inv_eps, 1.0, electric_conductivity=conductivity),
        config,
        inv_eps,
    )


def test_complex_current_uses_cosine_sine_quadrature() -> None:
    profile = jnp.asarray([2.0 + 3.0j, -4.0 + 5.0j, 0.0j], dtype=jnp.complex64)[
        :, None, None, None
    ]
    source, config, inv_eps = _placed_source(profile)
    zero = jnp.zeros((3, 5, 5, 5), dtype=jnp.float32)

    one_period_step = source.wave_character.get_period() / config.time_step_duration
    at_zero = source.update_E(
        zero, inv_eps, 1.0, jnp.asarray(one_period_step), False
    )
    expected_zero = -config.courant_number * np.asarray([2.0, -4.0, 0.0])
    np.testing.assert_allclose(
        np.asarray(at_zero[:, 2, 2, 2]), expected_zero, rtol=2e-6, atol=2e-6
    )

    quarter_step = 1.25 * source.wave_character.get_period() / config.time_step_duration
    at_quarter = source.update_E(
        zero, inv_eps, 1.0, jnp.asarray(quarter_step), False
    )
    expected_quarter = -config.courant_number * np.asarray([3.0, 5.0, 0.0])
    np.testing.assert_allclose(
        np.asarray(at_quarter[:, 2, 2, 2]),
        expected_quarter,
        rtol=2e-6,
        atol=2e-6,
    )


def test_wirtinger_and_current_normalization() -> None:
    electric = jnp.asarray([1.0 + 2.0j, 2.0 - 1.0j], dtype=jnp.complex64)
    coefficient = jnp.asarray([3.0, 4.0], dtype=jnp.float32)
    derivative = MODULE.quadratic_wirtinger_derivative(electric, coefficient)
    current = MODULE.adjoint_current_from_wirtinger(derivative, 0.25)
    np.testing.assert_allclose(
        np.asarray(derivative), np.asarray([3.0 + 6.0j, 8.0 - 4.0j])
    )
    np.testing.assert_allclose(
        np.asarray(current), np.asarray([-12.0 + 24.0j, -32.0 - 16.0j])
    )


def test_conductive_source_uses_electric_update_denominator() -> None:
    profile = jnp.asarray([1.0 + 0.0j, 0.0j, 0.0j], dtype=jnp.complex64)[
        :, None, None, None
    ]
    sigma = 2.5e4
    source, config, inv_eps = _placed_source_with_conductivity(profile, sigma)
    zero = jnp.zeros((3, 5, 5, 5), dtype=jnp.float32)
    one_period_step = source.wave_character.get_period() / config.time_step_duration
    updated = source.update_E(
        zero, inv_eps, 1.0, jnp.asarray(one_period_step), False
    )
    denominator = (
        1.0
        + config.courant_number
        * sigma
        * float(MODULE.eta0)
        * 0.25
        / 2.0
    )
    expected = -config.courant_number * 0.25 / denominator
    np.testing.assert_allclose(
        np.asarray(updated[0, 2, 2, 2]), expected, rtol=2e-6, atol=2e-6
    )


def test_harmonic_gradient_is_component_local_and_real() -> None:
    forward = jnp.asarray([1.0 + 0.5j, 2.0 - 0.25j], dtype=jnp.complex64)
    adjoint = jnp.asarray([0.2 - 0.1j, -0.3 + 0.4j], dtype=jnp.complex64)
    derivative = jnp.asarray([3.0 + 2.0j, -1.0 + 0.5j], dtype=jnp.complex64)
    omega = 2.0 * np.pi * 3.0e13
    dt = 1.0e-16
    result = MODULE.harmonic_material_gradient(
        forward, adjoint, derivative, omega, dt
    )
    z = np.exp(-1j * omega * dt)
    expected = -2.0 * np.real(
        (z - 1.0)
        * np.asarray(derivative)
        * np.asarray(forward)
        * np.asarray(adjoint)
    )
    np.testing.assert_allclose(np.asarray(result), expected, rtol=2e-6, atol=2e-6)
    assert np.isrealobj(np.asarray(result))
