"""Checkpoint-free harmonic adjoint helpers for dispersive FDTDX models.

This module deliberately does not call ``jax.grad`` through the FDTD time
loop.  A forward continuous-wave solve records the settled electric-field
phasor, a second solve is driven by the complex distributed current associated
with the objective's Wirtinger derivative, and the two phasors are contracted
locally.  Only the two final phasor fields are retained; no time-history or
checkpoint stack is allocated.

The implementation is intentionally kept separate from the existing
checkpointed certificate.  Its normalization and material contraction must be
validated against central finite differences before it is used by production
optimization.
"""

from __future__ import annotations

import math
from typing import Self

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, field, private_field
from fdtdx.core.null import Null
from fdtdx.constants import eta0
from fdtdx.objects.sources.source import Source


@autoinit
class DistributedElectricCurrentSource(Source):
    """Soft electric-current source with a complex vector spatial profile.

    ``complex_profile`` is the desired current phasor on the native electric
    Yee components, with shape ``(3, Nx, Ny, Nz)`` over this source object's
    placed grid slice.  FDTDX evolves real-valued time-domain fields, so the
    source injects the real and imaginary profile parts in cosine and sine
    quadrature.  With ``SingleFrequencyProfile(phase_shift=0)`` and a zero
    ``WaveCharacter.phase_shift`` this realizes

    ``J(t) = Re[J_tilde exp(-i omega t)]``.

    The electric update follows the same impressed-current convention as
    :class:`fdtdx.PointDipoleSource`:

    ``E <- E - courant * inv_eps_base * J(t)``.

    The base (instantaneous) inverse permittivity is intentional.  After the
    electric update is multiplied by the base permittivity, this produces the
    harmonic residual right-hand side ``-courant*J``.  In conductive cells the
    same Crank--Nicolson denominator as FDTDX's electric-field update is also
    applied.  Using the carrier-frequency effective inverse permittivity here
    would instead precondition the requested adjoint right-hand side by
    ``1/epsilon_eff`` and is not the transpose solve required by the material
    gradient.

    The source is a numerical adjoint source, not a physical illumination
    model.  It must be placed on exactly the same component-specific Yee
    support used to form the objective derivative.
    """

    complex_profile: jax.Array = field()
    _inv_eps_local: jax.Array = private_field()

    def __post_init__(self) -> None:
        profile = jnp.asarray(self.complex_profile)
        if profile.ndim != 4 or profile.shape[0] != 3:
            raise ValueError(
                "complex_profile must have shape (3, Nx, Ny, Nz), got "
                f"{profile.shape}"
            )
        if not jnp.issubdtype(profile.dtype, jnp.complexfloating):
            raise TypeError(
                "complex_profile must be complex so its phase is explicit, got "
                f"{profile.dtype}"
            )

    def apply(
        self: Self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        dispersive_c1: jax.Array | None = None,
        dispersive_c2: jax.Array | None = None,
        dispersive_c3: jax.Array | None = None,
        electric_conductivity: jax.Array | None = None,
        dispersive_c4: jax.Array | None = None,
    ) -> Self:
        del key, inv_permeabilities

        if tuple(self.complex_profile.shape[1:]) != tuple(self.grid_shape):
            raise ValueError(
                "complex_profile spatial shape must equal the realized source grid: "
                f"{self.complex_profile.shape[1:]} != {self.grid_shape}"
            )
        component_slice = (slice(None), *self.grid_slice)
        inv_eps_slice = inv_permittivities[component_slice]
        if electric_conductivity is not None:
            sigma_slice = electric_conductivity[component_slice]
            denominator = (
                1.0
                + self._config.courant_number
                * sigma_slice
                * eta0
                * inv_eps_slice
                / 2.0
            )
            inv_eps_slice = inv_eps_slice / denominator
        del dispersive_c1, dispersive_c2, dispersive_c3, dispersive_c4
        return self.aset("_inv_eps_local", inv_eps_slice, create_new_ok=True)

    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permeabilities
        dt = self._config.time_step_duration
        cosine = self.temporal_profile.get_amplitude(
            time=time_step * dt,
            period=self.wave_character.get_period(),
            phase_shift=self.wave_character.phase_shift,
        )
        sine = self.temporal_profile.get_amplitude(
            time=time_step * dt,
            period=self.wave_character.get_period(),
            phase_shift=self.wave_character.phase_shift - 0.5 * math.pi,
        )
        current = (
            jnp.real(self.complex_profile) * cosine
            + jnp.imag(self.complex_profile) * sine
        )
        if isinstance(self._inv_eps_local, Null):
            inv_eps = inv_permittivities[(slice(None), *self.grid_slice)]
        else:
            inv_eps = self._inv_eps_local
        sign = 1.0 if inverse else -1.0
        injection = (
            self._config.courant_number
            * self.static_amplitude_factor
            * inv_eps
            * current
        )
        return E.at[(slice(None), *self.grid_slice)].add(
            sign * injection.astype(E.dtype)
        )

    def update_H(
        self,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permittivities, inv_permeabilities, time_step, inverse
        return H


def quadratic_wirtinger_derivative(
    electric_phasor: jax.Array,
    coefficient: jax.Array,
) -> jax.Array:
    """Return ``d sum(coefficient*|E|^2) / dE*`` on the Yee grid."""

    if electric_phasor.shape != coefficient.shape:
        raise ValueError(
            f"electric/coefficient shape mismatch: {electric_phasor.shape} != "
            f"{coefficient.shape}"
        )
    return coefficient * electric_phasor


def adjoint_current_from_wirtinger(
    wirtinger_derivative: jax.Array,
    courant_number: float,
) -> jax.Array:
    """Map a phasor-domain objective derivative to FDTDX source current.

    For the forward residual convention used by FDTDX, an impressed current
    contributes ``-courant*J`` before multiplication by ``inv_eps``.  The
    reciprocal adjoint right-hand side is the complex conjugate Wirtinger
    derivative, hence ``J_adj=-conj(dF/dE*)/courant``.
    """

    if not math.isfinite(courant_number) or courant_number <= 0.0:
        raise ValueError(f"invalid courant_number={courant_number}")
    return -jnp.conj(wirtinger_derivative) / courant_number


def harmonic_material_gradient(
    forward_electric_phasor: jax.Array,
    adjoint_electric_phasor: jax.Array,
    d_relative_permittivity: jax.Array,
    angular_frequency: float,
    time_step_duration: float,
) -> jax.Array:
    """Return the field-mediated gradient on the native component grid.

    Eliminating the ADE polarization at the settled harmonic frequency gives
    the exact finite-time-step relative permittivity.  In the FDTDX electric
    update, its residual multiplier is ``z-1`` with
    ``z=exp(-i*omega*dt)``.  Reciprocity then gives the unconjugated
    forward/adjoint contraction below.  Spatial reduction (including the
    3-D-to-2-D design transpose) is intentionally left to the caller.
    """

    if forward_electric_phasor.shape != adjoint_electric_phasor.shape:
        raise ValueError(
            "forward/adjoint field shape mismatch: "
            f"{forward_electric_phasor.shape} != {adjoint_electric_phasor.shape}"
        )
    if d_relative_permittivity.shape != forward_electric_phasor.shape:
        raise ValueError(
            "permittivity derivative must be component-specific and collocated: "
            f"{d_relative_permittivity.shape} != {forward_electric_phasor.shape}"
        )
    z = jnp.exp(-1j * angular_frequency * time_step_duration)
    return -2.0 * jnp.real(
        (z - 1.0)
        * d_relative_permittivity
        * forward_electric_phasor
        * adjoint_electric_phasor
    )
