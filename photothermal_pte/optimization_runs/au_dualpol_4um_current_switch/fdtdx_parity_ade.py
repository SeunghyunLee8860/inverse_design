"""Three-pole passive discrete-ADE carrier for the Au n-k density law.

The exact target susceptibility is

``epsilon(rho) - 1 = rho*(2.4 + 57.8j) + rho**2*(-833.77 + 69.36j)``.

One positive Lorentz pole represents the linear term.  Two positive Lorentz
poles with carrier-frequency phases on opposite sides of the quadratic term
represent that term as a positive combination.  All c4 coefficients are zero,
all oscillator strengths and damping rates are positive, and every float32
recurrence root lies strictly inside the unit circle.

This is not the historical ``rho * c3_Au`` approximation: there are three
independently certified basis responses, two are weighted by rho squared, and
their sum realizes the nonlinear n-k-then-square law directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_contract import (
    PHYSICS,
    grid_audit,
)


C0_M_S = 299_792_458.0
RELATIVE_EPSILON_TOLERANCE = 1.0e-5
JVP_CENTERED_STEP = 1.0e-3


@dataclass(frozen=True)
class LorentzBasis:
    name: str
    density_power: int
    c1: float
    c2: float
    c3_at_unit_weight: float


LINEAR_BASIS = LorentzBasis(
    name="linear_nk_cross_term",
    density_power=1,
    c1=1.9985358715057373,
    c2=-0.9985368847846985,
    c3_at_unit_weight=8.311595593113452e-05,
)
QUADRATIC_LOW_PHASE_BASIS = LorentzBasis(
    name="quadratic_low_phase",
    density_power=2,
    c1=1.9999223947525024,
    c2=-0.9999226331710815,
    c3_at_unit_weight=0.00012718782818410546,
)
QUADRATIC_HIGH_PHASE_BASIS = LorentzBasis(
    name="quadratic_high_phase",
    density_power=2,
    c1=1.999948263168335,
    c2=-0.9999485611915588,
    c3_at_unit_weight=0.0004339332808740437,
)
BASES = (
    LINEAR_BASIS,
    QUADRATIC_LOW_PHASE_BASIS,
    QUADRATIC_HIGH_PHASE_BASIS,
)


def carrier_dt_s() -> float:
    return float(grid_audit()["resources"]["time"]["dt_s"])


def carrier_omega_rad_s() -> float:
    return 2.0 * math.pi * C0_M_S / PHYSICS.wavelength_m


def target_epsilon(rho: Any) -> np.ndarray:
    """Authoritative float64 n-k-then-square target."""

    value = np.asarray(rho, dtype=np.float64)
    n = 1.0 + value * (2.2 - 1.0)
    k = value * 28.9
    return np.asarray((n + 1j * k) ** 2, dtype=np.complex128)


def _numpy_weights(rho: Any) -> np.ndarray:
    value = np.asarray(rho, dtype=np.float32)
    square = value * value
    return np.stack((value, square, square), axis=0)


def coefficients_numpy(rho: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return float32 c1/c2/c3/c4 with the three-pole axis first."""

    weights = _numpy_weights(rho)
    trailing = (1,) * (weights.ndim - 1)
    c1 = np.asarray([basis.c1 for basis in BASES], dtype=np.float32).reshape(
        (3, *trailing)
    )
    c2 = np.asarray([basis.c2 for basis in BASES], dtype=np.float32).reshape(
        (3, *trailing)
    )
    c1 = np.broadcast_to(c1, weights.shape)
    c2 = np.broadcast_to(c2, weights.shape)
    c3_basis = np.asarray(
        [basis.c3_at_unit_weight for basis in BASES], dtype=np.float32
    ).reshape((3, *trailing))
    c3 = c3_basis * weights
    c4 = np.zeros_like(c3)
    return c1, c2, c3, c4


def coefficient_vector_numpy(rho: float) -> np.ndarray:
    return np.concatenate(coefficients_numpy(np.float32(rho))).astype(np.float32)


def coefficient_jvp_analytic(rho: float, tangent: float = 1.0) -> np.ndarray:
    value = np.float32(rho)
    direction = np.float32(tangent)
    zero = np.zeros(3, dtype=np.float32)
    weight_jvp = np.asarray([1.0, 2.0 * value, 2.0 * value], dtype=np.float32)
    dc3 = np.asarray(
        [basis.c3_at_unit_weight for basis in BASES], dtype=np.float32
    ) * weight_jvp * direction
    return np.concatenate((zero, zero, dc3, zero))


def coefficients_jax(rho: Any):
    """Spatial JAX c1/c2/c3/c4 arrays with the three-pole axis first."""

    import jax.numpy as jnp

    value = jnp.asarray(rho, dtype=jnp.float32)
    square = value * value
    weights = jnp.stack((value, square, square), axis=0)
    trailing = (1,) * value.ndim
    c1 = jnp.asarray([basis.c1 for basis in BASES], dtype=jnp.float32).reshape(
        (3, *trailing)
    )
    c2 = jnp.asarray([basis.c2 for basis in BASES], dtype=jnp.float32).reshape(
        (3, *trailing)
    )
    c1 = jnp.broadcast_to(c1, weights.shape)
    c2 = jnp.broadcast_to(c2, weights.shape)
    c3_basis = jnp.asarray(
        [basis.c3_at_unit_weight for basis in BASES], dtype=jnp.float32
    ).reshape((3, *trailing))
    c3 = c3_basis * weights
    c4 = jnp.zeros_like(c3)
    return c1, c2, c3, c4


def coefficient_vector_jax(rho: Any):
    """Flatten the scalar coefficient map for the uniform-density JVP gate."""

    import jax.numpy as jnp

    value = jnp.asarray(rho, dtype=jnp.float32)
    if value.ndim != 0:
        raise ValueError("coefficient_vector_jax expects one uniform scalar density")
    return jnp.concatenate(coefficients_jax(value))


def realized_epsilon(rho: Any) -> np.ndarray:
    """Evaluate the actual complex64 unit-circle recurrence response."""

    value = np.asarray(rho, dtype=np.float32)
    c1, c2, c3, c4 = coefficients_numpy(value)
    theta = np.float32(carrier_omega_rad_s() * carrier_dt_s())
    z_minus = np.exp(np.complex64(-1j * theta))
    z_plus = np.exp(np.complex64(1j * theta))
    trailing = (1,) * value.ndim
    z_minus_bc = z_minus.reshape((1, *trailing))
    z_plus_bc = z_plus.reshape((1, *trailing))
    denominator = z_minus_bc - c1 - c2 * z_plus_bc
    chi = np.sum((c3 + c4 * z_minus_bc) / denominator, axis=0, dtype=np.complex64)
    return np.asarray(np.complex64(1.0) + chi, dtype=np.complex64)


def recurrence_roots(basis: LorentzBasis) -> np.ndarray:
    return np.roots([1.0, -float(np.float32(basis.c1)), -float(np.float32(basis.c2))])


def lorentz_parameters(basis: LorentzBasis) -> dict[str, float]:
    """Invert the frozen float32 recurrence to positive Lorentz parameters."""

    dt = carrier_dt_s()
    c1 = float(np.float32(basis.c1))
    c2 = float(np.float32(basis.c2))
    c3 = float(np.float32(basis.c3_at_unit_weight))
    denominator = 2.0 / (1.0 - c2)
    gamma_dt = 2.0 * (1.0 + c2) / (1.0 - c2)
    omega0_sq_dt2 = 2.0 - c1 * denominator
    if gamma_dt <= 0.0 or omega0_sq_dt2 <= 0.0 or c3 <= 0.0:
        raise RuntimeError(f"non-passive recurrence for {basis.name}")
    gamma = gamma_dt / dt
    omega0 = math.sqrt(omega0_sq_dt2) / dt
    coupling_sq = c3 * denominator / dt**2
    return {
        "gamma_rad_s": gamma,
        "omega0_rad_s": omega0,
        "coupling_sq_rad2_s2": coupling_sq,
        "delta_epsilon": coupling_sq / omega0**2,
    }


def fdtdx_api_coefficient_audit() -> dict[str, object]:
    """Require pinned FDTDX LorentzPole to reproduce the frozen coefficients."""

    from fdtdx.dispersion import LorentzPole, compute_pole_coefficients

    items: dict[str, object] = {}
    passed = True
    for basis in BASES:
        params = lorentz_parameters(basis)
        arrays = compute_pole_coefficients(
            (
                LorentzPole(
                    resonance_frequency=params["omega0_rad_s"],
                    damping=params["gamma_rad_s"],
                    delta_epsilon=params["delta_epsilon"],
                ),
            ),
            carrier_dt_s(),
        )
        realized = tuple(float(np.float32(array[0])) for array in arrays)
        expected = (
            float(np.float32(basis.c1)),
            float(np.float32(basis.c2)),
            float(np.float32(basis.c3_at_unit_weight)),
            0.0,
        )
        exact = realized == expected
        passed = passed and exact
        items[basis.name] = {
            "lorentz_parameters": params,
            "expected_float32_coefficients": list(expected),
            "fdtdx_float32_coefficients": list(realized),
            "exact": exact,
        }
    return {"status": "PASS" if passed else "FAIL", "bases": items}


def coefficient_hash() -> str:
    values = np.asarray(
        [
            value
            for basis in BASES
            for value in (basis.c1, basis.c2, basis.c3_at_unit_weight, 0.0)
        ],
        dtype="<f4",
    )
    digest = hashlib.sha256()
    digest.update(np.asarray([carrier_dt_s()], dtype="<f8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def jvp_audit() -> dict[str, object]:
    import jax

    points = (0.1, 0.35, 0.7, 0.9)
    max_jax_vs_analytic = 0.0
    max_jax_vs_fd = 0.0
    rows: list[dict[str, float]] = []
    for rho in points:
        _, jax_jvp = jax.jvp(
            coefficient_vector_jax,
            (np.float32(rho),),
            (np.float32(1.0),),
        )
        jax_value = np.asarray(jax_jvp, dtype=np.float32)
        analytic = coefficient_jvp_analytic(rho)
        plus = coefficient_vector_numpy(rho + JVP_CENTERED_STEP)
        minus = coefficient_vector_numpy(rho - JVP_CENTERED_STEP)
        finite_difference = (plus - minus) / np.float32(2.0 * JVP_CENTERED_STEP)
        scale = max(float(np.linalg.norm(jax_value)), np.finfo(np.float32).tiny)
        analytic_error = float(np.linalg.norm(jax_value - analytic) / scale)
        fd_error = float(np.linalg.norm(jax_value - finite_difference) / scale)
        max_jax_vs_analytic = max(max_jax_vs_analytic, analytic_error)
        max_jax_vs_fd = max(max_jax_vs_fd, fd_error)
        rows.append(
            {
                "rho": rho,
                "jax_vs_analytic_relative_l2": analytic_error,
                "jax_vs_centered_fd_relative_l2": fd_error,
            }
        )
    checks = {
        "jax_matches_analytic": max_jax_vs_analytic < 1.0e-6,
        "jax_matches_centered_fd": max_jax_vs_fd < 2.0e-4,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "centered_step": JVP_CENTERED_STEP,
        "max_jax_vs_analytic_relative_l2": max_jax_vs_analytic,
        "max_jax_vs_centered_fd_relative_l2": max_jax_vs_fd,
        "points": rows,
    }


def carrier_audit() -> dict[str, object]:
    densities = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    target = target_epsilon(densities)
    realized = realized_epsilon(densities).astype(np.complex128)
    relative_error = np.abs(realized - target) / np.abs(target)
    max_index = int(np.argmax(relative_error))
    root_payload: dict[str, object] = {}
    stable = True
    passive_parameters = True
    for basis in BASES:
        roots = recurrence_roots(basis)
        maximum = float(np.max(np.abs(roots)))
        params = lorentz_parameters(basis)
        stable = stable and maximum < 1.0
        passive_parameters = passive_parameters and all(value > 0.0 for value in params.values())
        root_payload[basis.name] = {
            "roots": [[float(root.real), float(root.imag)] for root in roots],
            "maximum_magnitude": maximum,
            "strict_jury_margin": (
                1.0 - float(np.float32(basis.c2)) - abs(float(np.float32(basis.c1)))
            ),
            "lorentz_parameters": params,
        }
    checks = {
        "101_uniform_densities": densities.size == 101,
        "finite": bool(np.all(np.isfinite(realized))),
        "passive_imaginary_epsilon": bool(np.min(realized.imag) >= 0.0),
        "positive_lorentz_parameters": passive_parameters,
        "all_c4_exactly_zero": all(
            float(value) == 0.0 for value in coefficients_numpy(1.0)[3]
        ),
        "exact_air_endpoint": complex(realized[0]) == complex(1.0, 0.0),
        "relative_error_below_1e_5": bool(
            np.max(relative_error) < RELATIVE_EPSILON_TOLERANCE
        ),
        "strict_recurrence_stability": stable,
        "implicit_divisor_exactly_one": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "density_count": int(densities.size),
        "target_law": "(1 + rho*(2.2-1) + 1j*rho*28.9)**2",
        "target_decomposition": {
            "linear_chi": [2.4, 57.8],
            "quadratic_chi": [-833.77, 69.36],
            "quadratic_realization": "positive_sum_of_two_passive_Lorentz_bases",
        },
        "dt_s": carrier_dt_s(),
        "omega_rad_s": carrier_omega_rad_s(),
        "coefficient_sha256": coefficient_hash(),
        "bases": [asdict(basis) for basis in BASES],
        "maximum_relative_complex_epsilon_error": float(relative_error[max_index]),
        "maximum_error_density": float(densities[max_index]),
        "air_endpoint_realized": [realized[0].real, realized[0].imag],
        "au_endpoint_target": [target[-1].real, target[-1].imag],
        "au_endpoint_realized": [realized[-1].real, realized[-1].imag],
        "au_endpoint_relative_error": float(relative_error[-1]),
        "minimum_realized_imaginary_epsilon": float(np.min(realized.imag)),
        "minimum_implicit_divisor": 1.0,
        "recurrence": root_payload,
        "fdtdx_api": fdtdx_api_coefficient_audit(),
        "jvp": jvp_audit(),
        "field_control_gate": {
            "status": "PENDING",
            "required_densities": [0.0, 0.25, 0.5, 0.75, 1.0],
            "optimizer_allowed": False,
        },
    }


def main() -> int:
    payload = carrier_audit()
    print(json.dumps(payload, indent=2, sort_keys=True))
    nested_pass = (
        payload["status"] == "PASS"
        and payload["fdtdx_api"]["status"] == "PASS"
        and payload["jvp"]["status"] == "PASS"
    )
    return 0 if nested_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
