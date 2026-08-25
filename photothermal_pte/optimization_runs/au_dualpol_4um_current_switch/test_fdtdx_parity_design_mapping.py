from __future__ import annotations

import inspect

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_parity_design_mapping,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_design_mapping import (
    CELL_SHAPE,
    FIRST_CERTIFICATE_BETA,
    MAPPING,
    NODE_SHAPE,
    control_density,
    nodal_to_cell_average,
    nodal_to_cell_jvp,
    nodal_to_cell_vjp,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    NOMINAL_MAPPING as LUMERICAL_MAPPING,
)


def _latent() -> np.ndarray:
    return control_density("nonuniform_gray")["latent"]


def test_fresh_mapping_matches_committed_lumerical_parity_coefficients() -> None:
    assert np.array_equal(MAPPING.kernel, LUMERICAL_MAPPING.kernel)
    assert np.array_equal(MAPPING.normalization, LUMERICAL_MAPPING.normalization)
    latent = _latent()
    assert np.array_equal(MAPPING.filtered(latent), LUMERICAL_MAPPING.filtered(latent))
    assert np.array_equal(
        MAPPING.physical_nodes(latent, FIRST_CERTIFICATE_BETA),
        LUMERICAL_MAPPING.physical(latent, FIRST_CERTIFICATE_BETA),
    )


def test_mapping_contract_and_constants_are_exact() -> None:
    audit = MAPPING.audit()
    assert audit["status"] == "PASS"
    assert audit["latent_shape"] == [81, 81]
    assert audit["projected_nodal_shape"] == [81, 81]
    assert audit["physical_cell_shape"] == [80, 80]
    assert audit["filter_radius_m"] == 500e-9
    assert audit["projection_eta"] == 0.5
    assert audit["kernel_nonzero_count"] == 69
    assert audit["coefficient_sha256"] == "2c2f8cc8a613777f0bf08c7985efe3960992b8028ef46de399c5bce6b25f41d5"
    assert max(audit["constant_projection_max_abs_errors"].values()) < 1.0e-14
    assert audit["optimizer_enabled"] is False


def test_nodal_to_cell_map_has_exact_transpose_and_fd() -> None:
    rng = np.random.default_rng(20260825)
    nodes = 0.1 + 0.8 * rng.random(NODE_SHAPE)
    direction = rng.standard_normal(NODE_SHAPE)
    cotangent = rng.standard_normal(CELL_SHAPE)
    tangent = nodal_to_cell_jvp(direction)
    pullback = nodal_to_cell_vjp(cotangent)
    left = float(np.vdot(tangent, cotangent))
    right = float(np.vdot(direction, pullback))
    assert abs(left - right) / max(abs(left), abs(right), 1.0e-300) < 2.0e-15
    step = 1.0e-6
    finite = (
        nodal_to_cell_average(nodes + step * direction)
        - nodal_to_cell_average(nodes - step * direction)
    ) / (2.0 * step)
    assert np.linalg.norm(tangent - finite) / np.linalg.norm(finite) < 2.0e-10


def test_complete_numpy_mapping_jvp_vjp_and_fd() -> None:
    rng = np.random.default_rng(71)
    latent = _latent()
    direction = rng.standard_normal(NODE_SHAPE)
    direction /= np.max(np.abs(direction))
    cotangent = rng.standard_normal(CELL_SHAPE)
    beta = FIRST_CERTIFICATE_BETA
    tangent = MAPPING.cell_jvp(latent, direction, beta)
    pullback = MAPPING.cell_vjp(latent, cotangent, beta)
    left = float(np.vdot(tangent, cotangent))
    right = float(np.vdot(direction, pullback))
    assert abs(left - right) / max(abs(left), abs(right), 1.0e-300) < 1.0e-12
    step = 1.0e-6
    finite = (
        MAPPING.cell_density(latent + step * direction, beta)
        - MAPPING.cell_density(latent - step * direction, beta)
    ) / (2.0 * step)
    assert np.linalg.norm(tangent - finite) / np.linalg.norm(finite) < 1.0e-8


def test_jax_float32_mapping_matches_numpy_and_autodiff() -> None:
    import jax
    import jax.numpy as jnp

    latent = _latent().astype(np.float32)
    cotangent = np.random.default_rng(19).standard_normal(CELL_SHAPE).astype(np.float32)
    observed = np.asarray(MAPPING.jax_cell_density(jnp.asarray(latent), 4.0))
    expected = MAPPING.cell_density(latent.astype(np.float64), 4.0)
    assert np.max(np.abs(observed - expected)) < 5.0e-7
    jax_gradient = np.asarray(
        jax.grad(
            lambda value: jnp.vdot(
                MAPPING.jax_cell_density(value, 4.0), jnp.asarray(cotangent)
            )
        )(jnp.asarray(latent))
    )
    numpy_gradient = MAPPING.cell_vjp(
        latent.astype(np.float64), cotangent.astype(np.float64), 4.0
    )
    assert np.linalg.norm(jax_gradient - numpy_gradient) / np.linalg.norm(
        numpy_gradient
    ) < 2.0e-6


def test_control_densities_cannot_bypass_nodal_map() -> None:
    empty = control_density("empty")
    full = control_density("full")
    gray = control_density("nonuniform_gray")
    assert np.all(empty["latent"] == 0.0)
    assert np.all(empty["projected_nodes"] == 0.0)
    assert np.all(empty["cells"] == 0.0)
    assert np.all(full["latent"] == 1.0)
    assert np.all(full["projected_nodes"] == 1.0)
    assert np.all(full["cells"] == 1.0)
    assert 0.0 < gray["ranges"]["cells"][0] < gray["ranges"]["cells"][1] < 1.0
    with pytest.raises(ValueError, match="unknown density control"):
        control_density("legacy_checkpoint")


def test_mapping_has_no_legacy_fdtdx_or_material_fraction_dependency() -> None:
    source = inspect.getsource(fdtdx_parity_design_mapping)
    assert "material_fraction" not in source
    assert "fdtdx_4um_model" not in source
    assert "combined_4um" not in source
    assert "historical_checkpoint" not in source
    assert "rho**3" not in source and "rho ** 3" not in source
