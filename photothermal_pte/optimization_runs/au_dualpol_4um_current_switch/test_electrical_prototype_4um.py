from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    multiphysics_4um as legacy,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.electrical_prototype_4um import (
    block_mean,
    build_exact_binary_system,
    build_floored_binary_system,
    current_integrand_A_m2,
    electrical_load,
    evaluate_current_A,
    solve_weighting_cpu,
    system_sha256,
)


def _mask() -> np.ndarray:
    value = np.zeros((80, 80), dtype=np.uint8)
    value[31:49, 36:44] = 1
    value[38:42, 25:55] = 1
    return value


def _temperature(factor: int) -> np.ndarray:
    coordinate = (np.arange(160 * factor) + 0.5) / factor
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    return 0.3 * x + 0.002 * x**2 - 0.07 * y + 0.001 * x * y


def _direct_solution(system) -> np.ndarray:
    psi = np.zeros(system.full_matrix_S.shape[0], dtype=np.float64)
    psi[system.fixed] = system.fixed_values_V
    psi[system.free] = spsolve(system.reduced_matrix_S, system.reduced_rhs_A)
    return psi


def test_default_floored_builder_reproduces_legacy_operator() -> None:
    mask = _mask().astype(np.float64)
    temperature = _temperature(1)
    historical = legacy.build_electrical_system(mask, temperature)
    rebuilt = build_floored_binary_system(mask)
    difference = historical.full_matrix_S - rebuilt.full_matrix_S
    difference.eliminate_zeros()
    assert np.max(np.abs(difference.data), initial=0.0) <= 1e-15
    assert np.array_equal(historical.fixed, rebuilt.fixed)
    assert np.array_equal(historical.free, rebuilt.free)
    assert np.array_equal(historical.fixed_values_V, rebuilt.fixed_values_V)
    assert np.allclose(
        historical.reduced_rhs_A, rebuilt.reduced_rhs_A, rtol=0.0, atol=1e-15
    )
    historical_load = historical.objective_gradient_psi_A[: legacy.N_TA**2]
    rebuilt_load = electrical_load(temperature, 1)
    assert (
        np.linalg.norm(historical_load - rebuilt_load)
        / np.linalg.norm(historical_load)
        < 5e-16
    )


@pytest.mark.parametrize("factor", (1, 2, 4))
def test_exact_binary_system_has_only_physical_au_nodes(factor: int) -> None:
    mask = _mask()
    system = build_exact_binary_system(mask, factor)
    solid = int(np.count_nonzero(mask)) * factor**2
    assert system.ta_node_ids.shape == (160 * factor, 160 * factor)
    assert system.au_node_ids.shape == (80 * factor, 80 * factor)
    assert system.full_matrix_S.shape == (system.ta_node_ids.size + solid,) * 2
    assert np.count_nonzero(system.au_node_ids >= 0) == solid
    assert np.all(system.au_node_ids[~system.binary_mask] == -1)
    assert system.step_m == pytest.approx(100e-9 / factor)


def test_isolated_patterned_au_reduces_exactly_to_ta_sheet() -> None:
    system = build_exact_binary_system(
        _mask(), patterned_au_electrically_active=False
    )
    assert system.full_matrix_S.shape == (160**2, 160**2)
    assert np.all(system.au_node_ids == -1)
    assert system.void_model == "ta_only_isolated_au"


def test_cpu_cg_matches_direct_and_current_map_identity() -> None:
    system = build_exact_binary_system(_mask())
    iterative, audit = solve_weighting_cpu(system)
    direct = _direct_solution(system)
    assert audit["explicit_free_residual"] < 2e-10
    assert audit["terminal_balance_relative"] < 2e-10
    assert np.linalg.norm(iterative - direct) / np.linalg.norm(direct) < 2e-10
    temperature = _temperature(1)
    current = evaluate_current_A(system, iterative, temperature)
    mapped = float(
        np.sum(current_integrand_A_m2(system, iterative, temperature))
        * system.step_m**2
    )
    assert mapped == pytest.approx(current, rel=2e-13, abs=1e-24)


def test_exact_void_is_the_small_floor_limit() -> None:
    mask = _mask()
    temperature = _temperature(1)
    exact = build_exact_binary_system(mask)
    exact_current = evaluate_current_A(exact, _direct_solution(exact), temperature)
    relative_errors = []
    for floor in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        floored = build_floored_binary_system(
            mask,
            sigma_floor_fraction=floor,
            contact_floor_fraction=floor,
        )
        value = evaluate_current_A(floored, _direct_solution(floored), temperature)
        relative_errors.append(abs(value - exact_current) / abs(exact_current))
    assert np.all(np.isfinite(relative_errors))
    assert max(relative_errors) < 1e-8


def test_block_mean_is_mean_conservative() -> None:
    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    reduced = block_mean(values, 4)
    assert reduced.shape == (2, 2)
    assert np.mean(reduced) == pytest.approx(np.mean(values))


def test_system_hash_is_deterministic_and_parameter_sensitive() -> None:
    first = build_exact_binary_system(_mask(), electrical_contact_S_m2=1e8)
    second = build_exact_binary_system(_mask(), electrical_contact_S_m2=1e8)
    changed = build_exact_binary_system(_mask(), electrical_contact_S_m2=1e10)
    assert system_sha256(first) == system_sha256(second)
    assert system_sha256(first) != system_sha256(changed)


@pytest.mark.parametrize(
    ("builder", "kwargs"),
    (
        (build_exact_binary_system, {"refinement_factor": 3}),
        (build_exact_binary_system, {"electrical_contact_S_m2": 0.0}),
        (build_floored_binary_system, {"sigma_floor_fraction": 0.0}),
        (build_floored_binary_system, {"contact_floor_fraction": 1.0}),
    ),
)
def test_invalid_system_parameters_fail_closed(builder, kwargs) -> None:
    with pytest.raises(ValueError):
        builder(_mask(), **kwargs)


def test_nonbinary_mask_fails_closed() -> None:
    mask = _mask().astype(np.float64)
    mask[0, 0] = 0.5
    with pytest.raises(ValueError, match="exact 0/1"):
        build_exact_binary_system(mask)
