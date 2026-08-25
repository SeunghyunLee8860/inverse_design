"""Diagnostic exact-binary electrical operator for the assumed 4-um device.

This module does not define the target device and cannot authorize an
optimization.  It exists to separate three numerical questions in the current
rectangular prototype:

* lateral electrical pitch convergence;
* the zero-void limit of the historical floored Au network; and
* sensitivity to the assumed Au/TaIrTe4 electrical contact conductance.

The exact-binary builder removes void Au nodes instead of retaining an
arbitrarily weak sheet/contact network.  The floored builder is retained only
to reproduce and quantify the historical regularization.  Coordinates remain
solver x=crystal b and solver y=crystal a, with ideal full-edge x terminals.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    AU_THICKNESS_M,
    CONTACT_FLOOR_FRACTION,
    ELECTRICAL_CONTACT_S_M2,
    N_DESIGN,
    N_TA,
    SEEBECK_TA_XY_V_K,
    SIGMA_AU_S_M,
    SIGMA_FLOOR_FRACTION,
    SIGMA_TA_XY_S_M,
    STEP_M,
    TA_THICKNESS_M,
)


ALLOWED_REFINEMENT_FACTORS = (1, 2, 4)


@dataclass(frozen=True)
class PrototypeElectricalSystem:
    """Sparse weighting-potential system on one nested lateral grid."""

    full_matrix_S: sparse.csr_matrix
    reduced_matrix_S: sparse.csr_matrix
    reduced_rhs_A: np.ndarray
    free: np.ndarray
    fixed: np.ndarray
    fixed_values_V: np.ndarray
    ta_node_ids: np.ndarray
    au_node_ids: np.ndarray
    binary_mask: np.ndarray
    refinement_factor: int
    step_m: float
    electrical_contact_S_m2: float
    void_model: str
    sigma_floor_fraction: float | None
    contact_floor_fraction: float | None


def _validated_factor(refinement_factor: int) -> int:
    if (
        not isinstance(refinement_factor, int)
        or isinstance(refinement_factor, bool)
        or refinement_factor not in ALLOWED_REFINEMENT_FACTORS
    ):
        raise ValueError(
            f"electrical refinement factor must be one of {ALLOWED_REFINEMENT_FACTORS}"
        )
    return refinement_factor


def _validated_binary_mask(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask)
    if value.shape != CONTRACT.design_shape:
        raise ValueError(f"binary mask must have shape {CONTRACT.design_shape}")
    if not np.all(np.isfinite(value)) or not np.all((value == 0) | (value == 1)):
        raise ValueError("binary mask must contain only finite exact 0/1 values")
    return value.astype(bool, copy=True)


def _validated_contact(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("electrical contact conductance must be finite and positive")
    return result


def _validated_floor(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be finite and strictly between zero and one")
    return result


def _append_edges(
    rows: list[np.ndarray],
    cols: list[np.ndarray],
    data: list[np.ndarray],
    left: np.ndarray,
    right: np.ndarray,
    conductance_S: float | np.ndarray,
) -> None:
    lhs = np.asarray(left, dtype=np.int64).reshape(-1)
    rhs = np.asarray(right, dtype=np.int64).reshape(-1)
    if lhs.shape != rhs.shape:
        raise ValueError("edge endpoint arrays must have equal shape")
    conductance_input = np.asarray(conductance_S, dtype=np.float64)
    if conductance_input.ndim == 0:
        conductance = np.full(lhs.shape, float(conductance_input), dtype=np.float64)
    else:
        conductance = conductance_input.reshape(-1).copy()
        if conductance.shape != lhs.shape:
            raise ValueError("edge conductance array must match the endpoint count")
    if np.any(~np.isfinite(conductance)) or np.any(conductance <= 0.0):
        raise ValueError("all assembled conductances must be finite and positive")
    rows.append(np.concatenate((lhs, rhs, lhs, rhs)))
    cols.append(np.concatenate((lhs, rhs, rhs, lhs)))
    data.append(
        np.concatenate((conductance, conductance, -conductance, -conductance))
    )


def _ta_network(
    refinement_factor: int,
    rows: list[np.ndarray],
    cols: list[np.ndarray],
    data: list[np.ndarray],
) -> np.ndarray:
    n_ta = N_TA * refinement_factor
    ids = np.arange(n_ta * n_ta, dtype=np.int64).reshape(n_ta, n_ta)
    _append_edges(
        rows,
        cols,
        data,
        ids[:-1, :],
        ids[1:, :],
        SIGMA_TA_XY_S_M[0] * TA_THICKNESS_M,
    )
    _append_edges(
        rows,
        cols,
        data,
        ids[:, :-1],
        ids[:, 1:],
        SIGMA_TA_XY_S_M[1] * TA_THICKNESS_M,
    )
    return ids


def _finalize_system(
    *,
    rows: list[np.ndarray],
    cols: list[np.ndarray],
    data: list[np.ndarray],
    node_count: int,
    ta_node_ids: np.ndarray,
    au_node_ids: np.ndarray,
    binary_mask: np.ndarray,
    refinement_factor: int,
    electrical_contact_S_m2: float,
    void_model: str,
    sigma_floor_fraction: float | None,
    contact_floor_fraction: float | None,
) -> PrototypeElectricalSystem:
    matrix = sparse.coo_matrix(
        (
            np.concatenate(data),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(node_count, node_count),
    ).tocsr()
    matrix.sum_duplicates()
    low = ta_node_ids[0, :].copy()
    high = ta_node_ids[-1, :].copy()
    fixed = np.concatenate((low, high))
    fixed_values = np.concatenate(
        (np.zeros(low.size, dtype=np.float64), np.ones(high.size, dtype=np.float64))
    )
    free_mask = np.ones(node_count, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)
    reduced = matrix[free][:, free].tocsr()
    rhs = -np.asarray(matrix[free][:, fixed] @ fixed_values).reshape(-1)
    return PrototypeElectricalSystem(
        full_matrix_S=matrix,
        reduced_matrix_S=reduced,
        reduced_rhs_A=rhs,
        free=free,
        fixed=fixed,
        fixed_values_V=fixed_values,
        ta_node_ids=ta_node_ids,
        au_node_ids=au_node_ids,
        binary_mask=binary_mask,
        refinement_factor=refinement_factor,
        step_m=STEP_M / refinement_factor,
        electrical_contact_S_m2=electrical_contact_S_m2,
        void_model=void_model,
        sigma_floor_fraction=sigma_floor_fraction,
        contact_floor_fraction=contact_floor_fraction,
    )


def build_exact_binary_system(
    mask: np.ndarray,
    refinement_factor: int = 1,
    *,
    electrical_contact_S_m2: float = ELECTRICAL_CONTACT_S_M2,
    patterned_au_electrically_active: bool = True,
) -> PrototypeElectricalSystem:
    """Build a nested-grid system with no Au degrees of freedom in void cells."""

    factor = _validated_factor(refinement_factor)
    base_mask = _validated_binary_mask(mask)
    contact = _validated_contact(electrical_contact_S_m2)
    refined_mask = np.repeat(np.repeat(base_mask, factor, axis=0), factor, axis=1)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    ta_ids = _ta_network(factor, rows, cols, data)
    n_design = N_DESIGN * factor
    au_ids = np.full((n_design, n_design), -1, dtype=np.int64)
    next_id = ta_ids.size
    if patterned_au_electrically_active:
        au_ids[refined_mask] = np.arange(
            next_id, next_id + int(np.count_nonzero(refined_mask)), dtype=np.int64
        )
        next_id += int(np.count_nonzero(refined_mask))
        for left_slice, right_slice in (
            ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
            ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ):
            pair = refined_mask[left_slice] & refined_mask[right_slice]
            _append_edges(
                rows,
                cols,
                data,
                au_ids[left_slice][pair],
                au_ids[right_slice][pair],
                SIGMA_AU_S_M * AU_THICKNESS_M,
            )
        offset = (ta_ids.shape[0] - n_design) // 2
        ta_under_au = ta_ids[
            offset : offset + n_design, offset : offset + n_design
        ]
        _append_edges(
            rows,
            cols,
            data,
            ta_under_au[refined_mask],
            au_ids[refined_mask],
            contact * (STEP_M / factor) ** 2,
        )
    return _finalize_system(
        rows=rows,
        cols=cols,
        data=data,
        node_count=next_id,
        ta_node_ids=ta_ids,
        au_node_ids=au_ids,
        binary_mask=refined_mask,
        refinement_factor=factor,
        electrical_contact_S_m2=contact,
        void_model=("exact_binary_active_au_only" if patterned_au_electrically_active else "ta_only_isolated_au"),
        sigma_floor_fraction=None,
        contact_floor_fraction=None,
    )


def build_floored_binary_system(
    mask: np.ndarray,
    refinement_factor: int = 1,
    *,
    electrical_contact_S_m2: float = ELECTRICAL_CONTACT_S_M2,
    sigma_floor_fraction: float = SIGMA_FLOOR_FRACTION,
    contact_floor_fraction: float = CONTACT_FLOOR_FRACTION,
) -> PrototypeElectricalSystem:
    """Reproduce the historical all-Au-node regularization for a binary mask."""

    factor = _validated_factor(refinement_factor)
    base_mask = _validated_binary_mask(mask)
    contact = _validated_contact(electrical_contact_S_m2)
    sigma_floor = _validated_floor(sigma_floor_fraction, "sigma floor fraction")
    contact_floor = _validated_floor(
        contact_floor_fraction, "contact floor fraction"
    )
    refined_mask = np.repeat(np.repeat(base_mask, factor, axis=0), factor, axis=1)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    ta_ids = _ta_network(factor, rows, cols, data)
    n_design = N_DESIGN * factor
    au_ids = np.arange(
        ta_ids.size, ta_ids.size + n_design * n_design, dtype=np.int64
    ).reshape(n_design, n_design)
    sigma = SIGMA_AU_S_M * (
        sigma_floor + refined_mask.astype(np.float64) * (1.0 - sigma_floor)
    )
    step = STEP_M / factor
    for left_slice, right_slice in (
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ):
        resistance = (
            0.5 * step / sigma[left_slice] + 0.5 * step / sigma[right_slice]
        )
        _append_edges(
            rows,
            cols,
            data,
            au_ids[left_slice],
            au_ids[right_slice],
            AU_THICKNESS_M * step / resistance,
        )
    offset = (ta_ids.shape[0] - n_design) // 2
    ta_under_au = ta_ids[offset : offset + n_design, offset : offset + n_design]
    contact_map = contact * (
        contact_floor
        + refined_mask.astype(np.float64) * (1.0 - contact_floor)
    )
    _append_edges(
        rows,
        cols,
        data,
        ta_under_au,
        au_ids,
        contact_map * step**2,
    )
    return _finalize_system(
        rows=rows,
        cols=cols,
        data=data,
        node_count=ta_ids.size + au_ids.size,
        ta_node_ids=ta_ids,
        au_node_ids=au_ids,
        binary_mask=refined_mask,
        refinement_factor=factor,
        electrical_contact_S_m2=contact,
        void_model="historical_floored_all_au_nodes",
        sigma_floor_fraction=sigma_floor,
        contact_floor_fraction=contact_floor,
    )


def solve_weighting_cpu(
    system: PrototypeElectricalSystem,
    *,
    relative_tolerance: float = 1.0e-11,
    max_iterations: int = 100_000,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Solve the SPD weighting problem with Jacobi-preconditioned CPU CG."""

    diagonal = system.reduced_matrix_S.diagonal()
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise RuntimeError("electrical reduced matrix has invalid diagonal")
    preconditioner = sparse_linalg.LinearOperator(
        system.reduced_matrix_S.shape, matvec=lambda value: value / diagonal
    )
    iteration_count = 0

    def count_iteration(_: np.ndarray) -> None:
        nonlocal iteration_count
        iteration_count += 1

    solution, info = sparse_linalg.cg(
        system.reduced_matrix_S,
        system.reduced_rhs_A,
        rtol=relative_tolerance,
        atol=0.0,
        maxiter=max_iterations,
        M=preconditioner,
        callback=count_iteration,
    )
    if info != 0:
        raise RuntimeError(f"electrical CPU CG failed with info={info}")
    psi = np.zeros(system.full_matrix_S.shape[0], dtype=np.float64)
    psi[system.fixed] = system.fixed_values_V
    psi[system.free] = solution
    residual = np.asarray(system.full_matrix_S @ psi).reshape(-1)
    explicit = np.linalg.norm(residual[system.free]) / max(
        np.linalg.norm(system.reduced_rhs_A), np.finfo(float).tiny
    )
    terminal_count = system.ta_node_ids.shape[1]
    low = float(np.sum(residual[system.fixed[:terminal_count]]))
    high = float(np.sum(residual[system.fixed[terminal_count:]]))
    balance = abs(low + high) / max(abs(low), abs(high), np.finfo(float).tiny)
    return psi, {
        "solver": "scipy_cg_jacobi_cpu",
        "iterations": iteration_count,
        "explicit_free_residual": float(explicit),
        "terminal_balance_relative": float(balance),
        "low_terminal_A_per_V": low,
        "high_terminal_A_per_V": high,
    }


def electrical_load(temperature_K: np.ndarray, refinement_factor: int) -> np.ndarray:
    """Return the Ta-only Shockley--Ramo load on a nested electrical grid."""

    factor = _validated_factor(refinement_factor)
    n_ta = N_TA * factor
    temperature = np.asarray(temperature_K, dtype=np.float64)
    if temperature.shape != (n_ta, n_ta) or np.any(~np.isfinite(temperature)):
        raise ValueError(f"temperature must be finite with shape {(n_ta, n_ta)}")
    ids = np.arange(n_ta * n_ta, dtype=np.int64).reshape(n_ta, n_ta)
    load = np.zeros(n_ta * n_ta, dtype=np.float64)
    for axis, (left, right) in enumerate(
        ((ids[:-1, :], ids[1:, :]), (ids[:, :-1], ids[:, 1:]))
    ):
        delta = np.diff(temperature, axis=axis)
        value = (
            SIGMA_TA_XY_S_M[axis]
            * TA_THICKNESS_M
            * SEEBECK_TA_XY_V_K[axis]
            * delta
        ).reshape(-1)
        np.add.at(load, left.reshape(-1), value)
        np.add.at(load, right.reshape(-1), -value)
    return load


def evaluate_current_A(
    system: PrototypeElectricalSystem,
    psi: np.ndarray,
    temperature_K: np.ndarray,
) -> float:
    potential = np.asarray(psi, dtype=np.float64)
    if potential.shape != (system.full_matrix_S.shape[0],):
        raise ValueError("weighting potential shape does not match system")
    load = electrical_load(temperature_K, system.refinement_factor)
    return float(load @ potential[: system.ta_node_ids.size])


def current_integrand_A_m2(
    system: PrototypeElectricalSystem,
    psi: np.ndarray,
    temperature_K: np.ndarray,
) -> np.ndarray:
    """Return a Ta cell-centred map whose area integral equals total current."""

    factor = system.refinement_factor
    n_ta = N_TA * factor
    temperature = np.asarray(temperature_K, dtype=np.float64)
    if temperature.shape != (n_ta, n_ta):
        raise ValueError(f"temperature must have shape {(n_ta, n_ta)}")
    ta_psi = np.asarray(psi, dtype=np.float64)[system.ta_node_ids]
    result = np.zeros_like(temperature)
    step2 = system.step_m**2
    for axis in (0, 1):
        delta_t = np.diff(temperature, axis=axis)
        delta_psi = np.diff(ta_psi, axis=axis)
        edge_current = -(
            SIGMA_TA_XY_S_M[axis]
            * TA_THICKNESS_M
            * SEEBECK_TA_XY_V_K[axis]
            * delta_t
            * delta_psi
        )
        if axis == 0:
            result[:-1, :] += 0.5 * edge_current / step2
            result[1:, :] += 0.5 * edge_current / step2
        else:
            result[:, :-1] += 0.5 * edge_current / step2
            result[:, 1:] += 0.5 * edge_current / step2
    return result


def block_mean(values: np.ndarray, block_factor: int) -> np.ndarray:
    """Conservatively average a 2-D nested cell field by an integer factor."""

    array = np.asarray(values, dtype=np.float64)
    if (
        not isinstance(block_factor, int)
        or isinstance(block_factor, bool)
        or block_factor < 1
        or array.ndim != 2
        or array.shape[0] % block_factor
        or array.shape[1] % block_factor
        or np.any(~np.isfinite(array))
    ):
        raise ValueError("invalid 2-D field or block factor")
    return array.reshape(
        array.shape[0] // block_factor,
        block_factor,
        array.shape[1] // block_factor,
        block_factor,
    ).mean(axis=(1, 3))


def system_sha256(system: PrototypeElectricalSystem) -> str:
    """Hash the realized CSR operator, boundary vectors, and grid identity."""

    digest = hashlib.sha256()
    for value in (
        system.full_matrix_S.indptr.astype("<i8", copy=False),
        system.full_matrix_S.indices.astype("<i8", copy=False),
        system.full_matrix_S.data.astype("<f8", copy=False),
        system.fixed.astype("<i8", copy=False),
        system.fixed_values_V.astype("<f8", copy=False),
        system.binary_mask.astype(np.uint8, copy=False),
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(str(system.refinement_factor).encode("ascii"))
    digest.update(np.float64(system.electrical_contact_S_m2).tobytes())
    digest.update(system.void_model.encode("ascii"))
    return digest.hexdigest()
