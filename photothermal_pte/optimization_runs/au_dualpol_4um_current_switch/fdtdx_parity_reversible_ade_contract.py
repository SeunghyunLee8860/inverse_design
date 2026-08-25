"""Discrete contract for a possible sliced reversible ADE adjoint.

This module does not enable reversible FDTDX.  It isolates the exact diagonal,
c4-free material substep used by the parity target and audits why the pinned
non-dispersive custom VJP cannot be enabled by deleting its dispersion guard.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_ade import (
    BASES,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_fixed_materials import (
    TA_A,
    TA_B,
)


def _source_sha256(obj: Any) -> str:
    path = Path(inspect.getsourcefile(obj) or "")
    if not path.is_file():
        raise RuntimeError(f"cannot resolve source file for {obj}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned_reversible_source_audit() -> dict[str, Any]:
    """Fail closed unless the inspected FDTDX limitation is still present."""

    from fdtdx.fdtd.fdtd import reversible_fdtd
    from fdtdx.fdtd.forward import forward_single_args_wrapper
    from fdtdx.fdtd.update import update_E_reverse

    reversible_source = inspect.getsource(reversible_fdtd)
    reverse_e_source = inspect.getsource(update_E_reverse)
    wrapper_parameters = tuple(inspect.signature(forward_single_args_wrapper).parameters)
    missing_wrapper_state = tuple(
        name
        for name in (
            "dispersive_P_curr",
            "dispersive_P_prev",
            "dispersive_c1",
            "dispersive_c2",
            "dispersive_c3",
        )
        if name not in wrapper_parameters
    )
    checks = {
        "reversible_explicitly_rejects_dispersion": (
            "arrays.dispersive_c1 is not None" in reversible_source
            and "Use GradientConfig(method='checkpointed')" in reversible_source
        ),
        "reverse_E_explicitly_rejects_ADE_state": (
            "arrays.fields.dispersive_P_curr is not None" in reverse_e_source
            and "Use GradientConfig(method='checkpointed')" in reverse_e_source
        ),
        "wrapper_omits_all_ADE_state_and_coefficients": len(missing_wrapper_state)
        == 5,
    }
    return {
        "schema": "fdtdx_4um_parity_reversible_ADE_source_contract_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "forward_wrapper_parameters": list(wrapper_parameters),
        "missing_wrapper_state": list(missing_wrapper_state),
        "source_sha256": {
            "fdtd": _source_sha256(reversible_fdtd),
            "forward": _source_sha256(forward_single_args_wrapper),
            "update": _source_sha256(update_E_reverse),
        },
        "guard_removal_is_valid_implementation": False,
        "required_custom_vjp_outputs": [
            "dispersive_P_curr",
            "dispersive_P_prev",
            "dispersive_c3_cotangent",
        ],
    }


def forward_diagonal_c4_free_ade(
    *,
    E_prev: Any,
    P_curr: Any,
    P_prev: Any,
    inv_eps: Any,
    courant_curl_H: Any,
    c1: Any,
    c2: Any,
    c3: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Target material substep matching FDTDX's diagonal c4-free branch."""

    E_prev = np.asarray(E_prev)
    P_curr = np.asarray(P_curr)
    P_prev = np.asarray(P_prev)
    inv_eps = np.asarray(inv_eps)
    drive = np.asarray(courant_curl_H)
    P_next = np.asarray(c1) * P_curr + np.asarray(c2) * P_prev + np.asarray(c3) * E_prev
    E_next = E_prev + inv_eps * drive + inv_eps * np.sum(P_curr - P_next, axis=0)
    return E_next, P_next, P_curr


def reverse_diagonal_c4_free_ade(
    *,
    E_next: Any,
    P_next: Any,
    P_curr: Any,
    inv_eps: Any,
    courant_curl_H: Any,
    c1: Any,
    c2: Any,
    c3: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Algebraically invert the target material substep.

    Zero-c2 entries are allowed only when their recurrence numerator is exactly
    zero; this is the padded non-dispersive/inactive-pole state.  A safe divisor
    avoids evaluating a divide by zero in either NumPy or a future JAX ``where``.
    """

    E_next = np.asarray(E_next)
    P_next = np.asarray(P_next)
    P_curr = np.asarray(P_curr)
    inv_eps = np.asarray(inv_eps)
    drive = np.asarray(courant_curl_H)
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    c3 = np.asarray(c3)
    E_prev = E_next - inv_eps * drive - inv_eps * np.sum(P_curr - P_next, axis=0)
    numerator = P_next - c1 * P_curr - c3 * E_prev
    active = c2 != 0
    inactive_residual = np.where(active, 0, numerator)
    if np.any(inactive_residual != 0):
        raise ValueError("zero-c2 ADE entry has nonzero reverse recurrence residual")
    safe_c2 = np.where(active, c2, np.ones((), dtype=c2.dtype))
    P_prev = np.where(active, numerator / safe_c2, np.zeros((), dtype=numerator.dtype))
    return E_prev, P_curr, P_prev


def target_reverse_gain_audit() -> dict[str, Any]:
    """Report the unavoidable inverse damping of each frozen recurrence."""

    coefficients = {
        **{f"Au:{basis.name}": float(np.float32(basis.c2)) for basis in BASES},
        "TaIrTe4:a": float(np.float32(TA_A.c2)),
        "TaIrTe4:b/c": float(np.float32(TA_B.c2)),
    }
    rows = {
        name: {
            "c2": value,
            "single_step_inverse_c2_magnitude": 1.0 / abs(value),
            "conservative_4096_step_power": math.exp(
                4096.0 * -math.log(abs(value))
            ),
        }
        for name, value in coefficients.items()
    }
    maximum = max(
        row["single_step_inverse_c2_magnitude"] for row in rows.values()
    )
    return {
        "schema": "fdtdx_4um_parity_reversible_ADE_gain_v1",
        "status": "PASS_REQUIRES_SLICED_RESETS",
        "coefficients": rows,
        "maximum_single_step_inverse_c2_magnitude": maximum,
        "unsliced_256163_step_reverse_allowed": False,
        "sliced_exact_state_resets_required": True,
    }
