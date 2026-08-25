from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_sparse_ade_support import (
    sparse_ade_coefficient_support_audit,
)


def _arrays(*, leak_outside: bool):
    shape = (1, 3, 4, 4, 4)
    region = (slice(1, 3), slice(1, 3), slice(1, 3))
    index = (slice(None), slice(None), *region)
    c1 = jnp.zeros(shape).at[index].set(0.5)
    c2 = jnp.zeros(shape).at[index].set(-0.2)
    c3 = jnp.zeros(shape).at[index].set(0.1)
    if leak_outside:
        c3 = c3.at[(0, 0, 0, 0, 0)].set(1.0e-6)
    return (
        SimpleNamespace(
            dispersive_c1=c1,
            dispersive_c2=c2,
            dispersive_c3=c3,
            dispersive_c4=None,
        ),
        region,
    )


def test_support_audit_passes_exact_regional_coefficients() -> None:
    arrays, region = _arrays(leak_outside=False)
    audit = sparse_ade_coefficient_support_audit(
        arrays, regions=(region,), jax_module=jax
    )
    assert audit["status"] == "PASS"
    assert all(
        value == 0.0
        for value in audit["maximum_abs_coefficient_outside_regions"].values()
    )
    assert audit["outside_dispersion_allowed"] is False


def test_support_audit_blocks_any_outside_c3() -> None:
    arrays, region = _arrays(leak_outside=True)
    audit = sparse_ade_coefficient_support_audit(
        arrays, regions=(region,), jax_module=jax
    )
    assert audit["status"] == "FAIL"
    assert audit["maximum_abs_coefficient_outside_regions"]["dispersive_c3"] > 0.0
