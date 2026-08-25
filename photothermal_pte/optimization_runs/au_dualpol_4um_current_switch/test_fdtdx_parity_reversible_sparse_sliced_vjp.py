from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_reversible_sliced_vjp import (
    reversible_ade_cpml_phasor_sliced_fdtd_prototype,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_reversible_sparse_sliced_vjp import (
    reversible_ade_cpml_phasor_sparse_sliced_fdtd,
    reversible_sparse_slice_checkpoint_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_sparse_ade_support import (
    sparse_ade_coefficient_support_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.test_fdtdx_parity_reversible_cpml_detector_vjp import (
    _small_cpml_phasor_scene,
)


def test_sparse_sliced_vjp_matches_full_slice_regional_c3_gradient() -> None:
    _, jax, objects, arrays, config, key = _small_cpml_phasor_scene()
    region = (slice(3, 9), slice(3, 9), slice(4, 8))
    regions = (region,)
    c3 = arrays.dispersive_c3
    assert c3 is not None
    prefix = (slice(None), slice(None))
    regional_c3 = c3[prefix + region]
    support_audit = sparse_ade_coefficient_support_audit(
        arrays,
        regions=regions,
        jax_module=jax,
    )
    assert support_audit["status"] == "PASS"
    checkpoint_audit = reversible_sparse_slice_checkpoint_audit(
        arrays,
        regions=regions,
        jax_module=jax,
    )
    assert checkpoint_audit["status"] == "PASS"
    assert (
        checkpoint_audit["sparse_slice_checkpoint_bytes"]
        < checkpoint_audit["full_slice_checkpoint_bytes"]
    )
    assert checkpoint_audit["detector_state_emitted_per_slice"] is False

    def arrays_for_regional(value):
        return arrays.aset(
            "dispersive_c3",
            c3.at[prefix + region].set(value),
        )

    def detector_loss(output):
        phasor = output.detector_states["late_slab"]["phasor"]
        return jax.numpy.mean(jax.numpy.square(jax.numpy.abs(phasor)))

    def full_loss(value):
        _, output = reversible_ade_cpml_phasor_sliced_fdtd_prototype(
            arrays=arrays_for_regional(value),
            objects=objects,
            config=config,
            key=key,
            steps_per_slice=6,
        )
        return detector_loss(output)

    def sparse_loss(value):
        _, output = reversible_ade_cpml_phasor_sparse_sliced_fdtd(
            arrays=arrays_for_regional(value),
            objects=objects,
            config=config,
            key=key,
            steps_per_slice=6,
            regions=regions,
            support_audit=support_audit,
        )
        return detector_loss(output)

    full_value, full_gradient = jax.value_and_grad(full_loss)(regional_c3)
    sparse_value, sparse_gradient = jax.value_and_grad(sparse_loss)(regional_c3)
    assert float(sparse_value) == pytest.approx(
        float(full_value), rel=1.0e-6, abs=1.0e-10
    )
    np.testing.assert_allclose(
        np.asarray(sparse_gradient),
        np.asarray(full_gradient),
        rtol=5.0e-4,
        atol=3.0e-10,
    )
    assert np.count_nonzero(np.asarray(sparse_gradient)) > 0
