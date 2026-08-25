from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_reversible_design_sliced_vjp import (
    reversible_ade_cpml_phasor_design_sliced_fdtd,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_sparse_ade_support import (
    sparse_ade_coefficient_support_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.test_fdtdx_parity_reversible_sliced_long import (
    _long_cpml_phasor_scene,
)


def test_long_design_only_vjp_matches_direct_with_partial_final_slice() -> None:
    fdtdx, jax, objects, arrays, config, key = _long_cpml_phasor_scene()
    design_region = (slice(3, 9), slice(3, 9), slice(4, 8))
    regions = (design_region,)
    c3 = arrays.dispersive_c3
    assert c3 is not None
    prefix = (slice(None), slice(None))
    design_c3 = c3[prefix + design_region]
    support_audit = sparse_ade_coefficient_support_audit(
        arrays,
        regions=regions,
        jax_module=jax,
    )
    assert support_audit["status"] == "PASS"

    def arrays_for_design(value):
        return arrays.aset(
            "dispersive_c3",
            c3.at[prefix + design_region].set(value),
        )

    def detector_loss(output):
        phasor = output.detector_states["late_slab"]["phasor"]
        return jax.numpy.mean(jax.numpy.square(jax.numpy.abs(phasor)))

    def direct_loss(value):
        state = (
            jax.numpy.asarray(0, dtype=jax.numpy.int32),
            arrays_for_design(value),
        )
        for _ in range(config.time_steps_total):
            state = fdtdx.fdtd.forward.forward(
                state=state,
                config=config,
                objects=objects,
                key=key,
                record_detectors=True,
                record_boundaries=False,
                simulate_boundaries=True,
            )
        return detector_loss(state[1])

    def design_loss(value):
        _, output = reversible_ade_cpml_phasor_design_sliced_fdtd(
            arrays=arrays_for_design(value),
            objects=objects,
            config=config,
            key=key,
            steps_per_slice=16,
            regions=regions,
            design_region=design_region,
            support_audit=support_audit,
        )
        return detector_loss(output)

    direct_value, direct_gradient = jax.value_and_grad(direct_loss)(design_c3)
    design_value, design_gradient = jax.value_and_grad(design_loss)(design_c3)
    assert float(design_value) == pytest.approx(
        float(direct_value), rel=1.0e-6, abs=1.0e-10
    )
    np.testing.assert_allclose(
        np.asarray(design_gradient),
        np.asarray(direct_gradient),
        rtol=1.0e-3,
        atol=3.0e-10,
    )
    assert np.count_nonzero(np.asarray(design_gradient)) > 0
