import numpy as np
from scipy import sparse

from photothermal_pte.finite_inverse_design.yee_material_jacobian import (
    SparseYeeMaterialJacobian,
)


def test_sparse_complex_component_jacobian_transpose():
    rng = np.random.default_rng(911)
    matrices = {
        component: sparse.random(
            24,
            12,
            density=0.2,
            dtype=np.complex128,
            random_state=rng,
            data_rvs=lambda size: (
                rng.normal(size=size) + 1j * rng.normal(size=size)
            ),
        )
        for component in "xyz"
    }
    operator = SparseYeeMaterialJacobian(
        density_shape=(3, 4),
        component_shapes={component: (2, 3, 4) for component in "xyz"},
        matrices=matrices,
    )
    direction = rng.normal(size=(3, 4))
    tangent = operator.jvp(direction)
    cotangent = {
        component: (
            rng.normal(size=(2, 3, 4))
            + 1j * rng.normal(size=(2, 3, 4))
        )
        for component in "xyz"
    }
    left = np.real(
        sum(
            np.sum(cotangent[component] * tangent[component])
            for component in "xyz"
        )
    )
    right = np.vdot(direction, operator.vjp(cotangent))
    assert abs(left - right) / max(abs(left), abs(right)) < 1.0e-12
