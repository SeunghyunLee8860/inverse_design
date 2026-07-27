"""Explicit component-wise density-to-Yee material Jacobians."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import sparse


COMPONENTS = "xyz"


class SparseYeeMaterialJacobian:
    """Store Jx/Jy/Jz and expose matrix-free JVP/VJP operations."""

    def __init__(
        self,
        *,
        density_shape: tuple[int, int],
        component_shapes: Mapping[str, tuple[int, int, int]],
        matrices: Mapping[str, sparse.spmatrix],
    ) -> None:
        self.density_shape = tuple(density_shape)
        self.component_shapes = {
            component: tuple(component_shapes[component])
            for component in COMPONENTS
        }
        parameter_count = int(np.prod(self.density_shape))
        self.matrices = {}
        for component in COMPONENTS:
            matrix = sparse.csr_matrix(matrices[component])
            expected = (
                int(np.prod(self.component_shapes[component])),
                parameter_count,
            )
            if matrix.shape != expected:
                raise ValueError(
                    f"{component} J shape {matrix.shape} != {expected}"
                )
            self.matrices[component] = matrix

    def jvp(self, direction: np.ndarray) -> dict[str, np.ndarray]:
        vector = np.asarray(direction, float)
        if vector.shape != self.density_shape:
            raise ValueError("JVP direction shape differs from density")
        return {
            component: (
                self.matrices[component] @ vector.reshape(-1)
            ).reshape(self.component_shapes[component])
            for component in COMPONENTS
        }

    def vjp(self, cotangent: Mapping[str, np.ndarray]) -> np.ndarray:
        """Transpose under Re(sum(cotangent_c * J_c direction))."""

        result = np.zeros(int(np.prod(self.density_shape)), float)
        for component in COMPONENTS:
            weight = np.asarray(cotangent[component])
            if weight.shape != self.component_shapes[component]:
                raise ValueError(f"{component} cotangent shape differs")
            result += np.real(
                self.matrices[component].transpose()
                @ weight.reshape(-1)
            )
        return result.reshape(self.density_shape)
