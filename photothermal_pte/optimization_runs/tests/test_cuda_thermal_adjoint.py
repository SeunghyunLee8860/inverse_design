from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


@unittest.skipUnless(cuda_available(), "CUDA is unavailable in this test environment")
class CudaThermalAdjointTest(unittest.TestCase):
    def test_forward_adjoint_reciprocity(self) -> None:
        from photothermal_pte.optimization_runs.cuda_thermal_adjoint import (
            solve_forward_adjoint_cuda,
        )

        matrix = sparse.csr_matrix(
            np.asarray(
                [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
            )
        )
        result = solve_forward_adjoint_cuda(
            matrix,
            np.asarray([1.0, 0.0, 1.0]),
            np.asarray([0.0, 1.0, 0.0]),
            relative_tolerance=1.0e-12,
        )
        self.assertLess(result.forward.explicit_relative_residual, 1.0e-12)
        self.assertLess(result.adjoint.explicit_relative_residual, 1.0e-12)
        self.assertLess(result.reciprocity_relative_error, 1.0e-12)

    def test_literal_anisotropic_fvm_operator_pair(self) -> None:
        from photothermal_pte.optimization_runs.cuda_thermal_adjoint import (
            solve_forward_adjoint_cuda,
        )
        from photothermal_pte.validation.photothermal_stage1.anisotropic_heat_fvm import (
            assemble_steady_diagonal_kappa,
        )

        edges = np.linspace(-0.5e-6, 0.5e-6, 9)
        shape = (8, 8, 8)
        kappa = np.empty((*shape, 3), float)
        kappa[..., 0] = 14.4
        kappa[..., 1] = 3.8
        kappa[..., 2] = 1.0
        system = assemble_steady_diagonal_kappa(
            x_edges_m=edges,
            y_edges_m=edges,
            z_edges_m=edges,
            kappa_W_mK=kappa,
            dirichlet_temperature_K={"z_min": 300.0, "z_max": 300.0},
        )
        source = np.zeros(shape, float)
        source[3:5, 3:5, 3:5] = 1.0e15
        active_source = system.active_source(source)
        forward_rhs = np.asarray(
            system.source_volume_operator_m3 @ active_source
            + system.boundary_load_W
        )
        # Keep the reciprocity signal away from the exact antisymmetric /
        # symmetric cancellation that would make a relative error ill posed.
        adjoint_rhs = np.linspace(0.1, 1.1, forward_rhs.size)
        result = solve_forward_adjoint_cuda(
            system.matrix_W_K,
            forward_rhs,
            adjoint_rhs,
            relative_tolerance=1.0e-10,
        )
        self.assertLess(result.forward.explicit_relative_residual, 1.0e-10)
        self.assertLess(result.adjoint.explicit_relative_residual, 1.0e-10)
        self.assertLess(result.reciprocity_relative_error, 1.0e-9)


if __name__ == "__main__":
    unittest.main()
