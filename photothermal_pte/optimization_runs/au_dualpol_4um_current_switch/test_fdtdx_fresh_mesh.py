from __future__ import annotations

import inspect
import unittest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_mesh import (
    mesh_context,
)


class FreshMeshBridgeTest(unittest.TestCase):
    def test_context_installs_and_restores_exact_local_refinement(self) -> None:
        from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
            fdtdx_4um_model as optical_model,
        )

        original_layout = optical_model.LAYOUT
        original_edges_function = optical_model.grid_edges
        spec = MeshSpec(design_xy_factor=2, outer_xy_factor=1, z_factor=4)
        with mesh_context(spec) as audit:
            self.assertEqual(optical_model.LAYOUT.au_xy_cells, 160)
            self.assertEqual(optical_model.LAYOUT.flake_xy_cells, 240)
            self.assertEqual(optical_model.LAYOUT.source_xy_cells, 240)
            self.assertEqual(optical_model.LAYOUT.non_pml_xy_cells, 260)
            edges = optical_model.grid_edges()
            self.assertEqual([len(axis) - 1 for axis in edges], [276, 276, 160])
            self.assertEqual(audit["grid_shape_xyz"], [276, 276, 160])
            self.assertAlmostEqual(edges[0][58], -4.0e-6)
            self.assertAlmostEqual(edges[0][218], 4.0e-6)
        self.assertIs(optical_model.LAYOUT, original_layout)
        self.assertIs(optical_model.grid_edges, original_edges_function)

    def test_historical_builder_accepts_but_does_not_require_explicit_pml(self) -> None:
        from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
            fdtdx_4um_model as optical_model,
        )

        parameter = inspect.signature(optical_model.build_model).parameters[
            "pml_face_parameters"
        ]
        self.assertIsNone(parameter.default)


if __name__ == "__main__":
    unittest.main()
