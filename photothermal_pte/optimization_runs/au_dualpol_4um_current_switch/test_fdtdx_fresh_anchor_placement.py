from __future__ import annotations

import unittest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_anchor_placement import (
    expected_placement,
    expected_pml_slices,
)


class FdtdxFreshAnchorPlacementTest(unittest.TestCase):
    def test_anchor_material_and_monitor_slices_are_exact(self) -> None:
        placement = expected_placement(MeshSpec())
        self.assertEqual(placement["fixed_silicon_substrate"], [[0, 196], [0, 196], [0, 52]])
        self.assertEqual(placement["fixed_285nm_sio2"], [[0, 196], [0, 196], [52, 64]])
        self.assertEqual(placement["fixed_tairte4"], [[18, 178], [18, 178], [64, 84]])
        self.assertEqual(placement["au_design"], [[58, 138], [58, 138], [84, 92]])
        self.assertEqual(placement["gaussian_source"][2], [116, 117])
        self.assertEqual(placement["incident_plane"][2], [112, 113])
        self.assertEqual(placement["target_field"][2], [108, 109])
        self.assertEqual(placement["material_flux"], [[8, 188], [8, 188], [48, 108]])

    def test_pml_slices_follow_independent_lateral_and_z_factors(self) -> None:
        spec = MeshSpec(pml_xy_factor=2, z_factor=8)
        pml = expected_pml_slices(spec)
        self.assertEqual(pml["minx"], [[0, 16], [0, 212], [0, 320]])
        self.assertEqual(pml["maxy"], [[0, 212], [196, 212], [0, 320]])
        self.assertEqual(pml["minz"], [[0, 212], [0, 212], [0, 64]])
        self.assertEqual(pml["maxz"], [[0, 212], [0, 212], [256, 320]])


if __name__ == "__main__":
    unittest.main()
