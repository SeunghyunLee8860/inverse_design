from __future__ import annotations

import unittest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    CURRENT_SIGN_GUARD_A,
    MeshSpec,
    axis_levels,
    campaign_contract,
    endpoint_sign_gate,
    grid_edges,
    layout,
    mask_audit,
    mesh_audit,
    pml_parameters,
    reference_mask,
    upsample_mask,
)


class ExactBinaryConvergenceContractTest(unittest.TestCase):
    def test_reference_masks_are_binary_and_have_stable_occupancy(self) -> None:
        expected = {
            "empty": 0,
            "full_design_window": 6400,
            "centered_square_2um": 400,
            "x_bar_4um_by_1um": 400,
            "y_bar_1um_by_4um": 400,
            "l_shape_4um_with_1um_arms": 700,
        }
        hashes = set()
        for name, solid_cells in expected.items():
            audit = mask_audit(name)
            self.assertTrue(audit["binary"])
            self.assertEqual(audit["solid_cells"], solid_cells)
            hashes.add(audit["mask_sha256"])
        self.assertEqual(len(hashes), len(expected))

    def test_piecewise_constant_upsampling_preserves_geometry_and_area(self) -> None:
        mask = reference_mask("l_shape_4um_with_1um_arms")
        fine = upsample_mask(mask, 4)
        self.assertEqual((len(fine), len(fine[0])), (320, 320))
        self.assertEqual(sum(value for row in fine for value in row), 700 * 16)
        for x_index in range(80):
            for y_index in range(80):
                block = {
                    fine[4 * x_index + di][4 * y_index + dj]
                    for di in range(4)
                    for dj in range(4)
                }
                self.assertEqual(block, {mask[x_index][y_index]})

    def test_local_xy_and_full_z_factors_are_independent(self) -> None:
        anchor = MeshSpec()
        audit = mesh_audit(anchor)
        self.assertEqual(audit["grid_shape_xyz"], [196, 196, 160])
        self.assertEqual(audit["yee_cell_count"], 6_146_560)
        self.assertAlmostEqual(audit["design_solver_pitch_m"], 100.0e-9)
        self.assertAlmostEqual(audit["outer_solver_pitch_m"], 100.0e-9)

        local_four = mesh_audit(
            MeshSpec(design_xy_factor=4, outer_xy_factor=1, z_factor=4)
        )
        self.assertEqual(local_four["grid_shape_xyz"], [436, 436, 160])
        self.assertEqual(local_four["yee_cell_count"], 30_415_360)
        self.assertAlmostEqual(local_four["design_solver_pitch_m"], 25.0e-9)
        self.assertAlmostEqual(local_four["outer_solver_pitch_m"], 100.0e-9)

        z_eight = mesh_audit(MeshSpec(z_factor=8))
        self.assertEqual(z_eight["grid_shape_xyz"], [196, 196, 320])
        self.assertEqual(z_eight["yee_cell_count"], 12_293_120)
        self.assertNotEqual(audit["grid_contract_sha256"], z_eight["grid_contract_sha256"])

    def test_all_required_physical_edges_and_layout_indices_are_exact(self) -> None:
        spec = MeshSpec(design_xy_factor=2, outer_xy_factor=1, z_factor=4)
        x, y, z = grid_edges(spec)
        for required in (-8e-6, -4e-6, 4e-6, 8e-6):
            self.assertTrue(any(abs(value - required) <= 2e-18 for value in x))
            self.assertTrue(any(abs(value - required) <= 2e-18 for value in y))
        for required in (-0.385e-6, -0.100e-6, 0.0, 0.050e-6):
            self.assertTrue(any(abs(value - required) <= 2e-18 for value in z))
        value = layout(spec)
        self.assertEqual(value["au_xy_cells"], 160)
        self.assertEqual(value["flake_xy_cells"], 240)
        self.assertEqual(value["source_xy_cells"], 240)
        self.assertEqual(value["non_pml_xy_cells"], 260)

    def test_each_axis_ladder_changes_only_its_named_axis(self) -> None:
        anchor = MeshSpec()
        design = axis_levels("design_xy", anchor)
        self.assertEqual([item.design_xy_factor for item in design], [1, 2, 4])
        self.assertEqual({item.z_factor for item in design}, {anchor.z_factor})
        vertical = axis_levels("full_domain_z", anchor)
        self.assertEqual([item.z_factor for item in vertical], [2, 4, 8])
        self.assertEqual(
            {item.design_xy_factor for item in vertical},
            {anchor.design_xy_factor},
        )

    def test_pml_profile_is_explicitly_tied_to_4um(self) -> None:
        profile = pml_parameters(1.0e-6)
        expected_alpha = (
            0.01 * 2.0 * 3.141592653589793 * 299_792_458.0 / 4.0e-6
            * 8.854_187_812_8e-12
        )
        self.assertAlmostEqual(profile["alpha_start"], expected_alpha)
        self.assertEqual(profile["alpha_reference_wavelength_m"], 4.0e-6)
        self.assertGreater(profile["sigma_end"], 0.0)

    def test_sign_gate_has_requested_orientation_and_guard_band(self) -> None:
        self.assertTrue(endpoint_sign_gate(2.0e-9, -3.0e-9)["pass"])
        self.assertFalse(endpoint_sign_gate(-2.0e-9, 3.0e-9)["pass"])
        self.assertFalse(
            endpoint_sign_gate(0.5 * CURRENT_SIGN_GUARD_A, -3.0e-9)["pass"]
        )

    def test_campaign_is_fail_closed_before_any_solver_run(self) -> None:
        contract = campaign_contract()
        self.assertFalse(contract["historical_optimizer_may_resume"])
        self.assertFalse(contract["gray_density_allowed_in_reference_campaign"])
        self.assertFalse(contract["promotion"]["is_mesh_certificate"])
        self.assertFalse(contract["promotion"]["optimizer_start_allowed"])
        self.assertTrue(contract["rules"]["two_successive_pair_comparisons_required"])


if __name__ == "__main__":
    unittest.main()
