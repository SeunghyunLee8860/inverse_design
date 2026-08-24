from __future__ import annotations

from dataclasses import asdict
import unittest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    DOWNSTREAM_GATES_NOT_ACTIVE,
    OPTICAL_PAIR_GATES,
    MeshSpec,
    axis_levels,
    campaign_contract,
    evaluate_pair,
    grid_edges,
    mask_audit,
    mesh_audit,
    reference_mask,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_pml import (
    face_parameters,
)


class FreshConvergenceV2Test(unittest.TestCase):
    def test_v2_anchor_is_bitwise_geometry_compatible_with_endpoint_controls(self) -> None:
        audit = mesh_audit(MeshSpec())
        self.assertEqual(audit["grid_shape_xyz"], [196, 196, 160])
        self.assertEqual(audit["yee_cell_count"], 6_146_560)
        self.assertEqual(audit["bounds_m"], [[-10e-6, 10e-6]] * 2 + [[-3e-6, 3e-6]])
        self.assertAlmostEqual(audit["z_pml_solver_pitch_m"], 50e-9)
        self.assertAlmostEqual(audit["bottom_si_solver_pitch_m"], 50.75e-9)
        self.assertAlmostEqual(audit["source_air_solver_pitch_m"], 54.16666666666667e-9)

    def test_every_vertical_extent_ladder_changes_only_its_named_field(self) -> None:
        anchor = MeshSpec()
        expected = {
            "bottom_si_buffer": (
                "bottom_si_buffer_m",
                [1.015e-6, 2.030e-6, 3.045e-6],
                [160, 180, 200],
            ),
            "top_source_to_pml_gap": (
                "top_source_to_pml_gap_m",
                [0.650e-6, 1.300e-6, 1.950e-6],
                [160, 172, 184],
            ),
            "z_pml_thickness": (
                "z_pml_thickness_m",
                [1.600e-6, 2.400e-6, 3.200e-6],
                [160, 192, 224],
            ),
        }
        anchor_values = asdict(anchor)
        for axis, (field, values, z_cells) in expected.items():
            levels = axis_levels(axis, anchor)
            self.assertEqual([getattr(level, field) for level in levels], values)
            self.assertEqual(
                [mesh_audit(level)["grid_shape_xyz"][2] for level in levels],
                z_cells,
            )
            for level in levels:
                changed = {
                    name
                    for name, value in asdict(level).items()
                    if value != anchor_values[name]
                }
                self.assertLessEqual(changed, {field})
                z_edges = grid_edges(level)[2]
                for physical_edge in (-0.385e-6, -0.100e-6, 0.0, 0.050e-6, 0.750e-6):
                    self.assertTrue(
                        any(abs(value - physical_edge) <= 2e-18 for value in z_edges)
                    )

    def test_z_pml_profiles_follow_each_mesh_spec_physical_thickness(self) -> None:
        anchor = face_parameters(MeshSpec())
        thicker = face_parameters(MeshSpec(z_pml_thickness_m=2.4e-6))
        self.assertEqual(anchor["minx"]["sigma_end"], thicker["minx"]["sigma_end"])
        self.assertNotEqual(anchor["minz"]["sigma_end"], thicker["minz"]["sigma_end"])
        self.assertAlmostEqual(
            thicker["minz"]["sigma_end"] / anchor["minz"]["sigma_end"],
            1.6 / 2.4,
        )
        self.assertEqual(thicker["minz"], thicker["maxz"])

    def test_primary_and_gap_stress_references_are_exact_500nm_binary_masks(self) -> None:
        l_name = "l_shape_4um_with_500nm_arms"
        gap_name = "parallel_bars_4um_by_500nm_with_500nm_gap"
        self.assertEqual(mask_audit(l_name)["solid_cells"], 375)
        self.assertEqual(mask_audit(gap_name)["solid_cells"], 400)
        self.assertEqual(mask_audit(l_name)["policy"]["minimum_feature_m"], 500e-9)
        self.assertEqual(mask_audit(gap_name)["policy"]["minimum_gap_m"], 500e-9)
        gap_mask = reference_mask(gap_name)
        for x_index in range(20, 60):
            self.assertEqual([gap_mask[x_index][y] for y in range(25, 30)], [0] * 5)
        for name in (l_name, gap_name):
            self.assertEqual(
                {value for row in reference_mask(name) for value in row},
                {0, 1},
            )

    def test_optical_pair_evaluator_cannot_accept_thermal_or_current_metrics(self) -> None:
        passing = {name: 0.0 for name in OPTICAL_PAIR_GATES}
        passing.update(
            coarse_current_A=-1e9,
            fine_current_A=1e9,
            Ta_temperature_NRMSE=1e9,
        )
        result = evaluate_pair(passing)
        self.assertTrue(result["pass"])
        self.assertEqual(set(result["checks"]), set(OPTICAL_PAIR_GATES))
        self.assertTrue(set(OPTICAL_PAIR_GATES).isdisjoint(DOWNSTREAM_GATES_NOT_ACTIVE))

    def test_campaign_is_staged_fail_closed_and_serializes_all_missing_axes(self) -> None:
        contract = campaign_contract()
        self.assertEqual(
            contract["reference_execution"]["primary_full_ladder"],
            "l_shape_4um_with_500nm_arms",
        )
        self.assertEqual(
            contract["time_convergence"]["settling_ladder"]["total_periods"],
            [16, 24, 32],
        )
        self.assertEqual(
            contract["time_convergence"]["courant_ladder_after_settling"]["courant_factors"],
            [0.5, 0.375, 0.25, 0.1875],
        )
        self.assertTrue(
            {
                "bottom_si_buffer",
                "top_source_to_pml_gap",
                "z_pml_thickness",
                "lateral_pml_thickness",
            }
            <= set(contract["axis_ladders"])
        )
        self.assertTrue(
            contract["comparison_contract"]["complex_E"][
                "array_index_comparison_forbidden"
            ]
        )
        self.assertFalse(contract["promotion"]["is_mesh_certificate"])
        self.assertFalse(contract["promotion"]["optimizer_start_allowed"])
        self.assertFalse(contract["rules"]["thermal_or_current_metrics_may_certify_optical_mesh"])


if __name__ == "__main__":
    unittest.main()
