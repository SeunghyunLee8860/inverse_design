from __future__ import annotations

import unittest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_pml import (
    PML_FACES,
    SOLVER_PARAMETER_NAMES,
    boundary_config_kwargs,
    face_parameters,
    solver_parameters,
)


class FreshPmlContractTest(unittest.TestCase):
    def test_every_face_has_a_complete_explicit_4um_profile(self) -> None:
        profiles = face_parameters(MeshSpec())
        self.assertEqual(set(profiles), set(PML_FACES))
        clean = solver_parameters(profiles)
        for face in PML_FACES:
            self.assertEqual(set(clean[face]), set(SOLVER_PARAMETER_NAMES))
            self.assertEqual(profiles[face]["alpha_reference_wavelength_m"], 4.0e-6)
        self.assertEqual(clean["minx"]["alpha_start"], clean["minz"]["alpha_start"])
        self.assertNotEqual(clean["minx"]["sigma_end"], clean["minz"]["sigma_end"])

    def test_boundary_keywords_are_expanded_without_metadata(self) -> None:
        values = boundary_config_kwargs(face_parameters(MeshSpec()))
        self.assertEqual(len(values), len(PML_FACES) * len(SOLVER_PARAMETER_NAMES))
        self.assertIn("alpha_start_minx", values)
        self.assertIn("sigma_end_maxz", values)
        self.assertNotIn("target_reflection_minx", values)
        self.assertNotIn("alpha_reference_wavelength_m_minx", values)

    def test_missing_face_or_solver_parameter_fails_closed(self) -> None:
        profiles = face_parameters(MeshSpec())
        profiles.pop("maxz")
        with self.assertRaises(ValueError):
            solver_parameters(profiles)
        profiles = face_parameters(MeshSpec())
        profiles["minx"].pop("sigma_end")
        with self.assertRaises(ValueError):
            solver_parameters(profiles)


if __name__ == "__main__":
    unittest.main()
