from __future__ import annotations

import unittest

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_only import (
    extract_detector_fields,
)


class FdtdxSourceOnlyDetectorSchemaTest(unittest.TestCase):
    def _states(self):
        phasor = np.zeros((1, 1, 3, 2, 2, 1), dtype=np.complex64)
        return {
            "au_previous": {"phasor": phasor},
            "au_late": {"phasor": phasor},
            "tairte4_previous": {"phasor": phasor},
            "tairte4_late": {"phasor": phasor},
            "target_field": {"phasor": phasor},
            "incident_plane": {"phasor": phasor},
            "material_flux_td": {"poynting_flux": np.zeros((4, 1))},
            "material_flux": {
                f"phasor_axis{axis}_{side}": phasor
                for axis in range(3)
                for side in ("min", "max")
            },
        }

    def test_six_shell_faces_are_saved_under_distinct_raw_names(self) -> None:
        fields = extract_detector_fields(self._states())
        shell = sorted(name for name in fields if name.startswith("closed_phasor"))
        self.assertEqual(len(shell), 6)
        self.assertIn("closed_phasor_axis0_min", shell)
        self.assertIn("closed_phasor_axis2_max", shell)

    def test_legacy_single_phasor_key_is_rejected(self) -> None:
        states = self._states()
        states["material_flux"] = {"phasor": np.zeros((1,))}
        with self.assertRaises(RuntimeError):
            extract_detector_fields(states)


if __name__ == "__main__":
    unittest.main()
