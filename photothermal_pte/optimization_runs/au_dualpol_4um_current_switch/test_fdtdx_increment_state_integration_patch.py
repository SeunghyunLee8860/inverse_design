from __future__ import annotations

import hashlib
from pathlib import Path
import re
import unittest


RUN_DIR = Path(__file__).resolve().parent
PATCH = (
    RUN_DIR
    / "fdtdx_patches/0002-feat-dispersion-integrate-opt-in-increment-state-ADE.patch"
)
EXPECTED_SHA256 = "1532e032fbe3656b4397f6c8d94314339f4bd94b0e2583c162c317c106b901cb"
EXPECTED_COMMIT = "fc09ce54dc32ea13e27d2af799cdb3771801bf65"
EXPECTED_PATHS = {
    "src/fdtdx/config.py",
    "src/fdtdx/dispersion.py",
    "src/fdtdx/fdtd/container.py",
    "src/fdtdx/fdtd/initialization.py",
    "src/fdtdx/fdtd/update.py",
    "src/fdtdx/increment_state.py",
    "src/fdtdx/materials.py",
    "src/fdtdx/objects/detectors/mode.py",
    "src/fdtdx/objects/sources/dipole.py",
    "src/fdtdx/objects/sources/linear_polarization.py",
    "src/fdtdx/objects/sources/mode.py",
    "src/fdtdx/objects/sources/tfsf.py",
    "src/fdtdx/objects/sources/tfsf_region.py",
    "tests/simulation/fdtd/test_increment_state_fdtd.py",
    "tests/simulation/fdtd/test_time_reversal.py",
    "tests/unit/test_increment_state_spectrum.py",
}


class IncrementStateIntegrationPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = PATCH.read_bytes()
        cls.text = cls.payload.decode("utf-8")

    def test_patch_hash_is_pinned(self) -> None:
        self.assertEqual(hashlib.sha256(self.payload).hexdigest(), EXPECTED_SHA256)

    def test_patch_commit_is_pinned(self) -> None:
        self.assertTrue(self.text.startswith(f"From {EXPECTED_COMMIT} "))
        self.assertIn(
            "Subject: [PATCH] feat(dispersion): integrate opt-in increment-state ADE",
            self.text,
        )

    def test_patch_scope_is_exact(self) -> None:
        paths = set(
            re.findall(r"^diff --git a/(\S+) b/\1$", self.text, flags=re.MULTILINE)
        )
        self.assertEqual(paths, EXPECTED_PATHS)

    def test_patch_contains_required_production_gates(self) -> None:
        for marker in (
            'dispersive_state_representation: Literal["polarization", "increment"]',
            'if config.dispersive_state_representation == "increment":',
            "update_dispersive_increment_state(",
            "state_representation=state_representation",
            "increment-state ADE currently supports Lorentz/Drude poles only",
            "test_checkpointed_increment_b_gradient_matches_finite_difference",
        ):
            self.assertIn(marker, self.text)

    def test_patch_does_not_touch_lumerical(self) -> None:
        self.assertNotIn("lumerical", "\n".join(sorted(EXPECTED_PATHS)).lower())


if __name__ == "__main__":
    unittest.main()
