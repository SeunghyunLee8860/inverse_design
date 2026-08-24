from __future__ import annotations

from pathlib import Path
import unittest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_runtime_preflight import (
    load_runtime_lock,
    package_version_checks,
    parse_single_gpu_index,
    path_is_within,
)


class FdtdxRuntimePreflightTest(unittest.TestCase):
    def test_lock_contains_exact_cuda_jax_and_fdtdx_versions(self) -> None:
        lock = load_runtime_lock()
        self.assertEqual(lock["python_version"], [3, 12, 3])
        packages = lock["required_packages"]
        self.assertEqual(packages["fdtdx"], "0.6.2")
        self.assertEqual(packages["jax"], packages["jaxlib"])
        self.assertIn("jax-cuda13-plugin", packages)
        self.assertIn("nvidia-cudnn-cu13", packages)

    def test_package_checks_reject_missing_and_mismatched_versions(self) -> None:
        checks = package_version_checks(
            {"jax": "0.11.1", "fdtdx": "0.6.2"},
            {"jax": "0.11.0", "fdtdx": None},
        )
        self.assertEqual(checks, {"package:jax": False, "package:fdtdx": False})

    def test_exactly_one_numeric_physical_gpu_is_required(self) -> None:
        self.assertEqual(parse_single_gpu_index("7"), 7)
        for invalid in ("", "2,3", "GPU-uuid", "-1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_single_gpu_index(invalid)

    def test_import_path_must_be_below_pinned_source_package(self) -> None:
        root = Path("/dependencies/pinned/src/fdtdx")
        self.assertTrue(path_is_within(root / "__init__.py", root))
        self.assertFalse(path_is_within(Path("/venv/site-packages/fdtdx/__init__.py"), root))


if __name__ == "__main__":
    unittest.main()
