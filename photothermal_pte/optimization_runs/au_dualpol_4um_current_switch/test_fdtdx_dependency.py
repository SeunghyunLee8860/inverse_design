from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_dependency import (
    LOCK_PATH,
    audit_source,
    load_lock,
)


class FdtdxDependencyTest(unittest.TestCase):
    def test_lock_has_full_commit_tree_and_critical_implementations(self) -> None:
        lock = load_lock()
        self.assertEqual(len(lock["commit"]), 40)
        self.assertEqual(len(lock["tree"]), 40)
        files = set(lock["critical_files_sha256"])
        self.assertIn("src/fdtdx/fdtd/update.py", files)
        self.assertIn(
            "src/fdtdx/objects/boundaries/perfectly_matched_layer.py", files
        )
        self.assertIn("src/fdtdx/objects/static_material/static.py", files)
        self.assertEqual(json.loads(LOCK_PATH.read_text())["license"], "MIT")

    def test_missing_source_fails_closed_without_importing_fdtdx(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "not-cloned"
            result = audit_source(missing)
        self.assertFalse(result["ready"])
        self.assertEqual(result["status"], "BLOCKED_PINNED_FDTDX_SOURCE_MISSING")
        self.assertFalse(result["checks"]["source_directory_exists"])


if __name__ == "__main__":
    unittest.main()
