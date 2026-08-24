from __future__ import annotations

import os
from unittest import mock
import unittest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_runtime_preflight as runtime,
)


class FdtdxRuntimeFailClosedTest(unittest.TestCase):
    def test_external_compute_process_prevents_all_solver_imports(self) -> None:
        lock = runtime.load_runtime_lock()
        gpu = {
            "physical_index": 2,
            "uuid": "GPU-occupied",
            "name": lock["host_observation"]["gpu_model"],
            "driver_version": lock["host_observation"]["nvidia_driver"],
            "memory_used_mib_before_jax": 1024,
            "memory_total_mib": 183359,
            "utilization_percent_before_jax": 99,
            "compute_applications_before_jax": [
                {
                    "gpu_uuid": "GPU-occupied",
                    "pid": os.getpid() + 1000,
                    "process_name": "fdtd-engine",
                    "used_memory_mib": 1024,
                }
            ],
        }
        environment = {
            "CUDA_VISIBLE_DEVICES": "2",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
        with (
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch.object(runtime.sys, "prefix", "/test/venv"),
            mock.patch.object(runtime.sys, "base_prefix", "/usr"),
            mock.patch.object(
                runtime,
                "installed_versions",
                return_value=dict(lock["required_packages"]),
            ),
            mock.patch.object(runtime, "require_source", return_value={"actual": {}}),
            mock.patch.object(runtime, "gpu_observation", return_value=gpu),
            mock.patch.object(runtime, "import_module") as import_module,
        ):
            result = runtime.audit_runtime(runtime.configured_source(), 2)

        self.assertFalse(result["ready"])
        self.assertFalse(result["checks"]["gpu_has_no_external_compute_process"])
        self.assertFalse(result["checks"]["safe_to_initialize_jax"])
        self.assertIn("CUDA initialization skipped", result["errors"][-1])
        import_module.assert_not_called()


if __name__ == "__main__":
    unittest.main()
