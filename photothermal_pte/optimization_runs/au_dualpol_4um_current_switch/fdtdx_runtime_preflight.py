"""Fail-closed runtime preflight for the fresh, source-pinned FDTDX route."""

from __future__ import annotations

import argparse
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_dependency import (
    configured_source,
    require_source,
)


HERE = Path(__file__).resolve().parent
RUNTIME_LOCK_PATH = HERE / "fdtdx_runtime_lock.json"


def load_runtime_lock() -> dict[str, Any]:
    value = json.loads(RUNTIME_LOCK_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("FDTDX runtime lock root must be an object")
    required = {"python_version", "required_packages", "host_observation"}
    missing = required - set(value)
    if missing:
        raise RuntimeError(f"FDTDX runtime lock is missing {sorted(missing)}")
    return value


def installed_versions(package_names: list[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            result[package_name] = version(package_name)
        except PackageNotFoundError:
            result[package_name] = None
    return result


def package_version_checks(
    expected: Mapping[str, str], actual: Mapping[str, str | None]
) -> dict[str, bool]:
    return {
        f"package:{name}": actual.get(name) == expected_version
        for name, expected_version in expected.items()
    }


def parse_single_gpu_index(value: str) -> int:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if len(tokens) != 1 or not tokens[0].isdigit():
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain exactly one non-negative physical GPU index"
        )
    return int(tokens[0])


def path_is_within(path: Path, directory: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
    except ValueError:
        return False
    return True


def _nvidia_smi(*arguments: str) -> str:
    completed = subprocess.run(
        ("nvidia-smi", *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def gpu_observation(physical_index: int) -> dict[str, Any]:
    row = _nvidia_smi(
        f"--id={physical_index}",
        "--query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    rows = [line.strip() for line in row.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one GPU row for index {physical_index}: {rows}")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != 7:
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {rows[0]!r}")
    index, uuid, name, driver, used, total, utilization = values

    applications_text = _nvidia_smi(
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    applications: list[dict[str, Any]] = []
    for line in applications_text.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) == 4 and fields[0] == uuid:
            applications.append(
                {
                    "gpu_uuid": fields[0],
                    "pid": int(fields[1]),
                    "process_name": fields[2],
                    "used_memory_mib": int(fields[3]),
                }
            )
    return {
        "physical_index": int(index),
        "uuid": uuid,
        "name": name,
        "driver_version": driver,
        "memory_used_mib_before_jax": int(used),
        "memory_total_mib": int(total),
        "utilization_percent_before_jax": int(utilization),
        "compute_applications_before_jax": applications,
    }


def _resolved_module_file(module: Any) -> Path:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError("imported module has no __file__")
    return Path(module_file).resolve()


def audit_runtime(source: Path, requested_gpu_index: int) -> dict[str, Any]:
    lock = load_runtime_lock()
    checks: dict[str, bool] = {}
    errors: list[str] = []
    actual: dict[str, Any] = {
        "python_version": list(sys.version_info[:3]),
        "python_executable": str(Path(sys.executable).resolve()),
        "virtual_environment": sys.prefix != sys.base_prefix,
    }
    expected_packages = lock["required_packages"]
    actual_packages = installed_versions(list(expected_packages))
    actual["packages"] = actual_packages
    checks.update(package_version_checks(expected_packages, actual_packages))
    checks["python_version_exact"] = actual["python_version"] == lock["python_version"]
    checks["running_in_virtual_environment"] = actual["virtual_environment"] is True

    visible_value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    actual["CUDA_VISIBLE_DEVICES"] = visible_value
    try:
        visible_index = parse_single_gpu_index(visible_value)
        checks["one_numeric_cuda_visible_device"] = True
        checks["requested_gpu_matches_visible_gpu"] = visible_index == requested_gpu_index
    except ValueError as error:
        visible_index = None
        checks["one_numeric_cuda_visible_device"] = False
        checks["requested_gpu_matches_visible_gpu"] = False
        errors.append(str(error))
    checks["jax_memory_preallocation_disabled"] = (
        os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() == "false"
    )

    try:
        source_audit = require_source(source)
        actual["source"] = source_audit["actual"]
        checks["source_provenance"] = True
    except Exception as error:  # keep a machine-readable fail-closed report
        actual["source"] = {"path": str(Path(source).expanduser().resolve())}
        checks["source_provenance"] = False
        errors.append(f"source provenance: {error}")

    try:
        gpu = gpu_observation(requested_gpu_index)
        actual["gpu"] = gpu
        checks["nvidia_gpu_index_exact"] = gpu["physical_index"] == requested_gpu_index
        checks["gpu_model_exact"] = gpu["name"] == lock["host_observation"]["gpu_model"]
        checks["nvidia_driver_exact"] = (
            gpu["driver_version"] == lock["host_observation"]["nvidia_driver"]
        )
        external = [
            app
            for app in gpu["compute_applications_before_jax"]
            if app["pid"] != os.getpid()
        ]
        actual["gpu"]["external_compute_applications_before_jax"] = external
        checks["gpu_has_no_external_compute_process"] = not external
    except Exception as error:
        checks["nvidia_gpu_index_exact"] = False
        checks["gpu_model_exact"] = False
        checks["nvidia_driver_exact"] = False
        checks["gpu_has_no_external_compute_process"] = False
        errors.append(f"GPU observation: {error}")

    checks["safe_to_initialize_jax"] = all(checks.values()) and not errors
    if not checks["safe_to_initialize_jax"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        return {
            "status": "BLOCKED_FDTDX_GPU_RUNTIME",
            "ready": False,
            "checks": checks,
            "failed_checks": failed,
            "errors": errors
            + ["CUDA initialization skipped because a pre-import guard failed"],
            "expected": lock,
            "actual": actual,
        }

    source_root = Path(source).expanduser().resolve() / "src"
    try:
        source_root_text = str(source_root)
        if source_root_text in sys.path:
            sys.path.remove(source_root_text)
        sys.path.insert(0, source_root_text)
        fdtdx = import_module("fdtdx")
        fdtdx_file = _resolved_module_file(fdtdx)
        actual["fdtdx_import_file"] = str(fdtdx_file)
        checks["fdtdx_imported_from_pinned_source"] = path_is_within(
            fdtdx_file, source_root / "fdtdx"
        )
    except Exception as error:
        checks["fdtdx_imported_from_pinned_source"] = False
        errors.append(f"FDTDX import: {error}")

    if visible_index is not None:
        try:
            jax = import_module("jax")
            jnp = import_module("jax.numpy")
            devices = jax.devices()
            actual["jax_backend"] = jax.default_backend()
            actual["jax_devices"] = [str(device) for device in devices]
            actual["jax_device_kinds"] = [device.device_kind for device in devices]
            checks["jax_backend_gpu"] = actual["jax_backend"] == "gpu"
            checks["jax_sees_exactly_one_device"] = len(devices) == 1
            checks["jax_device_model_exact"] = (
                len(devices) == 1
                and devices[0].device_kind == lock["host_observation"]["gpu_model"]
            )
            x = jnp.arange(256, dtype=jnp.float32).reshape(16, 16)
            value = (x @ x.T).sum().block_until_ready()
            checksum = float(value)
            actual["jax_smoke_checksum"] = checksum
            checks["jax_smoke_checksum_exact"] = (
                checksum == lock["host_observation"]["smoke_checksum"]
            )
        except Exception as error:
            checks["jax_backend_gpu"] = False
            checks["jax_sees_exactly_one_device"] = False
            checks["jax_device_model_exact"] = False
            checks["jax_smoke_checksum_exact"] = False
            errors.append(f"JAX GPU smoke: {error}")

    failed = sorted(name for name, passed in checks.items() if not passed)
    ready = not failed and not errors
    return {
        "status": "VALIDATED_FDTDX_GPU_RUNTIME" if ready else "BLOCKED_FDTDX_GPU_RUNTIME",
        "ready": ready,
        "checks": checks,
        "failed_checks": failed,
        "errors": errors,
        "expected": lock,
        "actual": actual,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=configured_source())
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    result = audit_runtime(args.source, args.gpu_index)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_ready and not result["ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
