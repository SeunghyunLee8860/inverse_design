"""Fail-closed provenance checks shared by diagnostic validation scripts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    audit as material_fraction_audit,
)


SOURCE_CALIBRATION_STATUS = "VALIDATED_FDTDX_4UM_SOURCE_POWER_CALIBRATION"
OPTICAL_DECOMPOSITION_STATUS = "DIAGNOSTIC_ONLY_NOT_AN_OPTIMIZATION_GATE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    """Hash an array with explicit shape and dtype, not only its byte payload."""

    import numpy as np

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def require_status(payload: dict[str, Any], expected: str, label: str) -> None:
    actual = payload.get("status")
    if actual != expected:
        raise RuntimeError(f"{label} status is {actual!r}, expected {expected!r}")


def require_material_fraction(payload: dict[str, Any], label: str) -> None:
    expected = material_fraction_audit()
    actual = payload.get("au_material_fraction")
    if actual != expected:
        raise RuntimeError(
            f"{label} was not generated with the current common Au material law: "
            f"found {actual!r}, expected {expected!r}"
        )


def require_single_visible_gpu() -> str:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if len(devices) != 1 or devices[0] == "-1":
        raise RuntimeError(
            "set CUDA_VISIBLE_DEVICES to exactly one physical GPU before this solve"
        )
    return devices[0]
