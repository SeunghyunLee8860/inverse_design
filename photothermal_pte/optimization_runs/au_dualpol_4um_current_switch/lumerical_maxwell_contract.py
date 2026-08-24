"""Fail-closed contract for exact-Au Lumerical plus custom GPU PDE solvers.

Optimizer parameters may be continuous, but every physical evaluation must
realize one exact binary Au geometry.  That same geometry is consumed by the
optical, thermal, and electrical solvers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
LUMERICAL_ROOT = Path("/home/seunghyun/lumerical_r12/opt/lumerical/v261")
LUMAPI_PATH = LUMERICAL_ROOT / "api/python/lumapi.py"
LEGACY_LUMOPT_TOPOLOGY = LUMERICAL_ROOT / "api/python/lumopt/geometries/topology.py"
LUMOPT2_TOPOLOGY = LUMERICAL_ROOT / "api/python/lumopt2/parametrization/topology.py"
LUMOPT2_DEPS = LUMERICAL_ROOT / "api/python/lumopt2/parametrization/d_eps_calculator.py"


@dataclass(frozen=True)
class LumericalMaxwellContract:
    policy_version: int = 3
    maxwell_solver: str = "Ansys Lumerical FDTD v261 (2026 R1.2)"
    thermal_solver: str = "repository custom CUDA finite-volume steady heat solver"
    electrical_solver: str = (
        "repository custom CUDA weighting-potential finite-element solver"
    )
    maxwell_accelerator_required: str = "NVIDIA B200"
    maxwell_execution_mode: str = "3D FDTD GPU"
    thermal_execution_mode: str = "custom sparse linear solve on CUDA"
    electrical_execution_mode: str = "custom sparse linear solve on CUDA"
    optimization_design_map: str = (
        "continuous shape/level-set parameters -> 500-nm DFM geometry map -> "
        "one exact binary Au mask"
    )
    shared_design_field: str = (
        "one exact 0/1 Au mask with one shape and SHA-256 is passed to all three "
        "physical solvers"
    )
    continuous_geometry_parameters_allowed: bool = True
    gray_au_material_in_maxwell_allowed: bool = False
    gray_au_material_in_thermal_allowed: bool = False
    gray_au_material_in_electrical_allowed: bool = False
    exact_binary_required_for_every_physics_evaluation: bool = True
    numerical_interface_cut_cells_allowed: bool = True
    different_optical_thermal_electrical_design_fields_allowed: bool = False
    exact_binary_required_for_final_promotion: bool = True
    exact_dispersive_au_required_in_every_maxwell_evaluation: bool = True
    np_density_as_au_topology_variable_allowed: bool = False
    optical_geometry_gradient_status: str = (
        "BLOCKED_PENDING_4UM_EXACT_AU_SHAPE_ADFD_OR_VALIDATED_BINARY_ESTIMATOR"
    )
    bundled_lumopt_topology_gradient_allowed_without_au_adfd: bool = False
    fdtdx_allowed: bool = False
    jax_maxwell_allowed: bool = False
    required_polarizations: tuple[str, str] = ("Ea", "Eb")
    objective: str = "maximize min(+I_Ea,-I_Eb)"
    current_sign: str = "positive conventional current along solver +x (x_min to x_max)"
    solver_axes: str = "x=crystal b, y=crystal a, z=repository c=b optical closure"
    minimum_solid_feature_m: float = 500e-9
    minimum_void_feature_m: float = 500e-9


CONTRACT = LumericalMaxwellContract()


def canonical_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Validate the physical Au geometry and return exact uint8 values."""

    value = np.asarray(mask)
    if value.ndim != 2 or value.size == 0:
        raise ValueError("Au mask must be a non-empty 2-D array")
    if not np.all(np.isfinite(value)):
        raise ValueError("Au mask contains a non-finite value")
    if not np.all((value == 0) | (value == 1)):
        raise ValueError("physical Au mask must contain only exact 0/1 values")
    return np.ascontiguousarray(value, dtype=np.uint8)


def binary_mask_sha256(mask: np.ndarray) -> str:
    """Hash shape, dtype, and values of the physical mask."""

    value = canonical_binary_mask(mask)
    digest = hashlib.sha256()
    digest.update(b"lumerical-exact-binary-au-mask-v1\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0uint8\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _run_nvidia_smi() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return [{"inventory_error": f"{type(exc).__name__}: {exc}"}]
    result: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            result.append({"inventory_error": f"unparsed nvidia-smi row: {line}"})
            continue
        result.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
                "memory_total_MiB": int(fields[3]),
                "compute_capability": fields[4],
                "driver_version": fields[5],
            }
        )
    return result


def _source_audit() -> dict[str, Any]:
    """Prove why bundled generic topology gradients need an Au-specific gate."""

    paths = (LEGACY_LUMOPT_TOPOLOGY, LUMOPT2_TOPOLOGY, LUMOPT2_DEPS)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return {"passed": False, "missing": missing}
    legacy = LEGACY_LUMOPT_TOPOLOGY.read_text(encoding="utf-8")
    topology2 = LUMOPT2_TOPOLOGY.read_text(encoding="utf-8")
    deps2 = LUMOPT2_DEPS.read_text(encoding="utf-8")
    evidence = {
        "legacy_eps_levels_scalar_format": "params.eps_levels=[{0},{1}]" in legacy,
        "legacy_real_dF_dEps": "dF_dEps = real(dF_dEps)" in legacy,
        "lumopt2_real_material_index_annotation": bool(
            re.search(r"material_index:\s*float", topology2)
        ),
        "lumopt2_discards_complex_index_in_deps": "real(D{label}.index_x^2)" in deps2,
        "lumopt2_discards_complex_sparse_difference": (
            "real(d_eps_ctr_sparse.eps_x.data)" in deps2
        ),
        "lumopt2_clips_negative_real_epsilon": (
            "eps_real_data = np.where(eps_real_data < 0" in deps2
        ),
        "lumopt2_lossless_cauchy_assumption": "n>>k~0" in deps2,
    }
    return {
        "passed": all(evidence.values()),
        "evidence": evidence,
        "files": [str(path) for path in paths],
    }


def audit_environment(*, requested_gpu_index: int | None = None) -> dict[str, Any]:
    """Audit the Maxwell environment and require an actual NVIDIA B200."""

    inventory = _run_nvidia_smi()
    candidates = [
        item
        for item in inventory
        if "inventory_error" not in item
        and (requested_gpu_index is None or item["index"] == requested_gpu_index)
    ]
    b200 = [item for item in candidates if "B200" in str(item["name"]).upper()]
    source = _source_audit()
    gates = {
        "lumapi_v261_present": LUMAPI_PATH.is_file(),
        "installed_lumopt_requires_custom_au_route": bool(source.get("passed")),
        "requested_gpu_exists": bool(candidates),
        "requested_gpu_is_nvidia_b200": bool(b200),
    }
    return {
        "status": (
            "READY_FOR_LUMERICAL_B200_MAXWELL_DEVELOPMENT"
            if all(gates.values())
            else "BLOCKED_LUMERICAL_B200_PREFLIGHT"
        ),
        "contract": asdict(CONTRACT),
        "requested_gpu_index": requested_gpu_index,
        "gpu_inventory": inventory,
        "matching_b200": b200,
        "installed_source_audit": source,
        "gates": gates,
        "notes": [
            "Only FDTD time stepping uses the B200; Lumerical meshing and scripts use CPU.",
            "Thermal and electrical solves remain the repository custom CUDA PDE solvers.",
            "No Lumerical HEAT or CHARGE license is assumed or required.",
            "Continuous parameters may move a boundary; gray Au material is prohibited.",
            "Every solver must consume the same hash-identified exact binary Au geometry.",
            "The exact-Au shape derivative or binary search estimator must pass same-step validation before use.",
            "No Maxwell solve is authorized when the B200 gate is false.",
        ],
    }


def require_b200(*, requested_gpu_index: int | None = None) -> dict[str, Any]:
    result = audit_environment(requested_gpu_index=requested_gpu_index)
    if result["status"] != "READY_FOR_LUMERICAL_B200_MAXWELL_DEVELOPMENT":
        inventory = result["gpu_inventory"]
        raise RuntimeError(
            "Lumerical B200 preflight failed; refusing Maxwell solve. "
            f"requested_gpu_index={requested_gpu_index}, inventory={inventory}"
        )
    return result
