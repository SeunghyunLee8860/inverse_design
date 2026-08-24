"""Fail-closed contract for Lumerical density topology plus custom GPU PDEs.

One filtered/projected topology occupancy is shared by the optical, thermal,
and electrical constitutive maps.  A continuous occupancy is allowed during
gradient optimization; exact binary dispersive Au is required at endpoint
controls and final promotion.  The occupancy is neither an ``np density``
carrier field nor a physical electron/hole density.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.au_density_relaxation import (
    audit as density_relaxation_audit,
)


HERE = Path(__file__).resolve().parent
LUMERICAL_ROOT = Path(
    os.environ.get(
        "AU_LUMERICAL_ROOT",
        "/home/seunghyun/lumerical_r12/opt/lumerical/v261",
    )
).expanduser()
LUMAPI_PATH = LUMERICAL_ROOT / "api/python/lumapi.py"
LEGACY_LUMOPT_TOPOLOGY = LUMERICAL_ROOT / "api/python/lumopt/geometries/topology.py"
LUMOPT2_TOPOLOGY = LUMERICAL_ROOT / "api/python/lumopt2/parametrization/topology.py"
LUMOPT2_DEPS = LUMERICAL_ROOT / "api/python/lumopt2/parametrization/d_eps_calculator.py"


@dataclass(frozen=True)
class LumericalMaxwellContract:
    policy_version: int = 4
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
        "latent rho -> 500-nm density filter -> tanh projection with beta "
        "continuation -> one shared projected topology occupancy"
    )
    shared_design_field: str = (
        "one 81x81 nodal projected occupancy plus physical coordinates and "
        "SHA-256; every solver input is derived by a tested discrete map; the "
        "final 0/1 mask uses a separate canonical physical-geometry hash"
    )
    projected_density_grid: str = "81x81 nodes over 80x80 100-nm physical cells"
    custom_pde_density_map: str = (
        "four-node arithmetic cell average with exact discrete transpose"
    )
    density_topology_required: bool = True
    shape_or_level_set_required: bool = False
    continuous_relaxation_allowed_during_optimization: bool = True
    exact_binary_required_for_every_physics_evaluation: bool = False
    numerical_interface_cut_cells_allowed: bool = True
    different_optical_thermal_electrical_design_fields_allowed: bool = False
    exact_binary_required_for_final_promotion: bool = True
    exact_dispersive_au_required_at_material_endpoint: bool = True
    exact_dispersive_au_required_for_final_reevaluation: bool = True
    optical_relaxation_law: str = "christiansen_nk_then_square_v1"
    optical_rho_power: float | None = None
    np_density_as_au_topology_variable_allowed: bool = False
    continuous_lumerical_au_carrier_status: str = (
        "SOLVER_FREE_NK_LAW_IMPLEMENTED; "
        "BLOCKED_PENDING_B200_ENDPOINT_BANDWIDTH_RESONANCE_AND_SAME_STEP_ADFD"
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


def canonical_projected_density(density: np.ndarray) -> np.ndarray:
    """Validate and canonicalize the shared projected topology occupancy."""

    value = np.asarray(density, dtype=np.float64)
    if value.ndim != 2 or value.size == 0:
        raise ValueError("projected Au topology density must be a non-empty 2-D array")
    if not np.all(np.isfinite(value)):
        raise ValueError("projected Au topology density contains a non-finite value")
    tolerance = 1.0e-12
    if np.any(value < -tolerance) or np.any(value > 1.0 + tolerance):
        raise ValueError("projected Au topology density must remain in [0,1]")
    return np.ascontiguousarray(np.clip(value, 0.0, 1.0))


def projected_density_sha256(density: np.ndarray) -> str:
    """Hash the exact relaxed design state passed to every constitutive map."""

    value = canonical_projected_density(density)
    digest = hashlib.sha256()
    digest.update(b"lumerical-au-projected-topology-density-v4\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0float64-c\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def canonical_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Validate an endpoint/final-promotion mask and return exact uint8 values."""

    value = np.asarray(mask)
    if value.ndim != 2 or value.size == 0:
        raise ValueError("Au mask must be a non-empty 2-D array")
    if value.dtype.kind not in "biuf":
        raise ValueError("Au mask must have a numeric or boolean dtype")
    if not np.all(np.isfinite(value)):
        raise ValueError("Au mask contains a non-finite value")
    if not np.all((value == 0) | (value == 1)):
        raise ValueError("final Au mask must contain only exact 0/1 values")
    return np.ascontiguousarray(value, dtype=np.uint8)


def binary_mask_sha256(mask: np.ndarray) -> str:
    """Hash only mask shape and values, not its physical geometry.

    This is useful as a payload checksum.  It is deliberately not accepted as
    the cross-solver geometry identity because it omits pitch, origin, axes,
    and Au thickness.  Use :func:`exact_au_geometry_sha256` for that purpose.
    """

    value = canonical_binary_mask(mask)
    digest = hashlib.sha256()
    digest.update(b"lumerical-exact-binary-au-mask-v1\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0uint8\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_edges(
    values: np.ndarray, expected_size: int, label: str
) -> np.ndarray:
    edges = np.asarray(values)
    if edges.dtype.kind not in "biuf":
        raise ValueError(f"{label} must have a numeric dtype")
    edges = np.ascontiguousarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size != expected_size:
        raise ValueError(f"{label} must have exactly {expected_size} entries")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError(f"{label} must be finite and strictly increasing")
    return edges


def canonical_exact_au_geometry(
    mask: np.ndarray,
    *,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    z_bounds_m: np.ndarray,
    axis_x: str,
    axis_y: str,
) -> dict[str, Any]:
    """Validate one grid-aligned exact-Au geometry shared by all solvers.

    ``mask[ix, iy]`` occupies the complete rectangular prism bounded by the
    matching x/y cell edges and the common Au z bounds.  Solver-specific
    conformal or cut-cell representations may be derived from this geometry,
    but they may not reinterpret its orientation, scale, origin, or thickness.
    """

    value = canonical_binary_mask(mask)
    x_edges = _canonical_edges(x_edges_m, value.shape[0] + 1, "x_edges_m")
    y_edges = _canonical_edges(y_edges_m, value.shape[1] + 1, "y_edges_m")
    z_bounds = _canonical_edges(z_bounds_m, 2, "z_bounds_m")
    if (axis_x, axis_y) != ("b", "a"):
        raise ValueError("production coordinate contract requires solver x=b and y=a")
    return {
        "schema": "exact-grid-aligned-au-geometry-v1",
        "mask_index_order": "mask[ix,iy] maps to solver x-cell ix and y-cell iy",
        "mask": value,
        "x_edges_m": x_edges,
        "y_edges_m": y_edges,
        "z_bounds_m": z_bounds,
        "axis_mapping": {"x": axis_x, "y": axis_y},
    }


def exact_au_geometry_sha256(
    mask: np.ndarray,
    *,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    z_bounds_m: np.ndarray,
    axis_x: str = "b",
    axis_y: str = "a",
) -> str:
    """Hash exact occupancy together with its complete physical placement."""

    geometry = canonical_exact_au_geometry(
        mask,
        x_edges_m=x_edges_m,
        y_edges_m=y_edges_m,
        z_bounds_m=z_bounds_m,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    digest = hashlib.sha256()
    digest.update(b"exact-grid-aligned-au-geometry-v1\0")
    digest.update(
        json.dumps(
            {
                "mask_shape": list(geometry["mask"].shape),
                "mask_dtype": "uint8",
                "mask_index_order": geometry["mask_index_order"],
                "coordinate_dtype": "float64",
                "coordinate_units": "m",
                "axis_mapping": geometry["axis_mapping"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for label in ("mask", "x_edges_m", "y_edges_m", "z_bounds_m"):
        digest.update(b"\0" + label.encode("ascii") + b"\0")
        digest.update(np.ascontiguousarray(geometry[label]).tobytes(order="C"))
    return digest.hexdigest()


def exact_au_geometry_audit(
    mask: np.ndarray,
    *,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    z_bounds_m: np.ndarray,
    axis_x: str = "b",
    axis_y: str = "a",
) -> dict[str, Any]:
    """Return a JSON-safe manifest record without embedding the full mask."""

    geometry = canonical_exact_au_geometry(
        mask,
        x_edges_m=x_edges_m,
        y_edges_m=y_edges_m,
        z_bounds_m=z_bounds_m,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    value = geometry["mask"]
    return {
        "schema": geometry["schema"],
        "geometry_sha256": exact_au_geometry_sha256(
            value,
            x_edges_m=geometry["x_edges_m"],
            y_edges_m=geometry["y_edges_m"],
            z_bounds_m=geometry["z_bounds_m"],
            axis_x=axis_x,
            axis_y=axis_y,
        ),
        "mask_payload_sha256": binary_mask_sha256(value),
        "mask_shape_xy": list(value.shape),
        "occupied_cell_count": int(np.count_nonzero(value)),
        "occupied_area_fraction": float(np.mean(value)),
        "x_bounds_m": [
            float(geometry["x_edges_m"][0]),
            float(geometry["x_edges_m"][-1]),
        ],
        "y_bounds_m": [
            float(geometry["y_edges_m"][0]),
            float(geometry["y_edges_m"][-1]),
        ],
        "z_bounds_m": [float(item) for item in geometry["z_bounds_m"]],
        "axis_mapping": geometry["axis_mapping"],
        "mask_index_order": geometry["mask_index_order"],
    }


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
    density = density_relaxation_audit()
    gates = {
        "lumapi_v261_present": LUMAPI_PATH.is_file(),
        "installed_lumopt_requires_custom_au_route": bool(source.get("passed")),
        "solver_free_au_density_law_passed": bool(
            density["passive_on_uniform_density_sweep"]
            and density["exact_background_endpoint"]
            and density["exact_au_endpoint"]
            and not density["rho_cubed_used"]
        ),
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
        "au_density_relaxation": density,
        "gates": gates,
        "notes": [
            "Only FDTD time stepping uses the B200; Lumerical meshing and scripts use CPU.",
            "Thermal and electrical solves remain the repository custom CUDA PDE solvers.",
            "No Lumerical HEAT or CHARGE license is assumed or required.",
            "Density topology, not shape/level-set optimization, is the selected design method.",
            "One hash-identified projected occupancy is shared by all constitutive maps.",
            "The optical law is n-k interpolation followed by epsilon=(n+ik)^2; rho**3 is not used.",
            "The custom Au density-to-Yee derivative must pass same-step AD-FD before use.",
            "Final promotion requires independent ordinary dispersive-Au binary reevaluation.",
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
