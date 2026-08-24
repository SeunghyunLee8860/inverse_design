"""Place and read back the fresh FDTDX anchor without advancing Maxwell time."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_dependency import (
    configured_source,
    require_source,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
    grid_edges,
    layout,
    mesh_audit,
    reference_mask,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_material import (
    arrays_for_exact_binary,
    readback_exact_binary,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_mesh import (
    build_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_pml import (
    PML_FACES,
    SOLVER_PARAMETER_NAMES,
    solver_parameters,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_runtime_preflight import (
    load_runtime_lock,
)


ANCHOR_SPEC = MeshSpec()
DEFAULT_REFERENCE = "centered_square_2um"
REPORT_NAME = "FDTDX_FRESH_ANCHOR_PLACEMENT.json"
EDGE_ATOL_M = 2.0e-12


def _edge_index(edges: tuple[float, ...], coordinate_m: float) -> int:
    matches = [
        index
        for index, value in enumerate(edges)
        if math.isclose(value, coordinate_m, rel_tol=0.0, abs_tol=2.0e-18)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"edge {coordinate_m:.9e} m occurs {len(matches)} times")
    return matches[0]


def _slice_from_bounds(
    edges: tuple[tuple[float, ...], ...],
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> list[list[int]]:
    return [
        [_edge_index(edges[axis], lower), _edge_index(edges[axis], upper)]
        for axis, (lower, upper) in enumerate(bounds)
    ]


def expected_placement(spec: MeshSpec) -> dict[str, list[list[int]]]:
    """Return solver-independent object slices for one mesh specification."""

    edges = grid_edges(spec)
    x_domain = (edges[0][0], edges[0][-1])
    y_domain = (edges[1][0], edges[1][-1])
    source_z = _edge_index(edges[2], 0.750e-6)
    incident_z = _edge_index(edges[2], 0.500e-6)
    target_z = _edge_index(edges[2], 0.250e-6)
    pml_cells = layout(spec)["pml_cells_xy"]
    non_pml_x = (edges[0][pml_cells], edges[0][-pml_cells - 1])
    non_pml_y = (edges[1][pml_cells], edges[1][-pml_cells - 1])
    result = {
        "fixed_silicon_substrate": _slice_from_bounds(
            edges, (x_domain, y_domain, (-3.000e-6, -0.385e-6))
        ),
        "fixed_285nm_sio2": _slice_from_bounds(
            edges, (x_domain, y_domain, (-0.385e-6, -0.100e-6))
        ),
        "fixed_tairte4": _slice_from_bounds(
            edges, ((-8.0e-6, 8.0e-6), (-8.0e-6, 8.0e-6), (-0.100e-6, 0.0))
        ),
        "au_design": _slice_from_bounds(
            edges, ((-4.0e-6, 4.0e-6), (-4.0e-6, 4.0e-6), (0.0, 0.050e-6))
        ),
        "gaussian_source": [
            [_edge_index(edges[0], -8.0e-6), _edge_index(edges[0], 8.0e-6)],
            [_edge_index(edges[1], -8.0e-6), _edge_index(edges[1], 8.0e-6)],
            [source_z, source_z + 1],
        ],
        "incident_plane": [
            [_edge_index(edges[0], -8.0e-6), _edge_index(edges[0], 8.0e-6)],
            [_edge_index(edges[1], -8.0e-6), _edge_index(edges[1], 8.0e-6)],
            [incident_z, incident_z + 1],
        ],
        "target_field": [
            [_edge_index(edges[0], -8.0e-6), _edge_index(edges[0], 8.0e-6)],
            [_edge_index(edges[1], -8.0e-6), _edge_index(edges[1], 8.0e-6)],
            [target_z, target_z + 1],
        ],
        "material_flux": _slice_from_bounds(
            edges, (non_pml_x, non_pml_y, (-0.588e-6, 0.250e-6))
        ),
        "material_flux_td": _slice_from_bounds(
            edges, (non_pml_x, non_pml_y, (-0.588e-6, 0.250e-6))
        ),
    }
    return result


def expected_pml_slices(spec: MeshSpec) -> dict[str, list[list[int]]]:
    shape = mesh_audit(spec)["grid_shape_xyz"]
    xy = layout(spec)["pml_cells_xy"]
    z = layout(spec)["pml_cells_z"]
    return {
        "minx": [[0, xy], [0, shape[1]], [0, shape[2]]],
        "maxx": [[shape[0] - xy, shape[0]], [0, shape[1]], [0, shape[2]]],
        "miny": [[0, shape[0]], [0, xy], [0, shape[2]]],
        "maxy": [[0, shape[0]], [shape[1] - xy, shape[1]], [0, shape[2]]],
        "minz": [[0, shape[0]], [0, shape[1]], [0, z]],
        "maxz": [[0, shape[0]], [0, shape[1]], [shape[2] - z, shape[2]]],
    }


def _slice_list(value: tuple[slice, slice, slice]) -> list[list[int]]:
    return [[int(part.start), int(part.stop)] for part in value]


def _bounds_from_slice(
    edges: tuple[np.ndarray, np.ndarray, np.ndarray], value: list[list[int]]
) -> list[list[float]]:
    return [
        [float(edges[axis][lower]), float(edges[axis][upper])]
        for axis, (lower, upper) in enumerate(value)
    ]


def _pml_readback(model: dict[str, Any], spec: MeshSpec) -> dict[str, Any]:
    expected_profiles = solver_parameters(model["pml_face_parameters"])
    expected_slices = expected_pml_slices(spec)
    edges = tuple(np.asarray(model["grid"].edges(axis)) for axis in range(3))
    records: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    pml_objects = model["placed"].pml_objects
    checks["exactly_six_pml_objects"] = len(pml_objects) == 6
    for pml in pml_objects:
        face = pml.descriptive_name.replace("_", "")
        if face not in PML_FACES:
            raise RuntimeError(f"unexpected PML face {pml.descriptive_name!r}")
        actual_slice = _slice_list(pml.grid_slice)
        parameters = {
            name: float(getattr(pml, name)) for name in SOLVER_PARAMETER_NAMES
        }
        parameter_exact = all(
            math.isclose(
                parameters[name],
                expected_profiles[face][name],
                rel_tol=1.0e-12,
                abs_tol=1.0e-30,
            )
            for name in SOLVER_PARAMETER_NAMES
        )
        coefficient_arrays = {
            name: np.asarray(getattr(pml, name))
            for name in (
                "pml_a_E",
                "pml_b_E",
                "inv_kappa_E",
                "pml_a_H",
                "pml_b_H",
                "inv_kappa_H",
            )
        }
        finite = all(np.all(np.isfinite(value)) for value in coefficient_arrays.values())
        lossy = bool(
            np.any(coefficient_arrays["pml_b_E"] < 1.0)
            and np.any(coefficient_arrays["pml_b_H"] < 1.0)
        )
        kappa_one = bool(
            np.array_equal(
                coefficient_arrays["inv_kappa_E"],
                np.ones_like(coefficient_arrays["inv_kappa_E"]),
            )
            and np.array_equal(
                coefficient_arrays["inv_kappa_H"],
                np.ones_like(coefficient_arrays["inv_kappa_H"]),
            )
        )
        checks[f"{face}:slice_exact"] = actual_slice == expected_slices[face]
        checks[f"{face}:parameters_exact"] = parameter_exact
        checks[f"{face}:coefficients_finite"] = bool(finite)
        checks[f"{face}:coefficients_lossy"] = lossy
        checks[f"{face}:kappa_unity"] = kappa_one
        records[face] = {
            "object_name": pml.name,
            "axis": int(pml.axis),
            "direction": pml.direction,
            "slice": actual_slice,
            "bounds_m": _bounds_from_slice(edges, actual_slice),
            "physical_thickness_m": float(pml._physical_thickness()),
            "parameters": parameters,
            "coefficient_shapes": {
                name: list(value.shape) for name, value in coefficient_arrays.items()
            },
        }
    checks["all_six_face_labels_present"] = set(records) == set(PML_FACES)
    return {"checks": checks, "faces": records, "ready": all(checks.values())}


def audit_anchor(
    model: dict[str, Any],
    arrays: Any,
    *,
    spec: MeshSpec,
    reference: str,
    polarization: str,
) -> dict[str, Any]:
    expected_edges = tuple(np.asarray(axis, dtype=np.float64) for axis in grid_edges(spec))
    realized_edges = tuple(np.asarray(model["grid"].edges(axis)) for axis in range(3))
    checks: dict[str, bool] = {}
    edge_records: dict[str, Any] = {}
    for axis_name, expected, realized in zip(
        "xyz", expected_edges, realized_edges, strict=True
    ):
        cast_expected = expected.astype(realized.dtype)
        error = np.abs(realized.astype(np.float64) - expected)
        checks[f"{axis_name}_edges_exact_after_solver_dtype_cast"] = np.array_equal(
            realized, cast_expected
        )
        checks[f"{axis_name}_physical_edge_error_below_tolerance"] = bool(
            np.max(error) <= EDGE_ATOL_M
        )
        checks[f"{axis_name}_edges_strictly_increasing"] = bool(
            np.all(np.diff(realized) > 0.0)
        )
        edge_records[axis_name] = {
            "dtype": str(realized.dtype),
            "count": int(realized.size),
            "bounds_m": [float(realized[0]), float(realized[-1])],
            "minimum_step_m": float(np.min(np.diff(realized))),
            "maximum_step_m": float(np.max(np.diff(realized))),
            "max_absolute_error_vs_float64_contract_m": float(np.max(error)),
        }

    expected_objects = expected_placement(spec)
    actual_objects = model["placement"]
    object_records: dict[str, Any] = {}
    for name, expected_slice in expected_objects.items():
        actual_slice = actual_objects.get(name)
        checks[f"object:{name}:slice_exact"] = actual_slice == expected_slice
        if actual_slice is not None:
            object_records[name] = {
                "slice": actual_slice,
                "bounds_m": _bounds_from_slice(realized_edges, actual_slice),
            }
    checks["no_adjoint_source_in_placement_audit"] = (
        "distributed_adjoint_source" not in model["placed"]
    )
    checks["polarization_exact"] = tuple(
        model["placed"]["gaussian_source"].fixed_E_polarization_vector
    ) == ((0.0, 1.0, 0.0) if polarization == "Ea" else (1.0, 0.0, 0.0))

    material = readback_exact_binary(
        model, arrays, reference_mask(reference), spec
    )
    checks.update(
        {f"material:{name}": passed for name, passed in material["checks"].items()}
    )
    pml = _pml_readback(model, spec)
    checks.update({f"pml:{name}": passed for name, passed in pml["checks"].items()})
    checks["maxwell_time_stepping_was_not_called"] = True
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": (
            "VALIDATED_FDTDX_FRESH_ANCHOR_PLACEMENT"
            if not failed
            else "BLOCKED_FDTDX_FRESH_ANCHOR_PLACEMENT"
        ),
        "ready": not failed,
        "failed_checks": failed,
        "checks": checks,
        "spec": mesh_audit(spec),
        "reference": reference,
        "polarization": polarization,
        "grid_edges": edge_records,
        "time_contract": {
            "total_periods": 16,
            "window_periods": 4,
            "courant_factor": 0.5,
            "time_step_s": float(model["config"].time_step_duration),
            "time_steps_total": int(model["config"].time_steps_total),
            "field_dtype": str(model["config"].dtype),
        },
        "objects": object_records,
        "material": material,
        "pml": pml,
        "solver_array_shapes": {
            name: list(getattr(arrays, name).shape)
            for name in COEFFICIENT_ARRAY_NAMES
        },
        "maxwell_run_fdtd_calls": 0,
    }


COEFFICIENT_ARRAY_NAMES = (
    "inv_permittivities",
    "dispersive_c1",
    "dispersive_c2",
    "dispersive_c3",
)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _output_directory(value: Path) -> Path:
    path = value.expanduser().resolve()
    if not path.is_absolute() or not path.is_dir():
        raise RuntimeError("output directory must be an existing absolute directory")
    if any(path.iterdir()):
        raise RuntimeError("output directory must be empty before anchor placement")
    return path


def run(output_directory: Path, source: Path, reference: str, polarization: str) -> dict[str, Any]:
    output_directory = _output_directory(output_directory)
    source_audit = require_source(source)
    repository = Path(__file__).resolve().parents[3]
    provenance = {
        "repository_commit": _git(repository, "rev-parse", "HEAD"),
        "repository_dirty_porcelain": _git(
            repository, "status", "--porcelain", "--untracked-files=all"
        ),
        "fdtdx_source": source_audit["actual"],
        "runtime_lock": load_runtime_lock(),
    }
    started = time.perf_counter()
    model = build_model(
        ANCHOR_SPEC,
        polarization,
        total_periods=16,
        window_periods=4,
        courant_factor=0.5,
        include_adjoint_source=False,
        air_only_source_calibration=False,
    )
    arrays = arrays_for_exact_binary(
        model, reference_mask(reference), ANCHOR_SPEC
    )
    result = audit_anchor(
        model,
        arrays,
        spec=ANCHOR_SPEC,
        reference=reference,
        polarization=polarization,
    )
    result["placement_and_readback_runtime_s"] = time.perf_counter() - started
    result["provenance"] = provenance
    _atomic_json(output_directory / REPORT_NAME, result)
    return result


def main() -> int:
    configured_output = os.environ.get("FDTDX_FRESH_OUTPUT_DIR", "").strip()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(configured_output) if configured_output else None,
    )
    parser.add_argument("--source", type=Path, default=configured_source())
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--polarization", choices=("Ea", "Eb"), default="Ea")
    args = parser.parse_args()
    if args.output_dir is None:
        parser.error("--output-dir or FDTDX_FRESH_OUTPUT_DIR is required")
    try:
        result = run(args.output_dir, args.source, args.reference, args.polarization)
    except Exception as error:
        failure = {
            "status": "BLOCKED_FDTDX_FRESH_ANCHOR_PLACEMENT_EXCEPTION",
            "ready": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "maxwell_run_fdtd_calls": 0,
        }
        output = Path(args.output_dir).expanduser().resolve()
        if output.is_dir() and not any(output.iterdir()):
            _atomic_json(output / REPORT_NAME, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    summary = {
        "status": result["status"],
        "ready": result["ready"],
        "failed_checks": result["failed_checks"],
        "grid_shape_xyz": result["spec"]["grid_shape_xyz"],
        "yee_cell_count": result["spec"]["yee_cell_count"],
        "material_law": result["material"]["material_law"],
        "report": str(Path(args.output_dir).resolve() / REPORT_NAME),
        "maxwell_run_fdtd_calls": 0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
