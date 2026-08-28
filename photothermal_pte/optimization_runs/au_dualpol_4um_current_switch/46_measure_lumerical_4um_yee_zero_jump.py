#!/usr/bin/env python3
"""Measure, rather than assume, a rho=0 component-Yee epsilon jump.

The experiment is deliberately layout/index-only.  It never calls
``fdtd.run`` or starts a Maxwell engine.  Three increasingly realistic maps
are measured in the same Lumerical version:

1. a minimal uniform imported-index cube, sampled well inside every boundary;
2. the production thin-film layout with a spatially uniform imported density;
3. one exact-zero node in a frozen nonuniform beta=2 production density.

Every reported epsilon is the square of Lumerical's raw complex
``index_detail`` component.  Complex samples, deltas, delta/rho, coordinates,
and small-positive linear fits are retained in an NPZ in addition to the
human-readable JSON/CSV/Markdown/PNG summaries.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_INDEX,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.au_density_relaxation import (  # noqa: E402
    CONTRACT as RELAXATION_CONTRACT,
    epsilon_relaxation,
    nk_relaxation,
    ordal_au_index,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (  # noqa: E402
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (  # noqa: E402
    density_nodes,
    load_projected_density_file,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_exact_au import (  # noqa: E402
    SOURCE_WAVELENGTH_BAND_M,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_yee_jacobian import (  # noqa: E402
    COMPONENTS,
    component_coordinates,
    read_lumerical_index_detail,
    set_lumerical_projected_density,
    validate_index_detail,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_maxwell_contract import (  # noqa: E402
    LUMAPI_PATH,
    LUMERICAL_ROOT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_only_boundary import (  # noqa: E402
    require_lumerical_only_source_boundary,
)


C0_M_S = 299_792_458.0
UNIFORM_RHOS = np.asarray(
    [0.0, 1.0e-7, 3.0e-7, 1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5, 1.0e-4, 1.0e-3],
    dtype=np.float64,
)
NONUNIFORM_RHOS = np.asarray(
    [0.0, 1.0e-7, 1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5, 1.0e-4],
    dtype=np.float64,
)
EXACT_ZERO_GROUP_SCREEN_RHOS = np.asarray([1.0e-7, 1.0e-6], dtype=np.float64)
FIT_MAX_RHO = 3.0e-6
MINIMAL_IMPORT_NAME = "zero_jump_uniform_import"
MINIMAL_PABS_GROUP = "zero_jump_uniform_pabs"
MINIMAL_INDEX_MONITOR = f"{MINIMAL_PABS_GROUP}::index"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-project", required=True, type=Path)
    parser.add_argument("--density-file", required=True, type=Path)
    parser.add_argument("--density-key", default="projected_density_nodal")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-index", type=int, default=2)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _complex_json(value: complex) -> list[float]:
    item = complex(value)
    return [float(item.real), float(item.imag)]


def _configure_lumapi(gpu_index: int) -> None:
    os.environ["VC_LUMERICAL_ROOT"] = str(LUMERICAL_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(LUMERICAL_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(LUMAPI_PATH.parent)
    os.environ["PATH"] = f"{LUMERICAL_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
    # No engine is launched in this script.  Keep the session isolated anyway
    # so an accidental future GPU allocation cannot touch the production GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_index))
    if str(LUMAPI_PATH.parent) not in sys.path:
        sys.path.insert(0, str(LUMAPI_PATH.parent))


def _single_frequency(item: Any) -> None:
    for name, value in (
        ("override global monitor settings", True),
        ("use source limits", False),
        ("use wavelength spacing", True),
        ("wavelength center", CONTRACT.wavelength_m),
        ("wavelength span", 0.0),
        ("frequency points", 1),
    ):
        try:
            item[name] = value
        except Exception:
            pass


def _build_minimal_uniform_layout(fdtd: Any) -> dict[str, Any]:
    """Create a large import cube and a much smaller central index region."""

    fdtd.eval("selectall; delete;")
    solver = fdtd.addfdtd()
    solver["dimension"] = "3D"
    for axis in COMPONENTS:
        solver[f"{axis} min"] = -2.0e-6
        solver[f"{axis} max"] = 2.0e-6
        solver[f"{axis} min bc"] = "PML"
        solver[f"{axis} max bc"] = "PML"
    solver["mesh type"] = "auto non-uniform"
    solver["mesh refinement"] = "staircase"
    solver["mesh accuracy"] = 3
    solver["override simulation bandwidth for mesh generation"] = True
    solver["mesh wavelength min"] = SOURCE_WAVELENGTH_BAND_M[0]
    solver["mesh wavelength max"] = SOURCE_WAVELENGTH_BAND_M[1]

    # An imported material is fitted over the active source band even when no
    # propagation is run.  Include the production band so this layout-only
    # readback does not silently use Lumerical's unrelated default band.
    source = fdtd.addplane()
    source["name"] = "zero_jump_band_definition_only"
    source["injection axis"] = "z"
    source["direction"] = "forward"
    source["x span"] = 1.0e-6
    source["y span"] = 1.0e-6
    source["z"] = -0.50e-6
    source["override global source settings"] = True
    source["wavelength start"] = SOURCE_WAVELENGTH_BAND_M[0]
    source["wavelength stop"] = SOURCE_WAVELENGTH_BAND_M[1]

    mesh = fdtd.addmesh()
    mesh["name"] = "zero_jump_uniform_mesh"
    for axis in COMPONENTS:
        mesh[f"{axis} min"] = -0.75e-6
        mesh[f"{axis} max"] = 0.75e-6
        mesh[f"override {axis} mesh"] = True
        mesh[f"d{axis}"] = 50.0e-9

    axes = np.asarray([-1.0e-6, 0.0, 1.0e-6], dtype=np.float64)
    index = np.ones((axes.size, axes.size, axes.size), dtype=np.complex128)
    fdtd.addimport({"name": MINIMAL_IMPORT_NAME, "x": 0.0, "y": 0.0, "z": 0.0})
    imported = fdtd.importnk2(index, axes, axes, axes)
    if imported is not None and int(imported) != 1:
        raise RuntimeError("minimal uniform importnk2 returned failure")

    pabs = fdtd.addobject("pabs_adv")
    pabs["name"] = MINIMAL_PABS_GROUP
    for axis in COMPONENTS:
        pabs[axis] = 0.0
        pabs[f"{axis} span"] = 0.50e-6
    for internal in (
        f"{MINIMAL_PABS_GROUP}::field",
        MINIMAL_INDEX_MONITOR,
    ):
        for name, value in (
            ("override global monitor settings", True),
            ("use source limits", False),
            ("use wavelength spacing", True),
            ("wavelength center", CONTRACT.wavelength_m),
            ("wavelength span", 0.0),
            ("frequency points", 1),
        ):
            try:
                fdtd.setnamed(internal, name, value)
            except Exception:
                pass
    return {
        "import_bounds_m": {axis: [-1.0e-6, 1.0e-6] for axis in COMPONENTS},
        "monitor_bounds_m": {axis: [-0.25e-6, 0.25e-6] for axis in COMPONENTS},
        "mesh_step_m": 50.0e-9,
        "distance_from_monitor_to_import_boundary_m": 0.75e-6,
        "conformal_mesh": "staircase",
        "source_band_m": list(SOURCE_WAVELENGTH_BAND_M),
        "source_role": "layout-only imported-material fit-band definition; never propagated",
    }


def _set_minimal_uniform_density(fdtd: Any, rho: float) -> None:
    axes = np.asarray([-1.0e-6, 0.0, 1.0e-6], dtype=np.float64)
    value = complex(nk_relaxation(np.asarray([rho]))[0])
    index = np.full((axes.size, axes.size, axes.size), value, np.complex128)
    fdtd.select(MINIMAL_IMPORT_NAME)
    imported = fdtd.importnk2(index, axes, axes, axes)
    if imported is not None and int(imported) != 1:
        raise RuntimeError("minimal uniform density importnk2 returned failure")


def _epsilon(detail: Mapping[str, np.ndarray], component: str) -> np.ndarray:
    return np.asarray(detail[f"epsilon_{component}"], dtype=np.complex128)


def _nearest_sample(
    detail: Mapping[str, np.ndarray], component: str, target_xyz: tuple[float, float, float]
) -> tuple[tuple[int, int, int], tuple[float, float, float], complex]:
    coordinates = component_coordinates(detail, component)
    index = tuple(
        int(np.argmin(np.abs(axis - target)))
        for axis, target in zip(coordinates, target_xyz, strict=True)
    )
    actual = tuple(float(coordinates[axis][index[axis]]) for axis in range(3))
    return index, actual, complex(_epsilon(detail, component)[index])


def _grid_comparison(
    baseline: Mapping[str, np.ndarray], detail: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Quantify every coordinate change without treating a hash mismatch as physics."""

    baseline_audit = validate_index_detail(baseline)
    audit = validate_index_detail(detail)
    for key in ("base_shape", "component_shapes"):
        if audit[key] != baseline_audit[key]:
            raise RuntimeError(f"index_detail changed incompatible Yee grid {key}")
    frequency_scale = max(abs(float(baseline_audit["frequency_hz"])), 1.0)
    frequency_difference = float(audit["frequency_hz"]) - float(
        baseline_audit["frequency_hz"]
    )
    if abs(frequency_difference) > 1.0e-12 * frequency_scale:
        raise RuntimeError("index_detail changed objective frequency")
    coordinate_records: dict[str, Any] = {}
    maximum = 0.0
    for axis in COMPONENTS:
        for key in (axis, f"{axis}_offset"):
            reference = np.asarray(baseline[key], dtype=np.float64).reshape(-1)
            current = np.asarray(detail[key], dtype=np.float64).reshape(-1)
            if current.shape != reference.shape:
                raise RuntimeError(f"index_detail changed coordinate shape {key}")
            difference = current - reference
            key_maximum = float(np.max(np.abs(difference)))
            maximum = max(maximum, key_maximum)
            coordinate_records[key] = {
                "exactly_equal": bool(np.array_equal(current, reference)),
                "maximum_absolute_difference_m": key_maximum,
                "rms_absolute_difference_m": float(
                    np.sqrt(np.mean(np.abs(difference) ** 2))
                ),
            }
    return {
        "coordinate_sha256": audit["coordinate_sha256"],
        "coordinate_hash_matches_rho_zero": bool(
            audit["coordinate_sha256"] == baseline_audit["coordinate_sha256"]
        ),
        "maximum_absolute_coordinate_difference_m": maximum,
        "frequency_difference_hz": frequency_difference,
        "coordinate_differences": coordinate_records,
    }


def _fit_complex(rho: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(rho, dtype=np.float64)
    y = np.asarray(values, dtype=np.complex128)
    mask = (x > 0.0) & (x <= FIT_MAX_RHO)
    if np.count_nonzero(mask) < 3:
        raise RuntimeError("small-positive fit has fewer than three points")
    design = np.column_stack((np.ones(np.count_nonzero(mask)), x[mask]))
    coefficients, _, _, _ = np.linalg.lstsq(design, y[mask], rcond=None)
    fitted = design @ coefficients
    residual = y[mask] - fitted
    return {
        "positive_fit_rhos": x[mask].tolist(),
        "intercept_zero_plus": _complex_json(coefficients[0]),
        "slope": _complex_json(coefficients[1]),
        "maximum_abs_residual": float(np.max(np.abs(residual))),
        "rms_abs_residual": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
    }


def _uniform_case(
    *,
    name: str,
    rhos: np.ndarray,
    evaluate: Any,
    target_xyz: tuple[float, float, float],
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    details: list[dict[str, np.ndarray]] = []
    audits: list[dict[str, Any]] = []
    for rho in rhos:
        detail = evaluate(float(rho))
        details.append(detail)
        audits.append(validate_index_detail(detail))
    baseline_audit = audits[0]
    grid_comparisons = [_grid_comparison(details[0], detail) for detail in details]

    analytic = np.asarray(epsilon_relaxation(rhos), dtype=np.complex128)
    arrays: dict[str, np.ndarray] = {
        f"{name}_rho": np.array(rhos, copy=True),
        f"{name}_epsilon_material": analytic,
    }
    records: dict[str, Any] = {
        "name": name,
        "rho_values": rhos.tolist(),
        "baseline_grid": baseline_audit,
        "grid_comparison_to_exact_zero_by_rho": [
            {"rho": float(rho), **comparison}
            for rho, comparison in zip(rhos, grid_comparisons, strict=True)
        ],
        "components": {},
    }
    table: list[dict[str, Any]] = []
    for component in COMPONENTS:
        samples = [
            _nearest_sample(detail, component, target_xyz) for detail in details
        ]
        indices = np.asarray([sample[0] for sample in samples], dtype=int)
        coordinates = np.asarray([sample[1] for sample in samples], dtype=float)
        values = np.asarray([sample[2] for sample in samples], dtype=np.complex128)
        delta = values - values[0]
        ratio = np.full(values.shape, np.nan + 1j * np.nan, np.complex128)
        ratio[1:] = delta[1:] / rhos[1:]
        fit = _fit_complex(rhos, values)
        fit_delta = complex(*fit["intercept_zero_plus"]) - values[0]
        fit["actual_epsilon_zero"] = _complex_json(values[0])
        fit["delta_zero_plus_minus_actual_zero"] = _complex_json(fit_delta)
        fit["delta_magnitude"] = float(abs(fit_delta))
        arrays[f"{name}_{component}_representative_index_all_rhos"] = indices
        arrays[f"{name}_{component}_representative_coordinate_m_all_rhos"] = coordinates
        arrays[f"{name}_{component}_epsilon"] = values
        arrays[f"{name}_{component}_delta"] = delta
        arrays[f"{name}_{component}_delta_over_rho"] = ratio
        # Retain the full central-monitor raw component arrays for the minimal
        # case and the exact sampled component axes for both cases.
        arrays[f"{name}_{component}_raw_epsilon_all_rhos"] = np.stack(
            [_epsilon(detail, component) for detail in details], axis=0
        )
        for axis_index, axis in enumerate(COMPONENTS):
            coordinate_axes = [component_coordinates(detail, component)[axis_index]
                               for detail in details]
            arrays[f"{name}_{component}_{axis}_coordinate_m_all_rhos"] = np.stack(
                coordinate_axes, axis=0
            )
        component_records = []
        for row_index, rho in enumerate(rhos):
            error = values[row_index] - analytic[row_index]
            row = {
                "case": name,
                "component": component,
                "rho": float(rho),
                "x_m": float(coordinates[row_index, 0]),
                "y_m": float(coordinates[row_index, 1]),
                "z_m": float(coordinates[row_index, 2]),
                "epsilon_lumerical": _complex_json(values[row_index]),
                "epsilon_material": _complex_json(analytic[row_index]),
                "epsilon_error": _complex_json(error),
                "delta_from_exact_zero": _complex_json(delta[row_index]),
                "delta_over_rho": (
                    _complex_json(ratio[row_index]) if row_index > 0 else None
                ),
            }
            component_records.append(row)
            table.append(row)
        records["components"][component] = {
            "representative_array_index_by_rho": indices.tolist(),
            "representative_coordinate_m_by_rho": coordinates.tolist(),
            "samples": component_records,
            "small_positive_linear_fit": fit,
        }
    return records, arrays, table


def _choose_exact_zero_node(
    rho: np.ndarray,
    *,
    preferred_xy_m: tuple[float, float] | None = None,
) -> tuple[int, int, dict[str, Any]]:
    candidates = np.argwhere(rho == 0.0)
    if candidates.size == 0:
        raise RuntimeError("nonuniform density has no exact-zero node")
    x, y, _ = density_nodes()
    best: tuple[float, float, int, int, np.ndarray] | None = None
    for ix, iy in candidates:
        if ix < 2 or iy < 2 or ix >= rho.shape[0] - 2 or iy >= rho.shape[1] - 2:
            continue
        neighborhood = rho[ix - 1 : ix + 2, iy - 1 : iy + 2]
        score = float(np.max(neighborhood))
        distance = (
            float(np.hypot(x[ix] - preferred_xy_m[0], y[iy] - preferred_xy_m[1]))
            if preferred_xy_m is not None
            else 0.0
        )
        candidate = (
            -distance,
            score,
            int(ix),
            int(iy),
            np.array(neighborhood, copy=True),
        )
        if best is None or candidate[:4] > best[:4]:
            best = candidate
    if best is None:
        raise RuntimeError("no exact-zero node is sufficiently far from design boundary")
    ix, iy = best[2], best[3]
    return ix, iy, {
        "selection": (
            "interior exact-zero node nearest group-screen Ex maximum; neighbor maximum breaks ties"
            if preferred_xy_m is not None
            else "interior exact-zero node with largest 3x3 neighbor maximum"
        ),
        "preferred_group_screen_coordinate_xy_m": (
            list(preferred_xy_m) if preferred_xy_m is not None else None
        ),
        "distance_to_preferred_coordinate_m": -best[0],
        "neighbor_maximum_rho": best[1],
        "node_index_xy": [ix, iy],
        "node_coordinate_m": [float(x[ix]), float(y[iy])],
        "neighborhood_3x3": best[4].tolist(),
    }


def _remap_component_to_reference_grid(
    reference: Mapping[str, np.ndarray],
    current: Mapping[str, np.ndarray],
    component: str,
) -> tuple[np.ndarray, float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    reference_axes = component_coordinates(reference, component)
    current_axes = component_coordinates(current, component)
    mappings = tuple(
        np.argmin(np.abs(current_axis[:, None] - reference_axis[None, :]), axis=0)
        for reference_axis, current_axis in zip(
            reference_axes, current_axes, strict=True
        )
    )
    maximum_mapping_distance = max(
        float(np.max(np.abs(current_axis[mapping] - reference_axis)))
        for reference_axis, current_axis, mapping in zip(
            reference_axes, current_axes, mappings, strict=True
        )
    )
    remapped = _epsilon(current, component)[np.ix_(*mappings)]
    return remapped, maximum_mapping_distance, mappings


def _exact_zero_group_screen(
    *,
    base_rho: np.ndarray,
    baseline: Mapping[str, np.ndarray],
    evaluate_density: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray], tuple[float, float]]:
    """Reproduce the old simultaneous-zero direction and audit grid alignment."""

    exact_zero = base_rho == 0.0
    details: list[dict[str, np.ndarray]] = []
    for value in EXACT_ZERO_GROUP_SCREEN_RHOS:
        state = np.array(base_rho, copy=True)
        state[exact_zero] = float(value)
        details.append(evaluate_density(state))
    comparisons = [_grid_comparison(baseline, detail) for detail in details]
    records: dict[str, Any] = {
        "description": "all exact-zero nodes raised simultaneously, matching the old diagnostic direction",
        "exact_zero_node_count": int(np.count_nonzero(exact_zero)),
        "rho_values": EXACT_ZERO_GROUP_SCREEN_RHOS.tolist(),
        "grid_comparison_to_exact_zero": [
            {"rho": float(rho), **comparison}
            for rho, comparison in zip(
                EXACT_ZERO_GROUP_SCREEN_RHOS, comparisons, strict=True
            )
        ],
        "components": {},
    }
    arrays: dict[str, np.ndarray] = {
        "exact_zero_group_screen_rho": np.array(
            EXACT_ZERO_GROUP_SCREEN_RHOS, copy=True
        ),
    }
    preferred_xy: tuple[float, float] | None = None
    for component in COMPONENTS:
        coordinate_cubes = _local_coordinate_arrays(baseline, component)
        design_region = (
            (np.abs(coordinate_cubes[0]) <= 0.5 * CONTRACT.design_span_x_m + 5.0e-9)
            & (np.abs(coordinate_cubes[1]) <= 0.5 * CONTRACT.design_span_y_m + 5.0e-9)
            & (coordinate_cubes[2] >= -5.0e-9)
            & (coordinate_cubes[2] <= CONTRACT.design_thickness_m + 5.0e-9)
        )
        region_flat = np.flatnonzero(design_region)
        reference_coordinates = np.column_stack(
            [cube.reshape(-1)[region_flat] for cube in coordinate_cubes]
        )
        baseline_region = _epsilon(baseline, component).reshape(-1)[region_flat]
        naive_values: list[np.ndarray] = []
        remapped_values: list[np.ndarray] = []
        mapping_distances: list[float] = []
        samples: list[dict[str, Any]] = []
        for rho, detail in zip(EXACT_ZERO_GROUP_SCREEN_RHOS, details, strict=True):
            naive_region = _epsilon(detail, component).reshape(-1)[region_flat]
            remapped, mapping_distance, _ = _remap_component_to_reference_grid(
                baseline, detail, component
            )
            remapped_region = remapped.reshape(-1)[region_flat]
            naive_delta = naive_region - baseline_region
            remapped_delta = remapped_region - baseline_region
            naive_worst = int(np.argmax(np.abs(naive_delta)))
            remapped_worst = int(np.argmax(np.abs(remapped_delta)))
            samples.append({
                "rho": float(rho),
                "maximum_grid_mapping_distance_m": mapping_distance,
                "naive_same_array_index": {
                    "maximum_abs_delta": float(abs(naive_delta[naive_worst])),
                    "maximum_abs_delta_over_rho": float(
                        abs(naive_delta[naive_worst] / rho)
                    ),
                    "coordinate_m": reference_coordinates[naive_worst].tolist(),
                    "epsilon_zero": _complex_json(baseline_region[naive_worst]),
                    "epsilon_positive": _complex_json(naive_region[naive_worst]),
                    "delta": _complex_json(naive_delta[naive_worst]),
                    "delta_over_rho": _complex_json(naive_delta[naive_worst] / rho),
                },
                "nearest_physical_coordinate_remap": {
                    "maximum_abs_delta": float(abs(remapped_delta[remapped_worst])),
                    "maximum_abs_delta_over_rho": float(
                        abs(remapped_delta[remapped_worst] / rho)
                    ),
                    "coordinate_m": reference_coordinates[remapped_worst].tolist(),
                    "epsilon_zero": _complex_json(baseline_region[remapped_worst]),
                    "epsilon_positive": _complex_json(remapped_region[remapped_worst]),
                    "delta": _complex_json(remapped_delta[remapped_worst]),
                    "delta_over_rho": _complex_json(
                        remapped_delta[remapped_worst] / rho
                    ),
                },
            })
            if component == "x" and rho == EXACT_ZERO_GROUP_SCREEN_RHOS[0]:
                preferred_xy = (
                    float(reference_coordinates[naive_worst, 0]),
                    float(reference_coordinates[naive_worst, 1]),
                )
            naive_values.append(naive_region)
            remapped_values.append(remapped_region)
            mapping_distances.append(mapping_distance)
        naive_values_array = np.stack(naive_values, axis=0)
        remapped_values_array = np.stack(remapped_values, axis=0)
        scaling_records: dict[str, Any] = {}
        for label, values_array in (
            ("naive_same_array_index", naive_values_array),
            ("nearest_physical_coordinate_remap", remapped_values_array),
        ):
            delta_array = values_array - baseline_region
            worst = int(np.argmax(np.abs(delta_array[0])))
            first = abs(delta_array[0, worst])
            second = abs(delta_array[1, worst])
            delta_growth = float(second / first) if first > 0.0 else float("inf")
            ratio_quotient = float(
                (first / EXACT_ZERO_GROUP_SCREEN_RHOS[0])
                / (second / EXACT_ZERO_GROUP_SCREEN_RHOS[1])
            ) if second > 0.0 else float("inf")
            scaling_records[label] = {
                "same_sample_coordinate_m": reference_coordinates[worst].tolist(),
                "delta_abs_at_rho_1e_7": float(first),
                "delta_abs_at_rho_1e_6": float(second),
                "delta_growth_1e_6_over_1e_7": delta_growth,
                "delta_over_rho_1e_7_over_1e_6": ratio_quotient,
                "fixed_jump_scaling_signature": bool(
                    0.5 <= delta_growth <= 2.0 and ratio_quotient >= 5.0
                ),
                "smooth_linear_scaling_signature": bool(
                    5.0 <= delta_growth <= 20.0
                    and 0.5 <= ratio_quotient <= 2.0
                ),
            }
        arrays[f"exact_zero_group_{component}_region_flat_indices"] = region_flat
        arrays[f"exact_zero_group_{component}_reference_coordinates_m"] = reference_coordinates
        arrays[f"exact_zero_group_{component}_epsilon_zero"] = baseline_region
        arrays[f"exact_zero_group_{component}_epsilon_positive_naive"] = naive_values_array
        arrays[f"exact_zero_group_{component}_epsilon_positive_remapped"] = remapped_values_array
        arrays[f"exact_zero_group_{component}_delta_naive"] = naive_values_array - baseline_region
        arrays[f"exact_zero_group_{component}_delta_remapped"] = remapped_values_array - baseline_region
        arrays[f"exact_zero_group_{component}_maximum_mapping_distance_m"] = np.asarray(
            mapping_distances, dtype=np.float64
        )
        records["components"][component] = {
            "samples": samples,
            "same_sample_scaling": scaling_records,
        }
    if preferred_xy is None:
        raise RuntimeError("exact-zero group screen did not select an Ex coordinate")
    records["single_node_target_coordinate_xy_m"] = list(preferred_xy)
    return records, arrays, preferred_xy


def _local_coordinate_arrays(
    detail: Mapping[str, np.ndarray], component: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = component_coordinates(detail, component)
    return np.meshgrid(*axes, indexing="ij", sparse=False)


def _nonuniform_case(
    *,
    base_rho: np.ndarray,
    rhos: np.ndarray,
    evaluate_density: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    baseline_detail = evaluate_density(base_rho)
    group_records, group_arrays, preferred_xy = _exact_zero_group_screen(
        base_rho=base_rho,
        baseline=baseline_detail,
        evaluate_density=evaluate_density,
    )
    ix, iy, node = _choose_exact_zero_node(
        base_rho, preferred_xy_m=preferred_xy
    )
    x_node, y_node = node["node_coordinate_m"]
    details: list[dict[str, np.ndarray]] = [baseline_detail]
    for value in rhos[1:]:
        state = np.array(base_rho, copy=True)
        state[ix, iy] = float(value)
        details.append(evaluate_density(state))
    baseline_audit = validate_index_detail(details[0])
    grid_comparisons = [_grid_comparison(details[0], detail) for detail in details]

    arrays: dict[str, np.ndarray] = {
        **group_arrays,
        "nonuniform_rho": np.array(rhos, copy=True),
        "nonuniform_base_density": np.array(base_rho, copy=True),
        "nonuniform_node_index_xy": np.asarray([ix, iy], int),
        "nonuniform_node_coordinate_m": np.asarray([x_node, y_node], float),
    }
    records: dict[str, Any] = {
        "rho_values": rhos.tolist(),
        "selected_node": node,
        "simultaneous_exact_zero_group_screen": group_records,
        "baseline_grid": baseline_audit,
        "grid_comparison_to_exact_zero_by_rho": [
            {"rho": float(rho), **comparison}
            for rho, comparison in zip(rhos, grid_comparisons, strict=True)
        ],
        "components": {},
    }
    table: list[dict[str, Any]] = []
    for component in COMPONENTS:
        coordinate_cubes = _local_coordinate_arrays(details[0], component)
        local = (
            (np.abs(coordinate_cubes[0] - x_node) <= 225.0e-9)
            & (np.abs(coordinate_cubes[1] - y_node) <= 225.0e-9)
            & (coordinate_cubes[2] >= -5.0e-9)
            & (coordinate_cubes[2] <= CONTRACT.design_thickness_m + 5.0e-9)
        )
        candidate_flat_indices = np.flatnonzero(local)
        if candidate_flat_indices.size == 0:
            raise RuntimeError(f"nonuniform component {component} has no local samples")
        candidate_reference_coordinates = np.column_stack(
            [cube.reshape(-1)[candidate_flat_indices] for cube in coordinate_cubes]
        )
        candidate_values_by_rho: list[np.ndarray] = []
        candidate_mapped_indices_by_rho: list[np.ndarray] = []
        candidate_actual_coordinates_by_rho: list[np.ndarray] = []
        for detail in details:
            axes = component_coordinates(detail, component)
            mapped_axis_indices = np.column_stack(
                [
                    np.argmin(
                        np.abs(axis_values[:, None] - candidate_reference_coordinates[None, :, axis_index]),
                        axis=0,
                    )
                    for axis_index, axis_values in enumerate(axes)
                ]
            )
            mapped_flat_indices = np.ravel_multi_index(
                tuple(mapped_axis_indices[:, axis] for axis in range(3)),
                _epsilon(detail, component).shape,
            )
            actual_coordinates = np.column_stack(
                [
                    axes[axis][mapped_axis_indices[:, axis]]
                    for axis in range(3)
                ]
            )
            candidate_values_by_rho.append(
                _epsilon(detail, component).reshape(-1)[mapped_flat_indices]
            )
            candidate_mapped_indices_by_rho.append(mapped_flat_indices)
            candidate_actual_coordinates_by_rho.append(actual_coordinates)
        candidate_values = np.stack(candidate_values_by_rho, axis=0)
        candidate_mapped_indices = np.stack(candidate_mapped_indices_by_rho, axis=0)
        candidate_actual_coordinates = np.stack(
            candidate_actual_coordinates_by_rho, axis=0
        )
        difference = candidate_values[1:] - candidate_values[0]
        change_scale = float(np.max(np.abs(difference)))
        threshold = max(1.0e-12, 1.0e-13 * change_scale)
        affected_candidates = np.max(np.abs(difference), axis=0) > threshold
        affected_candidate_indices = np.flatnonzero(affected_candidates)
        if affected_candidate_indices.size == 0:
            raise RuntimeError(f"nonuniform component {component} has no local response")
        flat_indices = candidate_flat_indices[affected_candidate_indices]
        local_values = candidate_values[:, affected_candidate_indices]
        local_delta = local_values - local_values[0]
        local_ratio = local_delta[1:] / rhos[1:, None]
        flat_coordinates = candidate_reference_coordinates[affected_candidate_indices]
        actual_coordinates = candidate_actual_coordinates[:, affected_candidate_indices]
        mapped_indices = candidate_mapped_indices[:, affected_candidate_indices]
        fit_intercept = np.empty(flat_indices.size, np.complex128)
        fit_slope = np.empty(flat_indices.size, np.complex128)
        fit_residual = np.empty(flat_indices.size, np.float64)
        for sample_index in range(flat_indices.size):
            fit = _fit_complex(rhos, local_values[:, sample_index])
            fit_intercept[sample_index] = complex(*fit["intercept_zero_plus"])
            fit_slope[sample_index] = complex(*fit["slope"])
            fit_residual[sample_index] = float(fit["maximum_abs_residual"])
        fit_delta = fit_intercept - local_values[0]
        worst = int(np.argmax(np.abs(fit_delta)))
        arrays[f"nonuniform_{component}_local_candidate_flat_indices"] = candidate_flat_indices
        arrays[f"nonuniform_{component}_local_candidate_reference_coordinates_m"] = candidate_reference_coordinates
        arrays[f"nonuniform_{component}_local_candidate_mapped_indices_all_rhos"] = candidate_mapped_indices
        arrays[f"nonuniform_{component}_local_candidate_actual_coordinates_m_all_rhos"] = candidate_actual_coordinates
        arrays[f"nonuniform_{component}_local_candidate_raw_epsilon_all_rhos"] = candidate_values
        arrays[f"nonuniform_{component}_flat_indices"] = flat_indices
        arrays[f"nonuniform_{component}_reference_coordinates_m"] = flat_coordinates
        arrays[f"nonuniform_{component}_mapped_indices_all_rhos"] = mapped_indices
        arrays[f"nonuniform_{component}_actual_coordinates_m_all_rhos"] = actual_coordinates
        arrays[f"nonuniform_{component}_epsilon"] = local_values
        arrays[f"nonuniform_{component}_delta"] = local_delta
        arrays[f"nonuniform_{component}_delta_over_rho"] = local_ratio
        arrays[f"nonuniform_{component}_fit_intercept_zero_plus"] = fit_intercept
        arrays[f"nonuniform_{component}_fit_slope"] = fit_slope
        arrays[f"nonuniform_{component}_fit_delta"] = fit_delta
        arrays[f"nonuniform_{component}_fit_max_abs_residual"] = fit_residual
        sample_records: list[dict[str, Any]] = []
        for sample_index, flat_index in enumerate(flat_indices):
            for rho_index, rho in enumerate(rhos):
                row = {
                    "component": component,
                    "local_sample_index": int(sample_index),
                    "flat_array_index_at_rho_zero": int(flat_index),
                    "mapped_flat_array_index": int(mapped_indices[rho_index, sample_index]),
                    "rho": float(rho),
                    "x_m": float(actual_coordinates[rho_index, sample_index, 0]),
                    "y_m": float(actual_coordinates[rho_index, sample_index, 1]),
                    "z_m": float(actual_coordinates[rho_index, sample_index, 2]),
                    "epsilon_lumerical": _complex_json(local_values[rho_index, sample_index]),
                    "delta_from_exact_zero": _complex_json(local_delta[rho_index, sample_index]),
                    "delta_over_rho": (
                        _complex_json(local_delta[rho_index, sample_index] / rho)
                        if rho_index > 0
                        else None
                    ),
                }
                sample_records.append(row)
                table.append(row)
        records["components"][component] = {
            "local_candidate_sample_count": int(candidate_flat_indices.size),
            "affected_local_sample_count": int(flat_indices.size),
            "response_threshold": threshold,
            "sample_mapping": "nearest native component-Yee sample to each rho=0 physical coordinate",
            "worst_fitted_jump_sample": {
                "local_sample_index": worst,
                "flat_array_index": int(flat_indices[worst]),
                "coordinate_m": flat_coordinates[worst].tolist(),
                "actual_epsilon_zero": _complex_json(local_values[0, worst]),
                "extrapolated_epsilon_zero_plus": _complex_json(fit_intercept[worst]),
                "delta_zero_plus_minus_actual_zero": _complex_json(fit_delta[worst]),
                "delta_magnitude": float(abs(fit_delta[worst])),
                "slope": _complex_json(fit_slope[worst]),
                "maximum_abs_fit_residual": float(fit_residual[worst]),
                "samples": [
                    row
                    for row in sample_records
                    if row["local_sample_index"] == worst
                ],
            },
        }
    return records, arrays, table


def _write_uniform_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case", "component", "rho", "x_m", "y_m", "z_m",
        "epsilon_lum_real", "epsilon_lum_imag",
        "epsilon_material_real", "epsilon_material_imag",
        "error_real", "error_imag", "delta_real", "delta_imag",
        "delta_over_rho_real", "delta_over_rho_imag",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            ratio = row["delta_over_rho"]
            writer.writerow({
                "case": row["case"], "component": row["component"],
                "rho": row["rho"], "x_m": row["x_m"], "y_m": row["y_m"],
                "z_m": row["z_m"],
                "epsilon_lum_real": row["epsilon_lumerical"][0],
                "epsilon_lum_imag": row["epsilon_lumerical"][1],
                "epsilon_material_real": row["epsilon_material"][0],
                "epsilon_material_imag": row["epsilon_material"][1],
                "error_real": row["epsilon_error"][0],
                "error_imag": row["epsilon_error"][1],
                "delta_real": row["delta_from_exact_zero"][0],
                "delta_imag": row["delta_from_exact_zero"][1],
                "delta_over_rho_real": ratio[0] if ratio is not None else "",
                "delta_over_rho_imag": ratio[1] if ratio is not None else "",
            })


def _write_nonuniform_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "component", "local_sample_index", "flat_array_index_at_rho_zero",
        "mapped_flat_array_index", "rho",
        "x_m", "y_m", "z_m", "epsilon_real", "epsilon_imag",
        "delta_real", "delta_imag", "delta_over_rho_real", "delta_over_rho_imag",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            ratio = row["delta_over_rho"]
            writer.writerow({
                "component": row["component"],
                "local_sample_index": row["local_sample_index"],
                "flat_array_index_at_rho_zero": row["flat_array_index_at_rho_zero"],
                "mapped_flat_array_index": row["mapped_flat_array_index"],
                "rho": row["rho"],
                "x_m": row["x_m"], "y_m": row["y_m"], "z_m": row["z_m"],
                "epsilon_real": row["epsilon_lumerical"][0],
                "epsilon_imag": row["epsilon_lumerical"][1],
                "delta_real": row["delta_from_exact_zero"][0],
                "delta_imag": row["delta_from_exact_zero"][1],
                "delta_over_rho_real": ratio[0] if ratio is not None else "",
                "delta_over_rho_imag": ratio[1] if ratio is not None else "",
            })


def _plot_uniform(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    colors = {"minimal_bulk_uniform": "tab:blue", "production_stack_uniform": "tab:orange"}
    for column, component in enumerate(COMPONENTS):
        for case, color in colors.items():
            rho = arrays[f"{case}_rho"][1:]
            delta = arrays[f"{case}_{component}_delta"][1:]
            ratio = arrays[f"{case}_{component}_delta_over_rho"][1:]
            axes[0, column].plot(rho, np.abs(delta), "o-", color=color, label=case)
            axes[1, column].plot(rho, np.abs(ratio), "o-", color=color, label=case)
        axes[0, column].set_xscale("log")
        axes[0, column].set_yscale("log")
        axes[0, column].set_title(f"E{component}: |epsilon(rho)-epsilon(0)|")
        axes[0, column].set_xlabel("rho")
        axes[0, column].set_ylabel("|Delta epsilon|")
        axes[0, column].grid(True, which="both", alpha=0.3)
        axes[1, column].set_xscale("log")
        axes[1, column].set_yscale("log")
        axes[1, column].set_title(f"E{component}: |Delta epsilon/rho|")
        axes[1, column].set_xlabel("rho")
        axes[1, column].set_ylabel("|Delta epsilon/rho|")
        axes[1, column].grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Lumerical layout-only uniform rho endpoint measurement")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_nonuniform(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    rho = arrays["nonuniform_rho"]
    for column, component in enumerate(COMPONENTS):
        fit_delta = arrays[f"nonuniform_{component}_fit_delta"]
        worst = int(np.argmax(np.abs(fit_delta)))
        delta = arrays[f"nonuniform_{component}_delta"][:, worst]
        ratio = arrays[f"nonuniform_{component}_delta_over_rho"][:, worst]
        axes[0, column].plot(rho[1:], delta[1:].real, "o-", label="real")
        axes[0, column].plot(rho[1:], delta[1:].imag, "s-", label="imag")
        axes[1, column].plot(rho[1:], ratio.real, "o-", label="real")
        axes[1, column].plot(rho[1:], ratio.imag, "s-", label="imag")
        for row in axes[:, column]:
            row.set_xscale("log")
            row.grid(True, which="both", alpha=0.3)
            row.set_xlabel("single-node rho")
        axes[0, column].set_title(f"E{component} worst fit-jump sample: Delta epsilon")
        axes[1, column].set_title(f"E{component} worst fit-jump sample: Delta epsilon/rho")
    axes[0, 0].legend()
    axes[1, 0].legend()
    figure.suptitle("Lumerical layout-only nonuniform exact-zero node measurement")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_exact_zero_group(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    rho = arrays["exact_zero_group_screen_rho"]
    for column, component in enumerate(COMPONENTS):
        for suffix, label, marker in (
            ("naive", "same array index", "o-"),
            ("remapped", "physical-coordinate remap", "s--"),
        ):
            delta_all = arrays[f"exact_zero_group_{component}_delta_{suffix}"]
            worst = int(np.argmax(np.abs(delta_all[0])))
            delta = delta_all[:, worst]
            axes[0, column].plot(rho, np.abs(delta), marker, label=label)
            axes[1, column].plot(rho, np.abs(delta / rho), marker, label=label)
        for row in axes[:, column]:
            row.set_xscale("log")
            row.set_yscale("log")
            row.grid(True, which="both", alpha=0.3)
            row.set_xlabel("rho assigned to all exact-zero nodes")
        axes[0, column].set_title(f"E{component}: |Delta epsilon|")
        axes[1, column].set_title(f"E{component}: |Delta epsilon/rho|")
    axes[0, 0].legend()
    axes[1, 0].legend()
    figure.suptitle("Lumerical layout-only simultaneous exact-zero-node direction")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    def format_complex(value: list[float] | None) -> str:
        if value is None:
            return "--"
        return f"{value[0]:+.9e}{value[1]:+.9e}i"

    lines = [
        "# Lumerical rho=0 Yee-epsilon jump measurement",
        "",
        "This report contains layout/index-only measurements. Maxwell solves: **0**.",
        "",
        "## Small-positive extrapolated jump",
        "",
        "| case | component | actual epsilon(0) | extrapolated epsilon(0+) | delta | |delta| |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case_name in ("minimal_bulk_uniform", "production_stack_uniform"):
        case = result[case_name]
        for component in COMPONENTS:
            fit = case["components"][component]["small_positive_linear_fit"]
            lines.append(
                f"| {case_name} | E{component} | {fit['actual_epsilon_zero']} | "
                f"{fit['intercept_zero_plus']} | "
                f"{fit['delta_zero_plus_minus_actual_zero']} | {fit['delta_magnitude']:.9e} |"
            )
    nonuniform = result["nonuniform_exact_zero_node"]
    for component in COMPONENTS:
        worst = nonuniform["components"][component]["worst_fitted_jump_sample"]
        lines.append(
            f"| nonuniform exact-zero node | E{component} | "
            f"{worst['actual_epsilon_zero']} | {worst['extrapolated_epsilon_zero_plus']} | "
            f"{worst['delta_zero_plus_minus_actual_zero']} | {worst['delta_magnitude']:.9e} |"
        )
    lines.extend([
        "",
        "## Uniform representative raw complex samples",
        "",
        "Each component is sampled at the Yee point nearest the requested interior physical coordinate.",
        "",
        "| case | comp. | rho | epsilon Lum | epsilon material | Lum-material | Delta from rho=0 | Delta/rho |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for case_name in ("minimal_bulk_uniform", "production_stack_uniform"):
        for component in COMPONENTS:
            for sample in result[case_name]["components"][component]["samples"]:
                lines.append(
                    f"| {case_name} | E{component} | {sample['rho']:.1e} | "
                    f"{format_complex(sample['epsilon_lumerical'])} | "
                    f"{format_complex(sample['epsilon_material'])} | "
                    f"{format_complex(sample['epsilon_error'])} | "
                    f"{format_complex(sample['delta_from_exact_zero'])} | "
                    f"{format_complex(sample['delta_over_rho'])} |"
                )
    group = nonuniform["simultaneous_exact_zero_group_screen"]
    lines.extend([
        "",
        "## Simultaneous exact-zero-node direction",
        "",
        "This reproduces the old direction that raised all 63 exact-zero nodes together. The same physical sample is compared at rho=1e-7 and 1e-6.",
        "",
        "| comp. | comparison | |Delta|(1e-7) | |Delta|(1e-6) | Delta growth | (|Delta/rho| at 1e-7)/(at 1e-6) | fixed-jump signature |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for component in COMPONENTS:
        scaling = group["components"][component]["same_sample_scaling"]
        for comparison in (
            "naive_same_array_index",
            "nearest_physical_coordinate_remap",
        ):
            item = scaling[comparison]
            lines.append(
                f"| E{component} | {comparison} | "
                f"{item['delta_abs_at_rho_1e_7']:.9e} | "
                f"{item['delta_abs_at_rho_1e_6']:.9e} | "
                f"{item['delta_growth_1e_6_over_1e_7']:.9e} | "
                f"{item['delta_over_rho_1e_7_over_1e_6']:.9e} | "
                f"{item['fixed_jump_scaling_signature']} |"
            )
    lines.extend([
        "",
        "## Nonuniform beta=2 worst local samples",
        "",
        "All affected local samples are in the CSV/NPZ; this table shows the largest fitted-intercept-offset sample per component.",
        "",
        "| comp. | rho | epsilon Lum | Delta from rho=0 | Delta/rho |",
        "|---|---:|---:|---:|---:|",
    ])
    for component in COMPONENTS:
        worst = nonuniform["components"][component]["worst_fitted_jump_sample"]
        for sample in worst["samples"]:
            lines.append(
                f"| E{component} | {sample['rho']:.1e} | "
                f"{format_complex(sample['epsilon_lumerical'])} | "
                f"{format_complex(sample['delta_from_exact_zero'])} | "
                f"{format_complex(sample['delta_over_rho'])} |"
            )
    lines.extend([
        "",
        "## Coordinate-grid drift",
        "",
        "| case | maximum absolute coordinate difference over positive rho (m) |",
        "|---|---:|",
    ])
    for case_name in (
        "minimal_bulk_uniform",
        "production_stack_uniform",
        "nonuniform_exact_zero_node",
    ):
        comparisons = result[case_name]["grid_comparison_to_exact_zero_by_rho"]
        maximum = max(
            float(item["maximum_absolute_coordinate_difference_m"])
            for item in comparisons[1:]
        )
        lines.append(f"| {case_name} | {maximum:.9e} |")
    conclusion = result["conclusion"]
    lines.extend([
        "",
        "## Evidence conclusion",
        "",
        f"- Any directly observed fixed jump: **{conclusion['any_fixed_jump_directly_observed']}**",
        f"- Stage localization: {conclusion['stage_localization']}",
        f"- Component classifications: `{conclusion['fixed_jump_directly_observed_by_residual_separation']}`",
        f"- Symmetric floor excludes exact zero during relaxed optimization: **{conclusion['floor_3e_5_excludes_exact_zero_during_relaxed_optimization']}**",
        f"- This experiment demonstrates that the floor avoids a measured fixed jump: **{conclusion['floor_3e_5_avoids_a_fixed_jump_measured_by_this_experiment']}**",
        f"- Floor evidence note: {conclusion['floor_evidence_note']}",
    ])
    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- Raw complex arrays: `{result['artifacts']['raw_npz']['path']}`",
        f"- Uniform table: `{result['artifacts']['uniform_csv']['path']}`",
        f"- Nonuniform local table: `{result['artifacts']['nonuniform_csv']['path']}`",
        f"- Uniform plot: `{result['artifacts']['uniform_plot']['path']}`",
        f"- Nonuniform plot: `{result['artifacts']['nonuniform_plot']['path']}`",
        f"- Simultaneous exact-zero group plot: `{result['artifacts']['exact_zero_group_plot']['path']}`",
        "",
        "The evidence-based interpretation is written after the numeric run in the JSON conclusion block.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _jump_conclusion(result: Mapping[str, Any]) -> dict[str, Any]:
    minimal = {
        component: float(
            result["minimal_bulk_uniform"]["components"][component]
            ["small_positive_linear_fit"]["delta_magnitude"]
        )
        for component in COMPONENTS
    }
    stack = {
        component: float(
            result["production_stack_uniform"]["components"][component]
            ["small_positive_linear_fit"]["delta_magnitude"]
        )
        for component in COMPONENTS
    }
    nonuniform = {
        component: float(
            result["nonuniform_exact_zero_node"]["components"][component]
            ["worst_fitted_jump_sample"]["delta_magnitude"]
        )
        for component in COMPONENTS
    }
    group = result["nonuniform_exact_zero_node"][
        "simultaneous_exact_zero_group_screen"
    ]
    group_naive = {
        component: float(
            group["components"][component]["same_sample_scaling"]
            ["naive_same_array_index"]["delta_abs_at_rho_1e_7"]
        )
        for component in COMPONENTS
    }
    group_remapped = {
        component: float(
            group["components"][component]["same_sample_scaling"]
            ["nearest_physical_coordinate_remap"]["delta_abs_at_rho_1e_7"]
        )
        for component in COMPONENTS
    }
    # This is an evidence classifier, not an assumed physical threshold.  It
    # compares the fitted intercept offset with the measured fit residual.
    observed: dict[str, dict[str, bool]] = {}
    for case_name in ("minimal_bulk_uniform", "production_stack_uniform"):
        observed[case_name] = {}
        for component in COMPONENTS:
            fit = result[case_name]["components"][component]["small_positive_linear_fit"]
            observed[case_name][component] = bool(
                fit["delta_magnitude"] > 10.0 * max(fit["maximum_abs_residual"], 1.0e-12)
            )
    observed["nonuniform_exact_zero_node"] = {}
    for component in COMPONENTS:
        worst = result["nonuniform_exact_zero_node"]["components"][component][
            "worst_fitted_jump_sample"
        ]
        observed["nonuniform_exact_zero_node"][component] = bool(
            worst["delta_magnitude"]
            > 10.0 * max(worst["maximum_abs_fit_residual"], 1.0e-12)
        )
    observed["simultaneous_exact_zero_group_naive_same_array_index"] = {
        component: bool(
            group["components"][component]["same_sample_scaling"]
            ["naive_same_array_index"]["fixed_jump_scaling_signature"]
        )
        for component in COMPONENTS
    }
    observed["simultaneous_exact_zero_group_nearest_physical_coordinate_remap"] = {
        component: bool(
            group["components"][component]["same_sample_scaling"]
            ["nearest_physical_coordinate_remap"]["fixed_jump_scaling_signature"]
        )
        for component in COMPONENTS
    }
    any_observed = bool(
        any(value for case in observed.values() for value in case.values())
    )
    physical_jump_observed = bool(
        any(observed["nonuniform_exact_zero_node"].values())
        or any(
            observed[
                "simultaneous_exact_zero_group_nearest_physical_coordinate_remap"
            ].values()
        )
    )
    group_naive_any = any(
        observed["simultaneous_exact_zero_group_naive_same_array_index"].values()
    )
    group_remapped_any = any(
        observed[
            "simultaneous_exact_zero_group_nearest_physical_coordinate_remap"
        ].values()
    )
    if group_naive_any and not group_remapped_any:
        stage_localization = (
            "The old simultaneous-zero signature is removed by physical-coordinate remapping, "
            "so it enters at array-index/grid alignment rather than the realized epsilon field."
        )
    elif group_remapped_any:
        stage_localization = (
            "The simultaneous-zero signature survives physical-coordinate remapping; with bulk and "
            "uniform-stack tests smooth, it enters in nonuniform/interface material realization."
        )
    else:
        stage_localization = (
            "No fixed-jump scaling is present in bulk, uniform-stack, targeted single-node, or "
            "the simultaneous exact-zero direction; the earlier signature is not reproduced."
        )
    return {
        "jump_magnitude_by_case_and_component": {
            "minimal_bulk_uniform": minimal,
            "production_stack_uniform": stack,
            "nonuniform_exact_zero_node_worst_local_sample": nonuniform,
            "simultaneous_exact_zero_group_naive_at_rho_1e_7": group_naive,
            "simultaneous_exact_zero_group_remapped_at_rho_1e_7": group_remapped,
        },
        "fixed_jump_directly_observed_by_residual_separation": observed,
        "classification_rule": (
            "uniform/single: |fitted intercept - actual zero| > 10 * max(fit max residual, 1e-12); "
            "two-step group: delta growth in [0.5,2] while |delta/rho|(1e-7)/|delta/rho|(1e-6) >= 5"
        ),
        "any_fixed_jump_directly_observed": any_observed,
        "physical_coordinate_fixed_jump_directly_observed": physical_jump_observed,
        "stage_localization": stage_localization,
        "floor_3e_5_excludes_exact_zero_during_relaxed_optimization": True,
        "floor_3e_5_avoids_a_fixed_jump_measured_by_this_experiment": bool(
            physical_jump_observed
        ),
        "floor_scope": "relaxed optimization only; exact binary is independently reevaluated",
        "floor_evidence_note": (
            "A physical-coordinate Ex jump was measured between exact rho=0 and rho->0+, while "
            "the sampled positive branch from 1e-7 through and beyond 3e-5 is smooth.  The 3e-5 "
            "floor therefore lies on the smooth positive branch and excludes the measured endpoint "
            "transition during relaxed optimization."
            if physical_jump_observed
            else "No physical-coordinate fixed jump was measured, so this experiment only confirms "
            "that the floor excludes exact-zero evaluations; it does not justify the floor as a "
            "jump remedy."
        ),
        "interpretation_note": (
            "A bulk-uniform false and production/nonuniform true pattern locates the discontinuity "
            "after analytic n-k interpolation, in Lumerical spatial/interface/conformal assignment. "
            "A true bulk-uniform result instead implicates importnk2/index realization itself."
        ),
    }


def main() -> int:
    require_lumerical_only_source_boundary()
    args = _parse_args()
    forward_project = args.forward_project.expanduser().resolve()
    density_file = args.density_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not forward_project.is_file():
        raise FileNotFoundError(forward_project)
    if not density_file.is_file():
        raise FileNotFoundError(density_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.relative_to(REPOSITORY.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError("diagnostic output must be outside the Git worktree")

    result_json = output_dir / "yee_zero_jump_measurement.json"
    raw_npz = output_dir / "yee_zero_jump_raw_complex.npz"
    uniform_csv = output_dir / "uniform_representative_samples.csv"
    nonuniform_csv = output_dir / "nonuniform_local_yee_samples.csv"
    uniform_plot = output_dir / "uniform_delta_and_delta_over_rho.png"
    nonuniform_plot = output_dir / "nonuniform_delta_and_delta_over_rho.png"
    group_plot = output_dir / "exact_zero_group_delta_and_delta_over_rho.png"
    report_md = output_dir / "YEE_ZERO_JUMP_MEASUREMENT.md"
    minimal_fsp = output_dir / "minimal_uniform_import_layout_only.fsp"

    result: dict[str, Any] = {
        "status": "RUNNING_LUMERICAL_YEE_ZERO_JUMP_MEASUREMENT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "scope": "layout/index-only raw complex component-Yee epsilon measurement",
        "Maxwell_solves": 0,
        "fdtd_run_called": False,
        "requested_physical_gpu_index": int(args.gpu_index),
        "inputs": {
            "forward_project": _artifact(forward_project),
            "beta2_density": _artifact(density_file),
            "density_key": args.density_key,
        },
        "analytic_material_law": {
            "law": RELAXATION_CONTRACT.law,
            "formula": "[n_bg + rho*(n_Au-n_bg)]^2",
            "background_index": [RELAXATION_CONTRACT.background_n, RELAXATION_CONTRACT.background_k],
            "Au_index_4um": _complex_json(ordal_au_index()),
        },
    }
    _write_json(result_json, result)
    started = time.perf_counter()
    minimal_fdtd = None
    production_fdtd = None
    all_arrays: dict[str, np.ndarray] = {}
    try:
        _configure_lumapi(args.gpu_index)
        import lumapi

        minimal_fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        result["solver_version"] = str(minimal_fdtd.version())
        result["minimal_bulk_layout"] = _build_minimal_uniform_layout(minimal_fdtd)
        minimal_fdtd.save(str(minimal_fsp))

        def evaluate_minimal(rho: float) -> dict[str, np.ndarray]:
            _set_minimal_uniform_density(minimal_fdtd, rho)
            return read_lumerical_index_detail(
                minimal_fdtd, monitor_name=MINIMAL_INDEX_MONITOR
            )

        minimal_records, minimal_arrays, uniform_rows = _uniform_case(
            name="minimal_bulk_uniform",
            rhos=UNIFORM_RHOS,
            evaluate=evaluate_minimal,
            target_xyz=(0.0, 0.0, 0.0),
        )
        result["minimal_bulk_uniform"] = minimal_records
        all_arrays.update(minimal_arrays)
        minimal_fdtd.close()
        minimal_fdtd = None

        production_fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        production_fdtd.load(str(forward_project))
        production_fdtd.switchtolayout()

        def evaluate_production_uniform(rho: float) -> dict[str, np.ndarray]:
            state = np.full(CONTRACT.design_node_shape, rho, dtype=np.float64)
            set_lumerical_projected_density(production_fdtd, state)
            return read_lumerical_index_detail(production_fdtd, monitor_name=PABS_INDEX)

        stack_records, stack_arrays, stack_rows = _uniform_case(
            name="production_stack_uniform",
            rhos=UNIFORM_RHOS,
            evaluate=evaluate_production_uniform,
            target_xyz=(0.0, 0.0, 0.5 * CONTRACT.design_thickness_m),
        )
        result["production_stack_uniform"] = stack_records
        all_arrays.update(stack_arrays)
        uniform_rows.extend(stack_rows)

        beta2_density = load_projected_density_file(density_file, key=args.density_key)

        def evaluate_nonuniform(state: np.ndarray) -> dict[str, np.ndarray]:
            set_lumerical_projected_density(production_fdtd, state)
            return read_lumerical_index_detail(production_fdtd, monitor_name=PABS_INDEX)

        nonuniform_records, nonuniform_arrays, nonuniform_rows = _nonuniform_case(
            base_rho=beta2_density,
            rhos=NONUNIFORM_RHOS,
            evaluate_density=evaluate_nonuniform,
        )
        result["nonuniform_exact_zero_node"] = nonuniform_records
        all_arrays.update(nonuniform_arrays)
        # Exact roundtrip to the input density is mandatory.
        set_lumerical_projected_density(production_fdtd, beta2_density)
        result["production_roundtrip"] = validate_index_detail(
            read_lumerical_index_detail(production_fdtd, monitor_name=PABS_INDEX)
        )

        np.savez_compressed(raw_npz, **all_arrays)
        _write_uniform_csv(uniform_csv, uniform_rows)
        _write_nonuniform_csv(nonuniform_csv, nonuniform_rows)
        _plot_uniform(uniform_plot, all_arrays)
        _plot_nonuniform(nonuniform_plot, all_arrays)
        _plot_exact_zero_group(group_plot, all_arrays)
        result["conclusion"] = _jump_conclusion(result)
        result["status"] = "COMPLETED_LUMERICAL_YEE_ZERO_JUMP_MEASUREMENT"
        result["passed"] = True
        result["artifacts"] = {
            "minimal_layout_fsp": _artifact(minimal_fsp),
            "raw_npz": _artifact(raw_npz),
            "uniform_csv": _artifact(uniform_csv),
            "nonuniform_csv": _artifact(nonuniform_csv),
            "uniform_plot": _artifact(uniform_plot),
            "nonuniform_plot": _artifact(nonuniform_plot),
            "exact_zero_group_plot": _artifact(group_plot),
        }
        _write_report(report_md, result)
        result["artifacts"]["report_md"] = _artifact(report_md)
    except Exception as exc:
        result["status"] = "FAILED_LUMERICAL_YEE_ZERO_JUMP_MEASUREMENT"
        result["passed"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        for fdtd in (minimal_fdtd, production_fdtd):
            if fdtd is not None:
                try:
                    fdtd.close()
                except Exception:
                    pass
        result["wall_time_s"] = time.perf_counter() - started
        _write_json(result_json, result)
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
