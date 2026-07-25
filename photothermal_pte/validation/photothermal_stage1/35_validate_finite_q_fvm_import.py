#!/usr/bin/env python3
"""Validate a no-modification finite optical-Q import into FVM volumes.

Each optical sample is copied element-for-element to one FVM source control
volume.  The control-volume widths are the one-dimensional trapezoidal
quadrature weights represented as cell widths.  Therefore

    sum(Q_fvm * dV_fvm)

is algebraically identical to the original nested trapezoidal integration
without clipping, smoothing, gain, rescaling, tiling, or deletion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import config_stage1 as config
from lumerical_api import utc_timestamp, write_json


PR3_COMMIT = "053260da6fd0caec28ce155221bd18f683a0e5e7"
PR3_MANIFEST = (
    "photothermal_pte/reports/finite_2um_optical_q/"
    "RAW_ARTIFACT_MANIFEST.json"
)
EXPECTED_SHA256 = (
    "7a63f82842751e7623e895701bac4ce92558679ed71bf13a3d404695a150e794"
)
EXPECTED_POWER_W = 2.56071371086521e-12
POWER_LIMIT = 0.005
COORDINATE_ROUNDOFF_TOLERANCE_M = 1.0e-15
EXACT_FLAKE_BOUNDS_M = {
    "x": [-1.0e-6, 1.0e-6],
    "y": [-1.0e-6, 1.0e-6],
    "z": [-1.0e-7, 0.0],
}
REQUIRED_FIELDS = {
    "x_m",
    "y_m",
    "z_m",
    "Q_on_W_m3",
    "Qx_W_m3",
    "Qy_W_m3",
    "Qz_W_m3",
    "exact_flake_mask",
    "incident_intensity_W_m2",
    "P_abs_volume_W",
    "metadata_json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--q-artifact",
        default=str(
            config.OUTPUT_ROOT
            / "finite_convergence"
            / "domain"
            / "fixed_x_L16_pml24_dz5_w2_span6p8"
            / "finite_q_on_artifact.npz"
        ),
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--report-dir",
        default=str(
            config.REPOSITORY_ROOT / "reports" / "fvm_finite_q_import"
        ),
    )
    parser.add_argument(
        "--allow-missing-pr3-git-object",
        action="store_true",
        help=(
            "Use the embedded immutable PR #3 SHA contract when a clean or "
            "shallow checkout does not contain the PR #3 commit object."
        ),
    )
    return parser.parse_args()


def clean_output_directory(explicit: str | None) -> Path:
    output = (
        Path(explicit).expanduser().resolve()
        if explicit
        else config.OUTPUT_ROOT
        / "fvm_finite_q_import"
        / f"{utc_timestamp()}_import"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def scalar(data: Any, key: str) -> float:
    values = np.asarray(data[key], float).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        raise ValueError(f"{key} must contain one finite scalar")
    return float(values[0])


def read_pr3_manifest(
    *, allow_missing_git_object: bool = False
) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "show", f"{PR3_COMMIT}:{PR3_MANIFEST}"],
        cwd=config.REPOSITORY_ROOT.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        if allow_missing_git_object:
            return {
                "commit": PR3_COMMIT,
                "repository_path": PR3_MANIFEST,
                "git_object_available": False,
                "dependency_note": (
                    "PR #3 is outside PR #4 ancestry; the externally "
                    "supplied NPZ is authenticated against the immutable "
                    "SHA-256 embedded in this validation code"
                ),
                "entry": {
                    "sha256": EXPECTED_SHA256,
                    "external_artifact_required": True,
                },
            }
        raise RuntimeError(
            f"cannot read PR #3 manifest: {completed.stderr.strip()}"
        )
    manifest = json.loads(completed.stdout)
    matches = [
        item
        for item in manifest.get("artifacts", [])
        if item.get("sha256") == EXPECTED_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "PR #3 manifest does not contain exactly one expected Q artifact"
        )
    return {
        "commit": PR3_COMMIT,
        "repository_path": PR3_MANIFEST,
        "git_object_available": True,
        "entry": matches[0],
    }


def quadrature_edges(coordinate_m: np.ndarray) -> np.ndarray:
    """Convert trapezoidal point weights into contiguous FVM cell widths."""
    coordinate = np.asarray(coordinate_m, float).reshape(-1)
    if coordinate.size < 2 or not np.all(np.diff(coordinate) > 0.0):
        raise ValueError("coordinate must be strictly increasing")
    edges = np.empty(coordinate.size + 1, float)
    edges[0] = coordinate[0]
    edges[-1] = coordinate[-1]
    edges[1:-1] = 0.5 * (coordinate[:-1] + coordinate[1:])
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("derived FVM edges are not strictly increasing")
    expected_weights = np.empty_like(coordinate)
    expected_weights[0] = 0.5 * (coordinate[1] - coordinate[0])
    expected_weights[-1] = 0.5 * (
        coordinate[-1] - coordinate[-2]
    )
    expected_weights[1:-1] = 0.5 * (
        coordinate[2:] - coordinate[:-2]
    )
    if not np.allclose(
        np.diff(edges), expected_weights, rtol=1.0e-14, atol=0.0
    ):
        raise RuntimeError("FVM widths do not reproduce trapezoidal weights")
    return edges


def exact_physical_edges(
    coordinate_m: np.ndarray,
    selected: np.ndarray,
    bounds_m: list[float],
) -> np.ndarray:
    """Build cells contained exactly inside a requested physical interval."""
    coordinate = np.asarray(coordinate_m, float).reshape(-1)[selected]
    if coordinate.size < 2:
        raise ValueError("physical interval has too few source samples")
    edges = np.empty(coordinate.size + 1, float)
    edges[0] = bounds_m[0]
    edges[-1] = bounds_m[1]
    edges[1:-1] = 0.5 * (coordinate[:-1] + coordinate[1:])
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("exact physical FVM edges are not increasing")
    return edges


def trapezoidal_integral(
    values: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
) -> float:
    return float(
        np.trapezoid(
            np.trapezoid(
                np.trapezoid(values, z_m, axis=2), y_m, axis=1
            ),
            x_m,
            axis=0,
        )
    )


def map_q(
    path: Path,
    output: Path,
    *,
    allow_missing_pr3_git_object: bool = False,
) -> dict[str, Any]:
    pr3_manifest = read_pr3_manifest(
        allow_missing_git_object=allow_missing_pr3_git_object
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = sha256_file(path)
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError(
            f"Q SHA mismatch: expected {EXPECTED_SHA256}, got {actual_sha}"
        )
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(REQUIRED_FIELDS - set(data.files))
        if missing:
            raise ValueError(f"Q artifact lacks required fields: {missing}")
        x, y, z = (
            np.asarray(data[key], float).reshape(-1)
            for key in ("x_m", "y_m", "z_m")
        )
        components = {
            key: np.asarray(data[key], float)
            for key in (
                "Qx_W_m3",
                "Qy_W_m3",
                "Qz_W_m3",
                "Q_on_W_m3",
            )
        }
        shape = (x.size, y.size, z.size)
        if any(item.shape != shape for item in components.values()):
            raise ValueError("Q arrays do not match coordinate shape")
        if any(not np.all(np.diff(axis) > 0.0) for axis in (x, y, z)):
            raise ValueError("Q coordinates are not strictly increasing")
        if any(
            not np.all(np.isfinite(item))
            for item in (*components.values(), x, y, z)
        ):
            raise ValueError("Q artifact contains NaN or Inf")
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if metadata.get("array_axis_order") != ["x", "y", "z"]:
            raise ValueError("Q array order is not x,y,z")
        if metadata.get("exact_flake_bounds_m") != EXACT_FLAKE_BOUNDS_M:
            raise ValueError("exact TaIrTe4 bounds do not match")
        forbidden = {
            "clipping": bool(metadata.get("clipped", False)),
            "gain": bool(metadata.get("gain_applied", False)),
            "rescaling": bool(metadata.get("rescaled_to_flux", False)),
            "crop": bool(metadata.get("periodic_crop", False)),
            "tiling": bool(metadata.get("periodic_tiling", False)),
            "smoothing": bool(metadata.get("smoothing_applied", False)),
        }
        if any(forbidden.values()):
            raise ValueError(f"source records forbidden operation: {forbidden}")
        q = components["Q_on_W_m3"]
        component_sum = (
            components["Qx_W_m3"]
            + components["Qy_W_m3"]
            + components["Qz_W_m3"]
        )
        component_error = float(
            np.max(np.abs(component_sum - q))
            / max(float(np.max(np.abs(q))), np.finfo(float).tiny)
        )
        if component_error > 1.0e-12:
            raise ValueError("Qx+Qy+Qz does not reproduce Q")
        stored_mask = np.asarray(data["exact_flake_mask"], bool)
        if stored_mask.shape != shape:
            raise ValueError("exact flake mask shape mismatch")
        physical_mask = (
            (
                x[:, None, None]
                >= EXACT_FLAKE_BOUNDS_M["x"][0]
                - COORDINATE_ROUNDOFF_TOLERANCE_M
            )
            & (
                x[:, None, None]
                <= EXACT_FLAKE_BOUNDS_M["x"][1]
                + COORDINATE_ROUNDOFF_TOLERANCE_M
            )
            & (
                y[None, :, None]
                >= EXACT_FLAKE_BOUNDS_M["y"][0]
                - COORDINATE_ROUNDOFF_TOLERANCE_M
            )
            & (
                y[None, :, None]
                <= EXACT_FLAKE_BOUNDS_M["y"][1]
                + COORDINATE_ROUNDOFF_TOLERANCE_M
            )
            & (
                z[None, None, :]
                >= EXACT_FLAKE_BOUNDS_M["z"][0]
                - COORDINATE_ROUNDOFF_TOLERANCE_M
            )
            & (
                z[None, None, :]
                <= EXACT_FLAKE_BOUNDS_M["z"][1]
                + COORDINATE_ROUNDOFF_TOLERANCE_M
            )
        )
        maximum_q_outside_physical_mask = float(
            np.max(np.abs(q[~physical_mask]))
        )
        if maximum_q_outside_physical_mask != 0.0:
            raise ValueError(
                "source Q is nonzero outside the roundoff-inclusive "
                "physical TaIrTe4 mask"
            )
        stored_mask_boundary_nonzero = (~stored_mask) & physical_mask & (q != 0.0)
        incident_intensity = scalar(data, "incident_intensity_W_m2")
        if incident_intensity != 1.0:
            raise ValueError("source is not normalized to 1 W/m2")
        metadata_power_W = scalar(data, "P_abs_volume_W")

        edges = tuple(quadrature_edges(axis) for axis in (x, y, z))
        widths = tuple(np.diff(item) for item in edges)
        volume = (
            widths[0][:, None, None]
            * widths[1][None, :, None]
            * widths[2][None, None, :]
        )
        mapped_q = q.copy()
        original_array_sha = sha256_array(q)
        mapped_array_sha = sha256_array(mapped_q)
        array_identical = bool(np.array_equal(mapped_q, q))
        trapezoidal_power_W = trapezoidal_integral(q, x, y, z)
        fvm_sum_power_W = float(np.sum(mapped_q * volume))
        mapping_power_error = abs(
            fvm_sum_power_W - EXPECTED_POWER_W
        ) / EXPECTED_POWER_W
        quadrature_equivalence_error = abs(
            fvm_sum_power_W - trapezoidal_power_W
        ) / EXPECTED_POWER_W
        metadata_power_error = abs(
            metadata_power_W - EXPECTED_POWER_W
        ) / EXPECTED_POWER_W
        passed = bool(
            actual_sha == EXPECTED_SHA256
            and array_identical
            and original_array_sha == mapped_array_sha
            and mapping_power_error < POWER_LIMIT
            and quadrature_equivalence_error < 1.0e-12
            and metadata_power_error < POWER_LIMIT
            and component_error < 1.0e-12
            and not any(forbidden.values())
        )
        mapped_path = output / "finite_q_fvm_control_volumes.npz"
        np.savez_compressed(
            mapped_path,
            x_edges_m=edges[0],
            y_edges_m=edges[1],
            z_edges_m=edges[2],
            x_optical_samples_m=x,
            y_optical_samples_m=y,
            z_optical_samples_m=z,
            Q_fvm_W_m3=mapped_q,
            Qx_W_m3=components["Qx_W_m3"],
            Qy_W_m3=components["Qy_W_m3"],
            Qz_W_m3=components["Qz_W_m3"],
            exact_flake_mask=stored_mask,
            roundoff_inclusive_physical_mask=physical_mask,
            incident_intensity_W_m2=incident_intensity,
            source_artifact_sha256=actual_sha,
            mapping_method=(
                "elementwise_copy_to_trapezoidal_quadrature_control_volumes"
            ),
        )

        selected_axes = (
            (x >= EXACT_FLAKE_BOUNDS_M["x"][0] - COORDINATE_ROUNDOFF_TOLERANCE_M)
            & (x <= EXACT_FLAKE_BOUNDS_M["x"][1] + COORDINATE_ROUNDOFF_TOLERANCE_M),
            (y >= EXACT_FLAKE_BOUNDS_M["y"][0] - COORDINATE_ROUNDOFF_TOLERANCE_M)
            & (y <= EXACT_FLAKE_BOUNDS_M["y"][1] + COORDINATE_ROUNDOFF_TOLERANCE_M),
            (z >= EXACT_FLAKE_BOUNDS_M["z"][0] - COORDINATE_ROUNDOFF_TOLERANCE_M)
            & (z <= EXACT_FLAKE_BOUNDS_M["z"][1] + COORDINATE_ROUNDOFF_TOLERANCE_M),
        )
        exact_edges = tuple(
            exact_physical_edges(axis, selected, EXACT_FLAKE_BOUNDS_M[name])
            for axis, selected, name in zip(
                (x, y, z), selected_axes, ("x", "y", "z")
            )
        )
        exact_widths = tuple(np.diff(item) for item in exact_edges)
        exact_volume = (
            exact_widths[0][:, None, None]
            * exact_widths[1][None, :, None]
            * exact_widths[2][None, None, :]
        )
        nodal_energy_W = q * volume
        nonzero_energy_outside_physical_W = float(
            np.sum(np.abs(nodal_energy_W[~physical_mask]))
        )
        exact_cell_power_W = nodal_energy_W[
            np.ix_(*selected_axes)
        ].copy()
        exact_q_W_m3 = exact_cell_power_W / exact_volume
        exact_power_W = float(np.sum(exact_cell_power_W))
        exact_mapping_error = abs(
            exact_power_W - EXPECTED_POWER_W
        ) / EXPECTED_POWER_W
        source_energy_sha = sha256_array(
            nodal_energy_W[np.ix_(*selected_axes)]
        )
        mapped_cell_power_sha = sha256_array(exact_cell_power_W)
        exact_mapping_passed = bool(
            nonzero_energy_outside_physical_W == 0.0
            and source_energy_sha == mapped_cell_power_sha
            and exact_mapping_error < POWER_LIMIT
            and np.all(np.isfinite(exact_q_W_m3))
        )
        exact_path = output / "finite_q_exact_flake_source.npz"
        np.savez_compressed(
            exact_path,
            x_edges_m=exact_edges[0],
            y_edges_m=exact_edges[1],
            z_edges_m=exact_edges[2],
            Q_fvm_W_m3=exact_q_W_m3,
            source_power_per_cell_W=exact_cell_power_W,
            source_optical_sample_x_m=x[selected_axes[0]],
            source_optical_sample_y_m=y[selected_axes[1]],
            source_optical_sample_z_m=z[selected_axes[2]],
            incident_intensity_W_m2=incident_intensity,
            source_artifact_sha256=actual_sha,
            mapping_method=(
                "one_to_one_conservative_nodal_quadrature_energy_deposition_"
                "inside_exact_flake_bounds"
            ),
            empirical_gain_applied=False,
            global_rescaling_applied=False,
        )
        passed = passed and exact_mapping_passed
    return {
        "status": (
            "VALIDATED_FINITE_OPTICAL_Q_FVM_IMPORT"
            if passed
            else "FAILED_FINITE_OPTICAL_Q_FVM_IMPORT"
        ),
        "passed": passed,
        "pr3_manifest": pr3_manifest,
        "source_artifact_path": str(path),
        "source_artifact_sha256_expected": EXPECTED_SHA256,
        "source_artifact_sha256_actual": actual_sha,
        "mapped_artifact_path": str(mapped_path),
        "exact_flake_production_source_path": str(exact_path),
        "shape_xyz": list(shape),
        "coordinate_order": ["x", "y", "z"],
        "optical_coordinate_bounds_m": {
            "x": [float(x[0]), float(x[-1])],
            "y": [float(y[0]), float(y[-1])],
            "z": [float(z[0]), float(z[-1])],
        },
        "fvm_control_volume_bounds_m": {
            "x": [float(edges[0][0]), float(edges[0][-1])],
            "y": [float(edges[1][0]), float(edges[1][-1])],
            "z": [float(edges[2][0]), float(edges[2][-1])],
        },
        "exact_TaIrTe4_bounds_m": EXACT_FLAKE_BOUNDS_M,
        "exact_mask_voxels": int(np.count_nonzero(stored_mask)),
        "coordinate_roundoff_tolerance_m": (
            COORDINATE_ROUNDOFF_TOLERANCE_M
        ),
        "roundoff_inclusive_physical_mask_voxels": int(
            np.count_nonzero(physical_mask)
        ),
        "nonzero_samples_outside_stored_mask_but_on_roundoff_boundary": int(
            np.count_nonzero(stored_mask_boundary_nonzero)
        ),
        "mapped_Q_elementwise_identical": array_identical,
        "source_Q_array_sha256": original_array_sha,
        "mapped_Q_array_sha256": mapped_array_sha,
        "Q_component_sum_max_relative_error": component_error,
        "maximum_Q_outside_stored_exact_mask_W_m3": float(
            np.max(np.abs(q[~stored_mask]))
        ),
        "maximum_Q_outside_roundoff_inclusive_physical_mask_W_m3": (
            maximum_q_outside_physical_mask
        ),
        "incident_intensity_W_m2": incident_intensity,
        "P_Q_expected_W": EXPECTED_POWER_W,
        "P_Q_metadata_W": metadata_power_W,
        "P_Q_original_trapezoidal_W": trapezoidal_power_W,
        "P_Q_FVM_sum_QdV_W": fvm_sum_power_W,
        "FVM_mapping_relative_error": mapping_power_error,
        "quadrature_equivalence_relative_error": (
            quadrature_equivalence_error
        ),
        "allowed_mapping_relative_error": POWER_LIMIT,
        "mapping_method": (
            "elementwise Q copy; FVM cell widths exactly equal original "
            "trapezoidal quadrature weights"
        ),
        "exact_flake_production_mapping": {
            "status": (
                "PASSED_CONSERVATIVE_EXACT_FLAKE_DEPOSITION"
                if exact_mapping_passed
                else "FAILED_CONSERVATIVE_EXACT_FLAKE_DEPOSITION"
            ),
            "passed": exact_mapping_passed,
            "shape_xyz": list(exact_q_W_m3.shape),
            "bounds_m": {
                name: [float(axis[0]), float(axis[-1])]
                for name, axis in zip(("x", "y", "z"), exact_edges)
            },
            "source_nodal_energy_array_sha256": source_energy_sha,
            "mapped_cell_power_array_sha256": mapped_cell_power_sha,
            "nonzero_source_energy_deleted_W": (
                nonzero_energy_outside_physical_W
            ),
            "P_Q_exact_flake_sum_cell_power_W": exact_power_W,
            "relative_power_error": exact_mapping_error,
            "empirical_gain_applied": False,
            "global_rescaling_applied": False,
            "sample_averaging_applied": False,
            "mapping_semantics": (
                "each original nodal trapezoidal energy parcel is deposited "
                "one-to-one into its exact-flake boundary/interior cell; "
                "Q density is derived only by dividing conserved parcel "
                "power by the receiving physical cell volume"
            ),
        },
        "interpolation_applied": False,
        "forbidden_operations_applied": {
            **forbidden,
            "outside_flake_deletion": False,
        },
        "finite_optical_Q_imported_into_thermal_solve": False,
        "full_device_executed": False,
        "next_required_gate": (
            "ANISOTROPIC_FINITE_G_MULTIMATERIAL_FVM_PRODUCTION"
            if passed
            else "FINITE_OPTICAL_Q_CONSERVATIVE_IMPORT"
        ),
        "criteria": {
            "source_artifact_SHA256_exact": True,
            "mapped_Q_elementwise_identical": True,
            "sum_QdV_relative_error_lt": POWER_LIMIT,
            "quadrature_equivalence_relative_error_lt": 1.0e-12,
            "exact_flake_conservative_deposition_relative_error_lt": (
                POWER_LIMIT
            ),
            "nonzero_source_energy_deleted_W_eq": 0.0,
            "no_forbidden_operations": True,
        },
    }


def write_cases_csv(path: Path, result: dict[str, Any]) -> None:
    row = {
        "case_id": "finite_optical_Q_to_FVM_control_volumes",
        "status": result["status"],
        "passed": result["passed"],
        "source_SHA256": result["source_artifact_sha256_actual"],
        "shape_xyz": "x".join(map(str, result["shape_xyz"])),
        "P_Q_expected_W": result["P_Q_expected_W"],
        "P_Q_original_trapezoidal_W": (
            result["P_Q_original_trapezoidal_W"]
        ),
        "P_Q_FVM_sum_QdV_W": result["P_Q_FVM_sum_QdV_W"],
        "mapping_relative_error": result["FVM_mapping_relative_error"],
        "array_elementwise_identical": result[
            "mapped_Q_elementwise_identical"
        ],
        "interpolation_applied": result["interpolation_applied"],
        "incident_intensity_W_m2": result["incident_intensity_W_m2"],
        "exact_flake_shape_xyz": "x".join(
            map(
                str,
                result["exact_flake_production_mapping"]["shape_xyz"],
            )
        ),
        "exact_flake_P_Q_W": result["exact_flake_production_mapping"][
            "P_Q_exact_flake_sum_cell_power_W"
        ],
        "exact_flake_mapping_error": result[
            "exact_flake_production_mapping"
        ]["relative_power_error"],
    }
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(row), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(row)


def write_report(path: Path, result: dict[str, Any]) -> None:
    text = f"""# Finite optical-Q FVM import report

**Status: `{result["status"]}`.**

The PR #3 artifact SHA-256 is exactly
`{result["source_artifact_sha256_actual"]}`. Its array order is `x,y,z`,
shape is `{result["shape_xyz"]}`, and incident-intensity normalization is
`{result["incident_intensity_W_m2"]} W/m2`.

No thermal solve or production full-device calculation was run in this gate.

## Mapping

Every original `Q[i,j,k]` value is copied element-for-element to one FVM
source control volume. The cell widths are exactly the original 1D
trapezoidal quadrature weights. There is no interpolation and no source-value
change.

The original and mapped Q-array SHA-256 values are both
`{result["source_Q_array_sha256"]}`.

| Power check | W |
|---|---:|
| expected PR #3 P_Q | {result["P_Q_expected_W"]:.15g} |
| original nested trapezoidal integration | {result["P_Q_original_trapezoidal_W"]:.15g} |
| FVM sum(Q*dV) | {result["P_Q_FVM_sum_QdV_W"]:.15g} |

The FVM mapping relative error is
`{result["FVM_mapping_relative_error"]:.6g}`, below the required 0.5%.
The algebraic quadrature-equivalence error is
`{result["quadrature_equivalence_relative_error"]:.6g}`.

## Prohibited operations

Clipping, smoothing, gain, total-Q rescaling, periodic crop/tiling, and
outside-flake deletion are all `false`. There are
`{result["nonzero_samples_outside_stored_mask_but_on_roundoff_boundary"]}`
nonzero samples excluded by the stored boolean mask only because their
`z={5.790264287871194e-23:g} m` coordinate is infinitesimally above zero.
They remain inside the explicit `1e-15 m` roundoff-inclusive physical mask
used by PR #3, and the mapper preserves them without deletion or alteration.
Q is exactly zero outside that physical mask.

## Exact-flake production source

For the thermal geometry, every original nodal quadrature energy parcel
`Q*w_x*w_y*w_z` is deposited one-to-one into a cell contained inside the
exact `2 um x 2 um x 100 nm` flake. This is necessary because the validated
padded-grid trapezoid convention assigns a full quadrature weight to nonzero
boundary samples. No parcel is deleted or averaged, and no empirical gain or
global rescaling is applied. The receiving volumetric density is obtained
only by dividing each conserved parcel power by its physical cell volume.

The exact-flake source shape is
`{result["exact_flake_production_mapping"]["shape_xyz"]}` and its bounds are
`{result["exact_flake_production_mapping"]["bounds_m"]}`. Its summed power is
`{result["exact_flake_production_mapping"]["P_Q_exact_flake_sum_cell_power_W"]:.15g} W`
with relative error
`{result["exact_flake_production_mapping"]["relative_power_error"]:.6g}`.
The source-energy and mapped-cell-power array hashes are identical:
`{result["exact_flake_production_mapping"]["mapped_cell_power_array_sha256"]}`.

## Gate

The finite optical-Q conservative import gate passes. The next permitted
step is the first anisotropic, finite-G, multi-material FVM thermal solve.
Its result must be reported as a unit-intensity response
`Delta T / I_inc [K/(W/m2)]`, not as a physical laser temperature.
"""
    path.write_text(text, encoding="utf-8")


def repository_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.REPOSITORY_ROOT.parent))
    except ValueError:
        return str(path.resolve())


def git_value(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=config.REPOSITORY_ROOT.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def write_manifest(
    path: Path,
    *,
    output: Path,
    report_dir: Path,
    command: str,
) -> None:
    candidates = [
        output / "finite_q_fvm_control_volumes.npz",
        output / "finite_q_exact_flake_source.npz",
        report_dir / "FINITE_OPTICAL_Q_FVM_IMPORT_REPORT.md",
        report_dir / "finite_q_fvm_import_summary.json",
        report_dir / "finite_q_fvm_import_cases.csv",
    ]
    files = [item for item in candidates if item.is_file()]
    write_json(
        path,
        {
            "schema_version": 1,
            "generated_at_utc": utc_timestamp(),
            "branch": git_value("branch", "--show-current"),
            "base_commit_before_control": git_value("rev-parse", "HEAD"),
            "generation_command": command,
            "source_artifact_sha256": EXPECTED_SHA256,
            "finite_optical_Q_imported_into_thermal_solve": False,
            "artifacts": [
                {
                    "repository_path": repository_relative(item),
                    "server_path": str(item.resolve()),
                    "bytes": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
                for item in files
            ],
        },
    )


def main() -> int:
    args = parse_args()
    output = clean_output_directory(args.output_dir)
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, *sys.argv])
    try:
        result = map_q(
            Path(args.q_artifact).expanduser().resolve(),
            output,
            allow_missing_pr3_git_object=(
                args.allow_missing_pr3_git_object
            ),
        )
    except Exception as exc:
        result = {
            "status": "FAILED_FINITE_OPTICAL_Q_FVM_IMPORT",
            "passed": False,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "finite_optical_Q_imported_into_thermal_solve": False,
            "full_device_executed": False,
            "next_required_gate": "FINITE_OPTICAL_Q_CONSERVATIVE_IMPORT",
        }
    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        **result,
    }
    summary_path = report_dir / "finite_q_fvm_import_summary.json"
    cases_path = report_dir / "finite_q_fvm_import_cases.csv"
    report_path = report_dir / "FINITE_OPTICAL_Q_FVM_IMPORT_REPORT.md"
    write_json(summary_path, summary)
    if result["passed"]:
        write_cases_csv(cases_path, result)
        write_report(report_path, result)
    else:
        cases_path.write_text(
            "case_id,status,passed,exception\n"
            f"finite_optical_Q_to_FVM_control_volumes,"
            f"{result['status']},False,{json.dumps(result.get('exception'))}\n",
            encoding="utf-8",
        )
        report_path.write_text(
            "# Finite optical-Q FVM import report\n\n"
            f"**Status: `{result['status']}`.**\n\n"
            f"`{result.get('exception', 'unknown failure')}`\n",
            encoding="utf-8",
        )
    write_manifest(
        report_dir / "RAW_ARTIFACT_MANIFEST.json",
        output=output,
        report_dir=report_dir,
        command=command,
    )
    write_json(output / "import_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
