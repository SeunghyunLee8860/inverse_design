#!/usr/bin/env python3
"""Compare two Device-A empty-stack meshes on one physical target plane.

The raw case directories are read-only inputs.  In particular, this script
does not rewrite a historical FAILED_ACCEPTANCE result.  It records a
separate corrected acceptance audit for an intentionally offset Gaussian:
the absolute lateral power relative to incident power is the relevant
quantity, whereas an opposite-face ratio becomes ill-conditioned when both
face powers are nearly zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:
    from photothermal_pte.validation.paper_ir_sanity.coordinate_plot import (
        center_field,
    )
except ModuleNotFoundError:  # Direct execution outside repository cwd.
    from coordinate_plot import center_field


RELATIVE_GATE = 5.0e-3
LATERAL_FLUX_GATE = 1.0e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-case", type=Path, required=True)
    parser.add_argument("--candidate-case", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--common-step-nm", type=float, default=100.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "server_path": str(path.resolve()),
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
        "committed_to_git": False,
    }


def relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(second), np.finfo(float).tiny)


def load_case(directory: Path) -> dict[str, Any]:
    result_path = directory / "case_result.json"
    incident_path = directory / "incident_reference.npz"
    if not result_path.is_file() or not incident_path.is_file():
        raise FileNotFoundError(f"incomplete empty-stack case: {directory}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with np.load(incident_path) as archive:
        x = np.asarray(archive["x_m"], float).reshape(-1)
        y = np.asarray(archive["y_m"], float).reshape(-1)
        intensity = np.asarray(archive["downward_intensity_W_m2"], float)
    if intensity.shape != (x.size, y.size):
        raise RuntimeError(f"incident field shape mismatch in {directory}")
    if np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0):
        raise RuntimeError(f"non-monotone incident coordinates in {directory}")
    if not np.all(np.isfinite(intensity)) or np.any(intensity < 0.0):
        raise RuntimeError(f"invalid downward intensity in {directory}")
    return {
        "directory": directory.resolve(),
        "result_path": result_path.resolve(),
        "incident_path": incident_path.resolve(),
        "result": result,
        "x": x,
        "y": y,
        "intensity": intensity,
    }


def corrected_offset_acceptance(case: dict[str, Any]) -> dict[str, Any]:
    result = case["result"]
    raw = dict(result["run_result"]["acceptance"])
    source = result["pre_run_contract"]["geometry"]["source"]
    center = np.asarray(source["beam_center_m"], float)
    is_offset = bool(np.linalg.norm(center) > 1.0e-12)
    faces = result["run_result"]["six_face"]["faces"]
    lateral = max(
        abs(float(faces[name]["normalized_signed_axis_flux"]))
        for name in ("x_min", "x_max", "y_min", "y_max")
    )
    corrected = dict(raw)
    obsolete = corrected.pop("opposite_lateral_flux_asymmetry_lt_1e_4", None)
    if is_offset:
        corrected["offset_source_max_absolute_lateral_flux_lt_1e_4"] = (
            lateral < LATERAL_FLUX_GATE
        )
    elif obsolete is not None:
        corrected["opposite_lateral_flux_asymmetry_lt_1e_4"] = bool(obsolete)
    return {
        "raw_status": result["status"],
        "raw_acceptance": raw,
        "source_center_m": center.tolist(),
        "offset_source": is_offset,
        "max_absolute_lateral_flux_over_incident": lateral,
        "corrected_acceptance": corrected,
        "corrected_all_pass": bool(all(corrected.values())),
        "raw_result_immutable": True,
    }


def common_axis(first: np.ndarray, second: np.ndarray, step_m: float) -> np.ndarray:
    low = max(float(first[0]), float(second[0]))
    high = min(float(first[-1]), float(second[-1]))
    count = int(np.floor((high - low) / step_m))
    if count < 2:
        raise RuntimeError("empty common incident-field support")
    margin = 0.5 * ((high - low) - count * step_m)
    return low + margin + step_m * np.arange(count + 1, dtype=float)


def interpolate(case: dict[str, Any], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (case["x"], case["y"]),
        case["intensity"],
        bounds_error=True,
    )
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.asarray(interpolator(np.column_stack((xx.ravel(), yy.ravel())))).reshape(
        xx.shape
    )


def moments(x: np.ndarray, y: np.ndarray, intensity: np.ndarray) -> dict[str, Any]:
    total = float(np.sum(intensity))
    if total <= 0.0:
        raise RuntimeError("nonpositive target-plane intensity integral")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    cx = float(np.sum(intensity * xx) / total)
    cy = float(np.sum(intensity * yy) / total)
    sx = float(np.sqrt(np.sum(intensity * (xx - cx) ** 2) / total))
    sy = float(np.sqrt(np.sum(intensity * (yy - cy) ** 2) / total))
    return {
        "centroid_m": [cx, cy],
        "sigma_m": [sx, sy],
        "intensity_1_over_e2_waist_m": [2.0 * sx, 2.0 * sy],
    }


def main() -> None:
    args = parse_args()
    if args.common_step_nm <= 0.0:
        raise ValueError("common step must be positive")
    reference = load_case(args.reference_case.resolve())
    candidate = load_case(args.candidate_case.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    step_m = args.common_step_nm * 1.0e-9
    x = common_axis(reference["x"], candidate["x"], step_m)
    y = common_axis(reference["y"], candidate["y"], step_m)
    fields = {
        "reference": interpolate(reference, x, y),
        "candidate": interpolate(candidate, x, y),
    }
    area = step_m**2
    powers = {name: float(np.sum(field) * area) for name, field in fields.items()}
    normalized = {name: field / powers[name] for name, field in fields.items()}
    delta = normalized["candidate"] - normalized["reference"]
    nrmse = float(
        np.linalg.norm(delta.ravel())
        / max(np.linalg.norm(normalized["reference"].ravel()), np.finfo(float).tiny)
    )
    correlation = float(
        np.corrcoef(
            normalized["reference"].ravel(), normalized["candidate"].ravel()
        )[0, 1]
    )
    field_moments = {name: moments(x, y, field) for name, field in fields.items()}
    centroid_shift = float(
        np.linalg.norm(
            np.asarray(field_moments["candidate"]["centroid_m"])
            - np.asarray(field_moments["reference"]["centroid_m"])
        )
    )
    waist_relative = [
        relative(candidate_value, reference_value)
        for candidate_value, reference_value in zip(
            field_moments["candidate"]["intensity_1_over_e2_waist_m"],
            field_moments["reference"]["intensity_1_over_e2_waist_m"],
        )
    ]

    def scalar(case: dict[str, Any], *keys: str) -> float:
        value: Any = case["result"]
        for key in keys:
            value = value[key]
        return float(value)

    scalar_metrics = {
        "incident_power_W_at_1_W_m2": {
            "reference": scalar(
                reference, "run_result", "normalization", "incident_power_W_at_1_W_m2"
            ),
            "candidate": scalar(
                candidate, "run_result", "normalization", "incident_power_W_at_1_W_m2"
            ),
        },
        "central_incident_intensity_native_W_m2": {
            "reference": scalar(
                reference,
                "run_result",
                "incident_reference",
                "central_incident_intensity_W_m2",
            ),
            "candidate": scalar(
                candidate,
                "run_result",
                "incident_reference",
                "central_incident_intensity_W_m2",
            ),
        },
        "common_grid_integrated_downward_power_native_W": {
            "reference": powers["reference"],
            "candidate": powers["candidate"],
        },
    }
    for metric in scalar_metrics.values():
        metric["relative_difference"] = relative(
            metric["candidate"], metric["reference"]
        )

    reference_acceptance = corrected_offset_acceptance(reference)
    candidate_acceptance = corrected_offset_acceptance(candidate)
    gates = {
        "reference_corrected_empty_stack_acceptance": reference_acceptance[
            "corrected_all_pass"
        ],
        "candidate_corrected_empty_stack_acceptance": candidate_acceptance[
            "corrected_all_pass"
        ],
        "incident_power_relative_difference_lt_0p5_percent": scalar_metrics[
            "incident_power_W_at_1_W_m2"
        ]["relative_difference"]
        < RELATIVE_GATE,
        "central_intensity_relative_difference_lt_0p5_percent": scalar_metrics[
            "central_incident_intensity_native_W_m2"
        ]["relative_difference"]
        < RELATIVE_GATE,
        "normalized_spatial_intensity_NRMSE_lt_0p5_percent": nrmse
        < RELATIVE_GATE,
        "both_second_moment_waists_relative_difference_lt_0p5_percent": bool(
            max(waist_relative) < RELATIVE_GATE
        ),
    }
    status = (
        "VALIDATED_DEVICE_A_EMPTY_STACK_FAST_MESH"
        if all(gates.values())
        else "FAILED_DEVICE_A_EMPTY_STACK_FAST_MESH_VALIDATION"
    )
    summary = {
        "status": status,
        "scope": (
            "empty Palik SiO2/Si layered stack only; this does not validate "
            "finite Device-A Q, thermal temperature, PTE current, or optimization"
        ),
        "reference": {
            "directory": str(reference["directory"]),
            "case_result_sha256": sha256(reference["result_path"]),
            "incident_reference_sha256": sha256(reference["incident_path"]),
            "acceptance": reference_acceptance,
        },
        "candidate": {
            "directory": str(candidate["directory"]),
            "case_result_sha256": sha256(candidate["result_path"]),
            "incident_reference_sha256": sha256(candidate["incident_path"]),
            "acceptance": candidate_acceptance,
        },
        "common_grid": {
            "step_m": step_m,
            "shape": [int(x.size), int(y.size)],
            "x_bounds_m": [float(x[0]), float(x[-1])],
            "y_bounds_m": [float(y[0]), float(y[-1])],
            "interpolation": "linear RegularGridInterpolator; no clipping or rescaling",
        },
        "scalar_metrics": scalar_metrics,
        "normalized_spatial_intensity": {
            "NRMSE": nrmse,
            "correlation": correlation,
        },
        "moments": field_moments,
        "centroid_shift_m": centroid_shift,
        "waist_relative_difference": waist_relative,
        "gates": gates,
    }
    (output / "device_a_empty_mesh_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    manifest_entries = []
    for label, case in (("reference", reference), ("candidate", candidate)):
        for filename, role in (
            ("case_result.json", "case_result"),
            ("incident_reference.npz", "target_plane_incident_field"),
            ("finite_2um_optical_q.fsp", "raw_Lumerical_project"),
            ("finite_2um_optical_q_p0.log", "raw_solver_log"),
        ):
            path = case["directory"] / filename
            if path.is_file():
                manifest_entries.append(
                    artifact_record(path, f"{label}_{role}")
                )
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(
            {
                "status": status,
                "artifacts": manifest_entries,
                "raw_NPZ_or_FSP_committed_to_git": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with (output / "device_a_empty_mesh_validation_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("metric", "reference", "candidate", "relative_difference"))
        for name, metric in scalar_metrics.items():
            writer.writerow(
                (name, metric["reference"], metric["candidate"], metric["relative_difference"])
            )
        writer.writerow(("normalized_spatial_intensity_NRMSE", "", nrmse, ""))
        writer.writerow(("normalized_spatial_intensity_correlation", "", correlation, ""))
        writer.writerow(("centroid_shift_m", "", centroid_shift, ""))
        writer.writerow(("waist_x_relative_difference", "", waist_relative[0], ""))
        writer.writerow(("waist_y_relative_difference", "", waist_relative[1], ""))

    vmax = max(float(np.max(value / np.max(value))) for value in fields.values())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, name in zip(axes[:2], ("reference", "candidate")):
        display = fields[name] / np.max(fields[name])
        image = center_field(
            axis,
            x,
            y,
            display,
            coordinate_scale=1e6,
            vmin=0.0,
            vmax=vmax,
            cmap="viridis",
        )
        axis.set_title(f"{name}: target-plane intensity / peak")
        axis.set_xlabel("x (um)")
        axis.set_ylabel("y (um)")
        fig.colorbar(image, ax=axis)
    difference = (
        normalized["candidate"] - normalized["reference"]
    ) / float(np.max(normalized["reference"]))
    limit = max(float(np.max(np.abs(difference))), np.finfo(float).eps)
    image = center_field(
        axes[2],
        x,
        y,
        difference,
        coordinate_scale=1e6,
        vmin=-limit,
        vmax=limit,
        cmap="coolwarm",
    )
    axes[2].set_title("normalized candidate - reference\n(reference peak units)")
    axes[2].set_xlabel("x (um)")
    axes[2].set_ylabel("y (um)")
    fig.colorbar(image, ax=axes[2])
    fig.savefig(output / "device_a_empty_mesh_validation_fields.png", dpi=180)
    plt.close(fig)

    report = f"""# Device-A empty-stack fast-mesh validation

Status: `{status}`

This is an empty Palik SiO2/Si propagation and source-normalization check.
It does **not** validate finite Device-A absorption, thermal temperature, PTE
current, or optimization.

The raw case JSON files were not changed.  For an intentionally offset beam,
the old opposite-face *ratio* is ill-conditioned when both lateral powers are
nearly zero.  The separate audit uses the maximum absolute lateral face flux
relative to incident power, with a `1e-4` gate.

## Comparison

- incident-power relative difference: `{scalar_metrics['incident_power_W_at_1_W_m2']['relative_difference']:.6%}`
- central-intensity relative difference: `{scalar_metrics['central_incident_intensity_native_W_m2']['relative_difference']:.6%}`
- normalized target-plane intensity NRMSE: `{nrmse:.6%}`
- normalized target-plane intensity correlation: `{correlation:.12f}`
- centroid displacement: `{centroid_shift * 1e9:.6f} nm`
- second-moment waist relative difference x/y: `{waist_relative[0]:.6%}` / `{waist_relative[1]:.6%}`

## Gates

""" + "\n".join(f"- {name}: `{passed}`" for name, passed in gates.items()) + "\n"
    (output / "DEVICE_A_EMPTY_STACK_FAST_MESH_VALIDATION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps({"status": status, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
