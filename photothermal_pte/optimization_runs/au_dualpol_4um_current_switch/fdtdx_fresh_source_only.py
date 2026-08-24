"""Fresh all-air source-only solve on the validated FDTDX anchor contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_dependency import (
    configured_source,
    require_source,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
    mesh_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_mesh import (
    build_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_runtime_preflight import (
    load_runtime_lock,
)


ANCHOR_SPEC = MeshSpec()
TOTAL_PERIODS = 16
WINDOW_PERIODS = 4
COURANT_FACTOR = 0.5
STATIONARITY_LIMIT = 5.0e-3
POLARIZATION_PURITY_MINIMUM = 0.999
CLOSED_FLUX_FRACTION_LIMIT = 2.0e-2
BEAM_CENTER_LIMIT_M = 0.10e-6
BEAM_WAIST_RELATIVE_LIMIT = 0.10
JSON_NAME = "FDTDX_FRESH_SOURCE_ONLY.json"
RAW_NAME = "FDTDX_FRESH_SOURCE_ONLY_FIELDS.npz"


def weighted_complex_nrmse(
    late: np.ndarray, previous: np.ndarray, weights: np.ndarray
) -> float:
    late_value = np.asarray(late)
    previous_value = np.asarray(previous)
    weight = np.asarray(weights, dtype=np.float64)
    if late_value.shape != previous_value.shape:
        raise ValueError("late and previous fields must have identical shape")
    if late_value.ndim != 4 or weight.shape != late_value.shape[1:]:
        raise ValueError("fields must be (component,x,y,z) with matching 3D weights")
    numerator = float(np.sum(np.abs(late_value - previous_value) ** 2 * weight[None]))
    denominator = float(np.sum(np.abs(late_value) ** 2 * weight[None]))
    return math.sqrt(numerator / max(denominator, np.finfo(float).tiny))


def polarization_audit(
    field: np.ndarray, polarization: str, area_weights: np.ndarray
) -> dict[str, Any]:
    value = np.asarray(field)
    weights = np.asarray(area_weights, dtype=np.float64)
    if value.ndim != 4 or value.shape[0] != 3 or weights.shape != value.shape[1:]:
        raise ValueError("field must be (3,x,y,z) with matching area weights")
    desired = 1 if polarization == "Ea" else 0 if polarization == "Eb" else None
    if desired is None:
        raise ValueError(f"unknown polarization {polarization!r}")
    energy = np.asarray(
        [np.sum(np.abs(value[component]) ** 2 * weights) for component in range(3)],
        dtype=np.float64,
    )
    total = float(np.sum(energy))
    purity = float(energy[desired] / max(total, np.finfo(float).tiny))
    return {
        "desired_component": ("Ey" if desired == 1 else "Ex"),
        "component_energy_fraction": {
            name: float(item / max(total, np.finfo(float).tiny))
            for name, item in zip(("Ex", "Ey", "Ez"), energy, strict=True)
        },
        "purity": purity,
    }


def beam_moments(
    field: np.ndarray,
    x_centers_m: np.ndarray,
    y_centers_m: np.ndarray,
    area_weights_xy: np.ndarray,
) -> dict[str, float]:
    value = np.asarray(field)
    if value.ndim != 4 or value.shape[-1] != 1:
        raise ValueError("target field must be a one-cell z plane")
    intensity = np.sum(np.abs(value[:, :, :, 0]) ** 2, axis=0)
    weights = intensity * np.asarray(area_weights_xy, dtype=np.float64)
    total = float(np.sum(weights))
    if not total > 0.0:
        raise RuntimeError("target plane has non-positive field intensity")
    x = np.asarray(x_centers_m, dtype=np.float64)
    y = np.asarray(y_centers_m, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    center_x = float(np.sum(weights * xx) / total)
    center_y = float(np.sum(weights * yy) / total)
    variance_x = float(np.sum(weights * (xx - center_x) ** 2) / total)
    variance_y = float(np.sum(weights * (yy - center_y) ** 2) / total)
    return {
        "center_x_m": center_x,
        "center_y_m": center_y,
        "second_moment_waist_x_m": 2.0 * math.sqrt(max(variance_x, 0.0)),
        "second_moment_waist_y_m": 2.0 * math.sqrt(max(variance_y, 0.0)),
    }


def _region_volume_weights(model: dict[str, Any], object_name: str) -> np.ndarray:
    grid = model["grid"]
    grid_slice = model["slices"][object_name]
    selected = [
        np.asarray(grid.cell_widths(axis), dtype=np.float64)[grid_slice[axis]]
        for axis in range(3)
    ]
    return (
        selected[0][:, None, None]
        * selected[1][None, :, None]
        * selected[2][None, None, :]
    )


def _target_coordinates(model: dict[str, Any]):
    grid = model["grid"]
    target_slice = model["slices"]["target_field"]
    widths = [
        np.asarray(grid.cell_widths(axis), dtype=np.float64)[target_slice[axis]]
        for axis in range(3)
    ]
    centers = [
        np.asarray(grid.centers(axis), dtype=np.float64)[target_slice[axis]]
        for axis in range(3)
    ]
    volume = (
        widths[0][:, None, None]
        * widths[1][None, :, None]
        * widths[2][None, None, :]
    )
    area_xy = widths[0][:, None] * widths[1][None, :]
    return centers, volume, area_xy


def all_air_arrays(model: dict[str, Any]):
    arrays = (
        model["base"]
        .reset()
        .aset("dispersive_c1", model["fixed_c1"])
        .aset("dispersive_c2", model["fixed_c2"])
        .aset("dispersive_c3", model["fixed_c3"])
    )
    jnp = model["jnp"]
    coefficient_maxima = {
        name: float(jnp.max(jnp.abs(getattr(arrays, name))))
        for name in ("dispersive_c1", "dispersive_c2", "dispersive_c3")
    }
    inverse_error = float(jnp.max(jnp.abs(arrays.inv_permittivities - 1.0)))
    checks = {
        "inverse_permittivity_exactly_one": inverse_error == 0.0,
        **{
            f"{name}_exactly_zero": maximum == 0.0
            for name, maximum in coefficient_maxima.items()
        },
    }
    return arrays, {
        "checks": checks,
        "ready": all(checks.values()),
        "inverse_permittivity_max_absolute_error": inverse_error,
        "ADE_coefficient_max_absolute_values": coefficient_maxima,
    }


def extract_detector_fields(states: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract the pinned detector schema, including six hollow shell faces."""

    fields = {
        "au_previous": np.asarray(states["au_previous"]["phasor"][0, 0]),
        "au_late": np.asarray(states["au_late"]["phasor"][0, 0]),
        "tairte4_previous": np.asarray(
            states["tairte4_previous"]["phasor"][0, 0]
        ),
        "tairte4_late": np.asarray(states["tairte4_late"]["phasor"][0, 0]),
        "target": np.asarray(states["target_field"]["phasor"][0, 0]),
        "incident_phasor": np.asarray(states["incident_plane"]["phasor"]),
        "closed_td": np.asarray(states["material_flux_td"]["poynting_flux"]),
    }
    closed_state = states["material_flux"]
    expected_closed_keys = {
        f"phasor_axis{axis}_{side}"
        for axis in range(3)
        for side in ("min", "max")
    }
    if set(closed_state) != expected_closed_keys:
        raise RuntimeError(
            f"unexpected closed-surface phasor state keys: {sorted(closed_state)}"
        )
    fields.update(
        {
            f"closed_{name}": np.asarray(closed_state[name])
            for name in sorted(expected_closed_keys)
        }
    )
    return fields


def evaluate_output(
    model: dict[str, Any], output: Any, polarization: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    states = output.detector_states
    fields = extract_detector_fields(states)
    finite = all(np.all(np.isfinite(value)) for value in fields.values())
    stationarity = {
        "au_complex_E_NRMSE": weighted_complex_nrmse(
            fields["au_late"],
            fields["au_previous"],
            _region_volume_weights(model, "au_design"),
        ),
        "tairte4_complex_E_NRMSE": weighted_complex_nrmse(
            fields["tairte4_late"],
            fields["tairte4_previous"],
            _region_volume_weights(model, "fixed_tairte4"),
        ),
    }
    stationarity["maximum_complex_E_NRMSE"] = max(stationarity.values())

    centers, target_volume, target_area = _target_coordinates(model)
    polarization_result = polarization_audit(
        fields["target"], polarization, target_volume
    )
    beam = beam_moments(
        fields["target"], centers[0], centers[1], target_area
    )
    fdtdx = model["fdtdx"]
    eta0 = float(fdtdx.constants.eta0)
    incident_power = float(
        eta0
        * np.asarray(
            model["placed"]["incident_plane"].compute_poynting_flux(
                states["incident_plane"]
            )
        )[0]
    )
    closed_phasor = float(
        eta0
        * np.asarray(
            model["placed"]["material_flux"].compute_net_flux(
                states["material_flux"]
            )
        )[0]
    )
    closed_td = float(eta0 * np.mean(fields["closed_td"][:, 0]))
    denominator = max(abs(incident_power), np.finfo(float).tiny)
    flux = {
        "incident_plane_signed_W": incident_power,
        "closed_box_inward_phasor_signed_W": closed_phasor,
        "closed_box_inward_td_mean_signed_W": closed_td,
        "closed_phasor_over_incident_absolute": abs(closed_phasor) / denominator,
        "closed_td_over_incident_absolute": abs(closed_td) / denominator,
        "normalization": "eta0 times FDTDX normalized-field Poynting flux",
    }
    waist_errors = [
        abs(beam[name] - CONTRACT.gaussian_waist_m) / CONTRACT.gaussian_waist_m
        for name in ("second_moment_waist_x_m", "second_moment_waist_y_m")
    ]
    gates = {
        "all_raw_detector_values_finite": bool(finite),
        "incident_power_positive": math.isfinite(incident_power) and incident_power > 0.0,
        "complex_field_stationarity": (
            stationarity["maximum_complex_E_NRMSE"] <= STATIONARITY_LIMIT
        ),
        "polarization_purity": (
            polarization_result["purity"] >= POLARIZATION_PURITY_MINIMUM
        ),
        "closed_phasor_flux_residual": (
            flux["closed_phasor_over_incident_absolute"]
            <= CLOSED_FLUX_FRACTION_LIMIT
        ),
        "closed_td_flux_residual": (
            flux["closed_td_over_incident_absolute"] <= CLOSED_FLUX_FRACTION_LIMIT
        ),
        "beam_center": (
            abs(beam["center_x_m"]) <= BEAM_CENTER_LIMIT_M
            and abs(beam["center_y_m"]) <= BEAM_CENTER_LIMIT_M
        ),
        "beam_waist": max(waist_errors) <= BEAM_WAIST_RELATIVE_LIMIT,
    }
    return {
        "finite": bool(finite),
        "stationarity": stationarity,
        "polarization": polarization_result,
        "beam": beam,
        "beam_waist_max_relative_error": max(waist_errors),
        "flux": flux,
        "gates": gates,
        "ready": all(gates.values()),
    }, fields


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _output_directory(value: Path) -> Path:
    path = value.expanduser().resolve()
    if not path.is_absolute() or not path.is_dir():
        raise RuntimeError("output directory must be an existing absolute directory")
    if any(path.iterdir()):
        raise RuntimeError("output directory must be empty before source-only solve")
    return path


def run(
    output_directory: Path,
    source: Path,
    polarization: str,
) -> dict[str, Any]:
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
    model = build_model(
        ANCHOR_SPEC,
        polarization,
        total_periods=TOTAL_PERIODS,
        window_periods=WINDOW_PERIODS,
        courant_factor=COURANT_FACTOR,
        include_adjoint_source=False,
        air_only_source_calibration=True,
    )
    arrays, air_audit = all_air_arrays(model)
    if not air_audit["ready"]:
        raise RuntimeError(f"all-air material readback failed: {air_audit}")
    started = time.perf_counter()
    _, output = model["fdtdx"].run_fdtd(
        arrays,
        model["placed"],
        model["config"],
        model["key"],
        show_progress=False,
    )
    marker = output.detector_states["target_field"]["phasor"]
    model["jax"].block_until_ready(marker)
    solve_runtime = time.perf_counter() - started
    evaluation, fields = evaluate_output(model, output, polarization)

    raw_path = output_directory / RAW_NAME
    _atomic_npz(raw_path, **fields)
    payload = {
        "status": (
            "VALIDATED_FDTDX_FRESH_SOURCE_ONLY_CASE"
            if evaluation["ready"]
            else "BLOCKED_FDTDX_FRESH_SOURCE_ONLY_CASE"
        ),
        "ready": evaluation["ready"],
        "polarization": polarization,
        "scope": "all-air source-only on validated fresh anchor",
        "mesh": mesh_audit(ANCHOR_SPEC),
        "time_contract": {
            "total_periods": TOTAL_PERIODS,
            "window_periods": WINDOW_PERIODS,
            "source_startup_periods": 4,
            "courant_factor": COURANT_FACTOR,
            "time_step_s": float(model["config"].time_step_duration),
            "time_steps_total": int(model["config"].time_steps_total),
        },
        "all_air_material_readback": air_audit,
        "evaluation": evaluation,
        "solve_runtime_s": solve_runtime,
        "reporting_incident_power_W": CONTRACT.reporting_incident_power_W,
        "per_case_scale_not_authorized_until_pair_comparison": True,
        "raw": {
            "path": str(raw_path),
            "sha256": _sha256(raw_path),
            "arrays": {name: list(value.shape) for name, value in fields.items()},
        },
        "provenance": provenance,
    }
    _atomic_json(output_directory / JSON_NAME, payload)
    return payload


def main() -> int:
    configured_output = os.environ.get("FDTDX_FRESH_OUTPUT_DIR", "").strip()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(configured_output) if configured_output else None,
    )
    parser.add_argument("--source", type=Path, default=configured_source())
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    args = parser.parse_args()
    if args.output_dir is None:
        parser.error("--output-dir or FDTDX_FRESH_OUTPUT_DIR is required")
    try:
        result = run(args.output_dir, args.source, args.polarization)
    except Exception as error:
        failure = {
            "status": "BLOCKED_FDTDX_FRESH_SOURCE_ONLY_EXCEPTION",
            "ready": False,
            "polarization": args.polarization,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        output = Path(args.output_dir).expanduser().resolve()
        if output.is_dir() and not any(output.iterdir()):
            _atomic_json(output / JSON_NAME, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    summary = {
        "status": result["status"],
        "ready": result["ready"],
        "polarization": result["polarization"],
        "incident_power_W": result["evaluation"]["flux"][
            "incident_plane_signed_W"
        ],
        "maximum_complex_E_NRMSE": result["evaluation"]["stationarity"][
            "maximum_complex_E_NRMSE"
        ],
        "polarization_purity": result["evaluation"]["polarization"]["purity"],
        "gates": result["evaluation"]["gates"],
        "solve_runtime_s": result["solve_runtime_s"],
        "report": str(Path(args.output_dir).resolve() / JSON_NAME),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
