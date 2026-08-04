#!/usr/bin/env python3
"""Compare scalar and imported-permittivity optical endpoints."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .native_yee_q import trapezoid_weights


STATUS_PASS = "VALIDATED_IMPORTED_PERMITTIVITY_ENDPOINT_EQUIVALENCE"
STATUS_FAIL = "FAILED_IMPORTED_PERMITTIVITY_ENDPOINT_EQUIVALENCE"
METRIC_LIMIT = 5.0e-3
LOW_POWER_FRACTION = 1.0e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help=(
            "endpoint,dz_nm,scalar_result,scalar_npz,"
            "imported_result,imported_npz"
        ),
    )
    parser.add_argument(
        "--refinement-case",
        action="append",
        default=[],
        help="endpoint,dz_nm,representation,result,npz",
    )
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--generation-command", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
    }


def parse_pair(value: str) -> dict[str, object]:
    parts = value.split(",")
    if len(parts) != 6:
        raise ValueError(f"invalid pair: {value}")
    endpoint, dz, scalar_result, scalar_npz, imported_result, imported_npz = (
        parts
    )
    if endpoint not in {"rho0", "rho1"}:
        raise ValueError(endpoint)
    paths = [
        Path(item).expanduser().resolve()
        for item in (
            scalar_result,
            scalar_npz,
            imported_result,
            imported_npz,
        )
    ]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(paths)
    return {
        "endpoint": endpoint,
        "dz_nm": float(dz),
        "scalar_result_path": paths[0],
        "scalar_npz_path": paths[1],
        "imported_result_path": paths[2],
        "imported_npz_path": paths[3],
    }


def parse_refinement_case(value: str) -> dict[str, object]:
    parts = value.split(",")
    if len(parts) != 5:
        raise ValueError(f"invalid refinement case: {value}")
    endpoint, dz, representation, result, npz = parts
    if endpoint not in {"rho0", "rho1"}:
        raise ValueError(endpoint)
    if representation not in {"scalar", "imported"}:
        raise ValueError(representation)
    paths = [
        Path(item).expanduser().resolve() for item in (result, npz)
    ]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(paths)
    return {
        "endpoint": endpoint,
        "dz_nm": float(dz),
        "representation": representation,
        "result_path": paths[0],
        "npz_path": paths[1],
    }


def coordinates(data, component: str) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(data[f"Q{component}_{axis}_m"], float)
        for axis in "xyz"
    )


def interpolate(
    values: np.ndarray,
    source_coordinates: tuple[np.ndarray, ...],
    target_coordinates: tuple[np.ndarray, ...],
) -> np.ndarray:
    if all(
        np.array_equal(source, target)
        for source, target in zip(source_coordinates, target_coordinates)
    ):
        return np.asarray(values)
    mesh = np.meshgrid(*target_coordinates, indexing="ij")
    points = np.column_stack([axis.reshape(-1) for axis in mesh])
    real = RegularGridInterpolator(
        source_coordinates,
        np.real(values),
        method="linear",
        bounds_error=True,
    )(points)
    if np.iscomplexobj(values):
        imag = RegularGridInterpolator(
            source_coordinates,
            np.imag(values),
            method="linear",
            bounds_error=True,
        )(points)
        real = real + 1j * imag
    return np.asarray(real).reshape(
        tuple(axis.size for axis in target_coordinates)
    )


def weighted_norm_squared(
    values: np.ndarray, coordinate: tuple[np.ndarray, ...]
) -> float:
    weights = [trapezoid_weights(axis) for axis in coordinate]
    return float(
        np.einsum(
            "i,j,k,ijk->",
            weights[0],
            weights[1],
            weights[2],
            np.abs(values) ** 2,
            optimize=True,
        )
    )


def relative_l2(
    value: np.ndarray,
    reference: np.ndarray,
    coordinate: tuple[np.ndarray, ...],
) -> float:
    delta = weighted_norm_squared(value - reference, coordinate)
    norm = max(
        weighted_norm_squared(value, coordinate),
        weighted_norm_squared(reference, coordinate),
        np.finfo(float).tiny,
    )
    return float(np.sqrt(delta / norm))


def relative_scalar(value: float, reference: float) -> float:
    return abs(value - reference) / max(
        abs(value), abs(reference), np.finfo(float).tiny
    )


def compare_npz(
    reference_path: Path,
    candidate_path: Path,
) -> dict[str, object]:
    reference = np.load(reference_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)
    field_delta = 0.0
    field_norm = 0.0
    q_delta = 0.0
    q_norm = 0.0
    index_delta = 0.0
    index_norm = 0.0
    components = {}
    all_coordinates_equal = True
    for component in "xyz":
        target_coordinate = coordinates(reference, component)
        source_coordinate = coordinates(candidate, component)
        coordinate_equal = all(
            np.array_equal(left, right)
            for left, right in zip(target_coordinate, source_coordinate)
        )
        all_coordinates_equal &= coordinate_equal
        e_reference = np.asarray(reference[f"E{component}_V_m"])
        e_candidate = interpolate(
            np.asarray(candidate[f"E{component}_V_m"]),
            source_coordinate,
            target_coordinate,
        )
        q_reference = np.asarray(reference[f"Q{component}_W_m3"], float)
        q_candidate = interpolate(
            np.asarray(candidate[f"Q{component}_W_m3"], float),
            source_coordinate,
            target_coordinate,
        )
        n_reference = np.asarray(reference[f"index_{component}"])
        n_candidate = interpolate(
            np.asarray(candidate[f"index_{component}"]),
            source_coordinate,
            target_coordinate,
        )
        e_delta = weighted_norm_squared(
            e_candidate - e_reference, target_coordinate
        )
        e_norm = max(
            weighted_norm_squared(e_candidate, target_coordinate),
            weighted_norm_squared(e_reference, target_coordinate),
        )
        q_component_delta = weighted_norm_squared(
            q_candidate - q_reference, target_coordinate
        )
        q_component_norm = max(
            weighted_norm_squared(q_candidate, target_coordinate),
            weighted_norm_squared(q_reference, target_coordinate),
        )
        n_delta = weighted_norm_squared(
            n_candidate - n_reference, target_coordinate
        )
        n_norm = max(
            weighted_norm_squared(n_candidate, target_coordinate),
            weighted_norm_squared(n_reference, target_coordinate),
        )
        field_delta += e_delta
        field_norm += e_norm
        q_delta += q_component_delta
        q_norm += q_component_norm
        index_delta += n_delta
        index_norm += n_norm
        components[component] = {
            "coordinate_equal": coordinate_equal,
            "complex_field_NRMSE": float(
                np.sqrt(e_delta / max(e_norm, np.finfo(float).tiny))
            ),
            "spatial_Q_NRMSE": (
                0.0
                if q_component_norm == 0.0
                and q_component_delta == 0.0
                else float(
                    np.sqrt(
                        q_component_delta
                        / max(q_component_norm, np.finfo(float).tiny)
                    )
                )
            ),
            "index_NRMSE": float(
                np.sqrt(n_delta / max(n_norm, np.finfo(float).tiny))
            ),
        }
    return {
        "all_component_coordinates_equal": all_coordinates_equal,
        "common_grid": (
            "reference native Yee coordinates; candidate is used directly "
            "when identical, otherwise trilinearly interpolated"
        ),
        "combined_complex_field_NRMSE": float(
            np.sqrt(field_delta / max(field_norm, np.finfo(float).tiny))
        ),
        "combined_spatial_Q_NRMSE": (
            0.0
            if q_norm == 0.0 and q_delta == 0.0
            else float(np.sqrt(q_delta / max(q_norm, np.finfo(float).tiny)))
        ),
        "combined_index_NRMSE": float(
            np.sqrt(index_delta / max(index_norm, np.finfo(float).tiny))
        ),
        "components": components,
    }


def power_metrics(
    reference: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    ref_components = reference["Q"]["component_power_W"]
    candidate_components = candidate["Q"]["component_power_W"]
    p_reference = float(reference["Q"]["P_Q_W"])
    component = {}
    for axis in "xyz":
        ref = float(ref_components[axis])
        value = float(candidate_components[axis])
        fraction = max(abs(ref), abs(value)) / max(
            abs(p_reference), np.finfo(float).tiny
        )
        component[axis] = {
            "reference_W": ref,
            "candidate_W": value,
            "relative_difference": (
                0.0
                if ref == 0.0 and value == 0.0
                else relative_scalar(value, ref)
            ),
            "maximum_fraction_of_total_P_Q": fraction,
            "low_power_diagnostic": fraction < LOW_POWER_FRACTION,
        }
    return {
        "P_Q_reference_W": p_reference,
        "P_Q_candidate_W": float(candidate["Q"]["P_Q_W"]),
        "P_Q_relative_difference": relative_scalar(
            float(candidate["Q"]["P_Q_W"]), p_reference
        ),
        "P_six_reference_W": float(reference["six_face"]["P_six_W"]),
        "P_six_candidate_W": float(candidate["six_face"]["P_six_W"]),
        "P_six_relative_difference": relative_scalar(
            float(candidate["six_face"]["P_six_W"]),
            float(reference["six_face"]["P_six_W"]),
        ),
        "Q_components": component,
    }


def main() -> int:
    args = parse_args()
    pairs = [parse_pair(value) for value in args.pair]
    if {(pair["endpoint"], pair["dz_nm"]) for pair in pairs} != {
        ("rho0", 5.0),
        ("rho0", 2.5),
        ("rho1", 5.0),
        ("rho1", 2.5),
    }:
        raise RuntimeError("exact rho0/rho1 x dz5/dz2.5 matrix required")
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    artifacts = []
    loaded: dict[tuple[str, float, str], tuple[dict, Path]] = {}
    for pair in pairs:
        scalar = json.loads(pair["scalar_result_path"].read_text())
        imported = json.loads(pair["imported_result_path"].read_text())
        if not scalar.get("passed") or not imported.get("passed"):
            raise RuntimeError("all forward cases must pass closure first")
        if scalar["design_representation_requested"] != "scalar":
            raise RuntimeError("scalar case mislabeled")
        if (
            imported["design_representation_requested"]
            != "imported-permittivity"
        ):
            raise RuntimeError("imported case mislabeled")
        bounds = imported["design_representation"][
            "object_bounds_readback_m"
        ]
        expected = {
            "x": [-1e-6, 1e-6],
            "y": [-1e-6, 1e-6],
            "z": [0.0, 600e-9],
        }
        bounds_error = max(
            abs(bounds[axis][side] - expected[axis][side])
            for axis in "xyz"
            for side in (0, 1)
        )
        spatial = compare_npz(
            pair["scalar_npz_path"], pair["imported_npz_path"]
        )
        power = power_metrics(scalar, imported)
        major_component_errors = [
            value["relative_difference"]
            for value in power["Q_components"].values()
            if not value["low_power_diagnostic"]
        ]
        gated = {
            "P_Q": power["P_Q_relative_difference"],
            "P_six": power["P_six_relative_difference"],
            "complex_field": spatial["combined_complex_field_NRMSE"],
            "spatial_Q": spatial["combined_spatial_Q_NRMSE"],
            "index": spatial["combined_index_NRMSE"],
            "major_Q_component": max(major_component_errors, default=0.0),
        }
        comparison = {
            "kind": "representation_equivalence",
            "endpoint": pair["endpoint"],
            "flake_dz_nm": pair["dz_nm"],
            "scalar_status": scalar["status"],
            "imported_status": imported["status"],
            "imported_sample_shape": imported["design_representation"][
                "sample_shape"
            ],
            "imported_bounds_max_abs_error_m": bounds_error,
            "power": power,
            "spatial": spatial,
            "gated_metrics": gated,
            "maximum_gated_metric": max(gated.values()),
        }
        comparisons.append(comparison)
        for role, result_path, npz_path in (
            (
                "scalar",
                pair["scalar_result_path"],
                pair["scalar_npz_path"],
            ),
            (
                "imported",
                pair["imported_result_path"],
                pair["imported_npz_path"],
            ),
        ):
            loaded[(pair["endpoint"], pair["dz_nm"], role)] = (
                scalar if role == "scalar" else imported,
                npz_path,
            )
            artifacts.extend(
                [
                    {
                        "role": (
                            f"{pair['endpoint']}_dz{pair['dz_nm']:g}_"
                            f"{role}_result"
                        ),
                        **artifact(result_path),
                    },
                    {
                        "role": (
                            f"{pair['endpoint']}_dz{pair['dz_nm']:g}_"
                            f"{role}_native_field_Q"
                        ),
                        **artifact(npz_path),
                    },
                    {
                        "role": (
                            f"{pair['endpoint']}_dz{pair['dz_nm']:g}_"
                            f"{role}_FSP"
                        ),
                        **(
                            scalar if role == "scalar" else imported
                        )["artifacts"]["fsp"],
                    },
                ]
            )

    refinement_cases = [
        parse_refinement_case(value) for value in args.refinement_case
    ]
    for case in refinement_cases:
        key = (
            case["endpoint"],
            case["dz_nm"],
            case["representation"],
        )
        if key in loaded:
            raise RuntimeError(f"duplicate refinement case: {key}")
        result = json.loads(case["result_path"].read_text())
        if not result.get("passed"):
            raise RuntimeError(f"refinement forward case failed: {key}")
        expected_representation = (
            "imported-permittivity"
            if case["representation"] == "imported"
            else "scalar"
        )
        if (
            result["design_representation_requested"]
            != expected_representation
        ):
            raise RuntimeError(f"refinement case mislabeled: {key}")
        loaded[key] = (result, case["npz_path"])
        artifacts.extend(
            [
                {
                    "role": (
                        f"{case['endpoint']}_dz{case['dz_nm']:g}_"
                        f"{case['representation']}_result"
                    ),
                    **artifact(case["result_path"]),
                },
                {
                    "role": (
                        f"{case['endpoint']}_dz{case['dz_nm']:g}_"
                        f"{case['representation']}_native_field_Q"
                    ),
                    **artifact(case["npz_path"]),
                },
                {
                    "role": (
                        f"{case['endpoint']}_dz{case['dz_nm']:g}_"
                        f"{case['representation']}_FSP"
                    ),
                    **result["artifacts"]["fsp"],
                },
            ]
        )

    mesh_comparisons = []
    for endpoint in ("rho0", "rho1"):
        for representation in ("scalar", "imported"):
            dz_values = sorted(
                (
                    key[1]
                    for key in loaded
                    if key[0] == endpoint and key[2] == representation
                ),
                reverse=True,
            )
            for coarse_dz, fine_dz in zip(dz_values, dz_values[1:]):
                coarse_result, coarse_npz = loaded[
                    (endpoint, coarse_dz, representation)
                ]
                fine_result, fine_npz = loaded[
                    (endpoint, fine_dz, representation)
                ]
                spatial = compare_npz(coarse_npz, fine_npz)
                power = power_metrics(coarse_result, fine_result)
                major_component_errors = [
                    value["relative_difference"]
                    for value in power["Q_components"].values()
                    if not value["low_power_diagnostic"]
                ]
                gated = {
                    "P_Q": power["P_Q_relative_difference"],
                    "P_six": power["P_six_relative_difference"],
                    "complex_field": spatial[
                        "combined_complex_field_NRMSE"
                    ],
                    "spatial_Q": spatial["combined_spatial_Q_NRMSE"],
                    "major_Q_component": max(
                        major_component_errors, default=0.0
                    ),
                }
                promoted = bool(
                    representation == "scalar"
                    and fine_dz == min(dz_values)
                )
                mesh_comparisons.append(
                    {
                        "kind": (
                            f"mesh_convergence_dz{coarse_dz:g}_to_"
                            f"dz{fine_dz:g}"
                        ),
                        "endpoint": endpoint,
                        "representation": representation,
                        "coarse_flake_dz_nm": coarse_dz,
                        "fine_flake_dz_nm": fine_dz,
                        "promoted_for_gate": promoted,
                        "power": power,
                        "spatial": spatial,
                        "gated_metrics": gated,
                        "maximum_gated_metric": max(gated.values()),
                    }
                )
    worst_representation = max(
        record["maximum_gated_metric"] for record in comparisons
    )
    promoted_mesh = [
        record
        for record in mesh_comparisons
        if record["promoted_for_gate"]
    ]
    if {record["endpoint"] for record in promoted_mesh} != {
        "rho0",
        "rho1",
    }:
        raise RuntimeError("one promoted scalar mesh pair per endpoint required")
    worst_mesh = max(
        record["maximum_gated_metric"] for record in promoted_mesh
    )
    worst_all_mesh = max(
        record["maximum_gated_metric"] for record in mesh_comparisons
    )
    worst_bounds = max(
        record["imported_bounds_max_abs_error_m"]
        for record in comparisons
    )
    passed = bool(
        worst_representation < METRIC_LIMIT
        and worst_mesh < METRIC_LIMIT
        and worst_bounds < 2e-18
    )
    summary = {
        "status": STATUS_PASS if passed else STATUS_FAIL,
        "passed": passed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "v261 matched CPU-TFSF scalar-material versus 81x81x13 "
            "imported-permittivity endpoint equivalence"
        ),
        "metric_limit": METRIC_LIMIT,
        "low_Q_component_power_fraction": LOW_POWER_FRACTION,
        "comparisons": comparisons,
        "mesh_convergence": mesh_comparisons,
        "gates": {
            "worst_representation_equivalence_metric": (
                worst_representation
            ),
            "worst_mesh_convergence_metric": worst_mesh,
            "worst_all_recorded_mesh_metric_including_coarse_failures": (
                worst_all_mesh
            ),
            "worst_imported_bounds_error_m": worst_bounds,
        },
        "mesh_gate_policy": (
            "Scalar/imported representation equivalence is gated at dz=5 "
            "and 2.5 nm. Physical mesh convergence is gated on the finest "
            "available scalar pair for each endpoint; all coarser failures "
            "remain in mesh_convergence. This is valid only because the two "
            "representations have identical discrete fields and Q at both "
            "matched meshes."
        ),
        "next_gate": (
            "COMBINED_PHYSICAL_RHO_PTE_ADFD"
            if passed
            else "STOP_FAIL_CLOSED"
        ),
        "optimization_run": False,
    }
    summary_path = (
        report_dir / "imported_permittivity_endpoint_summary.json"
    )
    csv_path = report_dir / "imported_permittivity_endpoint_cases.csv"
    report_path = (
        report_dir / "IMPORTED_PERMITTIVITY_ENDPOINT_REPORT.md"
    )
    manifest_path = (
        report_dir
        / "IMPORTED_PERMITTIVITY_ENDPOINT_RAW_ARTIFACT_MANIFEST.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    rows = []
    for record in comparisons + mesh_comparisons:
        rows.append(
            {
                "kind": record["kind"],
                "endpoint": record["endpoint"],
                "representation": record.get(
                    "representation", "scalar_vs_imported"
                ),
                "flake_dz_nm": (
                    record["flake_dz_nm"]
                    if "flake_dz_nm" in record
                    else (
                        f"{record['coarse_flake_dz_nm']:g}_to_"
                        f"{record['fine_flake_dz_nm']:g}"
                    )
                ),
                "promoted_for_gate": record.get(
                    "promoted_for_gate", False
                ),
                **record["gated_metrics"],
                "maximum_gated_metric": record["maximum_gated_metric"],
            }
        )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    table = [
        "| "
        + " | ".join(
            [
                row["kind"],
                row["endpoint"],
                str(row["representation"]),
                str(row["flake_dz_nm"]),
                str(row["promoted_for_gate"]),
                f"{row['P_Q']:.6e}",
                f"{row['P_six']:.6e}",
                f"{row['complex_field']:.6e}",
                f"{row['spatial_Q']:.6e}",
                f"{row['maximum_gated_metric']:.6e}",
            ]
        )
        + " |"
        for row in rows
    ]
    report = f"""# Imported-permittivity endpoint equivalence

Status: `{summary['status']}`

Every case uses the matched rho=0.5 checkpoint environment except for the
requested endpoint and representation: CPU TFSF, six PML, PML 32,
stabilized x/y and standard z, 7.2 µm outer x-y, identical source/Q bounds,
and central incident intensity 1 W/m².

The imported object uses exact 81×81×13 samples on x,y=[-1,1] µm and
z=[0,600] nm.  Scalar and imported endpoints are compared on common native
Yee coordinates.  Complex fields are compared without phase fitting.
Qx/Qy/Qz powers and spatial fields are retained separately; a component
below `{LOW_POWER_FRACTION:.1e}` of total P_Q is reported as a low-power
diagnostic and is not allowed to dominate the major-component gate.

| kind | endpoint | representation | flake dz (nm) | promoted | P_Q diff | P_six diff | field NRMSE | spatial-Q NRMSE | worst gate |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

Gate limit: `{METRIC_LIMIT:.6e}`.

- Worst scalar/imported endpoint metric:
  `{worst_representation:.6e}`.
- Worst promoted finest-pair mesh metric:
  `{worst_mesh:.6e}`.
- Worst recorded mesh metric including the preserved coarse failures:
  `{worst_all_mesh:.6e}`.
- Worst imported-object bounds error:
  `{worst_bounds:.6e} m`.

The rho1 raw spatial-Q trace is explicitly retained through
5→2.5→1.25→0.625 nm. The gate uses the finest scalar pair, while endpoint
representation equivalence is independently checked at both 5 and 2.5 nm.
There is no bitwise-equality requirement. No thermal solve, adjoint,
gradient, transient, or optimization is run by this checkpoint.
"""
    report_path.write_text(report)
    manifest = {
        "status": summary["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_command": args.generation_command,
        "artifacts": artifacts,
        "git_policy": (
            "FSP and native field/Q NPZ artifacts stay outside Git; "
            "paths, sizes, and SHA-256 values are committed here"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
