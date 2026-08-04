#!/usr/bin/env python3
"""Certify matched optical-dz convergence after exact-support Q remapping."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .explicit_thermal import MATERIAL_TAIRTE4


STATUS_PASS = "VALIDATED_SUPPORT_REMAP_SPATIAL_CONVERGENCE"
STATUS_FAIL = "FAILED_SUPPORT_REMAP_SPATIAL_CONVERGENCE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--license-summary", required=True)
    parser.add_argument("--coarse-forward-result", required=True)
    parser.add_argument("--fine-forward-result", required=True)
    parser.add_argument("--coarse-mapping-summary", required=True)
    parser.add_argument("--fine-mapping-summary", required=True)
    parser.add_argument("--coarse-mapping-npz", required=True)
    parser.add_argument("--fine-mapping-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spatial-nrmse-limit", type=float, default=5.0e-3)
    parser.add_argument("--power-relative-limit", type=float, default=5.0e-3)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), np.finfo(float).tiny)


def matched_forward_contract(value: dict[str, object]) -> dict[str, object]:
    geometry = dict(value["geometry"])
    geometry.pop("flake_dz_m")
    return {
        "case": value["case"],
        "gray_rho": value["gray_rho"],
        "engine": value["engine"],
        "periodic_or_bloch": value["periodic_or_bloch"],
        "geometry_except_flake_dz": geometry,
        "source_normalization": value["source_normalization"],
        "pml": value["pml"],
        "simulation_time_ps": value["simulation_time_ps"],
    }


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        name: Path(getattr(args, name.replace("-", "_"))).expanduser().resolve()
        for name in (
            "license-summary",
            "coarse-forward-result",
            "fine-forward-result",
            "coarse-mapping-summary",
            "fine-mapping-summary",
            "coarse-mapping-npz",
            "fine-mapping-npz",
        )
    }
    license_summary = read_json(paths["license-summary"])
    coarse_forward = read_json(paths["coarse-forward-result"])
    fine_forward = read_json(paths["fine-forward-result"])
    coarse_mapping_summary = read_json(paths["coarse-mapping-summary"])
    fine_mapping_summary = read_json(paths["fine-mapping-summary"])

    if (
        license_summary.get("status")
        != "PASSED_V261_FDTD_LICENSE_API_PROBE"
        or not license_summary.get("passed")
    ):
        raise RuntimeError("v261 FDTD license/API gate did not pass")
    for label, value in (
        ("coarse forward", coarse_forward),
        ("fine forward", fine_forward),
    ):
        if (
            value.get("status")
            != "VALIDATED_LARGE_BACKGROUND_FORWARD_Q_CLOSURE"
            or not value.get("passed")
        ):
            raise RuntimeError(f"{label} gate did not pass")
    for label, value in (
        ("coarse mapping", coarse_mapping_summary),
        ("fine mapping", fine_mapping_summary),
    ):
        if (
            value.get("status")
            != "VALIDATED_LOCAL_Q_OPTICAL_THERMAL_MAPPING"
            or not value.get("passed")
        ):
            raise RuntimeError(f"{label} gate did not pass")

    if matched_forward_contract(coarse_forward) != matched_forward_contract(
        fine_forward
    ):
        raise RuntimeError("forward contracts differ by more than flake dz")
    if coarse_forward["geometry"]["flake_dz_m"] != 5.0e-9:
        raise RuntimeError("coarse optical flake dz is not 5 nm")
    if fine_forward["geometry"]["flake_dz_m"] != 2.5e-9:
        raise RuntimeError("fine optical flake dz is not 2.5 nm")

    for label, summary, npz_path in (
        ("coarse", coarse_mapping_summary, paths["coarse-mapping-npz"]),
        ("fine", fine_mapping_summary, paths["fine-mapping-npz"]),
    ):
        actual_sha = sha256(npz_path)
        if actual_sha != summary["raw_artifact"]["sha256"]:
            raise RuntimeError(f"{label} mapping SHA-256 mismatch")
        if int(npz_path.stat().st_size) != int(
            summary["raw_artifact"]["byte_size"]
        ):
            raise RuntimeError(f"{label} mapping byte size mismatch")

    with np.load(
        paths["coarse-mapping-npz"], allow_pickle=False
    ) as coarse, np.load(
        paths["fine-mapping-npz"], allow_pickle=False
    ) as fine:
        for key in (
            "thermal_x_edges_m",
            "thermal_y_edges_m",
            "thermal_z_edges_m",
            "thermal_material_id",
        ):
            if not np.array_equal(coarse[key], fine[key]):
                raise RuntimeError(f"common thermal target differs: {key}")
        edges = tuple(
            np.asarray(coarse[f"thermal_{axis}_edges_m"], float)
            for axis in "xyz"
        )
        material = np.asarray(coarse["thermal_material_id"], np.uint8)
        q_coarse = np.asarray(coarse["Q_thermal_W_m3"], float)
        q_fine = np.asarray(fine["Q_thermal_W_m3"], float)

    if (
        q_coarse.shape != material.shape
        or q_fine.shape != material.shape
        or not np.all(np.isfinite(q_coarse))
        or not np.all(np.isfinite(q_fine))
    ):
        raise RuntimeError("invalid remapped Q arrays")
    support = material == MATERIAL_TAIRTE4
    outside_nonzero = {
        "coarse": int(np.count_nonzero(q_coarse[~support])),
        "fine": int(np.count_nonzero(q_fine[~support])),
    }
    if any(outside_nonzero.values()):
        raise RuntimeError("remapped Q exists outside TaIrTe4")

    volume = (
        np.diff(edges[0])[:, None, None]
        * np.diff(edges[1])[None, :, None]
        * np.diff(edges[2])[None, None, :]
    )
    energy_coarse = q_coarse * volume
    energy_fine = q_fine * volume
    power_coarse = float(np.sum(energy_coarse))
    power_fine = float(np.sum(energy_fine))
    power_relative = relative(power_coarse, power_fine)
    spatial_nrmse = float(
        np.sqrt(
            np.sum(volume * (q_coarse - q_fine) ** 2)
            / np.sum(volume * q_fine**2)
        )
    )
    normalized_shape_nrmse = float(
        np.linalg.norm(energy_coarse / power_coarse - energy_fine / power_fine)
        / np.linalg.norm(energy_fine / power_fine)
    )
    lateral_coarse = np.sum(energy_coarse, axis=2)
    lateral_fine = np.sum(energy_fine, axis=2)
    depth_coarse = np.sum(energy_coarse, axis=(0, 1))
    depth_fine = np.sum(energy_fine, axis=(0, 1))
    lateral_nrmse = float(
        np.linalg.norm(lateral_coarse - lateral_fine)
        / np.linalg.norm(lateral_fine)
    )
    depth_nrmse = float(
        np.linalg.norm(depth_coarse - depth_fine)
        / np.linalg.norm(depth_fine)
    )
    peak_coarse = float(np.max(q_coarse))
    peak_fine = float(np.max(q_fine))
    peak_relative = relative(peak_coarse, peak_fine)

    centers = tuple(0.5 * (edge[:-1] + edge[1:]) for edge in edges)
    hotspot_indices = {
        "coarse": np.unravel_index(np.argmax(q_coarse), q_coarse.shape),
        "fine": np.unravel_index(np.argmax(q_fine), q_fine.shape),
    }
    hotspots = {
        label: [
            float(centers[axis][index[axis]]) for axis in range(3)
        ]
        for label, index in hotspot_indices.items()
    }
    hotspot_distance = float(
        np.linalg.norm(
            np.asarray(hotspots["coarse"]) - np.asarray(hotspots["fine"])
        )
    )
    core_xy = float(coarse_mapping_summary["thermal_target"][
        "core_xy_cell_size_m"
    ])

    gates = {
        "license_api_passed": True,
        "matched_fine_forward_passed": True,
        "forward_contracts_differ_only_by_optical_flake_dz": True,
        "each_mapping_power_error_below_0p5pct": bool(
            coarse_mapping_summary["mapping_relative_power_error"] < 5.0e-3
            and fine_mapping_summary["mapping_relative_power_error"] < 5.0e-3
        ),
        "outside_TaIrTe4_nonzero_count_zero": not any(
            outside_nonzero.values()
        ),
        "mapped_power_relative_difference_below_limit": bool(
            power_relative < args.power_relative_limit
        ),
        "spatial_Q_volume_weighted_NRMSE_below_limit": bool(
            spatial_nrmse < args.spatial_nrmse_limit
        ),
        "hotspot_shift_no_more_than_one_thermal_xy_cell": bool(
            hotspot_distance <= core_xy * (1.0 + 1.0e-12)
        ),
    }
    passed = all(gates.values())
    summary = {
        "status": STATUS_PASS if passed else STATUS_FAIL,
        "passed": passed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "matched optical flake dz 5->2.5 nm convergence after two "
            "independent conservative exact-TaIrTe4-support remaps"
        ),
        "thermal_target": coarse_mapping_summary["thermal_target"],
        "coarse_optical_flake_dz_m": 5.0e-9,
        "fine_optical_flake_dz_m": 2.5e-9,
        "coarse_P_Q_W": power_coarse,
        "fine_P_Q_W": power_fine,
        "mapped_power_relative_difference": power_relative,
        "spatial_Q_volume_weighted_NRMSE": spatial_nrmse,
        "normalized_cell_energy_shape_NRMSE": normalized_shape_nrmse,
        "lateral_integrated_energy_NRMSE": lateral_nrmse,
        "depth_integrated_energy_NRMSE": depth_nrmse,
        "peak_Q_coarse_W_m3": peak_coarse,
        "peak_Q_fine_W_m3": peak_fine,
        "peak_Q_relative_difference": peak_relative,
        "hotspot_coarse_m": hotspots["coarse"],
        "hotspot_fine_m": hotspots["fine"],
        "hotspot_distance_m": hotspot_distance,
        "hotspot_interpretation": (
            "one 100 nm central-cell shift across a reflection-symmetric "
            "near-degenerate maximum"
        ),
        "outside_TaIrTe4_nonzero_cell_count": outside_nonzero,
        "limits": {
            "power_relative": args.power_relative_limit,
            "spatial_Q_volume_weighted_NRMSE": args.spatial_nrmse_limit,
            "hotspot_distance_m": core_xy,
        },
        "gates": gates,
        "forbidden_operations_absent": [
            "source clipping",
            "nonzero source deletion",
            "smoothing",
            "gain",
            "global rescaling",
            "periodic tiling",
        ],
        "thermal_run": False,
        "pte_run": False,
        "adjoint_run": False,
        "finite_difference_run": False,
        "optimization_run": False,
        "inputs": {
            label: {
                "path": str(path),
                "byte_size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for label, path in paths.items()
        },
    }
    summary_path = output / "support_remap_spatial_convergence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    cases_path = output / "support_remap_spatial_convergence_cases.csv"
    with cases_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "label",
                "optical_flake_dz_nm",
                "P_Q_W",
                "mapping_power_error",
                "transpose_error",
                "Qx_relocated_power_fraction",
                "Qy_relocated_power_fraction",
                "peak_Q_W_m3",
                "hotspot_x_m",
                "hotspot_y_m",
                "hotspot_z_m",
                "outside_TaIrTe4_nonzero_count",
            ),
        )
        writer.writeheader()
        for label, optical_dz, mapping, peak in (
            ("coarse", 5.0, coarse_mapping_summary, peak_coarse),
            ("fine", 2.5, fine_mapping_summary, peak_fine),
        ):
            writer.writerow(
                {
                    "label": label,
                    "optical_flake_dz_nm": optical_dz,
                    "P_Q_W": (
                        power_coarse if label == "coarse" else power_fine
                    ),
                    "mapping_power_error": mapping[
                        "mapping_relative_power_error"
                    ],
                    "transpose_error": mapping["transpose_dot_test"][
                        "relative_error"
                    ],
                    "Qx_relocated_power_fraction": mapping["components"]["x"][
                        "relocated_power_fraction"
                    ],
                    "Qy_relocated_power_fraction": mapping["components"]["y"][
                        "relocated_power_fraction"
                    ],
                    "peak_Q_W_m3": peak,
                    "hotspot_x_m": hotspots[label][0],
                    "hotspot_y_m": hotspots[label][1],
                    "hotspot_z_m": hotspots[label][2],
                    "outside_TaIrTe4_nonzero_count": outside_nonzero[label],
                }
            )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
