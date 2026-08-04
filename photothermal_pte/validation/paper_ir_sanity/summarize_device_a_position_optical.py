#!/usr/bin/env python3
"""Summarize the frozen Device-A three-position optical cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dual_widths(coordinates: np.ndarray, bounds: list[float]) -> np.ndarray:
    values = np.asarray(coordinates, float)
    edges = np.empty(values.size + 1, float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0], edges[-1] = map(float, bounds)
    edges = np.clip(edges, float(bounds[0]), float(bounds[1]))
    if np.any(np.diff(edges) < 0.0):
        raise ValueError("non-monotonic bounded dual-cell edges")
    return np.diff(edges)


def artifact(path: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "committed_to_git": False,
    }


def load_case(label: str, polarization: str, s_um: float, path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    result_path = path / "case_result.json"
    q_path = path / "finite_q_on_artifact.npz"
    fsp_path = path / "finite_2um_optical_q.fsp"
    result = json.loads(result_path.read_text())
    run = result["run_result"]
    with np.load(q_path) as raw:
        metadata = json.loads(str(raw["metadata_json"][0]))
        bounds = metadata["realized_pre_run_contract"]["geometry"][
            "pabs_nominal_control_volume_bounds_m"
        ]
        wx = dual_widths(raw["x_m"], bounds["x"])
        wy = dual_widths(raw["y_m"], bounds["y"])
        wz = dual_widths(raw["z_m"], bounds["z"])
        q = np.asarray(raw["Q_on_W_m3"], float)
        reintegrated = float(
            np.sum(q * wx[:, None, None] * wy[None, :, None] * wz[None, None, :])
        )
        finite = bool(np.all(np.isfinite(q)))
        negative = int(np.count_nonzero(q < 0.0))
        shape = list(q.shape)
        coordinate_bounds = {
            "x": [float(raw["x_m"][0]), float(raw["x_m"][-1])],
            "y": [float(raw["y_m"][0]), float(raw["y_m"][-1])],
            "z": [float(raw["z_m"][0]), float(raw["z_m"][-1])],
        }
    acceptance = run["acceptance"]
    source = result["pre_run_contract"]["geometry"]["source"]
    pq = float(run["P_Q_W"])
    clearance = source.get("source_aperture_PML_clearance_m")
    if clearance is None:
        domain = result["pre_run_contract"]["geometry"]["domain_bounds_m"]
        half_span = 0.5 * float(source["source_span_m"])
        center_x, center_y = map(float, source["beam_center_m"])
        clearance = {
            "x_min": center_x - half_span - float(domain["x"][0]),
            "x_max": float(domain["x"][1]) - center_x - half_span,
            "y_min": center_y - half_span - float(domain["y"][0]),
            "y_max": float(domain["y"][1]) - center_y - half_span,
        }
    row: dict[str, object] = {
        "position_label": label,
        "signed_s_from_edge_um": s_um,
        "polarization": polarization,
        "source_center_x_um": float(source["beam_center_m"][0]) * 1e6,
        "source_center_y_um": float(source["beam_center_m"][1]) * 1e6,
        "P_Q_W_at_1_W_m2": pq,
        "P_Q_reintegrated_W": reintegrated,
        "Q_reintegration_relative_error": abs(reintegrated - pq) / abs(pq),
        "P_six_W_at_1_W_m2": float(run["P_six_face_W"]),
        "six_face_closure": float(run["six_face_relative_closure"]),
        "auto_shutoff_final": float(run["auto_shutoff"]["final_value"]),
        "P_Qx_W": float(run["component_power_W"]["x"]),
        "P_Qy_W": float(run["component_power_W"]["y"]),
        "P_Qz_W": float(run["component_power_W"]["z"]),
        "negative_Q_voxel_count": negative,
        "Q_finite": finite,
        "Q_shape": shape,
        "Q_coordinate_bounds_m": coordinate_bounds,
        "source_aperture_minimum_PML_clearance_m": min(
            float(value) for value in clearance.values()
        ),
        "case_acceptance_all_true": bool(all(acceptance.values())),
        "top_level_promoted_validated_flag": bool(result["validated"]),
        "top_level_flag_interpretation": (
            "legacy base-script promotion flag; individual production optical gates are audited independently"
        ),
    }
    raw_artifacts = [
        artifact(q_path, f"raw Q {label} E||{polarization}"),
        artifact(fsp_path, f"raw FSP {label} E||{polarization}"),
        artifact(result_path, f"case result {label} E||{polarization}"),
    ]
    return row, raw_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for label in ("sminus1", "s0", "splus1"):
        for polarization in ("a", "b"):
            parser.add_argument(f"--{label}-{polarization}", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scan = json.loads(args.scan_contract.read_text())
    scan_by_label = {item["label"]: item for item in scan["cases"]}
    scan_alias = {"sminus1": "s-1um", "s0": "s0", "splus1": "s+1um"}
    rows: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    for label in ("sminus1", "s0", "splus1"):
        for polarization in ("a", "b"):
            row, case_artifacts = load_case(
                label,
                polarization,
                float(scan_by_label[scan_alias[label]]["signed_s_from_edge_um"]),
                getattr(args, f"{label}_{polarization}"),
            )
            rows.append(row)
            artifacts.extend(case_artifacts)
    fixed_shapes = {tuple(row["Q_shape"]) for row in rows}
    fixed_bounds = {
        json.dumps(row["Q_coordinate_bounds_m"], sort_keys=True) for row in rows
    }
    gates = {
        "all_case_acceptance_items_true": all(
            bool(row["case_acceptance_all_true"]) for row in rows
        ),
        "all_six_face_closure_lt_0p5percent": all(
            float(row["six_face_closure"]) < 0.005 for row in rows
        ),
        "all_auto_shutoff_lt_or_equal_1e_5": all(
            float(row["auto_shutoff_final"]) <= 1.0e-5 for row in rows
        ),
        "all_Q_reintegration_error_lt_0p5percent": all(
            float(row["Q_reintegration_relative_error"]) < 0.005 for row in rows
        ),
        "all_Q_finite_nonnegative": all(
            bool(row["Q_finite"]) and int(row["negative_Q_voxel_count"]) == 0
            for row in rows
        ),
        "fixed_common_Q_shape_and_bounds": len(fixed_shapes) == 1
        and len(fixed_bounds) == 1,
        "source_aperture_PML_clearance_at_least_1um": all(
            float(row["source_aperture_minimum_PML_clearance_m"]) >= 1.0e-6
            for row in rows
        ),
    }
    summary = {
        "status": (
            "VALIDATED_DEVICE_A_THREE_POSITION_OPTICAL_GATE"
            if all(gates.values())
            else "FAILED_DEVICE_A_THREE_POSITION_OPTICAL_GATE"
        ),
        "scope": "six optical cases: four new GPU solves plus two immutable s0 artifacts",
        "promotion_note": (
            "case_result top-level validated=false is preserved; promotion here is based on the explicit per-case gates and independent raw-Q reintegration"
        ),
        "gates": gates,
        "cases": rows,
        "no_Q_clipping_smoothing_gain_rescaling": True,
        "no_CPU_FDTD_fallback": True,
        "no_25nm_or_12p5nm_refinement": True,
    }
    (args.output_dir / "device_a_position_optical_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                key: json.dumps(value) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    with (args.output_dir / "device_a_position_optical_cases.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    manifest = {
        "status": "RECORDED_EXTERNAL_RAW_ARTIFACTS_NOT_COMMITTED",
        "artifacts": artifacts,
        "generation_command": " ".join(sys.argv),
    }
    (args.output_dir / "RAW_ARTIFACT_MANIFEST_POSITION_OPTICAL.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for polarization, color in (("a", "tab:blue"), ("b", "tab:orange")):
        selected = sorted(
            (row for row in rows if row["polarization"] == polarization),
            key=lambda row: float(row["signed_s_from_edge_um"]),
        )
        s = [float(row["signed_s_from_edge_um"]) for row in selected]
        pq = [float(row["P_Q_W_at_1_W_m2"]) for row in selected]
        closure = [100.0 * float(row["six_face_closure"]) for row in selected]
        axes[0].plot(s, pq, "o-", color=color, label=rf"$E\parallel {polarization}$")
        axes[1].plot(s, closure, "o-", color=color, label=rf"$E\parallel {polarization}$")
    axes[0].set(xlabel="signed s from digitized edge (µm)", ylabel=r"raw $P_Q$ at 1 W m$^{-2}$ (W)")
    axes[1].set(xlabel="signed s from digitized edge (µm)", ylabel="six-face closure (%)")
    axes[1].axhline(0.5, color="black", linestyle="--", label="0.5% gate")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(args.output_dir / "DEVICE_A_POSITION_OPTICAL_POWER_AND_CLOSURE.png", dpi=180)
    plt.close(figure)
    lines = []
    for row in rows:
        lines.append(
            f"| {row['signed_s_from_edge_um']:.1f} | E||{row['polarization']} | "
            f"{row['P_Q_W_at_1_W_m2']:.9e} | {row['P_six_W_at_1_W_m2']:.9e} | "
            f"{100.0 * row['six_face_closure']:.4f}% | {row['auto_shutoff_final']:.6e} |"
        )
    report = f"""# Device-A three-position optical checkpoint

Status: `{summary['status']}`

Four new GPU FDTD solves and the two immutable `s0` artifacts were audited on
one fixed Device-A/PML/monitor/material/mesh contract. Only source center and
polarization change. No CPU FDTD fallback or Q clipping, smoothing, gain,
rescaling, or polarization matching was used.

| signed s (um) | polarization | P_Q (W) | P_six (W) | closure | auto-shutoff |
|---:|---|---:|---:|---:|---:|
{chr(10).join(lines)}

The legacy top-level `validated=false` field in each production case result is
preserved. This checkpoint uses the explicit acceptance items plus an
independent raw-NPZ dual-cell reintegration; it does not rewrite raw metadata.
"""
    (args.output_dir / "DEVICE_A_POSITION_OPTICAL_REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
