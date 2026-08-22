#!/usr/bin/env python3
"""Compare full and Au-truncated v261 backplane controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0e-300)


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.shape != bb.shape:
        raise ValueError(f"shape mismatch: {aa.shape} != {bb.shape}")
    return float(
        np.linalg.norm((aa - bb).reshape(-1))
        / max(np.linalg.norm(aa.reshape(-1)), np.linalg.norm(bb.reshape(-1)), 1.0e-300)
    )


def load_case(directory: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    summary = json.loads((directory / "backplane_case_result.json").read_text())
    if summary["status"] != "COMPLETED_BACKPLANE_TRUNCATION_FORWARD":
        raise RuntimeError(f"case is not completed: {directory}: {summary['status']}")
    artifacts = summary.get("raw_artifacts", [])
    npz_rows = [row for row in artifacts if str(row["path"]).endswith(".npz")]
    if len(npz_rows) != 1:
        raise RuntimeError(f"case must contain exactly one NPZ manifest row: {directory}")
    npz_path = Path(npz_rows[0]["path"])
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if npz_path.stat().st_size != int(npz_rows[0]["size_bytes"]):
        raise RuntimeError(f"NPZ size mismatch: {npz_path}")
    if sha256(npz_path) != npz_rows[0]["sha256"]:
        raise RuntimeError(f"NPZ SHA mismatch: {npz_path}")
    with np.load(npz_path) as raw:
        arrays = {name: np.asarray(raw[name]) for name in raw.files}
    return summary, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--truncated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    full, full_arrays = load_case(args.full_dir.resolve())
    truncated, truncated_arrays = load_case(args.truncated_dir.resolve())
    if full["geometry"]["architecture"] != truncated["geometry"]["architecture"]:
        raise RuntimeError("architecture mismatch")
    if full["geometry"]["substrate_mode"] != "full":
        raise RuntimeError("--full-dir is not a full-substrate case")
    if truncated["geometry"]["substrate_mode"] != "au_truncated":
        raise RuntimeError("--truncated-dir is not an Au-truncated case")

    field_component_nrmse = {
        component: nrmse(
            full_arrays[f"top_E{component}"],
            truncated_arrays[f"top_E{component}"],
        )
        for component in "xyz"
    }
    field_vector_nrmse = nrmse(
        np.stack([full_arrays[f"top_E{component}"] for component in "xyz"], axis=-1),
        np.stack(
            [truncated_arrays[f"top_E{component}"] for component in "xyz"], axis=-1
        ),
    )
    metrics = {
        "P_Q_relative_difference": relative(full["P_Q_W"], truncated["P_Q_W"]),
        "absorptance_relative_difference": relative(
            full["absorptance"], truncated["absorptance"]
        ),
        "reflectance_relative_difference": relative(
            full["reflection"], truncated["reflection"]
        ),
        "reflectance_absolute_difference": abs(
            full["reflection"] - truncated["reflection"]
        ),
        "top_field_vector_NRMSE": field_vector_nrmse,
        "top_field_component_NRMSE": field_component_nrmse,
        "full_transmission": float(full["transmission"]),
        "truncated_transmission": float(truncated["transmission"]),
        "full_closure_relative": float(full["closure_relative"]),
        "truncated_closure_relative": float(truncated["closure_relative"]),
    }
    gates = {
        "both_case_closure_lt_0p5pct": (
            metrics["full_closure_relative"] < 0.005
            and metrics["truncated_closure_relative"] < 0.005
        ),
        "P_Q_relative_difference_lt_0p5pct": metrics["P_Q_relative_difference"] < 0.005,
        "absorptance_relative_difference_lt_0p5pct": (
            metrics["absorptance_relative_difference"] < 0.005
        ),
        "reflectance_absolute_difference_lt_0p5pct": (
            metrics["reflectance_absolute_difference"] < 0.005
        ),
        "top_field_vector_NRMSE_lt_0p5pct": field_vector_nrmse < 0.005,
        "full_transmission_lt_1e_6": abs(metrics["full_transmission"]) < 1.0e-6,
        "no_Q_clipping_smoothing_gain_or_rescaling": True,
    }
    passed = all(gates.values())
    status = (
        "VALIDATED_OPTICAL_SUBSTRATE_TRUNCATION_BELOW_AU_BACKPLANE"
        if passed
        else "FAILED_OPTICAL_SUBSTRATE_TRUNCATION_BELOW_AU_BACKPLANE"
    )
    payload = {
        "status": status,
        "architecture": full["geometry"]["architecture"],
        "scope": (
            "planar mirror discriminator only; validates or rejects removal of SiO2/Si "
            "from the Maxwell domain below the Au backplane; does not validate a reduced "
            "thermal substrate or a full T/Z device"
        ),
        "metrics": metrics,
        "gates": gates,
        "raw_inputs": {
            "full": args.full_dir.resolve().as_posix(),
            "truncated": args.truncated_dir.resolve().as_posix(),
        },
    }
    json_path = output / "backplane_truncation_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    csv_path = output / "backplane_truncation_cases.csv"
    rows = []
    for label, case in (("full", full), ("au_truncated", truncated)):
        rows.append(
            {
                "case": label,
                "P_Q_W": case["P_Q_W"],
                "absorptance": case["absorptance"],
                "reflectance": case["reflection"],
                "transmission": case["transmission"],
                "closure_relative": case["closure_relative"],
                "wall_time_s": case["solver_wall_time_s"],
                "final_auto_shutoff": case["log_audit"]["final_auto_shutoff"],
            }
        )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = ["P_Q", "absorptance", "reflectance", "top E NRMSE"]
    values = [
        100.0 * metrics["P_Q_relative_difference"],
        100.0 * metrics["absorptance_relative_difference"],
        100.0 * metrics["reflectance_absolute_difference"],
        100.0 * metrics["top_field_vector_NRMSE"],
    ]
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    axis.bar(labels, values, color=["#3569a8", "#4c956c", "#d17b36", "#7353ba"])
    axis.axhline(0.5, color="black", linestyle="--", label="0.5% gate")
    axis.set_ylabel("difference (%)")
    axis.set_title("Full SiO2/Si vs Au-truncated optical backplane control")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    plot_path = output / "backplane_truncation_comparison.png"
    figure.savefig(plot_path, dpi=220)
    plt.close(figure)

    report_path = output / "OPTICAL_BACKPLANE_TRUNCATION_REPORT.md"
    report_path.write_text(
        f"""# Optical substrate truncation below the Au backplane

Status: `{status}`

This is a planar v261 GPU discriminator. It does not replace the TaIrTe4 T/Z
device calculation and it does not validate any thermal reduction.

| metric | value |
|---|---:|
| P_Q relative difference | {100*metrics['P_Q_relative_difference']:.6f}% |
| absorptance relative difference | {100*metrics['absorptance_relative_difference']:.6f}% |
| reflectance absolute difference | {100*metrics['reflectance_absolute_difference']:.6f}% |
| top-field vector NRMSE | {100*metrics['top_field_vector_NRMSE']:.6f}% |
| full transmission | {metrics['full_transmission']:.6e} |
| full closure | {100*metrics['full_closure_relative']:.6f}% |
| truncated closure | {100*metrics['truncated_closure_relative']:.6f}% |

No Q clipping, smoothing, gain, global rescaling, or polarization matching was used.
The full paper stack remains the reference unless every gate above passes.

![comparison](backplane_truncation_comparison.png)
"""
    )
    manifest = {
        "status": status,
        "generated_files": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (json_path, csv_path, plot_path, report_path)
        ],
        "raw_inputs_not_copied_to_git": [
            row
            for case in (full, truncated)
            for row in case.get("raw_artifacts", [])
        ],
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"status": status, "metrics": metrics, "gates": gates}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
