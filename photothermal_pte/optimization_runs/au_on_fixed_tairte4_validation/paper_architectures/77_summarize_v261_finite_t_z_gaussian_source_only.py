#!/usr/bin/env python3
"""Publish the finite T/Z source-only certificate without raw FSP/NPZ files."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_source_only")
OUTPUT = HERE / "results_finite_T_Z_gaussian_source_only"
VALIDATED = {"T": RAW_ROOT / "T_validated", "Z": RAW_ROOT / "Z_validated"}
HISTORY = {
    "T": ["T", "T_calibrated", "T_final", "T_validated"],
    "Z": ["Z", "Z_calibrated", "Z_final", "Z_validated"],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_result(directory: Path, key: str) -> dict:
    path = directory / f"FINITE_{key}_GAUSSIAN_SOURCE_ONLY.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    for artifact in result.get("raw_artifacts", []):
        raw = Path(artifact["path"])
        if raw.stat().st_size != int(artifact["size_bytes"]):
            raise RuntimeError(f"raw size mismatch: {raw}")
        if _sha256(raw) != artifact["sha256"]:
            raise RuntimeError(f"raw SHA mismatch: {raw}")
    return result


def _history_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, directory_names in HISTORY.items():
        for ordinal, name in enumerate(directory_names, start=1):
            directory = RAW_ROOT / name
            path = directory / f"FINITE_{key}_GAUSSIAN_SOURCE_ONLY.json"
            if not path.is_file():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            target = result.get("target_plane_metrics", {})
            setup = result.get("setup", {})
            gates = result.get("gates", {})
            rows.append(
                {
                    "architecture": key,
                    "calibration_ordinal": ordinal,
                    "status": result.get("status"),
                    "source_z_um": float(setup.get("source_bounds_m", {}).get("z", np.nan)) * 1e6,
                    "source_object_w0_um": float(setup.get("Lumerical_source_object_w0_m", np.nan)) * 1e6,
                    "fitted_wx_um": float(target.get("fitted_waist_x_m", np.nan)) * 1e6,
                    "fitted_wy_um": float(target.get("fitted_waist_y_m", np.nan)) * 1e6,
                    "fit_NRMSE_percent": float(target.get("Gaussian_fit_NRMSE", np.nan)) * 100,
                    "ellipticity_percent": float(target.get("fitted_xy_ellipticity", np.nan)) * 100,
                    "power_closure_percent": abs(float(target.get("downward_Poynting_power_over_sourcepower", np.nan)) - 1) * 100,
                    "auto_shutoff": result.get("log_audit", {}).get("final_auto_shutoff"),
                    "wall_time_s": result.get("solver_wall_time_s"),
                    "GPU_memory_GiB": result.get("log_audit", {}).get("precise_GPU_memory_GiB"),
                    "gates_all": bool(gates) and all(gates.values()),
                    "raw_JSON_path": str(path),
                }
            )
    return rows


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {key: _load_result(directory, key) for key, directory in VALIDATED.items()}
    for key, result in results.items():
        expected = f"VALIDATED_FINITE_{key}_GAUSSIAN_SOURCE_ONLY"
        if result.get("status") != expected or not all(result["gates"].values()):
            raise RuntimeError(f"{key} source gate is not validated")

    rows = _history_rows()
    csv_path = OUTPUT / "finite_T_Z_source_calibration_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), constrained_layout=True)
    for column, key in enumerate(("T", "Z")):
        npz = VALIDATED[key] / f"finite_{key}_gaussian_source_only_fields.npz"
        with np.load(npz) as data:
            x = np.asarray(data["target_x_m"]) * 1e6
            y = np.asarray(data["target_y_m"]) * 1e6
            intensity = np.asarray(data["target_downward_intensity_W_m2"], float)
        relative = intensity / np.max(intensity)
        im = axes[0, column].pcolormesh(x, y, relative.T, shading="auto", vmin=0, vmax=1, cmap="inferno")
        axes[0, column].set(
            title=f"{key}: realized target-plane intensity",
            xlabel="x=b (um)",
            ylabel="y=a (um)",
            aspect="equal",
        )
        fig.colorbar(im, ax=axes[0, column], label="I / Imax")
        ix = int(np.argmin(np.abs(x)))
        iy = int(np.argmin(np.abs(y)))
        axes[1, column].plot(x, relative[:, iy], label="x=b cut")
        axes[1, column].plot(y, relative[ix, :], "--", label="y=a cut")
        metrics = results[key]["target_plane_metrics"]
        axes[1, column].set(
            title=(
                f"wx={metrics['fitted_waist_x_m']*1e6:.4f} um, "
                f"wy={metrics['fitted_waist_y_m']*1e6:.4f} um\n"
                f"fit NRMSE={metrics['Gaussian_fit_NRMSE']*100:.4f}%, "
                f"ellipticity={metrics['fitted_xy_ellipticity']*100:.4f}%"
            ),
            xlabel="coordinate (um)",
            ylabel="I / Imax",
            xlim=(-9, 9),
            ylim=(0, 1.03),
        )
        axes[1, column].grid(alpha=0.25)
        axes[1, column].legend()
    fig.suptitle("Validated finite T/Z scalar-Gaussian source-only gates")
    plot = OUTPUT / "finite_T_Z_source_only_realized_beams.png"
    fig.savefig(plot, dpi=190)
    plt.close(fig)

    summary = {
        "status": "VALIDATED_FINITE_T_Z_GAUSSIAN_SOURCE_ONLY",
        "classification": "source-only; no Q, thermal, electrical, PTE, adjoint or optimization",
        "axes": "Lumerical x=b, y=a, propagation -z",
        "validated_cases": results,
        "calibration_history": rows,
        "interpretation": (
            "The source-object field is circular.  The target-plane Maxwell "
            "field has sub-1% polarization-oriented ellipticity and nonzero Ez "
            "because lambda/w0 is near unity.  No field symmetrization or power/Q "
            "rescaling was applied."
        ),
    }
    summary_path = OUTPUT / "FINITE_T_Z_GAUSSIAN_SOURCE_ONLY_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = OUTPUT / "FINITE_T_Z_GAUSSIAN_SOURCE_ONLY_REPORT.md"
    lines = [
        "# Finite T/Z Gaussian source-only report",
        "",
        "Status: `VALIDATED_FINITE_T_Z_GAUSSIAN_SOURCE_ONLY`",
        "",
        "This certificate contains no material, Q, thermal, electrical, PTE, adjoint, or optimization result.",
        "",
        "| case | wx (um) | wy (um) | fit NRMSE | ellipticity | power closure | shutoff | runtime | GPU memory |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("T", "Z"):
        result = results[key]
        metric = result["target_plane_metrics"]
        log = result["log_audit"]
        lines.append(
            f"| {key} | {metric['fitted_waist_x_m']*1e6:.6f} | "
            f"{metric['fitted_waist_y_m']*1e6:.6f} | "
            f"{metric['Gaussian_fit_NRMSE']*100:.6f}% | "
            f"{metric['fitted_xy_ellipticity']*100:.6f}% | "
            f"{abs(metric['downward_Poynting_power_over_sourcepower']-1)*100:.6f}% | "
            f"{log['final_auto_shutoff']:.6g} | {result['solver_wall_time_s']:.2f} s | "
            f"{log['precise_GPU_memory_GiB']:.3f} GiB |"
        )
    lines += [
        "",
        "The first uncalibrated attempts are retained as fail-closed diagnostics in the CSV. Source-object waist calibration is not Q or power rescaling.",
        "",
        "The sub-1% target ellipticity is retained. It is not removed by averaging or symmetrization.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    raw = []
    for key in ("T", "Z"):
        result = results[key]
        raw.extend(result["raw_artifacts"])
        json_path = VALIDATED[key] / f"FINITE_{key}_GAUSSIAN_SOURCE_ONLY.json"
        raw.append({"path": str(json_path), "size_bytes": json_path.stat().st_size, "sha256": _sha256(json_path)})
    published = []
    for path in (summary_path, report, csv_path, plot):
        published.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest = {
        "status": summary["status"],
        "raw_not_committed": raw,
        "published": published,
        "generation_command": f"python {Path(__file__).resolve()}",
    }
    (OUTPUT / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": summary["status"], "output": str(OUTPUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
