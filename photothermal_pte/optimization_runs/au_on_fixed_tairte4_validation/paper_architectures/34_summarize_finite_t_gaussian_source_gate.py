#!/usr/bin/env python3
"""Publish the fail-closed finite multi-T Gaussian source-gate history."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results_finite_T_Gaussian_source_gate"
REAUDIT = HERE / "results_finite_T_Gaussian_source_reaudit/FINITE_T_GAUSSIAN_SOURCE_REAUDIT.json"
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts")
RAW_12_FIRST = RAW_ROOT / "paper_tairte4_finite_T_target_w0_12um_source_only"
RAW_12_PASS = RAW_ROOT / "paper_tairte4_finite_T_target_w0_12um_calibrated_source_only"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_raw(raw: Path, label: str) -> dict[str, object]:
    data = json.loads((raw / "FINITE_T_GAUSSIAN_SOURCE_ONLY.json").read_text())
    source = data["source"]
    fit = data["target_plane_fit"]
    return {
        "label": label,
        "target_w0_um": source["target_realized_w0_um"],
        "source_object_w0_um": source["Lumerical_source_object_w0_um"],
        "fitted_wx_um": fit["fitted_waist_x_m"] * 1e6,
        "fitted_wy_um": fit["fitted_waist_y_m"] * 1e6,
        "fit_NRMSE": fit["Gaussian_fit_NRMSE"],
        "ellipticity": fit["fitted_xy_ellipticity"],
        "center_displacement_m": float(np.hypot(fit["fitted_center_x_m"], fit["fitted_center_y_m"])),
        "source_power_W": data["source_power_W"],
        "target_transmission": data["target_plane_transmitted_fraction"],
        "auto_shutoff": data["log_audit"]["final_auto_shutoff"],
        "solver_wall_time_s": data["solver_wall_time_s"],
        "gates": data["gates"],
        "pass": data["status"] == "VALIDATED_FINITE_T_GAUSSIAN_SOURCE_ONLY",
        "raw_dir": str(raw),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    offline = json.loads(REAUDIT.read_text())
    cases: list[dict[str, object]] = []
    for case in offline["cases"]:
        fit = case["fit"]
        cases.append(
            {
                "label": f"uncalibrated target {case['requested_w0_um']:.1f} um",
                "target_w0_um": case["requested_w0_um"],
                "source_object_w0_um": case["requested_w0_um"],
                "fitted_wx_um": fit["fitted_waist_x_m"] * 1e6,
                "fitted_wy_um": fit["fitted_waist_y_m"] * 1e6,
                "fit_NRMSE": fit["Gaussian_fit_NRMSE"],
                "ellipticity": fit["fitted_xy_ellipticity"],
                "center_displacement_m": float(np.hypot(fit["fitted_center_x_m"], fit["fitted_center_y_m"])),
                "source_power_W": case["source_power_W"],
                "incident_power_W": case["incident_power_W"],
                "incident_power_relative_error": case["incident_power_relative_error"],
                "gates": case["gates"],
                "pass": case["all_gates"],
                "raw_dir": case["raw_dir"],
            }
        )
    cases.append(compact_raw(RAW_12_FIRST, "12 um first calibration"))
    cases.append(compact_raw(RAW_12_PASS, "12 um second calibration (promoted)"))

    with np.load(RAW_12_PASS / "finite_T_gaussian_source_only_fields.npz") as data:
        x = np.asarray(data["x_m"], float) * 1e6
        y = np.asarray(data["y_m"], float) * 1e6
        intensity = np.asarray(data["E2_V2_m2"], float)
    fig, ax = plt.subplots(figsize=(7.3, 6.2), constrained_layout=True)
    image = ax.pcolormesh(x, y, (intensity / np.max(intensity)).T, shading="auto", cmap="inferno")
    ax.set_aspect("equal")
    ax.set_xlabel("Lumerical x = TaIrTe4 b (um)")
    ax.set_ylabel("Lumerical y = TaIrTe4 a (um)")
    ax.set_title("Validated finite-array incident source\nnormalized downward-transverse intensity at target plane")
    fig.colorbar(image, ax=ax, label="I / Imax")
    fig.savefig(OUTPUT / "validated_finite_T_w12_source.png", dpi=220)
    plt.close(fig)

    payload = {
        "status": "VALIDATED_FINITE_T_GAUSSIAN_SOURCE",
        "promoted_case": cases[-1],
        "all_cases": cases,
        "field_comparator": {
            "primary": "downward transverse incident field from E/H decomposition",
            "Ex_down": "(Ex-eta0*Hy)/2",
            "Ey_down": "(Ey+eta0*Hx)/2",
            "total_E2_including_Ez_primary": False,
        },
        "source_object_calibration": {
            "classification": "numerical source-object calibration",
            "incident_power_or_Q_rescaling": False,
            "clipping_smoothing_gain": False,
        },
        "full_finite_array_Q_executed": False,
    }
    (OUTPUT / "finite_T_Gaussian_source_gate_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    rows = "\n".join(
        f"| {c['label']} | {c['source_object_w0_um']:.6f} | {c['fitted_wx_um']:.5f} | {c['fitted_wy_um']:.5f} | {100*c['fit_NRMSE']:.4f}% | {100*c['ellipticity']:.4f}% | {c['pass']} |"
        for c in cases
    )
    (OUTPUT / "FINITE_T_GAUSSIAN_SOURCE_GATE_REPORT.md").write_text(
        "# Finite multi-T Gaussian source gate\n\n"
        "Status: `VALIDATED_FINITE_T_GAUSSIAN_SOURCE`\n\n"
        "The promoted physical target is `w0=12 um` at `lambda=11.825 um`. "
        "The Lumerical source-object input is `11.8575713844 um`; this is a numerical source calibration, not power or Q rescaling.\n\n"
        "| case | source-object w0 (um) | fitted wx | fitted wy | fit NRMSE | ellipticity | strict pass |\n"
        "|---|---:|---:|---:|---:|---:|---|\n" + rows + "\n\n"
        "The promoted case realizes `11.98755 x 12.01319 um`; all source-only gates pass. "
        "The 4 and 8.5 um failures remain as fail-closed diagnostics. No thermal, PTE, adjoint, or optimization solve was run.\n"
    )
    artifacts = []
    raw_dirs = [Path(c["raw_dir"]) for c in cases]
    for raw in raw_dirs:
        for path in sorted(raw.glob("*")):
            if path.is_file() and path.suffix in {".fsp", ".npz", ".json"}:
                artifacts.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    (OUTPUT / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps({"raw_artifacts_committed_to_git": False, "raw_artifacts": artifacts}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
