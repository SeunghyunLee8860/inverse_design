#!/usr/bin/env python3
"""Re-audit saved source-only fields using the downward transverse wave."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    validate_paper_ir_source_only_gpu as audit,
)


ETA0 = 376.730313668
MONITOR = "finite_T_Gaussian_target"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(fdtd, raw_dir: Path) -> dict[str, object]:
    prior = json.loads((raw_dir / "FINITE_T_GAUSSIAN_SOURCE_ONLY.json").read_text())
    w0 = float(prior["source"]["requested_w0_um"]) * 1.0e-6
    x = np.asarray(fdtd.getdata(MONITOR, "x", 1), float).reshape(-1)
    y = np.asarray(fdtd.getdata(MONITOR, "y", 1), float).reshape(-1)
    fields = {
        key: np.asarray(fdtd.getdata(MONITOR, key, 1)).squeeze()
        for key in ("Ex", "Ey", "Hx", "Hy")
    }
    ex_down = 0.5 * (fields["Ex"] - ETA0 * fields["Hy"])
    ey_down = 0.5 * (fields["Ey"] + ETA0 * fields["Hx"])
    intensity = np.abs(ex_down) ** 2 + np.abs(ey_down) ** 2
    fit = audit.fit_gaussian(x, y, intensity)
    incident_power = audit.integrate_xy(intensity / (2.0 * ETA0), x, y)
    source_power = float(prior["source_power_W"])
    gates = {
        "waist_x_within_0p5pct": abs(fit["fitted_waist_x_m"] - w0) / w0 < 0.005,
        "waist_y_within_0p5pct": abs(fit["fitted_waist_y_m"] - w0) / w0 < 0.005,
        "Gaussian_fit_NRMSE_lt_0p5pct": fit["Gaussian_fit_NRMSE"] < 0.005,
        "ellipticity_lt_0p5pct": fit["fitted_xy_ellipticity"] < 0.005,
        "center_displacement_lt_50nm": float(np.hypot(fit["fitted_center_x_m"], fit["fitted_center_y_m"])) < 50e-9,
        "incident_power_vs_sourcepower_lt_0p5pct": abs(incident_power - source_power) / source_power < 0.005,
    }
    return {
        "raw_dir": str(raw_dir),
        "requested_w0_um": w0 * 1.0e6,
        "comparator": "downward transverse incident field from E/H decomposition",
        "formula": {"Ex_down": "(Ex-eta0*Hy)/2", "Ey_down": "(Ey+eta0*Hx)/2"},
        "fit": fit,
        "incident_power_W": incident_power,
        "source_power_W": source_power,
        "incident_power_relative_error": abs(incident_power - source_power) / source_power,
        "gates": gates,
        "all_gates": all(gates.values()),
        "plot_arrays": {"x": x, "y": y, "intensity": intensity},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results_finite_T_Gaussian_source_reaudit",
    )
    parser.add_argument(
        "--raw-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_finite_T_w0_4um_source_only"),
            Path("/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_finite_T_w0_8p5um_source_only"),
        ],
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["VC_LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(audit.APPROVED_API)
    os.environ["PATH"] = f"{audit.APPROVED_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
    sys.path.insert(0, str(audit.APPROVED_API))
    import lumapi

    cases: list[dict[str, object]] = []
    for raw in args.raw_dirs:
        raw = raw.expanduser().resolve()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        try:
            fdtd.load(str(raw / "finite_T_gaussian_source_only.fsp"))
            cases.append(analyze(fdtd, raw))
        finally:
            fdtd.close()

    fig, axes = plt.subplots(1, len(cases), figsize=(7 * len(cases), 6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, case in zip(axes, cases):
        arrays = case.pop("plot_arrays")
        image = ax.pcolormesh(
            arrays["x"] * 1e6,
            arrays["y"] * 1e6,
            arrays["intensity"].T / np.max(arrays["intensity"]),
            shading="auto",
            cmap="inferno",
        )
        fit = case["fit"]
        ax.set_aspect("equal")
        ax.set_title(
            f"requested w0={case['requested_w0_um']:.1f} um\n"
            f"fit={fit['fitted_waist_x_m']*1e6:.3f} x {fit['fitted_waist_y_m']*1e6:.3f} um; "
            f"NRMSE={100*fit['Gaussian_fit_NRMSE']:.3f}%"
        )
        ax.set_xlabel("x=b (um)")
        ax.set_ylabel("y=a (um)")
        fig.colorbar(image, ax=ax, label="normalized downward transverse intensity")
    plot_path = output / "finite_T_Gaussian_downward_transverse_source_audit.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    passing = [case for case in cases if case["all_gates"]]
    payload = {
        "status": "VALIDATED_FINITE_T_GAUSSIAN_SOURCE" if passing else "BLOCKED_FINITE_T_GAUSSIAN_SOURCE_DISTORTION",
        "cases": cases,
        "promoted_w0_um": passing[0]["requested_w0_um"] if passing else None,
        "raw_FSP_NPZ_committed": False,
        "full_finite_T_Q_executed": False,
    }
    summary_path = output / "FINITE_T_GAUSSIAN_SOURCE_REAUDIT.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    report_path = output / "FINITE_T_GAUSSIAN_SOURCE_REAUDIT.md"
    rows = "\n".join(
        f"| {c['requested_w0_um']:.1f} | {c['fit']['fitted_waist_x_m']*1e6:.4f} | {c['fit']['fitted_waist_y_m']*1e6:.4f} | {100*c['fit']['Gaussian_fit_NRMSE']:.4f}% | {100*c['fit']['fitted_xy_ellipticity']:.4f}% | {c['all_gates']} |"
        for c in cases
    )
    report_path.write_text(
        "# Finite-T Gaussian source re-audit\n\n"
        f"Status: `{payload['status']}`\n\n"
        "The primary comparator is the downward transverse incident wave, not total |E|^2 including longitudinal Ez.\n\n"
        "| requested w0 (um) | fitted wx | fitted wy | NRMSE | ellipticity | pass |\n|---:|---:|---:|---:|---:|---|\n"
        + rows
        + "\n\nNo full finite-array Q, thermal, PTE, adjoint, or optimization solve was run.\n"
    )
    manifest_path = output / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "raw_artifacts_committed_to_git": False,
                "raw_artifacts": [
                    {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                    for raw in args.raw_dirs
                    for path in (
                        raw.resolve() / "finite_T_gaussian_source_only.fsp",
                        raw.resolve() / "finite_T_gaussian_source_only_fields.npz",
                        raw.resolve() / "FINITE_T_GAUSSIAN_SOURCE_ONLY.json",
                    )
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
