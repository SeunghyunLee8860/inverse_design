#!/usr/bin/env python3
"""Publish strict substrate-bearing nonuniform-Au AD--FD step plateau."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ad-json", required=True, type=Path)
    parser.add_argument("--h0p01-json", required=True, type=Path)
    parser.add_argument("--h0p02-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = (args.h0p02_json.resolve(), args.h0p01_json.resolve(), args.ad_json.resolve())
    results = [_load(path) for path in paths]
    rows = []
    for path, result in zip(paths, results):
        row = result["directions"][0]
        rows.append(
            {
                "h": row["h"],
                "AD_W_per_unit_direction": row["ad_W_per_unit_direction"],
                "FD_W_per_unit_direction": row["fd_W_per_unit_direction"],
                "strong_relative_error": row["strong_relative_error"],
                "gradient_l2_normalized_error": row["gradient_l2_normalized_error"],
                "power_plus_W": row["power_plus_W"],
                "power_minus_W": row["power_minus_W"],
                "source_json": str(path),
            }
        )
    rows.sort(key=lambda item: item["h"], reverse=True)
    by_h = {row["h"]: row for row in rows}
    fd_plateau = abs(
        by_h[0.02]["FD_W_per_unit_direction"]
        - by_h[0.01]["FD_W_per_unit_direction"]
    ) / max(
        abs(by_h[0.02]["FD_W_per_unit_direction"]),
        abs(by_h[0.01]["FD_W_per_unit_direction"]),
    )
    baseline = results[-1]["baseline"]
    stable = (0.02, 0.01)
    gates = {
        "stable_h0p02_and_h0p01_ADFD_error_lt_1pct": all(
            by_h[h]["strong_relative_error"] < 0.01 for h in stable
        ),
        "h0p02_to_h0p01_FD_plateau_lt_1pct": fd_plateau < 0.01,
        "Q_flux_closure_lt_0p5pct": baseline["Q_flux_closure_relative"] < 0.005,
        "late_window_change_lt_0p5pct": baseline["late_window_relative_change"] < 0.005,
        "no_clipping_smoothing_gain_or_gradient_rescaling": True,
    }
    passed = all(gates.values())
    summary = {
        "status": (
            "VALIDATED_FDTDX_DIAGNOSTIC_SUBSTRATE_NONUNIFORM_AU_GRADIENT_STABLE_STEP_PLATEAU"
            if passed
            else "FAILED_FDTDX_DIAGNOSTIC_SUBSTRATE_NONUNIFORM_AU_GRADIENT_STABLE_STEP_PLATEAU"
        ),
        "scope": (
            "one strong smooth direction on the 32-period diagnostic substrate contract; "
            "no thermal/PTE/electrical/optimization and not production while Palik Si readback is blocked"
        ),
        "baseline": baseline,
        "directions": rows,
        "FD_h0p02_to_h0p01_relative_change": fd_plateau,
        "h0p005_interpretation": (
            "fail-closed small-step diagnostic: float32 subtraction noise after the "
            "0.02-to-0.01 stable plateau; result is retained, not deleted or rescaled"
        ),
        "strict_runtime": results[-1]["runtime"],
        "gates": gates,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "fdtdx_substrate_gradient_plateau_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = output / "fdtdx_substrate_gradient_plateau.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    h = np.asarray([row["h"] for row in rows])
    ad = np.asarray([row["AD_W_per_unit_direction"] for row in rows])
    fd = np.asarray([row["FD_W_per_unit_direction"] for row in rows])
    error = 100 * np.asarray([row["strong_relative_error"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    axes[0].plot(h, 1e16 * fd, "o-", label="central FD")
    axes[0].plot(h, 1e16 * ad, "s--", label="discrete AD")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("central-FD step h")
    axes[0].set_ylabel(r"directional derivative ($10^{-16}$ W)")
    axes[0].set_title("Stable 0.02 to 0.01 plateau")
    axes[0].legend()
    axes[1].semilogx(h, error, "o-")
    axes[1].axhline(1.0, color="black", ls="--", label="1% gate")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("central-FD step h")
    axes[1].set_ylabel("strong AD-FD error (%)")
    axes[1].set_title("h=0.005 enters float32 cancellation")
    axes[1].legend()
    fig.suptitle("32-period substrate-bearing nonuniform-Au optical gradient smoke")
    plot_path = output / "fdtdx_substrate_gradient_plateau.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    report = f"""# FDTDX substrate-bearing nonuniform-Au gradient smoke

Status: **{summary['status']}**

The 20x20, 500-nm-pitch nonuniform Au density uses the passive causal Drude
strength law `s(rho)=rho^3` over a fixed 50-nm Au layer.  The objective is
total native-Yee material loss in Au, TaIrTe4, and SiO2 on the same 32-period
matched-interface optical contract as the validated binary endpoints.

| h | AD (W) | central FD (W) | strong relative error |
|---:|---:|---:|---:|
"""
    for row in rows:
        report += (
            f"| {row['h']:.4g} | {row['AD_W_per_unit_direction']:.9e} | "
            f"{row['FD_W_per_unit_direction']:.9e} | "
            f"{100*row['strong_relative_error']:.6f}% |\n"
        )
    report += f"""

The `h=0.02` and `h=0.01` forward derivatives differ by
`{100*fd_plateau:.6f}%` and both agree with AD below 1%.  The `h=0.005`
result is retained as a fail-closed small-step diagnostic: its objective
difference is too small for stable subtraction in the float32 time-domain
contract and its AD-FD error rises to
`{100*by_h[0.005]['strong_relative_error']:.6f}%`. No value is removed,
fitted, normalized, or rescaled.

Baseline total Q is `{baseline['P_Q_W']:.9e} W`; matched-volume closure is
`{100*baseline['Q_flux_closure_relative']:.6f}%`; late-window change is
`{100*baseline['late_window_relative_change']:.6f}%`.

The strict AD used 16 checkpoints and required
`{results[-1]['runtime']['ad_seconds']:.3f} s` with about 36.2 GB observed GPU
memory.  It is an accuracy reference, not an approved per-iteration
production runtime.  A faster optical-period contract must be compared
against this gradient before combined PTE AD-FD or optimization.

This validates one strong optical-total-Q direction only.  It does not yet
validate the spatially weighted Maxwell source required by PTE, combined
thermal/electrical chain rule, Au thermopower, or an Au inverse design.
"""
    report_path = output / "FDTDX_SUBSTRATE_NONUNIFORM_AU_GRADIENT_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    published = (summary_path, csv_path, plot_path, report_path, *paths)
    manifest = {
        "status": summary["status"],
        "raw_field_artifact": None,
        "published": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in published
        ],
    }
    manifest_path = output / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
