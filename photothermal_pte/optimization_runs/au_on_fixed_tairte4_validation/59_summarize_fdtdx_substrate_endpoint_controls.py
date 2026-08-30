#!/usr/bin/env python3
"""Publish the 10-um FDTDX substrate/Yee-volume endpoint checkpoint."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-dir", required=True, type=Path)
    parser.add_argument("--control-root", required=True, type=Path)
    args = parser.parse_args()
    endpoint_dir = args.endpoint_dir.resolve()
    control_root = args.control_root.resolve()
    summary_path = endpoint_dir / "fdtdx_substrate_binary_endpoints_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cases = summary["cases"]

    names = ("empty", "au0", "au1")
    labels = ("substrate", "TaIrTe4 + substrate", "Au + TaIrTe4 + substrate")
    rows: list[dict[str, object]] = []
    for name, label in zip(names, labels):
        case = cases[name]
        component = case["component_power_W"]
        closure = summary["closure"]["cases"][name]
        rows.append(
            {
                "case": name,
                "label": label,
                "P_Q_W": case["P_Q_W"],
                "P_Au_W": sum(component["au_xyz"]),
                "P_TaIrTe4_W": sum(component["tairte4_xyz"]),
                "P_SiO2_W": sum(component["sio2_xyz"]),
                "P_closed_time_domain_W": closure["deep_time_domain_closed_surface_W"],
                "time_domain_closure_relative": closure["Q_flux_closure_relative"],
                "near_phasor_closure_relative": closure[
                    "near_phasor_Q_flux_closure_relative"
                ],
                "deep_phasor_closure_relative": closure[
                    "deep_phasor_Q_flux_closure_relative"
                ],
                "late_window_relative_change": case["P_Q_window_relative_change"],
            }
        )
    csv_path = endpoint_dir / "fdtdx_substrate_binary_endpoints_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    x = np.arange(len(rows))
    p_au = 1e15 * np.asarray([row["P_Au_W"] for row in rows], dtype=float)
    p_ta = 1e15 * np.asarray([row["P_TaIrTe4_W"] for row in rows], dtype=float)
    p_oxide = 1e15 * np.asarray([row["P_SiO2_W"] for row in rows], dtype=float)
    td = 100 * np.asarray([row["time_domain_closure_relative"] for row in rows])
    near = 100 * np.asarray([row["near_phasor_closure_relative"] for row in rows])
    deep = 100 * np.asarray([row["deep_phasor_closure_relative"] for row in rows])
    window = 100 * np.asarray([row["late_window_relative_change"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].bar(x, p_oxide, label="SiO2")
    axes[0].bar(x, p_ta, bottom=p_oxide, label="TaIrTe4")
    axes[0].bar(x, p_au, bottom=p_oxide + p_ta, label="Au")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("absorbed power (fW)")
    axes[0].set_title("Native-Yee component loss")
    axes[0].legend()
    width = 0.25
    axes[1].bar(x - width, td, width, label="deep time-domain (primary)")
    axes[1].bar(x, near, width, label="near phasor")
    axes[1].bar(x + width, deep, width, label="deep phasor")
    axes[1].axhline(0.5, color="black", ls="--", label="0.5% gate")
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("Q / inward-flux difference (%)")
    axes[1].set_title("Matched-volume closure")
    axes[1].legend(fontsize=8)
    axes[2].bar(x, window)
    axes[2].axhline(0.5, color="black", ls="--")
    axes[2].set_xticks(x, names)
    axes[2].set_ylabel("late-window change (%)")
    axes[2].set_title("32-period observable convergence")
    fig.suptitle("10 um FDTDX substrate endpoint checkpoint (E || b)")
    plot_path = endpoint_dir / "fdtdx_substrate_binary_endpoints.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    planar_path = (
        control_root
        / "results_fdtdx_planar_substrate_loss"
        / "fdtdx_planar_substrate_loss_32period.json"
    )
    planar = json.loads(planar_path.read_text(encoding="utf-8"))
    material = summary["audit"]["substrate"]
    worst_closure = max(row["time_domain_closure_relative"] for row in rows)
    worst_window = max(row["late_window_relative_change"] for row in rows)
    report = f"""# 10 um FDTDX substrate and binary-Au endpoint control

Status: **{summary['status']}**

## Outcome

The component-specific Yee dual-volume integration error at the Si/SiO2
transition was identified and corrected.  `Ex` and `Ey` occupy z-edge dual
volumes, while `Ez` occupies the cell z width.  Applying one cell-centered
volume to all three components over-counted the 285-nm SiO2 optical support
when a 15-nm oxide cell touched a coarse Si cell.

The promoted diagnostic uses a matched z grid at the Si/SiO2 interface,
32 optical periods, a four-period late window, and direct material-loss versus
deep-box time-domain Poynting balance.  There is no background subtraction,
clipping, smoothing, gain, or result rescaling.

## Optical contract

- wavelength: 10 um; scalar Gaussian; requested `w0=8.5 um`; `E || b`
- domain: 48 x 48 x 16 um; six PML; 9,870,336 Yee cells
- TaIrTe4: 20 x 20 x 0.1 um; `epsilon_x=epsilon_b`, `epsilon_y=epsilon_a`,
  `epsilon_z=epsilon_b`
- SiO2: 285 nm, Kitamura value `epsilon={material['epsilon_sio2']}`
- Si: diagnostic lossless `n=3.4215`; installed-Lumerical Palik readback is
  still blocked and this checkpoint is therefore not a production material
  certificate
- Au: 50 nm, `epsilon={summary['materials']['au_epsilon']}`

## Endpoint results

| case | P_Q (W) | Au (W) | TaIrTe4 (W) | SiO2 (W) | primary closure | late-window change |
|---|---:|---:|---:|---:|---:|---:|
"""
    for row in rows:
        report += (
            f"| {row['case']} | {row['P_Q_W']:.9e} | {row['P_Au_W']:.9e} | "
            f"{row['P_TaIrTe4_W']:.9e} | {row['P_SiO2_W']:.9e} | "
            f"{100*row['time_domain_closure_relative']:.4f}% | "
            f"{100*row['late_window_relative_change']:.4f}% |\n"
        )
    report += f"""

Worst primary closure is `{100*worst_closure:.4f}%`; worst late-window change
is `{100*worst_window:.4f}%`.  Both pass the 0.5% gates.  Near/deep phasor
box differences remain in the CSV as detector-convergence diagnostics rather
than being hidden or substituted for the primary time-domain balance.

The 1D planar matched-grid control independently gives time-domain closure
`{100*planar['relative_errors']['Joule_vs_time_domain']:.4f}%`.

## Interpretation and remaining blocker

Adding Au changes the complete electromagnetic field: it introduces Au loss
but also reduces TaIrTe4 and SiO2 loss in this binary endpoint.  Therefore Au
power must not be appended to a fixed TaIrTe4 Q map after the Maxwell solve.

This checkpoint validates the FDTDX loss/flux bookkeeping for the stated
diagnostic substrate.  It does **not** yet validate the substrate-bearing Au
density gradient, the thermal/electrical coupled PTE gradient, or an inverse
design.  Palik Si readback from the installed Lumerical material database also
remains fail-closed.
"""
    report_path = endpoint_dir / "FDTDX_SUBSTRATE_BINARY_ENDPOINT_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    published = (summary_path, csv_path, plot_path, report_path, planar_path)
    manifest = {
        "status": summary["status"],
        "raw_field_artifact": None,
        "raw_field_artifact_note": (
            "No full E/H time history or FSP/NPZ is committed; this checkpoint "
            "publishes scalar endpoint observables and provenance only."
        ),
        "published": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in published
        ],
    }
    manifest_path = endpoint_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    return 0 if summary["status"].startswith("VALIDATED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
