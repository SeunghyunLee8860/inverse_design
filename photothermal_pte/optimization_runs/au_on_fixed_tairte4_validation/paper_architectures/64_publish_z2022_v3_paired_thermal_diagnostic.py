#!/usr/bin/env python3
"""Publish paired Ea/Eb Q, T and gradient maps for corrected Z M2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
INPUT = Path("/home/seunghyun/tairte4/raw_artifacts/paper_z2022_m2_v3_ea_eb_poynting_divergence_thermal")
OUTPUT = HERE / "results_Z_M2_periodic_Ea_Eb_poynting_thermal_diagnostic_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source = args.input_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((source / "Z2022_M2_POYNTING_DIVERGENCE_EA_EB_THERMAL.json").read_text())
    npz_path = source / "Z2022_M2_POYNTING_DIVERGENCE_EA_EB_THERMAL.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    x = arrays["x_m"] * 1e6
    y = arrays["y_m"] * 1e6
    extent = [x[0], x[-1], y[0], y[-1]]
    fields = [
        ("Qxy_W_m2", "depth-integrated signed Q", "W/m²", "coolwarm"),
        ("TaIrTe4_temperature_K", "TaIrTe₄ thickness-avg signed ΔT", "K per (W/m² incident)", "coolwarm"),
        ("dT_db_K_m", "∂T/∂b", "K/m per (W/m² incident)", "coolwarm"),
        ("dT_da_K_m", "∂T/∂a", "K/m per (W/m² incident)", "coolwarm"),
        ("gradT_K_m", "|∇T|", "K/m per (W/m² incident)", "viridis"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), constrained_layout=True)
    for row, pol in enumerate(("Ea", "Eb")):
        for col, (suffix, title, unit, cmap) in enumerate(fields):
            value = arrays[f"{pol}_{suffix}"].T
            if suffix in {"Qxy_W_m2", "TaIrTe4_temperature_K", "dT_db_K_m", "dT_da_K_m"}:
                # Signed diagnostic fields must expose, not hide, the negative
                # interface oscillations.  Use a polarization-specific robust
                # symmetric range; exact extrema remain in JSON/NPZ.
                vmax = np.nanpercentile(np.abs(arrays[f"{pol}_{suffix}"]), 99.5)
                im = axes[row, col].imshow(value, origin="lower", extent=extent, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
            else:
                im = axes[row, col].imshow(value, origin="lower", extent=extent, cmap=cmap, aspect="equal")
            axes[row, col].set_title(f"E∥{pol[-1].lower()}: {title}")
            axes[row, col].set_xlabel("x=b (µm)")
            axes[row, col].set_ylabel("y=a (µm)")
            fig.colorbar(im, ax=axes[row, col], label=unit, shrink=0.82)
    fig.suptitle("Corrected 2022 M2 Z: paired optical→thermal diagnostic (same operator)")
    plot = output / "Z2022_M2_V3_EA_EB_Q_T_GRADIENTS.png"
    fig.savefig(plot, dpi=220)
    plt.close(fig)

    csv_path = output / "Z2022_M2_V3_EA_EB_THERMAL_CASES.csv"
    keys = sorted(next(iter(summary["cases"].values())).keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["polarization", *keys], lineterminator="\n")
        writer.writeheader()
        for pol in ("Ea", "Eb"):
            writer.writerow({"polarization": pol, **summary["cases"][pol]})
    report = output / "Z2022_M2_V3_PAIRED_OPTICAL_THERMAL_REPORT.md"
    report.write_text(
        f"""# Corrected Z M2 paired optical→thermal diagnostic

Status: `{summary['status']}`

Both `E||a` and `E||b` use the same corrected figure-period geometry,
incident intensity, conservative remap, material tensors, periodic lateral
boundaries, bottom bath, and top adiabatic boundary.  The plot contains the
depth-integrated signed heat source, TaIrTe4 thickness-averaged temperature,
`dT/db`, `dT/da`, and in-plane gradient magnitude for **both** polarizations.

This is not a promoted physical thermal certificate.  The native Lumerical
volumetric-loss monitor failed the matched-volume closure gate, while the
independent Poynting-divergence construction closes by conservation but retains
signed metal-interface oscillations.  No clipping, smoothing, gain, or
rescaling was used.  The maps are therefore a fail-closed diagnostic only.

The periodic unit cell has no terminal pair, so weighting potential and PTE
current are not defined in this result.

## Case metrics

```json
{json.dumps(summary['cases'], indent=2)}
```
"""
    )
    summary_copy = output / "Z2022_M2_V3_PAIRED_OPTICAL_THERMAL_SUMMARY.json"
    summary_copy.write_text(json.dumps(summary, indent=2) + "\n")
    files = [report, summary_copy, csv_path, plot]
    manifest = {
        "status": summary["status"],
        "files": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ],
        "raw_npz": {"path": str(npz_path), "size_bytes": npz_path.stat().st_size, "sha256": sha256(npz_path)},
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
