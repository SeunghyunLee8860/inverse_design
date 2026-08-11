#!/usr/bin/env python3
"""Publish the Run-002 literal material-intersection Q audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = HERE / "plots"
MANIFEST = HERE / "manifests" / "RAW_ARTIFACT_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-directory", required=True, type=Path)
    args = parser.parse_args()
    raw = args.raw_directory.expanduser().resolve()
    raw_json = raw / "production_material_q_attribution.json"
    result = json.loads(raw_json.read_text())
    if not result.get("passed", False):
        raise RuntimeError("material-intersection attribution did not pass")
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary = RESULTS / "production_material_q_attribution_summary.json"
    report = RESULTS / "PRODUCTION_MATERIAL_Q_ATTRIBUTION_REPORT.md"
    plot = PLOTS / "production_material_q_attribution.png"
    summary.write_text(json.dumps(result, indent=2) + "\n")

    names = [
        "Si",
        "bottom_SiO2",
        "physical_TaIrTe4",
        "design_effective_SiO2",
        "artificial_extended_TaIrTe4_outside_physical_flake",
        "air_or_nonmaterial_intersection",
    ]
    labels = ["Si", "bottom SiO₂", "physical TaIrTe₄", "design SiO₂", "artificial\nbackground", "air/cut-cell\nremainder"]
    components = list("xyz")
    values = np.asarray(
        [
            [result["component_records"][c]["material_intersection_power_W"][name] for name in names]
            for c in components
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    bottoms = np.zeros(3)
    for column, label in enumerate(labels):
        axes[0].bar(components, values[:, column], bottom=bottoms, label=label)
        bottoms += values[:, column]
    axes[0].set(yscale="log", ylabel="power (W)", title="Literal material partition by Yee component")
    axes[0].legend(fontsize=8)
    fractions = np.asarray([result["power_W"][name] / result["power_W"]["forward_P_Q"] * 100 for name in names])
    axes[1].bar(np.arange(len(names)), fractions)
    axes[1].set_xticks(np.arange(len(names)), labels, rotation=25, ha="right")
    axes[1].set(ylabel="fraction of full P_Q (%)", title="No-relocation power attribution")
    fig.suptitle("Run 002 native Q → literal physical-material intersection")
    fig.savefig(plot, dpi=180)
    plt.close(fig)

    power = result["power_W"]
    relative = result["relative"]
    rows = "\n".join(
        f"| {label.replace(chr(10), ' ')} | {power[name]:.12e} | {100*power[name]/power['forward_P_Q']:.6f}% |"
        for name, label in zip(names, labels)
    )
    report.write_text(
        f"""# Production material-intersection Q attribution

Status: `{result['status']}`

The native component Yee-cell heat source was integrated only over the
literal volume shared with each physical thermal material. A cut-cell's full
power was **not** forced into TaIrTe4 or another nearest material. No clipping,
smoothing, gain, or global rescaling was used.

| partition | power (W) | fraction of full P_Q |
|:--|--:|--:|
{rows}

- full native P_Q: `{power['forward_P_Q']:.12e} W`
- native reintegrated P_Q: `{power['native_reintegrated']:.12e} W`
- material-attributed physical thermal source: `{power['physical_thermal_source']:.12e} W`
  (`{100*relative['physical_thermal_source_fraction_of_full_P_Q']:.6f}%`)
- reintegration error: `{relative['native_reintegration_error']:.6e}`
- partition identity error: `{relative['partition_identity_error']:.6e}`

The artificial long-TaIrTe4 optical background outside the finite 32×32 µm
thermal flake contributes only
`{100*relative['artificial_background_fraction_of_full_P_Q']:.6f}%` of full
P_Q and is explicitly excluded from the physical thermal RHS. The
`{100*relative['air_or_nonmaterial_fraction_of_full_P_Q']:.6f}%` air/cut-cell
remainder is reported rather than reassigned. Thus the thermal source is not
globally power-matched to the full optical control-volume P_Q.

This is an attribution gate only. It performs zero thermal, PTE, adjoint, or
optimization solves.
"""
    )
    manifest = json.loads(MANIFEST.read_text())
    manifest["production_material_intersection_q_attribution"] = {
        "status": result["status"],
        "raw_directory": str(raw),
        "artifacts": [
            {
                "path": str(raw_json),
                "size_bytes": raw_json.stat().st_size,
                "sha256": sha256(raw_json),
            }
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
