#!/usr/bin/env python3
"""Publish plots and a fail-closed report for Stage 52."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    result = args.result_dir.resolve()
    summary_path = result / "au_thermal_material_interface_adfd_summary.json"
    manifest_path = result / "RAW_ARTIFACT_MANIFEST.json"
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    raw_path = Path(manifest["raw_artifact"]["path"])
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    raw = np.load(raw_path)
    rows = np.genfromtxt(
        result / "au_thermal_material_interface_adfd_cases.csv",
        delimiter=",", names=True, dtype=None, encoding="utf-8",
    )

    names = list(summary["scenarios"])
    short = ["1 MW", "17.24 MW\nAu/MoS2 analogue", "100 MW", "perfect"]
    x_um = (np.arange(20) + 0.5 - 10) * 0.5
    extent = [x_um[0] - 0.25, x_um[-1] + 0.25, x_um[0] - 0.25, x_um[-1] + 0.25]
    fig, axes = plt.subplots(4, 3, figsize=(13.2, 15.2), constrained_layout=True)
    for index, (name, label) in enumerate(zip(names, short)):
        ta = raw[f"T_Ta_{name}"].T
        top = raw[f"T_top_{name}"].T
        gradient = raw[f"gradient_{name}"].T
        im = axes[index, 0].imshow(ta, origin="lower", extent=extent, cmap="inferno")
        fig.colorbar(im, ax=axes[index, 0], label="K")
        axes[index, 0].set_title(f"{label}: TaIrTe4 ΔT")
        im = axes[index, 1].imshow(top, origin="lower", extent=extent, cmap="inferno")
        fig.colorbar(im, ax=axes[index, 1], label="K")
        axes[index, 1].set_title(f"{label}: Au/air-layer ΔT")
        limit = float(np.max(np.abs(gradient)))
        im = axes[index, 2].imshow(
            gradient, origin="lower", extent=extent, cmap="coolwarm", vmin=-limit, vmax=limit
        )
        fig.colorbar(im, ax=axes[index, 2], label="K per rho")
        axes[index, 2].set_title(f"{label}: thermal adjoint gradient")
        for axis in axes[index]:
            axis.set_xlabel("Lumerical x=b (µm)")
            axis.set_ylabel("Lumerical y=a (µm)")
    fig.suptitle("Fixed-Q Au thermal-material/interface scenarios (not experimental predictions)")
    field_plot = result / "au_thermal_material_interface_fields.png"
    fig.savefig(field_plot, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
    for name, label in zip(names, short):
        selected = rows[rows["scenario"] == name]
        axes[0].scatter(selected["FD_K_per_rho"], selected["AD_K_per_rho"], s=18, label=label)
        for direction in np.unique(selected["direction"]):
            directional = selected[selected["direction"] == direction]
            order = np.argsort(directional["step"])[::-1]
            axes[1].loglog(
                directional["step"][order], directional["relative_error"][order], "o-", alpha=0.72
            )
    values = np.concatenate((rows["FD_K_per_rho"], rows["AD_K_per_rho"]))
    axes[0].plot([values.min(), values.max()], [values.min(), values.max()], "k--", label="ideal AD=FD")
    axes[0].set_xlabel("central-FD directional derivative (K)")
    axes[0].set_ylabel("adjoint directional derivative (K)")
    axes[0].set_title("All 4 G scenarios × 6 directions × 3 steps")
    axes[0].legend(fontsize=7)
    axes[1].axhline(0.01, color="black", ls="--", label="1% gate")
    axes[1].set_xlabel("FD step h")
    axes[1].set_ylabel("relative AD–FD error")
    axes[1].set_title("Central-step convergence")
    axes[1].legend()

    tmax = [summary["scenarios"][name]["Tmax_Ta_K"] for name in names]
    grad = [summary["scenarios"][name]["gradient_l2_K_per_rho"] for name in names]
    indices = np.arange(len(names))
    axes[2].plot(indices, tmax, "o-", color="#2878B5", label="TaIrTe4 Tmax")
    twin = axes[2].twinx()
    twin.plot(indices, grad, "s-", color="#D95319", label="gradient L2")
    axes[2].set_xticks(indices, short)
    axes[2].set_ylabel("Tmax (K)", color="#2878B5")
    twin.set_ylabel("gradient L2 (K per rho)", color="#D95319")
    axes[2].set_title("Physical-parameter sensitivity ≠ numerical error")
    handles = axes[2].get_lines() + twin.get_lines()
    axes[2].legend(handles, [line.get_label() for line in handles], fontsize=8)
    validation_plot = result / "au_thermal_material_interface_adfd_validation.png"
    fig.savefig(validation_plot, dpi=180)
    plt.close(fig)

    gates = summary["gates"]
    scenario_lines = []
    for name, label in zip(names, short):
        entry = summary["scenarios"][name]
        series = entry["interface_series_control"]
        g_text = "∞" if entry["G_Au_Ta_W_m2K"] == "inf" else f"{float(entry['G_Au_Ta_W_m2K']):.6g}"
        scenario_lines.append(
            f"| {label.replace(chr(10), ' ')} | {g_text} | {entry['Tmax_Ta_K']:.9g} | "
            f"{entry['gradient_l2_K_per_rho']:.9g} | {entry['worst_fine_step_ADFD_relative_error']:.3e} | "
            f"{series['analytic_interface_jump_K']:.6g} | {series['relative_jump_error']:.3e} |"
        )

    report = f"""# Au-on-fixed-TaIrTe4 thermal material/interface AD–FD control

Status: **{summary['status']}**

## What this checkpoint proves

This is a fixed-heat-source, GPU sparse-FVM control.  It validates the exact
discrete derivative of the Au/air thermal layer and the TaIrTe4-to-Au/air
contact-area relaxation.  It does **not** contain a Maxwell solve, PTE current,
electrical shunting, Au thermopower, or topology optimization.

The design is 20 x 20 physical 500-nm pixels over 10 x 10 um.  A fixed 100-nm
TaIrTe4 sheet (`x=b`, `y=a`, `z=c`) is covered by a 50-nm Au/air design layer.
The bottom is the paper-reduced thermally-grown-SiO2 Robin boundary and the
top is an ambient Robin boundary.  Lateral faces are adiabatic **for this
operator control only**, not as a promoted production boundary.

## Gray material and contact law

The physical density is interpreted as parallel contact-area fraction:

```text
g_face(rho) = A [(1-rho)/R_Ta-air + rho/R_Ta-Au]
```

The Au/air-layer conductivity is `k_air + rho (k_Au-k_air)`, and all harmonic
half-cell face resistances are differentiated.  No clipping, smoothing, gain,
or gradient rescaling was used.

`k_Au=317 W/(m K)` is a bulk reference scenario, not a certified 50-nm-film
value.  No direct Au/TaIrTe4 thermal-boundary-conductance measurement was
identified.  The 17.24 MW/(m2 K) case is derived from the *calculated*
Au/MoS2 resistance `5.8e-8 m2 K/W` in [Mao et al.](https://arxiv.org/abs/1407.2335),
and is explicitly **not** TaIrTe4 data.  The 1 and 100 MW/(m2 K) cases are
numerical sensitivity scenarios, not a confidence interval.

| scenario | G (W/m2K) | Ta Tmax (K) | ||dF/drho||2 | worst h=0.0025 AD–FD | analytic q/G jump (K) | series-control error |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(scenario_lines)}

## Numerical gates

- worst fine-step AD–FD relative error: `{gates['worst_h0p0025_ADFD_relative_error']:.6e}` (< 1%)
- worst explicit linear residual: `{gates['worst_linear_residual']:.6e}` (< 1e-8)
- worst energy-balance error: `{gates['worst_energy_balance']:.6e}` (< 1%)
- CPU linear-solve fallback: `{gates['CPU_linear_solve_fallback']}`
- fixed source power: `{summary['fixed_Q']['power_W']:.12e} W`

The variation of Tmax and gradient norm across G scenarios is physical-model
sensitivity; the AD–FD error is numerical derivative error.  They are reported
separately.

## Remaining blockers before Au PTE inverse design

1. The currently validated FDTDX optical checkpoint contains Au/TaIrTe4/air,
   but not the SiO2/Si optical substrate.  The optical geometry must be made
   identical to the thermal stack and re-close before coupling.
2. A coupled TaIrTe4 + floating Au electrical operator must validate how Au
   conductivity and Au/TaIrTe4 electrical contact alter the weighting field.
3. The spatial Maxwell sensitivity must be contracted with the thermal/PTE
   adjoint, then combined physical-density and latent AD–FD must pass.
4. Au/TaIrTe4 thermal and electrical contact values remain scenario parameters
   unless device-specific measurements are supplied.

Raw NPZ is not committed to Git.  Its absolute path, byte size, SHA-256, and
generation command are recorded in `RAW_ARTIFACT_MANIFEST.json`.
"""
    report_path = result / "AU_THERMAL_MATERIAL_INTERFACE_ADFD_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    published_paths = (
        summary_path,
        result / "au_thermal_material_interface_adfd_cases.csv",
        field_plot,
        validation_plot,
        report_path,
    )
    manifest["published"] = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in published_paths
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    return 0 if summary["status"].startswith("VALIDATED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
