#!/usr/bin/env python3
"""Publish plots/report for floating-Au weighting/electrical Stage 54."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    summary_path = result / "au_weighting_electrical_adfd_summary.json"
    manifest_path = result / "RAW_ARTIFACT_MANIFEST.json"
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    raw_path = Path(manifest["raw_artifact"]["path"])
    raw = np.load(raw_path)
    rows = np.genfromtxt(
        result / "au_weighting_electrical_adfd_cases.csv",
        delimiter=",", names=True, dtype=None, encoding="utf-8",
    )
    names = list(summary["scenarios"])
    labels = [r"$\rho_c=10^{-8}$", r"$10^{-10}$", r"$10^{-12}$", r"$10^{-14}$ Ωm²"]
    fig, axes = plt.subplots(4, 4, figsize=(17, 15), constrained_layout=True)
    ta_extent = (-10, 10, -10, 10)
    au_extent = (-5, 5, -5, 5)
    for index, (name, label) in enumerate(zip(names, labels)):
        psi_ta = raw[f"psi_Ta_{name}"].T
        psi_au = raw[f"psi_Au_{name}"].T
        gradient = raw[f"gradient_{name}"].T
        ta_under = raw[f"psi_Ta_{name}"][10:30, 10:30].T
        diff = psi_au - ta_under
        im = axes[index, 0].imshow(psi_ta, origin="lower", extent=ta_extent, vmin=0, vmax=1)
        fig.colorbar(im, ax=axes[index, 0], label=r"$\psi$")
        axes[index, 0].set_title(f"{label}: TaIrTe4 weighting potential")
        im = axes[index, 1].imshow(psi_au, origin="lower", extent=au_extent, vmin=0, vmax=1)
        fig.colorbar(im, ax=axes[index, 1], label=r"$\psi$")
        axes[index, 1].set_title("floating Au potential")
        limit = float(np.max(np.abs(diff)))
        im = axes[index, 2].imshow(
            diff, origin="lower", extent=au_extent, cmap="coolwarm", vmin=-limit, vmax=limit
        )
        fig.colorbar(im, ax=axes[index, 2], label=r"$\psi_{Au}-\psi_{Ta}$")
        axes[index, 2].set_title("vertical contact drop")
        limit = float(np.max(np.abs(gradient)))
        im = axes[index, 3].imshow(
            gradient, origin="lower", extent=au_extent, cmap="coolwarm", vmin=-limit, vmax=limit
        )
        fig.colorbar(im, ax=axes[index, 3], label="A per rho")
        axes[index, 3].set_title("electrical adjoint gradient")
        for axis in axes[index]:
            axis.set_xlabel("Lumerical x=b (µm)")
            axis.set_ylabel("Lumerical y=a (µm)")
    fig.suptitle("Floating Au changes the TaIrTe4 weighting solution (fixed-temperature control)")
    fields_plot = result / "au_weighting_electrical_fields.png"
    fig.savefig(fields_plot, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
    for name, label in zip(names, labels):
        selected = rows[rows["scenario"] == name]
        axes[0].scatter(selected["FD_A_per_rho"], selected["AD_A_per_rho"], s=18, label=label)
        for direction in np.unique(selected["direction"]):
            directional = selected[selected["direction"] == direction]
            order = np.argsort(directional["step"])[::-1]
            axes[1].loglog(
                directional["step"][order], directional["relative_error"][order], "o-", alpha=0.7
            )
    values = np.concatenate((rows["FD_A_per_rho"], rows["AD_A_per_rho"]))
    axes[0].plot([values.min(), values.max()], [values.min(), values.max()], "k--", label="ideal AD=FD")
    axes[0].set_xlabel("central-FD directional derivative (A)")
    axes[0].set_ylabel("adjoint directional derivative (A)")
    axes[0].set_title("4 contacts × 6 directions × 3 steps")
    axes[0].legend(fontsize=7)
    axes[1].axhline(0.01, color="black", ls="--", label="1% gate")
    axes[1].set_xlabel("FD step h")
    axes[1].set_ylabel("relative AD–FD error")
    axes[1].set_title("Electrical-adjoint convergence")
    axes[1].legend()
    currents = [1e9 * summary["scenarios"][name]["base_current_A_for_fixed_unit_temperature_field"] for name in names]
    gradients = [1e9 * summary["scenarios"][name]["gradient_l2_A_per_rho"] for name in names]
    indices = np.arange(4)
    axes[2].plot(indices, currents, "o-", label="fixed-T current")
    twin = axes[2].twinx()
    twin.plot(indices, gradients, "s-", color="#D95319", label="gradient L2")
    axes[2].set_xticks(indices, labels)
    axes[2].set_ylabel("current functional (nA)")
    twin.set_ylabel("gradient L2 (nA per rho)", color="#D95319")
    axes[2].set_title("Unknown contact is a physical uncertainty")
    handles = axes[2].get_lines() + twin.get_lines()
    axes[2].legend(handles, [item.get_label() for item in handles], fontsize=8)
    validation_plot = result / "au_weighting_electrical_adfd_validation.png"
    fig.savefig(validation_plot, dpi=180)
    plt.close(fig)

    scenario_lines = []
    for name, label in zip(names, labels):
        item = summary["scenarios"][name]
        scenario_lines.append(
            f"| {label} | {item['contact_conductance_S_m2']:.3e} | "
            f"{1e9*item['base_current_A_for_fixed_unit_temperature_field']:.6g} | "
            f"{1e9*item['gradient_l2_A_per_rho']:.6g} | "
            f"{item['worst_fine_step_ADFD_relative_error']:.3e} | "
            f"{item['audit']['terminal_current_balance']:.3e} |"
        )
    gates = summary["gates"]
    report = f"""# Floating-Au weighting/electrical AD–FD control

Status: **{summary['status']}**

## Physical question

The Au pattern is a floating optical nanostructure, not a measurement
electrode.  Nevertheless, direct Au/TaIrTe4 electrical contact creates a
parallel conducting sheet.  It changes current crowding and the TaIrTe4
weighting solution even when the Au Seebeck coefficient is set to zero.

The high/low terminals are applied only to fixed TaIrTe4 (`y=a` max/min).  The
Au potential is solved as a floating unknown.  The objective uses a fixed
asymmetric TaIrTe4 temperature field and the paper TaIrTe4 conductivity and
Seebeck tensors.  Thus this checkpoint isolates the electrical contribution
to `dI/drho`; it is not a coupled Maxwell/thermal PTE prediction.

## Material and gray laws

- `sigma_Ta(x=b,y=a) = (1.10e5, 4.91e5) S/m`
- `S_Ta(x=b,y=a) = (27, -6) uV/K`
- `sigma_Au = {summary['materials']['sigma_Au_bulk_reference_S_m']:.9g} S/m`
  (bulk reference, not certified 50-nm film transport)
- `S_Au=0` in this isolation control
- Au sheet: `sigma_floor + rho (sigma_Au-sigma_floor)`
- vertical contact: `A (G_floor + rho G_contact)`

The fixed-shape numerical floors are reported in JSON and are not interpreted
as physical air conduction.

No device-specific Au/TaIrTe4 electrical contact resistivity was identified.
The four values below are numerical scenarios, not a confidence interval.

| contact resistivity | G contact (S/m2) | fixed-T current (nA) | ||dI/drho|| (nA) | worst h=0.0025 AD–FD | terminal imbalance |
|---|---:|---:|---:|---:|---:|
{chr(10).join(scenario_lines)}

The large change in both current and gradient proves that a directly touching
Au nanoantenna cannot automatically be treated as optically active but
electrically invisible.  Conversely, the numerical sweep does not identify
which contact value belongs to the fabricated device.

## Numerical gates

- worst fine-step AD–FD error: `{gates['worst_h0p0025_ADFD_relative_error']:.6e}` (< 1%)
- worst linear residual: `{gates['worst_linear_residual']:.6e}` (< 1e-8)
- worst terminal-current imbalance: `{gates['worst_terminal_current_balance']:.6e}` (< 1e-8)
- CPU linear-solve fallback: `{gates['CPU_linear_solve_fallback']}`

## Next coupled gate

The optical substrate must first be added to the FDTDX Au/TaIrTe4 model.
After that, `Q_Au + Q_Ta` is conservatively mapped into the validated thermal
operator, and the resulting temperature is passed to this electrical
operator.  The combined gradient is then the sum of Maxwell-Q, thermal
material/contact, and electrical weighting/contact terms.  Nonzero Au
thermopower remains a separate sensitivity case.

Raw NPZ is outside Git; path, size and SHA-256 are in the manifest.
"""
    report_path = result / "AU_WEIGHTING_ELECTRICAL_ADFD_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    published = (
        summary_path,
        result / "au_weighting_electrical_adfd_cases.csv",
        fields_plot,
        validation_plot,
        report_path,
    )
    manifest["published"] = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in published
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(report_path)
    return 0 if summary["status"].startswith("VALIDATED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
