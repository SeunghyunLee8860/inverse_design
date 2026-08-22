#!/usr/bin/env python3
"""Publish paired corrected-Z optical-Q and periodic thermal maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "paper_z2022_m2_figure_period_corrected_ea_eb_periodic_thermal_v3"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results_Z_M2_periodic_Ea_Eb_thermal_v3"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.input_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((source / "Z2022_M2_PERIODIC_EA_EB_THERMAL.json").read_text())
    allowed = {
        "VALIDATED_Z2022_M2_PERIODIC_EA_EB_THERMAL_SCREEN",
        "DIAGNOSTIC_Z2022_M2_PERIODIC_EA_EB_THERMAL_BLOCKED_OPTICAL_CLOSURE",
    }
    if summary["status"] not in allowed:
        raise RuntimeError("thermal pair is neither validated nor an allowed diagnostic")
    with np.load(source / "Z2022_M2_PERIODIC_EA_EB_THERMAL.npz", allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    x = arrays["x_m"] * 1.0e6
    y = arrays["y_m"] * 1.0e6
    extent = [x[0], x[-1], y[0], y[-1]]
    fields = [
        ("Qxy_W_m2", "depth-integrated $Q$", "W/m²", "inferno", False),
        ("TaIrTe4_temperature_K", "$\\Delta T$ in TaIrTe$_4$", "K/(W/m²)", "magma", False),
        ("dT_db_K_m", "$\\partial_b T$", "K m⁻¹/(W m⁻²)", "coolwarm", True),
        ("dT_da_K_m", "$\\partial_a T$", "K m⁻¹/(W m⁻²)", "coolwarm", True),
        ("gradT_K_m", "$|\\nabla_{ab}T|$", "K m⁻¹/(W m⁻²)", "viridis", False),
    ]
    fig, axes = plt.subplots(2, len(fields), figsize=(22, 8), constrained_layout=True)
    for column, (suffix, title, unit, cmap, signed) in enumerate(fields):
        pair = [arrays[f"{pol}_{suffix}"] for pol in ("Ea", "Eb")]
        if signed:
            vmax = max(float(np.max(np.abs(value))) for value in pair)
            vmin = -vmax
        else:
            vmin = min(float(np.min(value)) for value in pair)
            vmax = max(float(np.max(value)) for value in pair)
        for row, (pol, value) in enumerate(zip(("Ea", "Eb"), pair)):
            image = axes[row, column].imshow(
                value.T,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axes[row, column].set_title(f"$E\\parallel {pol[-1].lower()}$: {title}")
            axes[row, column].set_xlabel("Lumerical x=b (µm)")
            axes[row, column].set_ylabel("Lumerical y=a (µm)")
            fig.colorbar(image, ax=axes[row, column], label=unit, shrink=0.82)
    fig.suptitle(
        "Corrected 2022 M2 Z cell — paired raw Maxwell Q → identical periodic thermal operator\n"
        "1 W/m² normal incidence; no polarization matching or Q rescaling",
        fontsize=15,
    )
    figure = output / "Z2022_M2_periodic_Ea_Eb_Q_temperature_gradients.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    cases = summary["cases"]
    report = f"""# Corrected Z-M2 paired optical-to-thermal screen

Status: `{summary['status']}`

Optical closure: Ea = {summary['optical_closure_relative']['Ea']:.3%},
Eb = {summary['optical_closure_relative']['Eb']:.3%}.  If either exceeds 0.5%,
the maps below are diagnostic and are not a promoted thermal certificate.

This is a paired periodic unit-cell thermal screen of the figure-period-corrected
M2 reconstruction. It is **not** a finite-device PTE-current result and it does
not contain a weighting field because a terminal pair is absent in the periodic
unit cell.

Both polarizations use the same geometry, incident intensity, conservative
Yee-to-FVM remap, material tensors, periodic lateral faces, adiabatic top and
fixed-temperature bottom. Raw polarization-dependent Q is never matched or
rescaled to the other polarization.

| metric | E parallel a | E parallel b |
|---|---:|---:|
| P_Q at 1 W/m2 incident (W) | {cases['Ea']['P_Q_W_at_1_W_m2_incident']:.8e} | {cases['Eb']['P_Q_W_at_1_W_m2_incident']:.8e} |
| TaIrTe4 Tmax (K per W/m2) | {cases['Ea']['TaIrTe4_Tmax_K_per_W_m2']:.8e} | {cases['Eb']['TaIrTe4_Tmax_K_per_W_m2']:.8e} |
| max abs dT/db | {cases['Ea']['max_abs_dT_db_K_m_per_W_m2']:.8e} | {cases['Eb']['max_abs_dT_db_K_m_per_W_m2']:.8e} |
| max abs dT/da | {cases['Ea']['max_abs_dT_da_K_m_per_W_m2']:.8e} | {cases['Eb']['max_abs_dT_da_K_m_per_W_m2']:.8e} |
| remap error | {cases['Ea']['Q_mapping_error_relative']:.3e} | {cases['Eb']['Q_mapping_error_relative']:.3e} |
| energy error | {cases['Ea']['energy_balance_relative']:.3e} | {cases['Eb']['energy_balance_relative']:.3e} |

For weighting potential and terminal current, a separate finite flake plus two
explicit electrodes must be specified. A periodic unit cell cannot supply that
quantity without inventing a device boundary condition.
"""
    (output / "Z2022_M2_PERIODIC_EA_EB_THERMAL_REPORT.md").write_text(report)
    published = dict(summary)
    published["published_figure"] = str(figure)
    (output / "Z2022_M2_PERIODIC_EA_EB_THERMAL_SUMMARY.json").write_text(
        json.dumps(published, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
