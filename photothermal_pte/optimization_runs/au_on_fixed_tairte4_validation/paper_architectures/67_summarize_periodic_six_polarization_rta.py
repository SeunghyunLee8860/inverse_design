#!/usr/bin/env python3
"""Summarize paired T/Z six-polarization periodic optical R/T/A spectra."""

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
RAW = Path("/home/seunghyun/tairte4/raw_artifacts/periodic_T_Z_six_polarization_20260822")
OUTPUT = HERE / "results_periodic_T_Z_six_polarization_optical"
POLS = (
    "x_b", "y_a", "linear_plus_45", "linear_minus_45", "CP_plus", "CP_minus"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_case(architecture: str, polarization: str) -> dict[str, object]:
    directory = RAW / architecture / polarization
    stem = "T2024_periodic_broadband_rta" if architecture == "T" else "Z2022_M2_periodic_broadband_rta"
    metadata_path = directory / f"{stem}.json"
    npz_path = directory / f"{stem}.npz"
    metadata = json.loads(metadata_path.read_text())
    expected = (
        "COMPLETED_T2024_PERIODIC_BROADBAND_RTA"
        if architecture == "T"
        else "COMPLETED_Z2022_M2_PERIODIC_BROADBAND_RTA"
    )
    if metadata.get("status") != expected:
        raise RuntimeError(f"incomplete case {architecture}/{polarization}: {metadata.get('status')}")
    with np.load(npz_path) as raw:
        arrays = {key: np.asarray(raw[key], float) for key in ("wavelength_m", "R", "T", "A")}
    return {
        "metadata": metadata,
        "arrays": arrays,
        "metadata_path": metadata_path,
        "npz_path": npz_path,
        "fsp_path": directory / f"{stem}.fsp",
    }


def main() -> int:
    cases = {
        architecture: {pol: load_case(architecture, pol) for pol in POLS}
        for architecture in ("T", "Z")
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    with (OUTPUT / "periodic_T_Z_six_polarization_spectra.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("architecture", "polarization", "wavelength_um", "R", "T", "A"),
        )
        writer.writeheader()
        for architecture in ("T", "Z"):
            fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True, sharey=True, constrained_layout=True)
            for ax, polarization in zip(axes.flat, POLS):
                arrays = cases[architecture][polarization]["arrays"]
                wavelength_um = arrays["wavelength_m"] * 1e6
                for key, color in (("R", "#2166ac"), ("T", "#1b9e77"), ("A", "#d73027")):
                    ax.plot(wavelength_um, arrays[key], label=key, lw=1.6, color=color)
                peak = int(np.argmax(arrays["A"]))
                summaries.append(
                    {
                        "architecture": architecture,
                        "polarization": polarization,
                        "peak_A": float(arrays["A"][peak]),
                        "peak_wavelength_um": float(wavelength_um[peak]),
                        "min_A": float(np.min(arrays["A"])),
                        "solver_wall_time_s": cases[architecture][polarization]["metadata"].get("solver_wall_time_s"),
                    }
                )
                ax.axvline(wavelength_um[peak], color="0.45", ls="--", lw=0.8)
                ax.set_title(f"{architecture}: {polarization}; peak A={arrays['A'][peak]:.3f} @ {wavelength_um[peak]:.3f} um")
                ax.grid(alpha=0.2)
                for index in range(wavelength_um.size):
                    writer.writerow(
                        {
                            "architecture": architecture,
                            "polarization": polarization,
                            "wavelength_um": wavelength_um[index],
                            "R": arrays["R"][index],
                            "T": arrays["T"][index],
                            "A": arrays["A"][index],
                        }
                    )
            axes[0, 0].legend(ncol=3)
            for ax in axes[-1, :]:
                ax.set_xlabel("wavelength (um)")
            for ax in axes[:, 0]:
                ax.set_ylabel("power fraction")
            fig.suptitle(
                f"{architecture} periodic optical R/T/A — six incident polarization states\n"
                "normal incidence; no thermal, weighting field, or PTE"
            )
            fig.savefig(OUTPUT / f"{architecture}_six_polarization_RTA.png", dpi=220)
            plt.close(fig)

    manifest: list[dict[str, object]] = []
    for architecture in ("T", "Z"):
        for polarization in POLS:
            for key in ("metadata_path", "npz_path", "fsp_path"):
                path = cases[architecture][polarization][key]
                if path.is_file():
                    manifest.append(
                        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                    )
    payload = {
        "status": "VALIDATED_PERIODIC_T_Z_SIX_POLARIZATION_RTA_SCREENING",
        "scope": "periodic optical R/T/A only",
        "thermal_executed": False,
        "selected_volumetric_Q_status": "NOT_YET_RUN_AFTER_RTA_RESONANCE_SELECTION",
        "cases": summaries,
    }
    (OUTPUT / "PERIODIC_T_Z_SIX_POLARIZATION_RTA_SUMMARY.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    (OUTPUT / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Periodic T/Z six-polarization optical screening",
        "",
        "Status: `VALIDATED_PERIODIC_T_Z_SIX_POLARIZATION_RTA_SCREENING`",
        "",
        "This certificate contains periodic normal-incidence optical R/T/A only. "
        "No periodic temperature, weighting field, PTE current, adjoint, or optimization was run.",
        "",
        "The Z `centered_expanded_supercell_v4` is a project 5.1 x 5.1 um lattice "
        "that centers the figure-constrained M2 bars; it is not the paper M2 5.1 x 2.6 um lattice.",
        "",
        "## Peak absorptance",
        "",
        "| architecture | polarization | peak A | wavelength (um) |",
        "|---|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['architecture']} | {item['polarization']} | {item['peak_A']:.6f} | {item['peak_wavelength_um']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Selected-wavelength volumetric Q is a subsequent gate: each chosen Q must close "
            "against the corresponding flux absorption before any finite thermal solve.",
        ]
    )
    (OUTPUT / "PERIODIC_T_Z_SIX_POLARIZATION_RTA_REPORT.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
