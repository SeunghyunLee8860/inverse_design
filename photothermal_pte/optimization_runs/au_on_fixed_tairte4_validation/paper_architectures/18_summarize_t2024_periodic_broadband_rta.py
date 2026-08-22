#!/usr/bin/env python3
"""Publish the four-case periodic inverse-T broadband R/T/A comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CASES = ("T_Ea", "T_Eb", "bare_Ea", "bare_Eb")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_case(root: Path, key: str) -> dict[str, object]:
    directory = root / key
    metadata_path = directory / "T2024_periodic_broadband_rta.json"
    spectrum_path = directory / "T2024_periodic_broadband_rta.npz"
    metadata = json.loads(metadata_path.read_text())
    if metadata["status"] != "COMPLETED_T2024_PERIODIC_BROADBAND_RTA":
        raise RuntimeError(f"{key} did not pass: {metadata['status']}")
    arrays = dict(np.load(spectrum_path))
    return {
        "metadata": metadata,
        "arrays": arrays,
        "metadata_path": metadata_path,
        "spectrum_path": spectrum_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.raw_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = {key: load_case(root, key) for key in CASES}

    wavelength = np.asarray(cases["T_Ea"]["arrays"]["wavelength_m"], float)
    for key in CASES[1:]:
        if not np.allclose(wavelength, cases[key]["arrays"]["wavelength_m"], rtol=0.0, atol=1e-15):
            raise RuntimeError(f"wavelength grid mismatch: {key}")
    wavelength_um = wavelength * 1e6

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, key in zip(axes.reshape(-1), CASES):
        arrays = cases[key]["arrays"]
        ax.plot(wavelength_um, arrays["R"], label="R", lw=1.8)
        ax.plot(wavelength_um, arrays["T"], label="T", lw=1.8)
        ax.plot(wavelength_um, arrays["A"], label="A=1-R-T", lw=2.2)
        ax.set_title(key.replace("_", ", "))
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(ncol=3)
    for ax in axes[-1, :]:
        ax.set_xlabel("wavelength (µm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of incident power")
    fig.suptitle("Periodic inverse-T broadband optical screening — normal-incidence plane wave")
    fig.tight_layout()
    fig.savefig(output / "01_T2024_periodic_four_case_RTA.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    colors = {"Ea": "#d95f02", "Eb": "#1b9e77"}
    for pol in ("Ea", "Eb"):
        axes[0].plot(wavelength_um, cases[f"T_{pol}"]["arrays"]["A"], color=colors[pol], lw=2.3, label=f"T, E||{pol[-1].lower()}")
        axes[0].plot(wavelength_um, cases[f"bare_{pol}"]["arrays"]["A"], color=colors[pol], lw=1.4, ls="--", label=f"bare, E||{pol[-1].lower()}")
        axes[1].plot(
            wavelength_um,
            cases[f"T_{pol}"]["arrays"]["A"] - cases[f"bare_{pol}"]["arrays"]["A"],
            color=colors[pol],
            lw=2.2,
            label=f"E||{pol[-1].lower()}",
        )
    axes[0].set_ylabel("total absorption A")
    axes[1].set_ylabel("A(T) - A(bare)")
    axes[1].set_xlabel("wavelength (µm)")
    axes[1].axhline(0.0, color="black", lw=0.8)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
    fig.suptitle("What the top inverse-T adds to the matched TaIrTe4 stack")
    fig.tight_layout()
    fig.savefig(output / "02_T2024_T_vs_bare_absorption.png", dpi=220)
    plt.close(fig)

    a_t_ea = np.asarray(cases["T_Ea"]["arrays"]["A"], float)
    a_t_eb = np.asarray(cases["T_Eb"]["arrays"]["A"], float)
    selectivity = (a_t_eb - a_t_ea) / np.maximum(0.5 * (a_t_eb + a_t_ea), 1e-12)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(wavelength_um, selectivity, color="#6a3d9a", lw=2.2)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("wavelength (µm)")
    ax.set_ylabel("2(A$_b$-A$_a$)/(A$_b$+A$_a$)")
    ax.set_title("Periodic inverse-T polarization selectivity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "03_T2024_periodic_polarization_selectivity.png", dpi=220)
    plt.close(fig)

    metrics: dict[str, object] = {}
    for key in CASES:
        absorption = np.asarray(cases[key]["arrays"]["A"], float)
        index = int(np.argmax(absorption))
        metrics[key] = {
            "peak_wavelength_um": float(wavelength_um[index]),
            "peak_total_absorption": float(absorption[index]),
            "runtime_s": float(cases[key]["metadata"]["solver_wall_time_s"]),
            "final_auto_shutoff": float(cases[key]["metadata"]["log_audit"]["final_auto_shutoff"]),
        }
    enhancement: dict[str, object] = {}
    for pol in ("Ea", "Eb"):
        delta = np.asarray(cases[f"T_{pol}"]["arrays"]["A"]) - np.asarray(cases[f"bare_{pol}"]["arrays"]["A"])
        index = int(np.argmax(delta))
        enhancement[pol] = {
            "max_delta_A": float(delta[index]),
            "wavelength_um": float(wavelength_um[index]),
        }
    summary = {
        "status": "VALIDATED_T2024_PERIODIC_BROADBAND_RTA_SCREENING",
        "scope": "periodic flux-derived spectrum only; selected resonances still require single-frequency volumetric-Q closure",
        "wavelength_bounds_um": [float(wavelength_um[0]), float(wavelength_um[-1])],
        "wavelength_points": int(wavelength.size),
        "case_metrics": metrics,
        "T_minus_bare_enhancement": enhancement,
        "maximum_abs_selectivity": float(np.max(np.abs(selectivity))),
    }
    (output / "T2024_PERIODIC_BROADBAND_RTA_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (output / "T2024_periodic_broadband_spectra.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["wavelength_um"]
        for key in CASES:
            header.extend([f"{key}_R", f"{key}_T", f"{key}_A"])
        writer.writerow(header)
        for index, value in enumerate(wavelength_um):
            row: list[float] = [float(value)]
            for key in CASES:
                row.extend(float(cases[key]["arrays"][metric][index]) for metric in ("R", "T", "A"))
            writer.writerow(row)

    manifest = {
        "raw_artifacts_committed_to_git": False,
        "cases": {
            key: {
                "metadata": {"path": str(cases[key]["metadata_path"]), "size_bytes": cases[key]["metadata_path"].stat().st_size, "sha256": sha256(cases[key]["metadata_path"])},
                "spectrum": {"path": str(cases[key]["spectrum_path"]), "size_bytes": cases[key]["spectrum_path"].stat().st_size, "sha256": sha256(cases[key]["spectrum_path"])},
                "solver_artifacts": cases[key]["metadata"].get("raw_artifacts", []),
            }
            for key in CASES
        },
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    lines = [
        "# T2024 periodic broadband R/T/A screening",
        "",
        "Status: `VALIDATED_T2024_PERIODIC_BROADBAND_RTA_SCREENING`",
        "",
        "This is the infinite periodic, normal-incidence plane-wave resonance-screening problem. It is not the finite Gaussian-beam PTE device.",
        "",
        "The plotted absorption is the flux quantity `A=1-R-T`. No broadband 3-D Q monitor was retained. A selected resonance must therefore be rerun at one wavelength with component-resolved volumetric Q and six/control-volume closure before thermal use.",
        "",
        "| case | peak wavelength (µm) | peak total A | runtime (s) |",
        "|---|---:|---:|---:|",
    ]
    for key in CASES:
        item = metrics[key]
        lines.append(f"| {key} | {item['peak_wavelength_um']:.6f} | {item['peak_total_absorption']:.6f} | {item['runtime_s']:.2f} |")
    lines.extend(["", "## Interpretation", ""])
    for pol in ("Ea", "Eb"):
        item = enhancement[pol]
        lines.append(f"- {pol}: maximum signed T-minus-bare absorption enhancement is `{item['max_delta_A']:.6f}` at `{item['wavelength_um']:.6f} µm`.")
    lines.extend([
        "",
        "The next optical calculation is a single-frequency Q certificate at the physically selected resonance, followed by a finite multi-T array under a Gaussian beam.",
    ])
    (output / "T2024_PERIODIC_BROADBAND_RTA_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
