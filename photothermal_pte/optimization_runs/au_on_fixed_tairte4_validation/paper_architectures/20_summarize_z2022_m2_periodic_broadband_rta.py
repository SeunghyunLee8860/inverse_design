#!/usr/bin/env python3
"""Summarize LH/RH and CP+/CP- periodic Z spectra."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CASES = ("LH_CP_plus", "LH_CP_minus", "RH_CP_plus", "RH_CP_minus")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, dict[str, object]] = {}
    for key in CASES:
        directory = raw_root / key
        metadata_path = directory / "Z2022_M2_periodic_broadband_rta.json"
        npz_path = directory / "Z2022_M2_periodic_broadband_rta.npz"
        metadata = json.loads(metadata_path.read_text())
        if metadata["status"] != "COMPLETED_Z2022_M2_PERIODIC_BROADBAND_RTA":
            raise RuntimeError(f"{key} failed: {metadata['status']}")
        loaded[key] = {
            "metadata": metadata,
            "arrays": dict(np.load(npz_path)),
            "metadata_path": metadata_path,
            "npz_path": npz_path,
        }
    wavelength = np.asarray(loaded[CASES[0]]["arrays"]["wavelength_m"], float)
    for key in CASES[1:]:
        if not np.allclose(wavelength, loaded[key]["arrays"]["wavelength_m"], rtol=0.0, atol=1e-15):
            raise RuntimeError(f"wavelength mismatch: {key}")
    wavelength_um = wavelength * 1e6

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, key in zip(axes.reshape(-1), CASES):
        arrays = loaded[key]["arrays"]
        for metric, style in (("R", "-"), ("T", "--"), ("A", "-")):
            ax.plot(wavelength_um, arrays[metric], lw=2.2 if metric == "A" else 1.5, ls=style, label=metric)
        ax.set_title(key)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(ncol=3)
    for ax in axes[-1, :]:
        ax.set_xlabel("wavelength (µm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of incident power")
    fig.suptitle("Reconstructed Z2022 M2 periodic 4–12 µm R/T/A")
    fig.tight_layout()
    fig.savefig(output / "01_Z2022_M2_four_case_RTA.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    metrics: dict[str, object] = {}
    for handedness, color in (("LH", "#1f78b4"), ("RH", "#e31a1c")):
        plus = np.asarray(loaded[f"{handedness}_CP_plus"]["arrays"]["A"], float)
        minus = np.asarray(loaded[f"{handedness}_CP_minus"]["arrays"]["A"], float)
        g = 2.0 * (plus - minus) / np.maximum(plus + minus, 1e-12)
        axes[0].plot(wavelength_um, plus, color=color, lw=2.0, label=f"{handedness}, CP+")
        axes[0].plot(wavelength_um, minus, color=color, lw=1.6, ls="--", label=f"{handedness}, CP-")
        axes[1].plot(wavelength_um, g, color=color, lw=2.0, label=handedness)
        index = int(np.argmax(np.abs(g)))
        metrics[handedness] = {
            "maximum_abs_g": float(abs(g[index])),
            "signed_g_at_maximum": float(g[index]),
            "wavelength_um": float(wavelength_um[index]),
            "A_CP_plus": float(plus[index]),
            "A_CP_minus": float(minus[index]),
        }
    axes[0].set_ylabel("total absorption A")
    axes[1].set_ylabel("g = 2(A$_+$-A$_-$)/(A$_+$+A$_-$)")
    axes[1].set_xlabel("wavelength (µm)")
    axes[1].axhline(0.0, color="black", lw=0.8)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
    fig.suptitle("Circular-phase selectivity; CP± are solver phase definitions, not yet LCP/RCP names")
    fig.tight_layout()
    fig.savefig(output / "02_Z2022_M2_circular_phase_selectivity.png", dpi=220)
    plt.close(fig)

    summary = {
        "status": "VALIDATED_Z2022_M2_RECONSTRUCTED_PERIODIC_BROADBAND_RTA",
        "scope": "corner-joined figure reconstruction; periodic flux spectrum only",
        "LCP_RCP_name_assignment": "not promoted; CP+ and CP- are explicit Ex/Ey phase definitions",
        "case_metrics": metrics,
    }
    (output / "Z2022_M2_PERIODIC_BROADBAND_RTA_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "Z2022_M2_periodic_broadband_spectra.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavelength_um"] + [f"{key}_{metric}" for key in CASES for metric in ("R", "T", "A")])
        for index, value in enumerate(wavelength_um):
            writer.writerow([float(value)] + [float(loaded[key]["arrays"][metric][index]) for key in CASES for metric in ("R", "T", "A")])
    manifest = {
        "raw_artifacts_committed_to_git": False,
        "cases": {
            key: {
                "metadata": {"path": str(loaded[key]["metadata_path"]), "size_bytes": loaded[key]["metadata_path"].stat().st_size, "sha256": sha256(loaded[key]["metadata_path"])},
                "spectrum": {"path": str(loaded[key]["npz_path"]), "size_bytes": loaded[key]["npz_path"].stat().st_size, "sha256": sha256(loaded[key]["npz_path"])},
                "solver_artifacts": loaded[key]["metadata"].get("raw_artifacts", []),
            }
            for key in CASES
        },
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Reconstructed Z2022 M2 periodic broadband R/T/A",
        "",
        "Status: `VALIDATED_Z2022_M2_RECONSTRUCTED_PERIODIC_BROADBAND_RTA`",
        "",
        "The scalar M2 dimensions are published. The corner-joined placement is a documented figure reconstruction, not author CAD. CP+ and CP- retain their explicit solver phase definitions until the -z propagation/time convention is audited; they are not silently renamed LCP/RCP.",
        "",
        "No volumetric Q, temperature, PTE, adjoint, or optimization is included in this spectrum.",
        "",
    ]
    for handedness in ("LH", "RH"):
        item = metrics[handedness]
        lines.append(f"- {handedness}: max |g| = `{item['maximum_abs_g']:.6f}` at `{item['wavelength_um']:.6f} µm`.")
    (output / "Z2022_M2_PERIODIC_BROADBAND_RTA_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
