#!/usr/bin/env python3
"""Publish the eight-case finite T/Z Gaussian volumetric-Q checkpoint."""

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
RAW = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_Q")
OUTPUT = HERE / "results_finite_T_Z_gaussian_q"
VALID_CASES = {
    "T_Ea_Au_on": RAW / "T_Ea_Au_on_v2",
    "T_Eb_Au_on": RAW / "T_Eb_Au_on",
    "T_Ea_Au_off": RAW / "T_Ea_Au_off",
    "T_Eb_Au_off": RAW / "T_Eb_Au_off",
    "Z_Ea_Au_on": RAW / "Z_Ea_Au_on",
    "Z_Eb_Au_on": RAW / "Z_Eb_Au_on",
    "Z_Ea_Au_off": RAW / "Z_Ea_Au_off",
    "Z_Eb_Au_off": RAW / "Z_Eb_Au_off",
}
FAILED_DIAGNOSTIC = RAW / "T_Ea_Au_on"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(case: str, directory: Path) -> tuple[Path, dict]:
    path = next(directory.glob(f"FINITE_{case}_Q.json"))
    return path, json.loads(path.read_text())


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases: dict[str, dict] = {}
    rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for name, directory in VALID_CASES.items():
        result_path, result = _json(name, directory)
        if not str(result["status"]).startswith("VALIDATED"):
            raise RuntimeError(f"case failed: {name}: {result['status']}")
        if not all(result["gates"].values()):
            raise RuntimeError(f"case gate false: {name}")
        cases[name] = result
        powers = result["Q_component_power_native_W"]
        rows.append(
            {
                "case": name,
                "architecture": result["architecture"],
                "polarization": result["polarization"],
                "top_Au_present": result["top_Au_present"],
                "source_power_W": result["source_power_W"],
                "P_Q_W": result["P_Q_native_W"],
                "P_six_W": result["P_six_face_W"],
                "closure_percent": 100.0 * result["six_face_closure_relative"],
                "auto_shutoff": result["log_audit"]["final_auto_shutoff"],
                "Qx_W": powers["x"],
                "Qy_W": powers["y"],
                "Qz_W": powers["z"],
                "hotspot_x_um": 1e6 * result["hotspot"]["x_m"],
                "hotspot_y_um": 1e6 * result["hotspot"]["y_m"],
                "hotspot_z_nm": 1e9 * result["hotspot"]["z_m"],
                "solver_wall_time_s": result["solver_wall_time_s"],
            }
        )
        manifest.append(
            {
                "case": name,
                "result_json": str(result_path),
                "result_json_size_bytes": result_path.stat().st_size,
                "result_json_sha256": _sha256(result_path),
                "raw_artifacts": result["raw_artifacts"],
                "committed_to_git": False,
            }
        )

    comparisons: dict[str, dict[str, float]] = {}
    for architecture in ("T", "Z"):
        for polarization in ("Ea", "Eb"):
            on = float(cases[f"{architecture}_{polarization}_Au_on"]["P_Q_native_W"])
            off = float(cases[f"{architecture}_{polarization}_Au_off"]["P_Q_native_W"])
            comparisons[f"{architecture}_{polarization}"] = {
                "P_Q_Au_on_W": on,
                "P_Q_Au_off_W": off,
                "signed_Au_on_minus_off_W": on - off,
                "signed_Au_effect_relative_to_off": (on - off) / off,
            }

    csv_path = OUTPUT / "finite_T_Z_gaussian_q_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["case"] for row in rows]
    pq = np.asarray([row["P_Q_W"] for row in rows]) * 1e15
    qx = np.asarray([row["Qx_W"] for row in rows]) * 1e15
    qy = np.asarray([row["Qy_W"] for row in rows]) * 1e15
    qz = np.asarray([row["Qz_W"] for row in rows]) * 1e15
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0].bar(x, pq, color=["#d39b00" if row["top_Au_present"] else "#777" for row in rows])
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("raw P_Q (fW)")
    axes[0].set_title("Finite nonperiodic Gaussian Maxwell absorption; identical source within each architecture")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, qx, label="Qx")
    axes[1].bar(x, qy, bottom=qx, label="Qy")
    axes[1].bar(x, qz, bottom=qx + qy, label="Qz")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylabel("native-Yee component power (fW)")
    axes[1].set_title("Component-resolved absorption")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    plot = OUTPUT / "finite_T_Z_gaussian_q_power_and_components.png"
    fig.savefig(plot, dpi=190)
    plt.close(fig)

    status = "VALIDATED_FINITE_T_Z_AU_ON_OFF_EA_EB_VOLUMETRIC_Q"
    summary = {
        "status": status,
        "classification": (
            "finite nonperiodic v261 Gaussian Maxwell/Q checkpoint; raw source "
            "power; no thermal/electrical/PTE/adjoint/optimization"
        ),
        "axes": {"x": "b", "y": "a", "z": "c=b optical closure"},
        "cases": {name: result["status"] for name, result in cases.items()},
        "Au_effect_total_absorption": comparisons,
        "maximum_six_face_closure_relative": max(
            float(result["six_face_closure_relative"]) for result in cases.values()
        ),
        "maximum_auto_shutoff": max(
            float(result["log_audit"]["final_auto_shutoff"]) for result in cases.values()
        ),
        "failed_diagnostic_preserved": str(FAILED_DIAGNOSTIC),
        "Q_processing": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "global_rescaling": False,
            "tiling": False,
        },
        "next_gate": (
            "component-native material-loss-participation/cut-cell mapping to the "
            "same explicit finite 3-D thermal operator"
        ),
    }
    summary_path = OUTPUT / "FINITE_T_Z_GAUSSIAN_Q_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    manifest_payload = {
        "status": status,
        "raw_files_committed_to_git": False,
        "valid_cases": manifest,
        "failed_diagnostic": {
            "path": str(FAILED_DIAGNOSTIC),
            "reason": "post-solve Python import failed; completed FSP retained; not used",
        },
    }
    manifest_path = OUTPUT / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n")

    lines = [
        "# Finite T/Z Gaussian Maxwell Q — Au-on/off and E∥a/E∥b",
        "",
        f"Status: **{status}**",
        "",
        "All eight finite nonperiodic v261 GPU cases passed matched-volume six-face closure, auto-shutoff, native/pabs agreement, finite-array, and nonnegative-Q gates.",
        "The scalar Gaussian source is the separately validated w0=4 µm source. Raw Q is never matched between polarizations or Au states.",
        "",
        "## Signed top-Au effect on total absorption",
        "",
        "| Architecture/polarization | Au-on P_Q (fW) | Au-off P_Q (fW) | (on-off)/off |",
        "|---|---:|---:|---:|",
    ]
    for label, value in comparisons.items():
        lines.append(
            f"| {label} | {value['P_Q_Au_on_W']*1e15:.6f} | {value['P_Q_Au_off_W']*1e15:.6f} | {100*value['signed_Au_effect_relative_to_off']:.3f}% |"
        )
    lines += [
        "",
        "This total-power comparison does not by itself identify whether Au changes top-Au, TaIrTe4, mirror, or substrate heating. That decomposition is the next material-overlap thermal gate.",
        "",
        "Raw FSP/NPZ files remain outside Git; exact paths, sizes, and SHA-256 values are in the manifest.",
    ]
    report = OUTPUT / "FINITE_T_Z_GAUSSIAN_Q_REPORT.md"
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
