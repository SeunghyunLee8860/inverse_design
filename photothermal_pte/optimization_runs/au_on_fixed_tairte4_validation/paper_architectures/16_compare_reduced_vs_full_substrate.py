#!/usr/bin/env python3
"""Certify the 285-nm optical closure against the preserved 1.5-um control."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
FULL_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_architectures_sio2_si")
REDUCED_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_architectures_sio2_si_reduced")
OUTPUT = HERE / "results_actual_metasurfaces_sio2_si_reduced"
FULL_DIRS = {
    "T_Eb": "T2024_MIR_4750_xb_forward",
    "T_Ea": "T2024_MIR_4750_ya_forward",
    "bare_Eb": "T2024_MIR_4750_bare_xb_forward",
    "bare_Ea": "T2024_MIR_4750_bare_ya_forward",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    comparison = load_module(HERE / "09_compare_t2024_tairte4_polarizations.py", "reduced_full_case_loader")
    metrics_module = load_module(HERE / "14_compare_t2024_explicit_substrate.py", "reduced_full_metrics")
    helpers = comparison.load_summary_helpers()
    records = {}
    raw_paths = []
    for label, full_name in FULL_DIRS.items():
        full_dir = FULL_ROOT / full_name
        reduced_dir = REDUCED_ROOT / f"T2024_MIR_4750_{label}"
        full = comparison.load_case(full_dir, helpers)
        reduced = comparison.load_case(reduced_dir, helpers)
        full_material = comparison.material_totals(full)
        reduced_material = comparison.material_totals(reduced)
        lateral = metrics_module.lateral_metrics(full, reduced)
        records[label] = {
            "full_1500nm": {
                "P_Q_W": full["result"]["P_Q_pabs_periodic_W"],
                "TaIrTe4_Q_W": full_material["TaIrTe4_geometric"],
                "runtime_s": full["result"]["solver_wall_time_s"],
            },
            "reduced_285nm": {
                "P_Q_W": reduced["result"]["P_Q_pabs_periodic_W"],
                "TaIrTe4_Q_W": reduced_material["TaIrTe4_geometric"],
                "runtime_s": reduced["result"]["solver_wall_time_s"],
                "bottom_transmission": reduced["result"]["transmission_bottom_monitor"],
            },
            "relative_change_reduced_vs_full": {
                "P_Q": reduced["result"]["P_Q_pabs_periodic_W"] / full["result"]["P_Q_pabs_periodic_W"] - 1.0,
                "TaIrTe4_Q": reduced_material["TaIrTe4_geometric"] / full_material["TaIrTe4_geometric"] - 1.0,
            },
            "lateral_Q_metrics": lateral,
        }
        raw_paths.extend(
            path
            for directory in (full_dir, reduced_dir)
            for path in (
                directory / "T2024_TaIrTe4_optical_smoke.json",
                directory / "T2024_TaIrTe4_native_q.npz",
            )
        )

    max_power = max(abs(item["relative_change_reduced_vs_full"]["P_Q"]) for item in records.values())
    max_tair = max(abs(item["relative_change_reduced_vs_full"]["TaIrTe4_Q"]) for item in records.values())
    max_shape = max(
        metric["unit_power_shape_NRMSE"]
        for item in records.values()
        for metric in item["lateral_Q_metrics"].values()
        if metric["active_for_shape_gate"]
    )
    max_bottom = max(item["reduced_285nm"]["bottom_transmission"] for item in records.values())
    gates = {
        "max_total_power_change_lt_0p5pct": max_power < 0.005,
        "max_TaIrTe4_power_change_lt_0p5pct": max_tair < 0.005,
        "max_lateral_shape_NRMSE_lt_0p5pct": max_shape < 0.005,
        "max_bottom_transmission_lt_1e_8": max_bottom < 1e-8,
    }
    summary = {
        "status": "VALIDATED_REDUCED_285NM_OPTICAL_CLOSURE" if all(gates.values()) else "FAILED_REDUCED_285NM_OPTICAL_CLOSURE",
        "records": records,
        "maxima": {
            "absolute_P_Q_change": max_power,
            "absolute_TaIrTe4_Q_change": max_tair,
            "lateral_unit_power_shape_NRMSE": max_shape,
            "bottom_transmission": max_bottom,
        },
        "gates": gates,
        "interpretation": "The 285-nm layer is an optical numerical closure below an opaque 200-nm Au mirror, not a replacement for the physical thermal oxide.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT / "REDUCED_285NM_VS_FULL_1500NM_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    labels = list(FULL_DIRS)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, [records[k]["full_1500nm"]["P_Q_W"] * 1e15 for k in labels], width, label="full 1.5 um")
    axes[0].bar(x + width / 2, [records[k]["reduced_285nm"]["P_Q_W"] * 1e15 for k in labels], width, label="reduced 285 nm")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="P_Q (fW/cell)", title="raw absorbed power")
    axes[0].legend()
    axes[1].bar(x, [1e6 * records[k]["relative_change_reduced_vs_full"]["P_Q"] for k in labels])
    axes[1].axhline(5000, color="red", ls="--", label="0.5% gate")
    axes[1].axhline(-5000, color="red", ls="--")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="relative change (ppm)", title="reduced - full")
    axes[1].legend()
    axes[2].bar(x, [records[k]["reduced_285nm"]["runtime_s"] for k in labels], label="reduced")
    axes[2].bar(x, [records[k]["full_1500nm"]["runtime_s"] for k in labels], alpha=0.45, label="full")
    axes[2].set(xticks=x, xticklabels=labels, ylabel="GPU solve time (s)", title="measured runtime")
    axes[2].legend()
    plot_path = OUTPUT / "REDUCED_285NM_VS_FULL_1500NM.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    report_path = OUTPUT / "REDUCED_285NM_VS_FULL_1500NM_REPORT.md"
    report_path.write_text(
        f"""# Reduced 285-nm optical closure versus full 1.5-um control

Status: `{summary['status']}`

The physical 1.5-um SiO2 control is preserved. The production optical solve
uses only 285 nm because a 200-nm Au mirror blocks transmission before the
oxide. This does **not** alter the later physical thermal geometry.

- maximum total-power change: `{100*max_power:.9f}%`
- maximum TaIrTe4-power change: `{100*max_tair:.9f}%`
- maximum active-component lateral-shape NRMSE: `{100*max_shape:.9f}%`
- maximum bottom transmission: `{max_bottom:.3e}`

No Q array was clipped, smoothed, gained, or rescaled.
"""
    )
    manifest = {
        "raw_artifacts_not_committed": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in raw_paths],
        "published": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in (summary_path, report_path, plot_path)],
    }
    (OUTPUT / "REDUCED_285NM_VS_FULL_1500NM_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
