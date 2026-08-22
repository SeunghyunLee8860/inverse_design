#!/usr/bin/env python3
"""Compare the inverse-T cases against matched no-top-T controls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_architectures")
CASES = {
    "T_Eb": RAW_ROOT / "T2024_MIR_4750_xb_forward",
    "T_Ea": RAW_ROOT / "T2024_MIR_4750_ya_forward",
    "bare_Eb": RAW_ROOT / "T2024_MIR_4750_bare_xb_forward",
    "bare_Ea": RAW_ROOT / "T2024_MIR_4750_bare_ya_forward",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(filename: str, name: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results_actual_metasurfaces")
    args = parser.parse_args()
    cases = {
        "T_Eb": args.raw_root / "T2024_MIR_4750_xb_forward",
        "T_Ea": args.raw_root / "T2024_MIR_4750_ya_forward",
        "bare_Eb": args.raw_root / "T2024_MIR_4750_bare_xb_forward",
        "bare_Ea": args.raw_root / "T2024_MIR_4750_bare_ya_forward",
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    comparison = load_module(
        "09_compare_t2024_tairte4_polarizations.py", "t2024_comparison_helpers"
    )
    summary_helpers = comparison.load_summary_helpers()
    loaded = {
        key: comparison.load_case(path, summary_helpers)
        for key, path in cases.items()
    }
    for key in ("T_Eb", "T_Ea"):
        if not loaded[key]["result"]["contract"].get("top_Au_T_present", True):
            raise RuntimeError(f"{key} unexpectedly omits the T")
    for key in ("bare_Eb", "bare_Ea"):
        if loaded[key]["result"]["contract"].get("top_Au_T_present", True):
            raise RuntimeError(f"{key} unexpectedly contains the T")
    substrate_modes = {
        case["result"]["contract"].get("substrate", {}).get("mode")
        for case in loaded.values()
    }
    if len(substrate_modes) != 1:
        raise RuntimeError(f"Matched cases do not share one substrate mode: {substrate_modes}")

    records = {}
    for key, case in loaded.items():
        result = case["result"]
        materials = comparison.material_totals(case)
        records[key] = {
            "top_Au_T_present": result["contract"].get("top_Au_T_present", True),
            "polarization": result["contract"]["source"]["polarization"],
            "P_Q_periodic_W": result["P_Q_pabs_periodic_W"],
            "P_flux_absorbed_W": result["P_flux_absorbed_W"],
            "P_Q_native_W": result["P_Q_native_uncorrected_W"],
            "geometric_material_power_W": materials,
            "closure_relative": result["closure_relative"],
            "auto_shutoff": result["log_audit"]["final_auto_shutoff"],
            "solver_wall_time_s": result["solver_wall_time_s"],
        }

    enhancement = {}
    for label in ("Eb", "Ea"):
        t = records[f"T_{label}"]
        bare = records[f"bare_{label}"]
        enhancement[label] = {
            "total_P_Q_T_over_bare": t["P_Q_periodic_W"] / bare["P_Q_periodic_W"],
            "total_P_Q_relative_change": t["P_Q_periodic_W"] / bare["P_Q_periodic_W"] - 1.0,
            "TaIrTe4_power_T_over_bare": (
                t["geometric_material_power_W"]["TaIrTe4_geometric"]
                / bare["geometric_material_power_W"]["TaIrTe4_geometric"]
            ),
            "TaIrTe4_power_relative_change": (
                t["geometric_material_power_W"]["TaIrTe4_geometric"]
                / bare["geometric_material_power_W"]["TaIrTe4_geometric"]
                - 1.0
            ),
        }
    selectivity = {
        "bare_PQ_Eb_over_Ea": records["bare_Eb"]["P_Q_periodic_W"] / records["bare_Ea"]["P_Q_periodic_W"],
        "with_T_PQ_Eb_over_Ea": records["T_Eb"]["P_Q_periodic_W"] / records["T_Ea"]["P_Q_periodic_W"],
        "bare_TaIrTe4_Eb_over_Ea": (
            records["bare_Eb"]["geometric_material_power_W"]["TaIrTe4_geometric"]
            / records["bare_Ea"]["geometric_material_power_W"]["TaIrTe4_geometric"]
        ),
        "with_T_TaIrTe4_Eb_over_Ea": (
            records["T_Eb"]["geometric_material_power_W"]["TaIrTe4_geometric"]
            / records["T_Ea"]["geometric_material_power_W"]["TaIrTe4_geometric"]
        ),
    }
    summary = {
        "status": "VALIDATED_T2024_TOP_T_MATCHED_BARE_OPTICAL_COMPARISON",
        "records": records,
        "T_enhancement": enhancement,
        "polarization_selectivity": selectivity,
        "interpretation": (
            "At 4.75 um the figure-digitized top T enhances the raw E||b absorption "
            "but suppresses E||a; the TaIrTe4-only geometric partition is reported "
            "separately from Au and interface loss."
        ),
        "limitations": [
            "single wavelength only; not a resonance spectrum",
            "figure-digitized T rather than author CAD",
            "100-nm TaIrTe4 substitution rather than graphene",
            "geometric material partition retains an explicit interface/other residual",
        ],
        "no_clipping_smoothing_gain_rescaling_or_polarization_matching": True,
        "substrate_mode": next(iter(substrate_modes)),
    }
    summary_path = output / "T2024_TOP_T_MATCHED_BARE_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    order = ["bare_Eb", "T_Eb", "bare_Ea", "T_Ea"]
    labels = [r"bare, $E\parallel b$", r"T, $E\parallel b$", r"bare, $E\parallel a$", r"T, $E\parallel a$"]
    x = np.arange(4)
    axes[0, 0].bar(x, [records[key]["P_Q_periodic_W"] * 1e15 for key in order], color=["#88aadd", "#4477aa", "#ee99aa", "#cc6677"])
    axes[0, 0].set_xticks(x, labels, rotation=15)
    axes[0, 0].set_ylabel("raw periodic P_Q (fW/cell)")
    axes[0, 0].set_title("total absorption; no polarization matching")

    material_keys = [
        "TaIrTe4_geometric",
        "top_Au_T_geometric",
        "Au_backplane_geometric",
        "SiO2_geometric",
        "Si_geometric",
        "unassigned_interface_or_other",
    ]
    material_labels = ["TaIrTe4", "top Au region", "Au mirror", "SiO2", "Si", "interface/other"]
    material_colors = ["#c74c4c", "#f6c64e", "#be8f00", "#9fd7d0", "#547aa5", "#777777"]
    bottom = np.zeros(4)
    for key, label, color in zip(material_keys, material_labels, material_colors):
        values = np.array([records[item]["geometric_material_power_W"][key] for item in order]) * 1e15
        axes[0, 1].bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[0, 1].set_xticks(x, labels, rotation=15)
    axes[0, 1].set_ylabel("native geometric Q power (fW/cell)")
    axes[0, 1].set_title("material-resolved native Q")
    axes[0, 1].legend(fontsize=8)

    for col, (label, component) in enumerate((("Eb", "x"), ("Ea", "y"))):
        axis = axes[1, col]
        case_bare = loaded[f"bare_{label}"]["areal"][component]
        case_t = loaded[f"T_{label}"]["areal"][component]
        if not (np.array_equal(case_bare["x"], case_t["x"]) and np.array_equal(case_bare["y"], case_t["y"])):
            raise RuntimeError(f"{label} dominant-component coordinates differ")
        difference = case_t["Q_W_m2"] - case_bare["Q_W_m2"]
        vmax = float(np.max(np.abs(difference)))
        image = axis.pcolormesh(
            case_t["x"] * 1e9,
            case_t["y"] * 1e9,
            difference.T,
            shading="auto",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        axis.set_aspect("equal")
        axis.set_xlabel("x=b (nm)")
        axis.set_ylabel("y=a (nm)")
        axis.set_title(rf"T - bare: dominant native $Q_{component}$, $E\parallel {label[-1].lower()}$")
        fig.colorbar(image, ax=axis, label="depth-integrated difference (W/m$^2$)")

    plot_path = output / "T2024_top_T_matched_bare_comparison.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    eb = enhancement["Eb"]
    ea = enhancement["Ea"]
    report_path = output / "T2024_TOP_T_MATCHED_BARE_REPORT.md"
    report_path.write_text(
        f"""# 2024 inverse-T / TaIrTe4 matched bare-control comparison

Status: `VALIDATED_T2024_TOP_T_MATCHED_BARE_OPTICAL_COMPARISON`

The only geometry change is removal of the 33-nm top Au inverse-T. The
periodic cell, 100-nm TaIrTe4, 35-nm Al2O3, Au mirror, materials, source,
mesh, boundary conditions, and normalization remain identical.

| Quantity | E parallel b | E parallel a |
|---|---:|---:|
| total P_Q, bare (W/cell) | {records['bare_Eb']['P_Q_periodic_W']:.12e} | {records['bare_Ea']['P_Q_periodic_W']:.12e} |
| total P_Q, with T (W/cell) | {records['T_Eb']['P_Q_periodic_W']:.12e} | {records['T_Ea']['P_Q_periodic_W']:.12e} |
| T / bare, total | {eb['total_P_Q_T_over_bare']:.6f} | {ea['total_P_Q_T_over_bare']:.6f} |
| relative total change | {100*eb['total_P_Q_relative_change']:.4f}% | {100*ea['total_P_Q_relative_change']:.4f}% |
| T / bare, TaIrTe4-only geometric Q | {eb['TaIrTe4_power_T_over_bare']:.6f} | {ea['TaIrTe4_power_T_over_bare']:.6f} |
| relative TaIrTe4 change | {100*eb['TaIrTe4_power_relative_change']:.4f}% | {100*ea['TaIrTe4_power_relative_change']:.4f}% |

At this single wavelength the digitized T is polarization selective: it
enhances E||b absorption and suppresses E||a. The bare total `Eb/Ea` ratio is
{selectivity['bare_PQ_Eb_over_Ea']:.6f}; with the T it becomes
{selectivity['with_T_PQ_Eb_over_Ea']:.6f}. This is a forward optical result,
not yet a thermal or PTE improvement claim.

All four GPU cases passed closure (<0.5%), auto-shutoff (<1e-5), finite-Q,
and nonnegative-Q gates. No clipping, smoothing, gain, global rescaling, or
polarization matching was used. Qx/Qy/Qz remain on component-specific Yee
coordinates. The lower panels plot only the incident-dominant component on
its own identical grid; they are not cross-component sums.
"""
    )

    raw_paths = []
    for raw_dir in cases.values():
        raw_paths.extend(
            [
                raw_dir / "T2024_TaIrTe4_optical_smoke.fsp",
                raw_dir / "T2024_TaIrTe4_native_q.npz",
                raw_dir / "T2024_TaIrTe4_optical_smoke.json",
            ]
        )
    manifest = {
        "raw_artifacts_not_committed": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in raw_paths
        ],
        "published_artifacts": [
            {
                "path": str(path.relative_to(HERE)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (summary_path, report_path, plot_path)
        ],
    }
    manifest_path = output / "T2024_TOP_T_MATCHED_BARE_RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
