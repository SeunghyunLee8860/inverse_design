#!/usr/bin/env python3
"""Compare the two validated GPU polarizations of the 2024 inverse-T smoke.

The native Qx/Qy/Qz arrays remain on their own Yee component grids.  This
script integrates and plots every component independently; it never adds
equal array indices from staggered component grids.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(
    "/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_architectures"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_summary_helpers():
    path = HERE / "08_summarize_t2024_tairte4_optical_smoke.py"
    spec = importlib.util.spec_from_file_location("t2024_summary_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_case(raw_dir: Path, helpers) -> dict:
    result_path = raw_dir / "T2024_TaIrTe4_optical_smoke.json"
    npz_path = raw_dir / "T2024_TaIrTe4_native_q.npz"
    result = json.loads(result_path.read_text())
    if result["status"] != "COMPLETED_T2024_TAIRTE4_OPTICAL_SMOKE":
        raise RuntimeError(f"Incomplete case: {result_path}")
    if not all(result["gates"].values()):
        raise RuntimeError(f"Failed gate in {result_path}: {result['gates']}")
    raw = np.load(npz_path)
    partition, areal = helpers.material_partition(raw, result)
    polarization = result["contract"]["source"]["polarization"]
    return {
        "raw_dir": raw_dir,
        "result_path": result_path,
        "npz_path": npz_path,
        "result": result,
        "raw": raw,
        "partition": partition,
        "areal": areal,
        "polarization": polarization,
    }


def material_totals(case: dict) -> dict[str, float]:
    keys = (
        "TaIrTe4_geometric",
        "top_Au_T_geometric",
        "Au_backplane_geometric",
        "SiO2_geometric",
        "Si_geometric",
        "unassigned_interface_or_other",
    )
    return {
        key: float(sum(case["partition"][component][key] for component in "xyz"))
        for key in keys
    }


def component_fractions(result: dict) -> dict[str, float]:
    powers = result["Q_component_power_native_W"]
    total = sum(float(powers[component]) for component in "xyz")
    return {component: float(powers[component]) / total for component in "xyz"}


def normalized_areal_metrics(case_b: dict, case_a: dict) -> dict:
    metrics = {}
    for component in "xyz":
        left = case_b["areal"][component]
        right = case_a["areal"][component]
        if not (
            np.array_equal(left["x"], right["x"])
            and np.array_equal(left["y"], right["y"])
        ):
            raise RuntimeError(f"{component}-component coordinates differ between polarizations")
        qb = np.asarray(left["Q_W_m2"], float)
        qa = np.asarray(right["Q_W_m2"], float)
        qb_norm = qb / np.sum(qb)
        qa_norm = qa / np.sum(qa)
        denom = max(float(np.linalg.norm(qb_norm)), float(np.linalg.norm(qa_norm)), 1e-300)
        nrmse = float(np.linalg.norm(qb_norm - qa_norm) / denom)
        correlation = float(np.corrcoef(qb_norm.ravel(), qa_norm.ravel())[0, 1])
        metrics[component] = {
            "equal_component_coordinates": True,
            "unit_sum_spatial_NRMSE": nrmse,
            "spatial_correlation": correlation,
        }
    return metrics


def plot_comparison(output: Path, cases: list[dict], summaries: dict) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 11.2), constrained_layout=True)
    row_titles = {"x_b": r"$E\parallel b$ (Lumerical x)", "y_a": r"$E\parallel a$ (Lumerical y)"}
    global_max = {
        component: max(float(np.max(case["areal"][component]["Q_W_m2"])) for case in cases)
        for component in "xyz"
    }
    for row, case in enumerate(cases):
        for col, component in enumerate("xyz"):
            item = case["areal"][component]
            image = axes[row, col].pcolormesh(
                item["x"] * 1e9,
                item["y"] * 1e9,
                item["Q_W_m2"].T,
                shading="auto",
                cmap="inferno",
                vmin=0.0,
                vmax=global_max[component],
            )
            axes[row, col].set_aspect("equal")
            axes[row, col].set_xlabel("x=b (nm)")
            axes[row, col].set_ylabel("y=a (nm)")
            axes[row, col].set_title(f"{row_titles[case['polarization']]}: native Yee $Q_{component}$")
            fig.colorbar(image, ax=axes[row, col], label="depth integral (W/m$^2$)")

    labels = [r"$E\parallel b$", r"$E\parallel a$"]
    x = np.arange(2)
    native = np.array([summaries[case["polarization"]]["P_Q_native_uncorrected_W"] for case in cases]) * 1e15
    flux = np.array([summaries[case["polarization"]]["P_flux_absorbed_W"] for case in cases]) * 1e15
    pabs = np.array([summaries[case["polarization"]]["P_Q_pabs_periodic_W"] for case in cases]) * 1e15
    width = 0.23
    axes[2, 0].bar(x - width, pabs, width, label="periodic pabs")
    axes[2, 0].bar(x, flux, width, label="six/flux absorption")
    axes[2, 0].bar(x + width, native, width, label="native Q sum")
    axes[2, 0].set_xticks(x, labels)
    axes[2, 0].set_ylabel("absorbed power per cell (fW)")
    axes[2, 0].set_title("raw power; no polarization matching")
    axes[2, 0].legend(fontsize=8)

    bottom = np.zeros(2)
    colors = {"x": "#4477aa", "y": "#ee6677", "z": "#228833"}
    for component in "xyz":
        values = np.array([
            summaries[case["polarization"]]["component_fractions"][component]
            for case in cases
        ])
        axes[2, 1].bar(x, values, bottom=bottom, label=f"Q{component}", color=colors[component])
        bottom += values
    axes[2, 1].set_xticks(x, labels)
    axes[2, 1].set_ylim(0, 1)
    axes[2, 1].set_ylabel("fraction of native Q power")
    axes[2, 1].set_title("component power fractions")
    axes[2, 1].legend(fontsize=8)

    material_keys = [
        "TaIrTe4_geometric",
        "top_Au_T_geometric",
        "Au_backplane_geometric",
        "SiO2_geometric",
        "Si_geometric",
        "unassigned_interface_or_other",
    ]
    material_labels = ["TaIrTe4", "top Au T", "Au mirror", "SiO2", "Si", "interface/other"]
    material_colors = ["#c74c4c", "#f6c64e", "#be8f00", "#9fd7d0", "#547aa5", "#777777"]
    bottom = np.zeros(2)
    for key, label, color in zip(material_keys, material_labels, material_colors):
        values = np.array([
            summaries[case["polarization"]]["material_totals_W"][key]
            for case in cases
        ]) * 1e15
        axes[2, 2].bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[2, 2].set_xticks(x, labels)
    axes[2, 2].set_ylabel("native geometric power (fW)")
    axes[2, 2].set_title("material partition; no deletion/rescaling")
    axes[2, 2].legend(fontsize=7)

    path = output / "T2024_TaIrTe4_polarization_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xb-dir", type=Path, default=DEFAULT_ROOT / "T2024_MIR_4750_xb_forward")
    parser.add_argument("--ya-dir", type=Path, default=DEFAULT_ROOT / "T2024_MIR_4750_ya_forward")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results_actual_metasurfaces")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    helpers = load_summary_helpers()
    cases = [load_case(args.xb_dir.resolve(), helpers), load_case(args.ya_dir.resolve(), helpers)]
    if [case["polarization"] for case in cases] != ["x_b", "y_a"]:
        raise RuntimeError("Expected x_b then y_a cases")
    substrate_modes = {case["result"]["contract"].get("substrate", {}).get("mode") for case in cases}
    if len(substrate_modes) != 1:
        raise RuntimeError(f"Substrate modes differ between polarizations: {substrate_modes}")

    per_case = {}
    for case in cases:
        result = case["result"]
        per_case[case["polarization"]] = {
            "solver_wall_time_s": result["solver_wall_time_s"],
            "source_power_W": result["source_power_W"],
            "P_Q_pabs_periodic_W": result["P_Q_pabs_periodic_W"],
            "P_flux_absorbed_W": result["P_flux_absorbed_W"],
            "P_Q_native_uncorrected_W": result["P_Q_native_uncorrected_W"],
            "absorptance_pabs": result["P_Q_pabs_periodic_W"] / result["source_power_W"],
            "absorptance_flux": result["P_flux_absorbed_W"] / result["source_power_W"],
            "closure_relative": result["closure_relative"],
            "reflection": result["reflection"],
            "auto_shutoff": result["log_audit"]["final_auto_shutoff"],
            "Q_component_power_native_W": result["Q_component_power_native_W"],
            "component_fractions": component_fractions(result),
            "material_totals_W": material_totals(case),
            "gates": result["gates"],
        }

    q_ratio_a_over_b = per_case["y_a"]["P_Q_pabs_periodic_W"] / per_case["x_b"]["P_Q_pabs_periodic_W"]
    tair_ratio_a_over_b = (
        per_case["y_a"]["material_totals_W"]["TaIrTe4_geometric"]
        / per_case["x_b"]["material_totals_W"]["TaIrTe4_geometric"]
    )
    spatial = normalized_areal_metrics(cases[0], cases[1])
    summary = {
        "status": "VALIDATED_T2024_FIGURE_DIGITIZED_TAIRTE4_TWO_POLARIZATION_OPTICAL_SMOKE",
        "identity_limit": "paper-derived MIR inverse-T scenario with a 100-nm TaIrTe4 active-layer substitution; not exact paper CAD and not graphene experiment reproduction",
        "polarizations": per_case,
        "raw_power_ratios": {
            "P_Q_Ea_over_Eb": q_ratio_a_over_b,
            "geometric_TaIrTe4_power_Ea_over_Eb": tair_ratio_a_over_b,
        },
        "unit_power_spatial_shape_metrics_Eb_vs_Ea": spatial,
        "coordinate_rule": "Qx/Qy/Qz were integrated and plotted on component-specific Yee coordinates; no same-index cross-component sum was formed",
        "scope": "optical forward only; no thermal, PTE, adjoint or optimization",
        "substrate_mode": next(iter(substrate_modes)),
        "Z2022_status": "BLOCKED_EXACT_Z_TOPOLOGY_NOT_PUBLISHED_IN_PDF",
    }
    summary_path = output / "T2024_TAIRTE4_TWO_POLARIZATION_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    plot_path = plot_comparison(output, cases, per_case)

    report_path = output / "T2024_TAIRTE4_TWO_POLARIZATION_REPORT.md"
    xb = per_case["x_b"]
    ya = per_case["y_a"]
    report_path.write_text(
        f"""# 2024 MIR inverse-T / TaIrTe4 two-polarization optical smoke

Status: `VALIDATED_T2024_FIGURE_DIGITIZED_TAIRTE4_TWO_POLARIZATION_OPTICAL_SMOKE`

This is a paper-derived scalar-geometry scenario, not a reproduction of the
graphene experiment and not exact author CAD. The 2024 paper's MIR inverse-T,
period, spacer, and Au thickness contract is retained while the active 2-D
material alone is deliberately replaced by 100-nm anisotropic TaIrTe4. The T
arm widths/lengths are digitized from Supplementary Fig. 14 axes.

## GPU results

| Metric | E parallel b (x) | E parallel a (y) |
|---|---:|---:|
| wall time (s) | {xb['solver_wall_time_s']:.3f} | {ya['solver_wall_time_s']:.3f} |
| source power (W/cell) | {xb['source_power_W']:.12e} | {ya['source_power_W']:.12e} |
| periodic P_Q (W/cell) | {xb['P_Q_pabs_periodic_W']:.12e} | {ya['P_Q_pabs_periodic_W']:.12e} |
| absorbed flux (W/cell) | {xb['P_flux_absorbed_W']:.12e} | {ya['P_flux_absorbed_W']:.12e} |
| pabs absorptance | {xb['absorptance_pabs']:.9f} | {ya['absorptance_pabs']:.9f} |
| closure | {100*xb['closure_relative']:.6f}% | {100*ya['closure_relative']:.6f}% |
| reflection | {xb['reflection']:.9f} | {ya['reflection']:.9f} |
| auto-shutoff | {xb['auto_shutoff']:.6e} | {ya['auto_shutoff']:.6e} |

Raw total periodic-Q ratio `E||a / E||b` is **{q_ratio_a_over_b:.6f}**. The
geometrically assigned TaIrTe4-only native-Q ratio is **{tair_ratio_a_over_b:.6f}**.
These are raw equal-source results; no polarization matching, clipping,
smoothing, gain, or global rescaling was applied.

Qx/Qy/Qz are retained on their independent staggered Yee coordinates. Each
component is integrated and plotted separately. Equal array indices from
different component grids are never treated as the same physical coordinate.

No thermal, PTE, adjoint, or optimization calculation was run in this stage.

## 2022 Z status

The 2022 Supplementary Table 1 publishes M1-M5 scalar dimensions, but the PDFs
do not publish polygon vertices or a unique arm-junction construction. Those
numbers are sufficient for a dimension audit but not for a unique Maxwell CAD.
The Z case therefore remains fail-closed until author geometry is recovered or
an explicitly named approximation is approved.
"""
    )

    manifest = {
        "raw_artifacts_not_committed": [],
        "published_artifacts": [],
    }
    for case in cases:
        for path in (
            case["raw_dir"] / "T2024_TaIrTe4_optical_smoke.fsp",
            case["npz_path"],
            case["result_path"],
        ):
            manifest["raw_artifacts_not_committed"].append(
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    for path in (summary_path, report_path, plot_path):
        manifest["published_artifacts"].append(
            {
                "path": str(path.relative_to(HERE)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest_path = output / "T2024_TAIRTE4_TWO_POLARIZATION_RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
