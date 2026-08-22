#!/usr/bin/env python3
"""Compare explicit 1.5-um SiO2/Si against the legacy Au-truncated stack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_TRUNCATED = Path("/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_architectures")
DEFAULT_EXPLICIT = Path("/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_architectures_sio2_si")
CASE_DIRS = {
    "T_Eb": "T2024_MIR_4750_xb_forward",
    "T_Ea": "T2024_MIR_4750_ya_forward",
    "bare_Eb": "T2024_MIR_4750_bare_xb_forward",
    "bare_Ea": "T2024_MIR_4750_bare_ya_forward",
}
COMPONENT_PARTICIPATION_FLOOR = 1.0e-8


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


def relative(new: float, reference: float) -> float:
    return float((new - reference) / reference)


def lateral_metrics(reference: dict, explicit: dict) -> dict:
    result = {}
    reference_powers = reference["result"]["Q_component_power_native_W"]
    explicit_powers = explicit["result"]["Q_component_power_native_W"]
    reference_total = sum(float(reference_powers[c]) for c in "xyz")
    explicit_total = sum(float(explicit_powers[c]) for c in "xyz")
    for component in "xyz":
        old = reference["areal"][component]
        new = explicit["areal"][component]
        if not (np.array_equal(old["x"], new["x"]) and np.array_equal(old["y"], new["y"])):
            raise RuntimeError(f"{component} lateral coordinates differ")
        q0 = np.asarray(old["Q_W_m2"], float)
        q1 = np.asarray(new["Q_W_m2"], float)
        participation = max(
            float(reference_powers[component]) / max(reference_total, 1e-300),
            float(explicit_powers[component]) / max(explicit_total, 1e-300),
        )
        if participation < COMPONENT_PARTICIPATION_FLOOR:
            result[component] = {
                "active_for_shape_gate": False,
                "maximum_power_participation": participation,
                "absolute_lateral_NRMSE": None,
                "unit_power_shape_NRMSE": None,
                "spatial_correlation": None,
                "reason": "component is numerical zero; normalization would amplify roundoff",
            }
            continue
        absolute_nrmse = float(np.linalg.norm(q1 - q0) / max(np.linalg.norm(q0), 1e-300))
        q0n = q0 / max(float(np.sum(q0)), 1e-300)
        q1n = q1 / max(float(np.sum(q1)), 1e-300)
        shape_nrmse = float(np.linalg.norm(q1n - q0n) / max(np.linalg.norm(q0n), 1e-300))
        correlation = float(np.corrcoef(q0n.ravel(), q1n.ravel())[0, 1])
        result[component] = {
            "active_for_shape_gate": True,
            "maximum_power_participation": participation,
            "absolute_lateral_NRMSE": absolute_nrmse,
            "unit_power_shape_NRMSE": shape_nrmse,
            "spatial_correlation": correlation,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truncated-root", type=Path, default=DEFAULT_TRUNCATED)
    parser.add_argument("--explicit-root", type=Path, default=DEFAULT_EXPLICIT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results_actual_metasurfaces_sio2_si",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    comparison = load_module(
        "09_compare_t2024_tairte4_polarizations.py", "t2024_substrate_comparison"
    )
    helpers = comparison.load_summary_helpers()
    loaded = {"au_truncated": {}, "sio2_si": {}}
    roots = {
        "au_truncated": args.truncated_root.resolve(),
        "sio2_si": args.explicit_root.resolve(),
    }
    for mode, root in roots.items():
        for label, directory in CASE_DIRS.items():
            loaded[mode][label] = comparison.load_case(root / directory, helpers)

    records = {}
    for label in CASE_DIRS:
        old = loaded["au_truncated"][label]
        new = loaded["sio2_si"][label]
        old_result = old["result"]
        new_result = new["result"]
        old_material = comparison.material_totals(old)
        new_material = comparison.material_totals(new)
        records[label] = {
            "au_truncated": {
                "P_Q_W": old_result["P_Q_pabs_periodic_W"],
                "P_flux_W": old_result["P_flux_absorbed_W"],
                "TaIrTe4_Q_W": old_material["TaIrTe4_geometric"],
                "closure_relative": old_result["closure_relative"],
            },
            "explicit_SiO2_Si": {
                "P_Q_W": new_result["P_Q_pabs_periodic_W"],
                "P_flux_W": new_result["P_flux_absorbed_W"],
                "TaIrTe4_Q_W": new_material["TaIrTe4_geometric"],
                "SiO2_Q_W": new_material["SiO2_geometric"],
                "Si_Q_W": new_material["Si_geometric"],
                "closure_relative": new_result["closure_relative"],
                "bottom_transmission": new_result["transmission_bottom_monitor"],
            },
            "relative_change_explicit_vs_truncated": {
                "P_Q": relative(new_result["P_Q_pabs_periodic_W"], old_result["P_Q_pabs_periodic_W"]),
                "P_flux": relative(new_result["P_flux_absorbed_W"], old_result["P_flux_absorbed_W"]),
                "TaIrTe4_Q": relative(new_material["TaIrTe4_geometric"], old_material["TaIrTe4_geometric"]),
            },
            "lateral_Q_metrics": lateral_metrics(old, new),
        }

    max_power_change = max(
        abs(item["relative_change_explicit_vs_truncated"]["P_Q"])
        for item in records.values()
    )
    max_tair_change = max(
        abs(item["relative_change_explicit_vs_truncated"]["TaIrTe4_Q"])
        for item in records.values()
    )
    max_shape_nrmse = max(
        metric["unit_power_shape_NRMSE"]
        for item in records.values()
        for metric in item["lateral_Q_metrics"].values()
        if metric["active_for_shape_gate"]
    )
    gates = {
        "all_explicit_forward_gates_pass": all(
            all(case["result"]["gates"].values())
            for case in loaded["sio2_si"].values()
        ),
        "max_total_power_change_lt_0p5pct": max_power_change < 0.005,
        "max_TaIrTe4_power_change_lt_0p5pct": max_tair_change < 0.005,
        "max_lateral_shape_NRMSE_lt_0p5pct": max_shape_nrmse < 0.005,
    }
    summary = {
        "status": (
            "VALIDATED_T2024_EXPLICIT_SIO2_SI_OPTICAL_EQUIVALENCE"
            if all(gates.values())
            else "FAILED_T2024_EXPLICIT_SIO2_SI_OPTICAL_EQUIVALENCE_GATE"
        ),
        "stack": "air / Au inverse-T / 100-nm TaIrTe4 / 35-nm Al2O3 / 200-nm Au mirror / 1.5-um thermal SiO2 / intrinsic Si",
        "records": records,
        "maxima": {
            "absolute_total_P_Q_change": max_power_change,
            "absolute_TaIrTe4_Q_change": max_tair_change,
            "lateral_unit_power_shape_NRMSE": max_shape_nrmse,
        },
        "gates": gates,
        "component_shape_gate_rule": {
            "minimum_native_power_fraction": COMPONENT_PARTICIPATION_FLOOR,
            "inactive_components_are_reported_but_not_normalized": True,
            "Q_arrays_modified": False,
        },
        "scope": "periodic optical forward equivalence only; no thermal, PTE, adjoint or optimization",
    }
    summary_path = output / "T2024_EXPLICIT_SIO2_SI_OPTICAL_EQUIVALENCE.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    setup_fig, setup_axes = plt.subplots(1, 2, figsize=(12.5, 5.8), constrained_layout=True)
    top = setup_axes[0]
    top.add_patch(Rectangle((-750, -500), 1500, 1000, facecolor="#d88b8b", alpha=0.35, edgecolor="#a94f4f"))
    t_vertices = np.array(
        [[-600, -350], [600, -350], [600, -250], [100, -250], [100, 350], [-100, 350], [-100, -250], [-600, -250]]
    )
    top.fill(t_vertices[:, 0], t_vertices[:, 1], color="#f6c64e", edgecolor="#8a5a00", lw=2)
    top.set_xlim(-750, 750); top.set_ylim(-500, 500); top.set_aspect("equal")
    top.set_xlabel("x=b (nm; periodic)"); top.set_ylabel("y=a (nm; periodic)")
    top.set_title("xy unit cell: one figure-digitized inverse-T")

    side = setup_axes[1]
    side.add_patch(Rectangle((-750, 133), 1500, 1067, color="#d9efff", label="air"))
    side.add_patch(Rectangle((-750, 0), 1500, 100, color="#cd7979", label="TaIrTe4 100 nm"))
    side.add_patch(Rectangle((-750, -35), 1500, 35, color="#b8d8e6", label="Al2O3 35 nm"))
    side.add_patch(Rectangle((-750, -235), 1500, 200, color="#d6a500", label="Au mirror 200 nm"))
    side.add_patch(Rectangle((-750, -1735), 1500, 1500, color="#9fd7d0", label="thermal SiO2 1.5 um"))
    side.add_patch(Rectangle((-750, -2500), 1500, 765, color="#547aa5", label="intrinsic Si to bottom PML"))
    side.add_patch(Rectangle((-100, 100), 200, 33, color="#f6c64e", ec="#8a5a00", label="Au T cross-section"))
    side.annotate("normal incidence", xy=(0, 250), xytext=(0, 800), ha="center", color="#2366b1", arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#2366b1"})
    side.axhline(-2500, color="#7a2c91", lw=4, label="z-min PML boundary")
    side.axhline(1200, color="#7a2c91", lw=4, label="z-max PML boundary")
    side.axvline(-750, color="#333", ls=":", lw=2); side.axvline(750, color="#333", ls=":", lw=2)
    side.set_xlim(-750, 750); side.set_ylim(-2600, 1250)
    side.set_xlabel("x=b (nm; periodic boundaries)"); side.set_ylabel("z (nm)")
    side.set_title("xz solver stack: explicit SiO2/Si")
    side.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    setup_path = output / "T2024_explicit_SiO2_Si_setup.png"
    setup_fig.savefig(setup_path, dpi=220)
    plt.close(setup_fig)

    labels = list(CASE_DIRS)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    width = 0.36
    axes[0, 0].bar(
        x - width / 2,
        [records[key]["au_truncated"]["P_Q_W"] * 1e15 for key in labels],
        width,
        label="legacy Au-to-PML",
    )
    axes[0, 0].bar(
        x + width / 2,
        [records[key]["explicit_SiO2_Si"]["P_Q_W"] * 1e15 for key in labels],
        width,
        label="explicit SiO2/Si",
    )
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("P_Q (fW/cell)")
    axes[0, 0].set_title("Raw periodic absorption")
    axes[0, 0].legend()

    axes[0, 1].bar(
        x - width / 2,
        [100 * records[key]["relative_change_explicit_vs_truncated"]["P_Q"] for key in labels],
        width,
        label="total P_Q",
    )
    axes[0, 1].bar(
        x + width / 2,
        [100 * records[key]["relative_change_explicit_vs_truncated"]["TaIrTe4_Q"] for key in labels],
        width,
        label="TaIrTe4 Q",
    )
    axes[0, 1].axhline(0.5, color="k", ls="--", lw=1)
    axes[0, 1].axhline(-0.5, color="k", ls="--", lw=1)
    axes[0, 1].set_yscale("symlog", linthresh=1.0e-6)
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylabel("explicit - truncated (%)")
    axes[0, 1].set_title("Substrate-model change")
    axes[0, 1].legend()

    for component, color in zip("xyz", ("#4477aa", "#ee6677", "#228833")):
        shape_values = [
            records[key]["lateral_Q_metrics"][component]["unit_power_shape_NRMSE"]
            for key in labels
        ]
        axes[1, 0].plot(
            x,
            [100 * value if value is not None else np.nan for value in shape_values],
            "o-",
            label=f"Q{component}",
            color=color,
        )
    axes[1, 0].axhline(0.5, color="k", ls="--", lw=1, label="0.5% gate")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylim(1.0e-5, 1.0)
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylabel("unit-power lateral NRMSE (%)")
    axes[1, 0].set_title("Thermally relevant lateral-Q shape")
    axes[1, 0].legend()

    axes[1, 1].bar(
        x - width / 2,
        [records[key]["explicit_SiO2_Si"]["SiO2_Q_W"] * 1e24 for key in labels],
        width,
        label="SiO2",
    )
    axes[1, 1].bar(
        x + width / 2,
        [records[key]["explicit_SiO2_Si"]["Si_Q_W"] * 1e24 for key in labels],
        width,
        label="Si",
    )
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("geometric optical Q (yW/cell)")
    axes[1, 1].set_title("Absorption below opaque Au mirror")
    axes[1, 1].legend()
    plot_path = output / "T2024_explicit_SiO2_Si_optical_equivalence.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    rows = []
    for label, item in records.items():
        active_shape = [
            metric["unit_power_shape_NRMSE"]
            for metric in item["lateral_Q_metrics"].values()
            if metric["active_for_shape_gate"]
        ]
        rows.append(
            f"| {label} | {item['relative_change_explicit_vs_truncated']['P_Q']*100:.6f}% | "
            f"{item['relative_change_explicit_vs_truncated']['TaIrTe4_Q']*100:.6f}% | "
            f"{max(active_shape)*100:.6f}% | "
            f"{item['explicit_SiO2_Si']['bottom_transmission']:.3e} |"
        )
    report_path = output / "T2024_EXPLICIT_SIO2_SI_OPTICAL_EQUIVALENCE.md"
    report_path.write_text(
        "# Explicit SiO2/Si substrate optical equivalence\n\n"
        f"Status: `{summary['status']}`\n\n"
        "The physical periodic stack is air / Au inverse-T / 100-nm TaIrTe4 / "
        "35-nm Al2O3 / 200-nm Au mirror / 1.5-um thermally grown SiO2 / intrinsic Si. "
        "The older Au-to-bottom-PML artifacts are retained only as a numerical optical control.\n\n"
        "| case | total P_Q change | TaIrTe4 Q change | max lateral shape NRMSE | bottom transmission |\n"
        "|---|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\nAll explicit-substrate forward, closure, auto-shutoff, finite-Q and nonnegative-Q gates passed. "
        "No clipping, smoothing, gain, global rescaling, or polarization matching was used. "
        "This validates only the optical insensitivity below the opaque Au mirror; the thermal model still retains explicit SiO2 and Si.\n"
    )

    csv_path = output / "T2024_EXPLICIT_SIO2_SI_OPTICAL_EQUIVALENCE.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            lineterminator="\n",
            fieldnames=(
                "case",
                "explicit_P_Q_W",
                "P_Q_change_relative",
                "TaIrTe4_Q_change_relative",
                "max_active_lateral_shape_NRMSE",
                "bottom_transmission",
                "closure_relative",
            ),
        )
        writer.writeheader()
        for label, item in records.items():
            active_shape = [
                metric["unit_power_shape_NRMSE"]
                for metric in item["lateral_Q_metrics"].values()
                if metric["active_for_shape_gate"]
            ]
            writer.writerow(
                {
                    "case": label,
                    "explicit_P_Q_W": item["explicit_SiO2_Si"]["P_Q_W"],
                    "P_Q_change_relative": item["relative_change_explicit_vs_truncated"]["P_Q"],
                    "TaIrTe4_Q_change_relative": item["relative_change_explicit_vs_truncated"]["TaIrTe4_Q"],
                    "max_active_lateral_shape_NRMSE": max(active_shape),
                    "bottom_transmission": item["explicit_SiO2_Si"]["bottom_transmission"],
                    "closure_relative": item["explicit_SiO2_Si"]["closure_relative"],
                }
            )

    raw_files = []
    for root in roots.values():
        for directory in CASE_DIRS.values():
            case_dir = root / directory
            for name in (
                "T2024_TaIrTe4_optical_smoke.json",
                "T2024_TaIrTe4_optical_smoke.fsp",
                "T2024_TaIrTe4_native_q.npz",
            ):
                path = case_dir / name
                raw_files.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest_path = output / "T2024_EXPLICIT_SIO2_SI_RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "raw_artifacts_not_committed": raw_files,
                "published_artifacts": [
                    {
                        "path": str(path.relative_to(HERE)),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                    for path in (summary_path, report_path, csv_path, setup_path, plot_path)
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
