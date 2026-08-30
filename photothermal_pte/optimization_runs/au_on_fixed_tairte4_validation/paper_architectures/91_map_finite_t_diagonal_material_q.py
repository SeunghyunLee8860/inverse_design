#!/usr/bin/env python3
"""Conservatively map finite T11x15 +/-45-degree Yee Q to the thermal grid."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "86_map_finite_t_z_array_material_q.py"
RAW = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_array_Q")
RAW_OUT = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_diagonal_material_Q")
OUTPUT = HERE / "results_finite_T_diagonal_material_Q_mapping"
CASES = {
    "T11x15_linear_plus_45_Au_on": RAW / "T11x15_linear_plus_45_Au_on",
    "T11x15_linear_minus_45_Au_on": RAW / "T11x15_linear_minus_45_Au_on",
}


def load_source():
    spec = importlib.util.spec_from_file_location("finite_t_diagonal_mapping_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    base = load_source()
    base.RAW_OUT = RAW_OUT
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = {name: base.map_case(name, directory) for name, directory in CASES.items()}
    gates = {
        "all_power_conservation_lt_1e-12": max(
            value["power_conservation_relative_error"] for value in cases.values()
        ) < 1e-12,
        "no_positive_power_cell_without_loss_overlap": sum(
            value["positive_power_cells_without_loss_overlap"] for value in cases.values()
        ) == 0,
    }
    status = (
        "VALIDATED_FINITE_T11X15_DIAGONAL_COMPONENT_MATERIAL_Q_MAPPING"
        if all(gates.values())
        else "FAILED_FINITE_T11X15_DIAGONAL_COMPONENT_MATERIAL_Q_MAPPING"
    )
    summary = {
        "status": status,
        "classification": (
            "component-specific Yee power partitioned by exact material-loss overlap "
            "and conservatively deposited on the unchanged explicit thermal grid"
        ),
        "polarizations": {
            "linear_plus_45": "(Eb+Ea)/sqrt(2)",
            "linear_minus_45": "(-Eb+Ea)/sqrt(2) under the Lumerical source-angle convention",
        },
        "gates": gates,
        "cases": cases,
    }
    summary_path = OUTPUT / "FINITE_T_DIAGONAL_MATERIAL_Q_MAPPING_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    fields = [
        "case", "input_total_power_W", "mapped_total_power_W",
        "power_conservation_relative_error", "Si_W", "SiO2_W", "Au_mirror_W",
        "TaIrTe4_W", "top_Au_W",
    ]
    with (OUTPUT / "finite_T_diagonal_material_Q_mapping_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, value in cases.items():
            material = value["mapped_power_by_material_W"]
            writer.writerow(
                {
                    "case": name,
                    "input_total_power_W": value["input_total_power_W"],
                    "mapped_total_power_W": value["mapped_total_power_W"],
                    "power_conservation_relative_error": value["power_conservation_relative_error"],
                    **{f"{key}_W": material[key] for key in ("Si", "SiO2", "Au_mirror", "TaIrTe4", "top_Au")},
                }
            )
    lines = [
        "# Finite T11x15 diagonal-polarization material-Q mapping",
        "",
        f"Status: `{status}`",
        "",
        "No Q clipping, smoothing, gain, global rescaling, tiling, or whole-cell material reassignment was used.",
        "",
        "| case | TaIrTe4 | top Au | mirror Au | SiO2 | Si | mapping error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in cases.items():
        material = value["mapped_power_by_material_W"]
        lines.append(
            f"| {name} | {material['TaIrTe4']:.9e} | {material['top_Au']:.9e} | "
            f"{material['Au_mirror']:.9e} | {material['SiO2']:.9e} | {material['Si']:.9e} | "
            f"{value['power_conservation_relative_error']:.3e} |"
        )
    (OUTPUT / "FINITE_T_DIAGONAL_MATERIAL_Q_MAPPING_REPORT.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "status": status,
        "raw_files_committed_to_git": False,
        "artifacts": {name: value["raw_mapped_artifact"] for name, value in cases.items()},
    }
    (OUTPUT / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": status, "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
