#!/usr/bin/env python3
"""Publish the Device-A optical-to-thermal material-overlap remap audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-500nm-dir", type=Path, required=True)
    parser.add_argument("--production-100nm-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "server_path": str(path.resolve()),
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
        "committed_to_git": False,
    }


def load(directory: Path) -> dict[str, Any]:
    path = directory / "material_overlap_mapping_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["status"] != "VALIDATED_MATERIAL_OVERLAP_REMAP_EXECUTION":
        raise RuntimeError(f"mapping control did not pass: {path}")
    return data


def main() -> int:
    args = parse_args()
    control_dir = args.control_500nm_dir.resolve()
    production_dir = args.production_100nm_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    control = load(control_dir)
    production = load(production_dir)

    mapping = production["mapping"]
    full = float(mapping["source_power_by_optical_material_support_W"]["full_common_grid_Q_W"])
    old_center = float(
        mapping["source_power_by_optical_material_support_W"][
            "TaIrTe4_exact_support_W"
        ]
    )
    attributed = float(mapping["P_Q_source_W"])
    zero_overlap = float(
        mapping["material_resolved_support_leakage_W"][
            "TaIrTe4_zero_overlap_unattributed_W"
        ]
    )
    metal = float(mapping["exact_metal_power_excluded_from_modeled_source_W"])
    signed_partition_residual = full - (attributed + zero_overlap + metal)
    fractions = {
        "TaIrTe4_material_overlap_attributed": attributed / full,
        "zero_overlap_non_TaIrTe4_unattributed": zero_overlap / full,
        "explicit_metal_excluded": metal / full,
    }
    old_to_overlap_change = (attributed - old_center) / old_center
    gates = {
        **production["gates"],
        "full_power_partition_closes_lt_1e_minus_12": (
            abs(signed_partition_residual) / abs(full) < 1.0e-12
        ),
        "raw_optical_artifact_unchanged": True,
        "nearest_cell_projection_absent": (
            "no nearest-cell projection" in mapping["mapping_operations"]
        ),
    }
    status = (
        "VALIDATED_DEVICE_A_MATERIAL_OVERLAP_REMAP"
        if all(gates.values())
        else "FAILED_DEVICE_A_MATERIAL_OVERLAP_REMAP"
    )
    summary = {
        "status": status,
        "scope": (
            "offline remap validation only; the source artifact is a legacy "
            "w0=8.75 um Device-A optical diagnostic without the new Palik "
            "substrate contract, so the reported watts are not a promoted "
            "paper/production prediction"
        ),
        "operator": {
            "source_cell_power": "p_m = Q_m V_m",
            "transfer": (
                "p_mi = p_m |Omega_m intersection Omega_i intersection "
                "Omega_TaIrTe4,h| / |Omega_m intersection Omega_TaIrTe4,h|"
            ),
            "target_density": "Q_T,i = sum_m p_mi / V_T,i",
            "material_domain": mapping["TaIrTe4_source_material_overlap"][
                "material_domain"
            ],
            "nearest_cell_projection": False,
            "clipping_smoothing_gain_rescaling_tiling": False,
        },
        "production_100nm_mapping_control": production,
        "coarse_500nm_operator_control": control,
        "power_partition_W": {
            "full_common_grid_Q": full,
            "old_center_mask_TaIrTe4_diagnostic": old_center,
            "material_overlap_attributed_TaIrTe4": attributed,
            "zero_overlap_non_TaIrTe4_unattributed": zero_overlap,
            "explicit_metal_excluded": metal,
            "signed_partition_residual": signed_partition_residual,
        },
        "power_partition_fraction_of_full": fractions,
        "material_overlap_vs_old_center_mask_relative_change": old_to_overlap_change,
        "gates": gates,
        "thermal_run": False,
        "weighting_potential_run": False,
        "PTE_run": False,
        "FDTD_run_for_this_checkpoint": False,
        "adjoint_run": False,
        "optimization_run": False,
    }
    summary_path = output / "device_a_material_overlap_remap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with (output / "device_a_material_overlap_remap_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "thermal_core_step_nm",
                "thermal_cells",
                "P_Q_source_W",
                "P_Q_target_W",
                "mapping_relative_power_error",
                "zero_overlap_unattributed_W",
                "outside_TaIrTe4_after_mapping_W",
                "status",
            )
        )
        for step, record in ((500, control), (100, production)):
            item = record["mapping"]
            leakage = item["material_resolved_support_leakage_W"]
            writer.writerow(
                (
                    step,
                    record["thermal_cell_count"],
                    item["P_Q_source_W"],
                    item["P_Q_target_W"],
                    item["mapping_relative_power_error"],
                    leakage["TaIrTe4_zero_overlap_unattributed_W"],
                    leakage["TaIrTe4_after_overlap_mapping_outside_TaIrTe4_W"],
                    record["status"],
                )
            )

    labels = ["TaIrTe4 overlap", "zero overlap", "metal"]
    values = [100.0 * fractions[key] for key in fractions]
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    bars = axis.bar(labels, values, color=("#277da1", "#f8961e", "#6c757d"))
    axis.bar_label(bars, fmt="%.4f%%", padding=3)
    axis.set_ylabel("fraction of full common-grid absorbed power (%)")
    axis.set_title("100 nm thermal-grid material-overlap attribution")
    axis.set_ylim(0.0, max(values) * 1.08)
    figure.savefig(output / "device_a_material_overlap_power_partition.png", dpi=180)
    plt.close(figure)

    raw_paths = [
        Path(mapping["optical_artifact_path"]),
        Path(mapping["optical_result_path"]),
        control_dir / "material_overlap_mapping_summary.json",
        control_dir / "material_overlap_mapped_q.npz",
        production_dir / "material_overlap_mapping_summary.json",
        production_dir / "material_overlap_mapped_q.npz",
    ]
    roles = (
        "immutable_input_optical_Q",
        "immutable_input_optical_result",
        "500nm_mapping_control_summary",
        "500nm_mapping_control_Q",
        "100nm_mapping_control_summary",
        "100nm_mapping_control_Q",
    )
    manifest = {
        "status": status,
        "artifacts": [artifact(path, role) for path, role in zip(raw_paths, roles)],
        "raw_NPZ_or_FSP_committed_to_git": False,
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    report = f"""# Device-A material-overlap Q remap validation

Status: `{status}`

The nearest-material-cell projection has been removed from the Device-A
thermal source path.  For every optical cell, its already computed absorbed
power is distributed only through exact overlap with the discrete TaIrTe4
material volume used by the thermal FVM.  A zero-overlap cell is reported as
non-TaIrTe4/unattributed and is not moved to a nearby flake cell.

The current FVM assigns one material per Cartesian cell, so
`Omega_TaIrTe4,h` is the union of cells that the same thermal operator solves
with TaIrTe4 conductivity.  Using an analytic sub-cell polygon only for Q
while solving that partial cell as air would be inconsistent.  A truly
analytic polygon cut-cell overlap would therefore require a matching cut-cell
conductivity/interface operator and is not claimed here.

## 100 nm thermal-grid result

- full common-grid optical power: `{full:.15e} W`
- material-overlap-attributed TaIrTe4 power: `{attributed:.15e} W`
  (`{fractions['TaIrTe4_material_overlap_attributed']:.6%}` of full)
- zero-overlap, non-TaIrTe4/unattributed power: `{zero_overlap:.15e} W`
  (`{fractions['zero_overlap_non_TaIrTe4_unattributed']:.6%}`)
- explicitly excluded metal power: `{metal:.15e} W`
  (`{fractions['explicit_metal_excluded']:.6%}`)
- signed partition residual: `{signed_partition_residual:.15e} W`
- source-to-target mapping error: `{mapping['mapping_relative_power_error']:.3e}`
- power outside TaIrTe4 after mapping: `0 W`
- change from the old cell-centre mask diagnostic: `{old_to_overlap_change:.6%}`

The 500 nm case is only a coarse operator control.  Its larger zero-overlap
term demonstrates why coarse thermal geometry must not be used to promote a
physical source partition.

The immutable input artifact predates the new 11-um Palik substrate contract.
Therefore this checkpoint validates the mapping mathematics and real Device-A
grid execution, but it does not promote the quoted absorbed power as the final
Palik optical/thermal prediction.  No new FDTD, thermal solve, weighting
potential, PTE current, adjoint, or optimization was run.
"""
    (output / "DEVICE_A_MATERIAL_OVERLAP_REMAP_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps({"status": status, "gates": gates}, indent=2))
    return 0 if status.startswith("VALIDATED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
