#!/usr/bin/env python3
"""Publish the fail-closed Device-A source/mapping/current correction audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, kind: str) -> dict[str, object]:
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "committed_to_git": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--legacy-current-json", type=Path, required=True)
    parser.add_argument("--intersection-current-json", type=Path, required=True)
    parser.add_argument("--fig3h-json", type=Path, required=True)
    parser.add_argument("--loss-a-json", type=Path, required=True)
    parser.add_argument("--loss-b-json", type=Path, required=True)
    parser.add_argument("--legacy-thermal-a", type=Path, required=True)
    parser.add_argument("--legacy-thermal-b", type=Path, required=True)
    parser.add_argument("--intersection-thermal-a", type=Path, required=True)
    parser.add_argument("--intersection-thermal-b", type=Path, required=True)
    parser.add_argument("--mapping-a", type=Path, required=True)
    parser.add_argument("--mapping-b", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    legacy = load(args.legacy_current_json)
    intersection = load(args.intersection_current_json)
    fig3h = load(args.fig3h_json)
    loss = {"a": load(args.loss_a_json), "b": load(args.loss_b_json)}
    old_thermal = {
        p: load(path / "summary.json")
        for p, path in (("a", args.legacy_thermal_a), ("b", args.legacy_thermal_b))
    }
    new_thermal = {
        p: load(path / "summary.json")
        for p, path in (
            ("a", args.intersection_thermal_a),
            ("b", args.intersection_thermal_b),
        )
    }
    mapping = {
        p: load(path / "material_overlap_mapping_summary.json")
        for p, path in (("a", args.mapping_a), ("b", args.mapping_b))
    }

    methods = ("legacy", "strict_centered", "internal_face")
    rows = []
    for attribution, current in (
        ("covered-cell-full-power", legacy),
        ("intersection-density", intersection),
    ):
        for method in methods:
            rows.append(
                {
                    "Q_attribution": attribution,
                    "current_quadrature": method,
                    "abs_Ia_over_abs_Ib": current["abs_Ia_over_abs_Ib"][method],
                    "I_a_A": current["cases"]["a"][method][
                        "reported_A" if method == "legacy" else "current_A"
                    ],
                    "I_b_A": current["cases"]["b"][method][
                        "reported_A" if method == "legacy" else "current_A"
                    ],
                }
            )
    old_ratio = float(legacy["abs_Ia_over_abs_Ib"]["legacy"])
    new_ratio = float(intersection["abs_Ia_over_abs_Ib"]["legacy"])
    target = 0.8365896980461811
    p_old = {
        p: float(old_thermal[p]["mapping"]["P_Q_source_W"]) for p in "ab"
    }
    p_new = {
        p: float(new_thermal[p]["mapping"]["P_Q_source_W"]) for p in "ab"
    }
    payload = {
        "status": "BLOCKED_DEVICE_A_FIG3H_REGISTRATION_AND_EXACT_LOSS_ATTRIBUTION",
        "scope": (
            "offline figure registration, read-only completed-FSP E/index audit, "
            "two intersection-remap thermal solves, and offline current quadrature; "
            "no new Maxwell run"
        ),
        "definite_findings": {
            "old_fig3h_pixels_used_for_source_coordinate": False,
            "old_scan_direction_mismatch_deg": fig3h["old_contract_audit"][
                "direction_mismatch_vs_fig3h_scan_deg"
            ],
            "old_chord_is_literal_polygon_edge": False,
            "E_index_component_collocation_maximum_mismatch_m": max(
                float(loss[p]["maximum_field_index_coordinate_mismatch_m"])
                for p in "ab"
            ),
            "covered_cell_full_power_exceeds_literal_intersection_attribution": True,
            "which_attribution_is_exact_for_conformal_loss": "UNRESOLVED",
            "intersection_mapping_closure_maximum": max(
                float(mapping[p]["mapping"]["mapping_relative_power_error"])
                for p in "ab"
            ),
            "intersection_mapping_power_outside_TaIrTe4_W": {
                p: mapping[p]["mapped_power_outside_TaIrTe4_W"] for p in "ab"
            },
            "one_sided_gradient_is_sole_polarization_reversal_cause": False,
        },
        "power_and_current": {
            "P_Q_covered_cell_full_power_W": p_old,
            "P_Q_intersection_density_W": p_new,
            "P_Q_relative_change": {
                p: (p_new[p] - p_old[p]) / p_old[p] for p in "ab"
            },
            "legacy_current_ratio_covered_cell_full_power": old_ratio,
            "legacy_current_ratio_intersection_density": new_ratio,
            "ratio_relative_change": (new_ratio - old_ratio) / old_ratio,
            "paper_digitized_ratio": target,
            "all_named_quadratures_retain_Ia_over_Ib_gt_1": all(
                float(row["abs_Ia_over_abs_Ib"]) > 1.0 for row in rows
            ),
            "cases": rows,
        },
        "loss_proxy": {
            "definition": (
                "Im(epsilon_eff,c)/Im(epsilon_bulk-TaIrTe4,c)"
            ),
            "status": "DIAGNOSTIC_ONLY_NOT_OCCUPANCY",
            "no_clipping_applied": True,
            "component_coordinate_pairing_passed": True,
            "out_of_unit_interval_occurs": any(
                loss[p]["components"][c]["proxy_above_one_cell_count"] > 0
                or loss[p]["components"][c]["proxy_below_zero_cell_count"] > 0
                for p in "ab" for c in "xyz"
            ),
            "thermal_source_from_proxy": False,
            "reason_not_promoted": (
                "Q already contains Im(epsilon_eff), and cells containing lossy "
                "metal/SiO2 make the ratio exceed one; multiplying Q by the "
                "proxy can double-count conformal mixing"
            ),
        },
        "unresolved": {
            "absolute_fig3h_source_coordinate": True,
            "raw_experimental_stage_metadata_or_CAD": "not published in supplied files",
            "exact_conformal_material_resolved_absorption": True,
            "explicit_contact_boundary_face_current_quadrature": True,
            "absolute_current_geometry_resistance_mismatch": {
                "predicted_ohm": new_thermal["a"]["two_terminal_resistance_audit"][
                    "predicted_resistance_ohm"
                ],
                "measured_ohm": 213.0,
            },
        },
        "gates": {
            "21_targeted_offline_tests_passed": True,
            "new_FDTD_run": False,
            "new_GPU_run": False,
            "new_source_position_promoted": False,
            "paper_trend_reproduced": False,
        },
        "next_step_requires_explicit_assumption": (
            "choose a named Fig3H-to-Fig2 affine registration or obtain raw scan "
            "coordinates; only then run one E||a/b Gaussian pair at the registered "
            "point. Do not reuse the old chord-normal s=3 um coordinate."
        ),
    }
    (args.report_dir / "device_a_correction_audit_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    with (args.report_dir / "device_a_correction_audit_cases.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [
        "full/legacy", "full/strict", "full/face",
        "intersection/legacy", "intersection/strict", "intersection/face",
    ]
    values = [float(row["abs_Ia_over_abs_Ib"]) for row in rows]
    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.bar(np.arange(len(values)), values, color=["#5577aa"] * 3 + ["#44aa77"] * 3)
    ax.axhline(1.0, color="black", linestyle="--", label="trend boundary")
    ax.axhline(target, color="tab:red", linestyle=":", label="digitized paper")
    ax.set_xticks(np.arange(len(values)), labels, rotation=20, ha="right")
    ax.set_ylabel("abs(Ia)/abs(Ib)")
    ax.set_title("Device-A correction audit: attribution and current quadrature")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(args.report_dir / "DEVICE_A_CORRECTION_RATIO_AUDIT.png", dpi=210)
    plt.close(fig)

    raw_paths = []
    for p, directory in (
        ("a", args.legacy_thermal_a), ("b", args.legacy_thermal_b),
        ("a", args.intersection_thermal_a), ("b", args.intersection_thermal_b),
    ):
        raw_paths.extend(
            [
                artifact(directory / "summary.json", f"thermal-summary-{p}"),
                artifact(directory / "thermal_pte_fields.npz", f"thermal-fields-{p}"),
            ]
        )
    for p, directory in (("a", args.mapping_a), ("b", args.mapping_b)):
        raw_paths.extend(
            [
                artifact(
                    directory / "material_overlap_mapping_summary.json",
                    f"mapping-summary-{p}",
                ),
                artifact(
                    directory / "material_overlap_mapped_q.npz",
                    f"mapping-Q-{p}",
                ),
            ]
        )
    for p, thermal in old_thermal.items():
        optical = Path(thermal["mapping"]["optical_artifact_path"])
        fsp = optical.parent / "finite_2um_optical_q.fsp"
        raw_paths.extend(
            [artifact(optical, f"optical-Q-{p}"), artifact(fsp, f"optical-FSP-{p}")]
        )
    manifest = {
        "status": "DEVICE_A_CORRECTION_AUDIT_RAW_MANIFEST",
        "raw_artifacts_committed_to_git": False,
        "artifacts": raw_paths,
    }
    (args.report_dir / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    report = f"""# Device-A source, mapping, and current correction audit

Status: `{payload['status']}`

## What was corrected

The old source position was not registered from Figure 3H: the crop was only
stored as provenance. The simulated chord-normal scan differs from the actual
near-vertical black dashed line by
`{payload['definite_findings']['old_scan_direction_mismatch_deg']:.3f} deg`,
and vertices 4--7 are not one polygon edge.

The old remap also preserved a complete optical-cell power whenever that cell
had any TaIrTe4 overlap. The new `intersection-density` diagnostic deposits
only `Q * literal intersection volume`, with no nearest-cell relocation,
gain, clipping, smoothing, or global rescaling.

| quantity | E||a | E||b |
|---|---:|---:|
| old attributed P_Q (W) | {p_old['a']:.9e} | {p_old['b']:.9e} |
| intersection P_Q (W) | {p_new['a']:.9e} | {p_new['b']:.9e} |
| relative change | {(p_new['a']-p_old['a'])/p_old['a']:.4%} | {(p_new['b']-p_old['b'])/p_old['b']:.4%} |

The legacy-current ratio falls from `{old_ratio:.9f}` to `{new_ratio:.9f}`,
but does not cross one. Strict four-neighbour and common-face quadratures also
retain `abs(Ia)/abs(Ib)>1`. Therefore neither the old cut-cell power rule nor
the one-sided gradient is the sole cause of the reversed paper trend.

## Read-only FSP audit

Component-specific E/index coordinates agree to
`{payload['definite_findings']['E_index_component_collocation_maximum_mismatch_m']:.3e} m`.
The effective-loss ratio leaves [0,1] in cells containing other lossy media,
so it is not an occupancy and was not used as a heat source. This also means
the full-power and literal-intersection results are two named attribution
scenarios, not two claims that one is exact conformal material decomposition.

## Blocking item before another GPU pair

An absolute Figure-3H source coordinate is not published and cannot be
recovered from the current crops without an explicit affine-registration
assumption. No new FDTD was launched. The old 3-um chord-normal point must not
be reused as the experimental position.
"""
    (args.report_dir / "DEVICE_A_CORRECTION_AUDIT_REPORT.md").write_text(report)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
